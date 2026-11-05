"""False-positive pipeline for one (paper, reviewer system).

Loads the review of the clean copy, extracts its flags, routes each module's
flags to its adjudicator, and writes data/{paper_id}/fp/{reviewer_system}_v0.json.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cost import combined_total_cost_usd as _total_cost
from .fp_citations import adjudicate_many_citation_flags
from .fp_extractor import extract_flags
from .fp_general import adjudicate_general_flags
from .fp_maths import adjudicate_maths_flags
from .fp_numerics import adjudicate_numerics_flags
from .fp_schema import (
    AdjudicationVerdict,
    FPFlag,
    Module,
    PaperFPReport,
    ReviewerSystem,
    Verdict,
)


def _load_review(
    paper_dir: Path,
    reviewer_system: str,
    provider_tag: Optional[str] = None,
) -> Optional[str]:
    if provider_tag:
        candidate_dirs = [
            paper_dir / "reviews" / reviewer_system / f"v0__{provider_tag}.md",
            paper_dir / "reviews" / reviewer_system.lower() / f"v0__{provider_tag}.md",
        ]
    else:
        candidate_dirs = [
            paper_dir / "reviews" / reviewer_system / "v0.md",
            paper_dir / "reviews" / reviewer_system.lower() / "v0.md",
        ]
    for p in candidate_dirs:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def _load_paper_pdf(paper_dir: Path) -> Optional[Path]:
    for candidate in [
        paper_dir / "v0" / "paper.pdf",
        paper_dir / "v0" / "v0.pdf",
        paper_dir / "paper.pdf",
        paper_dir / "v0.pdf",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_reference_entries(paper_dir: Path) -> List[str]:
    """Reference strings for the citations adjudicator, from the .bbl file or paper.md; [] if none."""
    for bbl in sorted(paper_dir.glob("source/tex/*.bbl")):
        text = bbl.read_text(encoding="utf-8", errors="replace")
        entries = re.split(r"\\bibitem\b[^\n]*\n", text)
        result = [e.strip().replace("\n", " ") for e in entries if e.strip()]
        if result:
            return result[:200]

    for candidate in [paper_dir / "v0" / "paper.md", paper_dir / "paper.md"]:
        if candidate.exists():
            md = candidate.read_text(encoding="utf-8")
            m = re.search(
                r"^#{1,3}\s*(References|Bibliography|Works Cited)\s*$",
                md, re.IGNORECASE | re.MULTILINE,
            )
            if m:
                section = md[m.end():]
                entries = re.split(r"\n\s*\[?\d+\]?\s+", section)
                result = [e.strip().replace("\n", " ") for e in entries if e.strip()]
                if result:
                    return result[:200]

    return []


def _compute_fp_rates(verdicts: List[AdjudicationVerdict]) -> Dict[str, float]:
    if not verdicts:
        return {}
    total = len(verdicts)
    fp_count = sum(1 for v in verdicts if v.verdict == "FP")
    return {"fp_rate": fp_count / total, "total": total, "fp_count": fp_count}


def run_fp_pipeline(
    *,
    paper_id: str,
    paper_dir: Path,
    reviewer_system: ReviewerSystem,
    params_skepthical: Dict[str, Any],
    api_keys: Optional[Dict[str, Any]] = None,
    general_dedup_cache: Optional[Dict[str, AdjudicationVerdict]] = None,
    citations_dedup_cache: Optional[Dict[str, AdjudicationVerdict]] = None,
    corpus_csv: Optional[Path] = None,
    provider_tag: Optional[str] = None,
) -> Optional[PaperFPReport]:
    """Run the false-positive pipeline for one (paper, reviewer system).

        Pass the same dedup caches across reviewers of a paper to adjudicate shared
        flags once. Returns None if the review file is not found.
    """
    started_at = datetime.now(timezone.utc)
    cost_at_start = _total_cost()

    review_text = _load_review(paper_dir, reviewer_system, provider_tag)
    if review_text is None:
        tag = f"/{provider_tag}" if provider_tag else ""
        print(f"[fp_orchestrator] review not found for {paper_id}/{reviewer_system}{tag}/v0", flush=True)
        return None

    pdf_path = _load_paper_pdf(paper_dir)
    reference_entries = _load_reference_entries(paper_dir)
    work_dir = paper_dir / "fp" / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cell_label = f"{reviewer_system}/{provider_tag}" if provider_tag else reviewer_system
    print(f"[fp_orchestrator] extracting flags: {paper_id}/{cell_label}", flush=True)
    flags: List[FPFlag] = extract_flags(
        report_text=review_text,
        reviewer_system=reviewer_system,
        paper_id=paper_id,
        paper_dir=paper_dir,
        api_keys=api_keys,
    )
    print(f"[fp_orchestrator] extracted {len(flags)} flags", flush=True)

    if not flags:
        extractor_only = float(_total_cost() - cost_at_start)
        report = PaperFPReport(
            paper_id=paper_id,
            reviewer_system=reviewer_system,
            per_module_verdicts={},
            per_module_fp_rate={},
            per_module_unadjudicated_count={},
            extractor_cost_usd=extractor_only,
            total_cost_usd=extractor_only,
            provider_tag=provider_tag,
        )
        _write_report(paper_dir, reviewer_system, report, provider_tag=provider_tag)
        return report

    by_module: Dict[Module, List[FPFlag]] = {"citations": [], "numerics": [], "mathematics": [], "general": []}
    for flag in flags:
        by_module[flag.module].append(flag)

    per_module_verdicts: Dict[Module, List[AdjudicationVerdict]] = {}

    def _module_failed(mod: Module, flags_in_mod: List[FPFlag], reason: str) -> List[AdjudicationVerdict]:
        return [
            AdjudicationVerdict(
                flag_id=f.flag_id, paper_id=paper_id,
                reviewer_systems_attributed=[f.reviewer_system],
                module=mod, verdict="UNADJUDICATABLE",
                confidence="low", reason=reason,
                evidence={"claim": f.claim, "excerpt": f.excerpt},
            )
            for f in flags_in_mod
        ]

    if by_module["citations"]:
        print(f"[fp_orchestrator] adjudicating {len(by_module['citations'])} citations flags", flush=True)
        try:
            per_module_verdicts["citations"] = adjudicate_many_citation_flags(
                by_module["citations"],
                reference_entries=reference_entries,
                work_dir=work_dir,
                paper_id=paper_id,
                paper_dir=paper_dir,
                params_skepthical=params_skepthical,
                api_keys=api_keys,
                dedup_cache=citations_dedup_cache,
            )
        except Exception as exc:
            print(f"[fp_orchestrator] citations module failed: {type(exc).__name__}: {exc}", flush=True)
            per_module_verdicts["citations"] = _module_failed(
                "citations", by_module["citations"],
                f"citations module raised {type(exc).__name__}: {exc}",
            )

    if by_module["numerics"] and pdf_path is not None:
        print(f"[fp_orchestrator] adjudicating {len(by_module['numerics'])} numerics flags", flush=True)
        try:
            per_module_verdicts["numerics"] = adjudicate_numerics_flags(
                flags=by_module["numerics"],
                pdf_path=pdf_path,
                paper_id=paper_id,
                paper_dir=paper_dir,
                work_dir=work_dir,
                params_skepthical=params_skepthical,
                api_keys=api_keys,
            )
        except Exception as exc:
            print(f"[fp_orchestrator] numerics module failed: {type(exc).__name__}: {exc}", flush=True)
            per_module_verdicts["numerics"] = _module_failed(
                "numerics", by_module["numerics"],
                f"numerics module raised {type(exc).__name__}: {exc}",
            )
    elif by_module["numerics"]:
        print(f"[fp_orchestrator] no PDF found — numerics flags marked UNADJUDICATABLE", flush=True)
        per_module_verdicts["numerics"] = _module_failed(
            "numerics", by_module["numerics"], "Paper PDF not found.",
        )

    if by_module["mathematics"] and pdf_path is not None:
        print(f"[fp_orchestrator] adjudicating {len(by_module['mathematics'])} mathematics flags", flush=True)
        try:
            per_module_verdicts["mathematics"] = adjudicate_maths_flags(
                flags=by_module["mathematics"],
                pdf_path=pdf_path,
                paper_id=paper_id,
                paper_dir=paper_dir,
                work_dir=work_dir,
                params_skepthical=params_skepthical,
                api_keys=api_keys,
                corpus_csv=corpus_csv,
            )
        except Exception as exc:
            print(f"[fp_orchestrator] mathematics module failed: {type(exc).__name__}: {exc}", flush=True)
            per_module_verdicts["mathematics"] = _module_failed(
                "mathematics", by_module["mathematics"],
                f"mathematics module raised {type(exc).__name__}: {exc}",
            )
    elif by_module["mathematics"]:
        per_module_verdicts["mathematics"] = _module_failed(
            "mathematics", by_module["mathematics"], "Paper PDF not found.",
        )

    if by_module["general"]:
        print(f"[fp_orchestrator] adjudicating {len(by_module['general'])} general flags", flush=True)
        try:
            per_module_verdicts["general"] = adjudicate_general_flags(
                flags=by_module["general"],
                pdf_path=pdf_path,
                paper_id=paper_id,
                paper_dir=paper_dir,
                api_keys=api_keys,
                dedup_cache=general_dedup_cache,
            )
        except Exception as exc:
            print(f"[fp_orchestrator] general module failed: {type(exc).__name__}: {exc}", flush=True)
            per_module_verdicts["general"] = _module_failed(
                "general", by_module["general"],
                f"general module raised {type(exc).__name__}: {exc}",
            )

    per_module_fp_rate: Dict[Module, float] = {}
    per_module_unadj: Dict[Module, int] = {}
    per_module_cost: Dict[Module, float] = {}
    verdict_cost_total = 0.0
    for mod, verdicts in per_module_verdicts.items():
        adjudicated = [v for v in verdicts if v.verdict != "UNADJUDICATABLE"]
        unadj_count = sum(1 for v in verdicts if v.verdict == "UNADJUDICATABLE")
        fp_count = sum(1 for v in adjudicated if v.verdict == "FP")
        per_module_fp_rate[mod] = (fp_count / len(adjudicated)) if adjudicated else 0.0
        per_module_unadj[mod] = unadj_count
        mod_cost = sum(float(v.cost_usd) for v in verdicts)
        per_module_cost[mod] = mod_cost
        verdict_cost_total += mod_cost

    cost_at_end = _total_cost()
    total_cost = float(cost_at_end - cost_at_start)
    residual_cost = max(total_cost - verdict_cost_total, 0.0)

    report = PaperFPReport(
        paper_id=paper_id,
        reviewer_system=reviewer_system,
        per_module_verdicts=per_module_verdicts,
        per_module_fp_rate=per_module_fp_rate,
        per_module_unadjudicated_count=per_module_unadj,
        per_module_cost_usd=per_module_cost,
        extractor_cost_usd=residual_cost,
        total_cost_usd=total_cost,
        provider_tag=provider_tag,
    )

    _write_report(paper_dir, reviewer_system, report, provider_tag=provider_tag)
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    print(
        f"[fp_orchestrator] done in {elapsed:.1f}s: {paper_id}/{cell_label} "
        f"(cost ${total_cost:.4f} — verdicts ${verdict_cost_total:.4f}, "
        f"residual ${residual_cost:.4f})",
        flush=True,
    )
    return report


def _write_report(
    paper_dir: Path,
    reviewer_system: str,
    report: PaperFPReport,
    *,
    provider_tag: Optional[str] = None,
) -> Path:
    out_dir = paper_dir / "fp"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{reviewer_system}__{provider_tag}_v0" if provider_tag else f"{reviewer_system}_v0"
    out_path = out_dir / f"{stem}.json"
    # Atomic write so a crash never leaves a truncated report.
    import os
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, out_path)
    return out_path
