"""False-positive adjudicator for maths flags.

For each paper: extract the flagged equations from the PDF, check the
transcription, run mechanical checks (Wolfram / pint), then have an LLM
interpret the results.
  missing equation or refuted transcription -> FP
  interpreter PASS                          -> FP
  interpreter FAIL, confidence >= medium    -> CONFIRMED_ISSUE
  otherwise                                 -> UNADJUDICATABLE

Flags raised by L3 are not adjudicated automatically, since L3 runs this
same maths pipeline; they are written out for manual review instead.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cost import combined_total_cost_usd as _total_cost, log_call
from .llm_client import _openai_get_or_upload
from .fp_schema import AdjudicationVerdict, FPFlag, ReviewerSystem, Verdict


def _import_maths():
    try:
        import skepthical.maths as _m
        return _m
    except ImportError:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import skepthical.maths as _m
        return _m


def _import_maths_engines():
    try:
        from skepthical.maths_engines import WolframSession, wolfram_available
        return WolframSession, wolfram_available
    except ImportError:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from skepthical.maths_engines import WolframSession, wolfram_available
        return WolframSession, wolfram_available


def _import_math_schemas():
    try:
        from skepthical.schemas import MathStatement, MechanicalEvidence, ToolResult
        return MathStatement, MechanicalEvidence, ToolResult
    except ImportError:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from skepthical.schemas import MathStatement, MechanicalEvidence, ToolResult
        return MathStatement, MechanicalEvidence, ToolResult


_TARGETED_EXTRACTOR_INSTRUCTIONS = """\
You are extracting specific equations from a scientific paper PDF for
mechanical verification. Unlike the default maths extractor, you are NOT
surveying the whole paper — you are given SPECIFIC locations (section,
equation number, or verbatim cue) and must extract ONLY those equations.

For each flagged location, find the corresponding equation in the PDF and
produce a MathStatement with:
  id:                    reuse the input flag_id exactly
  location:              section / equation number (e.g. "Eq. (3), Sec 2.1")
  kind:                  "equation" | "derivation" | "identity" | "normalisation" | etc.
  latex:                 list of LaTeX strings for the equations at that location
  claim:                 one-sentence paraphrase of what the equation asserts
  assumptions:           list of stated assumptions (may be empty)
  importance:            "critical" | "moderate" | "minor"
  units_critical:        true only if dimensional failure would invalidate the claim
  check_plan:            list of mechanical checks [{engine, args, description}]
                         where engine is "wolfram" or "pint"
  verbatim_source_snippet: short verbatim excerpt of the paper text around the equation
                            (REQUIRED — the transcription-fidelity checker uses this)

If an equation is NOT found at the stated location, add the flag_id to "missing".

STRICT JSON — no prose, no code fences:
{
  "statements": [ MathStatement, ... ],
  "missing": [{"flag_id": "...", "reason": "equation not found at stated location"}]
}
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _build_targeted_extractor_prompt(flags: List[Dict[str, Any]]) -> str:
    items = [
        {"flag_id": f.get("flag_id", ""),
         "location_cue": f.get("canonical_entity", ""),
         "reviewer_claim": f.get("claim", "")}
        for f in flags
    ]
    return (
        _TARGETED_EXTRACTOR_INSTRUCTIONS
        + "\n\nFLAGGED LOCATIONS:\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
    )


def _parse_targeted_extractor_output(
    text: str, all_flag_ids: List[str], MathStatement: type,
) -> Tuple[list, List[str]]:
    s = (text or "").strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    statements = []
    missing_ids: List[str] = []

    try:
        obj = json.loads(s, strict=False)
    except json.JSONDecodeError:
        return [], list(all_flag_ids)

    for raw_stmt in obj.get("statements", []):
        try:
            stmt = MathStatement.model_validate(raw_stmt)
            statements.append(stmt)
        except Exception:
            fid = raw_stmt.get("id", "")
            if fid:
                missing_ids.append(fid)

    for miss in obj.get("missing", []):
        fid = miss.get("flag_id", "")
        if fid:
            missing_ids.append(fid)

    accounted = {s.id for s in statements} | set(missing_ids)
    for fid in all_flag_ids:
        if fid not in accounted:
            missing_ids.append(fid)

    return statements, missing_ids


def _verdict_from_checked_item(item: Any) -> Tuple[Verdict, str, str]:
    v = getattr(item, "verdict", "UNCERTAIN")
    conf = getattr(item, "confidence", "low")
    notes = getattr(item, "notes", "") or ""

    if v == "PASS":
        return "FP", conf, f"Maths interpreter: PASS. {notes}".strip()
    if v == "FAIL":
        if conf in ("high", "medium"):
            return "CONFIRMED_ISSUE", conf, f"Maths interpreter: FAIL. {notes}".strip()
        return "UNADJUDICATABLE", "low", f"Maths interpreter: FAIL (low confidence). {notes}".strip()
    return "UNADJUDICATABLE", "low", f"Maths interpreter: UNCERTAIN. {notes}".strip()


def _adjudicate_maths_batch(
    *,
    pdf_path: str,
    flags: List[Dict[str, Any]],
    params_skepthical: Dict[str, Any],
    api_keys: Optional[Dict[str, Any]] = None,
    work_dir: str,
    paper_id: str,
    paper_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Returns {flag_id: {verdict, confidence, reason, evidence}}."""
    if not flags:
        return {}

    _maths = _import_maths()
    WolframSession, wolfram_available = _import_maths_engines()
    MathStatement, MechanicalEvidence, ToolResult = _import_math_schemas()

    models = params_skepthical.get("models", {}) or {}
    extractor_model = models.get("maths_extractor", "gemini-3.1-pro-preview")
    transcription_model = models.get("maths_transcription_checker", "gemini-3.1-pro-preview")
    interpreter_model = models.get("maths_interpreter", "claude-sonnet-4-6")
    max_workers = int(params_skepthical.get("maths_parallel_workers", 10))
    timeout_s: Optional[float] = params_skepthical.get("timeout", None)
    use_wolfram = bool(params_skepthical.get("maths_wolfram_persistent", True))

    openai_client = _maths._make_openai_client(api_keys)
    anthropic_client = _maths._make_anthropic_client(api_keys)

    debug_dir = os.path.join(work_dir, "_fp_maths_debug")
    os.makedirs(debug_dir, exist_ok=True)

    all_flag_ids = [f.get("flag_id", "") for f in flags]
    results: Dict[str, Dict[str, Any]] = {}
    uploaded_id: Optional[str] = None
    wolfram_session = None

    try:
        uploaded_id = _openai_get_or_upload(openai_client, Path(pdf_path))

        targeted_prompt = _build_targeted_extractor_prompt(flags)
        from skepthical.schemas import FPMathsStructurerOutput
        from skepthical.llm_call import resolve_provider as _resolve_provider
        from .llm_client import call_with_pdf as _call_with_pdf
        ext_result = _call_with_pdf(
            provider=_resolve_provider(extractor_model),
            model=extractor_model,
            prompt=targeted_prompt,
            pdf_path=Path(pdf_path),
            agent_name="fp_maths_targeted_extractor",
            paper_id=paper_id, paper_dir=paper_dir,
            max_tokens=128000,
            api_keys=api_keys,
            schema=FPMathsStructurerOutput,
        )
        ext_text = ext_result.text
        statements, missing_ids = _parse_targeted_extractor_output(ext_text, all_flag_ids, MathStatement)

        for fid in missing_ids:
            results[fid] = {
                "verdict": "FP", "confidence": "medium",
                "reason": "Targeted extractor could not find the flagged equation in the paper.",
                "evidence": {"extractor_status": "missing"},
            }

        if not statements:
            return results

        fidelity_by_id = _maths._stage1b_transcription_fidelity(
            openai_client=openai_client, anthropic_client=anthropic_client,
            api_keys=api_keys,
            model=transcription_model,
            statements=statements, max_workers=max_workers, debug_dir=debug_dir,
        )

        for stmt in list(statements):
            fid_result = fidelity_by_id.get(stmt.id, {})
            if fid_result.get("status") == "refuted":
                results[stmt.id] = {
                    "verdict": "FP", "confidence": "medium",
                    "reason": "Transcription-fidelity check: equation in paper does not match reviewer's description.",
                    "evidence": {"fidelity": fid_result},
                }

        active_stmts = [s for s in statements if s.id not in results]
        if not active_stmts:
            return results

        if use_wolfram and wolfram_available():
            try:
                wolfram_session = WolframSession()
            except Exception:
                wolfram_session = None

        evidence_by_id = _maths._stage2_mechanical(
            active_stmts, debug_dir=debug_dir, wolfram_session=wolfram_session,
        )

        for sid, fid_data in fidelity_by_id.items():
            ev = evidence_by_id.setdefault(sid, MechanicalEvidence(statement_id=sid, results=[]))
            ev.results.insert(0, ToolResult(
                tool="llm",
                status=fid_data.get("status", "inconclusive"),
                detail=str(fid_data.get("detail", ""))[:2000],
                seconds=float(fid_data.get("seconds", 0.0)),
                description="transcription-fidelity check (cross-family LLM)",
            ))

        checked_items = _maths._stage3_interpret(
            openai_client=openai_client, anthropic_client=anthropic_client,
            api_keys=api_keys,
            model=interpreter_model,
            statements=active_stmts, evidence_by_id=evidence_by_id,
            timeout_s=timeout_s, max_workers=max_workers, debug_dir=debug_dir,
        )

        for item in checked_items:
            verdict, conf, reason = _verdict_from_checked_item(item)
            results[item.id] = {
                "verdict": verdict, "confidence": conf, "reason": reason,
                "evidence": {
                    "maths_verdict": getattr(item, "verdict", None),
                    "confidence": getattr(item, "confidence", None),
                    "impact": getattr(item, "impact", None),
                    "notes": getattr(item, "notes", None),
                    "tools_used": getattr(item, "tools_used", []),
                },
            }

    except Exception as exc:
        print(f"[fp_maths] unexpected error: {type(exc).__name__}: {exc}", flush=True)
        for fid in all_flag_ids:
            if fid not in results:
                results[fid] = {
                    "verdict": "UNADJUDICATABLE", "confidence": "low",
                    "reason": f"Adjudicator error: {type(exc).__name__}: {exc}",
                    "evidence": {},
                }
    finally:
        if wolfram_session is not None:
            try:
                wolfram_session.close()
            except Exception:
                pass

    return results


def emit_l3_maths_manual_review(
    *,
    paper_id: str,
    paper_dir: Path,
    l3_maths_flags: List[FPFlag],
    corpus_csv: Path,
) -> Path:
    """Write data/{paper_id}/l3_maths_for_manual_review.md and append rows to the review CSV."""
    out_path = paper_dir / "l3_maths_for_manual_review.md"
    lines = [
        f"# L3 Maths Manual Review — {paper_id}\n",
        "Fill in `MANUAL: [CONFIRMED_ISSUE | FP | AMBIGUOUS] — <reason>` for each flag.\n\n",
        "---\n",
    ]
    for i, flag in enumerate(l3_maths_flags, 1):
        lines.append(f"## Flag {i}: `{flag.canonical_entity or flag.flag_id}`\n\n")
        lines.append(f"**Claim:** {flag.claim}\n\n")
        lines.append(f"**Excerpt:**\n\n> {flag.excerpt}\n\n")
        lines.append(f"**flag_id:** `{flag.flag_id}`\n\n")
        lines.append("**MANUAL:** [CONFIRMED_ISSUE | FP | AMBIGUOUS] — _reason here_\n\n")
        lines.append("---\n")

    out_path.write_text("".join(lines), encoding="utf-8")

    corpus_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not corpus_csv.exists()
    with open(corpus_csv, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "paper_id", "flag_id", "canonical_entity", "claim", "excerpt", "manual_verdict", "manual_reason",
        ])
        if write_header:
            writer.writeheader()
        for flag in l3_maths_flags:
            writer.writerow({
                "paper_id": paper_id,
                "flag_id": flag.flag_id,
                "canonical_entity": flag.canonical_entity,
                "claim": flag.claim,
                "excerpt": flag.excerpt[:300],
                "manual_verdict": "",
                "manual_reason": "",
            })

    return out_path


def adjudicate_maths_flags(
    *,
    flags: List[FPFlag],
    pdf_path: Path,
    paper_id: str,
    paper_dir: Path,
    work_dir: Path,
    params_skepthical: Dict[str, Any],
    api_keys: Optional[Dict[str, Any]] = None,
    corpus_csv: Optional[Path] = None,
) -> List[AdjudicationVerdict]:
    if not flags:
        return []

    l3_flags = [f for f in flags if f.reviewer_system == "L3"]
    non_l3_flags = [f for f in flags if f.reviewer_system != "L3"]

    results: List[AdjudicationVerdict] = []

    if l3_flags:
        _csv = corpus_csv or (paper_dir.parent.parent / "papers" / "l3_maths_manual_review_queue.csv")
        md_path = emit_l3_maths_manual_review(
            paper_id=paper_id,
            paper_dir=paper_dir,
            l3_maths_flags=l3_flags,
            corpus_csv=_csv,
        )
        print(f"[fp_maths] L3 maths flags → manual review: {md_path}", flush=True)
        for flag in l3_flags:
            results.append(AdjudicationVerdict(
                flag_id=flag.flag_id,
                paper_id=paper_id,
                reviewer_systems_attributed=["L3"],
                module="mathematics",
                verdict="UNADJUDICATABLE",
                confidence="low",
                reason="L3 maths flags require manual review (automated adjudication would be circular).",
                evidence={"l3_manual_review_path": str(md_path)},
            ))

    if non_l3_flags:
        flag_dicts = [f.as_dict() for f in non_l3_flags]
        cost_at_start = _total_cost()
        from skepthical.cost import ledger_context as _ledger_ctx
        with _ledger_ctx(
            paper_id=paper_id, paper_dir=paper_dir,
            method="fp_maths",
        ):
            raw_results = _adjudicate_maths_batch(
                pdf_path=str(pdf_path),
                flags=flag_dicts,
                params_skepthical=params_skepthical,
                api_keys=api_keys,
                work_dir=str(work_dir),
                paper_id=paper_id,
                paper_dir=paper_dir,
            )
        batch_cost = max(_total_cost() - cost_at_start, 0.0)

        id_to_reviewers: Dict[str, list] = {}
        for flag in non_l3_flags:
            id_to_reviewers.setdefault(flag.flag_id, [])
            if flag.reviewer_system not in id_to_reviewers[flag.flag_id]:
                id_to_reviewers[flag.flag_id].append(flag.reviewer_system)

        n_unique = len({f.flag_id for f in non_l3_flags})
        per_flag_cost = batch_cost / max(n_unique, 1)

        seen: set[str] = set()
        for flag in non_l3_flags:
            fid = flag.flag_id
            if fid in seen:
                continue
            seen.add(fid)
            raw = raw_results.get(fid, {})
            verdict_raw: Verdict = raw.get("verdict", "UNADJUDICATABLE")
            if verdict_raw not in ("FP", "CONFIRMED_ISSUE", "AMBIGUOUS", "UNADJUDICATABLE"):
                verdict_raw = "UNADJUDICATABLE"
            conf_raw = raw.get("confidence", "low")
            if conf_raw not in ("high", "medium", "low"):
                conf_raw = "low"
            results.append(AdjudicationVerdict(
                flag_id=fid,
                paper_id=paper_id,
                reviewer_systems_attributed=id_to_reviewers.get(fid, [flag.reviewer_system]),
                module="mathematics",
                verdict=verdict_raw,
                confidence=conf_raw,  # type: ignore[arg-type]
                reason=raw.get("reason", ""),
                evidence=raw.get("evidence", {}),
                cost_usd=per_flag_cost,
            ))

    return results
