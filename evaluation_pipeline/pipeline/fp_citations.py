"""False-positive adjudicator for citation flags.

Two cross-family judges read the cited paper's PDF; a third breaks ties.
Only CONTRADICTED and UNRELATED flags are adjudicated. Results are cached
per (paper, cited arXiv ID, claim) and attributed to every reviewer that
raised the flag.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cost import combined_total_cost_usd as _total_cost, log_call
from .fp_schema import AdjudicationVerdict, FPFlag, ReviewerSystem, SubVerdict, Verdict
from .llm_client import call_with_pdf

_SUPPORTED_VERDICTS = {"SUPPORTED", "WEAKLY_SUPPORTED", "PARTIALLY_SUPPORTED"}
_WRONG_VERDICTS = {"CONTRADICTED", "UNRELATED"}

_SECONDARY_MODEL = "claude-sonnet-4-6"
_TIEBREAK_MODEL = "gemini-3.1-pro-preview"

_VERIF_PROMPT = """\
You are an expert scientific reviewer. Using ONLY the attached paper(s),
classify the following statement's relationship to the attached evidence
into exactly one of these verdicts:
  - SUPPORTED: the statement's specific claims (including quantitative detail)
    are directly supported by the attached paper(s).
  - WEAKLY_SUPPORTED: the statement is consistent with the attached paper(s)
    but supporting detail is qualitative or indirect.
  - PARTIALLY_SUPPORTED: some aspects are supported and others are not
    addressed or are at odds with the evidence.
  - CONTRADICTED: the attached paper(s) make claims that contradict the statement.
  - UNRELATED: the attached paper(s) do not discuss the statement's subject matter.

Statement: {claim}

Cited paper label: {cite_label}

Return JSON only:
{{"verdict": "SUPPORTED"|"WEAKLY_SUPPORTED"|"PARTIALLY_SUPPORTED"|"CONTRADICTED"|"UNRELATED",
  "reason": "<brief explanation>"}}
"""

_VALID_VERDICTS = {"SUPPORTED", "WEAKLY_SUPPORTED", "PARTIALLY_SUPPORTED",
                   "CONTRADICTED", "UNRELATED", "UNVERIFIABLE"}


def _dedup_key(paper_id: str, arxiv_id: Optional[str], claim: str) -> str:
    h = hashlib.sha256()
    h.update(paper_id.encode())
    h.update(b"|")
    h.update((arxiv_id or "").lower().encode())
    h.update(b"|")
    h.update(claim.lower().strip()[:200].encode())
    return h.hexdigest()[:16]


def _parse_verif_json(text: str) -> Tuple[str, str]:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
    try:
        obj = json.loads(s)
        verdict = obj.get("verdict", "UNVERIFIABLE")
        reason = obj.get("reason", "")
        if verdict not in _VALID_VERDICTS:
            verdict = "UNVERIFIABLE"
        return verdict, reason
    except json.JSONDecodeError:
        return "UNVERIFIABLE", "JSON parse error"


def _resolve_arxiv_pdf(
    canonical_entity: str,
    reference_entries: List[str],
    work_dir: Path,
    params_skepthical: Dict[str, Any],
    api_keys: Dict[str, Any],
) -> Tuple[Optional[str], Optional[Path]]:
    """Resolve a bibkey/label to an arXiv ID and download the PDF; (None, None) on failure."""
    try:
        from skepthical.ref import resolve_reference_with_agent, download_arxiv_pdf
    except ImportError:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from skepthical.ref import resolve_reference_with_agent, download_arxiv_pdf
        except ImportError:
            print("[fp_citations] cannot import skepthical.ref — skipping PDF resolution.")
            return None, None

    if not reference_entries:
        return None, None

    result = resolve_reference_with_agent(
        canonical_entity, reference_entries, params_skepthical, api_keys,
    )
    if result.get("resolution_status") != "resolved":
        return None, None

    arxiv_id = result.get("arxiv_id")
    if not arxiv_id:
        return None, None

    pdf_path = download_arxiv_pdf(arxiv_id, str(work_dir))
    if pdf_path is None or not Path(str(pdf_path)).exists():
        return arxiv_id, None

    return arxiv_id, Path(str(pdf_path))


def _verify_primary(
    claim: str,
    pdf_path: Path,
    arxiv_id: str,
    params_skepthical: Dict[str, Any],
    api_keys: Dict[str, Any],
    work_dir: Path,
) -> Tuple[str, str]:
    try:
        from skepthical.ref import verify_claim_with_paper
    except ImportError:
        return "UNVERIFIABLE", "Could not import skepthical.ref"
    try:
        verdict, reason = verify_claim_with_paper(
            claim, [str(pdf_path)], params_skepthical, api_keys,
            ref_labels=[arxiv_id], work_dir=str(work_dir),
        )
        return verdict, reason
    except Exception as exc:
        return "UNVERIFIABLE", f"Primary verifier exception: {exc}"


def _verify_secondary_or_tiebreak(
    claim: str,
    pdf_path: Path,
    provider: str,
    model: str,
    arxiv_id: str,
    paper_id: str,
    paper_dir: Path,
    agent_name: str,
    api_keys: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    prompt = _VERIF_PROMPT.format(
        claim=claim,
        cite_label=arxiv_id,
    )
    try:
        from .fp_schema import FPCitationsVerdictModel
        result = call_with_pdf(
            provider=provider,
            model=model,
            prompt=prompt,
            pdf_path=str(pdf_path),
            agent_name=agent_name,
            paper_id=paper_id,
            paper_dir=paper_dir,
            api_keys=api_keys,
            schema=FPCitationsVerdictModel,
        )
        return _parse_verif_json(result.text)
    except Exception as exc:
        return "UNVERIFIABLE", f"{provider}/{model} exception: {exc}"


def _majority_2plus1(
    primary: str, secondary: str, tiebreak: Optional[str],
) -> str:
    def _direction(v: str) -> str:
        if v in _SUPPORTED_VERDICTS:
            return "fp"
        if v in _WRONG_VERDICTS:
            return "issue"
        return "unknown"

    dp, ds = _direction(primary), _direction(secondary)
    if dp == ds and dp != "unknown":
        return primary
    if tiebreak is not None:
        dt = _direction(tiebreak)
        if dp == dt and dp != "unknown":
            return primary
        if ds == dt and ds != "unknown":
            return secondary
    return "UNVERIFIABLE"


def _fp_verdict_from_verif(
    aggregated_verif: str,
    reviewer_citation_verdict: Optional[str],
) -> Verdict:
    if aggregated_verif in ("UNVERIFIABLE",):
        return "UNADJUDICATABLE"
    if aggregated_verif in _SUPPORTED_VERDICTS:
        return "FP"
    if aggregated_verif in _WRONG_VERDICTS:
        return "CONFIRMED_ISSUE"
    return "UNADJUDICATABLE"


def adjudicate_one_citation_flag(
    flag: FPFlag,
    *,
    reference_entries: List[str],
    work_dir: Path,
    paper_id: str,
    paper_dir: Path,
    params_skepthical: Dict[str, Any],
    api_keys: Optional[Dict[str, Any]] = None,
    dedup_cache: Optional[Dict[str, AdjudicationVerdict]] = None,
) -> AdjudicationVerdict:
    if dedup_cache is None:
        dedup_cache = {}

    ak = api_keys or {}
    cost_at_start = _total_cost()

    arxiv_id, pdf_path = _resolve_arxiv_pdf(
        flag.canonical_entity, reference_entries, work_dir, params_skepthical, ak,
    )

    if pdf_path is None:
        cost_used = _total_cost() - cost_at_start
        return AdjudicationVerdict(
            flag_id=flag.flag_id,
            paper_id=paper_id,
            reviewer_systems_attributed=[flag.reviewer_system],
            module="citations",
            verdict="UNADJUDICATABLE",
            confidence="low",
            reason=f"Could not retrieve cited paper PDF for {flag.canonical_entity!r}.",
            evidence={"arxiv_id": arxiv_id, "canonical_entity": flag.canonical_entity},
            cost_usd=max(cost_used, 0.0),
        )

    arxiv_id_str = arxiv_id or flag.canonical_entity

    dk = _dedup_key(paper_id, arxiv_id_str, flag.claim)
    if dk in dedup_cache:
        adv = dedup_cache[dk]
        if flag.reviewer_system not in adv.reviewer_systems_attributed:
            adv.reviewer_systems_attributed.append(flag.reviewer_system)
        return adv

    primary_verdict, primary_reason = _verify_primary(
        flag.claim, pdf_path, arxiv_id_str,
        params_skepthical, ak, work_dir,
    )

    secondary_verdict, secondary_reason = _verify_secondary_or_tiebreak(
        flag.claim, pdf_path,
        provider="anthropic", model=_SECONDARY_MODEL,
        arxiv_id=arxiv_id_str, paper_id=paper_id, paper_dir=paper_dir,
        agent_name="fp_citations_secondary", api_keys=ak,
    )

    def _dir(v: str) -> str:
        return "fp" if v in _SUPPORTED_VERDICTS else "issue" if v in _WRONG_VERDICTS else "unknown"

    tiebreak_verdict: Optional[str] = None
    tiebreak_reason = ""
    if _dir(primary_verdict) != _dir(secondary_verdict):
        tiebreak_verdict, tiebreak_reason = _verify_secondary_or_tiebreak(
            flag.claim, pdf_path,
            provider="gemini", model=_TIEBREAK_MODEL,
            arxiv_id=arxiv_id_str, paper_id=paper_id, paper_dir=paper_dir,
            agent_name="fp_citations_tiebreak", api_keys=ak,
        )

    aggregated = _majority_2plus1(primary_verdict, secondary_verdict, tiebreak_verdict)
    final_verdict: Verdict = _fp_verdict_from_verif(aggregated, flag.citations_verdict)

    judge_panel: List[SubVerdict] = [
        SubVerdict(
            judge_provider="openai",
            judge_model=params_skepthical.get("models", {}).get("paper_verification_agent", "gpt-5"),
            verdict=final_verdict if primary_verdict not in _VALID_VERDICTS else (
                "FP" if primary_verdict in _SUPPORTED_VERDICTS else
                "CONFIRMED_ISSUE" if primary_verdict in _WRONG_VERDICTS else "UNADJUDICATABLE"
            ),
            confidence="medium",
            reason=primary_reason,
        ),
        SubVerdict(
            judge_provider="anthropic",
            judge_model=_SECONDARY_MODEL,
            verdict=(
                "FP" if secondary_verdict in _SUPPORTED_VERDICTS else
                "CONFIRMED_ISSUE" if secondary_verdict in _WRONG_VERDICTS else "UNADJUDICATABLE"
            ),
            confidence="medium",
            reason=secondary_reason,
        ),
    ]
    if tiebreak_verdict:
        judge_panel.append(SubVerdict(
            judge_provider="gemini",
            judge_model=_TIEBREAK_MODEL,
            verdict=(
                "FP" if tiebreak_verdict in _SUPPORTED_VERDICTS else
                "CONFIRMED_ISSUE" if tiebreak_verdict in _WRONG_VERDICTS else "UNADJUDICATABLE"
            ),
            confidence="medium",
            reason=tiebreak_reason,
        ))

    cost_used = _total_cost() - cost_at_start
    adv = AdjudicationVerdict(
        flag_id=flag.flag_id,
        paper_id=paper_id,
        reviewer_systems_attributed=[flag.reviewer_system],
        module="citations",
        verdict=final_verdict,
        confidence="medium",
        reason=f"Primary: {primary_verdict}. Secondary: {secondary_verdict}."
               + (f" Tiebreak: {tiebreak_verdict}." if tiebreak_verdict else ""),
        evidence={
            "arxiv_id": arxiv_id_str,
            "canonical_entity": flag.canonical_entity,
            "primary_verdict": primary_verdict,
            "secondary_verdict": secondary_verdict,
            "tiebreak_verdict": tiebreak_verdict,
            "aggregated_verif": aggregated,
        },
        judge_panel=judge_panel,
        cost_usd=max(cost_used, 0.0),
    )
    dedup_cache[dk] = adv
    return adv


def adjudicate_many_citation_flags(
    flags: List[FPFlag],
    *,
    reference_entries: List[str],
    work_dir: Path,
    paper_id: str,
    paper_dir: Path,
    params_skepthical: Dict[str, Any],
    api_keys: Optional[Dict[str, Any]] = None,
    dedup_cache: Optional[Dict[str, AdjudicationVerdict]] = None,
) -> List[AdjudicationVerdict]:
    if dedup_cache is None:
        dedup_cache = {}
    results: List[AdjudicationVerdict] = []
    from skepthical.cost import ledger_context as _ledger_ctx
    with _ledger_ctx(
        paper_id=paper_id, paper_dir=paper_dir,
        method="fp_citations",
    ):
        for flag in flags:
            adv = adjudicate_one_citation_flag(
                flag,
                reference_entries=reference_entries,
                work_dir=work_dir,
                paper_id=paper_id,
                paper_dir=paper_dir,
                params_skepthical=params_skepthical,
                api_keys=api_keys,
                dedup_cache=dedup_cache,
            )
            results.append(adv)
    return results
