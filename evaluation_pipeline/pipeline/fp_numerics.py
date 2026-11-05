"""False-positive adjudicator for numerics flags.

All reviewers' flags for a paper are structured in one call, their values
re-read from the PDF, and checked by a mechanical predicate.
  value not found at the cited location -> FP
  predicate PASS                        -> FP
  predicate FAIL                        -> CONFIRMED_ISSUE
  predicate UNCERTAIN or flag skipped   -> UNADJUDICATABLE
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cost import combined_total_cost_usd as _total_cost, log_call
from .llm_client import _openai_get_or_upload
from .fp_schema import AdjudicationVerdict, FPFlag, Verdict


def _import_skepthical_numerics():
    """Return the skepthical.numerics module, adding its parent to sys.path."""
    try:
        import skepthical.numerics as _n
        return _n
    except ImportError:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import skepthical.numerics as _n
        return _n


def _import_schemas():
    try:
        from skepthical.schemas import NumericsCandidate, NumericsRereadOutput
        return NumericsCandidate, NumericsRereadOutput
    except ImportError:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from skepthical.schemas import NumericsCandidate, NumericsRereadOutput
        return NumericsCandidate, NumericsRereadOutput


def _import_predicates():
    try:
        from skepthical.numerics_predicates import run_predicate
        return run_predicate
    except ImportError:
        root = Path(__file__).parent.parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from skepthical.numerics_predicates import run_predicate
        return run_predicate


_CHECK_TYPE_ENUM_LINES = """\
check_type enum (use exactly one per candidate):
  "sum_eq_total"           — parts sum to a stated total
  "percent_to_100"         — percentages / fractions sum to 100 / 1
  "repeated_constant"      — the same constant is quoted multiple times
  "table_row_total"        — sum_eq_total over a table row
  "table_col_total"        — sum_eq_total over a table column
  "abstract_vs_body_match" — a value in the abstract matches its body restatement
  "rounded_match"          — two values agree after rounding to the coarser precision
  "integer_equal"          — exact integer equality
  "percent_of"             — base × percent / 100 = result
  "ratio_eq"               — numerator / denominator = ratio
  "difference_eq"          — minuend − subtrahend = difference
"""

_STRUCTURER_INSTRUCTIONS = (
    "You are converting reviewer-raised numerical flags into structured candidates "
    "for mechanical verification. You are given the paper PDF and each flag's claim.\n\n"
    "For each input flag, produce at most ONE candidate record. Skip flags you cannot "
    "map to a specific numeric in the paper (list them in 'skipped').\n\n"
    + _CHECK_TYPE_ENUM_LINES
    + "\nFor each candidate you produce:\n"
    "  id:             the input flag_id (unchanged)\n"
    "  check_type:     one of the enum above\n"
    "  location:       section / table / equation reference\n"
    "  statement:      one-sentence paraphrase of what is being verified\n"
    "  values:         list of objects, each with:\n"
    "    label:          role per check_type (e.g. 'part_1', 'total', 'percent_1')\n"
    "    raw:            exact numeric string as printed in the paper (e.g. '3.14', '42%')\n"
    "    value:          parsed float (without % or unit)\n"
    "    units:          unit string or null\n"
    "    location_cue:   verbatim ≥8-word phrase from the paper containing 'raw'\n"
    "  predicate_params: {}\n\n"
    "STRICT JSON — no prose, no code fences:\n"
    "{\n"
    '  "candidates": [ {NumericsCandidate}, ... ],\n'
    '  "skipped": [ {"flag_id": "...", "reason": "..."}, ... ]\n'
    "}\n\n"
    "Skip flags where you cannot find the exact raw value in the PDF, or where "
    "the flag is too vague to map to a specific numeric."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _build_structurer_prompt(flags: List[Dict[str, Any]]) -> str:
    items = [
        {"flag_id": f.get("flag_id", ""), "claim": f.get("claim", ""),
         "excerpt": f.get("excerpt", "")}
        for f in flags
    ]
    return _STRUCTURER_INSTRUCTIONS + "\n\nFLAGS:\n" + json.dumps(items, ensure_ascii=False, indent=2)


def _parse_structurer_output(
    text: str, all_flag_ids: List[str], NumericsCandidate: type,
) -> Tuple[list, List[str]]:
    s = (text or "").strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    candidates = []
    skipped_ids: List[str] = []

    try:
        obj = json.loads(s, strict=False)
    except json.JSONDecodeError:
        return [], list(all_flag_ids)

    for raw_cand in obj.get("candidates", []):
        try:
            cand = NumericsCandidate.model_validate(raw_cand)
            candidates.append(cand)
        except Exception:
            fid = raw_cand.get("id", "")
            if fid:
                skipped_ids.append(fid)

    for skip in obj.get("skipped", []):
        fid = skip.get("flag_id", "")
        if fid:
            skipped_ids.append(fid)

    accounted = {c.id for c in candidates} | set(skipped_ids)
    for fid in all_flag_ids:
        if fid not in accounted:
            skipped_ids.append(fid)

    return candidates, skipped_ids


def _adjudicate_flags_batch(
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

    NumericsCandidate, NumericsRereadOutput = _import_schemas()
    run_predicate = _import_predicates()
    _num = _import_skepthical_numerics()

    openai_key = (api_keys or {}).get("OPENAI") or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return {
            f.get("flag_id", ""): {
                "verdict": "UNADJUDICATABLE", "confidence": "low",
                "reason": "OPENAI_API_KEY not set.", "evidence": {},
            }
            for f in flags
        }

    from openai import OpenAI
    models = params_skepthical.get("models", {}) or {}
    rereader_model = models.get("numerics_rereader", "gpt-4.1")
    structurer_model = models.get("numerics_structurer", "gpt-4.1")
    timeout_s: Optional[float] = params_skepthical.get("timeout", None)

    from skepthical.llm_call import make_hardened_openai_client
    client = make_hardened_openai_client(openai_key)

    debug_dir = os.path.join(work_dir, "_fp_numerics_debug")
    os.makedirs(debug_dir, exist_ok=True)

    all_flag_ids = [f.get("flag_id", "") for f in flags]
    results: Dict[str, Dict[str, Any]] = {}
    uploaded_id: Optional[str] = None

    try:
        uploaded_id = _openai_get_or_upload(client, Path(pdf_path))

        structurer_prompt = _build_structurer_prompt(flags)
        struct_kwargs = _num._build_responses_kwargs(
            model=structurer_model, temperature=None, timeout_s=timeout_s,
            prompt_text=structurer_prompt, file_id=uploaded_id,
        )
        struct_kwargs["prompt_cache_key"] = f"paper-pdf-{uploaded_id}"
        from skepthical.schemas import FPNumericsStructurerOutput
        try:
            struct_resp = client.responses.parse(
                text_format=FPNumericsStructurerOutput, **struct_kwargs,
            )
        except Exception as _e:
            try:
                from openai import BadRequestError  # type: ignore
                _is_400 = isinstance(_e, BadRequestError)
            except Exception:
                _is_400 = "BadRequest" in type(_e).__name__
            if not _is_400:
                raise
            struct_resp = client.responses.create(**struct_kwargs)
        usage = getattr(struct_resp, "usage", None)
        in_t = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        out_t = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cached_t = 0
        if usage is not None:
            details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_t = int(getattr(details, "cached_tokens", 0) or 0)
        log_call(
            paper_id=paper_id, paper_dir=paper_dir,
            agent_name="fp_numerics_structurer", model=structurer_model,
            input_tokens=in_t, output_tokens=out_t,
            provider="openai", cache_read_tokens=cached_t,
        )

        struct_text = _num._extract_output_text(struct_resp)
        candidates, skipped_ids = _parse_structurer_output(struct_text, all_flag_ids, NumericsCandidate)

        for fid in skipped_ids:
            results[fid] = {
                "verdict": "UNADJUDICATABLE", "confidence": "low",
                "reason": "Flag-structurer could not map to a specific numeric in the paper.",
                "evidence": {"structurer_status": "skipped"},
            }

        if not candidates:
            return results

        rereader_out = _num._run_rereader(
            client=client, model=rereader_model, temperature=None, timeout_s=timeout_s,
            file_id=uploaded_id, candidates=candidates, debug_dir=debug_dir,
        )
        rr_by_id = {rc.id: rc for rc in rereader_out.candidates}

        for cand in candidates:
            rr_cand = rr_by_id.get(cand.id)

            if rr_cand is None:
                results[cand.id] = {
                    "verdict": "FP", "confidence": "medium",
                    "reason": "Re-reader could not locate the flagged numeric in the paper.",
                    "evidence": {"rereader_status": "not_returned"},
                }
                continue

            all_none = all(v.raw_observed is None and v.value_observed is None for v in rr_cand.values)
            if all_none:
                results[cand.id] = {
                    "verdict": "FP", "confidence": "medium",
                    "reason": "Re-reader found the location but could not read any numeric value there.",
                    "evidence": {"rereader_status": "no_value_found"},
                }
                continue

            pred_result = run_predicate(cand)
            pred_verdict = pred_result.get("verdict", "UNCERTAIN")
            pred_notes = pred_result.get("notes", "")

            if pred_verdict == "PASS":
                final_v, conf = "FP", "high"
                reason = f"Mechanical predicate passed: {pred_notes}" if pred_notes else "Predicate confirmed the numeric is correct."
            elif pred_verdict == "FAIL":
                final_v, conf = "CONFIRMED_ISSUE", "high"
                reason = f"Mechanical predicate failed: {pred_notes}" if pred_notes else "Predicate found a numerical discrepancy."
            else:
                final_v, conf = "UNADJUDICATABLE", "low"
                reason = f"Predicate uncertain: {pred_notes}" if pred_notes else "Predicate could not evaluate."

            results[cand.id] = {
                "verdict": final_v, "confidence": conf, "reason": reason,
                "evidence": {
                    "check_type": cand.check_type,
                    "predicate": pred_result,
                    "rereader": [rv.model_dump() for rv in rr_cand.values],
                },
            }

    except Exception as exc:
        print(f"[fp_numerics] unexpected error: {type(exc).__name__}: {exc}", flush=True)
        for fid in all_flag_ids:
            if fid not in results:
                results[fid] = {
                    "verdict": "UNADJUDICATABLE", "confidence": "low",
                    "reason": f"Adjudicator error: {type(exc).__name__}: {exc}",
                    "evidence": {},
                }
    finally:
        pass

    return results


def adjudicate_numerics_flags(
    *,
    flags: List[FPFlag],
    pdf_path: Path,
    paper_id: str,
    paper_dir: Path,
    work_dir: Path,
    params_skepthical: Dict[str, Any],
    api_keys: Optional[Dict[str, Any]] = None,
) -> List[AdjudicationVerdict]:
    if not flags:
        return []

    flag_dicts = [f.as_dict() for f in flags]

    cost_at_start = _total_cost()
    from skepthical.cost import ledger_context as _ledger_ctx
    with _ledger_ctx(
        paper_id=paper_id, paper_dir=paper_dir,
        method="fp_numerics",
    ):
        raw_results = _adjudicate_flags_batch(
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
    for flag in flags:
        id_to_reviewers.setdefault(flag.flag_id, [])
        if flag.reviewer_system not in id_to_reviewers[flag.flag_id]:
            id_to_reviewers[flag.flag_id].append(flag.reviewer_system)

    results: List[AdjudicationVerdict] = []
    seen: set[str] = set()
    n_unique = len({f.flag_id for f in flags})
    per_flag_cost = batch_cost / max(n_unique, 1)
    for flag in flags:
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
            module="numerics",
            verdict=verdict_raw,
            confidence=conf_raw,  # type: ignore[arg-type]
            reason=raw.get("reason", ""),
            evidence=raw.get("evidence", {}),
            cost_usd=per_flag_cost,
        ))

    return results
