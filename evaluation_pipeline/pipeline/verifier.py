"""Score reviewer flags against injected errors.

One VerificationReport per (paper, injected copy, reviewer system), with one
verdict per injection: the majority of a three-judge cross-family panel.
Low-confidence or ambiguous verdicts are re-judged with the paper PDF attached.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .cost import log_call
from .llm_client import Provider, call_text, call_with_pdf
from .review_schema import ReviewerSystem
from .verification_schema import (
    InjectedCopyId,
    SubVerdict,
    VerificationReport,
    VerificationVerdict,
)


@dataclass(frozen=True)
class Judge:
    provider: Provider
    model: str

    @property
    def tag(self) -> str:
        return f"{self.provider}:{self.model}"


DEFAULT_PANEL: Tuple[Judge, Judge, Judge] = (
    Judge("anthropic", "claude-sonnet-4-6"),
    Judge("openai",    "gpt-5.4"),
    Judge("gemini",    "gemini-3-flash-preview"),
)


DEFAULT_N_JUDGES: Dict[str, int] = {
    "citations": 3,
    "numerics": 3,
    "mathematics": 3,
    "general": 3,
}


def _injection_ref(module: str, severity: str, t: int) -> str:
    return f"{module}-{severity}-t{t}"


def _load_copy_injections(manifest_path: Path, copy_id: str) -> List[Dict[str, Any]]:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    for copy in m["copies"]:
        if copy["copy_id"] == copy_id:
            return copy["injections"]
    raise RuntimeError(f"no copy {copy_id!r} in manifest {manifest_path}")


def _load_review_md(paper_dir: Path, system: str, copy_id: str,
                    tag: Optional[str] = None) -> str:
    """`tag` is the panel provider or B3 mode; None reads `<copy>.md`."""
    stem = copy_id if not tag else f"{copy_id}__{tag}"
    path = paper_dir / "reviews" / system / f"{stem}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"review MD not found: {path} — run `scripts/review_paper.py` "
            f"for {system}{':'+tag if tag else ''} × {copy_id} first"
        )
    return path.read_text(encoding="utf-8")


def _normalise_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _load_l3_citation_sources(
    paper_dir: Path, copy_id: str, reviewer_system: str,
    reviewer_tag: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Citations checklist item from the L3 review's `_raw.json`, or None."""
    if reviewer_system != "L3":
        return None
    stem = copy_id if not reviewer_tag else f"{copy_id}__{reviewer_tag}"
    raw_path = paper_dir / "reviews" / reviewer_system / f"{stem}_raw.json"
    if not raw_path.is_file():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    checklist: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        checklist = raw.get("checklist") or []
        if not checklist:
            for k in ("verified_report", "merged_report", "report"):
                v = raw.get(k)
                if isinstance(v, dict) and v.get("checklist"):
                    checklist = v["checklist"]
                    break
    for item in checklist:
        if not isinstance(item, dict):
            continue
        if "statements" not in item or "sources" not in item:
            continue
        return item
    return None


def _compute_extraction_source(
    inj: Dict[str, Any],
    citation_checklist: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Source tag (`primary`, `second_pass` or `not_extracted`) of the L3 statement
        that best matches a citation injection; None for other modules.
    """
    if inj.get("module") != "citations":
        return None
    if not citation_checklist:
        return "not_extracted"
    truth = inj.get("truth") or {}
    target = (truth.get("claim_sentence_verbatim") or "").strip()
    if not target:
        return "not_extracted"
    norm_target = _normalise_for_match(target)
    statements = citation_checklist.get("statements") or []
    sources = citation_checklist.get("sources") or []
    if not statements:
        return "not_extracted"
    def _src_at(i: int) -> str:
        if i < len(sources) and isinstance(sources[i], str):
            return sources[i]
        return "primary"
    # A shared 8-token shingle counts as a match; robust to LaTeX vs plain text.
    target_tokens = norm_target.split()
    if len(target_tokens) < 4:
        return "not_extracted"
    target_shingles = {
        " ".join(target_tokens[i : i + 8])
        for i in range(max(1, len(target_tokens) - 7))
    }
    best_idx = -1
    best_score = 0
    for i, s in enumerate(statements):
        norm_s = _normalise_for_match(str(s))
        s_tokens = norm_s.split()
        if len(s_tokens) < 4:
            continue
        s_shingles = {
            " ".join(s_tokens[j : j + 8])
            for j in range(max(1, len(s_tokens) - 7))
        }
        score = len(target_shingles & s_shingles)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0 or best_score == 0:
        return "not_extracted"
    return _src_at(best_idx)


def _copy_pdf(paper_dir: Path, copy_id: str) -> Path:
    path = paper_dir / copy_id / f"{copy_id}.pdf"
    if not path.is_file():
        raise FileNotFoundError(
            f"copy PDF not found: {path} — run `scripts/compile_versions.py` first"
        )
    return path


_JUDGE_PROMPT_HEADER = """\
You are a scorer for the Skepthical-1 AI-review evaluation study. A \
reviewer system has produced a peer-review report on a scientific \
paper that has had 4 errors deliberately INJECTED into it. For each \
injection below you receive the full structured ground truth \
(module, severity, type, and the verbatim text that was changed). \
Your job: for EACH of the 4 injections, decide whether the reviewer's \
report contains a flag that corresponds to that specific injection.

A detection must be SPECIFIC: vague complaints about "citations look \
sketchy" or "there are some issues" DO NOT count as detection of a \
particular injection. The reviewer must have flagged something that \
clearly corresponds to the edit we made. When in doubt, prefer \
`confidence: low` so a secondary pass can resolve ambiguity.

NOTES:
- A reviewer may flag pre-existing defects in the paper (issues the \
  authors already had). Those are NOT credited as detections.
- A single reviewer-flag may correspond to ONE or to MORE injections. \
  Evaluate each injection independently.
- The reviewer did NOT know which errors were injected or where; they \
  reviewed the whole paper.

PER-MUTATION-KIND DETECTION RULES (citations module).
The reviewer cannot reasonably name a "dropped qualifier" or
"removed scope condition" — they did not see the original text. The
operative check depends on `truth.mutation_kind`:

(A) BROADENED-CLAIM rule.  For `mutation_kind` in:
  * `claim_condition_dropped_or_over_generalised`
  * `claim_methodology_drift`
  * `claim_method_category_swap`
  * `claim_topic_mis_attribution`
  * `claim_direction_reversal`

a reviewer flag of the *post-edit* claim sentence as `[CONTRADICTED]`,
`[PARTIALLY_SUPPORTED]`, "overgeneralised", "not supported by",
"contradicted by", or "mischaracterises" the SAME cited source counts
as DETECTION even if the specific dropped qualifier / scope condition
is not named in the report. The operative check: did the reviewer
challenge the post-edit claim against the cited source it points to?
If yes — detected.

(B) VALUE-OR-NUMERIC rule.  For `mutation_kind` in:
  * `claim_qualifier_within_error`
  * `claim_qualifier_1to2_sigma`
  * `claim_numerical_mis_quote`

the reviewer must either NAME the cited paper and challenge its value,
or QUOTE the specific value (or a same-quantity value) as inconsistent
with the cited source. A flag that names neither the cited paper nor
the value is NOT detection.

(C) For all other `mutation_kind` values — `cite_key_swap_*` (cite-key
swap variants), or any future enum addition — the standard "specific
match" rule applies: the reviewer must flag something clearly
corresponding to the edit (the swapped key, the inserted/deleted
citation), not a vague "citations look sketchy".

OUTPUT FORMAT — STRICT JSON ONLY, no prose, no markdown, no code fences:

{
  "verdicts": [
    {
      "injection_ref": "<module>-<severity>-t<type>",
      "detected": true | false,
      "matching_evidence": "<verbatim quote(s) from the review — empty if not detected>",
      "confidence": "high" | "medium" | "low",
      "reasoning": "<1-2 sentences>"
    },
    ... exactly 4 entries, one per injection, in the order given below
  ]
}
"""


def _format_injection(inj: Dict[str, Any]) -> str:
    ref = _injection_ref(inj["module"], inj["severity"], inj["type"])
    truth = inj.get("truth", {})
    lines: List[str] = []
    lines.append(f"### Injection {ref}")
    lines.append(f"- module: {inj['module']}")
    lines.append(f"- severity: {inj['severity']}")
    lines.append(f"- type: {inj['type']}")
    lines.append(f"- file: {inj['file_path']}")
    if inj.get("rationale"):
        lines.append(f"- proposer rationale: {inj['rationale']}")
    if inj.get("ground_truth_description"):
        lines.append(f"- ground-truth summary: {inj['ground_truth_description']}")

    orig = (inj.get("original_text") or "").strip()
    injected = (inj.get("injected_text") or "").strip()
    if len(orig) > 1200:
        orig = orig[:600] + " [...] " + orig[-600:]
    if len(injected) > 1200:
        injected = injected[:600] + " [...] " + injected[-600:]
    lines.append(f"- original_text (before edit): {orig!r}")
    lines.append(f"- injected_text (after edit):  {injected!r}")

    if truth.get("kind") == "citations":
        for k in ("mutation_kind", "original_cite_key", "injected_cite_key",
                  "section_location", "detection_target_verdict"):
            if truth.get(k):
                lines.append(f"- truth.{k}: {truth[k]}")
        if truth.get("claim_sentence_verbatim"):
            cs = truth["claim_sentence_verbatim"]
            if len(cs) > 400:
                cs = cs[:200] + " [...] " + cs[-200:]
            lines.append(f"- truth.claim_sentence_verbatim: {cs!r}")
    elif truth.get("kind") == "numerics":
        for k in ("predicate", "value_before_raw", "value_after_raw",
                  "quantity_label", "paired_location", "expected_bucket"):
            if truth.get(k) is not None:
                lines.append(f"- truth.{k}: {truth[k]}")
        if truth.get("location_cue"):
            cue = truth["location_cue"]
            if len(cue) > 300:
                cue = cue[:300] + "…"
            lines.append(f"- truth.location_cue: {cue!r}")
    elif truth.get("kind") == "mathematics":
        for k in ("equation_label", "equation_location", "term_role",
                  "expected_engine", "expected_verifier_outcome", "units_critical"):
            if truth.get(k) is not None:
                lines.append(f"- truth.{k}: {truth[k]}")
    elif truth.get("kind") == "general":
        for k in ("checklist_category", "checklist_item_id", "change_kind",
                  "expected_bucket", "section_location"):
            if truth.get(k) is not None:
                lines.append(f"- truth.{k}: {truth[k]}")
    return "\n".join(lines)


def _build_prompt(injections: List[Dict[str, Any]], review_md: str,
                  reviewer_system: str) -> str:
    injections_block = "\n\n".join(_format_injection(inj) for inj in injections)
    ordered_refs = ", ".join(
        _injection_ref(inj["module"], inj["severity"], inj["type"]) for inj in injections
    )
    return (
        _JUDGE_PROMPT_HEADER
        + f"\n\nReviewer system that produced the report below: **{reviewer_system}**\n"
        + f"\nINJECTION ORDER (return verdicts in exactly this order):\n  {ordered_refs}\n"
        + "\n\n=== INJECTIONS (ground truth) ===\n\n"
        + injections_block
        + "\n\n=== REVIEWER REPORT (markdown) ===\n\n"
        + review_md
        + "\n\n=== END ===\n"
        + "\nReturn the JSON now."
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate, strict=False)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _call_anthropic_text(
    *, model: str, prompt: str, paper_id: str, paper_dir: Path,
    agent_name: str, max_tokens: int = 16000,
    api_keys: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    import anthropic
    key = (api_keys or {}).get("ANTHROPIC") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    from skepthical.llm_call import make_hardened_anthropic_client
    client = make_hardened_anthropic_client(key)
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, timeout=3600.0,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(resp, "usage", None)
    in_t = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    out_t = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
    log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=in_t, output_tokens=out_t,
        provider="anthropic",
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
        extras={"provider": "anthropic", "stage": "verifier"},
    )
    parts: List[str] = []
    for blk in resp.content or []:
        t = getattr(blk, "text", None)
        if isinstance(t, str) and t:
            parts.append(t)
    return "\n".join(parts).strip(), in_t, out_t


def _call_openai_text(
    *, model: str, prompt: str, paper_id: str, paper_dir: Path,
    agent_name: str, max_tokens: int = 16000,
    api_keys: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    """Text-only OpenAI Responses call; reasoning is kept minimal since the task is bounded."""
    key = (api_keys or {}).get("OPENAI") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    from skepthical.llm_call import make_hardened_openai_client
    client = make_hardened_openai_client(key)

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "max_output_tokens": max_tokens,
    }
    mlow = model.lower()
    if mlow.startswith(("gpt-5", "gpt-5.", "o1", "o3", "o4")):
        kwargs["reasoning"] = {"effort": "none"}
        kwargs["text"] = {"verbosity": "low"}
    resp = client.responses.create(**kwargs)
    text = getattr(resp, "output_text", None) or ""
    if not text:
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for blk in getattr(item, "content", []) or []:
                    t = getattr(blk, "text", None)
                    if isinstance(t, str) and t:
                        parts.append(t)
        text = "\n".join(parts).strip()
    usage = getattr(resp, "usage", None)
    in_t = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0) if usage else 0
    out_t = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0) if usage else 0
    cache_read = 0
    if usage is not None:
        details = (
            getattr(usage, "input_tokens_details", None)
            or getattr(usage, "prompt_tokens_details", None)
        )
        if details is not None:
            cache_read = int(getattr(details, "cached_tokens", 0) or 0)
    log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=in_t, output_tokens=out_t,
        provider="openai",
        cache_read_tokens=cache_read,
        extras={"provider": "openai", "stage": "verifier"},
    )
    return text, in_t, out_t


def _call_gemini_text(
    *, model: str, prompt: str, paper_id: str, paper_dir: Path,
    agent_name: str, max_tokens: int = 16000,
    api_keys: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    key = (
        (api_keys or {}).get("GEMINI")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    try:
        from google import genai
        from google.genai import types as gtypes
        client = genai.Client(api_key=key)
        cfg_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
        m = model.lower()
        if "flash" in m or "2.5-pro" in m or m.startswith("gemini-3"):
            try:
                cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        config = gtypes.GenerateContentConfig(**cfg_kwargs)
        resp = client.models.generate_content(
            model=model, contents=[prompt], config=config,
        )
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_t = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_t = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    except ModuleNotFoundError:
        import google.generativeai as legacy
        legacy.configure(api_key=key)
        m = legacy.GenerativeModel(model)
        resp = m.generate_content([prompt], generation_config={"max_output_tokens": max_tokens})
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_t = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_t = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    cache_read = int(getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0
    log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=in_t, output_tokens=out_t,
        provider="gemini",
        cache_read_tokens=cache_read,
        extras={"provider": "gemini", "stage": "verifier"},
    )
    return text, in_t, out_t


def _is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(k in msg for k in ("503", "unavailable", "overloaded",
                              "rate limit", "429", "timeout",
                              "connection reset", "service unavailable")):
        return True
    return False


def _call_judge_text(
    judge: Judge, *, prompt: str, paper_id: str, paper_dir: Path,
    agent_name: str, max_tokens: int = 16000,
    api_keys: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    """Dispatch to the provider, retrying transient errors with exponential backoff."""
    import time as _t
    last_exc: Optional[BaseException] = None
    delays = [10.0, 30.0, 90.0]
    for attempt in range(len(delays) + 1):
        try:
            if judge.provider == "anthropic":
                return _call_anthropic_text(model=judge.model, prompt=prompt, paper_id=paper_id,
                                            paper_dir=paper_dir, agent_name=agent_name,
                                            max_tokens=max_tokens, api_keys=api_keys)
            if judge.provider == "openai":
                return _call_openai_text(model=judge.model, prompt=prompt, paper_id=paper_id,
                                         paper_dir=paper_dir, agent_name=agent_name,
                                         max_tokens=max_tokens, api_keys=api_keys)
            if judge.provider == "gemini":
                return _call_gemini_text(model=judge.model, prompt=prompt, paper_id=paper_id,
                                         paper_dir=paper_dir, agent_name=agent_name,
                                         max_tokens=max_tokens, api_keys=api_keys)
            raise ValueError(f"unknown judge provider: {judge.provider}")
        except Exception as e:
            last_exc = e
            if attempt >= len(delays) or not _is_transient_error(e):
                raise
            _t.sleep(delays[attempt])
    raise last_exc  # type: ignore[misc]
    raise ValueError(f"unknown judge provider {judge.provider!r}")


def _run_judge_pass(
    judge: Judge,
    *,
    injections: List[Dict[str, Any]],
    review_md: str,
    reviewer_system: str,
    paper_id: str,
    paper_dir: Path,
    copy_id: str,
    pass_tag: str,
    pdf_path: Optional[Path] = None,
    api_keys: Optional[Dict[str, Any]] = None,
) -> Tuple[List[SubVerdict], int, int]:
    prompt = _build_prompt(injections, review_md, reviewer_system)
    agent = f"verifier.{copy_id}.{reviewer_system}.{pass_tag}.{judge.provider}"
    from .fp_schema import VerifierBatchOutput

    if pdf_path is None:
        result = call_text(
            provider=judge.provider, model=judge.model,
            prompt=prompt,
            agent_name=agent, paper_id=paper_id, paper_dir=paper_dir,
            max_tokens=32000, api_keys=api_keys,
            schema=VerifierBatchOutput,
        )
    else:
        result = call_with_pdf(
            provider=judge.provider, model=judge.model,
            prompt=prompt, pdf_path=pdf_path,
            agent_name=agent, paper_id=paper_id, paper_dir=paper_dir,
            max_tokens=32000, api_keys=api_keys,
            schema=VerifierBatchOutput,
        )

    text, in_t, out_t = result.text, result.input_tokens, result.output_tokens

    if result.parsed is not None:
        obj = result.parsed.model_dump()
    else:
        obj = _extract_json_object(text)
    if not obj or "verdicts" not in obj:
        raise RuntimeError(
            f"judge {judge.tag} pass={pass_tag} returned no parseable JSON: {text[:400]!r}"
        )

    verdicts_json = obj["verdicts"]
    if not isinstance(verdicts_json, list) or len(verdicts_json) != len(injections):
        raise RuntimeError(
            f"judge {judge.tag} pass={pass_tag} returned wrong number of verdicts: "
            f"{len(verdicts_json) if isinstance(verdicts_json, list) else 'not-a-list'} "
            f"vs {len(injections)} expected"
        )

    subs: List[SubVerdict] = []
    for v, inj in zip(verdicts_json, injections):
        subs.append(SubVerdict(
            judge_provider=judge.provider,
            judge_model=judge.model,
            detected=bool(v.get("detected", False)),
            matching_evidence=str(v.get("matching_evidence", "") or ""),
            confidence=v.get("confidence", "low") if v.get("confidence") in {"high", "medium", "low"} else "low",
            reasoning=str(v.get("reasoning", "") or ""),
            escalated_with_pdf=(pdf_path is not None),
        ))
    return subs, in_t, out_t


def _aggregate_module_verdict(
    *,
    module: str,
    n_required: int,
    primary_sub: SubVerdict,
    secondary_sub: Optional[SubVerdict],
    tertiary_sub: Optional[SubVerdict],
) -> Tuple[bool, bool, str, SubVerdict, List[SubVerdict], str]:
    """Per-module majority rule.

        Returns (final_detected, ambiguous, aggregate_reason,
        aggregate_confidence_source, counted_panel, summary).
    """
    if n_required == 1:
        return (
            primary_sub.detected,
            False,
            "single judge (as configured for this module)",
            primary_sub,
            [primary_sub],
            "unanimous (n=1)",
        )

    if n_required == 2:
        if secondary_sub is None:
            return (
                primary_sub.detected, False,
                "secondary missing; fell back to single judge",
                primary_sub, [primary_sub], "primary-only",
            )
        if primary_sub.detected == secondary_sub.detected:
            lower_conf = primary_sub if _conf_rank(primary_sub.confidence) <= _conf_rank(secondary_sub.confidence) else secondary_sub
            return (
                primary_sub.detected, False,
                "2-judge agreement",
                lower_conf,
                [primary_sub, secondary_sub],
                "unanimous (primary + secondary)",
            )
        if tertiary_sub is None:
            return (
                False, True,
                "2-judge disagreement + no tiebreak → ambiguous",
                primary_sub,
                [primary_sub, secondary_sub],
                "split (no tiebreak)",
            )
        return (
            tertiary_sub.detected, False,
            "2-judge disagreement resolved by tiebreak",
            tertiary_sub,
            [primary_sub, secondary_sub, tertiary_sub],
            "tiebreak (tertiary decides)",
        )

    if n_required == 3:
        if secondary_sub is None or tertiary_sub is None:
            return (
                primary_sub.detected, False,
                "3-judge panel incomplete; fell back to available judges",
                primary_sub, [primary_sub], "degraded",
            )
        votes = [primary_sub.detected, secondary_sub.detected, tertiary_sub.detected]
        agree_yes = sum(votes)
        if agree_yes >= 2:
            majority_subs = [s for s, v in zip(
                [primary_sub, secondary_sub, tertiary_sub], votes) if v]
            lowest_conf = min(majority_subs, key=lambda s: _conf_rank(s.confidence))
            return (
                True, False,
                f"3-judge majority ({agree_yes}/3 detected)",
                lowest_conf,
                [primary_sub, secondary_sub, tertiary_sub],
                f"majority-{agree_yes}/3",
            )
        majority_subs = [s for s, v in zip(
            [primary_sub, secondary_sub, tertiary_sub], votes) if not v]
        lowest_conf = min(majority_subs, key=lambda s: _conf_rank(s.confidence))
        return (
            False, False,
            f"3-judge majority ({3 - agree_yes}/3 not-detected)",
            lowest_conf,
            [primary_sub, secondary_sub, tertiary_sub],
            f"majority-{3 - agree_yes}/3",
        )

    raise ValueError(f"unsupported n_required={n_required}")


_CONF_ORDER = {"high": 2, "medium": 1, "low": 0}


def _conf_rank(c: str) -> int:
    return _CONF_ORDER.get(c, 0)


def verify_review(
    *,
    paper_id: str,
    copy_id: InjectedCopyId,
    reviewer_system: ReviewerSystem,
    paper_dir: Path,
    reviewer_tag: Optional[str] = None,
    n_judges_per_module: Optional[Dict[str, int]] = None,
    uniform_n_judges: Optional[int] = None,
    panel: Tuple[Judge, Judge, Judge] = DEFAULT_PANEL,
    escalate_with_pdf: bool = True,
    api_keys: Optional[Dict[str, Any]] = None,
) -> VerificationReport:
    manifest_path = paper_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    injections = _load_copy_injections(manifest_path, copy_id)
    if not 1 <= len(injections) <= 4:
        raise RuntimeError(
            f"expected 1-4 injections on {copy_id}, got {len(injections)}"
        )
    if len(injections) < 4:
        print(
            f"[verifier] {copy_id}: only {len(injections)}/4 injections "
            "in manifest (some cells were skipped by the injector — "
            "denominator adjusts accordingly)",
            flush=True,
        )

    review_md = _load_review_md(paper_dir, reviewer_system, copy_id, tag=reviewer_tag)
    citation_checklist = _load_l3_citation_sources(
        paper_dir, copy_id, reviewer_system, reviewer_tag,
    )

    if uniform_n_judges is not None:
        effective_n = {m: int(uniform_n_judges) for m in ["citations", "numerics", "mathematics", "general"]}
    else:
        effective_n = dict(n_judges_per_module or DEFAULT_N_JUDGES)

    started = datetime.now(timezone.utc)
    total_in, total_out = 0, 0
    escalation_count = 0

    needs_secondary = any(effective_n[i["module"]] >= 2 for i in injections)
    needs_tertiary_unconditional = any(effective_n[i["module"]] >= 3 for i in injections)

    primary_subs, in_t, out_t = _run_judge_pass(
        panel[0], injections=injections, review_md=review_md,
        reviewer_system=reviewer_system, paper_id=paper_id, paper_dir=paper_dir,
        copy_id=copy_id, pass_tag="primary", api_keys=api_keys,
    )
    total_in += in_t; total_out += out_t
    primary_by_ref = {
        _injection_ref(inj["module"], inj["severity"], inj["type"]): s
        for inj, s in zip(injections, primary_subs)
    }

    secondary_by_ref: Dict[str, SubVerdict] = {}
    if needs_secondary:
        secondary_subs, in_t, out_t = _run_judge_pass(
            panel[1], injections=injections, review_md=review_md,
            reviewer_system=reviewer_system, paper_id=paper_id, paper_dir=paper_dir,
            copy_id=copy_id, pass_tag="secondary", api_keys=api_keys,
        )
        total_in += in_t; total_out += out_t
        for inj, s in zip(injections, secondary_subs):
            secondary_by_ref[_injection_ref(inj["module"], inj["severity"], inj["type"])] = s

    tertiary_by_ref: Dict[str, SubVerdict] = {}
    need_tertiary = needs_tertiary_unconditional
    if not need_tertiary and needs_secondary:
        for inj in injections:
            m = inj["module"]
            ref = _injection_ref(m, inj["severity"], inj["type"])
            if effective_n[m] == 2:
                if primary_by_ref[ref].detected != secondary_by_ref[ref].detected:
                    need_tertiary = True
                    break

    if need_tertiary:
        tertiary_subs, in_t, out_t = _run_judge_pass(
            panel[2], injections=injections, review_md=review_md,
            reviewer_system=reviewer_system, paper_id=paper_id, paper_dir=paper_dir,
            copy_id=copy_id, pass_tag="tertiary", api_keys=api_keys,
        )
        total_in += in_t; total_out += out_t
        for inj, s in zip(injections, tertiary_subs):
            tertiary_by_ref[_injection_ref(inj["module"], inj["severity"], inj["type"])] = s

    final_verdicts: List[VerificationVerdict] = []
    pdf_path = _copy_pdf(paper_dir, copy_id) if escalate_with_pdf else None

    for inj in injections:
        m = inj["module"]
        ref = _injection_ref(m, inj["severity"], inj["type"])
        n = effective_n[m]
        p_sub = primary_by_ref[ref]
        s_sub = secondary_by_ref.get(ref)
        t_sub = tertiary_by_ref.get(ref)

        final_detected, ambiguous, reason_detail, conf_src, counted_panel, summary = \
            _aggregate_module_verdict(
                module=m, n_required=n,
                primary_sub=p_sub, secondary_sub=s_sub, tertiary_sub=t_sub,
            )

        final_confidence = conf_src.confidence
        evidence = conf_src.matching_evidence if final_detected else ""

        escalated = False
        panel_trail: List[SubVerdict] = list(counted_panel)

        if (final_confidence == "low" or ambiguous) and pdf_path is not None:
            escalated = True
            escalation_count += 1
            esc_subs, in_t, out_t = _run_judge_pass(
                panel[0], injections=[inj], review_md=review_md,
                reviewer_system=reviewer_system, paper_id=paper_id, paper_dir=paper_dir,
                copy_id=copy_id, pass_tag="escalation", pdf_path=pdf_path,
                api_keys=api_keys,
            )
            total_in += in_t; total_out += out_t
            esc_sub = esc_subs[0]
            panel_trail.append(esc_sub)
            final_detected = esc_sub.detected
            final_confidence = esc_sub.confidence
            ambiguous = False
            reason_detail = "ambiguity-escalation with PDF attached"
            evidence = esc_sub.matching_evidence if final_detected else ""

        extraction_source = _compute_extraction_source(inj, citation_checklist)
        final_verdicts.append(VerificationVerdict(
            injection_ref=ref,
            module=m,
            detected=final_detected,
            ambiguous=ambiguous,
            confidence=final_confidence,
            matching_evidence=evidence,
            reasoning=f"{summary}: {reason_detail}",
            escalated_with_pdf=escalated,
            judge_panel=panel_trail,
            extraction_source=extraction_source,
        ))

    finished = datetime.now(timezone.utc)
    return VerificationReport(
        paper_id=paper_id, copy_id=copy_id, reviewer_system=reviewer_system,
        reviewer_tag=reviewer_tag,
        verdicts=final_verdicts,
        per_module_n_judges=effective_n,
        started_at=started, finished_at=finished,
        wall_seconds=(finished - started).total_seconds(),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        escalation_count=escalation_count,
    )


def write_verification(paper_dir: Path, report: VerificationReport) -> Path:
    out_dir = paper_dir / "verifications" / report.reviewer_system
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (report.copy_id if not report.reviewer_tag
            else f"{report.copy_id}__{report.reviewer_tag}")
    path = out_dir / f"{stem}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
