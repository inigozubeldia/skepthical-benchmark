"""Extract major-severity flags from a reviewer report as FPFlag records.

One call per report; the extractor sees the review text only, not the paper.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .cost import log_call
from .fp_schema import FPFlag, Module, ReviewerSystem


_EXTRACTOR_INSTRUCTIONS = """\
You are the FLAG EXTRACTOR for a peer-review false-positive audit.

Your ONLY job is to extract MAJOR-severity issues from a single
reviewer's report on a scientific paper. You do NOT see the paper
itself — only the review text.

--- DISPATCH BY INPUT SHAPE ---
If the review contains an explicit H2 section whose heading includes the words
"major" (e.g. "## Major issues", "## Major Comments", "## Major concerns",
"## Major problems") OR a JSON field `major_issues`: extract ONLY items from
that section / field.
Ignore sections or fields named "Minor", "Very minor", "Strengths", "Summary",
"Weaknesses", prose paragraphs outside a major-issue section, and anything
labelled "editorial", "stylistic", "trivial", "rounding", "typo", "nit".

If the review has no such structure (free-form prose, or baselines with no
standard headers): read the whole report and identify only MAJOR-severity
issues based on the reviewer's own language.
  IN:  "critical", "major concern", "significant issue", "substantive problem",
       "fundamental flaw", "serious limitation", "incorrect", "wrong".
  OUT: "minor", "editorial", "stylistic", "nit", "small", "would benefit from",
       "could be clearer", numeric rounding / precision quibbles (last digit,
       trailing zeros) unless explicitly framed as substantive.

--- PER-ISSUE EXTRACTION ---
For each major-severity issue, determine its MODULE:

- "citations": the issue is about a cited paper failing to support or
  contradicting a claim. ONLY include if the reviewer asserts the citation
  is WRONG in one of these senses:
    * the cited paper does NOT support the claim at all
    * the cited paper CONTRADICTS the claim
    * the reviewer says the citation is UNRELATED to the claim
  Set citations_verdict to "CONTRADICTED" or "UNRELATED" accordingly.
  If the flag is "weak support" / "partial support" / "rephrase the citation"
  / "add another citation" — DO NOT EXTRACT. Skip.

- "numerics": the issue is about a numerical inconsistency — wrong value,
  sums/ratios/percentages that do not balance, unit errors,
  abstract-vs-body mismatches (>10×). DO NOT EXTRACT rounding quibbles,
  trailing-zero differences, or last-decimal precision unless the reviewer
  explicitly frames them as substantive.

- "mathematics": the issue is about a mathematical derivation, identity,
  normalisation, sign error, or dimensional inconsistency in an equation —
  things Wolfram / pint could check.

- "general": everything else the reviewer flagged as a major issue —
  methodological concerns, under-justified assumptions, missing ablations,
  scope limitations, etc.

--- OUTPUT FORMAT ---
Return STRICT JSON only — no prose, no code fences:

{
  "flags": [
    {
      "module": "citations" | "numerics" | "mathematics" | "general",
      "canonical_entity": "<bibkey if citations / equation id or location if maths / numeric location if numerics / short identifying phrase if general>",
      "claim": "<one-sentence description of what the reviewer is flagging>",
      "reported_severity": "major",
      "excerpt": "<verbatim snippet from the review, up to ~300 chars>",
      "citations_verdict": "CONTRADICTED" | "UNRELATED" | null
    },
    ...
  ]
}

RULES:
- Every extracted flag is MAJOR severity. Never emit minor / very_minor.
- canonical_entity: best-effort handle (bibkey, equation number, numeric
  location, or short phrase). Empty string if none can be determined.
- excerpt must be a verbatim substring of the review text.
- If the review contains zero major-severity issues, return {"flags": []}.
- citations_verdict is null for non-citations flags.
"""


def _flag_id(paper_id: str, module: str, canonical_entity: str, claim: str) -> str:
    """Stable content hash for cross-reviewer deduplication."""
    h = hashlib.sha256()
    h.update(paper_id.encode())
    h.update(b"|")
    h.update(module.encode())
    h.update(b"|")
    h.update(canonical_entity.lower().strip().encode())
    h.update(b"|")
    h.update(claim.lower().strip()[:200].encode())
    return "fp-" + h.hexdigest()[:12]


def extract_flags(
    *,
    report_text: str,
    reviewer_system: ReviewerSystem,
    paper_id: str,
    paper_dir: Path,
    api_keys: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4.1",
) -> List[FPFlag]:
    """Returns an empty list on parse failure (never raises)."""
    _model_lc = (model or "").strip().lower()
    if _model_lc.startswith("claude"):
        _provider = "anthropic"
    elif _model_lc.startswith("gemini"):
        _provider = "gemini"
    else:
        _provider = "openai"

    user_message = (
        f"REVIEWER_SYSTEM: {reviewer_system}\n\n"
        f"REVIEW:\n{report_text}"
    )

    text = ""
    in_t = 0
    out_t = 0
    cache_read = 0
    cache_creation = 0
    try:
        if _provider == "openai":
            key = (api_keys or {}).get("OPENAI") or os.environ.get("OPENAI_API_KEY")
            if not key:
                print(f"[fp_extractor] OPENAI_API_KEY not set; returning empty flags for {paper_id}/{reviewer_system}.")
                return []
            from skepthical.llm_call import make_hardened_openai_client
            client = make_hardened_openai_client(key)
            from .fp_schema import FPExtractorOutput
            _oai_kwargs = dict(
                model=model,
                instructions=_EXTRACTOR_INSTRUCTIONS,
                input=[{
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_message}],
                }],
                max_output_tokens=32000,
            )
            try:
                resp = client.responses.parse(
                    text_format=FPExtractorOutput, **_oai_kwargs,
                )
            except Exception as _e:
                try:
                    from openai import BadRequestError  # type: ignore
                    _is_400 = isinstance(_e, BadRequestError)
                except Exception:
                    _is_400 = "BadRequest" in type(_e).__name__
                if not _is_400:
                    raise
                resp = client.responses.create(**_oai_kwargs)
            usage = getattr(resp, "usage", None)
            in_t = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
            out_t = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            details = (
                getattr(usage, "input_tokens_details", None)
                or getattr(usage, "prompt_tokens_details", None)
            ) if usage else None
            if details is not None:
                cache_read = int(getattr(details, "cached_tokens", 0) or 0)
            text = getattr(resp, "output_text", None) or ""
            if not text:
                parts: list[str] = []
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) == "message":
                        for blk in getattr(item, "content", []) or []:
                            t = getattr(blk, "text", None)
                            if isinstance(t, str) and t:
                                parts.append(t)
                text = "\n".join(parts).strip()

        elif _provider == "anthropic":
            akey = (api_keys or {}).get("ANTHROPIC") or os.environ.get("ANTHROPIC_API_KEY")
            if not akey:
                print(f"[fp_extractor] ANTHROPIC_API_KEY not set; returning empty flags for {paper_id}/{reviewer_system}.")
                return []
            from skepthical.llm_call import make_hardened_anthropic_client
            anth = make_hardened_anthropic_client(akey)
            from .fp_schema import FPExtractorOutput
            _anth_kwargs = dict(
                model=model,
                max_tokens=32000,
                system=_EXTRACTOR_INSTRUCTIONS,
                messages=[{"role": "user", "content": user_message}],
            )
            try:
                resp = anth.messages.parse(
                    output_format=FPExtractorOutput, **_anth_kwargs,
                )
            except Exception as _e:
                try:
                    from anthropic import BadRequestError  # type: ignore
                    _is_400 = isinstance(_e, BadRequestError)
                except Exception:
                    _is_400 = "BadRequest" in type(_e).__name__
                if not _is_400:
                    raise
                resp = anth.messages.create(**_anth_kwargs)
            usage = getattr(resp, "usage", None)
            in_t = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
            out_t = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
            parts = []
            for blk in resp.content or []:
                t = getattr(blk, "text", None)
                if isinstance(t, str) and t:
                    parts.append(t)
            text = "\n".join(parts).strip()

        else:
            try:
                from google import genai  # type: ignore
                from google.genai import types as gtypes  # type: ignore
            except ModuleNotFoundError:
                print(f"[fp_extractor] google-genai SDK missing; returning empty flags.")
                return []
            gkey = (api_keys or {}).get("GEMINI") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not gkey:
                print(f"[fp_extractor] GEMINI_API_KEY not set; returning empty flags for {paper_id}/{reviewer_system}.")
                return []
            real_model = model
            thinking_level: Optional[str] = None
            for lvl in ("none", "off", "low", "medium", "high"):
                suf = f"-thinking-{lvl}"
                if real_model.endswith(suf):
                    real_model = real_model[: -len(suf)]
                    thinking_level = "none" if lvl == "off" else lvl
                    break
            else:
                if real_model.endswith("-thinking"):
                    real_model = real_model[: -len("-thinking")]
                    thinking_level = "high"
            if thinking_level is None:
                thinking_level = "none"
            cfg_kwargs: Dict[str, Any] = {
                "max_output_tokens": 8000,
                "system_instruction": _EXTRACTOR_INSTRUCTIONS,
            }
            _budget_map = {
                "none": 0, "off": 0,
                "low": 2048, "medium": 8192, "high": 24576, "xhigh": 32768,
            }
            _budget = _budget_map.get(thinking_level)
            if _budget is not None:
                try:
                    cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(
                        thinking_budget=_budget)
                except Exception:
                    pass
            from .fp_schema import FPExtractorOutput
            from skepthical.llm_call import _to_strict_json_schema as _strictify
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_json_schema"] = _strictify(
                FPExtractorOutput.model_json_schema()
            )
            config = gtypes.GenerateContentConfig(**cfg_kwargs)
            gclient = genai.Client(api_key=gkey)
            resp = gclient.models.generate_content(
                model=real_model, contents=[user_message], config=config,
            )
            text = getattr(resp, "text", "") or ""
            usage = getattr(resp, "usage_metadata", None)
            # Gemini reports total prompt tokens (cached + uncached).
            in_t = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
            out_t = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
            cache_read = int(getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0

    except Exception as exc:
        print(f"[fp_extractor] LLM call failed for {paper_id}/{reviewer_system} ({_provider}): {exc}")
        return []

    log_call(
        paper_id=paper_id,
        paper_dir=paper_dir,
        agent_name="fp_extractor",
        model=model,
        input_tokens=in_t,
        output_tokens=out_t,
        provider=_provider,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
        extras={"reviewer_system": reviewer_system, "provider": _provider},
    )

    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"[fp_extractor] JSON parse failed for {paper_id}/{reviewer_system}: {exc}")
        return []

    flags: List[FPFlag] = []
    for item in data.get("flags", []):
        module: Module = item.get("module", "general")
        if module == "maths":
            module = "mathematics"
        if module not in ("citations", "numerics", "mathematics", "general"):
            module = "general"
        entity = item.get("canonical_entity", "") or ""
        claim = item.get("claim", "") or ""
        if not claim:
            continue
        f = FPFlag(
            paper_id=paper_id,
            reviewer_system=reviewer_system,
            module=module,
            canonical_entity=entity,
            claim=claim,
            reported_severity=item.get("reported_severity", "major"),
            excerpt=item.get("excerpt", "") or "",
            citations_verdict=item.get("citations_verdict") or None,
        )
        f.flag_id = _flag_id(paper_id, f.module, f.canonical_entity, f.claim)
        flags.append(f)

    return flags
