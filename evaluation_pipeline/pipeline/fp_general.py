"""False-positive adjudicator for general issues.

Three cross-family judges each receive the paper PDF and the batch of
deduplicated major general flags for that paper. Flags with the same
normalised claim are adjudicated once and attributed to every reviewer
that raised them.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fp_schema import (
    AdjudicationVerdict,
    FPFlag,
    ReviewerSystem,
    SubVerdict,
    Verdict,
)
from .llm_client import call_with_pdf


_SYSTEM = """\
You are a peer-review fact-checker. The attached PDF is the paper under review.
Read it carefully before adjudicating the flagged issues listed in the prompt.\
"""

_PANEL_PROMPT = """\
Below are flagged issues raised by an automated reviewer. For each flag,
judge whether it is a REAL issue with the paper or a SPURIOUS (false-positive)
flag. Base your verdict ONLY on the attached paper PDF.

--- FLAGGED ISSUES ---
{flags_json}

--- TASK ---
For each flag, return one verdict:
- "CONFIRMED_ISSUE": the paper genuinely has the problem the reviewer
  describes; the reviewer's specific concern is NOT resolved by the paper.
- "FP": the reviewer's specific concern is fully resolved or factually
  incorrect — i.e., (a) the paper performs exactly the analysis the reviewer
  asks for, (b) the reviewer's factual claim about the paper is wrong, or
  (c) the flag is so vague it makes no testable assertion.
- "AMBIGUOUS": genuinely unclear — could go either way.

--- CRITICAL RULES — read carefully ---

(R1) **"Acknowledged" ≠ "Addressed".** If the reviewer asks for a
*quantitative* assessment (e.g. "quantify the bias", "estimate the
magnitude", "show the distribution"), and the paper only *acknowledges* the
limitation in words or labels it as "future work", that is a CONFIRMED_ISSUE,
NOT an FP. Mere acknowledgement does not satisfy a request for quantification.

(R2) **Match the exact quantity.** Before voting FP, verify that the paper
content you are citing addresses the REVIEWER'S EXACT TECHNICAL CLAIM, not a
related-but-different quantity. Common conflations to avoid:
  - Training-time stochasticity (random init, SGD seed) ≠ data-realisation
    stochasticity (variability across catalogues / mock datasets).
  - Bias of estimator vs. truth ≠ agreement between two estimators
    (e.g. "SBI vs likelihood" is a different metric from "SBI vs truth").
  - Validity at one fiducial point ≠ validity across a wider prior volume.
  - Information content under one inference method ≠ information content
    under another (e.g. losslessness for likelihood ≠ losslessness for SBI).
If the paper text you would cite addresses a *related* quantity but not the
exact one the reviewer flagged, the verdict is CONFIRMED_ISSUE or
AMBIGUOUS — not FP.

(R3) **Cite the exact text.** Your "reason" must quote (in quotes) the
specific paper passage that resolves or fails to resolve the flag. If you
cannot find such a passage, you cannot vote FP.

(R4) **When uncertain, say AMBIGUOUS.** A 2-1 vote with one AMBIGUOUS
dissenter blocks the FP label downstream, so do not feel forced into a
binary call.

Also return for each verdict:
- confidence: "high" | "medium" | "low"
- reason: 1-2 sentences citing the specific paper passage (in quotes)

--- OUTPUT ---
STRICT JSON only — no prose, no code fences:
{{
  "verdicts": [
    {{
      "flag_id": "<id>",
      "verdict": "CONFIRMED_ISSUE" | "FP" | "AMBIGUOUS",
      "confidence": "high" | "medium" | "low",
      "reason": "<1-2 sentences, with quoted paper passage>"
    }},
    ...
  ]
}}
"""

_JUDGES: List[Tuple[str, str, str, Optional[Dict[str, Any]]]] = [
    ("anthropic", "claude-sonnet-4-6",       "fp_general_panel_1", None),
    ("openai",    "gpt-5.4",                 "fp_general_panel_2", {"effort": "medium"}),
    ("gemini",    "gemini-3.1-pro-preview",  "fp_general_panel_3", None),
]

_CONF_ORDER = {"high": 2, "medium": 1, "low": 0}


_PUNCT = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")


def _normalise_claim(claim: str) -> str:
    s = claim.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _dedup_key(paper_id: str, claim: str) -> str:
    normalised = _normalise_claim(claim)[:200]
    h = hashlib.sha256((paper_id + "|" + normalised).encode()).hexdigest()[:16]
    return h


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_verdicts(text: str) -> List[Dict[str, Any]]:
    s = text.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, dict):
        return obj.get("verdicts", [])
    return []


def _aggregate_3panel(
    sub_verdicts: List[SubVerdict],
) -> Tuple[Verdict, str, str]:
    """Aggregate three sub-verdicts into (verdict, confidence, reason).

        Biased against the FP label: >=2 CONFIRMED_ISSUE -> CONFIRMED_ISSUE;
        unanimous FP -> FP; anything else -> AMBIGUOUS.
    """
    counts: Dict[str, int] = {}
    for sv in sub_verdicts:
        counts[sv.verdict] = counts.get(sv.verdict, 0) + 1

    n_confirmed = counts.get("CONFIRMED_ISSUE", 0)
    n_fp = counts.get("FP", 0)

    if n_confirmed >= 2:
        majority_svs = [sv for sv in sub_verdicts if sv.verdict == "CONFIRMED_ISSUE"]
        conf_vals = [_CONF_ORDER.get(sv.confidence, 0) for sv in majority_svs]
        conf_str = ["low", "medium", "high"][min(conf_vals)]
        return "CONFIRMED_ISSUE", conf_str, majority_svs[0].reason

    if n_fp == 3:
        majority_svs = [sv for sv in sub_verdicts if sv.verdict == "FP"]
        conf_vals = [_CONF_ORDER.get(sv.confidence, 0) for sv in majority_svs]
        conf_str = ["low", "medium", "high"][min(conf_vals)]
        return "FP", conf_str, majority_svs[0].reason

    if n_fp == 2:
        dissent = [sv for sv in sub_verdicts if sv.verdict != "FP"]
        reason = (
            f"FP majority blocked by non-FP dissent — "
            f"{dissent[0].judge_provider}={dissent[0].verdict}: {dissent[0].reason}"
        )
        return "AMBIGUOUS", "low", reason

    reason = "Three-panel split: " + ", ".join(
        f"{sv.judge_provider}={sv.verdict}" for sv in sub_verdicts
    )
    return "AMBIGUOUS", "low", reason


def adjudicate_general_flags(
    *,
    flags: List[FPFlag],
    pdf_path: Optional[Path],
    paper_id: str,
    paper_dir: Path,
    api_keys: Optional[Dict[str, Any]] = None,
    dedup_cache: Optional[Dict[str, AdjudicationVerdict]] = None,
) -> List[AdjudicationVerdict]:
    if dedup_cache is None:
        dedup_cache = {}

    if pdf_path is None or not pdf_path.is_file():
        return [
            AdjudicationVerdict(
                flag_id=f.flag_id, paper_id=paper_id,
                reviewer_systems_attributed=[f.reviewer_system],
                module="general", verdict="UNADJUDICATABLE",
                confidence="low", reason="Paper PDF not found.",
                evidence={},
            )
            for f in flags
        ]

    key_to_flag: Dict[str, FPFlag] = {}
    key_to_reviewers: Dict[str, List[ReviewerSystem]] = {}
    for flag in flags:
        dk = _dedup_key(paper_id, flag.claim)
        if dk not in key_to_flag:
            key_to_flag[dk] = flag
            key_to_reviewers[dk] = []
        if flag.reviewer_system not in key_to_reviewers[dk]:
            key_to_reviewers[dk].append(flag.reviewer_system)

    results: List[AdjudicationVerdict] = []

    pending_keys = [dk for dk in key_to_flag if key_to_flag[dk].flag_id not in dedup_cache]
    pending_flags = [key_to_flag[dk] for dk in pending_keys]

    if pending_flags:
        flags_for_prompt = [
            {"flag_id": f.flag_id, "claim": f.claim, "excerpt": f.excerpt}
            for f in pending_flags
        ]
        flags_json = json.dumps(flags_for_prompt, ensure_ascii=False, indent=2)
        prompt = _PANEL_PROMPT.format(flags_json=flags_json)

        judge_verdicts: Dict[str, List[SubVerdict]] = {f.flag_id: [] for f in pending_flags}

        n_pending = max(len(pending_flags), 1)
        for provider, model, agent_name, reasoning in _JUDGES:
            try:
                from .fp_schema import FPGeneralPanelOutput
                result = call_with_pdf(
                    provider=provider,  # type: ignore[arg-type]
                    model=model,
                    prompt=prompt,
                    pdf_path=pdf_path,
                    agent_name=agent_name,
                    paper_id=paper_id,
                    paper_dir=paper_dir,
                    system=_SYSTEM,
                    max_tokens=32000,
                    api_keys=api_keys,
                    reasoning=reasoning,
                    schema=FPGeneralPanelOutput,
                )
                raw = result.text
            except Exception as exc:
                print(f"[fp_general] judge {provider}/{model} failed: {exc}")
                continue

            # One call covers the batch; split its cost evenly across the flags.
            per_flag_cost = float(result.cost_usd) / n_pending

            for item in _parse_verdicts(raw):
                fid = item.get("flag_id", "")
                verdict_raw = item.get("verdict", "AMBIGUOUS")
                if verdict_raw not in ("CONFIRMED_ISSUE", "FP", "AMBIGUOUS", "UNADJUDICATABLE"):
                    verdict_raw = "AMBIGUOUS"
                conf_raw = item.get("confidence", "low")
                if conf_raw not in ("high", "medium", "low"):
                    conf_raw = "low"
                if fid in judge_verdicts:
                    judge_verdicts[fid].append(SubVerdict(
                        judge_provider=provider,
                        judge_model=model,
                        verdict=verdict_raw,  # type: ignore[arg-type]
                        confidence=conf_raw,  # type: ignore[arg-type]
                        reason=item.get("reason", ""),
                        cost_usd=per_flag_cost,
                    ))

        for flag in pending_flags:
            svs = judge_verdicts.get(flag.flag_id, [])
            if not svs:
                final_verdict: Verdict = "UNADJUDICATABLE"
                final_conf = "low"
                final_reason = "No judge produced a verdict."
            elif len(svs) == 1:
                final_verdict = svs[0].verdict
                final_conf = svs[0].confidence
                final_reason = svs[0].reason
            elif len(svs) == 2:
                if svs[0].verdict == svs[1].verdict:
                    final_verdict = svs[0].verdict
                    conf_vals = [_CONF_ORDER.get(sv.confidence, 0) for sv in svs]
                    final_conf = ["low", "medium", "high"][min(conf_vals)]
                    final_reason = svs[0].reason
                else:
                    final_verdict = "AMBIGUOUS"
                    final_conf = "low"
                    final_reason = (
                        f"Two-judge split: {svs[0].judge_provider}={svs[0].verdict}, "
                        f"{svs[1].judge_provider}={svs[1].verdict}"
                    )
            else:
                final_verdict, final_conf, final_reason = _aggregate_3panel(svs)

            adv = AdjudicationVerdict(
                flag_id=flag.flag_id,
                paper_id=paper_id,
                reviewer_systems_attributed=[],
                module="general",
                verdict=final_verdict,
                confidence=final_conf,  # type: ignore[arg-type]
                reason=final_reason,
                evidence={"claim": flag.claim, "excerpt": flag.excerpt},
                judge_panel=svs,
                cost_usd=sum(sv.cost_usd for sv in svs),
            )
            dedup_cache[flag.flag_id] = adv

    for dk in key_to_flag:
        flag = key_to_flag[dk]
        reviewers = key_to_reviewers[dk]
        if flag.flag_id in dedup_cache:
            adv = dedup_cache[flag.flag_id]
            adv.reviewer_systems_attributed = list(
                set(list(adv.reviewer_systems_attributed) + reviewers)
            )
            results.append(adv)
        else:
            results.append(AdjudicationVerdict(
                flag_id=flag.flag_id,
                paper_id=paper_id,
                reviewer_systems_attributed=reviewers,
                module="general",
                verdict="UNADJUDICATABLE",
                confidence="low",
                reason="All judges failed.",
                evidence={"claim": flag.claim, "excerpt": flag.excerpt},
            ))

    return results
