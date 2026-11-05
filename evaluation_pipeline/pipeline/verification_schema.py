"""Verdict and report models for scoring reviewer flags against injected errors."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .review_schema import ReviewerSystem


InjectedCopyId = Literal["v1", "v2", "v3"]


class SubVerdict(BaseModel):
    judge_provider: str
    judge_model: str

    detected: bool
    matching_evidence: str = Field(
        default="",
        description="Verbatim excerpt from the review if detected, else empty.",
    )
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(description="1-2 sentences on why.")

    escalated_with_pdf: bool = Field(
        default=False,
        description="True if this subverdict came from the PDF-attached escalation.",
    )


class VerificationVerdict(BaseModel):
    """Per-injection final verdict: majority over the judge panel."""

    injection_ref: str = Field(
        description='Stable ref: "<module>-<severity>-t<type>" (e.g. "citations-S1-t2").'
    )
    module: str

    detected: bool = Field(
        description="Final majority verdict. For split panels with no majority → False with ambiguous=True."
    )
    ambiguous: bool = Field(
        default=False,
        description="True when the judge panel split without a clear majority "
                    "(e.g. 1-1 on a 2-judge citations panel with no tiebreak). "
                    "These are reported separately, not as undetected.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Aggregated confidence — lowest across judges whose vote counted in the majority.",
    )
    matching_evidence: str = Field(
        default="",
        description="Evidence text from the judge(s) who voted 'detected'."
    )
    reasoning: str = Field(
        description="Short aggregate reason; distinguishes unanimous vs majority-split vs tiebreak."
    )
    escalated_with_pdf: bool = Field(
        default=False,
        description="True if any subverdict was produced via the PDF-attached escalation.",
    )

    judge_panel: List[SubVerdict] = Field(
        description="Full trail of per-judge subverdicts (including tiebreak/escalation passes).",
    )

    extraction_source: Optional[str] = Field(
        default=None,
        description=(
            "For citations module only.  How the L3 ref.py pipeline "
            "discovered the statement that matches this injection: "
            '"primary" (Pass-1 statement_extractor), "second_pass" '
            "(Pass-2 suspicious-statement extractor), or "
            '"not_extracted" (neither pass picked it up — the injection '
            "is a total miss at the extraction layer).  Computed by "
            "verifier.py via fuzzy match of `truth.claim_sentence_verbatim` "
            "against the L3 raw JSON's checklist statement list."
        ),
    )


class VerificationReport(BaseModel):
    """One report per (paper, injected copy, reviewer system[, tag]).

        `reviewer_tag` is the panel provider for L1/L2 or the run mode for B3.
    """

    paper_id: str
    copy_id: InjectedCopyId
    reviewer_system: ReviewerSystem
    reviewer_tag: Optional[str] = None

    verdicts: List[VerificationVerdict]

    per_module_n_judges: Dict[str, int] = Field(
        description="Judge-panel size per module as actually used (honours any --uniform override)."
    )

    started_at: datetime
    finished_at: datetime
    wall_seconds: float

    total_cost_usd: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    escalation_count: int = 0

    notes: Optional[str] = None

    def detection_count(self) -> int:
        return sum(1 for v in self.verdicts if v.detected)

    def ambiguous_count(self) -> int:
        return sum(1 for v in self.verdicts if v.ambiguous)
