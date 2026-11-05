"""Schemas for the false-positive pipeline.

One FPFlag per reviewer-raised issue on a clean (v0) paper, one
AdjudicationVerdict per deduplicated flag, one PaperFPReport per
(paper, reviewer system).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _StrictFPModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

ReviewerSystem = Literal[
    "L1", "L2", "L3",
    "L25",
    "B1",
    "B2",
    "B3-Fast", "B3-Std", "B3-Deep",
    "C",
    "CL1",
    "CL25",
]
Module = Literal["citations", "numerics", "mathematics", "general"]
Verdict = Literal["CONFIRMED_ISSUE", "FP", "AMBIGUOUS", "UNADJUDICATABLE"]


@dataclass
class FPFlag:
    paper_id: str
    reviewer_system: ReviewerSystem
    module: Module
    canonical_entity: str
    claim: str
    reported_severity: str
    excerpt: str
    citations_verdict: Optional[str] = None
    flag_id: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "paper_id": self.paper_id,
            "reviewer_system": self.reviewer_system,
            "module": self.module,
            "canonical_entity": self.canonical_entity,
            "claim": self.claim,
            "reported_severity": self.reported_severity,
            "excerpt": self.excerpt,
            "citations_verdict": self.citations_verdict,
        }


class EligibilityScanOutput(_StrictFPModel):
    eligible: bool = True
    rationale: str = ""
    examples: List[str] = Field(default_factory=list)


class FPExtractedFlag(_StrictFPModel):
    """Extractor output for one flag, used as the strict-mode schema for all providers."""
    module: Literal["citations", "numerics", "mathematics", "general"] = "general"
    canonical_entity: str = ""
    claim: str
    reported_severity: Literal["major"] = "major"
    excerpt: str = ""
    citations_verdict: Optional[Literal["CONTRADICTED", "UNRELATED"]] = None


class FPExtractorOutput(_StrictFPModel):
    flags: List[FPExtractedFlag] = Field(default_factory=list)


class FPSubVerdictModel(_StrictFPModel):
    verdict: Literal["CONFIRMED_ISSUE", "FP", "AMBIGUOUS", "UNADJUDICATABLE"]
    confidence: Literal["high", "medium", "low"]
    reason: str


class VerifierBatchEntry(_StrictFPModel):
    injection_ref: str
    detected: bool
    matching_evidence: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    reasoning: str = ""


class VerifierBatchOutput(_StrictFPModel):
    verdicts: List[VerifierBatchEntry] = Field(default_factory=list)


class FPGeneralPanelEntry(_StrictFPModel):
    flag_id: str
    verdict: Literal["CONFIRMED_ISSUE", "FP", "AMBIGUOUS", "UNADJUDICATABLE"]
    confidence: Literal["high", "medium", "low"] = "low"
    reason: str = ""


class FPGeneralPanelOutput(_StrictFPModel):
    verdicts: List[FPGeneralPanelEntry] = Field(default_factory=list)


class FPCitationsVerdictModel(_StrictFPModel):
    verdict: Literal[
        "SUPPORTED", "WEAKLY_SUPPORTED", "PARTIALLY_SUPPORTED",
        "CONTRADICTED", "UNRELATED", "UNVERIFIABLE",
    ]
    reason: str = ""


@dataclass
class SubVerdict:
    judge_provider: str
    judge_model: str
    verdict: Verdict
    confidence: Literal["high", "medium", "low"]
    reason: str
    cost_usd: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reason": self.reason,
            "cost_usd": self.cost_usd,
        }


@dataclass
class AdjudicationVerdict:
    flag_id: str
    paper_id: str
    reviewer_systems_attributed: List[ReviewerSystem]
    module: Module
    verdict: Verdict
    confidence: Literal["high", "medium", "low"]
    reason: str
    evidence: Dict[str, Any]
    judge_panel: List[SubVerdict] = field(default_factory=list)
    cost_usd: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "paper_id": self.paper_id,
            "reviewer_systems_attributed": list(self.reviewer_systems_attributed),
            "module": self.module,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence": self.evidence,
            "judge_panel": [sv.as_dict() for sv in self.judge_panel],
            "cost_usd": self.cost_usd,
        }


@dataclass
class PaperFPReport:
    """Per-paper aggregate for one (paper, reviewer system[, provider])."""

    paper_id: str
    reviewer_system: ReviewerSystem
    per_module_verdicts: Dict[Module, List[AdjudicationVerdict]]
    per_module_fp_rate: Dict[Module, float]
    per_module_unadjudicated_count: Dict[Module, int]
    per_module_cost_usd: Dict[Module, float] = field(default_factory=dict)
    extractor_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    provider_tag: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "reviewer_system": self.reviewer_system,
            "provider_tag": self.provider_tag,
            "per_module_verdicts": {
                mod: [v.as_dict() for v in verdicts]
                for mod, verdicts in self.per_module_verdicts.items()
            },
            "per_module_fp_rate": dict(self.per_module_fp_rate),
            "per_module_unadjudicated_count": dict(self.per_module_unadjudicated_count),
            "per_module_cost_usd": dict(self.per_module_cost_usd),
            "extractor_cost_usd": self.extractor_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }
