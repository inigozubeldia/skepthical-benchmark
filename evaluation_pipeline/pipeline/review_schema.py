"""Pydantic models for reviewer outputs."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# Labels are dot-free ("L25" for L2.5): agent names are dot-delimited.
ReviewerSystem = Literal[
    "L1", "L2", "L25", "L3", "B1", "B2", "B3", "C", "CL1", "CL25",
]
CopyId = Literal["v0", "v1", "v2", "v3"]
Provider = Literal[
    "anthropic", "openai", "gemini",
    "hosted", "self-hosted", "cloudrun",
    "claude-code-subscription",
]


class ReviewMeta(BaseModel):
    """Metadata saved alongside every review (`<copy>_meta.json`)."""

    paper_id: str
    copy_id: CopyId
    reviewer_system: ReviewerSystem

    provider: Provider
    model: str

    started_at: datetime
    finished_at: datetime
    wall_seconds: float

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    cost_usd: Optional[float] = None

    md_path: str = Field(description="Relative path of the markdown review file")
    pdf_path: Optional[str] = Field(default=None, description="L3 only — relative path of rendered PDF review")

    raw_response_path: Optional[str] = Field(default=None, description="Optional: raw LLM response JSON for audit")
    notes: Optional[str] = None


class ReviewReport(BaseModel):
    md_text: str
    meta: ReviewMeta
    raw_response: Optional[dict] = None
