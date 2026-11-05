"""Data formats."""
from __future__ import annotations
import math
import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
_RAW_NUMERIC_RE = re.compile('^[+\\-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+\\-]?\\d+)?$')

def _is_numeric_raw(s: str) -> bool:
    if not isinstance(s, str):
        return False
    cleaned = s.strip().replace(',', '').replace('\u202f', '').replace('\xa0', '').replace(' ', '')
    return bool(_RAW_NUMERIC_RE.match(cleaned))

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')
VERDICT_VALUES: tuple = ('SUPPORTED', 'WEAKLY_SUPPORTED', 'PARTIALLY_SUPPORTED', 'CONTRADICTED', 'UNRELATED', 'NOT_FOUND', 'RESOLUTION_FAILED', 'UNVERIFIABLE')
VERIFIER_VERDICT_VALUES: tuple = ('SUPPORTED', 'WEAKLY_SUPPORTED', 'PARTIALLY_SUPPORTED', 'CONTRADICTED', 'UNRELATED')

class ReferenceResolverOutput(StrictModel):
    arxiv_id: Optional[str] = None
    reference_text: Optional[str] = None

class VerifierVerdict(StrictModel):
    verdict: Literal['SUPPORTED', 'WEAKLY_SUPPORTED', 'PARTIALLY_SUPPORTED', 'CONTRADICTED', 'UNRELATED']
    reason: str

class MathsCheckedItem(StrictModel):
    id: str
    location: str
    claim: str
    inputs_assumptions: List[str] = Field(default_factory=list)
    check_type: List[str] = Field(default_factory=list)
    verdict: Literal['PASS', 'FAIL', 'UNCERTAIN']
    confidence: Literal['high', 'medium', 'low']
    impact: Literal['critical', 'moderate', 'minor']
    notes: str
    tools_used: List[str] = Field(default_factory=list, description="One short string per mechanical check, e.g. 'Wolfram: FullSimplify → True (confirmed, 1.0s)' or 'pint: dimensional match (confirmed, 0.001s)'.")
    mechanical_evidence_summary: Optional[str] = Field(default=None, description='One-paragraph prose summary of the mechanical evidence used to reach this verdict. Empty for LLM-only items.')

class MathsAuditOutput(StrictModel):
    math_relevance: Literal['none', 'light', 'substantial']
    math_summary: str
    checked_items: List[MathsCheckedItem] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    excluded_equations: List['MathsExcludedEquation'] = Field(default_factory=list, description='Numbered equations the extractor chose NOT to audit, with a reason. Exposes extractor self-selection so readers can judge scope.')
    tool_aggregate: Optional['MathsToolAggregate'] = Field(default=None, description='Per-tool call counts across all checked items. Rendered at the foot of the maths appendix.')
    pattern_notes: Optional[str] = Field(default=None, description="Cross-item patterns the aggregator spotted (e.g. 'two independent sign errors in §3 suggest that section was revised after §4').")

class MathsExcludedEquation(StrictModel):
    eq_ref: str = Field(description="Equation number or short location, e.g. '(12)' or 'Sec 4.2 first eq'.")
    reason: str = Field(description='Why the extractor did not audit this equation.')

class MathsToolAggregate(StrictModel):
    sympy_calls: int = 0
    wolfram_calls: int = 0
    pint_calls: int = 0
    llm_only_items: int = 0

class MathCheckSpec(StrictModel):
    engine: Literal['wolfram', 'pint', 'sympy']
    args: dict = Field(default_factory=dict)
    description: str = ''

class MathStatement(StrictModel):
    id: str = Field(description="Short human-friendly tag (e.g. 'gaussian_normalisation'); NOT a code like 'SO-EQ1'.")
    location: str = Field(description="Human-readable location, e.g. 'Eqs. (3)-(7), Sec 3.2, p.7'.")
    kind: Literal['equation', 'derivation', 'limit', 'normalisation', 'dimensional', 'definition', 'inequality', 'substitution', 'property', 'assumption']
    latex: List[str] = Field(default_factory=list, description='One or more LaTeX strings for the equations in this statement. For derivations, list them in order.')
    claim: str = Field(description='Plain-language restatement of what the statement asserts (the thing we want to verify).')
    assumptions: List[str] = Field(default_factory=list, description='Premises the claim relies on — declared symbol ranges, regularity conditions, boundary behaviour, etc.')
    importance: Literal['critical', 'moderate', 'minor'] = 'moderate'
    units_critical: bool = Field(default=False, description="True iff failure of dimensional consistency would invalidate the claim. Default False so we don't run pint spuriously.")
    check_plan: List[MathCheckSpec] = Field(default_factory=list, description='Mechanical checks the extractor wants the Stage 2 dispatcher to run. May be empty for statements that are LLM-only.')
    verbatim_source_snippet: Optional[str] = Field(default=None, description="A short verbatim excerpt of the paper text around the equation(s) this statement is derived from. Used by the Stage 1b transcription-fidelity check to detect extractor drift between the paper's actual math and the python_expr / wl_code passed to the mechanical layer.")

class MathsStage1bPerCheck(StrictModel):
    index: int
    fidelity: Literal['faithful', 'drifted', 'unverifiable']
    notes_fidelity: str = ''

class MathsStage1bOutput(StrictModel):
    overall_fidelity: Literal['faithful', 'drifted', 'unverifiable']
    knowledge_audit: Literal['matches', 'does_not_match', 'unknown', 'not_applicable']
    knowledge_audit_findings: List[str] = Field(default_factory=list)
    per_check: List[MathsStage1bPerCheck] = Field(default_factory=list)

class MathsStage3Output(StrictModel):
    verdict: Literal['PASS', 'FAIL', 'UNCERTAIN']
    confidence: Literal['high', 'medium', 'low']
    impact: Literal['critical', 'moderate', 'minor']
    notes: str = ''

class ToolResult(StrictModel):
    tool: Literal['sympy', 'wolfram', 'pint', 'llm']
    status: Literal['confirmed', 'refuted', 'inconclusive', 'error', 'unavailable', 'skipped']
    detail: str = ''
    seconds: float = 0.0
    description: str = ''

class MechanicalEvidence(StrictModel):
    statement_id: str
    results: List[ToolResult] = Field(default_factory=list)

    def summary_lines(self) -> List[str]:
        out = []
        for r in self.results:
            bits = [f'{r.tool}']
            if r.description:
                bits.append(f': {r.description}')
            bits.append(f' → {r.status}')
            if r.seconds >= 0.01:
                bits.append(f' ({r.seconds:.2f}s)')
            out.append(''.join(bits))
        return out
MathsAuditOutput.model_rebuild()
NumericsCheckType = Literal['sum_eq_total', 'percent_to_100', 'repeated_constant', 'table_row_total', 'table_col_total', 'abstract_vs_body_match', 'rounded_match', 'integer_equal', 'percent_of', 'ratio_eq', 'difference_eq']

class NumericsExtractedValue(BaseModel):
    label: str
    raw: str
    value: float
    units: Optional[str] = None
    location_cue: str
    raw_original_corrupt: Optional[str] = None

    @field_validator('value')
    @classmethod
    def _value_must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f'NumericsExtractedValue.value must be finite; got {v}')
        return v

    @model_validator(mode='after')
    def _recover_corrupted_raw(self) -> 'NumericsExtractedValue':
        if _is_numeric_raw(self.raw):
            return self
        if not math.isfinite(self.value):
            raise ValueError(f'NumericsExtractedValue has corrupted raw {self.raw[:80]!r} AND non-finite value {self.value} — cannot recover.')
        original = self.raw
        recovered: Optional[str] = None
        if self.location_cue:
            best = None
            best_err = math.inf
            for m in re.finditer('-?\\d+(?:\\.\\d+)?(?:[eE][+\\-]?\\d+)?', self.location_cue):
                tok = m.group(0)
                try:
                    cand = float(tok)
                except ValueError:
                    continue
                err = abs(cand - self.value) / max(abs(self.value), 1.0)
                if err < best_err:
                    best, best_err = (tok, err)
            if best is not None and best_err < 0.01:
                recovered = best
        if recovered is None:
            recovered = '{:g}'.format(self.value) if abs(self.value) < 1000000000000000.0 else repr(self.value)
        if not _is_numeric_raw(recovered):
            raise ValueError(f'NumericsExtractedValue recovery failed: original raw {original[:80]!r}, value={self.value}, recovered={recovered!r} (still not numeric-shaped).')
        object.__setattr__(self, 'raw', recovered)
        object.__setattr__(self, 'raw_original_corrupt', original)
        return self

class NumericsCandidate(BaseModel):
    id: str
    check_type: NumericsCheckType
    location: str
    statement: str
    values: List[NumericsExtractedValue] = Field(default_factory=list)
    predicate_params: dict = Field(default_factory=dict)

class NumericsExtractorOutput(BaseModel):
    candidates: List[NumericsCandidate] = Field(default_factory=list)
    unverified_items: List['NumericsUnverifiedItem'] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

class NumericsRereadValue(BaseModel):
    label: str
    raw_observed: Optional[str] = None
    value_observed: Optional[float] = None
    notes: str = ''

class NumericsRereadCandidate(BaseModel):
    id: str
    values: List[NumericsRereadValue] = Field(default_factory=list)

class NumericsRereadOutput(BaseModel):
    candidates: List[NumericsRereadCandidate] = Field(default_factory=list)

class NumericsUnverifiedItem(BaseModel):
    id: str
    location: str
    statement: str
    reason: str

class FPNumericsSkippedFlag(BaseModel):
    flag_id: str
    reason: str

class FPNumericsStructurerOutput(BaseModel):
    candidates: List[NumericsCandidate] = Field(default_factory=list)
    skipped: List[FPNumericsSkippedFlag] = Field(default_factory=list)

class FPMathsMissingFlag(BaseModel):
    flag_id: str
    reason: str = ''

class FPMathsStructurerOutput(BaseModel):
    statements: List[MathStatement] = Field(default_factory=list)
    missing: List[FPMathsMissingFlag] = Field(default_factory=list)
NumericsExtractorOutput.model_rebuild()
