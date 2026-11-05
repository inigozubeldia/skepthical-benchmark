"""Arithmetic checks for numerical statements."""
from __future__ import annotations
import re
from typing import Any, Callable, Dict, List
from .schemas import NumericsCandidate, NumericsExtractedValue
_NUMERIC_RAW_RE = re.compile('^[+\\-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+\\-]?\\d+)?$')

def _split_mantissa_exp(raw: str) -> tuple[str, int]:
    s = (raw or '').strip()
    if not s:
        raise ValueError('empty raw')
    s = s.replace(',', '').replace('\u202f', '').replace('\xa0', '').replace(' ', '')
    if not _NUMERIC_RAW_RE.match(s):
        raise ValueError(f'raw is not a numeric literal (after stripping separators): {(raw or '')[:80]!r}')
    if s.startswith(('+', '-')):
        s = s[1:]
    if 'e' in s.lower():
        mant, exp = s.lower().split('e', 1)
        return (mant, int(exp))
    return (s, 0)

def decimals_in_raw(raw: str) -> int:
    mant, exp = _split_mantissa_exp(raw)
    if '.' in mant:
        _, dec = mant.split('.', 1)
        return len(dec) - exp
    stripped = mant.rstrip('0') or '0'
    trailing = len(mant) - len(stripped)
    return -trailing - exp

def sigfig_count(raw: str) -> int:
    mant, _ = _split_mantissa_exp(raw)
    if '.' in mant:
        int_part, dec_part = mant.split('.', 1)
        int_part = int_part.lstrip('0')
        if int_part:
            return len(int_part) + len(dec_part)
        return len(dec_part.lstrip('0')) or 1
    stripped_leading = mant.lstrip('0') or '0'
    return len(stripped_leading)

def halfulp(raw: str) -> float:
    return 0.5 * 10.0 ** (-decimals_in_raw(raw))

def combined_additive_tol(raws: List[str]) -> float:
    return sum((halfulp(r) for r in raws))

def relative_tol_from_sigfigs(n_sigfigs: int) -> float:
    n = max(1, int(n_sigfigs))
    return 0.5 * 10.0 ** (-(n - 1))

def _mk(verdict: str, *, computed: Any=None, reported: Any=None, abs_tol: Any=None, rel_tol: Any=None, diff_abs: Any=None, diff_rel: Any=None, notes: str='') -> Dict[str, Any]:
    return {'verdict': verdict, 'computed': computed, 'reported': reported, 'diff_abs': diff_abs, 'diff_rel': diff_rel, 'tolerance': {'abs_tol': abs_tol, 'rel_tol': rel_tol}, 'notes': notes}

def _by_label(values: List[NumericsExtractedValue]) -> Dict[str, List[NumericsExtractedValue]]:
    out: Dict[str, List[NumericsExtractedValue]] = {}
    for v in values:
        out.setdefault(v.label, []).append(v)
    return out

def p_sum_eq_total(cand: NumericsCandidate) -> Dict[str, Any]:
    total_label = (cand.predicate_params or {}).get('total_label', 'total')
    labels = _by_label(cand.values)
    totals = labels.get(total_label, [])
    if len(totals) != 1:
        return _mk('UNCERTAIN', notes=f'expected exactly one value labelled {total_label!r}')
    parts = [v for v in cand.values if v.label != total_label]
    if len(parts) < 2:
        return _mk('UNCERTAIN', notes='need at least two part values')
    total = totals[0]
    part_sum = sum((v.value for v in parts))
    tol = combined_additive_tol([v.raw for v in parts] + [total.raw])
    diff = part_sum - total.value
    verdict = 'PASS' if abs(diff) <= tol else 'FAIL'
    return _mk(verdict, computed=part_sum, reported=total.value, abs_tol=tol, diff_abs=diff, notes=f'sum of {len(parts)} parts = {part_sum}; reported total = {total.raw}; worst-case additive half-ULP tolerance = {tol}')

def p_percent_to_100(cand: NumericsCandidate) -> Dict[str, Any]:
    parts = list(cand.values)
    if len(parts) < 2:
        return _mk('UNCERTAIN', notes='need at least two percentages/fractions')
    mx = max((v.value for v in parts))
    target = (cand.predicate_params or {}).get('target')
    if target is None:
        target = 100.0 if mx > 1.001 else 1.0
    target = float(target)
    s = sum((v.value for v in parts))
    tol = combined_additive_tol([v.raw for v in parts])
    diff = s - target
    verdict = 'PASS' if abs(diff) <= tol else 'FAIL'
    return _mk(verdict, computed=s, reported=target, abs_tol=tol, diff_abs=diff, notes=f'sum = {s}; target = {target}; additive half-ULP tolerance = {tol}')

def p_repeated_constant(cand: NumericsCandidate) -> Dict[str, Any]:
    vals = list(cand.values)
    if len(vals) < 2:
        return _mk('UNCERTAIN', notes='need at least two occurrences')
    min_sigf = min((sigfig_count(v.raw) for v in vals))
    rel = relative_tol_from_sigfigs(min_sigf)
    mx = max((v.value for v in vals))
    mn = min((v.value for v in vals))
    scale = max(abs(mx), abs(mn), 1e-300)
    diff_rel = (mx - mn) / scale
    abs_tol = max((halfulp(v.raw) for v in vals))
    diff_abs = mx - mn
    passes = diff_rel <= rel or diff_abs <= abs_tol
    return _mk('PASS' if passes else 'FAIL', computed=mx, reported=mn, abs_tol=abs_tol, rel_tol=rel, diff_abs=diff_abs, diff_rel=diff_rel, notes=f'{len(vals)} occurrences; min sigfigs = {min_sigf}; range = [{mn}, {mx}]')

def p_abstract_vs_body_match(cand: NumericsCandidate) -> Dict[str, Any]:
    return p_repeated_constant(cand)

def p_rounded_match(cand: NumericsCandidate) -> Dict[str, Any]:
    vals = list(cand.values)
    if len(vals) < 2:
        return _mk('UNCERTAIN', notes='need at least two values')
    decimals = (cand.predicate_params or {}).get('decimals')
    if decimals is None:
        decimals = min((decimals_in_raw(v.raw) for v in vals))
    decimals = int(decimals)
    a, b = (vals[0].value, vals[1].value)
    ra, rb = (round(a, decimals), round(b, decimals))
    step = 10.0 ** (-decimals)
    abs_tol = 0.5 * step
    diff = abs(a - b)
    return _mk('PASS' if ra == rb else 'FAIL', computed=ra, reported=rb, abs_tol=abs_tol, diff_abs=diff, notes=f'rounded to {decimals} decimals (step {step}); {a} vs {b}')

def p_integer_equal(cand: NumericsCandidate) -> Dict[str, Any]:
    vals = list(cand.values)
    if len(vals) < 2:
        return _mk('UNCERTAIN', notes='need at least two values')
    ints: List[int] = []
    for v in vals:
        i = int(round(v.value))
        if abs(v.value - i) > 1e-09:
            return _mk('UNCERTAIN', notes=f'non-integer input for integer_equal: {v.raw}')
        ints.append(i)
    verdict = 'PASS' if len(set(ints)) == 1 else 'FAIL'
    return _mk(verdict, computed=ints[0], reported=ints[-1], abs_tol=0, diff_abs=max(ints) - min(ints), notes=f'exact integer equality; values = {ints}')

def _combined_verdict_from_abs_or_rel(*, diff_abs: float, diff_rel: float, abs_tol: float, rel_tol: float) -> str:
    return 'PASS' if diff_abs <= abs_tol or diff_rel <= rel_tol else 'FAIL'

def p_percent_of(cand: NumericsCandidate) -> Dict[str, Any]:
    by = _by_label(cand.values)
    try:
        base = by['base'][0]
        pct = by['percent'][0]
        result = by['result'][0]
    except (KeyError, IndexError):
        return _mk('UNCERTAIN', notes='need labels {base, percent, result}')
    computed = base.value * pct.value / 100.0
    min_sigf = min(sigfig_count(base.raw), sigfig_count(pct.raw))
    rel = relative_tol_from_sigfigs(min_sigf)
    scale = max(abs(result.value), 1e-300)
    diff_abs = abs(computed - result.value)
    diff_rel = diff_abs / scale
    abs_tol = halfulp(result.raw)
    return _mk(_combined_verdict_from_abs_or_rel(diff_abs=diff_abs, diff_rel=diff_rel, abs_tol=abs_tol, rel_tol=rel), computed=computed, reported=result.value, abs_tol=abs_tol, rel_tol=rel, diff_abs=diff_abs, diff_rel=diff_rel, notes=f'{base.raw} * {pct.raw}% = {computed}; claimed {result.raw}')

def p_ratio_eq(cand: NumericsCandidate) -> Dict[str, Any]:
    by = _by_label(cand.values)
    try:
        num = by['numerator'][0]
        den = by['denominator'][0]
        r = by['ratio'][0]
    except (KeyError, IndexError):
        return _mk('UNCERTAIN', notes='need labels {numerator, denominator, ratio}')
    if den.value == 0:
        return _mk('UNCERTAIN', notes='zero denominator')
    computed = num.value / den.value
    min_sigf = min(sigfig_count(num.raw), sigfig_count(den.raw))
    rel = relative_tol_from_sigfigs(min_sigf)
    scale = max(abs(r.value), 1e-300)
    diff_abs = abs(computed - r.value)
    diff_rel = diff_abs / scale
    abs_tol = halfulp(r.raw)
    return _mk(_combined_verdict_from_abs_or_rel(diff_abs=diff_abs, diff_rel=diff_rel, abs_tol=abs_tol, rel_tol=rel), computed=computed, reported=r.value, abs_tol=abs_tol, rel_tol=rel, diff_abs=diff_abs, diff_rel=diff_rel, notes=f'{num.raw}/{den.raw} = {computed}; claimed {r.raw}')

def p_difference_eq(cand: NumericsCandidate) -> Dict[str, Any]:
    by = _by_label(cand.values)
    try:
        m = by['minuend'][0]
        s = by['subtrahend'][0]
        d = by['difference'][0]
    except (KeyError, IndexError):
        return _mk('UNCERTAIN', notes='need labels {minuend, subtrahend, difference}')
    computed = m.value - s.value
    tol = combined_additive_tol([m.raw, s.raw, d.raw])
    diff = computed - d.value
    verdict = 'PASS' if abs(diff) <= tol else 'FAIL'
    return _mk(verdict, computed=computed, reported=d.value, abs_tol=tol, diff_abs=diff, notes=f'{m.raw} - {s.raw} = {computed}; claimed {d.raw}')

def p_table_row_total(cand: NumericsCandidate) -> Dict[str, Any]:
    return p_sum_eq_total(cand)

def p_table_col_total(cand: NumericsCandidate) -> Dict[str, Any]:
    return p_sum_eq_total(cand)
PREDICATES: Dict[str, Callable[[NumericsCandidate], Dict[str, Any]]] = {'sum_eq_total': p_sum_eq_total, 'percent_to_100': p_percent_to_100, 'repeated_constant': p_repeated_constant, 'table_row_total': p_table_row_total, 'table_col_total': p_table_col_total, 'abstract_vs_body_match': p_abstract_vs_body_match, 'rounded_match': p_rounded_match, 'integer_equal': p_integer_equal, 'percent_of': p_percent_of, 'ratio_eq': p_ratio_eq, 'difference_eq': p_difference_eq}

def run_predicate(cand: NumericsCandidate) -> Dict[str, Any]:
    fn = PREDICATES.get(cand.check_type)
    if fn is None:
        return _mk('UNCERTAIN', notes=f'unknown check_type {cand.check_type!r}')
    try:
        return fn(cand)
    except Exception as e:
        return _mk('UNCERTAIN', notes=f'predicate error: {type(e).__name__}: {e}')
