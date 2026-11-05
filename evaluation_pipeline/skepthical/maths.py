"""Mathematics checks used by the false-positive adjudication of mathematics flags."""
from __future__ import annotations
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import anthropic
from openai import OpenAI
from .cost import record_openai_response_cost, record_openai_tokens_cost
from .maths_engines import WolframSession, check_pint, check_wolfram
from .schemas import MathCheckSpec, MathStatement, MathsCheckedItem, MechanicalEvidence, ToolResult
_STAGE1B_PROMPT_TEMPLATE = 'You are a maths-statement auditor. An extractor LLM has converted a mathematical statement from a scientific paper into a mechanical check (Wolfram Language wl_code, or pint dimensional check). You will answer TWO INDEPENDENT questions about it.\n\nThe statement carries a `kind` field (one of `equation | derivation | limit | normalisation | dimensional | definition | inequality | substitution | property | assumption`). This determines whether Q2 applies.\n\n────────────────────────────────────────────────────────────────────\nQUESTION 1 — TRANSCRIPTION FIDELITY (always applies)\n────────────────────────────────────────────────────────────────────\nIs the mechanical form a faithful copy of what the paper actually contains in `verbatim_source_snippet`? Did the extractor silently "fix" a typo, swap a sign, drop a factor, reinterpret a subscript, or normalise to a canonical form that hides an error in the paper?\n\nThis question is purely about EXTRACTOR ↔ PAPER correspondence. It does NOT judge whether the paper\'s maths is itself correct.\n\n────────────────────────────────────────────────────────────────────\nQUESTION 2 — KNOWLEDGE AUDIT OF DROPPED-IN EQUATIONS\n────────────────────────────────────────────────────────────────────\nThis is the channel ORTHOGONAL to the mechanical engines. The mechanical engines (Wolfram / pint) check that derivations and identity-claims are internally consistent — i.e. given the equations the paper asserts as starting points, does the algebra hold step-by-step. They cannot tell you whether a starting equation imported from the literature is itself correct.\n\nQ2 is the converse: given an asserted starting equation that the paper imports / drops in / states without derivation, does it match the canonical / standard form for the quantity the paper itself names it as?\n\nAPPLICABILITY GATE.  Q2 applies ONLY to statements that ASSERT an equation, definition, normalisation, limit, inequality, dimensional relation, property, or assumption — i.e. dropped-in starting points that come from outside the paper\'s own algebra.\n  • `kind` ∈ {{`equation`, `definition`, `normalisation`, `limit`, `inequality`, `dimensional`, `property`, `assumption`}} → Q2 APPLIES.\n  • `kind` ∈ {{`derivation`, `substitution`}} → Q2 DOES NOT APPLY. The mechanical engines own the verdict on these. Return `knowledge_audit = "not_applicable"` and `knowledge_audit_findings = []`. Do not attempt to judge.\n\nA derivation that follows from a wrong dropped-in starting equation may be internally CORRECT (and the mechanical engine will rightly confirm it). The starting equation itself is what Q2 catches — and only on the statement that asserts it, not on downstream derivation steps.\n\nWHEN Q2 APPLIES.  Examine the ENTIRE `verbatim_source_snippet` — every prefactor, every exponent, every scaling factor, every sign, every subleading-correction term — and judge each, ONE BY ONE, against your training-data knowledge of the canonical / textbook form for the quantity the paper names.\n\nUse the paper\'s own prose to identify what the equation is SUPPOSED to be (the prose typically names the quantity: a probability density, a scaling relation, a likelihood, a canonical limit, a normalisation property, a standard physical law, …). Then judge each component term against your literature knowledge of that named form.\n\nPay particular attention to:\n  • sign conventions and signs on individual terms\n  • prefactors (½, ¼, 2π, √, etc.)\n  • exponents (especially fractional exponents, negative exponents, and exponents on subleading correction terms)\n  • argument orderings and dummy-index conventions\n  • normalisation conditions (does the form integrate / sum / limit to its expected value?)\n  • subleading correction terms — these are often the target of subtle injections and the bit the narrow check_plan ignores\n\nIf you can identify a canonical / textbook reference for ANY component of the asserted equation and it does NOT match, return `knowledge_audit = "does_not_match"` and itemise the specific factor / sign / exponent / coefficient that deviates AND the canonical form you compared against, in `knowledge_audit_findings`.\n\nIf the equation is bespoke (a quantity defined freshly by the paper with no standard literature form), return `knowledge_audit = "unknown"` — do NOT default to "matches".\n\nIf every component matches the literature form you know, return `knowledge_audit = "matches"` with empty findings.\n\nNEVER fall back to "matches" out of caution. Either you have a canonical reference for at least one component and the answer is matches / does_not_match, or you don\'t and the answer is unknown.\n\n────────────────────────────────────────────────────────────────────\nINSTRUCTION\n────────────────────────────────────────────────────────────────────\nEvaluate every entry in the statement\'s `check_plan` for Q1. For Q2, examine the WHOLE `verbatim_source_snippet` (including any factor / term outside the extractor\'s check_plan), gated by the applicability rule above.\n\nOutput ONLY valid JSON (no markdown):\n\n{{\n  "overall_fidelity":   "faithful" | "drifted" | "unverifiable",\n  "knowledge_audit":    "matches" | "does_not_match" | "unknown" | "not_applicable",\n  "knowledge_audit_findings": [\n    "<1 sentence per finding: name the specific factor / exponent / sign / coefficient, the canonical form you compared against, and the deviation observed. Empty list if knowledge_audit ∈ {{matches, unknown, not_applicable}}.>"\n  ],\n  "per_check": [\n    {{"index": 0,\n      "fidelity":   "faithful" | "drifted" | "unverifiable",\n      "notes_fidelity":  "<1 sentence; if drifted, the specific deviation>"}},\n    ...\n  ]\n}}\n\nThe STATEMENT to analyse follows below.  Everything ABOVE this point is\nthe stable instruction prefix (cached for subsequent per-statement calls\non the same paper); only the JSON below varies per call.\n'
_STAGE1B_PROMPT_SUFFIX = 'STATEMENT (extractor output):\n{statement_json}\n'

def _stage1b_transcription_fidelity(*, openai_client: OpenAI, anthropic_client: anthropic.Anthropic, api_keys: Optional[Dict[str, Any]]=None, model: str, statements: List[MathStatement], max_workers: int, debug_dir: str) -> Dict[str, Dict[str, Any]]:
    sem = _ApiSemaphore(max_workers)

    def _one(stmt: MathStatement) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        if not (stmt.verbatim_source_snippet and stmt.verbatim_source_snippet.strip()):
            return (stmt.id, {'status': 'skipped', 'detail': 'no verbatim_source_snippet provided by extractor', 'seconds': 0.0, 'overall': 'unverifiable', 'knowledge_status': 'skipped', 'knowledge_overall': 'unknown', 'knowledge_detail': 'no verbatim_source_snippet provided by extractor', 'knowledge_findings': [], 'per_check': []})
        prompt = _STAGE1B_PROMPT_SUFFIX.format(statement_json=json.dumps(stmt.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
        try:
            from .schemas import MathsStage1bOutput
            with sem:
                text = _dispatch_text_call(openai_client=openai_client, anthropic_client=anthropic_client, api_keys=api_keys, model=model, prompt=prompt, cache_prefix=_STAGE1B_PROMPT_TEMPLATE, max_tokens=128000, agent_name='maths_transcription_check', schema=MathsStage1bOutput)
            parsed = _safe_json_loads(text)
            overall = (parsed.get('overall_fidelity') or parsed.get('overall') or 'unverifiable').strip().lower()
            knowledge_overall = (parsed.get('knowledge_audit') or parsed.get('overall_canonical') or 'unknown').strip().lower()
            knowledge_findings = parsed.get('knowledge_audit_findings') or parsed.get('holistic_findings') or []
            if not isinstance(knowledge_findings, list):
                knowledge_findings = [str(knowledge_findings)]
            knowledge_findings = [str(x).strip() for x in knowledge_findings if str(x).strip()]
            per_check = parsed.get('per_check') or []
            if overall == 'faithful':
                status = 'confirmed'
            elif overall == 'drifted':
                status = 'refuted'
            else:
                status = 'inconclusive'
            if knowledge_overall == 'matches':
                knowledge_status = 'confirmed'
            elif knowledge_overall == 'does_not_match':
                knowledge_status = 'refuted'
            elif knowledge_overall == 'not_applicable':
                knowledge_status = 'skipped'
            else:
                knowledge_status = 'inconclusive'
            fid_bits = []
            for pc in per_check:
                if not isinstance(pc, dict):
                    continue
                idx = pc.get('index', '?')
                fidel = (pc.get('fidelity') or '?').lower()
                notes_fid = (pc.get('notes_fidelity') or pc.get('notes') or '').strip()
                fid_bits.append(f'check#{idx}:{fidel} — {notes_fid}')
            detail = 'overall_fidelity=' + overall + '; ' + ' | '.join(fid_bits) if fid_bits else f'overall_fidelity={overall}'
            knowledge_detail = 'knowledge_audit=' + knowledge_overall + ('; findings: ' + ' | '.join(knowledge_findings) if knowledge_findings else '')
            return (stmt.id, {'status': status, 'detail': detail, 'seconds': round(time.time() - t0, 3), 'overall': overall, 'knowledge_status': knowledge_status, 'knowledge_overall': knowledge_overall, 'knowledge_detail': knowledge_detail, 'knowledge_findings': knowledge_findings, 'per_check': per_check})
        except Exception as e:
            return (stmt.id, {'status': 'error', 'detail': f'stage1b call failed: {type(e).__name__}: {e}', 'seconds': round(time.time() - t0, 3), 'overall': 'unverifiable', 'knowledge_status': 'error', 'knowledge_overall': 'unknown', 'knowledge_detail': f'stage1b call failed: {type(e).__name__}: {e}', 'knowledge_findings': [], 'per_check': []})
    fidelity: Dict[str, Dict[str, Any]] = {}
    workers = min(max_workers, max(1, len(statements)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, s): s for s in statements}
        for fut in as_completed(futures):
            sid, res = fut.result()
            fidelity[sid] = res
    _debug_json(debug_dir, '01b_transcription_fidelity.json', fidelity)
    return fidelity

def _stage2_mechanical(statements: List[MathStatement], *, debug_dir: Optional[str]=None, wolfram_session: Optional[WolframSession]=None) -> Dict[str, MechanicalEvidence]:
    out: Dict[str, MechanicalEvidence] = {}
    for stmt in statements:
        results: List[ToolResult] = []
        for spec in stmt.check_plan:
            results.extend(_run_one_check(spec, stmt, wolfram_session=wolfram_session))
        out[stmt.id] = MechanicalEvidence(statement_id=stmt.id, results=results)
    if debug_dir:
        _debug_json(debug_dir, '02_evidence.json', {sid: ev.model_dump(exclude_none=True) for sid, ev in out.items()})
    return out

def _run_one_check(spec: MathCheckSpec, stmt: MathStatement, *, wolfram_session: Optional[WolframSession]=None) -> List[ToolResult]:
    results: List[ToolResult] = []
    engine = spec.engine
    args = dict(spec.args or {})
    description = spec.description or f'{engine} check'
    if engine == 'pint':
        if not stmt.units_critical:
            results.append(ToolResult(tool='pint', status='skipped', detail='units_critical=False — unit check skipped by policy', seconds=0.0, description=description))
            return results
        try:
            r = check_pint(**args)
        except Exception as e:
            r = {'tool': 'pint', 'status': 'error', 'detail': f'{type(e).__name__}: {e}', 'seconds': 0.0}
        results.append(ToolResult(description=description, **_normalise_driver_result(r, 'pint')))
        return results
    if engine == 'wolfram':
        try:
            r = check_wolfram(session=wolfram_session, **args)
        except Exception as e:
            r = {'tool': 'wolfram', 'status': 'error', 'detail': f'{type(e).__name__}: {e}', 'seconds': 0.0}
        results.append(ToolResult(description=description, **_normalise_driver_result(r, 'wolfram')))
        return results
    if engine == 'sympy':
        wl_args = _wolfram_from_latex(stmt, args)
        if wl_args is None:
            results.append(ToolResult(tool='wolfram', status='unavailable', description=description + ' (legacy sympy spec — no usable LaTeX for Wolfram fallback)', detail="legacy engine='sympy' check could not be re-routed: no LHS=RHS LaTeX available. Knowledge-audit / prose channels are load-bearing.", seconds=0.0))
            return results
        try:
            r = check_wolfram(session=wolfram_session, **wl_args)
        except Exception as e:
            r = {'tool': 'wolfram', 'status': 'error', 'detail': f'legacy sympy → wolfram fallback: {type(e).__name__}: {e}', 'seconds': 0.0}
        results.append(ToolResult(description=description + ' (legacy sympy → wolfram raw-LaTeX fallback)', **_normalise_driver_result(r, 'wolfram')))
        return results
    results.append(ToolResult(tool='llm', status='error', detail=f'unknown engine {engine!r}', seconds=0.0, description=description))
    return results

def _normalise_driver_result(r: Dict[str, Any], default_tool: str) -> Dict[str, Any]:
    return {'tool': r.get('tool') or default_tool, 'status': r.get('status') or 'error', 'detail': str(r.get('detail') or '')[:2000], 'seconds': float(r.get('seconds') or 0.0)}
_LATEX_EQ_RE = re.compile('^\\s*(.+?)\\s*=\\s*(.+?)\\s*$', re.DOTALL)

def _wolfram_from_latex(stmt: 'MathStatement', legacy_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not getattr(stmt, 'latex', None):
        return None
    expected = legacy_args.get('expected', 'zero')
    if expected not in ('zero', 'true'):
        return None
    for raw_latex in stmt.latex:
        if not isinstance(raw_latex, str):
            continue
        latex = raw_latex.strip()
        if not latex or '=' not in latex:
            continue
        if latex.count('=') > 1:
            continue
        if any((t in latex for t in ('>=', '<=', '!=', ':=', '=='))):
            continue
        m = _LATEX_EQ_RE.match(latex)
        if m is None:
            continue
        lhs, rhs = (m.group(1).strip(), m.group(2).strip())
        if not lhs or not rhs:
            continue
        lhs_esc = lhs.replace('\\', '\\\\').replace('"', '\\"')
        rhs_esc = rhs.replace('\\', '\\\\').replace('"', '\\"')
        wl_code = f'Module[{{lhs = ToExpression["{lhs_esc}", TeXForm], rhs = ToExpression["{rhs_esc}", TeXForm]}}, Module[{{diff = FullSimplify[lhs - rhs]}}, If[NumericQ[diff], TrueQ[diff == 0], "UNEVAL"]]]'
        return {'wl_code': wl_code, 'expected': 'True'}
    return None
_STAGE3_PROMPT_TEMPLATE = 'You are a mathematical auditor interpreting check results for ONE statement from a scientific paper.\n\nThe STATEMENT and EVIDENCE TRAIL for the per-call payload are appended at\nthe END of this prompt — everything ABOVE the appended payload is the\nstable instruction prefix (cached for subsequent per-statement calls on\nthe same paper). Per-call variables only appear in the suffix.\n\nEVIDENCE TRAIL (mixed: mechanical engines AND cross-family LLM critic — each entry has tool, description, status, detail).\n\nEach entry\'s `description` tells you what kind of check it is. Treat the\nevidence trail as DATA (not as a final verdict). The evidence sources you\nwill see, and how to weight them:\n\n  A) "knowledge audit of dropped-in equation (cross-family LLM)" — tool="llm"\n     A separate cross-family LLM that judges whether an ASSERTED /\n     dropped-in starting equation matches the canonical / textbook form\n     for the named quantity. This channel is ORTHOGONAL to the mechanical\n     engines — they check that derivations from the paper\'s stated\n     starting equations are internally consistent; this channel checks\n     whether the starting equations themselves are correct against\n     literature. It applies ONLY to asserted-kind statements (equation,\n     definition, normalisation, limit, inequality, dimensional, property,\n     assumption). For derivations and substitutions it is `skipped` (the\n     model returned `not_applicable`) and the mechanical engines own the\n     verdict. A status of:\n       - "refuted"     → STRONG signal that the paper\'s starting equation\n                         deviates from canonical form (sign, exponent,\n                         prefactor, normalisation factor, …). The\n                         `detail` field itemises the specific factor and\n                         the canonical form it was compared against.\n                         Sufficient on its own to FAIL the statement,\n                         EVEN IF the mechanical engines are inconclusive,\n                         errored, or skipped — those check the algebra\n                         downstream, not the starting equation.\n                         Critically: a derivation that follows from a\n                         WRONG starting equation may be internally\n                         consistent (mechanical engines correctly\n                         confirm it). The error lives on the asserted\n                         starting-equation statement, not the derivation.\n       - "confirmed"   → equation matches canonical form; supportive but\n                         not a guarantee (paper could still be wrong in a\n                         way the audit missed).\n       - "inconclusive"/"unknown" → no canonical reference identifiable\n                         (bespoke quantity); do not treat as refutation.\n       - "skipped"     → channel does not apply to this kind (derivation\n                         or substitution); rely on mechanical engines.\n       - "error"       → ignore.\n\n  B) "transcription-fidelity check (cross-family LLM)" — tool="llm"\n     This judges whether the EXTRACTOR copied the paper\'s maths faithfully\n     (independent of whether the maths is itself correct). A status of:\n       - "refuted" / "drifted" → the extractor drifted. Mechanical engine\n                                 results are UNRELIABLE (ran on a drifted\n                                 transcription). Do NOT use mechanical\n                                 results as confirmation or refutation;\n                                 fall back to the knowledge audit or\n                                 prose argument.\n       - "confirmed" / "faithful" → mechanical engine results can be\n                                 trusted; integrate them normally.\n\n  C) "<engine> check" with tool ∈ {{wolfram, pint}}\n     Mechanical engine results. status ∈ {{confirmed, refuted,\n     inconclusive, error, skipped, unavailable}}. Treat `error` and\n     `unavailable` as non-informative — but ONLY about that one engine.\n     They do NOT prevent a FAIL when channel A is refuted.\n\nYour job:\n  1. Synthesise the evidence trail above with the statement\'s assumptions.\n  2. Decide the final verdict:\n       PASS      — the claim is correct (mechanical confirmation on\n                   derivations / identities, OR knowledge-audit match on\n                   asserted equations, AND no contrary signal from the\n                   other applicable channel).\n       FAIL      — the claim is wrong. Trigger conditions (any one suffices):\n                   (a) any mechanical engine returns `refuted`;\n                   (b) the knowledge audit returns `refuted` with\n                       itemised findings;\n                   (c) clear prose-level reasoning error you can articulate.\n                   When (b) fires while (a) is `error` / `inconclusive` /\n                   `skipped`, issue FAIL — that is exactly the case the\n                   knowledge channel exists for (a wrong starting equation\n                   that mechanical engines can\'t refute on their own).\n                   Do NOT downgrade a knowledge-audit refutation because\n                   Wolfram couldn\'t parse the deliberately-broken term.\n       UNCERTAIN — cannot decide. Use this when:\n                   • for a derivation / substitution: the applicable\n                     mechanical engines are all\n                     inconclusive/skipped/unavailable AND no clear prose\n                     reasoning error;\n                   • for an asserted equation: the knowledge audit is\n                     inconclusive/unknown AND mechanical engines (if\n                     applicable) are not refuted.\n  3. Distinguish "paper omitted an intermediate step but the claim holds"\n     (PASS with note) from "paper got the math wrong" (FAIL).\n\nOutput ONLY valid JSON (no markdown) with this schema:\n\n{{\n  "verdict": "PASS" | "FAIL" | "UNCERTAIN",\n  "confidence": "high" | "medium" | "low",\n  "impact": "critical" | "moderate" | "minor",\n  "notes": "2-5 sentences. If FAIL, point to the first incorrect step and which evidence channel surfaced it (knowledge-audit / mechanical / prose).",\n  "mechanical_evidence_summary": "1-2 sentence prose summary of how the evidence trail was used. Name which channel(s) carried the verdict."\n}}\n'
_STAGE3_PROMPT_SUFFIX = 'STATEMENT (extracted earlier):\n{statement_json}\n\nEVIDENCE TRAIL:\n{evidence_json}\n'

def _stage3_interpret(*, openai_client: OpenAI, anthropic_client: anthropic.Anthropic, api_keys: Optional[Dict[str, Any]]=None, model: str, statements: List[MathStatement], evidence_by_id: Dict[str, MechanicalEvidence], timeout_s: Optional[float], max_workers: int, debug_dir: str) -> List[MathsCheckedItem]:
    sem = _ApiSemaphore(max_workers)

    def _one(stmt: MathStatement) -> MathsCheckedItem:
        ev = evidence_by_id.get(stmt.id, MechanicalEvidence(statement_id=stmt.id, results=[]))
        prompt = _STAGE3_PROMPT_SUFFIX.format(statement_json=json.dumps(stmt.model_dump(exclude_none=True), ensure_ascii=False, indent=2), evidence_json=json.dumps(ev.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
        try:
            from .schemas import MathsStage3Output
            with sem:
                text = _dispatch_text_call(openai_client=openai_client, anthropic_client=anthropic_client, api_keys=api_keys, model=model, prompt=prompt, cache_prefix=_STAGE3_PROMPT_TEMPLATE, max_tokens=128000, timeout_s=timeout_s, agent_name='maths_interpreter', schema=MathsStage3Output)
            parsed = _safe_json_loads(text)
            verdict = parsed.get('verdict', 'UNCERTAIN')
            confidence = parsed.get('confidence', 'low')
            impact = parsed.get('impact', 'minor')
            notes = parsed.get('notes', '')
            mech_summary = parsed.get('mechanical_evidence_summary', '')
        except Exception as e:
            print(f'[maths-rigorous] stage 3 fallback for {stmt.id!r}: {e}', flush=True)
            verdict, confidence, impact = ('UNCERTAIN', 'low', 'minor')
            notes = f'Interpreter call failed: {type(e).__name__}: {e}'
            mech_summary = ''
        if verdict not in ('PASS', 'FAIL', 'UNCERTAIN'):
            verdict = 'UNCERTAIN'
        if confidence not in ('high', 'medium', 'low'):
            confidence = 'low'
        if impact not in ('critical', 'moderate', 'minor'):
            impact = 'minor'
        return MathsCheckedItem(id=stmt.id, location=stmt.location, claim=stmt.claim, inputs_assumptions=list(stmt.assumptions or []), check_type=[spec.engine for spec in stmt.check_plan], verdict=verdict, confidence=confidence, impact=impact, notes=str(notes or '').strip(), tools_used=ev.summary_lines(), mechanical_evidence_summary=str(mech_summary).strip() or None)
    items: List[MathsCheckedItem] = []
    workers = min(max_workers, max(1, len(statements)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, s): s for s in statements}
        for fut in as_completed(futures):
            items.append(fut.result())
    order = {s.id: i for i, s in enumerate(statements)}
    items.sort(key=lambda it: order.get(it.id, 10000))
    _debug_json(debug_dir, '03_checked_items.json', [it.model_dump(exclude_none=True) for it in items])
    return items

class _ApiSemaphore:

    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()
        return False

def _safe_record_openai_cost(resp: Any, agent_name: str, model: str) -> None:
    try:
        record_openai_response_cost(resp, agent_name=agent_name, model=str(_display_model_name(model)), verbose=False)
    except Exception as e:
        print(f'[maths-rigorous] cost recorder skipped ({agent_name}): {type(e).__name__}: {e}', flush=True)

def _make_openai_client(api_keys: Optional[Dict[str, Any]]) -> OpenAI:
    from .llm_call import make_hardened_openai_client
    return make_hardened_openai_client(os.environ.get('OPENAI_API_KEY') or (api_keys or {}).get('OPENAI') or '')

def _make_anthropic_client(api_keys: Optional[Dict[str, Any]]) -> anthropic.Anthropic:
    key = os.environ.get('ANTHROPIC_API_KEY') or (api_keys or {}).get('ANTHROPIC')
    from .llm_call import make_hardened_anthropic_client
    return make_hardened_anthropic_client(key or '')

def _anthropic_call(client: anthropic.Anthropic, *, model: str, prompt: str, max_tokens: int=4096, timeout_s: Optional[float]=None, agent_name: str='anthropic_call', cache_prefix: Optional[str]=None, schema: Optional[Type[BaseModel]]=None) -> str:
    if cache_prefix:
        content_blocks: List[Dict[str, Any]] = [{'type': 'text', 'text': cache_prefix, 'cache_control': {'type': 'ephemeral'}}, {'type': 'text', 'text': prompt}]
        messages_content: Any = content_blocks
    else:
        messages_content = prompt
    kwargs: Dict[str, Any] = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': messages_content}]}
    if timeout_s is not None:
        kwargs['timeout'] = float(timeout_s)

    def _record_cost(_resp):
        try:
            usage = getattr(_resp, 'usage', None)
            in_tok = int(getattr(usage, 'input_tokens', 0) or 0) if usage else 0
            out_tok = int(getattr(usage, 'output_tokens', 0) or 0) if usage else 0
            cache_w = int(getattr(usage, 'cache_creation_input_tokens', 0) or 0) if usage else 0
            cache_r = int(getattr(usage, 'cache_read_input_tokens', 0) or 0) if usage else 0
            effective_input = int(in_tok + 1.25 * cache_w + 0.1 * cache_r)
            record_openai_tokens_cost(agent_name=agent_name, model=model, input_tokens=effective_input, output_tokens=out_tok, verbose=False)
        except Exception:
            pass
    if schema is not None:
        try:
            resp = client.messages.parse(output_format=schema, **kwargs)
            parsed_obj = getattr(resp, 'parsed', None)
            if parsed_obj is not None:
                _record_cost(resp)
                return parsed_obj.model_dump_json(exclude_none=False)
        except Exception as _e:
            try:
                from anthropic import BadRequestError
                _is_400 = isinstance(_e, BadRequestError)
            except Exception:
                _is_400 = 'BadRequest' in type(_e).__name__ or '400' in str(_e)[:80]
            if not _is_400:
                raise
    resp = client.messages.create(**kwargs)
    _record_cost(resp)
    parts: List[str] = []
    for blk in resp.content or []:
        t = getattr(blk, 'text', None)
        if isinstance(t, str) and t:
            parts.append(t)
    text = '\n'.join(parts).strip()
    if not text:
        raise RuntimeError('Anthropic returned no text content.')
    return text

def _is_anthropic_model(model: str) -> bool:
    return isinstance(model, str) and model.strip().lower().startswith('claude')

def _is_gemini_model(model: str) -> bool:
    return isinstance(model, str) and model.strip().lower().startswith('gemini')

def _split_gemini_thinking_suffix(model: str) -> Tuple[str, Optional[str]]:
    if not isinstance(model, str):
        return (model, None)
    for lvl in ('none', 'off', 'low', 'medium', 'high'):
        suffix = f'-thinking-{lvl}'
        if model.endswith(suffix):
            return (model[:-len(suffix)], 'none' if lvl == 'off' else lvl)
    if model.endswith('-thinking'):
        return (model[:-len('-thinking')], 'high')
    return (model, None)

def _gemini_thinking_config(level: Optional[str]):
    if level is None:
        return None
    try:
        from google.genai import types as gtypes
    except ModuleNotFoundError:
        return None
    budget_map = {'none': 0, 'off': 0, 'low': 2048, 'medium': 8192, 'high': 24576, 'xhigh': 32768}
    budget = budget_map.get(level)
    if budget is None:
        return None
    try:
        return gtypes.ThinkingConfig(thinking_budget=budget)
    except Exception:
        return None

def _gemini_record_cost(*, agent_name: str, model: str, usage) -> None:
    try:
        in_tok = int(getattr(usage, 'prompt_token_count', 0) or 0) if usage else 0
        out_tok = int(getattr(usage, 'candidates_token_count', 0) or 0) if usage else 0
        cache_r = int(getattr(usage, 'cached_content_token_count', 0) or 0) if usage else 0
        uncached_in = max(in_tok - cache_r, 0)
        effective_input = int(uncached_in + 0.25 * cache_r)
        record_openai_tokens_cost(agent_name=agent_name, model=str(model), input_tokens=effective_input, output_tokens=out_tok, verbose=False)
    except Exception:
        pass

def _gemini_call(*, api_keys: Optional[Dict[str, Any]], model: str, prompt: str, max_tokens: int=4096, timeout_s: Optional[float]=None, agent_name: str='gemini_call', cache_prefix: Optional[str]=None, schema: Optional[Type[BaseModel]]=None) -> str:
    import os as _os
    key = _os.environ.get('GEMINI_API_KEY') or _os.environ.get('GOOGLE_API_KEY') or (api_keys or {}).get('GEMINI')
    if not key:
        raise ValueError('GEMINI_API_KEY (or GOOGLE_API_KEY) missing.')
    try:
        from google import genai
        from google.genai import types as gtypes
    except ModuleNotFoundError as e:
        raise RuntimeError('google-genai SDK required for Gemini calls.') from e
    real_model, thinking_level = _split_gemini_thinking_suffix(model)
    full_prompt = f'{cache_prefix}\n\n{prompt}' if cache_prefix else prompt
    cfg_kwargs: Dict[str, Any] = {'max_output_tokens': max_tokens}
    tc = _gemini_thinking_config(thinking_level)
    if tc is not None:
        cfg_kwargs['thinking_config'] = tc
    if schema is not None:
        from .llm_call import _to_strict_json_schema as _strictify
        cfg_kwargs['response_mime_type'] = 'application/json'
        cfg_kwargs['response_json_schema'] = _strictify(schema.model_json_schema())
    config = gtypes.GenerateContentConfig(**cfg_kwargs)
    client = genai.Client(api_key=key)
    from .llm_call import gemini_generate_with_budget_retry
    resp, text_already = gemini_generate_with_budget_retry(client, model=real_model, contents=[full_prompt], config_kwargs=cfg_kwargs, label=f'gemini_text/{agent_name}')
    _gemini_record_cost(agent_name=agent_name, model=real_model, usage=getattr(resp, 'usage_metadata', None))
    text = text_already or (getattr(resp, 'text', '') or '')
    if not text.strip():
        raise RuntimeError('Gemini generate_content returned no text.')
    return text.strip()

def _dispatch_text_call(*, openai_client: OpenAI, anthropic_client: anthropic.Anthropic, api_keys: Optional[Dict[str, Any]]=None, model: str, prompt: str, timeout_s: Optional[float]=None, agent_name: str, max_tokens: int=4096, cache_prefix: Optional[str]=None, schema: Optional[Type[BaseModel]]=None) -> str:
    if _is_anthropic_model(model):
        return _anthropic_call(anthropic_client, model=model, prompt=prompt, max_tokens=max_tokens, timeout_s=timeout_s, agent_name=agent_name, cache_prefix=cache_prefix, schema=schema)
    if _is_gemini_model(model):
        return _gemini_call(api_keys=api_keys, model=model, prompt=prompt, max_tokens=max_tokens, timeout_s=timeout_s, agent_name=agent_name, cache_prefix=cache_prefix, schema=schema)
    full_prompt = f'{cache_prefix}\n\n{prompt}' if cache_prefix else prompt
    _stage_key = {'maths_transcription_check': 'maths_transcription_check_v2', 'maths_interpreter': 'maths_interpreter_v2'}.get(agent_name, agent_name)
    _kwargs = _build_responses_kwargs(model=model, temperature=None, timeout_s=timeout_s, prompt_text=full_prompt, file_id=None, cache_key=_stage_key)
    if schema is not None:
        try:
            resp = openai_client.responses.parse(text_format=schema, **_kwargs)
            parsed_obj = getattr(resp, 'output_parsed', None)
            if parsed_obj is not None:
                _safe_record_openai_cost(resp, agent_name, model)
                return parsed_obj.model_dump_json(exclude_none=False)
        except Exception as _e:
            try:
                from openai import BadRequestError
                _is_400 = isinstance(_e, BadRequestError)
            except Exception:
                _is_400 = 'BadRequest' in type(_e).__name__
            if not _is_400:
                raise
    resp = openai_client.responses.create(**_kwargs)
    _safe_record_openai_cost(resp, agent_name, model)
    return _extract_output_text(resp)

def _split_thinking_suffix(model: str) -> Tuple[str, Optional[str]]:
    if not isinstance(model, str):
        return (model, None)
    for eff in ('minimal', 'low', 'medium', 'high', 'xhigh'):
        suffix = f'-thinking-{eff}'
        if model.endswith(suffix):
            return (model[:-len(suffix)], eff)
    if model.endswith('-thinking'):
        return (model[:-len('-thinking')], 'high')
    return (model, None)

def _display_model_name(model: str) -> str:
    real, _ = _split_thinking_suffix(model)
    return real

def _build_responses_kwargs(*, model: str, temperature: Optional[float], timeout_s: Optional[float], prompt_text: str, file_id: Optional[str], cache_key: Optional[str]=None) -> Dict[str, Any]:
    model_name, effort = _split_thinking_suffix(model)
    reasoning: Optional[Dict[str, Any]] = {'effort': effort} if effort else None
    content: List[Dict[str, Any]] = [{'type': 'input_text', 'text': prompt_text}]
    if file_id:
        content.append({'type': 'input_file', 'file_id': file_id})
    kwargs: Dict[str, Any] = {'model': model_name, 'input': [{'role': 'user', 'content': content}]}
    if reasoning is not None:
        kwargs['reasoning'] = reasoning
    if timeout_s is not None:
        kwargs['timeout'] = timeout_s
    if temperature is not None:
        kwargs['temperature'] = float(temperature)
    if cache_key:
        kwargs['prompt_cache_key'] = cache_key
    return kwargs

def _extract_output_text(resp: Any) -> str:
    t = getattr(resp, 'output_text', None)
    if isinstance(t, str) and t.strip():
        return t.strip()
    out = getattr(resp, 'output', None)
    if isinstance(out, list) and out:
        chunks: List[str] = []
        for msg in out:
            content = getattr(msg, 'content', None)
            if not isinstance(content, list):
                continue
            for c in content:
                tt = getattr(c, 'text', None)
                if isinstance(tt, str) and tt.strip():
                    chunks.append(tt)
                elif isinstance(c, dict) and isinstance(c.get('text'), str) and c['text'].strip():
                    chunks.append(c['text'])
        if chunks:
            return '\n'.join(chunks).strip()
    raise RuntimeError('No text could be extracted from Responses API output.')
_JSON_RE = re.compile('(\\{.*\\}|\\[.*\\])', re.DOTALL)

def _safe_json_loads(text: str) -> Any:
    s = (text or '').strip()
    if not s:
        raise json.JSONDecodeError('Empty string', s, 0)
    m = _JSON_RE.search(s)
    if not m:
        raise json.JSONDecodeError('No JSON object/array found', s, 0)
    raw = m.group(1)
    return json.loads(raw, strict=False)

def _debug_json(debug_dir: str, name: str, obj: Any) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write('\n')
