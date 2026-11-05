"""Cost accounting for model calls."""
import contextlib
import contextvars
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
_LEDGER_CTX: 'contextvars.ContextVar[Optional[Dict[str, Any]]]' = contextvars.ContextVar('skepthical_ledger_ctx', default=None)

def set_ledger_context(*, paper_id: str, paper_dir: Any, version: Optional[str]=None, method: Optional[str]=None) -> contextvars.Token:
    return _LEDGER_CTX.set({'paper_id': str(paper_id), 'paper_dir': str(paper_dir), 'version': version, 'method': method})

def reset_ledger_context(token: contextvars.Token) -> None:
    _LEDGER_CTX.reset(token)

@contextlib.contextmanager
def ledger_context(*, paper_id: str, paper_dir: Any, version: Optional[str]=None, method: Optional[str]=None):
    token = set_ledger_context(paper_id=paper_id, paper_dir=paper_dir, version=version, method=method)
    try:
        yield
    finally:
        reset_ledger_context(token)

def _try_external_log(*, agent_name: str, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int=0) -> None:
    ctx = _LEDGER_CTX.get()
    if ctx is None:
        return
    try:
        from pipeline.cost import log_call as _log_call
        _log_call(paper_id=ctx['paper_id'], paper_dir=Path(ctx['paper_dir']), agent_name=agent_name, model=model, input_tokens=int(input_tokens), output_tokens=int(output_tokens), cache_read_tokens=int(cached_input_tokens or 0), provider='openai', extras={'via': 'skepthical-bridge', 'stage': agent_name}, version=ctx.get('version'), method=ctx.get('method'))
    except Exception:
        pass
_DIRECT_LLM_COST_LOCK = threading.Lock()
_DIRECT_LLM_COSTS: Dict[str, Dict[str, Any]] = {}

def _normalize_model_id(model: str) -> str:
    m = (model or '').strip().lower()
    m = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', m)
    return m
_CACHED_READ_PER_1M_SK: Dict[str, float] = {'claude-opus-4-7': 0.5, 'claude-opus-4-6': 0.5, 'claude-opus-4-5': 0.5, 'claude-opus-4-1': 1.5, 'claude-opus-4': 1.5, 'claude-sonnet-4-6': 0.3, 'claude-sonnet-4-5': 0.3, 'claude-sonnet-4': 0.3, 'claude-haiku-4-5': 0.1, 'claude-haiku-4-6': 0.1, 'gpt-5.5': 0.5, 'gpt-5.5-pro': 3.0, 'gpt-5.4': 0.25, 'gpt-5.4-mini': 0.075, 'gpt-5.4-nano': 0.02, 'gpt-5.4-pro': 3.0, 'gpt-5.2': 0.175, 'gpt-5.1': 0.125, 'gpt-5': 0.125, 'gpt-5-mini': 0.025, 'gpt-5-nano': 0.005, 'gpt-4.1': 0.5, 'gpt-4.1-mini': 0.1, 'gpt-4.1-nano': 0.025, 'gpt-4o': 1.25, 'gpt-4o-mini': 0.075, 'gemini-3.1-pro-preview': 0.2, 'gemini-3.1-pro': 0.2, 'gemini-3-pro-preview': 0.2, 'gemini-3-flash-preview': 0.05, 'gemini-3-flash': 0.05, 'gemini-3.1-flash-lite-preview': 0.025, 'gemini-3.1-flash-lite': 0.025, 'gemini-2.5-pro': 0.125, 'gemini-2.5-flash': 0.03, 'gemini-2.5-flash-lite': 0.01, 'gemini-2.0-flash': 0.025}

def _cached_read_rate_per_token(model: str, in_rate: float) -> float:
    m = (model or '').strip().lower()
    m = re.sub('-\\d{4}-\\d{2}-\\d{2}$', '', m)
    if m in _CACHED_READ_PER_1M_SK:
        return _CACHED_READ_PER_1M_SK[m] / 1000000.0
    for k, rate_per_M in _CACHED_READ_PER_1M_SK.items():
        if m.startswith(k):
            return rate_per_M / 1000000.0
    return 0.5 * in_rate

def _price_per_token(model: str) -> Tuple[float, float]:
    PRICES_1M: Dict[str, Tuple[float, float]] = {'gpt-5.5': (5.0, 30.0), 'gpt-5.5-pro': (30.0, 180.0), 'gpt-5.4-pro': (30.0, 180.0), 'gpt-5.4-mini': (0.75, 4.5), 'gpt-5.4-nano': (0.2, 1.25), 'gpt-5.4': (2.5, 15.0), 'gpt-5.2': (1.75, 14.0), 'gpt-5.1': (1.25, 10.0), 'gpt-5-mini': (0.25, 2.0), 'gpt-5-nano': (0.05, 0.4), 'gpt-5': (1.25, 10.0), 'gpt-4.1-mini': (0.4, 1.6), 'gpt-4.1-nano': (0.1, 0.4), 'gpt-4.1': (2.0, 8.0), 'gpt-4o': (2.5, 10.0), 'gpt-4o-mini': (0.15, 0.6), 'claude-opus-4-7': (5.0, 25.0), 'claude-opus-4-6': (5.0, 25.0), 'claude-sonnet-4-6': (3.0, 15.0), 'claude-sonnet-4-5': (3.0, 15.0), 'claude-haiku-4-5': (1.0, 5.0), 'gemini-3.1-pro-preview': (2.0, 12.0), 'gemini-3.1-pro': (2.0, 12.0), 'gemini-3-pro-preview': (2.0, 12.0), 'gemini-3-flash-preview': (0.5, 3.0), 'gemini-3-flash': (0.5, 3.0), 'gemini-3.1-flash-lite-preview': (0.25, 1.5), 'gemini-3.1-flash-lite': (0.25, 1.5), 'gemini-2.5-pro': (1.25, 10.0), 'gemini-2.5-flash': (0.3, 2.5), 'gemini-2.5-flash-lite': (0.1, 0.4)}
    m = _normalize_model_id(model)
    if m in PRICES_1M:
        inp, out = PRICES_1M[m]
        return (inp / 1000000.0, out / 1000000.0)
    for k, (inp, out) in PRICES_1M.items():
        if m.startswith(k):
            return (inp / 1000000.0, out / 1000000.0)
    inp, out = PRICES_1M['gpt-4o']
    return (inp / 1000000.0, out / 1000000.0)

def _extract_in_out_tokens_from_usage(usage: Any) -> Tuple[int, int]:
    if usage is None:
        return (0, 0)
    if isinstance(usage, dict):
        in_t = usage.get('input_tokens', usage.get('prompt_tokens', 0)) or 0
        out_t = usage.get('output_tokens', usage.get('completion_tokens', 0)) or 0
        return (int(in_t), int(out_t))
    in_t = getattr(usage, 'input_tokens', None)
    out_t = getattr(usage, 'output_tokens', None)
    if in_t is None and out_t is None:
        in_t = getattr(usage, 'prompt_tokens', 0)
        out_t = getattr(usage, 'completion_tokens', 0)
    return (int(in_t or 0), int(out_t or 0))

def _extract_cached_input_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
        if isinstance(details, dict):
            return int(details.get('cached_tokens', 0) or 0)
        return 0
    details = getattr(usage, 'input_tokens_details', None) or getattr(usage, 'prompt_tokens_details', None)
    if details is None:
        return 0
    return int(getattr(details, 'cached_tokens', 0) or 0)

def record_openai_tokens_cost(*, agent_name: str, model: str, input_tokens: int, output_tokens: int, cached_input_tokens: int=0, verbose: bool=False) -> float:
    in_rate, out_rate = _price_per_token(model)
    cached_rate = _cached_read_rate_per_token(model, in_rate)
    cached = max(int(cached_input_tokens), 0)
    total_in = int(input_tokens)
    uncached = max(total_in - cached, 0)
    cost = uncached * in_rate + cached * cached_rate + int(output_tokens) * out_rate
    with _DIRECT_LLM_COST_LOCK:
        row = _DIRECT_LLM_COSTS.get(agent_name)
        if row is None:
            row = {'Agent': agent_name, 'Cost': 0.0, 'Prompt Tokens': 0, 'Completion Tokens': 0, 'Cached Prompt Tokens': 0, 'Total Tokens': 0, 'Model': set()}
            _DIRECT_LLM_COSTS[agent_name] = row
        row['Cost'] += float(cost)
        row['Prompt Tokens'] += total_in
        row['Completion Tokens'] += int(output_tokens)
        row['Cached Prompt Tokens'] = row.get('Cached Prompt Tokens', 0) + cached
        row['Total Tokens'] += total_in + int(output_tokens)
        row['Model'].add(_normalize_model_id(model))
    if verbose:
        print(f'[COST] Agent={agent_name} | Model={model} | Input={total_in:,} (cached={cached:,}) | Output={int(output_tokens):,} | Est. cost=${cost:.6f}')
    _try_external_log(agent_name=agent_name, model=model, input_tokens=total_in, output_tokens=int(output_tokens), cached_input_tokens=cached)
    return float(cost)

def record_openai_response_cost(resp: Any, *, agent_name: str, model: str, verbose: bool=False) -> float:
    usage = getattr(resp, 'usage', None)
    if usage is None:
        d = None
        for fn in ('to_dict', 'model_dump', 'dict'):
            if hasattr(resp, fn):
                try:
                    d = getattr(resp, fn)()
                    break
                except Exception:
                    pass
        if isinstance(d, dict):
            usage = d.get('usage')
    in_t, out_t = _extract_in_out_tokens_from_usage(usage)
    cached_in = _extract_cached_input_tokens(usage)
    if (in_t == 0 and out_t == 0) and verbose:
        print(f'[COST][WARN] No usage found on response for agent={agent_name}, model={model}.')
    return record_openai_tokens_cost(agent_name=agent_name, model=model, input_tokens=in_t, output_tokens=out_t, cached_input_tokens=cached_in, verbose=verbose)

def get_direct_llm_cost_rows(clear: bool=False):
    with _DIRECT_LLM_COST_LOCK:
        rows = []
        for agent_name, row in _DIRECT_LLM_COSTS.items():
            name_str = str(agent_name) if agent_name is not None else ''
            models = sorted(list(row.get('Model') or []))
            rows.append({'Agent': name_str.replace('_', ' '), 'Cost ($)': round(float(row.get('Cost', 0.0)), 8), 'Prompt Tokens': int(row.get('Prompt Tokens', 0)), 'Cached Prompt Tokens': int(row.get('Cached Prompt Tokens', 0)), 'Completion Tokens': int(row.get('Completion Tokens', 0)), 'Total Tokens': int(row.get('Total Tokens', 0)), 'Model': ', '.join(models)})
        if clear:
            _DIRECT_LLM_COSTS.clear()
        return rows
