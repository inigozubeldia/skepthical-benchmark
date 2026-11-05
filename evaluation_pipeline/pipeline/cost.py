"""Per-call LLM cost logging.

Each call appends one JSON line to <paper_dir>/costs.jsonl and updates an
in-memory aggregator. Unknown models log zero cost with `model_known=false`.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# USD per 1M tokens (input, output).
PRICES_PER_1M: Dict[str, Tuple[float, float]] = {
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-haiku-4-6":  (1.00, 5.00),

    "gpt-5.5":      (5.00,  30.00),
    "gpt-5.5-pro":  (30.00, 180.00),

    "gpt-5.4-pro":  (30.00, 180.00),
    "gpt-5.4-mini": (0.75,  4.50),
    "gpt-5.4-nano": (0.20,  1.25),
    "gpt-5.4":      (2.50,  15.00),

    "gpt-5.2":      (1.75, 14.00),
    "gpt-5.1":      (1.25, 10.00),
    "gpt-5-mini":   (0.25, 2.00),
    "gpt-5-nano":   (0.05, 0.40),
    "gpt-5":        (1.25, 10.00),

    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1":      (2.00, 8.00),

    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15, 0.60),

    "gemini-3.1-pro-preview":         (2.00, 12.00),
    "gemini-3.1-pro":                 (2.00, 12.00),
    "gemini-3-pro-preview":           (2.00, 12.00),
    "gemini-3-flash-preview":         (0.50, 3.00),
    "gemini-3-flash":                 (0.50, 3.00),
    "gemini-3.1-flash-lite-preview":  (0.25, 1.50),
    "gemini-3.1-flash-lite":          (0.25, 1.50),
    "gemini-2.5-pro":                 (1.25, 10.00),
    "gemini-2.5-flash":               (0.30, 2.50),
    "gemini-2.5-flash-lite":          (0.10, 0.40),
    "gemini-2.0-flash":               (0.10, 0.40),
}


def _normalize_model_id(model: str) -> str:
    s = (model or "").strip().lower()
    parts = s.split("-")
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 6:
        s = "-".join(parts[:-1])
    return s


def price_per_token(model: str) -> Tuple[float, float, bool]:
    """(input_rate, output_rate, known), in USD per token."""
    m = _normalize_model_id(model)
    if m in PRICES_PER_1M:
        inp, out = PRICES_PER_1M[m]
        return inp / 1_000_000.0, out / 1_000_000.0, True
    for k, (inp, out) in PRICES_PER_1M.items():
        if m.startswith(k):
            return inp / 1_000_000.0, out / 1_000_000.0, True
    return 0.0, 0.0, False


CACHED_READ_PER_1M: Dict[str, float] = {
    "claude-opus-4-7":   0.50,
    "claude-opus-4-6":   0.50,
    "claude-opus-4-5":   0.50,
    "claude-opus-4-1":   1.50,
    "claude-opus-4":     1.50,
    "claude-sonnet-4-6": 0.30,
    "claude-sonnet-4-5": 0.30,
    "claude-sonnet-4":   0.30,
    "claude-haiku-4-5":  0.10,
    "claude-haiku-4-6":  0.10,
    "gpt-5.5":      0.50,
    "gpt-5.5-pro":  3.00,
    "gpt-5.4":      0.25,
    "gpt-5.4-mini": 0.075,
    "gpt-5.4-nano": 0.020,
    "gpt-5.4-pro":  3.00,
    "gpt-5.2":      0.175,
    "gpt-5.1":      0.125,
    "gpt-5":        0.125,
    "gpt-5-mini":   0.025,
    "gpt-5-nano":   0.005,
    "gpt-4.1":      0.50,
    "gpt-4.1-mini": 0.10,
    "gpt-4.1-nano": 0.025,
    "gpt-4o":       1.25,
    "gpt-4o-mini":  0.075,
    "gemini-3.1-pro-preview":         0.20,
    "gemini-3.1-pro":                 0.20,
    "gemini-3-pro-preview":           0.20,
    "gemini-3-flash-preview":         0.05,
    "gemini-3-flash":                 0.05,
    "gemini-3.1-flash-lite-preview":  0.025,
    "gemini-3.1-flash-lite":          0.025,
    "gemini-2.5-pro":                 0.125,
    "gemini-2.5-flash":               0.030,
    "gemini-2.5-flash-lite":          0.010,
    "gemini-2.0-flash":               0.025,
}


def cached_read_rate_per_token(model: str, provider: str, in_rate: float) -> float:
    m = _normalize_model_id(model)
    if m in CACHED_READ_PER_1M:
        return CACHED_READ_PER_1M[m] / 1_000_000.0
    for k, rate_per_M in CACHED_READ_PER_1M.items():
        if m.startswith(k):
            return rate_per_M / 1_000_000.0
    return _CACHE_READ_MULT.get(provider, 1.0) * in_rate


_CACHE_WRITE_MULT = {"anthropic": 1.25, "openai": 1.00, "gemini": 1.00}
_CACHE_READ_MULT  = {"anthropic": 0.10, "openai": 0.50, "gemini": 0.10}


def compute_call_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Provider-aware cost.

        Anthropic reports `input_tokens` as the uncached subset; OpenAI and Gemini
        report the total, with cached tokens as a subset.
    """
    in_rate, out_rate, _ = price_per_token(model)
    write_mult = _CACHE_WRITE_MULT.get(provider, 1.0)
    read_rate = cached_read_rate_per_token(model, provider, in_rate)

    if provider == "anthropic":
        billable_input = input_tokens
    else:
        billable_input = max(input_tokens - cache_read_tokens - cache_creation_tokens, 0)

    return (
        billable_input * in_rate
        + cache_creation_tokens * write_mult * in_rate
        + cache_read_tokens * read_rate
        + output_tokens * out_rate
    )


_LOCK = threading.Lock()
_ROWS: list[Dict[str, Any]] = []


def reset() -> None:
    with _LOCK:
        _ROWS.clear()


def summary() -> Dict[str, Any]:
    """Aggregate the rows recorded since the last reset()."""
    with _LOCK:
        rows = list(_ROWS)
    total_cost = sum(r["cost_usd"] for r in rows)
    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    per_agent: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        a = r["agent_name"]
        cell = per_agent.setdefault(a, {"calls": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0})
        cell["calls"] += 1
        cell["cost_usd"] += r["cost_usd"]
        cell["input_tokens"] += r["input_tokens"]
        cell["output_tokens"] += r["output_tokens"]
    return {
        "n_calls": len(rows),
        "total_cost_usd": total_cost,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "per_agent": per_agent,
    }


def _skepthical_cost_total() -> float:
    try:
        from skepthical.cost import get_direct_llm_cost_rows  # type: ignore
    except Exception:
        return 0.0
    try:
        rows = get_direct_llm_cost_rows(clear=False) or []
    except Exception:
        return 0.0
    return float(sum(float(r.get("Cost ($)", 0.0)) for r in rows))


def combined_total_cost_usd() -> float:
    """Total cost across this module and skepthical.cost, for delta-based attribution."""
    pipeline_total = float(summary().get("total_cost_usd", 0.0))
    return pipeline_total + _skepthical_cost_total()


_AGENT_NAME_RE = re.compile(
    r"""
    ^(?:
        (?P<top1>reviewer)\.(?P<m1>L[123]|B[123]|C)\.(?P<v1>v\d)        |
        (?P<top2>verifier)\.(?P<v2>v\d)\.(?P<m2>L[123]|B[123]|C)\.[\w.-]+  |
        (?P<top3>injector)\.(?P<mod>\w+)\.S\d\.t\d\.(?P<v3>v\d)(?:\.\w+)?  |
        (?P<top4>fp_extractor|fp_general_panel_\d|fp_maths_targeted_extractor|fp_numerics_structurer|fp_citations_\w+|fp_maths_\w+|fp_numerics_\w+)
    )$
    """,
    re.VERBOSE,
)


def _parse_agent_name(agent_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort (version, method) from agent_name; (None, None) if unrecognised."""
    m = _AGENT_NAME_RE.match(agent_name or "")
    if not m:
        return None, None
    if m.group("top1"):
        return m.group("v1"), m.group("m1")
    if m.group("top2"):
        return m.group("v2"), m.group("m2")
    if m.group("top3"):
        return m.group("v3"), "injector"
    if m.group("top4"):
        raw = m.group("top4")
        if raw.startswith("fp_general_panel"):
            return None, "fp_general"
        if raw.startswith("fp_maths"):
            return None, "fp_maths"
        if raw.startswith("fp_numerics"):
            return None, "fp_numerics"
        if raw.startswith("fp_citations"):
            return None, "fp_citations"
        return None, raw
    return None, None


def log_call(
    *,
    paper_id: str,
    paper_dir: Path,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    extras: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    version: Optional[str] = None,
    method: Optional[str] = None,
    cost_usd_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Record one LLM call: append a JSONL row and update the aggregator."""
    _, _, known = price_per_token(model)
    if cost_usd_override is not None:
        cost = float(cost_usd_override)
    elif provider:
        cost = compute_call_cost(
            provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
    else:
        in_rate, out_rate, _ = price_per_token(model)
        cost = float(input_tokens) * in_rate + float(output_tokens) * out_rate
    if version is None or method is None:
        v_parsed, m_parsed = _parse_agent_name(agent_name)
        if version is None:
            version = v_parsed
        if method is None:
            method = m_parsed
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "agent_name": agent_name,
        "version": version,
        "method": method,
        "model": model,
        "model_known": known,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cache_read_tokens": int(cache_read_tokens),
        "cache_creation_tokens": int(cache_creation_tokens),
        "cost_usd": cost,
    }
    if extras:
        row["extras"] = extras

    paper_dir.mkdir(parents=True, exist_ok=True)
    log_path = paper_dir / "costs.jsonl"
    with _LOCK:
        _ROWS.append(row)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def log_anthropic_response(
    resp: Any,
    *,
    paper_id: str,
    paper_dir: Path,
    agent_name: str,
    model: str,
    extras: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    usage = getattr(resp, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
    cache_read_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    return log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        extras=extras, version=version, method=method,
    )


def log_openai_response(
    resp: Any,
    *,
    paper_id: str,
    paper_dir: Path,
    agent_name: str,
    model: str,
    extras: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        for fn in ("to_dict", "model_dump", "dict"):
            if hasattr(resp, fn):
                try:
                    d = getattr(resp, fn)()
                    if isinstance(d, dict):
                        usage = d.get("usage")
                    break
                except Exception:
                    pass
    if usage is None:
        in_t = out_t = 0
    elif isinstance(usage, dict):
        in_t = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        out_t = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    else:
        in_t = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
        out_t = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
    return log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=in_t, output_tokens=out_t,
        extras=extras, version=version, method=method,
    )
