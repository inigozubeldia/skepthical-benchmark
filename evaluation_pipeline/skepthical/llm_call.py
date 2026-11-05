"""API clients and structured-output calls."""
from __future__ import annotations
import base64
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
from pydantic import BaseModel
from . import cost as _cost
Provider = Literal['anthropic', 'openai', 'gemini']
_CACHE_WRITE_MULT = {'anthropic': 1.25, 'openai': 1.0, 'gemini': 1.0}
_CACHE_READ_MULT = {'anthropic': 0.1, 'openai': 0.5, 'gemini': 0.25}

@dataclass
class LLMCallResult:
    text: str
    parsed: Optional[Any] = None
    provider: Provider = 'openai'
    model: str = ''
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: Any = None
_OPENAI_FILE_CACHE: Dict[str, str] = {}
_GEMINI_FILE_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()
_OPENAI_INFLIGHT: Dict[str, threading.Event] = {}
_GEMINI_INFLIGHT: Dict[str, threading.Event] = {}
_LLM_MAX_RETRIES_DEFAULT = 8
_LLM_TIMEOUT_DEFAULT_S = 300.0

def make_hardened_anthropic_client(api_key: str, *, max_retries: int=_LLM_MAX_RETRIES_DEFAULT, timeout_s: float=_LLM_TIMEOUT_DEFAULT_S):
    import anthropic as _anthropic
    return _anthropic.Anthropic(api_key=api_key, max_retries=int(max_retries), timeout=float(timeout_s))

def make_hardened_openai_client(api_key: str, *, max_retries: int=_LLM_MAX_RETRIES_DEFAULT, timeout_s: float=_LLM_TIMEOUT_DEFAULT_S):
    from openai import OpenAI as _OpenAI
    return _OpenAI(api_key=api_key, max_retries=int(max_retries), timeout=float(timeout_s))

def _is_transient_llm_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    cls = type(exc).__name__.lower()
    transient_substrings = ('502', '503', '504', '500', 'bad gateway', 'service unavailable', 'gateway timeout', 'internal server error', 'overloaded', 'rate limit', '429', 'timeout', 'deadline exceeded', 'connection reset', 'connection error', 'remote disconnected', 'broken pipe', 'temporarily unavailable', 'resource_exhausted')
    if any((s in msg for s in transient_substrings)):
        return True
    transient_classes = ('apiconnectionerror', 'apitimeouterror', 'rate_limit_error', 'internalservererror', 'serverr', 'sdkerror', 'deadlineexceeded', 'serviceunavailable', 'googleapierror')
    if any((s in cls for s in transient_classes)):
        return True
    return False

def gemini_generate_with_budget_retry(client: Any, *, model: str, contents: Any, config_kwargs: Dict[str, Any], label: str='gemini') -> Tuple[Any, str]:
    try:
        from google.genai import types as gtypes
    except ImportError as e:
        raise RuntimeError('google-genai SDK required') from e
    import copy as _copy
    cfg1 = gtypes.GenerateContentConfig(**config_kwargs)
    resp = with_transient_retry(client.models.generate_content, model=model, contents=contents, config=cfg1, label=label)
    text = (getattr(resp, 'text', '') or '').strip()
    usage = getattr(resp, 'usage_metadata', None)
    thoughts = int(getattr(usage, 'thoughts_token_count', 0) or 0) if usage else 0
    candidates = getattr(resp, 'candidates', None) or []
    finish = getattr(candidates[0], 'finish_reason', None) if candidates else None
    finish_str = str(finish) if finish is not None else ''
    budget_exhausted = 'MAX_TOKENS' in finish_str.upper()
    if text and (not budget_exhausted):
        return (resp, text)
    if not budget_exhausted:
        return (resp, text)
    old_max = int(config_kwargs.get('max_output_tokens') or 0)
    if old_max <= 0:
        return (resp, text)
    new_max = min(old_max * 4, 256000)
    if new_max <= old_max:
        return (resp, text)
    print(f'[gemini-budget-retry] {label}: finish_reason=MAX_TOKENS (thoughts={thoughts}, visible_chars={len(text)}); retrying with max_output_tokens={old_max}→{new_max}.', flush=True)
    new_kwargs = _copy.deepcopy(config_kwargs)
    new_kwargs['max_output_tokens'] = new_max
    cfg2 = gtypes.GenerateContentConfig(**new_kwargs)
    resp2 = with_transient_retry(client.models.generate_content, model=model, contents=contents, config=cfg2, label=f'{label}/budget-retry')
    text2 = (getattr(resp2, 'text', '') or '').strip()
    cands2 = getattr(resp2, 'candidates', None) or []
    finish2 = getattr(cands2[0], 'finish_reason', None) if cands2 else None
    if 'MAX_TOKENS' in str(finish2 or '').upper():
        usage2 = getattr(resp2, 'usage_metadata', None)
        thoughts2 = int(getattr(usage2, 'thoughts_token_count', 0) or 0) if usage2 else 0
        try:
            print(f'[gemini-budget-retry][CIRCUIT_BREAKER] {label}: SECOND attempt also returned finish_reason=MAX_TOKENS at max_output_tokens={new_max} (thoughts={thoughts2}, visible_chars={len(text2)}). NOT retrying again — this is a pathological prompt or thinking-loop. Caller will see truncated/empty text.', flush=True)
        except Exception:
            pass
    return (resp2, text2)

def with_transient_retry(fn, *args, max_attempts: int=5, base_delay_s: float=8.0, max_delay_s: float=90.0, label: str='llm_call', **kwargs):
    import time as _time
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:
            last_exc = e
            if attempt + 1 >= max_attempts or not _is_transient_llm_error(e):
                raise
            delay = min(base_delay_s * 2 ** attempt, max_delay_s)
            print(f'[transient-retry] {label} attempt {attempt + 1}/{max_attempts} failed ({type(e).__name__}: {str(e)[:160]!r}); sleeping {delay:.0f}s before retry.', flush=True)
            _time.sleep(delay)
    raise last_exc

def _pdf_hash(pdf_path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def _compute_cost(*, provider: str, model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int=0, cache_creation_tokens: int=0) -> float:
    in_rate, out_rate = _cost._price_per_token(model)
    write_mult = _CACHE_WRITE_MULT.get(provider, 1.0)
    read_mult = _CACHE_READ_MULT.get(provider, 1.0)
    if provider == 'anthropic':
        billable_input = input_tokens
    else:
        billable_input = max(input_tokens - cache_read_tokens - cache_creation_tokens, 0)
    return billable_input * in_rate + cache_creation_tokens * write_mult * in_rate + cache_read_tokens * read_mult * in_rate + output_tokens * out_rate

def _schema_to_dict(schema: Type[BaseModel]) -> Dict[str, Any]:
    return schema.model_json_schema()

def _schema_name(schema: Type[BaseModel]) -> str:
    return getattr(schema, '__name__', 'Output')

def _to_strict_json_schema(d: Any) -> Any:
    import copy as _copy
    return _strictify(_copy.deepcopy(d))

def _strictify(node: Any) -> Any:
    if isinstance(node, dict):
        node.pop('default', None)
        if node.get('type') == 'object' or 'properties' in node:
            props = node.get('properties')
            if isinstance(props, dict):
                node['additionalProperties'] = False
                node['required'] = list(props.keys())
                for k, v in list(props.items()):
                    props[k] = _strictify(v)
        if 'items' in node:
            node['items'] = _strictify(node['items'])
        for k in ('anyOf', 'oneOf', 'allOf'):
            if isinstance(node.get(k), list):
                node[k] = [_strictify(x) for x in node[k]]
        for defs_k in ('$defs', 'definitions'):
            if isinstance(node.get(defs_k), dict):
                node[defs_k] = {k: _strictify(v) for k, v in node[defs_k].items()}
        return node
    if isinstance(node, list):
        return [_strictify(x) for x in node]
    return node
_FENCE_RE = re.compile('```(?:json)?\\s*(.*?)\\s*```', re.DOTALL | re.IGNORECASE)

def _strip_fences(text: str) -> str:
    s = (text or '').strip()
    if not s:
        return s
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    if s and s[0] != '{':
        i = s.find('{')
        j = s.rfind('}')
        if i != -1 and j != -1 and (j > i):
            s = s[i:j + 1]
    elif s and s[0] != '[':
        pass
    return s

def _call_anthropic(*, model: str, system_prompt: Optional[str], user_prompt: str, schema: Optional[Type[BaseModel]], pdf_path: Optional[Path], max_tokens: int, api_keys: Optional[Dict[str, Any]], temperature: Optional[float]=None) -> LLMCallResult:
    import anthropic
    key = (api_keys or {}).get('ANTHROPIC') or os.environ.get('ANTHROPIC_API_KEY')
    if not key:
        raise RuntimeError('Anthropic API key not found (set ANTHROPIC_API_KEY).')
    client = make_hardened_anthropic_client(key)
    content: list[Dict[str, Any]] = []
    if pdf_path is not None:
        pdf_bytes = Path(pdf_path).read_bytes()
        content.append({'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': base64.standard_b64encode(pdf_bytes).decode('utf-8')}, 'cache_control': {'type': 'ephemeral'}})
    content.append({'type': 'text', 'text': user_prompt})
    sys_text = system_prompt or ''
    kwargs: Dict[str, Any] = {'model': model, 'max_tokens': max_tokens, 'messages': [{'role': 'user', 'content': content}], 'timeout': 3600.0}
    if sys_text:
        kwargs['system'] = [{'type': 'text', 'text': sys_text, 'cache_control': {'type': 'ephemeral'}}]
    if temperature is not None:
        kwargs['temperature'] = float(temperature)
    parsed_obj: Any = None
    if schema is not None:
        try:
            resp = client.messages.parse(output_format=schema, **kwargs)
            parsed_obj = getattr(resp, 'parsed', None)
        except Exception as e:
            try:
                from anthropic import BadRequestError
                is_400 = isinstance(e, BadRequestError)
            except Exception:
                is_400 = 'BadRequest' in type(e).__name__ or '400' in str(e)[:80]
            if not is_400:
                raise
            sys_text_fb = (sys_text + '\n\n' if sys_text else '') + ('Return ONLY a single JSON object matching this schema. No prose, no code fences:\n' + json.dumps(_schema_to_dict(schema), ensure_ascii=False))
            kwargs_fb = dict(kwargs)
            kwargs_fb['system'] = [{'type': 'text', 'text': sys_text_fb, 'cache_control': {'type': 'ephemeral'}}]
            resp = client.messages.create(**kwargs_fb)
    else:
        resp = client.messages.create(**kwargs)
    parts: list[str] = []
    for blk in resp.content or []:
        t = getattr(blk, 'text', None)
        if isinstance(t, str) and t:
            parts.append(t)
        if parsed_obj is None:
            inp = getattr(blk, 'input', None)
            if isinstance(inp, dict):
                try:
                    parsed_obj = schema.model_validate(inp) if schema is not None else None
                    parts.append(json.dumps(inp, ensure_ascii=False))
                except Exception:
                    pass
    text = '\n'.join(parts).strip()
    if parsed_obj is not None and (not text):
        try:
            text = parsed_obj.model_dump_json()
        except Exception:
            text = ''
    usage = getattr(resp, 'usage', None)
    in_t = int(getattr(usage, 'input_tokens', 0) or 0) if usage else 0
    out_t = int(getattr(usage, 'output_tokens', 0) or 0) if usage else 0
    cr = int(getattr(usage, 'cache_read_input_tokens', 0) or 0) if usage else 0
    cw = int(getattr(usage, 'cache_creation_input_tokens', 0) or 0) if usage else 0
    return LLMCallResult(text=text, parsed=parsed_obj, provider='anthropic', model=model, input_tokens=in_t, output_tokens=out_t, cache_read_tokens=cr, cache_creation_tokens=cw, raw_response=resp)

def _openai_get_or_upload(client: Any, pdf_path: Path) -> str:
    h = _pdf_hash(pdf_path)
    while True:
        with _CACHE_LOCK:
            cached = _OPENAI_FILE_CACHE.get(h)
            if cached:
                return cached
            ev = _OPENAI_INFLIGHT.get(h)
            if ev is None:
                ev = _OPENAI_INFLIGHT[h] = threading.Event()
                break
        ev.wait()
    try:
        with open(pdf_path, 'rb') as f:
            uploaded = client.files.create(file=f, purpose='user_data')
        file_id = uploaded.id
        with _CACHE_LOCK:
            _OPENAI_FILE_CACHE[h] = file_id
            _OPENAI_INFLIGHT.pop(h, None)
        ev.set()
        return file_id
    except BaseException:
        with _CACHE_LOCK:
            _OPENAI_INFLIGHT.pop(h, None)
        ev.set()
        raise

def _call_openai(*, model: str, system_prompt: Optional[str], user_prompt: str, schema: Optional[Type[BaseModel]], pdf_path: Optional[Path], max_tokens: int, api_keys: Optional[Dict[str, Any]], reasoning: Optional[Dict[str, Any]]=None, temperature: Optional[float]=None) -> LLMCallResult:
    key = (api_keys or {}).get('OPENAI') or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OpenAI API key not found (set OPENAI_API_KEY).')
    client = make_hardened_openai_client(key)
    content_blocks: list[Dict[str, Any]] = []
    file_id: Optional[str] = None
    if pdf_path is not None:
        file_id = _openai_get_or_upload(client, Path(pdf_path))
        content_blocks.append({'type': 'input_file', 'file_id': file_id})
    content_blocks.append({'type': 'input_text', 'text': user_prompt})
    create_kwargs: Dict[str, Any] = {'model': model, 'input': [{'role': 'user', 'content': content_blocks}], 'instructions': system_prompt or None, 'max_output_tokens': max_tokens}
    if file_id is not None:
        create_kwargs['prompt_cache_key'] = f'paper-pdf-{file_id}'
    else:
        import hashlib as _hashlib
        _key_seed = (system_prompt or '') + '\x1f' + (user_prompt or '')[:256] + '\x1f' + str(model)
        _key = _hashlib.sha1(_key_seed.encode('utf-8')).hexdigest()[:24]
        create_kwargs['prompt_cache_key'] = f'text-{_key}'
    if reasoning is not None:
        create_kwargs['reasoning'] = reasoning
    if temperature is not None:
        create_kwargs['temperature'] = float(temperature)
    parsed_obj: Any = None
    if schema is not None:
        try:
            resp = client.responses.parse(text_format=schema, **create_kwargs)
            parsed_obj = getattr(resp, 'output_parsed', None)
        except Exception as e:
            from openai import BadRequestError
            from pydantic import ValidationError as _PVE
            err_str = str(e)[:600]
            is_truncation = isinstance(e, _PVE) and ('Invalid JSON' in err_str or 'EOF while parsing' in err_str or 'json_invalid' in err_str)
            if is_truncation:
                old_max = int(create_kwargs.get('max_output_tokens') or 0)
                new_max = min(max(old_max * 4, 64000), 200000)
                if new_max > old_max:
                    print(f'[openai-truncation-retry] {agent_name}: JSON truncated mid-emission at max_output_tokens={old_max}; retrying with max_output_tokens={new_max}.', flush=True)
                    create_kwargs['max_output_tokens'] = new_max
                    try:
                        resp = client.responses.parse(text_format=schema, **create_kwargs)
                        parsed_obj = getattr(resp, 'output_parsed', None)
                    except Exception as e2:
                        if isinstance(e2, BadRequestError):
                            create_kwargs['text'] = {'format': {'type': 'json_schema', 'name': _schema_name(schema), 'schema': _to_strict_json_schema(_schema_to_dict(schema)), 'strict': False}}
                            resp = client.responses.create(**create_kwargs)
                        else:
                            print(f'[openai-truncation-retry] {agent_name}: SECOND attempt also failed ({type(e2).__name__}); raising original exception.', flush=True)
                            raise
                else:
                    raise
            elif isinstance(e, BadRequestError):
                create_kwargs['text'] = {'format': {'type': 'json_schema', 'name': _schema_name(schema), 'schema': _to_strict_json_schema(_schema_to_dict(schema)), 'strict': False}}
                resp = client.responses.create(**create_kwargs)
            else:
                raise
    else:
        resp = client.responses.create(**create_kwargs)
    text = getattr(resp, 'output_text', None) or ''
    if not text:
        parts: list[str] = []
        for item in getattr(resp, 'output', []) or []:
            if getattr(item, 'type', None) == 'message':
                for blk in getattr(item, 'content', []) or []:
                    t = getattr(blk, 'text', None)
                    if isinstance(t, str) and t:
                        parts.append(t)
        text = '\n'.join(parts).strip()
    usage = getattr(resp, 'usage', None)
    in_t = 0
    out_t = 0
    cache_read = 0
    if usage is not None:
        if isinstance(usage, dict):
            in_t = int(usage.get('input_tokens') or usage.get('prompt_tokens') or 0)
            out_t = int(usage.get('output_tokens') or usage.get('completion_tokens') or 0)
            details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
            if isinstance(details, dict):
                cache_read = int(details.get('cached_tokens') or 0)
        else:
            in_t = int(getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0) or 0)
            out_t = int(getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0) or 0)
            details = getattr(usage, 'input_tokens_details', None) or getattr(usage, 'prompt_tokens_details', None)
            if details is not None:
                cache_read = int(getattr(details, 'cached_tokens', 0) or 0)
    return LLMCallResult(text=text, parsed=parsed_obj, provider='openai', model=model, input_tokens=in_t, output_tokens=out_t, cache_read_tokens=cache_read, cache_creation_tokens=0, raw_response=resp)

def _gemini_get_or_upload(client: Any, pdf_path: Path) -> Any:
    h = _pdf_hash(pdf_path)
    while True:
        with _CACHE_LOCK:
            cached = _GEMINI_FILE_CACHE.get(h)
            if cached is not None:
                return cached
            ev = _GEMINI_INFLIGHT.get(h)
            if ev is None:
                ev = _GEMINI_INFLIGHT[h] = threading.Event()
                break
        ev.wait()
    try:
        uploaded = client.files.upload(file=str(pdf_path))
        with _CACHE_LOCK:
            _GEMINI_FILE_CACHE[h] = uploaded
            _GEMINI_INFLIGHT.pop(h, None)
        ev.set()
        return uploaded
    except BaseException:
        with _CACHE_LOCK:
            _GEMINI_INFLIGHT.pop(h, None)
        ev.set()
        raise

def _call_gemini(*, model: str, system_prompt: Optional[str], user_prompt: str, schema: Optional[Type[BaseModel]], pdf_path: Optional[Path], max_tokens: int, api_keys: Optional[Dict[str, Any]], temperature: Optional[float]=None) -> LLMCallResult:
    key = (api_keys or {}).get('GEMINI') or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not key:
        raise RuntimeError('Gemini API key not found (set GEMINI_API_KEY or GOOGLE_API_KEY).')
    try:
        from google import genai
        from google.genai import types as gtypes
        client = genai.Client(api_key=key)
        contents: list[Any] = []
        if pdf_path is not None:
            uploaded = _gemini_get_or_upload(client, Path(pdf_path))
            contents.append(uploaded)
        contents.append(user_prompt)
        cfg_kwargs: Dict[str, Any] = {'max_output_tokens': max_tokens}
        if system_prompt:
            cfg_kwargs['system_instruction'] = system_prompt
        if temperature is not None:
            cfg_kwargs['temperature'] = float(temperature)
        if schema is not None:
            cfg_kwargs['response_mime_type'] = 'application/json'
            cfg_kwargs['response_json_schema'] = _to_strict_json_schema(_schema_to_dict(schema))
        config = gtypes.GenerateContentConfig(**cfg_kwargs)
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        text = getattr(resp, 'text', '') or ''
        parsed_obj = None
        usage = getattr(resp, 'usage_metadata', None)
        in_t = int(getattr(usage, 'prompt_token_count', 0) or 0) if usage else 0
        out_visible = int(getattr(usage, 'candidates_token_count', 0) or 0) if usage else 0
        out_thoughts = int(getattr(usage, 'thoughts_token_count', 0) or 0) if usage else 0
        out_t = out_visible + out_thoughts
        cache_read = int(getattr(usage, 'cached_content_token_count', 0) or 0) if usage else 0
        return LLMCallResult(text=text, parsed=parsed_obj, provider='gemini', model=model, input_tokens=in_t, output_tokens=out_t, cache_read_tokens=cache_read, raw_response=resp)
    except ModuleNotFoundError:
        import google.generativeai as legacy
        legacy.configure(api_key=key)
        h = _pdf_hash(pdf_path) if pdf_path else None
        uploaded = None
        if pdf_path is not None and h is not None:
            with _CACHE_LOCK:
                uploaded = _GEMINI_FILE_CACHE.get(h)
            if uploaded is None:
                uploaded = legacy.upload_file(str(pdf_path))
                with _CACHE_LOCK:
                    _GEMINI_FILE_CACHE[h] = uploaded
        gen_config: Dict[str, Any] = {'max_output_tokens': max_tokens}
        if temperature is not None:
            gen_config['temperature'] = float(temperature)
        if schema is not None:
            gen_config['response_mime_type'] = 'application/json'
            user_prompt = user_prompt + '\n\nReturn ONLY a single JSON object matching this JSON schema. No prose, no code fences:\n' + json.dumps(_schema_to_dict(schema), ensure_ascii=False)
        m = legacy.GenerativeModel(model, system_instruction=system_prompt) if system_prompt else legacy.GenerativeModel(model)
        contents = [uploaded, user_prompt] if uploaded is not None else [user_prompt]
        resp = m.generate_content(contents, generation_config=gen_config)
        text = getattr(resp, 'text', '') or ''
        usage = getattr(resp, 'usage_metadata', None)
        in_t = int(getattr(usage, 'prompt_token_count', 0) or 0) if usage else 0
        out_visible = int(getattr(usage, 'candidates_token_count', 0) or 0) if usage else 0
        out_thoughts = int(getattr(usage, 'thoughts_token_count', 0) or 0) if usage else 0
        out_t = out_visible + out_thoughts
        cache_read = int(getattr(usage, 'cached_content_token_count', 0) or 0) if usage else 0
        return LLMCallResult(text=text, parsed=None, provider='gemini', model=model, input_tokens=in_t, output_tokens=out_t, cache_read_tokens=cache_read, raw_response=resp)

def resolve_provider(model: str) -> Provider:
    m = (model or '').lower()
    if m.startswith('claude') or 'anthropic/' in m:
        return 'anthropic'
    if m.startswith('gemini') or m.startswith('google/'):
        return 'gemini'
    return 'openai'

def call_with_schema(*, model: str, system_prompt: Optional[str], user_prompt: str, schema: Optional[Type[BaseModel]]=None, pdf_path: Optional[Union[str, Path]]=None, provider: Optional[Provider]=None, agent_name: str='skepthical', max_tokens: int=128000, api_keys: Optional[Dict[str, Any]]=None, reasoning: Optional[Dict[str, Any]]=None, temperature: Optional[float]=None) -> LLMCallResult:
    if provider is None:
        provider = resolve_provider(model)
    pdf_p = Path(pdf_path) if pdf_path is not None else None
    if pdf_p is not None and (not pdf_p.is_file()):
        raise FileNotFoundError(f'PDF not found: {pdf_p}')
    if provider == 'anthropic':
        result = _call_anthropic(model=model, system_prompt=system_prompt, user_prompt=user_prompt, schema=schema, pdf_path=pdf_p, max_tokens=max_tokens, api_keys=api_keys, temperature=temperature)
    elif provider == 'openai':
        result = _call_openai(model=model, system_prompt=system_prompt, user_prompt=user_prompt, schema=schema, pdf_path=pdf_p, max_tokens=max_tokens, api_keys=api_keys, reasoning=reasoning, temperature=temperature)
    elif provider == 'gemini':
        result = _call_gemini(model=model, system_prompt=system_prompt, user_prompt=user_prompt, schema=schema, pdf_path=pdf_p, max_tokens=max_tokens, api_keys=api_keys, temperature=temperature)
    else:
        raise ValueError(f'unknown provider {provider!r}')
    result.cost_usd = _compute_cost(provider=provider, model=model, input_tokens=result.input_tokens, output_tokens=result.output_tokens, cache_read_tokens=result.cache_read_tokens, cache_creation_tokens=result.cache_creation_tokens)
    try:
        if provider == 'anthropic':
            effective_in = int(result.input_tokens + 1.25 * result.cache_creation_tokens + 0.1 * result.cache_read_tokens)
            _cost.record_openai_tokens_cost(agent_name=agent_name, model=model, input_tokens=effective_in, output_tokens=result.output_tokens, cached_input_tokens=0, verbose=False)
        else:
            _cost.record_openai_tokens_cost(agent_name=agent_name, model=model, input_tokens=result.input_tokens, output_tokens=result.output_tokens, cached_input_tokens=result.cache_read_tokens, verbose=False)
    except Exception:
        pass
    if schema is not None and result.parsed is None and result.text:
        try:
            payload = _strip_fences(result.text)
            data = json.loads(payload)
            result.parsed = schema.model_validate(data)
        except Exception as _validation_err:
            try:
                _schema_label = getattr(schema, '__name__', str(schema))
                print(f'[call_with_schema] schema-validate FAILED for provider={result.provider} model={result.model} schema={_schema_label}: {type(_validation_err).__name__}: {str(_validation_err)[:240]}.  parsed=None; caller will fall back to manual parse.', flush=True)
            except Exception:
                pass
    return result
