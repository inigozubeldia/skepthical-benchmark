"""Provider-agnostic LLM client (Anthropic, OpenAI, Gemini) with PDF input and cost logging.

PDFs are uploaded once per process and reused across calls (keyed on the
SHA-256 of the bytes) so provider-side prefix caching applies.
"""
from __future__ import annotations

import base64
import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from .cost import log_call


Provider = Literal["anthropic", "openai", "gemini"]


@dataclass
class LLMResult:
    text: str
    provider: Provider
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    parsed: Optional[Any] = None


_OPENAI_FILE_CACHE: Dict[str, str] = {}
_GEMINI_FILE_CACHE: Dict[str, Any] = {}
_PDF_REFCOUNT: Dict[str, int] = {}
_CACHE_LOCK = threading.Lock()


def _pdf_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clear_pdf_caches() -> None:
    """Drop the in-process upload caches; remote files are not deleted."""
    with _CACHE_LOCK:
        _OPENAI_FILE_CACHE.clear()
        _GEMINI_FILE_CACHE.clear()
        _PDF_REFCOUNT.clear()


def acquire_pdf_ref(pdf_hash: str) -> None:
    with _CACHE_LOCK:
        _PDF_REFCOUNT[pdf_hash] = _PDF_REFCOUNT.get(pdf_hash, 0) + 1


def release_pdf_ref(pdf_hash: str) -> None:
    with _CACHE_LOCK:
        n = _PDF_REFCOUNT.get(pdf_hash, 0) - 1
        if n <= 0:
            _PDF_REFCOUNT.pop(pdf_hash, None)
        else:
            _PDF_REFCOUNT[pdf_hash] = n


def delete_pdf_remote(
    pdf_path: Path,
    *,
    api_keys: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, bool]:
    """Delete this PDF's OpenAI and Gemini uploads and drop its cache entry.

        A no-op returning {"deferred": True} while another call holds a reference,
        unless force=True.
    """
    h = _pdf_hash(pdf_path)
    out: Dict[str, bool] = {"openai": True, "gemini": True, "deferred": False}

    with _CACHE_LOCK:
        live = _PDF_REFCOUNT.get(h, 0)
        if live > 0 and not force:
            out["deferred"] = True
            return out
        oai_id = _OPENAI_FILE_CACHE.pop(h, None)
        gem_file = _GEMINI_FILE_CACHE.pop(h, None)
        _PDF_REFCOUNT.pop(h, None)

    if oai_id:
        try:
            from openai import OpenAI
            key = (api_keys or {}).get("OPENAI") or os.environ.get("OPENAI_API_KEY")
            if key:
                OpenAI(api_key=key).files.delete(oai_id)
        except Exception as exc:
            print(f"[llm_client] OpenAI file delete failed for {oai_id}: {exc}")
            out["openai"] = False

    if gem_file is not None:
        try:
            from google import genai
            key = (
                (api_keys or {}).get("GEMINI")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if key:
                client = genai.Client(api_key=key)
                name = getattr(gem_file, "name", None) or str(gem_file)
                client.files.delete(name=name)
        except ModuleNotFoundError:
            try:
                import google.generativeai as legacy
                key = (
                    (api_keys or {}).get("GEMINI")
                    or os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("GOOGLE_API_KEY")
                )
                if key:
                    legacy.configure(api_key=key)
                    name = getattr(gem_file, "name", None) or str(gem_file)
                    legacy.delete_file(name)
            except Exception as exc:
                print(f"[llm_client] Gemini (legacy) file delete failed: {exc}")
                out["gemini"] = False
        except Exception as exc:
            print(f"[llm_client] Gemini file delete failed: {exc}")
            out["gemini"] = False

    return out


def _call_anthropic(
    *,
    model: str,
    prompt: str,
    pdf_path: Path,
    system: Optional[str],
    max_tokens: int,
    api_keys: Optional[Dict[str, Any]],
    schema: Optional[type] = None,
) -> LLMResult:
    import anthropic
    key = (api_keys or {}).get("ANTHROPIC") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Anthropic API key not found (set ANTHROPIC_API_KEY).")

    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    # Explicit timeout: the SDK refuses large max_tokens without one.
    from skepthical.llm_call import make_hardened_anthropic_client
    client = make_hardened_anthropic_client(key)
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "timeout": 3600.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    if system:
        kwargs["system"] = system

    parsed_obj = None
    if schema is not None:
        try:
            resp = client.messages.parse(output_format=schema, **kwargs)
            parsed_obj = getattr(resp, "parsed", None)
        except Exception as _e:
            try:
                from anthropic import BadRequestError  # type: ignore
                _is_400 = isinstance(_e, BadRequestError)
            except Exception:
                _is_400 = "BadRequest" in type(_e).__name__
            if not _is_400:
                raise
            resp = client.messages.create(**kwargs)
    else:
        resp = client.messages.create(**kwargs)

    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0

    if parsed_obj is not None:
        text = parsed_obj.model_dump_json(exclude_none=False)
    else:
        parts: List[str] = []
        for blk in resp.content or []:
            t = getattr(blk, "text", None)
            if isinstance(t, str) and t:
                parts.append(t)
        text = "\n".join(parts).strip()

    return LLMResult(
        text=text,
        provider="anthropic", model=model,
        input_tokens=in_tok, output_tokens=out_tok,
        cache_read_tokens=cache_read, cache_creation_tokens=cache_write,
    )


def _openai_get_or_upload(client: Any, pdf_path: Path) -> str:
    h = _pdf_hash(pdf_path)
    with _CACHE_LOCK:
        cached = _OPENAI_FILE_CACHE.get(h)
    if cached:
        return cached
    with open(pdf_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")
    file_id = uploaded.id
    with _CACHE_LOCK:
        _OPENAI_FILE_CACHE[h] = file_id
    return file_id


def _call_openai(
    *,
    model: str,
    prompt: str,
    pdf_path: Path,
    system: Optional[str],
    max_tokens: int,
    api_keys: Optional[Dict[str, Any]],
    reasoning: Optional[Dict[str, Any]] = None,
    schema: Optional[type] = None,
) -> LLMResult:
    key = (api_keys or {}).get("OPENAI") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not found (set OPENAI_API_KEY).")

    from skepthical.llm_call import make_hardened_openai_client
    client = make_hardened_openai_client(key)

    file_id = _openai_get_or_upload(client, pdf_path)

    # Hold a reference so a concurrent delete_pdf_remote() cannot remove the file mid-call.
    _h_for_ref = _pdf_hash(pdf_path)
    acquire_pdf_ref(_h_for_ref)
    try:
        instructions = system if system else ""
        content_blocks: List[Dict[str, Any]] = [
            {"type": "input_file", "file_id": file_id},
            {"type": "input_text", "text": prompt},
        ]
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "input": [{"role": "user", "content": content_blocks}],
            "instructions": instructions or None,
            "max_output_tokens": max_tokens,
            "prompt_cache_key": f"paper-pdf-{file_id}",
        }
        if reasoning is not None:
            create_kwargs["reasoning"] = reasoning
        parsed_obj = None
        if schema is not None:
            try:
                resp = client.responses.parse(text_format=schema, **create_kwargs)
                parsed_obj = getattr(resp, "output_parsed", None)
            except Exception as _e:
                try:
                    from openai import BadRequestError  # type: ignore
                    _is_400 = isinstance(_e, BadRequestError)
                except Exception:
                    _is_400 = "BadRequest" in type(_e).__name__
                if not _is_400:
                    raise
                resp = client.responses.create(**create_kwargs)
        else:
            resp = client.responses.create(**create_kwargs)
    finally:
        release_pdf_ref(_h_for_ref)

    if parsed_obj is not None:
        text = parsed_obj.model_dump_json(exclude_none=False)
    else:
        text = getattr(resp, "output_text", None)
        if not text:
            parts: List[str] = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for blk in getattr(item, "content", []) or []:
                        t = getattr(blk, "text", None)
                        if isinstance(t, str) and t:
                            parts.append(t)
            text = "\n".join(parts).strip()

    usage = getattr(resp, "usage", None)
    in_tok = 0
    out_tok = 0
    cache_read = 0
    if usage is not None:
        if isinstance(usage, dict):
            in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                cache_read = int(details.get("cached_tokens") or 0)
        else:
            in_tok = int(
                getattr(usage, "input_tokens", 0)
                or getattr(usage, "prompt_tokens", 0)
                or 0
            )
            out_tok = int(
                getattr(usage, "output_tokens", 0)
                or getattr(usage, "completion_tokens", 0)
                or 0
            )
            details = (
                getattr(usage, "input_tokens_details", None)
                or getattr(usage, "prompt_tokens_details", None)
            )
            if details is not None:
                cache_read = int(getattr(details, "cached_tokens", 0) or 0)

    return LLMResult(
        text=text or "",
        provider="openai", model=model,
        input_tokens=in_tok, output_tokens=out_tok,
        cache_read_tokens=cache_read, cache_creation_tokens=0,
    )


def _gemini_get_or_upload(client: Any, pdf_path: Path) -> Any:
    h = _pdf_hash(pdf_path)
    with _CACHE_LOCK:
        cached = _GEMINI_FILE_CACHE.get(h)
    if cached is not None:
        return cached
    uploaded = client.files.upload(file=str(pdf_path))
    with _CACHE_LOCK:
        _GEMINI_FILE_CACHE[h] = uploaded
    return uploaded


def _call_gemini(
    *,
    model: str,
    prompt: str,
    pdf_path: Path,
    system: Optional[str],
    max_tokens: int,
    api_keys: Optional[Dict[str, Any]],
    schema: Optional[type] = None,
) -> LLMResult:
    key = (
        (api_keys or {}).get("GEMINI")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise RuntimeError("Gemini API key not found (set GEMINI_API_KEY or GOOGLE_API_KEY).")

    _h_for_ref = _pdf_hash(pdf_path)
    try:
        from google import genai
        from google.genai import types as gtypes
        client = genai.Client(api_key=key)
        uploaded = _gemini_get_or_upload(client, pdf_path)
        acquire_pdf_ref(_h_for_ref)
        try:
            cfg_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
            if system:
                cfg_kwargs["system_instruction"] = system
            if schema is not None:
                from skepthical.llm_call import _to_strict_json_schema as _strictify
                cfg_kwargs["response_mime_type"] = "application/json"
                cfg_kwargs["response_json_schema"] = _strictify(
                    schema.model_json_schema()
                )
            config = gtypes.GenerateContentConfig(**cfg_kwargs)
            resp = client.models.generate_content(
                model=model,
                contents=[uploaded, prompt],
                config=config,
            )
        finally:
            release_pdf_ref(_h_for_ref)
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        cache_read = int(getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0
        return LLMResult(
            text=text, provider="gemini", model=model,
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read_tokens=cache_read,
        )
    except ModuleNotFoundError:
        import google.generativeai as legacy
        legacy.configure(api_key=key)
        h = _pdf_hash(pdf_path)
        with _CACHE_LOCK:
            uploaded = _GEMINI_FILE_CACHE.get(h)
        if uploaded is None:
            uploaded = legacy.upload_file(str(pdf_path))
            with _CACHE_LOCK:
                _GEMINI_FILE_CACHE[h] = uploaded
        acquire_pdf_ref(h)
        try:
            gen_config: Dict[str, Any] = {"max_output_tokens": max_tokens}
            m = legacy.GenerativeModel(model, system_instruction=system) if system else legacy.GenerativeModel(model)
            resp = m.generate_content([uploaded, prompt], generation_config=gen_config)
        finally:
            release_pdf_ref(h)
        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        out_tok = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        cache_read = int(getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0
        return LLMResult(
            text=text, provider="gemini", model=model,
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read_tokens=cache_read,
        )


def call_with_pdf(
    *,
    provider: Provider,
    model: str,
    prompt: str,
    pdf_path: Path,
    agent_name: str,
    paper_id: str,
    paper_dir: Path,
    system: Optional[str] = None,
    max_tokens: int = 16000,
    api_keys: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    schema: Optional[type] = None,
) -> LLMResult:
    """Call `model` on `provider` with the PDF and prompt; log cost.

        With `schema` (a Pydantic class), uses strict structured output, falling back
        to a non-strict call if the provider rejects the schema.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if provider == "anthropic":
        result = _call_anthropic(
            model=model, prompt=prompt, pdf_path=pdf_path, system=system,
            max_tokens=max_tokens, api_keys=api_keys, schema=schema,
        )
    elif provider == "openai":
        result = _call_openai(
            model=model, prompt=prompt, pdf_path=pdf_path, system=system,
            max_tokens=max_tokens, api_keys=api_keys, reasoning=reasoning,
            schema=schema,
        )
    elif provider == "gemini":
        result = _call_gemini(
            model=model, prompt=prompt, pdf_path=pdf_path, system=system,
            max_tokens=max_tokens, api_keys=api_keys, schema=schema,
        )
    else:
        raise ValueError(f"unknown provider {provider!r}")

    row = log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        provider=provider,
        cache_read_tokens=result.cache_read_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        extras={
            "provider": provider,
            "pdf_path": str(pdf_path),
            "pdf_bytes": pdf_path.stat().st_size,
        },
    )
    result.cost_usd = float(row.get("cost_usd", 0.0))

    return result


def call_text(
    *,
    provider: Provider,
    model: str,
    prompt: str,
    agent_name: str,
    paper_id: str,
    paper_dir: Path,
    system: Optional[str] = None,
    max_tokens: int = 16000,
    api_keys: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    schema: Optional[type] = None,
) -> LLMResult:
    """Text-only counterpart of `call_with_pdf`."""
    from skepthical.llm_call import call_with_schema as _sk_call
    sk = _sk_call(
        model=model,
        system_prompt=system,
        user_prompt=prompt,
        schema=schema,
        pdf_path=None,
        provider=provider,
        agent_name=agent_name,
        max_tokens=max_tokens,
        api_keys=api_keys,
        reasoning=reasoning,
    )

    result = LLMResult(
        text=sk.text or "",
        provider=provider,
        model=model,
        input_tokens=int(sk.input_tokens or 0),
        output_tokens=int(sk.output_tokens or 0),
        cache_read_tokens=int(sk.cache_read_tokens or 0),
        cache_creation_tokens=int(sk.cache_creation_tokens or 0),
        parsed=sk.parsed,
    )

    row = log_call(
        paper_id=paper_id, paper_dir=paper_dir,
        agent_name=agent_name, model=model,
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        provider=provider,
        cache_read_tokens=result.cache_read_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
        extras={"provider": provider, "stage": "text"},
    )
    result.cost_usd = float(row.get("cost_usd", 0.0))
    return result
