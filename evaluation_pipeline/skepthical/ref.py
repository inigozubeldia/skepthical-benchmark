"""Reference resolution, arXiv download and claim verification against a cited paper."""
import re
import hashlib
from pathlib import Path
import requests
import json
import os
from .search import resolve_arxiv_id_from_reference
import json
from typing import Any, List, Optional
from .schemas import VERIFIER_VERDICT_VALUES, VERDICT_VALUES
_VERIFY_PROMPT_VERSION = 'v6-directional-contradiction'
_CROP_MAX_PAGES = 20
_CROP_HEAD_PAGES = 10
_CROP_TAIL_PAGES = 10

def _crop_pdf_if_too_large(src_path, work_dir, max_pages: int=_CROP_MAX_PAGES, head: int=_CROP_HEAD_PAGES, tail: int=_CROP_TAIL_PAGES):
    src = Path(src_path)
    if not src.exists():
        return (src, False)
    if work_dir is None:
        return (src, False)
    try:
        import pymupdf
    except Exception:
        try:
            import fitz as pymupdf
        except Exception as e:
            print(f'[WARN] pymupdf unavailable; skipping crop of {src.name}: {e}')
            return (src, False)
    try:
        doc = pymupdf.open(src)
    except Exception as e:
        print(f'[WARN] Could not open {src} for crop check: {e}')
        return (src, False)
    try:
        n_pages = doc.page_count
        if n_pages <= max_pages:
            return (src, False)
        out_dir = Path(work_dir) / 'papers_cropped'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{src.stem}_h{head}_t{tail}.pdf'
        if out_path.exists():
            try:
                if out_path.stat().st_size > 1024:
                    return (out_path, True)
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
        keep_head = list(range(min(head, n_pages)))
        keep_tail = list(range(max(n_pages - tail, head), n_pages))
        keep = sorted(set(keep_head + keep_tail))
        new_doc = pymupdf.open()
        try:
            for p in keep:
                new_doc.insert_pdf(doc, from_page=p, to_page=p)
            new_doc.save(out_path)
        finally:
            new_doc.close()
        print(f'[INFO] Cropped {src.name}: {n_pages} pages → {len(keep)} (head {head} + tail {tail}) at {out_path.name}')
        return (out_path, True)
    except Exception as e:
        print(f'[WARN] Crop failed for {src.name}; using original: {e}')
        return (src, False)
    finally:
        try:
            doc.close()
        except Exception:
            pass

def download_arxiv_pdf(arxiv_id, work_dir):
    papers_dir = Path(work_dir) / 'papers'
    papers_dir.mkdir(exist_ok=True, parents=True)
    safe_id = arxiv_id.replace('/', '_')
    pdf_path = papers_dir / f'{safe_id}.pdf'
    if pdf_path.exists():
        try:
            size = pdf_path.stat().st_size
            with open(pdf_path, 'rb') as fh:
                magic = fh.read(5)
            if size > 1024 and magic == b'%PDF-':
                print(f'[INFO] PDF already exists: {pdf_path.name}')
                return pdf_path
            print(f'[WARN] Cached PDF {pdf_path.name} looks corrupt (size={size}, magic={magic!r}); re-downloading.')
            pdf_path.unlink(missing_ok=True)
        except Exception as e:
            print(f'[WARN] Could not validate cached PDF {pdf_path.name}: {e}; re-downloading.')
            try:
                pdf_path.unlink(missing_ok=True)
            except Exception:
                pass
    if '/' in arxiv_id:
        url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    else:
        url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
    print(f'[INFO] Downloading PDF from: {url}')
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)
        print(f'[SUCCESS] Saved to {pdf_path}')
        return pdf_path
    except Exception as e:
        print(f'[WARN] Could not download {arxiv_id}: {e}')
        return None

def _hash_pdf_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def _verification_cache_key(stmt: str, pdf_paths: List[str], model: str, prompt_version: str) -> str:
    pdf_hashes = sorted((_hash_pdf_file(p) for p in pdf_paths))
    h = hashlib.sha256()
    h.update(prompt_version.encode('utf-8'))
    h.update(b'\x00')
    h.update(model.encode('utf-8'))
    h.update(b'\x00')
    h.update(stmt.encode('utf-8'))
    h.update(b'\x00')
    for ph in pdf_hashes:
        h.update(ph.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()

def _verification_cache_dir(work_dir) -> Optional[Path]:
    if not work_dir:
        return None
    d = Path(work_dir) / 'verification_cache'
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f'[WARN] Could not create verification cache dir {d}: {e}')
        return None
    return d

def _verification_cache_load(cache_dir: Optional[Path], key: str):
    if cache_dir is None:
        return None
    p = cache_dir / f'{key}.json'
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        verdict = data.get('verdict')
        reason = data.get('reason')
        if verdict in VERDICT_VALUES and isinstance(reason, str):
            return (verdict, reason)
    except Exception as e:
        print(f'[WARN] Could not read verification cache {p}: {e}')
    return None

def _verification_cache_store(cache_dir: Optional[Path], key: str, verdict: str, reason: str) -> None:
    if cache_dir is None:
        return
    p = cache_dir / f'{key}.json'
    try:
        p.write_text(json.dumps({'verdict': verdict, 'reason': reason}, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f'[WARN] Could not write verification cache {p}: {e}')

def verify_claim_with_paper(stmt, pdf_paths, params_skepthical, api_keys, ref_labels=None, work_dir=None):
    import os
    import json
    from pathlib import Path
    from openai import OpenAI
    from .cost import record_openai_response_cost
    model_type = params_skepthical['models'].get('paper_verification_agent', 'gpt-5')
    _model_lc = (model_type or '').strip().lower()
    if _model_lc.startswith('claude'):
        _provider = 'anthropic'
    elif _model_lc.startswith('gemini'):
        _provider = 'gemini'
    else:
        _provider = 'openai'
    _provider_key_map = {'openai': ('OPENAI', 'OPENAI_API_KEY'), 'anthropic': ('ANTHROPIC', 'ANTHROPIC_API_KEY'), 'gemini': ('GEMINI', 'GEMINI_API_KEY')}
    _ak, _envk = _provider_key_map[_provider]
    if not (api_keys and api_keys.get(_ak)) and (not os.environ.get(_envk)):
        print(f'[WARN] Missing {_ak} API key; marking verdict=UNVERIFIABLE.')
        return ('UNVERIFIABLE', f'Missing {_ak} API key; verification not attempted.')
    if isinstance(pdf_paths, (str, Path)):
        pdf_paths = [pdf_paths]
    pdf_paths = [str(p) for p in pdf_paths if p and os.path.exists(str(p))]
    if not pdf_paths:
        print(f'[WARN] No valid PDFs for statement:\n{stmt[:120]}...')
        return ('UNVERIFIABLE', 'No valid PDFs available; verification not attempted.')
    if not ref_labels or len(ref_labels) != len(pdf_paths):
        ref_labels = [f'Paper_{i + 1}' for i in range(len(pdf_paths))]
    cropped_flags: List[bool] = []
    cropped_paths: List[str] = []
    for p in pdf_paths:
        new_p, was_cropped = _crop_pdf_if_too_large(p, work_dir)
        cropped_paths.append(str(new_p))
        cropped_flags.append(was_cropped)
    pdf_paths = cropped_paths
    cropped_labels = [ref_labels[i] for i, flag in enumerate(cropped_flags) if flag]
    cache_dir = _verification_cache_dir(work_dir)
    cache_key = _verification_cache_key(stmt, pdf_paths, str(model_type), _VERIFY_PROMPT_VERSION)
    cached = _verification_cache_load(cache_dir, cache_key)
    if cached is not None:
        verdict, reason = cached
        print(f'[CACHE] Hit for verification (key={cache_key[:12]}…): verdict={verdict}')
        return (verdict, reason)
    _openai_key = (api_keys or {}).get('OPENAI') or os.environ.get('OPENAI_API_KEY') or 'no-key'
    from .llm_call import make_hardened_openai_client
    client = make_hardened_openai_client(_openai_key)
    plural = 'papers' if len(pdf_paths) > 1 else 'paper'
    mapping_text = '\n'.join([f'  - {label}: (attached as PDF_{i + 1})' + (' [CROPPED — first 10 + last 10 pages only]' if cropped_flags[i] else '') for i, label in enumerate(ref_labels)])
    verdict_list = ' | '.join((f'"{v}"' for v in VERIFIER_VERDICT_VALUES))
    crop_note = ''
    if cropped_labels:
        crop_note = f'\nNOTE ON CROPPED PDFS: One or more attached papers exceeded the context budget and have been cropped to the first 10 and last 10 pages only — the middle of those papers is NOT in the attachment. If the cited material would plausibly live in the body/middle of a long paper (a particular figure, table, derivation, or result that is neither in the introduction/abstract nor in the conclusion/appendix), and you cannot find it in the cropped pages, return verdict UNVERIFIABLE rather than CONTRADICTED — absence of evidence in a head+tail crop is not evidence of absence. Cropped papers: {', '.join(cropped_labels)}.\n'
    prompt = f"""You are an expert scientific reviewer. Using ONLY the attached {plural}, classify the following statement's relationship to the attached evidence into exactly one of these verdicts:\n  - SUPPORTED: the statement's specific claims (including quantitative\n    detail) are directly supported by the attached paper(s).\n  - WEAKLY_SUPPORTED: the statement is consistent with the attached\n    paper(s) but supporting detail is qualitative or indirect.\n  - PARTIALLY_SUPPORTED: some aspects of the statement are supported\n    and others are not addressed or are at odds with the evidence.\n    Includes SCOPE MISMATCHES — the cited paper makes the claim under\n    an explicit condition (a regime, dataset, parameter range, prior\n    assumption) that the citing sentence drops or generalises beyond.\n    If the cited paper's claim is bounded and the citing sentence is\n    broader, this is PARTIALLY_SUPPORTED, not SUPPORTED.\n  - CONTRADICTED: the attached paper(s) make claims that contradict\n    the statement.  This INCLUDES directional contradictions — if\n    the citing statement paraphrases the cited paper's content but\n    inverts the direction, polarity, sign of effect, or causal\n    arrow (e.g. 'X improves Y' cited to a paper that finds 'X\n    degrades Y', or 'A is consistent with B' cited to a paper that\n    finds 'A is inconsistent with B'), the verdict is CONTRADICTED,\n    not SUPPORTED — overlapping vocabulary alone is not support if\n    the conclusion of the cited paper points the other way.\n  - UNRELATED: the attached paper(s) do not discuss the statement's\n    subject matter.\n\nEach attached PDF corresponds to the following references:\n{mapping_text}\n{crop_note}\nWhen explaining your reasoning, refer to the papers by their reference labels (e.g., {', '.join(ref_labels)}), NOT as 'Paper_1' or 'PDF_1'.\n\nRespond with JSON only:\n{{ "verdict": """ + verdict_list + f', "reason": "<brief explanation>" }}\n\nStatement:\n{stmt}'
    from pathlib import Path as _Path
    text = ''
    try:
        if _provider == 'openai':
            from .llm_call import _openai_get_or_upload
            file_ids = []
            for path in pdf_paths:
                try:
                    file_ids.append(_openai_get_or_upload(client, _Path(path)))
                except Exception as e:
                    print(f'[WARN] Could not upload {path}: {e}')
            if not file_ids:
                return ('UNVERIFIABLE', 'All PDF uploads failed; verification not attempted.')
            messages = [{'role': 'system', 'content': 'You verify scientific claims using PDF evidence.'}, {'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}] + [{'type': 'input_file', 'file_id': fid} for fid in file_ids]}]
            from .schemas import VerifierVerdict
            _verifier_parsed: Any = None
            try:
                resp = client.responses.parse(model=model_type, input=messages, max_output_tokens=8192, text_format=VerifierVerdict)
                _verifier_parsed = getattr(resp, 'output_parsed', None)
            except Exception as _e:
                try:
                    from openai import BadRequestError
                    _is_400 = isinstance(_e, BadRequestError)
                except Exception:
                    _is_400 = 'BadRequest' in type(_e).__name__
                if not _is_400:
                    raise
                resp = client.responses.create(model=model_type, input=messages, max_output_tokens=8192)
            try:
                record_openai_response_cost(resp, agent_name='paper_verification_agent', model=str(model_type), verbose=False)
            except Exception:
                pass
            if _verifier_parsed is not None:
                text = _verifier_parsed.model_dump_json()
            else:
                text = getattr(resp, 'output_text', '') or ''
            if not text and getattr(resp, 'output', None):
                chunks = []
                for item in resp.output:
                    if getattr(item, 'content', None):
                        for c in item.content:
                            if getattr(c, 'type', None) == 'output_text' and hasattr(c, 'text'):
                                chunks.append(c.text)
                if chunks:
                    text = '\n'.join(chunks).strip()
        elif _provider == 'anthropic':
            import base64 as _b64
            import anthropic as _anthropic
            from .cost import record_openai_tokens_cost
            anth_key = (api_keys or {}).get('ANTHROPIC') or os.environ.get('ANTHROPIC_API_KEY')
            from .llm_call import make_hardened_anthropic_client
            anth_client = make_hardened_anthropic_client(anth_key or '')
            doc_blocks = []
            for path in pdf_paths:
                try:
                    with open(path, 'rb') as fh:
                        b64 = _b64.standard_b64encode(fh.read()).decode('utf-8')
                    doc_blocks.append({'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': b64}, 'cache_control': {'type': 'ephemeral'}})
                except Exception as e:
                    print(f'[WARN] Could not read {path}: {e}')
            if not doc_blocks:
                return ('UNVERIFIABLE', 'All PDF reads failed; verification not attempted.')
            from .schemas import VerifierVerdict
            _anth_kwargs = dict(model=str(model_type), max_tokens=8192, system='You verify scientific claims using PDF evidence.', messages=[{'role': 'user', 'content': doc_blocks + [{'type': 'text', 'text': prompt}]}])
            _anth_parsed: Any = None
            try:
                resp = anth_client.messages.parse(output_format=VerifierVerdict, **_anth_kwargs)
                _anth_parsed = getattr(resp, 'parsed', None)
            except Exception as _e:
                try:
                    from anthropic import BadRequestError
                    _is_400 = isinstance(_e, BadRequestError)
                except Exception:
                    _is_400 = 'BadRequest' in type(_e).__name__ or '400' in str(_e)[:80]
                if not _is_400:
                    raise
                resp = anth_client.messages.create(**_anth_kwargs)
            try:
                usage = getattr(resp, 'usage', None)
                in_tok = int(getattr(usage, 'input_tokens', 0) or 0) if usage else 0
                out_tok = int(getattr(usage, 'output_tokens', 0) or 0) if usage else 0
                cache_w = int(getattr(usage, 'cache_creation_input_tokens', 0) or 0) if usage else 0
                cache_r = int(getattr(usage, 'cache_read_input_tokens', 0) or 0) if usage else 0
                effective_input = int(in_tok + 1.25 * cache_w + 0.1 * cache_r)
                record_openai_tokens_cost(agent_name='paper_verification_agent', model=str(model_type), input_tokens=effective_input, output_tokens=out_tok, verbose=False)
            except Exception:
                pass
            if _anth_parsed is not None:
                text = _anth_parsed.model_dump_json()
            else:
                parts = []
                for blk in resp.content or []:
                    t = getattr(blk, 'text', None)
                    if isinstance(t, str) and t:
                        parts.append(t)
                text = '\n'.join(parts).strip()
        else:
            from .cost import record_openai_tokens_cost
            try:
                from google import genai
                from google.genai import types as gtypes
            except ModuleNotFoundError as e:
                raise RuntimeError('google-genai SDK required for Gemini verifier.') from e
            gkey = (api_keys or {}).get('GEMINI') or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            gclient = genai.Client(api_key=gkey)
            from .llm_call import _gemini_get_or_upload
            uploads = []
            for path in pdf_paths:
                try:
                    uploads.append(_gemini_get_or_upload(gclient, _Path(path)))
                except Exception as e:
                    print(f'[WARN] Could not upload {path} to Gemini: {e}')
            if not uploads:
                return ('UNVERIFIABLE', 'All PDF uploads failed; verification not attempted.')
            real_model = str(model_type)
            thinking_level: Optional[str] = None
            for lvl in ('none', 'off', 'low', 'medium', 'high'):
                suf = f'-thinking-{lvl}'
                if real_model.endswith(suf):
                    real_model = real_model[:-len(suf)]
                    thinking_level = 'none' if lvl == 'off' else lvl
                    break
            else:
                if real_model.endswith('-thinking'):
                    real_model = real_model[:-len('-thinking')]
                    thinking_level = 'high'
            cfg_kwargs = {'max_output_tokens': 128000, 'system_instruction': 'You verify scientific claims using PDF evidence.'}
            if thinking_level is not None:
                _budget_map = {'none': 0, 'off': 0, 'low': 2048, 'medium': 8192, 'high': 24576, 'xhigh': 32768}
                _budget = _budget_map.get(thinking_level)
                if _budget is not None:
                    try:
                        cfg_kwargs['thinking_config'] = gtypes.ThinkingConfig(thinking_budget=_budget)
                    except Exception:
                        pass
            from .schemas import VerifierVerdict
            from .llm_call import _to_strict_json_schema as _strictify
            cfg_kwargs['response_mime_type'] = 'application/json'
            cfg_kwargs['response_json_schema'] = _strictify(VerifierVerdict.model_json_schema())
            from .llm_call import gemini_generate_with_budget_retry
            resp, _text_pre = gemini_generate_with_budget_retry(gclient, model=real_model, contents=list(uploads) + [prompt], config_kwargs=cfg_kwargs, label='gemini_ref_verifier')
            try:
                usage = getattr(resp, 'usage_metadata', None)
                in_tok = int(getattr(usage, 'prompt_token_count', 0) or 0) if usage else 0
                out_tok = int(getattr(usage, 'candidates_token_count', 0) or 0) if usage else 0
                cache_r = int(getattr(usage, 'cached_content_token_count', 0) or 0) if usage else 0
                uncached_in = max(in_tok - cache_r, 0)
                effective_input = int(uncached_in + 0.25 * cache_r)
                record_openai_tokens_cost(agent_name='paper_verification_agent', model=real_model, input_tokens=effective_input, output_tokens=out_tok, verbose=False)
            except Exception:
                pass
            text = getattr(resp, 'text', '') or ''
        try:
            clean = text.strip().removeprefix('```json').removesuffix('```').strip('`').strip()
            parsed = json.loads(clean)
            raw_verdict = str(parsed.get('verdict', '')).strip().upper()
            reason = parsed.get('reason', '')
            if isinstance(reason, str):
                reason = reason.strip() or 'No explanation provided.'
            else:
                reason = 'No explanation provided.'
            if raw_verdict in VERIFIER_VERDICT_VALUES:
                verdict = raw_verdict
            else:
                try:
                    print(f'[ref.verify_claim_with_paper][unrecognised_verdict] model returned {raw_verdict!r}; coercing to UNVERIFIABLE. Allowed: {sorted(VERIFIER_VERDICT_VALUES)}')
                except Exception:
                    pass
                verdict = 'UNVERIFIABLE'
                reason = f'Verifier returned unrecognised verdict {raw_verdict!r}; {reason}'
        except Exception as e:
            print(f'[WARN] Could not parse model output ({e}):\n{text}')
            verdict = 'UNVERIFIABLE'
            reason = f'Could not parse verifier JSON output: {e}'
        print(f'[RESULT] Verification result ({len(pdf_paths)} PDFs):\n  Verdict: {verdict}\n  Reason: {reason}\n')
        _verification_cache_store(cache_dir, cache_key, verdict, reason)
        return (verdict, reason)
    except Exception as e:
        print(f'[ERROR] Verification failed with {model_type}: {e}')
        return ('UNVERIFIABLE', f'Verification call failed with {model_type}: {e}')
    finally:
        pass

def extract_arxiv_metadata_from_reference_string(ref_text: str) -> dict:
    if not ref_text:
        return {'arxiv_id': None, 'doi': None}
    text = ref_text.strip()
    arxiv_id = None
    m = re.search('arxiv\\s*:\\s*([0-9]{4}\\.[0-9]{4,5}|[a-z\\-]+/[0-9]{7})', text, re.IGNORECASE)
    if m:
        arxiv_id = m.group(1).strip()
    else:
        m = re.search('arxiv\\.org/(?:abs|pdf)/([0-9]{4}\\.[0-9]{4,5}|[a-z\\-]+/[0-9]{7})', text, re.IGNORECASE)
        if m:
            arxiv_id = m.group(1).strip()
    doi = None
    m = re.search('doi\\s*[:]\\s*(10\\.\\d{4,9}/\\S+)', text, re.IGNORECASE)
    if m:
        doi = m.group(1).strip()
    else:
        m = re.search('doi\\.org/(10\\.\\d{4,9}/\\S+)', text, re.IGNORECASE)
        if m:
            doi = m.group(1).strip()
    return {'arxiv_id': arxiv_id, 'doi': doi}

def resolve_reference_with_agent(citation_label, reference_entries, params_skepthical, api_keys):
    import json
    from openai import OpenAI
    from .cost import record_openai_response_cost
    if not reference_entries:
        print('[WARN] resolve_reference_with_agent: empty reference_entries.')
        return {'arxiv_id': None, 'reference_text': None, 'resolution_status': 'not_found'}
    if not api_keys or not api_keys.get('OPENAI'):
        print('[WARN] resolve_reference_with_agent: missing OPENAI API key.')
        return {'arxiv_id': None, 'reference_text': None, 'resolution_status': 'failed'}
    ref_lines = []
    for i, ref in enumerate(reference_entries, start=1):
        if isinstance(ref, str):
            s = ref.strip()
        else:
            raise TypeError(f'resolve_reference_with_agent: reference_entries must be a list of strings; entry {i} is a {type(ref).__name__}. Normalise upstream (e.g. in parse_paper) — do not pass dict/bib objects through here.')
        ref_lines.append(f'[{i}] {s}')
    refs_block = '\n'.join(ref_lines)
    model_type = params_skepthical['models'].get('reference_resolver', params_skepthical['models'].get('paper_verification_agent', 'gpt-5-mini'))
    from .llm_call import make_hardened_openai_client
    client = make_hardened_openai_client(api_keys['OPENAI'])
    instructions = 'You are given:\n1) A list of references as they appear at the end of a scientific paper.\n2) An in-text citation label exactly as it appears in the paper.\n\nYour task:\n- Find the single reference from the list that best matches the in-text citation label.\n- Return JSON only with:\n  {"arxiv_id": "<string or null>", "reference_text": "<full reference text or null>"}\n- If unsure, return nulls.\n'
    user_payload = f'IN-TEXT CITATION LABEL:\n{citation_label}\n\nREFERENCE LIST:\n{refs_block}\n'
    messages = [{'role': 'system', 'content': 'You are an expert at matching in-text citations to bibliography entries.'}, {'role': 'user', 'content': [{'type': 'input_text', 'text': instructions}]}, {'role': 'user', 'content': [{'type': 'input_text', 'text': user_payload}]}]
    try:
        from .schemas import ReferenceResolverOutput
        _resolver_parsed = None
        try:
            resp = client.responses.parse(model=model_type, input=messages, max_output_tokens=2048, text_format=ReferenceResolverOutput)
            _resolver_parsed = getattr(resp, 'output_parsed', None)
        except Exception as _e:
            try:
                from openai import BadRequestError
                _is_400 = isinstance(_e, BadRequestError)
            except Exception:
                _is_400 = 'BadRequest' in type(_e).__name__
            if not _is_400:
                raise
            resp = client.responses.create(model=model_type, input=messages, max_output_tokens=2048)
        try:
            record_openai_response_cost(resp, agent_name='reference_resolver', model=str(model_type), verbose=False)
        except Exception:
            pass
        if _resolver_parsed is not None:
            arxiv_id = _resolver_parsed.arxiv_id
            reference_text = _resolver_parsed.reference_text
        else:
            text = getattr(resp, 'output_text', '') or ''
            if not text and getattr(resp, 'output', None):
                chunks = []
                for item in resp.output:
                    if getattr(item, 'content', None):
                        for c in item.content:
                            if getattr(c, 'type', None) == 'output_text' and hasattr(c, 'text'):
                                chunks.append(c.text)
                if chunks:
                    text = '\n'.join(chunks).strip()
            arxiv_id = None
            reference_text = None
            try:
                clean = text.strip().removeprefix('```json').removesuffix('```').strip('`').strip()
                parsed = json.loads(clean)
                arxiv_id = parsed.get('arxiv_id')
                reference_text = parsed.get('reference_text')
            except Exception as e:
                print(f'[WARN] resolve_reference_with_agent: could not parse JSON ({e}):\n{text}')
        if isinstance(arxiv_id, str):
            arxiv_id = arxiv_id.strip() or None
        else:
            arxiv_id = None
        if isinstance(reference_text, str):
            reference_text = reference_text.strip() or None
        else:
            reference_text = None
        if reference_text and (not arxiv_id) and ('extract_arxiv_metadata_from_reference_string' in globals()):
            meta = extract_arxiv_metadata_from_reference_string(reference_text)
            arxiv_id = meta.get('arxiv_id') or arxiv_id
        fallback_failed = False
        if reference_text and (not arxiv_id):
            try:
                arxiv_id = resolve_arxiv_id_from_reference(reference_text)
                if arxiv_id:
                    print(f'    ✅ Resolved arXiv from reference text: {arxiv_id}')
            except Exception as e:
                print(f'[WARN] resolve_arxiv_id_from_reference failed (pdf path): {e}')
                fallback_failed = True
        if arxiv_id:
            status = 'resolved'
        elif fallback_failed:
            status = 'failed'
        else:
            status = 'not_found'
        return {'arxiv_id': arxiv_id, 'reference_text': reference_text, 'resolution_status': status}
    except Exception as e:
        print(f'[ERROR] resolve_reference_with_agent failed: {e}')
        return {'arxiv_id': None, 'reference_text': None, 'resolution_status': 'failed'}
import re
