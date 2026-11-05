"""Resolution of bibliography entries to arXiv identifiers."""
from __future__ import annotations
import os
import re
import time
import random
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, quote_plus
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
import requests
_TRACE_MODE = os.getenv('ARXIV_RESOLVER_TRACE_MODE', 'compact').lower()
import threading as _threading
_TLS = _threading.local()

def _get_trace_current() -> Optional[List[Dict[str, Any]]]:
    return getattr(_TLS, 'trace_current', None)

def _set_trace_current(val: Optional[List[Dict[str, Any]]]) -> None:
    _TLS.trace_current = val

def _get_resolved_via() -> Optional[str]:
    return getattr(_TLS, 'resolved_via', None)

def _set_resolved_via(val: Optional[str]) -> None:
    _TLS.resolved_via = val

def _trace(tool: str, outcome: str, *, candidate: Optional[str]=None, score: Optional[float]=None, detail: str='') -> None:
    buf = _get_trace_current()
    if buf is not None:
        buf.append({'tool': tool, 'outcome': outcome, 'candidate': candidate, 'score': score, 'detail': detail})
    if _TRACE_MODE == 'none':
        return
    if _TRACE_MODE == 'full':
        return
    parts = [f'[{_ts()}] {tool}: {outcome}']
    if candidate:
        parts.append(f'id={candidate}')
    if score is not None:
        parts.append(f'score={score:.2f}')
    if detail:
        parts.append(detail)
    print('  '.join(parts))

def _trace_summary(final_id: Optional[str]) -> None:
    if _TRACE_MODE == 'none':
        return
    buf = _get_trace_current()
    if not buf:
        print(f'[{_ts()}] TRACE: (empty) final={final_id or '(none)'}')
        return
    last_by_tool: Dict[str, Dict[str, Any]] = {}
    for e in buf:
        last_by_tool[e['tool']] = e
    print(f'\n[{_ts()}] TRACE SUMMARY')
    for tool in sorted(last_by_tool.keys()):
        e = last_by_tool[tool]
        line = f'  - {tool}: {e['outcome']}'
        if e.get('candidate'):
            line += f'  id={e['candidate']}'
        if e.get('score') is not None:
            line += f'  score={e['score']:.2f}'
        if e.get('detail'):
            line += f'  ({e['detail']})'
        print(line)
    via = _get_resolved_via() or 'unknown'
    print(f'[{_ts()}] RESULT: {final_id or '(none)'} via {via}\n')

def _ts() -> str:
    return _dt.datetime.now().strftime('%H:%M:%S')

def _log(icon: str, msg: str) -> None:
    print(f'[{_ts()}] {icon} {msg}')

def _stage(msg: str) -> None:
    _log('▶', msg)

def _info(msg: str) -> None:
    _log('ℹ️ ', msg)

def _warn(msg: str) -> None:
    _log('⚠️ ', msg)

def _ok(msg: str) -> None:
    _log('✅', msg)

def _fail(msg: str) -> None:
    _log('❌', msg)
_ARXIV_NEW_RE = re.compile('\\b(\\d{4}\\.\\d{4,5})(v\\d+)?\\b')
_ARXIV_OLD_RE = re.compile('\\b([a-z][a-z0-9\\-]*(?:\\.[a-z0-9\\-]+)?/\\d{7})(v\\d+)?\\b', re.IGNORECASE)
_DOI_RE = re.compile('\\b(10\\.\\d{4,9}/[-._;()/:A-Z0-9]+)\\b', re.IGNORECASE)
_ROMAN_RE = re.compile('^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.IGNORECASE)
_HTML_RE = re.compile('<[^>]+>')
_MATH_RE = re.compile('\\$[^$]{1,400}\\$')
_WS_RE = re.compile('\\s+')

def _arxiv_id_year_month(arx: str) -> Tuple[Optional[int], Optional[int]]:
    arx = _normalize_arxiv_id(arx) or ''
    if not arx:
        return (None, None)
    m = re.fullmatch('(\\d{2})(\\d{2})\\.\\d{4,5}', arx)
    if m:
        yy, mm = (int(m.group(1)), int(m.group(2)))
        return (2000 + yy, mm)
    m2 = re.fullmatch('[a-z0-9\\-\\.]+/(\\d{2})(\\d{2})\\d{3}', arx, flags=re.IGNORECASE)
    if m2:
        yy, mm = (int(m2.group(1)), int(m2.group(2)))
        year = 1900 + yy if yy >= 91 else 2000 + yy
        return (year, mm)
    return (None, None)

def _normalize_whitespace(s: str) -> str:
    return re.sub('\\s+', ' ', s or '').strip()

def _strip_trailing_punct(s: str) -> str:
    return (s or '').strip().rstrip(' .;,')

def _clean_title_text(s: str) -> str:
    s = s or ''
    s = _HTML_RE.sub(' ', s)
    s = _MATH_RE.sub(' ', s)
    s = s.replace('{', ' ').replace('}', ' ')
    s = s.replace('\\', ' ')
    return _WS_RE.sub(' ', s).strip()

def _canonicalize_doi(doi: str) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    m = re.search('(?:https?://)?(?:dx\\.)?doi\\.org/(10\\.\\d{4,9}/[^ \\t\\r\\n<>\\"\']+)', d, flags=re.IGNORECASE)
    if m:
        d = m.group(1)
    d = re.sub('^\\s*doi\\s*:\\s*', '', d, flags=re.IGNORECASE).strip()
    d = d.rstrip(').,;]}>"\' ')
    d = d.strip()
    if not _DOI_RE.search(d):
        if not d.lower().startswith('10.'):
            return None
    return d.lower()

def _valid_new_style_arxiv_id(arx: str) -> bool:
    m = re.fullmatch('(\\d{4})\\.(\\d{4,5})', arx)
    if not m:
        return False
    yymm = m.group(1)
    mm = int(yymm[2:4])
    return 1 <= mm <= 12

def _valid_old_style_arxiv_id(arx: str) -> bool:
    m = re.fullmatch('([a-z][a-z0-9\\-]*(?:\\.[a-z0-9\\-]+)?)/(\\d{7})', arx, flags=re.IGNORECASE)
    if not m:
        return False
    digits = m.group(2)
    mm = int(digits[2:4])
    return 1 <= mm <= 12

def _normalize_arxiv_id(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    s = re.sub('^\\s*arxiv\\s*:\\s*', '', s, flags=re.IGNORECASE).strip()
    s = re.sub('v\\d+\\s*$', '', s, flags=re.IGNORECASE).strip()
    m_new = _ARXIV_NEW_RE.search(s)
    if m_new:
        val = m_new.group(1)
        return val if _valid_new_style_arxiv_id(val) else None
    m_old = _ARXIV_OLD_RE.search(s)
    if m_old:
        val = m_old.group(1)
        if '/' in val:
            cat, rest = val.split('/', 1)
            val = f'{cat.lower()}/{rest}'
        return val if _valid_old_style_arxiv_id(val) else None
    return None

def _extract_arxiv_id_from_text(reference: str) -> Optional[str]:
    if not reference:
        return None
    m = re.search('arxiv\\.org/(?:abs|pdf)/([^?\\s#]+)', reference, flags=re.IGNORECASE)
    if m:
        tail = re.sub('\\.pdf$', '', m.group(1), flags=re.IGNORECASE)
        arx = _normalize_arxiv_id(tail)
        if arx:
            return arx
    m_new = _ARXIV_NEW_RE.search(reference)
    if m_new:
        arx = _normalize_arxiv_id(m_new.group(1))
        if arx:
            return arx
    m_old = _ARXIV_OLD_RE.search(reference)
    if m_old:
        arx = _normalize_arxiv_id(m_old.group(1))
        if arx:
            return arx
    m2 = re.search('arxiv\\s*:\\s*([0-9]{4}\\.[0-9]{4,5}(?:v\\d+)?)', reference, flags=re.IGNORECASE)
    if m2:
        return _normalize_arxiv_id(m2.group(1))
    m3 = re.search('arxiv\\s*:\\s*([a-z][a-z0-9\\-]*(?:\\.[a-z0-9\\-]+)?/\\d{7}(?:v\\d+)?)', reference, flags=re.IGNORECASE)
    if m3:
        return _normalize_arxiv_id(m3.group(1))
    return None

def _extract_doi_from_text(reference: str) -> Optional[str]:
    if not reference:
        return None
    mlink = re.search('(?:https?://)?(?:dx\\.)?doi\\.org/(10\\.\\d{4,9}/[^ \\t\\r\\n<>\\"\']+)', reference, flags=re.IGNORECASE)
    if mlink:
        return _canonicalize_doi(mlink.group(1))
    mpfx = re.search('\\bdoi\\s*:\\s*(10\\.\\d{4,9}/[-._;()/:A-Z0-9]+)', reference, flags=re.IGNORECASE)
    if mpfx:
        return _canonicalize_doi(mpfx.group(1))
    m = _DOI_RE.search(reference)
    if not m:
        return None
    return _canonicalize_doi(m.group(1))

def _clean_token(s: str) -> str:
    s = (s or '').lower()
    s = re.sub('[^a-z0-9]+', ' ', s)
    return _normalize_whitespace(s)

def _seq_ratio(a: str, b: str) -> float:
    a2 = _clean_token(a)
    b2 = _clean_token(b)
    a_toks = set(a2.split())
    b_toks = set(b2.split())
    if not a_toks or not b_toks:
        return 0.0
    inter = len(a_toks & b_toks)
    union = len(a_toks | b_toks)
    j = inter / union if union else 0.0
    if a2 and (a2 in b2 or b2 in a2):
        j = max(j, 0.85)
    char_ratio = SequenceMatcher(None, a2, b2).ratio() if a2 and b2 else 0.0
    return max(j, char_ratio)

def _heuristic_title_from_reference(reference: str) -> Optional[str]:
    ref = _normalize_whitespace(reference)
    quote_patterns = ['“([^”]{10,300})”', '\\"([^\\"]{10,300})\\"', '‘([^’]{10,300})’', "'([^']{10,300})'"]
    cands: List[str] = []
    for pat in quote_patterns:
        for m in re.finditer(pat, ref):
            s = _normalize_whitespace(m.group(1))
            if re.search('[A-Za-z]', s) and (not re.search('\\b10\\.\\d{4,9}/', s)):
                cands.append(s)
    if cands:
        cands.sort(key=len, reverse=True)
        return _clean_title_text(cands[0].strip(' .;,')) or None
    parts = [p.strip() for p in re.split('\\.\\s+', ref) if p.strip()]
    if len(parts) >= 3:
        cand = parts[1]
        if len(cand.split()) >= 6 and (not re.search('\\b10\\.\\d{4,9}/', cand)):
            return _clean_title_text(cand.strip(' .;,')[:500]) or None
    return None

def _title_snippet(title: str, max_words: int=14) -> str:
    t = _clean_title_text(title)
    toks = [tt for tt in re.split('\\s+', (t or '').strip()) if tt]
    return ' '.join(toks[:max_words])

def _best_title_candidate(titles: List[Optional[str]]) -> Optional[str]:
    uniq: Dict[str, str] = {}
    for t in titles:
        if not t or not str(t).strip():
            continue
        t2 = _clean_title_text(_normalize_whitespace(str(t)).strip())
        if not t2:
            continue
        key = _clean_token(t2)
        if not key:
            continue
        if key not in uniq or len(t2) > len(uniq[key]):
            uniq[key] = t2
    if not uniq:
        return None
    ranked = sorted(uniq.values(), key=lambda x: (min(len(x), 300), len(x.split())), reverse=True)
    return ranked[0][:500]

def _norm_numlike(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    return re.sub('[^0-9A-Za-z]+', '', ss)

def _is_proxy_author_token(tok: Optional[str]) -> bool:
    if not tok:
        return False
    t = tok.strip()
    if not t:
        return False
    if _ROMAN_RE.match(t):
        return True
    if t.lower() in {'collaboration', 'results', 'team', 'consortium', 'group', 'working', 'committee'}:
        return True
    if re.fullmatch('[A-Z]{2,6}', t):
        return True
    if re.fullmatch('[A-Z]{1,4}', t):
        return True
    return False

def _best_effort_surname(name: str) -> Optional[str]:
    if not name:
        return None
    s = _strip_trailing_punct(_normalize_whitespace(name))
    if not s:
        return None
    s = re.sub('^(the|a|an)\\s+', '', s, flags=re.IGNORECASE)
    toks = s.split()
    if not toks:
        return None
    last = toks[-1]
    if _is_proxy_author_token(last) or 'collaboration' in s.lower():
        for cand in toks[:-1]:
            if cand.isalpha() and (not _is_proxy_author_token(cand)):
                return cand
        return None
    if re.fullmatch('[A-Z]\\.?', last):
        return None
    return last

@dataclass
class RefMeta:
    raw: str
    year: Optional[int] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    page: Optional[str] = None
    first_author: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    title: Optional[str] = None
    author_is_proxy: bool = False
    collaboration: Optional[str] = None

def _parse_reference_metadata(reference: str) -> RefMeta:
    ref = _normalize_whitespace(reference)
    meta = RefMeta(raw=ref)
    years = re.findall('(?:19|20)\\d{2}', ref)
    if years:
        try:
            meta.year = int(years[-1])
        except ValueError:
            meta.year = None
    meta.title = _heuristic_title_from_reference(ref)
    ref_no_year = ref
    if meta.year is not None:
        ref_no_year = re.sub('\\(\\s*%d\\s*\\)\\.?\\s*$' % meta.year, '', ref_no_year).strip()
        ref_no_year = re.sub('\\b%d\\b\\.?\\s*$' % meta.year, '', ref_no_year).strip()
    m_jvp = re.search('\\b(?P<journal>[A-Za-z0-9&\\.\\- ]+?)\\s+(?P<vol>\\d{1,5})\\s*,\\s*(?P<page>[A-Za-z]?\\d{1,8}[A-Za-z]?)\\b', ref_no_year)
    if not m_jvp:
        m_jvp = re.search('\\b(?P<journal>[A-Za-z0-9&\\.\\- ]+?)\\s+(?P<vol>\\d{1,5})\\s+(?P<page>[A-Za-z]?\\d{1,8}[A-Za-z]?)\\b', ref_no_year)
    if not m_jvp:
        m_jvp = re.search('\\b(?P<journal>[A-Za-z0-9&\\.\\- ]+?)\\s+(?P<vol>\\d{1,5})\\s*:\\s*(?P<page>[A-Za-z]?\\d{1,8}[A-Za-z]?)\\b', ref_no_year)
    author_block = ref_no_year
    if m_jvp:
        meta.journal = _normalize_whitespace(m_jvp.group('journal')).strip(', ')
        meta.volume = m_jvp.group('vol')
        meta.page = m_jvp.group('page')
        author_block = ref_no_year[:m_jvp.start()].strip().rstrip(',')
    ab = author_block
    ab = ab.replace(' et al.', '').replace(' et al', '')
    ab = _strip_trailing_punct(ab)
    parts: List[str] = []
    for chunk in [c.strip() for c in ab.split(',') if c.strip()]:
        if ' and ' in chunk:
            parts.extend([x.strip() for x in chunk.split(' and ') if x.strip()])
        elif chunk.lower().startswith('and '):
            parts.append(chunk[4:].strip())
        else:
            parts.append(chunk)
    meta.authors = parts
    first_author_seg = parts[0] if parts else None
    if first_author_seg:
        seg_low = first_author_seg.lower()
        if 'collaboration' in seg_low or 'consortium' in seg_low or seg_low.endswith(' collaboration'):
            meta.collaboration = _strip_trailing_punct(first_author_seg)
            meta.author_is_proxy = True
        surname = _best_effort_surname(first_author_seg)
        last_tok = first_author_seg.split()[-1] if first_author_seg.split() else ''
        if _is_proxy_author_token(last_tok) or 'collaboration' in seg_low:
            meta.author_is_proxy = True
        if surname is None:
            meta.author_is_proxy = True
        meta.first_author = surname
    return meta

def _build_search_query(meta: RefMeta) -> str:
    if meta.title:
        bits: List[str] = []
        if meta.collaboration:
            bits.append(meta.collaboration)
        elif meta.first_author and (not meta.author_is_proxy):
            bits.append(meta.first_author)
        bits.append(meta.title)
        if meta.year:
            bits.append(str(meta.year))
        return _normalize_whitespace(' '.join(bits))
    bits: List[str] = []
    if meta.collaboration:
        bits.append(meta.collaboration)
    elif meta.first_author and (not meta.author_is_proxy):
        bits.append(meta.first_author)
    if meta.journal:
        canon = _canonical_journal_name(meta.journal)
        bits.append(meta.journal)
        if canon and canon != meta.journal:
            bits.append(canon)
    if meta.volume:
        bits.append(meta.volume)
    if meta.page:
        bits.append(meta.page)
    if meta.year:
        bits.append(str(meta.year))
    if not bits:
        bits.append(meta.raw)
    return _normalize_whitespace(' '.join(bits))
_DEFAULT_TIMEOUT = 12
_USER_AGENT = f'arxiv-id-resolver/1.6 (mailto:{os.getenv('ARXIV_RESOLVER_MAILTO', 'unknown@example.com')})'
_SESSION = requests.Session()
_HOST_MIN_INTERVAL = {'export.arxiv.org': 3.0, 'api.crossref.org': 0.1, 'api.openalex.org': 0.1, 'api.semanticscholar.org': 1.0 if os.getenv('SEMANTIC_SCHOLAR_API_KEY', '').strip() else 0.1, 'inspirehep.net': 0.1, 'api.tavily.com': 0.2}
_LAST_HOST_TIME: Dict[str, float] = {}
_THROTTLE_LOCK = _threading.Lock()
_HOST_PENALTY_STATE: Dict[str, Tuple[float, float]] = {}
_HOST_PENALTY_LOCK = _threading.Lock()
_PENALTY_CAP = 30.0
_PENALTY_DECAY_S = 120.0

def _get_host_penalty(host: str) -> float:
    with _HOST_PENALTY_LOCK:
        state = _HOST_PENALTY_STATE.get(host)
        if not state:
            return 0.0
        penalty, last_429 = state
        if time.time() - last_429 > _PENALTY_DECAY_S:
            _HOST_PENALTY_STATE.pop(host, None)
            return 0.0
        return penalty

def _record_429(host: str, base_iv: float) -> None:
    with _HOST_PENALTY_LOCK:
        penalty, _ = _HOST_PENALTY_STATE.get(host, (0.0, 0.0))
        new_penalty = min(_PENALTY_CAP, max(2.0 * base_iv, penalty * 2.0))
        _HOST_PENALTY_STATE[host] = (new_penalty, time.time())

def _record_host_success(host: str) -> None:
    with _HOST_PENALTY_LOCK:
        _HOST_PENALTY_STATE.pop(host, None)

def _sleep_backoff(attempt: int, base: float=0.6) -> None:
    wait = 2 ** attempt * base + random.random() * 0.25
    time.sleep(wait)

def _throttle(url: str) -> None:
    try:
        host = re.sub('^https?://', '', url).split('/', 1)[0].lower()
    except Exception:
        return
    base_iv = _HOST_MIN_INTERVAL.get(host)
    if not base_iv:
        return
    effective_iv = base_iv + _get_host_penalty(host)
    with _THROTTLE_LOCK:
        now = time.time()
        last = _LAST_HOST_TIME.get(host, 0.0)
        delta = now - last
        wait = effective_iv - delta if delta < effective_iv else 0.0
        _LAST_HOST_TIME[host] = now + wait
    if wait > 0:
        time.sleep(wait)

def _http_request(method: str, url: str, *, params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, str]]=None, timeout: int=_DEFAULT_TIMEOUT, max_retries: int=2, json_body: Optional[Dict[str, Any]]=None) -> Optional[requests.Response]:
    h = {'User-Agent': _USER_AGENT}
    if headers:
        h.update(headers)
    for attempt in range(max_retries + 1):
        try:
            _throttle(url)
            r = _SESSION.request(method, url, params=params, headers=h, timeout=timeout, json=json_body)
            if r.status_code == 429:
                try:
                    host_429 = re.sub('^https?://', '', url).split('/', 1)[0].lower()
                    base_iv_429 = _HOST_MIN_INTERVAL.get(host_429, 0.0) or 0.0
                    if base_iv_429:
                        _record_429(host_429, base_iv_429)
                except Exception:
                    base_iv_429 = 0.0
                ra = r.headers.get('Retry-After')
                wait: float
                if ra:
                    try:
                        wait = float(ra)
                    except ValueError:
                        wait = 2 ** attempt + random.random()
                else:
                    wait = 2 ** attempt + random.random()
                if base_iv_429:
                    effective_429 = base_iv_429 + _get_host_penalty(host_429)
                    if wait < effective_429:
                        wait = effective_429
                _warn(f'HTTP 429 rate-limited. Backing off {wait:.1f}s: {url}')
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                if attempt < max_retries:
                    _warn(f'HTTP {r.status_code} transient. Retrying: {url}')
                    _sleep_backoff(attempt)
                    continue
            if 400 <= r.status_code < 500 and r.status_code != 429:
                _warn(f'HTTP {r.status_code} (non-retryable): {url}')
                return r
            r.raise_for_status()
            try:
                host_ok = re.sub('^https?://', '', url).split('/', 1)[0].lower()
                _record_host_success(host_ok)
            except Exception:
                pass
            return r
        except Exception as e:
            if attempt < max_retries:
                _warn(f'HTTP error (retrying): {e}')
                _sleep_backoff(attempt)
            else:
                _warn(f'HTTP error (giving up): {e}')
    return None

def _http_get_json(url: str, params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, str]]=None, timeout: int=_DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    r = _http_request('GET', url, params=params, headers=headers, timeout=timeout)
    if not r:
        return None
    try:
        return r.json()
    except Exception:
        _warn(f'Non-JSON response from: {url}')
        return None

def _http_post_json(url: str, body: Dict[str, Any], headers: Optional[Dict[str, str]]=None, timeout: int=_DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    r = _http_request('POST', url, headers=headers, timeout=timeout, json_body=body)
    if not r:
        return None
    try:
        return r.json()
    except Exception:
        _warn(f'Non-JSON response from: {url}')
        return None

def _http_get_text(url: str, params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str, str]]=None, timeout: int=_DEFAULT_TIMEOUT) -> Optional[str]:
    r = _http_request('GET', url, params=params, headers=headers, timeout=timeout)
    if not r:
        return None
    return r.text

def _crossref_work_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    doi = _canonicalize_doi(doi) or doi
    url = 'https://api.crossref.org/works/' + quote(doi, safe='')
    return _http_get_json(url, timeout=15)

def _crossref_title_from_doi(doi: str) -> Optional[str]:
    data = _crossref_work_by_doi(doi)
    if not data or 'message' not in data:
        return None
    msg = data.get('message') or {}
    t = msg.get('title')
    if isinstance(t, list) and t:
        title = _clean_title_text(str(t[0]).strip())
        return title or None
    return None

def _crossref_year(item: Dict[str, Any]) -> Optional[int]:
    issued = item.get('issued', {}).get('date-parts')
    if issued and isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
        try:
            return int(issued[0][0])
        except Exception:
            return None
    return None

def _crossref_author_surnames(item: Dict[str, Any]) -> List[str]:
    out = []
    for a in item.get('author') or []:
        fam = a.get('family')
        if fam:
            out.append(str(fam).lower())
    return out

def _crossref_author_blob(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    for a in item.get('author') or []:
        nm = a.get('name') or ''
        fam = a.get('family') or ''
        giv = a.get('given') or ''
        s = ' '.join([str(x) for x in (nm, giv, fam) if x]).strip()
        if s:
            parts.append(s)
    return ' '.join(parts).lower()

def _score_crossref_item(meta: RefMeta, item: Dict[str, Any]) -> float:
    score = 0.0
    y = _crossref_year(item)
    if meta.year and y:
        if meta.year == y:
            score += 2.0
        elif abs(meta.year - y) <= 1:
            score += 0.5
        else:
            score -= 2.0
    if meta.journal:
        ct = ''
        container = item.get('container-title')
        if isinstance(container, list) and container:
            ct = container[0]
        elif isinstance(container, str):
            ct = container
        if ct:
            score += 1.0 * _seq_ratio(meta.journal, ct)
    mv = str(meta.volume).strip() if meta.volume else None
    iv = str(item.get('issue')).strip() if item.get('issue') is not None else None
    vv = str(item.get('volume')).strip() if item.get('volume') is not None else None
    if mv and (vv or iv):
        if mv == vv or mv == iv:
            score += 2.0
        else:
            score -= 1.0
    mp = str(meta.page).strip() if meta.page else None
    page = str(item.get('page') or '').strip()
    artnum = str(item.get('article-number') or '').strip()
    if mp and (page or artnum):
        ok = False
        for cp in [page, artnum]:
            if not cp:
                continue
            if cp == mp:
                ok = True
            if cp.lstrip('0') == mp.lstrip('0') and cp.lstrip('0'):
                ok = True
            if cp.startswith(mp + '-') or cp.endswith('-' + mp):
                ok = True
        if ok:
            score += 2.0
        else:
            return -999.0
    if meta.first_author and (not meta.author_is_proxy):
        surnames = _crossref_author_surnames(item)
        if surnames:
            if meta.first_author.lower() in surnames:
                score += 2.0
            else:
                score -= 2.0
    if meta.author_is_proxy and meta.collaboration:
        blob = _crossref_author_blob(item)
        toks = [t for t in _clean_token(meta.collaboration).split() if t]
        hits = sum((1 for t in toks if t in blob))
        if hits >= 2:
            score += 1.5
        else:
            score -= 0.5
    return score

def _crossref_reference_to_doi(reference: str, meta: RefMeta) -> Tuple[Optional[str], Optional[str]]:
    _stage('Tool: Crossref (reference → DOI)')
    url = 'https://api.crossref.org/works'

    def _build_params(include_container: bool, include_year_filter: bool) -> Dict[str, Any]:
        p: Dict[str, Any] = {'rows': 10, 'select': 'DOI,title,author,issued,container-title,volume,issue,page,article-number', 'query.bibliographic': reference}
        if meta.author_is_proxy and meta.collaboration:
            p['query.author'] = meta.collaboration
        elif meta.first_author and (not meta.author_is_proxy):
            p['query.author'] = meta.first_author
        if include_container and meta.journal:
            p['query.container-title'] = _canonical_journal_name(meta.journal) or meta.journal
        if include_year_filter and meta.year:
            p['filter'] = f'from-pub-date:{meta.year - 1}-01-01,until-pub-date:{meta.year + 1}-12-31'
        return p

    def _fetch(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = _http_get_json(url, params=params)
        if not data or 'message' not in data:
            return []
        return data.get('message', {}).get('items', []) or []
    items = _fetch(_build_params(include_container=True, include_year_filter=True))
    if not items:
        _info('Crossref: retrying without container-title filter.')
        items = _fetch(_build_params(include_container=False, include_year_filter=True))
    if not items:
        _info('Crossref: retrying without container-title or year filter.')
        items = _fetch(_build_params(include_container=False, include_year_filter=False))
    if not items:
        _warn('Crossref returned no candidates after 3 attempts.')
        return (None, None)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in items:
        if not it.get('DOI'):
            continue
        scored.append((_score_crossref_item(meta, it), it))
    if not scored:
        _warn('Crossref returned no DOI-bearing candidates.')
        return (None, None)
    scored.sort(key=lambda x: x[0], reverse=True)
    _info('Crossref candidates (top scores):')
    for i, (sc, it) in enumerate(scored[:3], start=1):
        doi_i = it.get('DOI')
        t = (it.get('title') or [''])[0] if isinstance(it.get('title'), list) and it.get('title') else it.get('title') or ''
        _info(f'  {i}) score={sc:.2f} doi={doi_i} title={_strip_trailing_punct(_clean_title_text(str(t)))[:120]}')
    best_score, best_item = scored[0]
    if best_score < 4.0:
        _warn(f'Crossref candidates not confident enough (best score={best_score:.2f}); ignoring.')
        return (None, None)
    best_doi = _canonicalize_doi(best_item.get('DOI')) if isinstance(best_item.get('DOI'), str) else None
    title = None
    t = best_item.get('title')
    if isinstance(t, list) and t:
        title = _clean_title_text(str(t[0]).strip()) or None
    _ok(f'Crossref → DOI: {best_doi}')
    return (best_doi, title)

def _parse_arxiv_atom_entries(atom_xml: str) -> List[Dict[str, Any]]:
    if not atom_xml:
        return []
    try:
        root = ET.fromstring(atom_xml)
    except Exception:
        return []
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    entries: List[Dict[str, Any]] = []
    for entry in root.findall('atom:entry', ns):
        id_el = entry.find('atom:id', ns)
        title_el = entry.find('atom:title', ns)
        if id_el is None or not id_el.text:
            continue
        m = re.search('arxiv\\.org/abs/(.+)$', id_el.text.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        arx = _normalize_arxiv_id(m.group(1))
        if not arx:
            continue
        title = ''
        if title_el is not None and title_el.text:
            title = _clean_title_text(_normalize_whitespace(title_el.text))
        authors: List[str] = []
        for ael in entry.findall('atom:author', ns):
            nel = ael.find('atom:name', ns)
            if nel is not None and nel.text:
                authors.append(_normalize_whitespace(nel.text))
        jr = ''
        jr_el = entry.find('arxiv:journal_ref', ns)
        if jr_el is not None and jr_el.text:
            jr = _normalize_whitespace(jr_el.text)
        doi = ''
        doi_el = entry.find('arxiv:doi', ns)
        if doi_el is not None and doi_el.text:
            doi = _normalize_whitespace(doi_el.text)
        entries.append({'arxiv_id': arx, 'title': title, 'authors': authors, 'journal_ref': jr, 'doi': doi})
    return entries

def _arxiv_api_query(search_query: str, max_results: int=8) -> List[Dict[str, Any]]:
    base = 'https://export.arxiv.org/api/query'
    params = {'search_query': search_query, 'start': 0, 'max_results': max_results, 'sortBy': 'relevance', 'sortOrder': 'descending'}
    txt = _http_get_text(base, params=params, timeout=15)
    return _parse_arxiv_atom_entries(txt or '')
_ARXIV_ID_METADATA_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}
_ARXIV_ID_METADATA_LOCK = _threading.Lock()

def _arxiv_api_id_list(arxiv_id: str) -> Optional[Dict[str, Any]]:
    normalised = _normalize_arxiv_id(arxiv_id) or arxiv_id
    with _ARXIV_ID_METADATA_LOCK:
        if normalised in _ARXIV_ID_METADATA_CACHE:
            return _ARXIV_ID_METADATA_CACHE[normalised]
    base = 'https://export.arxiv.org/api/query'
    params = {'id_list': arxiv_id}
    txt = _http_get_text(base, params=params, timeout=15)
    ents = _parse_arxiv_atom_entries(txt or '')
    result = ents[0] if ents else None
    if result is not None:
        with _ARXIV_ID_METADATA_LOCK:
            _ARXIV_ID_METADATA_CACHE[normalised] = result
    return result

def _arxiv_from_doi(doi: str) -> List[str]:
    doi = _canonicalize_doi(doi) or doi
    _info(f'arXiv API query: doi:"{doi}"')
    ents = _arxiv_api_query(f'doi:"{doi}"', max_results=5)
    return [e['arxiv_id'] for e in ents if e.get('arxiv_id')]

def _arxiv_from_title(title: str, first_author: Optional[str]) -> List[str]:
    if not title:
        return []
    t0 = _clean_title_text(title).replace('"', '').strip()
    if not t0:
        return []
    snip = _title_snippet(t0, 14)
    out: List[str] = []
    seen: set = set()
    if first_author:
        q = f'au:"{first_author}" AND ti:"{snip}"'
        _info(f'arXiv API query: {q}')
        for e in _arxiv_api_query(q, max_results=10):
            arx = e.get('arxiv_id')
            if arx and arx not in seen:
                out.append(arx)
                seen.add(arx)
    q2 = f'ti:"{snip}"'
    _info(f'arXiv API query: {q2}')
    for e in _arxiv_api_query(q2, max_results=10):
        arx = e.get('arxiv_id')
        if arx and arx not in seen:
            out.append(arx)
            seen.add(arx)
    return out

def _journal_ref_variants(meta: RefMeta) -> List[str]:
    j = meta.journal or ''
    if not j:
        return []
    j2 = j.strip()
    CANONICAL_ALIASES = {'Physical Review Letters': ['PRL', 'Phys. Rev. Lett.', 'Phys Rev Lett', 'Phys.Rev.Lett.', 'Physical Review Letters', 'Physical Review Letter'], 'Physical Review D': ['PRD', 'Phys. Rev. D', 'Phys Rev D', 'Phys.Rev.D', 'Physical Review D', 'Physical Review D: Particles, Fields, Gravitation and Cosmology'], 'Physical Review A': ['PRA', 'Phys. Rev. A', 'Phys Rev A', 'Phys.Rev.A', 'Physical Review A'], 'Physical Review B': ['PRB', 'Phys. Rev. B', 'Phys Rev B', 'Phys.Rev.B', 'Physical Review B'], 'Physical Review C': ['PRC', 'Phys. Rev. C', 'Phys Rev C', 'Phys.Rev.C', 'Physical Review C'], 'Physical Review E': ['PRE', 'Phys. Rev. E', 'Phys Rev E', 'Phys.Rev.E', 'Physical Review E'], 'Physical Review X': ['PRX', 'Phys. Rev. X', 'Phys Rev X', 'Phys.Rev.X', 'Physical Review X'], 'Physical Review Research': ['PRR', 'Phys. Rev. Research', 'Phys Rev Research', 'Physical Review Research'], 'Physical Review Applied': ['Phys. Rev. Applied', 'Phys Rev Applied', 'Physical Review Applied', 'PR Applied'], 'Physical Review Fluids': ['Phys. Rev. Fluids', 'Phys Rev Fluids', 'Physical Review Fluids', 'PR Fluids'], 'Physical Review Materials': ['Phys. Rev. Materials', 'Phys Rev Materials', 'Physical Review Materials', 'PR Materials'], 'Reviews of Modern Physics': ['RMP', 'Rev. Mod. Phys.', 'Rev Mod Phys', 'Rev.Mod.Phys.', 'Reviews of Modern Physics'], 'Journal of High Energy Physics': ['JHEP', 'J. High Energy Phys.', 'J High Energy Phys', 'Journal of High Energy Physics', 'J. High Energy Physics'], 'The European Physical Journal C': ['EPJC', 'Eur. Phys. J. C', 'Eur Phys J C', 'E.P.J. C', 'European Physical Journal C', 'The European Physical Journal C'], 'The European Physical Journal A': ['EPJA', 'Eur. Phys. J. A', 'Eur Phys J A', 'European Physical Journal A', 'The European Physical Journal A'], 'The European Physical Journal B': ['EPJB', 'Eur. Phys. J. B', 'Eur Phys J B', 'European Physical Journal B', 'The European Physical Journal B'], 'The European Physical Journal D': ['EPJD', 'Eur. Phys. J. D', 'Eur Phys J D', 'European Physical Journal D', 'The European Physical Journal D'], 'Physics Letters B': ['PLB', 'Phys. Lett. B', 'Phys Lett B', 'Phys.Lett.B', 'Physics Letters B'], 'Nuclear Physics A': ['Nucl. Phys. A', 'Nucl Phys A', 'Nucl.Phys.A', 'Nuclear Physics A', 'NPA'], 'Nuclear Physics B': ['Nucl. Phys. B', 'Nucl Phys B', 'Nucl.Phys.B', 'Nuclear Physics B', 'NPB'], 'Classical and Quantum Gravity': ['CQG', 'Class. Quantum Grav.', 'Class Quantum Grav', 'Classical and Quantum Gravity', 'Classical & Quantum Gravity'], 'Journal of Statistical Mechanics: Theory and Experiment': ['JSTAT', 'J. Stat. Mech.', 'J Stat Mech', 'Journal of Statistical Mechanics: Theory and Experiment'], 'Journal of Physics: Condensed Matter': ['JPCM', 'J. Phys.: Condens. Matter', 'J Phys Condens Matter', 'Journal of Physics: Condensed Matter', 'J. Phys. Condens. Matter'], 'Journal of Physics A: Mathematical and Theoretical': ['J. Phys. A', 'J Phys A', 'Journal of Physics A', 'J. Phys. A: Math. Theor.', 'J. Phys. A: Mathematical and Theoretical', 'JPA'], 'Journal of Applied Physics': ['J. Appl. Phys.', 'J Appl Phys', 'J.Appl.Phys.', 'Journal of Applied Physics', 'JAP'], 'Reports on Progress in Physics': ['Rep. Prog. Phys.', 'Rep Prog Phys', 'Reports on Progress in Physics', 'RPP'], 'Progress of Theoretical and Experimental Physics': ['Prog. Theor. Exp. Phys.', 'Prog Theor Exp Phys', 'Progress of Theoretical and Experimental Physics', 'PTEP'], 'Astronomy and Astrophysics': ['A&A', 'Astron. Astrophys.', 'Astron Astrophys', 'Astron. & Astrophys.', 'Astronomy & Astrophysics', 'Astronomy and Astrophysics'], 'The Astrophysical Journal': ['ApJ', 'Astrophys. J.', 'Astrophys J', 'Astrophys. J', 'The Astrophysical Journal', 'Astrophysical Journal'], 'The Astrophysical Journal Letters': ['ApJL', 'Astrophys. J. Lett.', 'Astrophys J Lett', 'ApJ Letters', 'The Astrophysical Journal Letters', 'Astrophysical Journal Letters'], 'The Astrophysical Journal Supplement Series': ['ApJS', 'Astrophys. J. Suppl.', 'Astrophys J Suppl', 'ApJ Suppl.', 'The Astrophysical Journal Supplement Series', 'Astrophysical Journal Supplement', 'Astrophysical Journal Supplement Series', 'Astrophys. J. Suppl. Ser.', 'Astrophys J Suppl Ser'], 'The Astronomical Journal': ['AJ', 'Astron. J.', 'Astron J', 'The Astronomical Journal', 'Astronomical Journal'], 'Monthly Notices of the Royal Astronomical Society': ['MNRAS', 'Mon. Not. R. Astron. Soc.', 'Mon Not R Astron Soc', 'Monthly Notices of the Royal Astronomical Society'], 'Publications of the Astronomical Society of the Pacific': ['PASP', 'Publ. Astron. Soc. Pac.', 'Publ Astron Soc Pac', 'Publications of the Astronomical Society of the Pacific'], 'Astronomy and Astrophysics Supplement Series': ['A&AS', 'Astron. Astrophys. Suppl.', 'Astron Astrophys Suppl', 'Astronomy & Astrophysics Supplement Series', 'Astronomy and Astrophysics Supplement Series'], 'Annual Review of Astronomy and Astrophysics': ['ARA&A', 'Annu. Rev. Astron. Astrophys.', 'Annu Rev Astron Astrophys', 'Annual Review of Astronomy and Astrophysics'], 'Nature Astronomy': ['Nat. Astron.', 'Nat Astron', 'Nature Astronomy'], 'Astroparticle Physics': ['Astropart. Phys.', 'Astropart Phys', 'Astroparticle Physics'], 'Journal of Cosmology and Astroparticle Physics': ['JCAP', 'J. Cosmology Astropart. Phys.', 'J Cosmology Astropart Phys', 'Journal of Cosmology and Astroparticle Physics'], 'Nature': ['Nature'], 'Science': ['Science'], 'Proceedings of the National Academy of Sciences': ['PNAS', 'Proc. Natl. Acad. Sci. U.S.A.', 'Proc Natl Acad Sci U S A', 'Proceedings of the National Academy of Sciences', 'Proceedings of the National Academy of Sciences of the United States of America'], 'PLOS ONE': ['PLOS ONE', 'PloS One', 'PLoS One'], 'Journal of the American Chemical Society': ['JACS', 'J. Am. Chem. Soc.', 'J Am Chem Soc', 'Journal of the American Chemical Society'], 'Angewandte Chemie International Edition': ['Angew. Chem. Int. Ed.', 'Angew Chem Int Ed', 'Angewandte Chemie International Edition', 'Angew. Chemie Int. Ed.', 'Angewandte Chemie Int. Ed.'], 'Chemical Reviews': ['Chem. Rev.', 'Chem Rev', 'Chemical Reviews'], 'Nature Chemistry': ['Nat. Chem.', 'Nat Chem', 'Nature Chemistry'], 'The Journal of Chemical Physics': ['J. Chem. Phys.', 'J Chem Phys', 'The Journal of Chemical Physics', 'JCP'], 'Physical Chemistry Chemical Physics': ['Phys. Chem. Chem. Phys.', 'Phys Chem Chem Phys', 'Physical Chemistry Chemical Physics', 'PCCP'], 'ACS Nano': ['ACS Nano', 'ACSNano'], 'Nano Letters': ['Nano Lett.', 'Nano Lett', 'Nano Letters'], 'Advanced Materials': ['Adv. Mater.', 'Adv Mater', 'Advanced Materials'], 'IEEE Transactions on Pattern Analysis and Machine Intelligence': ['IEEE Trans. Pattern Anal. Mach. Intell.', 'IEEE Trans Pattern Anal Mach Intell', 'IEEE TPAMI', 'TPAMI', 'IEEE Transactions on Pattern Analysis and Machine Intelligence'], 'IEEE Transactions on Neural Networks and Learning Systems': ['IEEE Trans. Neural Netw. Learn. Syst.', 'IEEE Trans Neural Netw Learn Syst', 'TNNLS', 'IEEE Transactions on Neural Networks and Learning Systems'], 'IEEE Transactions on Automatic Control': ['IEEE Trans. Autom. Control', 'IEEE Trans Autom Control', 'IEEE TAC', 'IEEE Transactions on Automatic Control'], 'Proceedings of the IEEE': ['Proc. IEEE', 'Proc IEEE', 'Proceedings of the IEEE'], 'Journal of Fluid Mechanics': ['J. Fluid Mech.', 'J Fluid Mech', 'Journal of Fluid Mechanics', 'JFM'], 'AIAA Journal': ['AIAA J.', 'AIAA Journal', 'AIAA J'], 'Journal of Dynamic Systems, Measurement, and Control': ['J. Dyn. Syst. Meas. Control', 'J Dyn Syst Meas Control', 'Journal of Dynamic Systems, Measurement, and Control'], 'Advances in Neural Information Processing Systems': ['NeurIPS', 'NIPS', 'Advances in Neural Information Processing Systems', 'Neural Information Processing Systems', 'Proc. NeurIPS', 'Proceedings of NeurIPS'], 'International Conference on Machine Learning': ['ICML', 'International Conference on Machine Learning', 'Proc. ICML', 'Proceedings of ICML'], 'International Conference on Learning Representations': ['ICLR', 'International Conference on Learning Representations', 'Proc. ICLR', 'Proceedings of ICLR'], 'International Conference on Artificial Intelligence and Statistics': ['AISTATS', 'International Conference on Artificial Intelligence and Statistics', 'Proc. AISTATS', 'Proceedings of AISTATS'], 'Proceedings of Machine Learning Research': ['PMLR', 'Proceedings of Machine Learning Research', 'Proc. PMLR', 'Proc PMLR'], 'Journal of Machine Learning Research': ['JMLR', 'J. Mach. Learn. Res.', 'J Mach Learn Res', 'Journal of Machine Learning Research'], 'Machine Learning': ['Mach. Learn.', 'Mach Learn', 'Machine Learning', 'MLJ'], 'Artificial Intelligence': ['Artif. Intell.', 'Artif Intell', 'Artificial Intelligence', 'AIJ'], 'Transactions on Machine Learning Research': ['Trans. Mach. Learn. Res.', 'Transactions on Machine Learning Research', 'TMLR'], 'Conference on Computer Vision and Pattern Recognition': ['CVPR', 'IEEE/CVF Conf. Comput. Vis. Pattern Recognit.', 'IEEE CVPR', 'Conference on Computer Vision and Pattern Recognition'], 'International Conference on Computer Vision': ['ICCV', 'International Conference on Computer Vision', 'Proc. ICCV', 'Proceedings of ICCV'], 'European Conference on Computer Vision': ['ECCV', 'European Conference on Computer Vision', 'Proc. ECCV', 'Proceedings of ECCV'], 'Annual Meeting of the Association for Computational Linguistics': ['ACL', 'Annual Meeting of the Association for Computational Linguistics', 'Proc. ACL', 'Proceedings of ACL'], 'Empirical Methods in Natural Language Processing': ['EMNLP', 'Empirical Methods in Natural Language Processing', 'Proc. EMNLP', 'Proceedings of EMNLP'], 'Cell': ['Cell'], 'Nature Biotechnology': ['Nat. Biotechnol.', 'Nat Biotechnol', 'Nature Biotechnology'], 'Nature Genetics': ['Nat. Genet.', 'Nat Genet', 'Nature Genetics'], 'Nature Methods': ['Nat. Methods', 'Nat Methods', 'Nature Methods'], 'Genome Research': ['Genome Res.', 'Genome Res', 'Genome Research'], 'Nucleic Acids Research': ['Nucleic Acids Res.', 'Nucleic Acids Res', 'Nucleic Acids Research', 'NAR'], 'Bioinformatics': ['Bioinformatics'], 'eLife': ['eLife'], 'The New England Journal of Medicine': ['NEJM', 'N. Engl. J. Med.', 'N Engl J Med', 'The New England Journal of Medicine', 'New England Journal of Medicine'], 'The Lancet': ['Lancet', 'The Lancet'], 'Journal of the American Medical Association': ['JAMA', 'J. Am. Med. Assoc.', 'J Am Med Assoc', 'Journal of the American Medical Association'], 'The BMJ': ['BMJ', 'Br. Med. J.', 'Br Med J', 'British Medical Journal', 'The BMJ'], 'Annals of Internal Medicine': ['Ann. Intern. Med.', 'Ann Intern Med', 'Annals of Internal Medicine'], 'Circulation': ['Circulation'], 'Journal of the American College of Cardiology': ['JACC', 'J. Am. Coll. Cardiol.', 'J Am Coll Cardiol', 'Journal of the American College of Cardiology'], 'Nature Medicine': ['Nat. Med.', 'Nat Med', 'Nature Medicine'], 'Nature Immunology': ['Nat. Immunol.', 'Nat Immunol', 'Nature Immunology'], 'Clinical Infectious Diseases': ['Clin. Infect. Dis.', 'Clin Infect Dis', 'Clinical Infectious Diseases'], 'Emerging Infectious Diseases': ['Emerg. Infect. Dis.', 'Emerg Infect Dis', 'Emerging Infectious Diseases'], 'BMC Biology': ['BMC Biol.', 'BMC Biol', 'BMC Biology'], 'BMC Medicine': ['BMC Med.', 'BMC Med', 'BMC Medicine'], 'Annals of Mathematics': ['Ann. of Math.', 'Ann of Math', 'Annals of Mathematics', 'AnnMath'], 'Inventiones Mathematicae': ['Invent. Math.', 'Invent Math', 'Inventiones Mathematicae', 'InventMath'], 'Journal of the American Mathematical Society': ['J. Amer. Math. Soc.', 'J Amer Math Soc', 'Journal of the American Mathematical Society', 'JAMS'], 'Communications on Pure and Applied Mathematics': ['Comm. Pure Appl. Math.', 'Comm Pure Appl Math', 'Communications on Pure and Applied Mathematics', 'CPAM'], 'Probability Theory and Related Fields': ['Probab. Theory Relat. Fields', 'Probab Theory Relat Fields', 'Probability Theory and Related Fields', 'PTRF'], 'The Annals of Statistics': ['Ann. Statist.', 'Ann Statist', 'The Annals of Statistics', 'Annals of Statistics', 'AoS'], 'Biometrika': ['Biometrika'], 'Journal of the Royal Statistical Society: Series B': ['J. R. Stat. Soc. Ser. B', 'J R Stat Soc Series B', 'Journal of the Royal Statistical Society: Series B', 'JRSSB'], 'Geophysical Research Letters': ['Geophys. Res. Lett.', 'Geophys Res Lett', 'Geophysical Research Letters', 'GRL'], 'Journal of Geophysical Research': ['J. Geophys. Res.', 'J Geophys Res', 'Journal of Geophysical Research', 'JGR'], 'Atmospheric Chemistry and Physics': ['Atmos. Chem. Phys.', 'Atmos Chem Phys', 'Atmospheric Chemistry and Physics', 'ACP'], 'Journal of Open Source Software': ['JOSS', 'J. Open Source Softw.', 'J Open Source Softw', 'Journal of Open Source Software'], 'Scientific Data': ['Sci. Data', 'Sci Data', 'Scientific Data', 'SciData']}
    import re
    from typing import Dict, List
    _WS_RE = re.compile('\\s+')
    _PUNCT_RE = re.compile('[^\\w\\s&/+\\-\\.]')

    def _norm_key(s: str) -> str:
        s = (s or '').strip().lower()
        s = s.replace('′', "'").replace('“', '"').replace('”', '"').replace('’', "'")
        s = _PUNCT_RE.sub(' ', s)
        s = _WS_RE.sub(' ', s).strip()
        return s

    def _variants_for_label(label: str) -> List[str]:
        base = (label or '').strip()
        if not base:
            return []
        vars_: List[str] = [base]
        vars_.append(_WS_RE.sub(' ', base).strip())
        vars_.append(base.replace('–', '-').replace('—', '-'))
        vars_.append(base.replace('&', 'and'))
        vars_.append(base.replace(' and ', ' & '))
        vars_.append(re.sub('\\.(?=\\s|$)', '', base))
        vars_.append(re.sub('^\\s*the\\s+', '', base, flags=re.IGNORECASE))
        m = re.match('^\\s*Journal\\s+of\\s+(.+)$', base, flags=re.IGNORECASE)
        if m:
            rest = m.group(1).strip()
            vars_.append(f'J. {rest}')
            vars_.append(f'J {rest}')
        m2 = re.match('^\\s*J\\.?\\s+(.+)$', base, flags=re.IGNORECASE)
        if m2:
            rest = m2.group(1).strip()
            if len(rest.split()) >= 2:
                vars_.append(f'Journal of {rest}')
        out: List[str] = []
        seen = set()
        for v in vars_:
            v = v.strip()
            if not v:
                continue
            if v not in seen:
                out.append(v)
                seen.add(v)
        return out

    def build_aliases(canonical_aliases: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for canonical, labels in (canonical_aliases or {}).items():
            all_labels: List[str] = []
            for x in [canonical, *(labels or [])]:
                if x and x not in all_labels:
                    all_labels.append(x)
            key_candidates: List[str] = []
            for v in _variants_for_label(canonical):
                key_candidates.append(v)
                key_candidates.append(_norm_key(v))
            for lab in labels or []:
                for v in _variants_for_label(lab):
                    key_candidates.append(v)
                    key_candidates.append(_norm_key(v))
            seen_keys = set()
            keys: List[str] = []
            for k in key_candidates:
                k = (k or '').strip()
                if not k:
                    continue
                if k not in seen_keys:
                    keys.append(k)
                    seen_keys.add(k)
            for k in keys:
                existing = out.get(k)
                if existing is None:
                    out[k] = list(all_labels)
                else:
                    for lab in all_labels:
                        if lab not in existing:
                            existing.append(lab)
        return out

    def journal_alias_variants(meta_journal: str, aliases: Dict[str, List[str]]) -> List[str]:
        j = (meta_journal or '').strip()
        if not j:
            return []
        cands = []
        cands.extend(aliases.get(j, []))
        cands.extend(aliases.get(_norm_key(j), []))
        for v in _variants_for_label(j):
            cands.extend(aliases.get(v, []))
            cands.extend(aliases.get(_norm_key(v), []))
        out: List[str] = []
        seen = set()
        for x in cands:
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out
    aliases = build_aliases(CANONICAL_ALIASES)
    out = aliases.get(j2, [j2])
    seen, uniq = (set(), [])
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def _journal_variants_for(journal: Optional[str]) -> List[str]:
    if not journal:
        return []
    dummy = RefMeta(raw='', journal=journal)
    return _journal_ref_variants(dummy)

def _canonical_journal_name(journal: Optional[str]) -> Optional[str]:
    variants = _journal_variants_for(journal)
    return variants[0] if variants else journal

def _arxiv_from_journal_ref(meta: RefMeta) -> List[str]:
    if not meta.journal or not meta.page or (not meta.year):
        return []
    jvars = _journal_ref_variants(meta)
    page = str(meta.page).strip()
    jcap_aliases = {'JCAP', 'J. Cosmology Astropart. Phys.', 'J. Cosmol. Astropart. Phys.', 'Journal of Cosmology and Astroparticle Physics'}
    is_jcap = (meta.journal or '') in jcap_aliases
    queries: List[str] = []
    if is_jcap:
        for j in jvars:
            base = f'jr:"{j}" AND jr:"{meta.year}" AND jr:"{page}"'
            if meta.collaboration:
                base = f'all:"{meta.collaboration}" AND ' + base
            elif meta.first_author and (not meta.author_is_proxy):
                base = f'au:"{meta.first_author}" AND ' + base
            queries.append(base)
    else:
        if not meta.volume:
            return []
        vol = str(meta.volume).strip()
        for j in jvars:
            base = f'jr:"{j}" AND jr:"{vol}" AND jr:"{page}" AND jr:"{meta.year}"'
            if meta.first_author and (not meta.author_is_proxy):
                base = f'au:"{meta.first_author}" AND ' + base
            queries.append(base)
    out: List[str] = []
    seen: set = set()
    for q in queries:
        _info(f'arXiv API query (journal_ref): {q}')
        for e in _arxiv_api_query(q, max_results=8):
            arx = e.get('arxiv_id')
            if arx and arx not in seen:
                out.append(arx)
                seen.add(arx)
        if out:
            return out
    return out

def _author_surname_list(authors: List[str]) -> List[str]:
    out: List[str] = []
    for nm in authors or []:
        s = _best_effort_surname(nm)
        if s:
            out.append(s.lower())
    return out

def _journal_ref_score(meta: RefMeta, journal_ref: str) -> float:
    if not journal_ref:
        return 0.0
    jr = journal_ref
    score = 0.0
    if meta.journal:
        score += 0.8 * _seq_ratio(meta.journal, jr)
    mv = _norm_numlike(meta.volume)
    mp = _norm_numlike(meta.page)
    if mv and re.search(f'\\b{re.escape(mv)}\\b', jr):
        score += 0.9
    if mp and re.search(f'\\b{re.escape(mp)}\\b', jr):
        score += 0.9
    else:
        mp2 = re.sub('^0+', '', mp or '')
        if mp2 and re.search(f'\\b{re.escape(mp2)}\\b', jr):
            score += 0.6
    if meta.year and re.search(f'\\b{meta.year}\\b', jr):
        score += 0.5
    return score

def _has_strong_contradiction(meta: RefMeta, ent: Dict[str, Any], expected_title: Optional[str]) -> bool:
    arx = ent.get('arxiv_id') or ''
    ay, _ = _arxiv_id_year_month(str(arx))
    if meta.year and ay and (ay > meta.year):
        return True
    cand_title = (ent.get('title') or '').strip()
    if expected_title and cand_title:
        sim = _seq_ratio(expected_title, cand_title)
        if sim < 0.45:
            return True
    jr = (ent.get('journal_ref') or '').strip()
    if jr and meta.journal and (meta.volume or meta.page):
        jsc = _journal_ref_score(meta, jr)
        if jsc < 0.55:
            return True
    return False

def _validation_threshold(meta: RefMeta, expected_title: Optional[str]) -> float:
    if expected_title:
        return 2.2
    evidence = 0
    if meta.year:
        evidence += 1
    if meta.journal:
        evidence += 1
    if meta.volume or meta.page:
        evidence += 1
    if meta.first_author and (not meta.author_is_proxy) or meta.collaboration:
        evidence += 1
    if evidence >= 3:
        thr = 1.8
    elif evidence == 2:
        thr = 1.4
    else:
        thr = 1.1
    if meta.author_is_proxy:
        thr = max(0.95, thr - 0.15)
    return thr

def _validate_arxiv_candidate(arxiv_id: str, meta: RefMeta, expected_title: Optional[str]=None, expected_author: Optional[str]=None) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
    arxiv_id = _normalize_arxiv_id(arxiv_id) or ''
    if not arxiv_id:
        return (False, 0.0, None)
    ay, _am = _arxiv_id_year_month(arxiv_id)
    if meta.year and ay and (ay > meta.year):
        return (False, -999.0, None)
    ent = _arxiv_api_id_list(arxiv_id)
    if not ent:
        return (False, 0.0, None)
    score = 0.0
    thr = _validation_threshold(meta, expected_title)
    if expected_title and ent.get('title'):
        sim = _seq_ratio(expected_title, ent['title'])
        score += 2.5 * sim
        if sim < 0.55:
            return (False, score, ent)
    a_exp = expected_author or meta.first_author
    if a_exp and (not meta.author_is_proxy):
        surn = _author_surname_list(ent.get('authors') or [])
        if a_exp.lower() in surn:
            score += 0.9
        else:
            score -= 0.25 if not expected_title else 0.6
    score += _journal_ref_score(meta, ent.get('journal_ref') or '')
    ok = score >= thr
    return (ok, score, ent)

def _accept_candidate(arxiv_id: Optional[str], meta: RefMeta, expected_title: Optional[str]=None, expected_author: Optional[str]=None, label: str='candidate') -> Optional[str]:
    if not arxiv_id:
        _trace(label, 'miss')
        return None
    arx = _normalize_arxiv_id(arxiv_id)
    if not arx:
        _trace(label, 'reject', candidate=str(arxiv_id), detail='normalize_failed')
        return None
    exp_t = _clean_title_text(expected_title) if expected_title else None
    ok, score, ent = _validate_arxiv_candidate(arx, meta, expected_title=exp_t, expected_author=expected_author)
    inspire_lenient = 'INSPIRE' in label.upper()
    if inspire_lenient and ent is not None and (not exp_t) and (not meta.title):
        if not _has_strong_contradiction(meta, ent, expected_title=None):
            _ok(f'Accepted {label}: {arx} (INSPIRE-lenient; validation score={score:.2f})')
            _trace(label, 'ok', candidate=arx, score=score, detail='inspire_lenient_accept')
            _set_resolved_via(label)
            return arx
    if ok:
        _ok(f'Accepted {label}: {arx} (validation score={score:.2f})')
        _trace(label, 'ok', candidate=arx, score=score)
        _set_resolved_via(label)
        return arx
    why = 'validation_failed'
    got_title = (ent or {}).get('title') or ''
    if got_title:
        why += f'; cand_title={got_title[:60]!r}'
    _warn(f'Rejected {label}: {arx} (validation score={score:.2f})')
    _trace(label, 'reject', candidate=arx, score=score, detail=why)
    return None

def _doi_consistency_score(doi: str, meta: RefMeta) -> float:
    data = _crossref_work_by_doi(doi)
    msg = (data or {}).get('message') or {}
    if not msg:
        return -999.0
    return _score_crossref_item(meta, msg)

def _doi_is_consistent_enough(doi: str, meta: RefMeta, *, threshold: float=3.5) -> bool:
    sc = _doi_consistency_score(doi, meta)
    _info(f'DOI consistency score vs reference: {sc:.2f} (threshold={threshold:.2f})')
    return sc >= threshold

def _openalex_mailto() -> Optional[str]:
    return os.getenv('ARXIV_RESOLVER_MAILTO', '').strip() or None

def _extract_arxiv_from_any_urlish(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        m = re.search('arxiv\\.org/(?:abs|pdf)/([^?\\s#]+)', obj, flags=re.IGNORECASE)
        if m:
            tail = re.sub('\\.pdf$', '', m.group(1), flags=re.IGNORECASE)
            return _normalize_arxiv_id(tail)
        return None
    if isinstance(obj, list):
        for x in obj:
            v = _extract_arxiv_from_any_urlish(x)
            if v:
                return v
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            got = _extract_arxiv_from_any_urlish(v)
            if got:
                return got
    return None

def _openalex_work_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    doi = _canonicalize_doi(doi) or doi
    work_id = quote('https://doi.org/' + doi, safe='')
    url = 'https://api.openalex.org/works/' + work_id
    params: Dict[str, Any] = {}
    m = _openalex_mailto()
    if m:
        params['mailto'] = m
    return _http_get_json(url, params=params, timeout=15)

def _openalex_search(query: str, per_page: int=5) -> Optional[List[Dict[str, Any]]]:
    url = 'https://api.openalex.org/works'
    params: Dict[str, Any] = {'search': query, 'per-page': per_page}
    m = _openalex_mailto()
    if m:
        params['mailto'] = m
    data = _http_get_json(url, params=params, timeout=15)
    return (data or {}).get('results') or None

def _openalex_doi_to_arxiv_title(doi: str) -> Tuple[Optional[str], Optional[str]]:
    _stage('Tool: OpenAlex (DOI → arXiv)')
    w = _openalex_work_by_doi(doi)
    if not w:
        _warn('OpenAlex did not return a record for DOI.')
        return (None, None)
    title = _clean_title_text(w.get('title') or '') or None
    arx = _extract_arxiv_from_any_urlish(w)
    if arx:
        _ok(f'OpenAlex → arXiv: {arx}')
    else:
        _warn('OpenAlex did not return arXiv.')
    return (arx, title)

def _openalex_biblio(w: Dict[str, Any]) -> Dict[str, Any]:
    return w.get('biblio') or {}

def _score_openalex_result(meta: RefMeta, w: Dict[str, Any]) -> float:
    score = 0.0
    if meta.year and w.get('publication_year'):
        if meta.year == w['publication_year']:
            score += 2.0
        elif abs(meta.year - w['publication_year']) <= 1:
            score += 0.5
        else:
            score -= 2.0
    if meta.journal:
        hv = w.get('host_venue') or {}
        dn = hv.get('display_name') or ''
        if dn:
            sim = _seq_ratio(meta.journal, dn)
            score += 0.9 * sim
            if sim < 0.2:
                score -= 1.5
    b = _openalex_biblio(w)
    if meta.volume and b.get('volume'):
        score += 1.6 if str(meta.volume).strip() == str(b['volume']).strip() else -1.0
    if meta.page and b.get('first_page'):
        fp = str(b.get('first_page') or '').strip()
        if fp:
            score += 1.6 if _norm_numlike(fp) == _norm_numlike(meta.page) else -1.0
    if meta.first_author and (not meta.author_is_proxy):
        auths = w.get('authorships') or []
        fams = []
        for a in auths:
            author = a.get('author') or {}
            disp = author.get('display_name') or ''
            fams.append((_best_effort_surname(disp) or '').lower())
        if any((meta.first_author.lower() == f for f in fams if f)):
            score += 1.2
        else:
            score -= 0.6
    if meta.collaboration:
        blob = ' '.join([(a.get('author') or {}).get('display_name') or '' for a in w.get('authorships') or []]).lower()
        toks = [t for t in _clean_token(meta.collaboration).split() if t]
        hits = sum((1 for t in toks if t in blob))
        if hits >= 2:
            score += 0.8
    if meta.title and w.get('title'):
        score += 1.2 * _seq_ratio(meta.title, w['title'])
    return score

def _openalex_reference_search_to_doi_arxiv_title(reference: str, meta: RefMeta) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    _stage('Tool: OpenAlex (reference search → metadata)')
    q = _build_search_query(meta)
    _info(f'OpenAlex search query: {q}')
    results = _openalex_search(q, per_page=5) or []
    if not results:
        _warn('OpenAlex search returned no results.')
        return (None, None, None)
    scored = sorted(((_score_openalex_result(meta, r), r) for r in results), key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    _info(f'OpenAlex best score: {best_score:.2f}')
    _SCORE_FLOOR = 1.5
    if best_score < _SCORE_FLOOR:
        _warn('OpenAlex best match not confident; ignoring OpenAlex search.')
        return (None, None, None)
    title = _clean_title_text(best.get('title') or '') or None
    doi = None
    doi_url = best.get('doi')
    if isinstance(doi_url, str) and doi_url.lower().startswith('https://doi.org/'):
        doi = _canonicalize_doi(doi_url.split('doi.org/', 1)[-1].strip())
    arx: Optional[str] = None
    for sc, r in scored:
        if sc < _SCORE_FLOOR:
            break
        cand = _extract_arxiv_from_any_urlish(r)
        if cand:
            arx = cand
            break
    if arx:
        _ok(f'OpenAlex search → arXiv: {arx}')
    return (doi, arx, title)

def _s2_headers() -> Dict[str, str]:
    h = {'User-Agent': _USER_AGENT}
    key = os.getenv('SEMANTIC_SCHOLAR_API_KEY', '').strip()
    if key:
        h['x-api-key'] = key
    return h
_S2_FIELDS = 'title,year,venue,externalIds,url,authors,publicationVenue,journal'

def _s2_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    doi = _canonicalize_doi(doi) or doi
    url = f'https://api.semanticscholar.org/graph/v1/paper/DOI:{quote_plus(doi)}'
    params = {'fields': _S2_FIELDS}
    return _http_get_json(url, params=params, headers=_s2_headers(), timeout=15)

def _s2_search(query: str, limit: int=5) -> Optional[List[Dict[str, Any]]]:
    url = 'https://api.semanticscholar.org/graph/v1/paper/search'
    params = {'query': query, 'limit': limit, 'fields': _S2_FIELDS}
    data = _http_get_json(url, params=params, headers=_s2_headers(), timeout=15)
    return (data or {}).get('data') or None

def _extract_arxiv_from_s2(record: Dict[str, Any]) -> Optional[str]:
    ext = record.get('externalIds') or {}
    arx = ext.get('ArXiv') or ext.get('arXiv') or ext.get('arxiv')
    if arx:
        return _normalize_arxiv_id(str(arx))
    return _extract_arxiv_from_any_urlish(record)

def _extract_doi_from_s2(record: Dict[str, Any]) -> Optional[str]:
    ext = record.get('externalIds') or {}
    doi = ext.get('DOI') or ext.get('doi')
    return _canonicalize_doi(doi) if isinstance(doi, str) else None

def _s2_doi_to_arxiv_title(doi: str) -> Tuple[Optional[str], Optional[str]]:
    _stage('Tool: Semantic Scholar (DOI → arXiv)')
    rec = _s2_paper_by_doi(doi)
    if not rec:
        _warn('Semantic Scholar did not return a record for DOI.')
        return (None, None)
    title = _clean_title_text(rec.get('title') or '') or None
    arx = _extract_arxiv_from_s2(rec)
    if arx:
        _ok(f'Semantic Scholar → arXiv: {arx}')
    else:
        _warn('Semantic Scholar did not return arXiv.')
    return (arx, title)

def _score_s2_result(meta: RefMeta, r: Dict[str, Any]) -> float:
    score = 0.0
    if meta.year and r.get('year'):
        if meta.year == r['year']:
            score += 2.0
        elif abs(meta.year - r['year']) <= 1:
            score += 0.5
        else:
            score -= 2.0
    if meta.first_author and (not meta.author_is_proxy):
        fams = []
        for a in r.get('authors') or []:
            name = a.get('name') or ''
            fams.append((_best_effort_surname(name) or '').lower())
        if any((meta.first_author.lower() == f for f in fams if f)):
            score += 1.2
        else:
            score -= 0.6
    if meta.journal:
        venue = r.get('venue') or ''
        pv = (r.get('publicationVenue') or {}).get('name') or ''
        jn = (r.get('journal') or {}).get('name') or ''
        sim = max(_seq_ratio(meta.journal, venue), _seq_ratio(meta.journal, pv), _seq_ratio(meta.journal, jn))
        score += 0.9 * sim
        if sim < 0.2:
            score -= 1.5
    j = r.get('journal') or {}
    if meta.volume and j.get('volume'):
        score += 1.4 if str(meta.volume).strip() == str(j['volume']).strip() else -0.8
    if meta.page and j.get('pages'):
        pages = str(j['pages'])
        if _norm_numlike(meta.page) and _norm_numlike(meta.page) in _norm_numlike(pages or ''):
            score += 1.4
        else:
            score -= 0.8
    if meta.collaboration:
        blob = ' '.join([a.get('name') or '' for a in r.get('authors') or []]).lower()
        toks = [t for t in _clean_token(meta.collaboration).split() if t]
        hits = sum((1 for t in toks if t in blob))
        if hits >= 2:
            score += 0.8
    if meta.title and r.get('title'):
        score += 1.2 * _seq_ratio(meta.title, r['title'])
    return score

def _s2_reference_search_to_doi_arxiv_title(reference: str, meta: RefMeta) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    _stage('Tool: Semantic Scholar (reference search → metadata)')
    q = _build_search_query(meta)
    _info(f'Semantic Scholar search query: {q}')
    results = _s2_search(q, limit=5) or []
    if not results:
        _warn('Semantic Scholar search returned no results.')
        return (None, None, None)
    scored = sorted(((_score_s2_result(meta, r), r) for r in results), key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    _info(f'Semantic Scholar best score: {best_score:.2f}')
    _SCORE_FLOOR = 1.5
    if best_score < _SCORE_FLOOR:
        _warn('Semantic Scholar best match not confident; ignoring S2 search.')
        return (None, None, None)
    title = _clean_title_text(best.get('title') or '') or None
    doi = _extract_doi_from_s2(best)
    arx: Optional[str] = None
    for sc, r in scored:
        if sc < _SCORE_FLOOR:
            break
        cand = _extract_arxiv_from_s2(r)
        if cand:
            arx = cand
            break
    if arx:
        _ok(f'Semantic Scholar search → arXiv: {arx}')
    return (doi, arx, title)

def _inspire_search(q: str, size: int=3) -> Optional[Dict[str, Any]]:
    url = 'https://inspirehep.net/api/literature'
    params = {'q': q, 'size': size}
    return _http_get_json(url, params=params, timeout=20)

def _extract_arxiv_from_inspire(hit: Dict[str, Any]) -> Optional[str]:
    md = (hit or {}).get('metadata') or {}
    ae = md.get('arxiv_eprints')
    if isinstance(ae, list):
        for e in ae:
            v = e.get('value')
            arx = _normalize_arxiv_id(v) if v else None
            if arx:
                return arx
    v2 = md.get('arxiv_eprint')
    arx2 = _normalize_arxiv_id(v2) if v2 else None
    if arx2:
        return arx2
    return _extract_arxiv_from_any_urlish(hit)

def _inspire_search_by_doi(doi: str) -> Optional[str]:
    _stage('Tool: INSPIRE (DOI → arXiv)')
    data = _inspire_search(f'doi "{doi}"', size=3)
    if not data:
        _warn('INSPIRE did not return results for DOI.')
        return None
    hits = (data.get('hits') or {}).get('hits') or []
    for h in hits:
        arx = _extract_arxiv_from_inspire(h)
        if arx:
            _ok(f'INSPIRE → arXiv: {arx}')
            return arx
    _warn('INSPIRE did not return arXiv.')
    return None

def _inspire_reference_search_to_arxiv(reference: str, meta: RefMeta) -> Optional[str]:
    _stage('Tool: INSPIRE (reference search → arXiv)')
    journal_variants = _journal_variants_for(meta.journal)[:3] if meta.journal else []
    qs: List[str] = []
    if meta.first_author and meta.journal and meta.volume and meta.page and meta.year and (not meta.author_is_proxy):
        for jv in journal_variants or [meta.journal]:
            qs.append(f'a {meta.first_author} and j "{jv}" and vol {meta.volume} and page {meta.page} and year {meta.year}')
    if meta.collaboration and meta.journal and meta.page and meta.year:
        for jv in journal_variants or [meta.journal]:
            qs.append(f'"{meta.collaboration}" and j "{jv}" and year {meta.year} and page {meta.page}')
    if meta.title and meta.first_author and (not meta.author_is_proxy):
        qs.append(f'a {meta.first_author} and t "{_title_snippet(meta.title, 10)}"')
    if meta.title:
        qs.append(f't "{_title_snippet(meta.title, 12)}"')
    qs.append(reference)
    for q in qs:
        _info(f'INSPIRE query: {q}')
        data = _inspire_search(q, size=3)
        if not data:
            continue
        hits = (data.get('hits') or {}).get('hits') or []
        for h in hits:
            arx = _extract_arxiv_from_inspire(h)
            if arx:
                _ok(f'INSPIRE search → arXiv: {arx}')
                return arx
    _warn('INSPIRE search did not return arXiv.')
    return None

def _doi_to_arxiv(doi: str, meta: RefMeta, title_hint: Optional[str]) -> Optional[str]:
    doi = _canonicalize_doi(doi) or doi.strip()
    _stage('Stage: DOI-based resolution')
    _info(f'DOI: {doi}')
    title = _clean_title_text(title_hint) if title_hint else None
    title = title or _crossref_title_from_doi(doi)
    for arx in _arxiv_from_doi(doi):
        accepted = _accept_candidate(arx, meta, expected_title=title, label='DOI→arXiv (arXiv-doi)')
        if accepted:
            return accepted
    arx_oa, title_oa = _openalex_doi_to_arxiv_title(doi)
    accepted = _accept_candidate(arx_oa, meta, expected_title=title_oa or title, label='DOI→arXiv (OpenAlex)')
    if accepted:
        return accepted
    arx_s2, title_s2 = _s2_doi_to_arxiv_title(doi)
    accepted = _accept_candidate(arx_s2, meta, expected_title=title_s2 or title, label='DOI→arXiv (S2)')
    if accepted:
        return accepted
    arx_i = _inspire_search_by_doi(doi)
    accepted = _accept_candidate(arx_i, meta, expected_title=title, label='DOI→arXiv (INSPIRE)')
    if accepted:
        return accepted
    if title:
        _stage('Stage: title-based resolution (from DOI metadata)')
        for arx_t in _arxiv_from_title(title, first_author=meta.first_author if meta.first_author and (not meta.author_is_proxy) else None):
            accepted = _accept_candidate(arx_t, meta, expected_title=title, label='DOI→arXiv (title)')
            if accepted:
                return accepted
    _warn('DOI → arXiv failed.')
    return None

def _tavily_key() -> Optional[str]:
    return os.getenv('TAVILY_API_KEY', '').strip() or None

def _tavily_search_raw(query: str, search_depth: str='basic', max_results: int=5) -> Optional[Dict[str, Any]]:
    key = _tavily_key()
    if not key:
        _warn('TAVILY_API_KEY is not set; cannot use Tavily fallback.')
        return None
    url = 'https://api.tavily.com/search'
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'User-Agent': _USER_AGENT}
    body = {'query': query, 'search_depth': search_depth, 'include_answer': True, 'include_raw_content': False, 'max_results': max_results, 'include_images': False, 'include_usage': False, 'auto_parameters': False}
    return _http_post_json(url, body=body, headers=headers, timeout=30)

def _tavily_extract_arxiv_ids(resp: Dict[str, Any]) -> List[str]:
    if not resp:
        return []
    texts: List[str] = []
    ans = resp.get('answer')
    if isinstance(ans, str) and ans.strip():
        texts.append(ans)
    for r in resp.get('results') or []:
        for key in ('title', 'content', 'url'):
            v = r.get(key)
            if isinstance(v, str) and v.strip():
                texts.append(v)
    found: List[str] = []
    for t in texts:
        for m in _ARXIV_NEW_RE.finditer(t):
            arx = _normalize_arxiv_id(m.group(1))
            if arx:
                found.append(arx)
        for m in _ARXIV_OLD_RE.finditer(t):
            arx = _normalize_arxiv_id(m.group(1))
            if arx:
                found.append(arx)
        m4 = re.search('arxiv\\.org/(?:abs|pdf)/([^?\\s#]+)', t, flags=re.IGNORECASE)
        if m4:
            tail = re.sub('\\.pdf$', '', m4.group(1), flags=re.IGNORECASE)
            arx = _normalize_arxiv_id(tail)
            if arx:
                found.append(arx)
    out: List[str] = []
    seen = set()
    for a in found:
        if a not in seen:
            out.append(a)
            seen.add(a)
    return out

def _tavily_fallback(reference: str, meta: RefMeta) -> Optional[str]:
    _stage('Tool: Tavily (web search → arXiv)')
    q_core = _build_search_query(meta)
    q = f'arXiv {q_core}'
    _info(f'Tavily query: {q}')
    resp = _tavily_search_raw(q, search_depth='basic', max_results=6)
    if not resp:
        _warn('Tavily did not return a usable response.')
        return None
    ids = _tavily_extract_arxiv_ids(resp)
    if not ids:
        _warn('Tavily response did not contain an arXiv ID.')
        return None
    _info(f'Tavily extracted arXiv IDs: {ids}')
    best = None
    best_score = -1000000000.0
    for arx in ids:
        ay, _ = _arxiv_id_year_month(arx)
        if meta.year and ay and (ay > meta.year):
            _info(f'Rejecting {arx}: arXiv year {ay} > pub year {meta.year}')
            continue
        ok, sc, ent = _validate_arxiv_candidate(arx, meta, expected_title=meta.title)
        if not ok and ent:
            if not _has_strong_contradiction(meta, ent, expected_title=meta.title):
                sc = max(sc, 1.0)
        if sc > best_score:
            best_score = sc
            best = arx
    if best and best_score >= 1.4:
        _ok(f'Tavily → selected best arXiv: {best} (score={best_score:.2f})')
        return best
    _warn('Tavily produced arXiv IDs but none scored confidently.')
    return None

def _doi_to_pmid(doi: str) -> Optional[str]:
    try:
        conv_url = f'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={quote(doi)}&format=json'
        resp = requests.get(conv_url, timeout=10)
        resp.raise_for_status()
        for rec in resp.json().get('records', []):
            pmid = rec.get('pmid')
            if pmid:
                return str(pmid)
    except Exception:
        pass
    return None

def resolve_reference(reference: str) -> Dict[str, Optional[str]]:
    _set_trace_current([])
    _set_resolved_via(None)
    final_id: Optional[str] = None
    best_doi: Optional[str] = None
    best_title: Optional[str] = None
    ref = _normalize_whitespace(reference)
    try:
        _stage('Start: resolve_reference')
        _info(f'Reference: {reference}')
        _trace('resolver', 'info', detail='start')
        _stage('Tool: direct parse (reference → arXiv)')
        arx_direct = _extract_arxiv_id_from_text(ref)
        if arx_direct:
            arx_direct = _normalize_arxiv_id(arx_direct)
            if arx_direct:
                _ok(f'Found arXiv id directly: {arx_direct}')
                _trace('direct_arxiv', 'ok', candidate=arx_direct, detail='no_network_return')
                _set_resolved_via('direct_arxiv')
                final_id = arx_direct
                return {'arxiv_id': final_id, 'doi': None, 'title': None, 'pmid': None}
        _warn('No arXiv id found directly.')
        _trace('direct_arxiv', 'miss')
        meta = _parse_reference_metadata(ref)
        _trace('meta', 'info', detail=f'author={meta.first_author or '-'} proxy={meta.author_is_proxy} collab={('yes' if meta.collaboration else 'no')} year={meta.year or '-'} journal={meta.journal or '-'} vol={meta.volume or '-'} page={meta.page or '-'} title={('yes' if meta.title else 'no')}')
        _stage('Tool: direct parse (reference → DOI)')
        doi_direct = _extract_doi_from_text(ref)
        if doi_direct:
            _ok(f'Found DOI directly: {doi_direct}')
            _trace('direct_doi', 'info', detail=f'doi={doi_direct}')
            best_doi = doi_direct
            arx = _doi_to_arxiv(doi_direct, meta, title_hint=meta.title)
            if arx:
                final_id = arx
                _trace('resolver', 'ok', candidate=final_id, detail='via=doi_pipeline(direct)')
                return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
            _trace('direct_doi', 'reject', detail='doi_pipeline_failed')
        else:
            _warn('No DOI found directly.')
            _trace('direct_doi', 'miss')
        jcap_aliases = {'JCAP', 'J. Cosmology Astropart. Phys.', 'J. Cosmol. Astropart. Phys.', 'Journal of Cosmology and Astroparticle Physics'}
        if meta.journal and meta.page and meta.year and (meta.volume or meta.journal in jcap_aliases):
            _stage('Tool: arXiv API (journal_ref search)')
            for arx_jr in _arxiv_from_journal_ref(meta):
                accepted = _accept_candidate(arx_jr, meta, expected_title=meta.title, expected_author=meta.first_author if not meta.author_is_proxy else None, label='arXiv journal_ref')
                if accepted:
                    final_id = accepted
                    _trace('resolver', 'ok', candidate=final_id, detail='via=arxiv(journal_ref)')
                    return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
        doi_cr, title_cr = _crossref_reference_to_doi(ref, meta)
        if doi_cr:
            _trace('crossref_ref2doi', 'ok', detail=f'doi={doi_cr}')
            best_doi = best_doi or doi_cr
            best_title = best_title or title_cr
            arx = _doi_to_arxiv(doi_cr, meta, title_hint=title_cr or meta.title)
            if arx:
                final_id = arx
                _trace('resolver', 'ok', candidate=final_id, detail='via=doi_pipeline(crossref)')
                return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
            _trace('crossref_ref2doi', 'reject', detail='doi_pipeline_failed')
        else:
            _trace('crossref_ref2doi', 'miss')
        oa_doi, oa_arx, oa_title = _openalex_reference_search_to_doi_arxiv_title(ref, meta)
        _trace('openalex_ref_search', 'info', detail=f'doi={oa_doi or '-'} arxiv={oa_arx or '-'} title={('yes' if oa_title else 'no')}')
        best_doi = best_doi or oa_doi
        best_title = best_title or oa_title
        if oa_arx:
            accepted = _accept_candidate(oa_arx, meta, expected_title=oa_title or meta.title, expected_author=meta.first_author if not meta.author_is_proxy else None, label='OpenAlex ref arXiv')
            if accepted:
                final_id = accepted
                _trace('resolver', 'ok', candidate=final_id, detail='via=openalex(arxiv)')
                return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
        if oa_doi:
            if _doi_is_consistent_enough(oa_doi, meta):
                arx = _doi_to_arxiv(oa_doi, meta, title_hint=oa_title or meta.title)
                if arx:
                    final_id = arx
                    _trace('resolver', 'ok', candidate=final_id, detail='via=openalex(doi_pipeline)')
                    return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
            else:
                _warn(f'Ignoring OpenAlex-derived DOI (inconsistent with reference): {oa_doi}')
        s2_doi, s2_arx, s2_title = _s2_reference_search_to_doi_arxiv_title(ref, meta)
        _trace('s2_ref_search', 'info', detail=f'doi={s2_doi or '-'} arxiv={s2_arx or '-'} title={('yes' if s2_title else 'no')}')
        best_doi = best_doi or s2_doi
        best_title = best_title or s2_title
        if s2_arx:
            accepted = _accept_candidate(s2_arx, meta, expected_title=s2_title or meta.title, expected_author=meta.first_author if not meta.author_is_proxy else None, label='S2 ref arXiv')
            if accepted:
                final_id = accepted
                _trace('resolver', 'ok', candidate=final_id, detail='via=s2(arxiv)')
                return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
        if s2_doi:
            if _doi_is_consistent_enough(s2_doi, meta):
                arx = _doi_to_arxiv(s2_doi, meta, title_hint=s2_title or meta.title)
                if arx:
                    final_id = arx
                    _trace('resolver', 'ok', candidate=final_id, detail='via=s2(doi_pipeline)')
                    return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
            else:
                _warn(f'Ignoring S2-derived DOI (inconsistent with reference): {s2_doi}')
        arx_ins = _inspire_reference_search_to_arxiv(ref, meta)
        if arx_ins:
            accepted = _accept_candidate(arx_ins, meta, expected_title=meta.title, expected_author=meta.first_author if not meta.author_is_proxy else None, label='INSPIRE ref arXiv')
            if accepted:
                final_id = accepted
                _trace('resolver', 'ok', candidate=final_id, detail='via=inspire(ref)')
                return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
        else:
            _trace('inspire_ref_search', 'miss')
        title_best = _best_title_candidate([title_cr, oa_title, s2_title, meta.title])
        if title_best:
            best_title = best_title or title_best
            _stage('Tool: arXiv API (title-only fallback)')
            _trace('arxiv_title_fallback', 'info', detail=_title_snippet(title_best, 10))
            for arx_t in _arxiv_from_title(title_best, first_author=meta.first_author if meta.first_author and (not meta.author_is_proxy) else None):
                accepted = _accept_candidate(arx_t, meta, expected_title=title_best, expected_author=meta.first_author if not meta.author_is_proxy else None, label='arXiv title fallback')
                if accepted:
                    final_id = accepted
                    _trace('resolver', 'ok', candidate=final_id, detail='via=arxiv(title)')
                    return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
        else:
            _warn('No usable title for arXiv title-only fallback.')
            _trace('arxiv_title_fallback', 'miss', detail='no_title')
        arx_tv = _tavily_fallback(ref, meta)
        if arx_tv:
            final_id = arx_tv
            _set_resolved_via('tavily')
            _trace('tavily', 'ok', candidate=final_id)
            _trace('resolver', 'ok', candidate=final_id, detail='via=tavily')
            return {'arxiv_id': final_id, 'doi': best_doi, 'title': best_title, 'pmid': None}
        if best_doi:
            _info(f'No arXiv ID found, but have DOI: {best_doi}')
            _trace('resolver', 'info', detail=f'fallback_doi={best_doi}')
            pmid = _doi_to_pmid(best_doi)
            if pmid:
                _info(f'Resolved DOI to PubMed ID: {pmid}')
            return {'arxiv_id': None, 'doi': best_doi, 'title': best_title, 'pmid': pmid}
        _fail('Failed: could not resolve reference to any identifier.')
        _trace('resolver', 'error', detail='no_id_found')
        return {'arxiv_id': None, 'doi': None, 'title': None, 'pmid': None}
    except Exception as e:
        _fail(f'Unhandled exception in resolver: {e}')
        _trace('resolver', 'error', detail=str(e))
        return {'arxiv_id': None, 'doi': None, 'title': None, 'pmid': None}
    finally:
        _trace_summary(final_id)
        _set_trace_current(None)
        _set_resolved_via(None)

def resolve_arxiv_id_from_reference(reference: str) -> Optional[str]:
    return resolve_reference(reference).get('arxiv_id')
assert 'resolve_arxiv_id_from_reference' in globals(), 'Public API function missing/renamed!'
assert 'resolve_reference' in globals(), 'Public API function missing/renamed!'
