"""Wolfram Engine, SymPy and pint engines."""
from __future__ import annotations
import shutil
import subprocess
import time
from typing import Dict, Optional
import pint
_WOLFRAM_BINARY: Optional[str] = None

def wolfram_available() -> bool:
    global _WOLFRAM_BINARY
    if _WOLFRAM_BINARY is None:
        for name in ('wolframscript', 'WolframKernel', 'wolfram'):
            path = shutil.which(name)
            if path:
                _WOLFRAM_BINARY = path
                break
        else:
            _WOLFRAM_BINARY = ''
    return bool(_WOLFRAM_BINARY)

def _build_wolfram_classifier_probe(wl_code: str) -> str:
    return 'Module[{$r = (' + wl_code + ')}, Which[ TrueQ[$r], "TRUE", $r === False, "FALSE", PossibleZeroQ[$r] || TrueQ[Simplify[$r == 0]], "ZERO", True, "UNEVAL:" <> ToString[$r, InputForm]]]'

def check_wolfram(wl_code: str, expected: str='True', timeout_s: float=60.0, session: Optional['WolframSession']=None) -> Dict:
    if not wolfram_available():
        return {'tool': 'wolfram', 'status': 'unavailable', 'detail': 'Wolfram Engine not installed (install WolframEngine + `wolframscript`)', 'seconds': 0.0}
    t0 = time.time()
    probe = _build_wolfram_classifier_probe(wl_code)
    if session is not None and session.alive():
        try:
            out = session.eval(probe, timeout_s=timeout_s)
            seconds = round(time.time() - t0, 3)
            return _classify_wolfram_output(out, expected, seconds, source='session')
        except Exception as e:
            fallback_detail = f'session call failed: {type(e).__name__}: {e}  — retrying via subprocess'
            t0 = time.time()
        else:
            fallback_detail = ''
    else:
        fallback_detail = ''
    cmd = [_WOLFRAM_BINARY, '-code', probe]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {'tool': 'wolfram', 'status': 'error', 'detail': f'timeout after {timeout_s:.0f}s', 'seconds': round(time.time() - t0, 3)}
    except Exception as e:
        return {'tool': 'wolfram', 'status': 'error', 'detail': f'{type(e).__name__}: {e}', 'seconds': round(time.time() - t0, 3)}
    seconds = round(time.time() - t0, 3)
    out = (proc.stdout or '').strip().strip('"')
    err = (proc.stderr or '').strip()
    if proc.returncode != 0:
        return {'tool': 'wolfram', 'status': 'error', 'detail': f'exit={proc.returncode}; stderr={err[:400]}', 'seconds': seconds}
    result = _classify_wolfram_output(out, expected, seconds, source='subprocess')
    if fallback_detail:
        result['detail'] = fallback_detail + '  || ' + result['detail']
    return result

def _classify_wolfram_output(out: str, expected: str, seconds: float, source: str) -> Dict:
    exp = str(expected).strip()
    detail = f'wolfram[{source}] classifier → {out}'
    if exp in ('True', 'False'):
        if out == 'TRUE':
            return {'tool': 'wolfram', 'status': 'confirmed' if exp == 'True' else 'refuted', 'detail': detail, 'seconds': seconds}
        if out == 'FALSE':
            return {'tool': 'wolfram', 'status': 'refuted' if exp == 'True' else 'confirmed', 'detail': detail, 'seconds': seconds}
        return {'tool': 'wolfram', 'status': 'inconclusive', 'detail': detail + '  (no boolean reduction)', 'seconds': seconds}
    if exp == '0':
        if out == 'ZERO':
            return {'tool': 'wolfram', 'status': 'confirmed', 'detail': detail, 'seconds': seconds}
        if out == 'TRUE':
            return {'tool': 'wolfram', 'status': 'inconclusive', 'detail': detail + '  (got boolean True, expected 0)', 'seconds': seconds}
        if out.startswith('UNEVAL:'):
            return {'tool': 'wolfram', 'status': 'inconclusive', 'detail': detail, 'seconds': seconds}
        return {'tool': 'wolfram', 'status': 'refuted', 'detail': detail, 'seconds': seconds}
    return {'tool': 'wolfram', 'status': 'inconclusive', 'detail': detail + f'  (no boolean reduction against expected={exp!r})', 'seconds': seconds}

class WolframSession:
    _BEGIN = '::SKEP_WL_BEGIN::'
    _END = '::SKEP_WL_END::'

    def __init__(self, binary: Optional[str]=None, spawn_timeout_s: float=60.0):
        if not wolfram_available():
            raise RuntimeError('wolframscript not on PATH; WolframSession requires it')
        self._binary = binary or _WOLFRAM_BINARY
        self._dead = False
        self._lock = None
        self._child = None
        try:
            import pexpect
        except ImportError as e:
            raise RuntimeError('WolframSession requires the `pexpect` package (pip install pexpect)') from e
        self._spawn(spawn_timeout_s)

    def _spawn(self, spawn_timeout_s: float) -> None:
        import pexpect
        import threading
        self._lock = threading.Lock()
        self._child = pexpect.spawn(self._binary, encoding='utf-8', timeout=spawn_timeout_s, dimensions=(40, 200), echo=False)
        try:
            self._child.setecho(False)
            self._child.waitnoecho(timeout=5)
        except Exception:
            pass
        warm = f'Print["{self._BEGIN}"]; Print["ready"]; Print["{self._END}"]'
        try:
            self._child.sendline(warm)
            self._child.expect_exact(self._END, timeout=spawn_timeout_s)
        except Exception:
            self._dead = True
            self._kill_child()
            raise

    def alive(self) -> bool:
        if self._dead:
            return False
        ch = self._child
        if ch is None:
            return False
        try:
            return bool(ch.isalive())
        except Exception:
            return False

    def eval(self, wl_expr: str, timeout_s: float=60.0) -> str:
        import pexpect
        if not self.alive():
            raise RuntimeError('WolframSession is not alive')
        assert self._lock is not None
        cmd = f'Module[{{$skepout = ({wl_expr})}}, Print["{self._BEGIN}"]; Print[$skepout]; Print["{self._END}"]]'
        with self._lock:
            try:
                self._child.sendline(cmd)
                self._child.expect_exact(self._END, timeout=timeout_s)
                before = self._child.before or ''
            except pexpect.TIMEOUT as e:
                self._dead = True
                self._kill_child()
                raise RuntimeError(f'WolframSession eval timed out after {timeout_s:.0f}s') from e
            except pexpect.EOF as e:
                self._dead = True
                self._kill_child()
                raise RuntimeError('WolframSession kernel exited unexpectedly') from e
            except Exception:
                self._dead = True
                self._kill_child()
                raise
        idx = before.rfind(self._BEGIN)
        if idx < 0:
            self._dead = True
            self._kill_child()
            raise RuntimeError('WolframSession: BEGIN sentinel not found; output was: ' + before[-400:])
        body = before[idx + len(self._BEGIN):]
        return body.strip().strip('"').strip()

    def close(self) -> None:
        self._dead = True
        self._kill_child()

    def _kill_child(self) -> None:
        if self._child is None:
            return
        try:
            self._child.sendline('Quit[]')
        except Exception:
            pass
        try:
            self._child.close(force=True)
        except Exception:
            pass
        self._child = None

    def __enter__(self) -> 'WolframSession':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
_ureg = pint.UnitRegistry()

def check_pint(lhs_units: str, rhs_units: str) -> Dict:
    t0 = time.time()
    try:
        lhs_q = _ureg(lhs_units)
        rhs_q = _ureg(rhs_units)
    except Exception as e:
        return {'tool': 'pint', 'status': 'error', 'detail': f'could not parse ({lhs_units!r} vs {rhs_units!r}): {e}', 'seconds': round(time.time() - t0, 3)}
    seconds = round(time.time() - t0, 3)
    lhs_dim = lhs_q.dimensionality
    rhs_dim = rhs_q.dimensionality
    detail = f'[{lhs_units}] = {lhs_dim}  vs  [{rhs_units}] = {rhs_dim}'
    if lhs_dim == rhs_dim:
        return {'tool': 'pint', 'status': 'confirmed', 'detail': detail, 'seconds': seconds}
    return {'tool': 'pint', 'status': 'refuted', 'detail': detail, 'seconds': seconds}

def _self_test():
    print('--- pint: energy dimensional match ---')
    print(check_pint('joule', 'kilogram * meter**2 / second**2'))
    print('\n--- pint: dimensional absurdity (length vs time) ---')
    print(check_pint('meter', 'second'))
    print('\n--- Wolfram: availability + one probe ---')
    print(f'wolfram_available() = {wolfram_available()}')
    print(check_wolfram('FullSimplify[(a+b)^2 - (a^2 + 2 a b + b^2)] == 0', expected='True'))
    print('\n--- Wolfram: sign-flip refutation on subleading factor ---')
    print(check_wolfram('FullSimplify[(M^alpha) * h^(1/2) - (M^alpha) * h^(-1/2)] === 0', expected='True'))
if __name__ == '__main__':
    _self_test()
