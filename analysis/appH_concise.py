"""Appendix H: concise structured prompt baseline (Table 11 and text numbers).

Table differences are printed from unrounded rates, with the difference of rounded
entries in brackets. Paired tests use injections adjudicated for both configurations
and a paired paper-level bootstrap (10,000 resamples, seed 20260506). Half-widths use
2000 resamples (seed 20260427).

Usage: python analysis/appH_concise.py [astro|ml]   (default: both)
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import CORPORA, CORPUS_NAMES, DATA

MODS = ("citations", "numerics", "mathematics", "general")
VER_RE = re.compile(r"^(v[123])(?:__(.+))?\.json$")

PROVIDERS = ("Claude Opus 4.7", "GPT-5.5", "Gemini 3.1 Pro")
FF, CON, STR = "free-form", "concise structured", "structured"


def cfg(model, prompt):
    return f"{model} ({prompt})"


CC = "Claude Code"
ARMS = {
    ("L1", "anthropic"): cfg(PROVIDERS[0], FF), ("L25", "anthropic"): cfg(PROVIDERS[0], CON),
    ("L2", "anthropic"): cfg(PROVIDERS[0], STR),
    ("L1", "openai"): cfg(PROVIDERS[1], FF), ("L25", "openai"): cfg(PROVIDERS[1], CON),
    ("L2", "openai"): cfg(PROVIDERS[1], STR),
    ("L1", "gemini"): cfg(PROVIDERS[2], FF), ("L25", "gemini"): cfg(PROVIDERS[2], CON),
    ("L2", "gemini"): cfg(PROVIDERS[2], STR),
    ("CL1", ""): cfg(CC, FF), ("CL25", ""): cfg(CC, CON), ("C", ""): cfg(CC, STR),
    ("L3", ""): "Skepthical",
}

_claude, _gpt, _gemini = PROVIDERS
PAIRS = [(cfg(_claude, CON), cfg(_claude, FF)), (cfg(_claude, STR), cfg(_claude, CON)),
         (cfg(_gpt, CON), cfg(_gpt, FF)), (cfg(_gpt, STR), cfg(_gpt, CON)),
         (cfg(_gemini, CON), cfg(_gemini, FF)), (cfg(_gemini, STR), cfg(_gemini, CON)),
         (cfg(CC, CON), cfg(CC, FF)), (cfg(CC, STR), cfg(CC, CON)),
         ("Skepthical", cfg(_gpt, CON)), ("Skepthical", cfg(_claude, CON)),
         ("Skepthical", cfg(_gemini, CON)), (cfg(CC, STR), cfg(_claude, CON)),
         (cfg(CC, CON), cfg(_claude, CON))]
W = max(len(a) for a in ARMS.values())


def load(papers):
    """D[arm][(paper, copy, module)] = 0/1 detected."""
    D = defaultdict(dict)
    for p in papers:
        for f in glob.glob(str(DATA / p / "verifications" / "*" / "v*.json")):
            system = Path(f).parent.name
            m = VER_RE.match(Path(f).name)
            if not m:
                continue
            arm = ARMS.get((system, (m.group(2) or "").lower()))
            if arm is None:
                continue
            for v in json.loads(Path(f).read_text()).get("verdicts", []):
                D[arm][(p, m.group(1), v["module"])] = int(bool(v.get("detected")))
    return D


def rate(D, arm, mod=None):
    xs = [d for (p, c, m), d in D[arm].items() if mod is None or m == mod]
    return (100 * np.mean(xs) if xs else float("nan")), len(xs)


def boot_halfwidth(D, arm, mod=None, n_boot=2000, seed=20260427):
    by = defaultdict(list)
    for (p, c, m), d in D[arm].items():
        if mod is None or m == mod:
            by[p].append(d)
    papers = list(by)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        smp = rng.choice(papers, size=len(papers), replace=True)
        vals = [x for q in smp for x in by[q]]
        if vals:
            bs.append(np.mean(vals))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 50 * (hi - lo)


def paired(D, a, b, mod=None, B=10000, seed=20260506):
    """(difference in pp on injections adjudicated for both, reversal probability, n)."""
    keys = [k for k in D[a] if k in D[b] and (mod is None or k[2] == mod)]
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append((D[a][k], D[b][k]))
    papers = list(by)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        smp = rng.choice(papers, size=len(papers), replace=True)
        xs = [t for q in smp for t in by[q]]
        diffs.append(np.mean([x for x, _ in xs]) - np.mean([y for _, y in xs]))
    diffs = np.array(diffs)
    d = 100 * (np.mean([D[a][k] for k in keys]) - np.mean([D[b][k] for k in keys]))
    p = float(np.mean(diffs <= 0)) if d > 0 else float(np.mean(diffs >= 0)) if d < 0 else 0.5
    return d, p, len(keys)


def fp(papers, arm):
    """(FP rate %, number of flags, {module: (FP, flags)}) on the clean copies."""
    system, tag = next(k for k, v in ARMS.items() if v == arm)
    nf = nfp = 0
    per_mod = defaultdict(lambda: [0, 0])
    for p in papers:
        f = DATA / p / "fp" / f"{system}{'__' + tag if tag else ''}_v0.json"
        if not f.exists():
            continue
        for mod, vs in (json.loads(f.read_text()).get("per_module_verdicts") or {}).items():
            for v in vs:
                nf += 1
                per_mod[mod][1] += 1
                if v.get("verdict") == "FP":
                    nfp += 1
                    per_mod[mod][0] += 1
    return (100 * nfp / nf if nf else float("nan")), nf, dict(per_mod)


def table_diff(R, a, b, col):
    """Difference of two table rates: (unrounded, difference of the rounded entries)."""
    x, y = R[a][col], R[b][col]
    return x - y, round(x, 1) - round(y, 1)


def report(which):
    papers = sorted(CORPORA[which])
    D = load(papers)
    print(f"\n{'=' * 78}\n{CORPUS_NAMES[which]}\n{'=' * 78}")

    print("\nTable 11: detection % (+- half-width, n adjudicated) | citations numerics "
          "mathematics general ; clean-copy FP % (flags) ; FP by error type")
    R, FP = {}, {}
    for arm in ARMS.values():
        if not D.get(arm):
            print(f"  {arm:<{W}} (no data)")
            continue
        tot, n = rate(D, arm)
        R[arm] = {"total": tot, **{m: rate(D, arm, m)[0] for m in MODS}}
        FP[arm] = fp(papers, arm)
        fpr, nf, per_mod = FP[arm]
        print(f"  {arm:<{W}} {tot:5.1f}+-{boot_halfwidth(D, arm):4.1f} (n={n}) | "
              + " ".join(f"{R[arm][m]:5.1f}" for m in MODS)
              + f" ; FP {fpr:5.1f} ({nf}) ; "
              + " ".join(f"{m[:4]} {per_mod.get(m, (0, 0))[0]}/{per_mod.get(m, (0, 0))[1]}" for m in MODS))

    def td(a, b, col="total"):
        x, r = table_diff(R, a, b, col)
        return f"{x:+.2f} (table entries: {r:+.1f})"

    print("\nTable differences quoted in the text (pp)")
    for prov in PROVIDERS:
        print(f"  {prov:<15} concise - free-form    {td(cfg(prov, CON), cfg(prov, FF))}")
    for prov in PROVIDERS:
        print(f"  {prov:<15} structured - concise   {td(cfg(prov, STR), cfg(prov, CON))}")
    print(f"  Claude Code concise - free-form    {td(cfg(CC, CON), cfg(CC, FF))}")
    print(f"  Claude Code structured - free-form {td(cfg(CC, STR), cfg(CC, FF))}")
    print(f"  Claude Code structured - concise   {td(cfg(CC, STR), cfg(CC, CON))}")
    print(f"  Skepthical - concise GPT-5.5       {td('Skepthical', cfg(_gpt, CON))}"
          f"   ({R['Skepthical']['total']:.2f} vs {R[cfg(_gpt, CON)]['total']:.2f} %)")
    print("  Structured - concise, averaged over the three providers, per error type:")
    for col in ("total",) + MODS:
        xs = [table_diff(R, cfg(p, STR), cfg(p, CON), col) for p in PROVIDERS]
        print(f"    {col:<12} {np.mean([x for x, _ in xs]):+.2f} "
              f"(table entries: {np.mean([r for _, r in xs]):+.2f})")
    for name, prompt in (("free-form", FF), ("concise", CON), ("structured", STR)):
        rs = [R[cfg(p, prompt)]["total"] for p in PROVIDERS]
        print(f"  spread between providers, {name:<10} {max(rs) - min(rs):5.2f} pp "
              f"(max {max(rs):.2f}, min {min(rs):.2f})")
    for name, prompt in (("free-form", FF), ("concise", CON), ("structured", STR)):
        rs = [FP[cfg(p, prompt)][0] for p in PROVIDERS]
        print(f"  FP range for single LLMs, {name:<10} {min(rs):.2f}-{max(rs):.2f} %")
    print("  Claude Code FP: " + ", ".join(f"{p} {FP[cfg(CC, p)][0]:.2f} %" for p in (FF, CON, STR)))

    print("\nPaired tests (a - b) on injections adjudicated for both: difference pp "
          "(reversal probability), n")
    for a, b in PAIRS:
        if not D.get(a) or not D.get(b):
            continue
        d, p, n = paired(D, a, b)
        per = " | ".join(f"{m[:3]} {paired(D, a, b, m)[0]:+5.1f} ({paired(D, a, b, m)[1]:.3f})" for m in MODS)
        print(f"  {a:<{W}} - {b:<{W}} total {d:+5.1f} ({p:.3f}) n={n} | {per}")

    print("\nStructured - concise pooled over the three providers (paired, paper-level resampling)")
    D["structured, pooled"] = {(k[0], prov + "_" + k[1], k[2]): v
                               for prov in PROVIDERS for k, v in D[cfg(prov, STR)].items()}
    D["concise, pooled"] = {(k[0], prov + "_" + k[1], k[2]): v
                            for prov in PROVIDERS for k, v in D[cfg(prov, CON)].items()}
    for m in (None,) + MODS:
        d, p, n = paired(D, "structured, pooled", "concise, pooled", m)
        print(f"  {m or 'total':<12} {d:+5.1f} ({p:.3f}) n={n}")


if __name__ == "__main__":
    for which in (sys.argv[1:] or ["astro", "ml"]):
        report(which)
