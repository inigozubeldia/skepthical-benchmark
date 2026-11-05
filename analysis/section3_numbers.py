"""Numbers quoted in Section 3.1 (detection rates) and Section 3.2 (false positives), per corpus.

    python analysis/section3_numbers.py [astro|ml]      (default: both corpora)

Twelve systems of Table 2 only. Prints paired bootstrap contrasts, detection by
severity, clean-copy FP rates, Claude Code run metadata and the Section 3.2 summaries.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from collections import defaultdict

import numpy as np

from common import (CORPORA, DATA, MODS, PROVIDER_MODEL, REPO, SYSTEMS, iter_verdicts,
                    load_detections, load_fp)

DERIVED = REPO / "analysis" / "derived"
STRONG = [("L1", "openai"), ("L1", "anthropic"), ("L2", "anthropic"), ("L2", "openai"), ("L2", "gemini"),
          ("L3", ""), ("C", ""), ("CL1", ""), ("B1", "")]
OPEN_WEIGHTS = [("B2", ""), ("B3", "deep")]
SEVERITIES = ("S1", "S2", "S3")


def paired(D, a, b, mods=None, n_boot=2000, seed=20260427):
    keys = [k for k in D[a] if k in D[b] and (mods is None or k[2] in mods)]
    byp = defaultdict(list)
    for k in keys:
        byp[k[0]].append((int(D[a][k]), int(D[b][k])))
    ps = list(byp)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        smp = rng.choice(ps, size=len(ps), replace=True)
        x = [v for p in smp for v in byp[p]]
        deltas.append(np.mean([u for u, _ in x]) - np.mean([w for _, w in x]))
    obs = np.mean([u for p in ps for u, _ in byp[p]]) - np.mean([w for p in ps for _, w in byp[p]])
    return 100 * obs, float(np.mean(np.array(deltas) <= 0)), len(keys)


def fp_rate(pairs, n_boot=2000, seed=20260427):
    if not pairs:
        return None
    by = defaultdict(list)
    for p, v in pairs:
        by[p].append(1 if v == "FP" else 0)
    ps = list(by)
    flat = [x for p in ps for x in by[p]]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        smp = rng.choice(ps, size=len(ps), replace=True)
        vals = [x for p in smp for x in by[p]]
        if vals:
            bs.append(np.mean(vals))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (100 * np.mean(flat), 100 * lo, 100 * hi, sum(flat), len(flat),
            sum(1 for _, v in pairs if v == "UNADJUDICATABLE"))


def load_records(papers):
    """Per-injection records for the twelve systems, severities from v<k>/injections.json."""
    sev = {}
    for p in papers:
        sev[p] = {}
        for copy in ("v1", "v2", "v3"):
            f = DATA / p / copy / "injections.json"
            if f.exists():
                sev[p][copy] = {r.get("module"): r.get("severity") for r in json.loads(f.read_text())}
    records = []
    for system, tag, lab in SYSTEMS:
        for p in papers:
            for copy in ("v1", "v2", "v3"):
                f = DATA / p / "verifications" / system / (f"{copy}__{tag}.json" if tag else f"{copy}.json")
                if not f.exists():
                    continue
                for v in json.loads(f.read_text()).get("verdicts", []):
                    m = v.get("module")
                    if m not in MODS or sev.get(p, {}).get(copy, {}).get(m) not in SEVERITIES:
                        continue
                    records.append((lab, p, m, bool(v.get("detected"))))
    return records


def per_paper_counts(records, papers, lab, module=None):
    out = {p: [0, 0] for p in papers}
    for rl, p, m, d in records:
        if rl == lab and (module is None or m == module):
            out[p][1] += 1
            out[p][0] += d
    return out


def pooled_rate(counts, sample=None):
    sample = list(counts) if sample is None else sample
    det = sum(counts[p][0] for p in sample)
    tot = sum(counts[p][1] for p in sample)
    return det / tot if tot else float("nan")


def prob_le_zero(records, papers, a, b, B=10_000, seed=20260506):
    """{column: Pr(rate(a) - rate(b) <= 0)} under paper-level resampling."""
    rng = random.Random(seed)
    n_p = len(papers)
    out = {}
    for col in ("total",) + MODS:
        mod = None if col == "total" else col
        ca, cb = per_paper_counts(records, papers, a, mod), per_paper_counts(records, papers, b, mod)
        n_le = n_valid = 0
        for _ in range(B):
            sample = [papers[rng.randrange(n_p)] for _ in range(n_p)]
            ra, rb = pooled_rate(ca, sample), pooled_rate(cb, sample)
            if ra != ra or rb != rb:
                continue
            n_valid += 1
            n_le += (ra - rb) <= 0
        out[col] = n_le / n_valid if n_valid else float("nan")
    return out


CONTRASTS = [
    ("Structured vs free-form, Claude Opus 4.7", "Claude Opus 4.7 (structured)", "Claude Opus 4.7 (free-form)"),
    ("Structured vs free-form, GPT-5.5", "GPT-5.5 (structured)", "GPT-5.5 (free-form)"),
    ("Structured vs free-form, Gemini 3.1 Pro", "Gemini 3.1 Pro (structured)", "Gemini 3.1 Pro (free-form)"),
    ("Structured vs free-form, Claude Code", "Claude Code (structured)", "Claude Code (free-form)"),
    ("Claude Code vs single call, free-form", "Claude Code (free-form)", "Claude Opus 4.7 (free-form)"),
    ("Claude Code vs single call, structured", "Claude Code (structured)", "Claude Opus 4.7 (structured)"),
    ("Skepthical vs Claude Code, structured", "Skepthical", "Claude Code (structured)"),
    ("Skepthical vs GPT-5.5, structured", "Skepthical", "GPT-5.5 (structured)"),
]


def contrasts(papers):
    R = load_records(papers)
    for name, a, b in CONTRASTS:
        pab, pba = prob_le_zero(R, papers, a, b), prob_le_zero(R, papers, b, a)
        cells = []
        for col in ("total",) + MODS:
            mod = None if col == "total" else col
            d = 100 * (pooled_rate(per_paper_counts(R, papers, a, mod)) - pooled_rate(per_paper_counts(R, papers, b, mod)))
            p = pab[col] if d > 0 else (pba[col] if d < 0 else 0.5)
            cells.append(f"{col[:4]} {d:+5.1f} p={p:.3f}")
        print(f"  {name:<42} " + " | ".join(cells))


def concise_contrast(papers, a=("L3", ""), b=("L25", "openai"), B=10_000, seed=20260506):
    D = defaultdict(dict)
    for sysd, tag, p, copy, v in iter_verdicts(sorted(papers)):
        if (sysd, tag) in (a, b):
            D[(sysd, tag)][(p, copy, v["module"])] = int(bool(v.get("detected")))
    rate = {s: 100 * np.mean(list(D[s].values())) for s in (a, b)}
    keys = [k for k in D[a] if k in D[b]]
    by = defaultdict(list)
    for k in keys:
        by[k[0]].append((D[a][k], D[b][k]))
    ps = list(by)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        smp = rng.choice(ps, size=len(ps), replace=True)
        xs = [t for q in smp for t in by[q]]
        diffs.append(np.mean([x for x, _ in xs]) - np.mean([y for _, y in xs]))
    diffs = np.array(diffs)
    d = 100 * (np.mean([D[a][k] for k in keys]) - np.mean([D[b][k] for k in keys]))
    p = float(np.mean(diffs <= 0)) if d > 0 else float(np.mean(diffs >= 0)) if d < 0 else 0.5
    return rate[a], rate[b], d, p, len(keys)


def run(cname):
    papers = CORPORA[cname]
    D, inj = load_detections(papers)
    print(f"\n################ {cname} ################")
    print("(1) paired paper bootstrap, Delta = Skepthical - other (pp), Pr(Delta<=0)")
    for other, lab in ((("L2", "openai"), "structured GPT-5.5"), (("L1", "openai"), "free-form GPT-5.5"),
                       (("C", ""), "Claude Code structured"), (("L2", "anthropic"), "structured Opus 4.7")):
        parts = []
        for mods, ml in ((None, "total"), (("citations",), "cit"), (("numerics",), "num"),
                         (("mathematics",), "math"), (("general",), "gen")):
            r = paired(D, ("L3", ""), other, mods)
            parts.append(f"{ml} {r[0]:+.1f} (Pr={r[1]:.3f})")
        print(f"  vs {lab:24s}: " + " | ".join(parts))

    print("(2) detection rate % at S1/S2/S3 per system and error type ('^' strictly increasing)")
    for sysd, tag, label in SYSTEMS:
        if not D[(sysd, tag)]:
            continue
        parts = []
        for m in MODS:
            r = [100 * np.mean([D[(sysd, tag)][k] for k in D[(sysd, tag)] if k[2] == m and inj[k] == s] or [np.nan])
                 for s in SEVERITIES]
            parts.append(f"{m[:3]} {r[0]:.0f}/{r[1]:.0f}/{r[2]:.0f}{'^' if r[0] < r[1] < r[2] else ' '}")
        print(f"  {label:30s} " + "  ".join(parts))
    for m in MODS:
        rates = [100 * np.mean([D[st][k] for st in STRONG if D[st] for k in D[st] if k[2] == m and inj[k] == s])
                 for s in SEVERITIES]
        dips = sum(1 for st in STRONG if D[st]
                   and np.mean([D[st][k] for k in D[st] if k[2] == m and inj[k] == "S2"])
                   < np.mean([D[st][k] for k in D[st] if k[2] == m and inj[k] == "S1"]))
        print(f"  pooled non-floor systems, {m:11s}: S1 {rates[0]:.0f} S2 {rates[1]:.0f} S3 {rates[2]:.0f}"
              f"   S2<S1 in {dips}/{sum(1 for st in STRONG if D[st])} systems")

    print("(3) clean-copy FP rate % [95% CI] (FP/flags; u = unadjudicated in denominator)")
    F = load_fp(papers)
    for sysd, tag, label in SYSTEMS:
        if not F[(sysd, tag)]:
            continue
        cells = []
        for m in MODS + ("all",):
            pairs = F[(sysd, tag)][m] if m != "all" else [x for mm in MODS for x in F[(sysd, tag)][mm]]
            r = fp_rate(pairs)
            cells.append(f"{m[:3]} {r[0]:.1f} [{r[1]:.0f},{r[2]:.0f}] ({r[3]}/{r[4]};u{r[5]})" if r else f"{m[:3]} --")
        print(f"  {label:30s} " + " | ".join(cells))
    for sysd, lab in (("L1", "free-form"), ("L2", "structured")):
        rs = [fp_rate(F[(sysd, prov)]["mathematics"]) for prov in ("anthropic", "openai", "gemini")]
        print(f"  single-LLM mathematics FP, {lab}: " + ", ".join(f"{r[0]:.1f} ({r[3]}/{r[4]})" if r else "--" for r in rs)
              + f"; macro-average {np.mean([r[0] for r in rs if r]):.1f}")

    print("(4) Claude Code run metadata")
    with open(DERIVED / f"claude_code_runs_{cname}.csv") as fh:
        runs = list(csv.DictReader(fh))
    for sysd, name in (("C", "Claude Code (structured)"), ("CL1", "Claude Code (free-form)")):
        turns = [int(r["num_turns"]) for r in runs if r["system"] == sysd and r["num_turns"] != ""]
        wall = defaultdict(list)
        for r in runs:
            if r["system"] == sysd and r["wall_seconds"] != "":
                wall[r["paper"]].append(float(r["wall_seconds"]))
        pp = [statistics.mean(v) for v in wall.values() if v]
        print(f"  {name}: {len(turns)} runs, turns median {statistics.median(turns)}, max {max(turns)}, "
              f">=15 turns {sum(t >= 15 for t in turns)}/{len(turns)}; per-paper mean wall-clock spread {max(pp)/min(pp):.1f}x")

    print("(5) Section 3.1 comparisons: Delta = rate(a) - rate(b) in pp, p = Pr(sign reverses), "
          "paired paper bootstrap B=10,000 seed 20260506")
    contrasts(papers)
    ra, rb, d, p, n = concise_contrast(papers)
    print(f"  Skepthical {ra:.1f} vs concise structured GPT-5.5 {rb:.1f}: paired Delta {d:+.1f} p={p:.3f} (n={n})")

    print("(6) detection rate % by severity, pooled over error types (S1 / S2 / S3)")
    for sysd, tag, label in SYSTEMS:
        d = D[(sysd, tag)]
        if d:
            r = [100 * np.mean([d[k] for k in d if inj[k] == s]) for s in SEVERITIES]
            print(f"  {label:30s} {r[0]:5.1f} / {r[1]:5.1f} / {r[2]:5.1f}")

    print("(7) Section 3.2 false-positive summaries")
    tot = {}
    for sysd, tag, label in SYSTEMS:
        pairs = [x for m in MODS for x in F[(sysd, tag)][m]]
        if pairs:
            tot[(sysd, tag)] = (label, 100 * sum(v == "FP" for _, v in pairs) / len(pairs))
    rest = {k: v for k, v in tot.items() if k not in OPEN_WEIGHTS}
    for thr in (9, 10):
        below = [lab for lab, r in rest.values() if r < thr]
        print(f"  systems other than the open-weights reviewers with FP rate < {thr}%: {len(below)}/{len(rest)}; "
              "above: " + (", ".join(f"{lab} {r:.1f}" for lab, r in rest.values() if r >= thr) or "none"))
    frontier = [k for k in tot if k not in OPEN_WEIGHTS]
    n_fp_all = sum(v == "FP" for k in frontier for m in MODS for _, v in F[k][m])
    for m in MODS:
        vs = [v for k in frontier for _, v in F[k][m]]
        nfp = sum(v == "FP" for v in vs)
        print(f"  frontier-model systems pooled, {m:11s}: {nfp}/{len(vs)} = {100*nfp/len(vs):.1f}% false; "
              f"{100*nfp/n_fp_all:.1f}% of their {n_fp_all} false positives")
    for a, b, lab in ([(("L1", p), ("L2", p), PROVIDER_MODEL[p]) for p in ("anthropic", "openai", "gemini")]
                      + [(("CL1", ""), ("C", ""), "Claude Code")]):
        ga, gb = len(F[a]["general"]), len(F[b]["general"])
        fa = sum(v == "FP" for m in MODS for _, v in F[a][m]); na = sum(len(F[a][m]) for m in MODS)
        fb = sum(v == "FP" for m in MODS for _, v in F[b][m]); nb = sum(len(F[b][m]) for m in MODS)
        print(f"  {lab:15s} free-form -> structured: general-issue flags {ga} -> {gb} (x{gb/ga:.2f}); "
              f"false positives {fa} -> {fb} on {na} -> {nb} flags")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["astro", "ml"]:
        run(name)
