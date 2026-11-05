"""Appendix G: significance tests (Tables 9 and 10) and the Section 3.1 paired tests.

Paired paper-level bootstrap (10,000 resamples, seed 20260506): the reported
probability is the fraction of resamples in which the sign of the difference
reverses (zero counts as a reversal; 0.5 when the full difference is zero).
Severity trend: Page's L on per-paper (S1, S2, S3) rates, one-sided permutation
p-value (10,000 permutations, seed 20260507), Benjamini-Hochberg at 0.05.

Usage: python analysis/appG_significance.py [astro|ml]   (default: both)
"""
from __future__ import annotations

import json
import random
import sys
from itertools import permutations

from common import CORPORA, CORPUS_NAMES, DATA

MODULES = ("citations", "numerics", "mathematics", "general")
COLS = ("total",) + MODULES
SEVERITIES = ("S1", "S2", "S3")
COPIES = ("v1", "v2", "v3")

BOOT_B = 10_000
BOOT_SEED = 20260506
PERM_B = 10_000
PERM_SEED = 20260507
ALPHA = 0.05

SYSTEMS = [
    ("L1", "anthropic", "Claude Opus 4.7 (free-form)"),
    ("L1", "openai", "GPT-5.5 (free-form)"),
    ("L1", "gemini", "Gemini 3.1 Pro (free-form)"),
    ("L2", "anthropic", "Claude Opus 4.7 (structured)"),
    ("L2", "openai", "GPT-5.5 (structured)"),
    ("L2", "gemini", "Gemini 3.1 Pro (structured)"),
    ("L3", "", "Skepthical"),
    ("C", "", "Claude Code (structured)"),
    ("CL1", "", "Claude Code (free-form)"),
    ("B1", "", "Stanford Agentic Reviewer"),
    ("B2", "", "OpenReviewer-8B"),
    ("B3", "deep", "PaperAudit-Bench Deep"),
]
LOW_DETECTION = ("OpenReviewer-8B", "PaperAudit-Bench Deep")

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


def load_records(papers):
    sev = {}
    for p in papers:
        for copy in COPIES:
            f = DATA / p / copy / "injections.json"
            if f.exists():
                sev[(p, copy)] = {r.get("module"): r.get("severity") for r in json.loads(f.read_text())}
    records = []
    for system, tag, label in SYSTEMS:
        for p in papers:
            for copy in COPIES:
                f = DATA / p / "verifications" / system / (f"{copy}__{tag}.json" if tag else f"{copy}.json")
                if not f.exists():
                    continue
                for v in json.loads(f.read_text()).get("verdicts", []):
                    module = v.get("module")
                    s = sev.get((p, copy), {}).get(module)
                    if module not in MODULES or s not in SEVERITIES:
                        continue
                    records.append({"label": label, "paper": p, "module": module,
                                    "severity": s, "detected": bool(v.get("detected"))})
    return records


def per_paper_counts(records, papers, label, module=None):
    """{paper: (detected, adjudicated)} for one reviewer (and optionally one error type)."""
    out = {p: [0, 0] for p in papers}
    for r in records:
        if r["label"] == label and (module is None or r["module"] == module):
            out[r["paper"]][1] += 1
            out[r["paper"]][0] += r["detected"]
    return {p: tuple(v) for p, v in out.items()}


def pooled_rate(counts, sample=None):
    sample = list(counts) if sample is None else sample
    det = sum(counts[p][0] for p in sample)
    tot = sum(counts[p][1] for p in sample)
    return det / tot if tot else float("nan")


def paired_bootstrap(records, papers, a, b, B=BOOT_B, seed=BOOT_SEED):
    """{column: (k, n)}: k of n valid resamples have rate(a) - rate(b) <= 0."""
    rng = random.Random(seed)
    n_p = len(papers)
    out = {}
    for col in COLS:
        mod = None if col == "total" else col
        ca = per_paper_counts(records, papers, a, mod)
        cb = per_paper_counts(records, papers, b, mod)
        k = n = 0
        for _ in range(B):
            sample = [papers[rng.randrange(n_p)] for _ in range(n_p)]
            ra, rb = pooled_rate(ca, sample), pooled_rate(cb, sample)
            if ra != ra or rb != rb:
                continue
            n += 1
            k += (ra - rb) <= 0
        out[col] = (k, n)
    return out


def contrast(records, papers, a, b):
    """[(column, difference in pp, reversal probability, k, n)]."""
    pab = paired_bootstrap(records, papers, a, b)
    pba = paired_bootstrap(records, papers, b, a)
    rows = []
    for col in COLS:
        mod = None if col == "total" else col
        d = (pooled_rate(per_paper_counts(records, papers, a, mod))
             - pooled_rate(per_paper_counts(records, papers, b, mod))) * 100
        if d > 0:
            k, n = pab[col]
        elif d < 0:
            k, n = pba[col]
        else:
            k, n = None, None
        p = 0.5 if k is None else k / n
        rows.append((col, d, p, k, n))
    return rows


def midranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def page_L(triples):
    return sum(j * r for t in triples for j, r in enumerate(midranks(t), start=1))


def page_L_pvalue(triples, B=PERM_B, seed=PERM_SEED):
    L_obs = page_L(triples)
    rng = random.Random(seed)
    perms = list(permutations(range(3)))
    n_ge = 0
    for _ in range(B):
        permuted = []
        for t in triples:
            q = perms[rng.randrange(6)]
            permuted.append([t[q[0]], t[q[1]], t[q[2]]])
        n_ge += page_L(permuted) >= L_obs
    return (n_ge + 1) / (B + 1)


def bh_reject(pvalues, alpha=ALPHA):
    n = len(pvalues)
    ranked = sorted(enumerate(pvalues), key=lambda x: x[1])
    last = -1
    for rank, (_, p) in enumerate(ranked):
        if p <= (rank + 1) * alpha / n:
            last = rank
    reject = [False] * n
    for rank, (i, _) in enumerate(ranked):
        reject[i] = rank <= last
    return reject


def trend_pooled(records, papers, reviewers):
    out = {}
    for module in MODULES:
        det = {(p, s): 0 for p in papers for s in SEVERITIES}
        tot = {(p, s): 0 for p in papers for s in SEVERITIES}
        for r in records:
            if r["module"] == module and r["label"] in reviewers:
                det[(r["paper"], r["severity"])] += r["detected"]
                tot[(r["paper"], r["severity"])] += 1
        triples = [[det[(p, s)] / tot[(p, s)] for s in SEVERITIES]
                   for p in papers if all(tot[(p, s)] for s in SEVERITIES)]
        out[module] = {"n_papers": len(triples), "p": page_L_pvalue(triples),
                       "rates": [sum(t[i] for t in triples) / len(triples) for i in range(3)]}
    for m, rej in zip(MODULES, bh_reject([out[m]["p"] for m in MODULES])):
        out[m]["bh"] = rej
    return out


def trend_per_reviewer(records, papers, labels):
    cells = []
    for label in labels:
        for module in MODULES:
            c = {(p, s): [0, 0] for p in papers for s in SEVERITIES}
            for r in records:
                if r["label"] == label and r["module"] == module:
                    c[(r["paper"], r["severity"])][1] += 1
                    c[(r["paper"], r["severity"])][0] += r["detected"]
            triples = [[c[(p, s)][0] / c[(p, s)][1] for s in SEVERITIES]
                       for p in papers if all(c[(p, s)][1] for s in SEVERITIES)]
            cells.append({"label": label, "module": module, "n_papers": len(triples),
                          "p": page_L_pvalue(triples),
                          "rates": [sum(t[i] for t in triples) / len(triples) for i in range(3)]})
    for c, rej in zip(cells, bh_reject([c["p"] for c in cells])):
        c["bh"] = rej
    return cells


def report(which):
    papers = list(CORPORA[which])
    records = load_records(papers)
    present = [lab for _, _, lab in SYSTEMS if any(r["label"] == lab for r in records)]
    print(f"\n{'=' * 78}\n{CORPUS_NAMES[which]}: {len(papers)} papers, {len(records):,} adjudicated "
          f"injections, {len(present)} reviewers with data\n{'=' * 78}")

    print("\nTable 9: difference (pp, a - b) and probability that its sign reverses")
    for name, a, b in CONTRASTS:
        rows = contrast(records, papers, a, b)
        print(f"  {name}")
        for col, d, p, k, n in rows:
            mark = "*" if p < 0.05 else " "
            detail = (f"{k}/{n} resamples" + ("  (no resample reversed)" if k == 0 else "")
                      if k is not None else "difference is zero; 0.5 by convention")
            print(f"    {col:<12} {d:+6.1f}  p = {p:.3f} {mark}  [{detail}]")
    print("  (* = probability below 0.05)")

    print("\nSkepthical vs every other reviewer: Pr(rate(Skepthical) - rate(other) <= 0)")
    for _, _, lab in SYSTEMS:
        if lab == "Skepthical" or lab not in present:
            continue
        pv = paired_bootstrap(records, papers, "Skepthical", lab)
        print(f"  {lab:<30} " + "  ".join(
            f"{c[:4]} " + ("<0.001" if pv[c][0] / pv[c][1] < 0.001 else f"{pv[c][0] / pv[c][1]:.3f}")
            for c in COLS))

    pooled_labels = [lab for lab in present if lab not in LOW_DETECTION]
    print(f"\nTable 10: severity trend pooled over {len(pooled_labels)} reviewers "
          f"(all with data except {', '.join(LOW_DETECTION)})")
    pooled = trend_pooled(records, papers, pooled_labels)
    for m in MODULES:
        v = pooled[m]
        r = v["rates"]
        print(f"  {m:<12} S1 {r[0] * 100:5.1f}  S2 {r[1] * 100:5.1f}  S3 {r[2] * 100:5.1f}  "
              f"p_raw {v['p']:.3f}  BH {'pass' if v['bh'] else '-'}  ({v['n_papers']} papers)")
    order = sorted(MODULES, key=lambda m: pooled[m]["p"])
    print("  BH thresholds by rank: " + ", ".join(
        f"{m} p={pooled[m]['p']:.4f} vs {(i + 1) * ALPHA / len(MODULES):.4f}" for i, m in enumerate(order)))

    cells = trend_per_reviewer(records, papers, present)
    n_papers = [c["n_papers"] for c in cells]
    print(f"\nPer-reviewer severity tests: {len(cells)} combinations ({len(present)} reviewers x "
          f"{len(MODULES)} error types), {min(n_papers)}-{max(n_papers)} papers per combination")
    print(f"  pass Benjamini-Hochberg: {sum(c['bh'] for c in cells)}; "
          f"nominal p < 0.05: {sum(c['p'] < 0.05 for c in cells)}")
    for c in cells:
        if c["bh"]:
            r = c["rates"]
            print(f"  rejected: {c['label']} / {c['module']}: S1 {r[0] * 100:.1f}  S2 {r[1] * 100:.1f}  "
                  f"S3 {r[2] * 100:.1f}  p_raw {c['p']:.4f}  ({c['n_papers']} papers)")


if __name__ == "__main__":
    for which in (sys.argv[1:] or ["astro", "ml"]):
        report(which)
