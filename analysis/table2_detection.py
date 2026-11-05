"""Table 2: detection rate (%) per error type and in total, with 95 % paper-level
bootstrap half-widths (2000 resamples, seed 20260427), for each corpus.

    python analysis/table2_detection.py [astro|ml]      (default: both corpora)

LaTeX rows go to stdout; the counts behind each cell go to stderr.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from common import CORPORA, CORPUS_NAMES, MODS, SYSTEMS, iter_verdicts, load_injections

LABELS = {("L3", ""): "\\skepthical{}", ("C", ""): "Claude Code, structured",
          ("CL1", ""): "Claude Code, free-form"}
GROUPS = [(0, 3, "\\emph{\\shortstack[l]{Single LLM\\\\free-form}}"),
          (3, 6, "\\emph{\\shortstack[l]{Single LLM\\\\structured}}"),
          (6, 9, "\\emph{Agentic}"),
          (9, 12, "\\emph{\\shortstack[l]{External\\\\baselines}}")]


def label(sysd, tag, long_label):
    return LABELS.get((sysd, tag), long_label.split(" (")[0])


def load(papers):
    """{(system, tag): {module: [(paper, detected), ...]}} over adjudicated injections."""
    inj = load_injections(papers)
    V = defaultdict(lambda: defaultdict(list))
    for sysd, tag, p, copy, v in iter_verdicts(papers):
        if (p, copy, v.get("module")) in inj:
            V[(sysd, tag)][v.get("module")].append((p, bool(v.get("detected"))))
    return V


def boot(pairs, n_boot=2000, seed=20260427, ci=0.95):
    """(rate %, half-width %, n) with a paper-level bootstrap; None if there are no verdicts."""
    by = defaultdict(list)
    for p, d in pairs:
        by[p].append(int(d))
    papers = list(by)
    flat = [x for p in papers for x in by[p]]
    if not flat:
        return None
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        smp = rng.choice(papers, size=len(papers), replace=True)
        vals = [x for p in smp for x in by[p]]
        if vals:
            bs.append(np.mean(vals))
    a = (1 - ci) / 2
    lo, hi = np.percentile(bs, [100 * a, 100 * (1 - a)])
    return 100 * np.mean(flat), 50 * (hi - lo), len(flat)


def fmt(cell, bold):
    if cell is None:
        return "--"
    rate, hw, n = cell
    if rate == 0:
        return "$0.0$"
    hw = round(hw, 1)
    body = f"{rate:.1f}\\pm {hw:.0f}" if hw >= 10 else f"{rate:.1f}\\pm {hw:.1f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def rows_for(name, bold=True):
    """LaTeX rows for one corpus block, plus {system label: [(rate, half-width, n) or None] * 5}."""
    V = load(CORPORA[name])
    table = []
    for sysd, tag, long_label in SYSTEMS:
        cells = [boot(V[(sysd, tag)][m]) for m in MODS] + [boot([x for m in MODS for x in V[(sysd, tag)][m]])]
        table.append((label(sysd, tag, long_label), long_label, cells))
    best = [max((c[j][0] for _, _, c in table if c[j]), default=None) for j in range(5)]
    out = []
    for gi, (a, b, head) in enumerate(GROUPS):
        run = [(lab, cells) for lab, _, cells in table[a:b] if any(cells)]
        if gi:
            out.append("\\midrule")
        out.append(f"\\multirow{{{len(run)}}}{{*}}{{{head}}}")
        for lab, cells in run:
            out.append(" & " + lab + " & " + " & ".join(
                fmt(c, bold and c is not None and c[0] > 0 and c[0] == best[j]) for j, c in enumerate(cells)) + " \\\\")
    return "\n".join(out), {long: [(c[0], c[1], c[2]) if c else None for c in cells] for _, long, cells in table}


if __name__ == "__main__":
    names = sys.argv[1:] or ["astro", "ml"]
    for name in names:
        latex, raw = rows_for(name)
        print(f"% {CORPUS_NAMES[name]}")
        print(latex)
        print(f"\n% {name} k/n check:", {k: [f"{v[0]:.1f}% (n={v[2]})" if v else None for v in vals]
                                       for k, vals in raw.items()}, file=sys.stderr)
