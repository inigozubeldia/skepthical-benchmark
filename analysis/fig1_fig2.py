#!/usr/bin/env python3
"""Figures 1 and 2, both corpora.

Figure 1: detection rate per error type and injected severity, with paper-level
bootstrap 68 % and 95 % intervals. Figure 2: clean-copy flags split into false
positives and ambiguous adjudications, with the FP rate and its 95 % interval.
Plotted numbers are also printed to stdout.
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from common import DATA, OUTPUT, REPO, SYSTEMS, corpus, iter_verdicts, load_fp, load_injections
from pathlib import Path

OUT = OUTPUT / "figures"
MODULES = ("citations", "numerics", "mathematics", "general")
SEVERITIES = ("S1", "S2", "S3")

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{textcomp}",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def L(s):
    return (s.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
             .replace("#", r"\#").replace("_", r"\_"))


MODEL_LABEL = {"anthropic": "Claude Opus 4.7", "openai": "GPT-5.5", "gemini": "Gemini 3.1 Pro"}
MODEL_COLOR = {"anthropic": "#c8581c", "openai": "#10a37f", "gemini": "#4285f4"}
NAME = {(system, tag): name for system, tag, name in SYSTEMS}


def boot_metric(by_paper, n_boot=1000, seed=20260427, ci=0.68):
    """Pooled rate and paper-level bootstrap interval."""
    papers = list(by_paper.keys())
    if not papers:
        return float("nan"), float("nan"), float("nan"), 0
    flat = [x for p in papers for x in by_paper[p]]
    rate = float(np.mean(flat))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(papers, size=len(papers), replace=True)
        vals = [x for p in sample for x in by_paper[p]]
        if vals:
            boots.append(np.mean(vals))
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    return rate, lo, hi, len(flat)


class Corpus:
    def __init__(self, name):
        self.papers = sorted(corpus(name))  # sorted order fixes the bootstrap draws
        self.inj = load_injections(self.papers)
        self.verdicts = defaultdict(lambda: defaultdict(list))
        for system, tag, paper, copy, v in iter_verdicts(self.papers):
            key = (paper, copy, v.get("module"))
            if key in self.inj:
                self.verdicts[(system, tag)][key].append(bool(v.get("detected")))
        order = {p: i for i, p in enumerate(self.papers)}
        self.fp = {k: sorted((pv for mod in MODULES for pv in by_mod.get(mod, [])), key=lambda pv: order[pv[0]])
                   for k, by_mod in load_fp(self.papers).items()}

    def cell(self, system, tag, module, sev):
        by_paper = defaultdict(list)
        for (paper, copy, m), vs in self.verdicts.get((system, tag), {}).items():
            if m == module and self.inj[(paper, copy, m)] == sev:
                by_paper[paper].extend(int(x) for x in vs)
        by_paper = {p: by_paper[p] for p in self.papers if p in by_paper}
        rate, lo68, hi68, n = boot_metric(by_paper, ci=0.68)
        _, lo95, hi95, _ = boot_metric(by_paper, ci=0.95)
        return rate, lo68, hi68, lo95, hi95, n

    def was_run(self, system, tag):
        stem = f"{system}__{tag}_v0.json" if tag else f"{system}_v0.json"
        return any((DATA / p / "fp" / stem).exists() for p in self.papers)


A, M = Corpus("astro"), Corpus("ml")
BLOCKS = [("Astrophysics corpus (15 papers, 177 injected errors)", A),
          ("Machine-learning corpus (15 papers, 180 injected errors)", M)]
for title, c in BLOCKS:
    assert f"{len(c.inj)} injected" in title, (title, len(c.inj))


SERIES = {
    "Single LLM, free-form":  [("L1", p, MODEL_COLOR[p]) for p in ("anthropic", "openai", "gemini")],
    "Single LLM, structured": [("L2", p, MODEL_COLOR[p]) for p in ("anthropic", "openai", "gemini")],
    "Agentic":            [("CL1", "", "#c4a8df"), ("C", "", "#7a4ab8"), ("L3", "", "#d62728")],
    "External baselines": [("B1", "", "#8c564b"), ("B2", "", "#dba73a"), ("B3", "deep", "#5cb85c")],
}
COLUMNS = list(SERIES)
MARKERS = ["o", "s", "D"]


def figure1():
    print("Figure 1: detection rate [68% interval] per (system, error type, severity)")
    fig = plt.figure(figsize=(13.5, 19.8))
    outer = fig.add_gridspec(2, 1, left=0.075, right=0.985, top=0.955, bottom=0.102, hspace=0.235)
    for b, (title, cp) in enumerate(BLOCKS):
        print(f"  [{title}]")
        inner = outer[b].subgridspec(4, 4, hspace=0.30, wspace=0.10)
        axes = np.empty((4, 4), dtype=object)
        for r, module in enumerate(MODULES):
            for c, col in enumerate(COLUMNS):
                ax = fig.add_subplot(inner[r, c]); axes[r, c] = ax
                x = np.arange(3)
                for k, (system, tag, color) in enumerate(SERIES[col]):
                    ys, l68, h68, l95, h95, txt = [], [], [], [], [], []
                    for sev in SEVERITIES:
                        rate, lo68, hi68, lo95, hi95, n = cp.cell(system, tag, module, sev)
                        ys.append(rate)
                        if np.isnan(rate):
                            l68.append(0); h68.append(0); l95.append(0); h95.append(0)
                            txt.append(f"{sev} -")
                        else:
                            l68.append(max(0, rate - lo68)); h68.append(max(0, hi68 - rate))
                            l95.append(max(0, rate - lo95)); h95.append(max(0, hi95 - rate))
                            txt.append(f"{sev} {rate:.3f} [{lo68:.3f},{hi68:.3f}] n={n}")
                    print(f"    {module:<11} {NAME[(system, tag)]:<29} " + "  ".join(txt))
                    jit = (k - 1) * 0.06
                    ax.errorbar(x + jit, ys, yerr=[l68, h68], marker=MARKERS[k], markersize=7,
                                linewidth=1.8, capsize=3, color=color, alpha=0.95)
                    ax.errorbar(x + jit, ys, yerr=[l95, h95], fmt="none", linewidth=0.9,
                                capsize=0, color=color, alpha=0.5)
                ax.set_xticks(x); ax.set_xlim(-0.5, 2.5)
                ax.set_ylim(-0.05, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.grid(axis="y", alpha=0.3, linestyle=":")
                if r == 3:
                    ax.set_xticklabels([r"$S_1$", r"$S_2$", r"$S_3$"], fontsize=18)
                    ax.set_xlabel("injected error severity", fontsize=18)
                else:
                    ax.set_xticklabels([])
                if c == 0:
                    ax.tick_params(axis="y", labelsize=16)
                    ax.set_ylabel(rf"\textbf{{{L(module)}}}\\\\detection rate", fontsize=18)
                else:
                    ax.set_yticklabels([])
                if r == 0:
                    ax.set_title(rf"\textbf{{{L(col)}}}", fontsize=18, pad=10)
        y_top = axes[0, 0].get_position().y1
        fig.text(0.53, y_top + 0.030, rf"\textbf{{{L(title)}}}", ha="center", va="bottom", fontsize=20)

    groups = [
        ("Single LLM (3 frontier models)",
         [(MODEL_COLOR[p], m, MODEL_LABEL[p]) for p, m in zip(("anthropic", "openai", "gemini"), "osD")]),
        ("Agentic", [("#c4a8df", "o", "Claude Code, free-form"), ("#7a4ab8", "s", "Claude Code, structured"),
                     ("#d62728", "D", "Skepthical")]),
        ("External baselines", [("#8c564b", "o", "Stanford (astrophysics only)"), ("#dba73a", "s", "OpenReviewer-8B"),
                                ("#5cb85c", "D", "PaperAudit-R (Deep)")]),
    ]
    for (grp, entries), xp in zip(groups, [0.18, 0.50, 0.82]):
        handles = [plt.Line2D([], [], color=c_, marker=m, markersize=7, linewidth=2.0) for c_, m, _ in entries]
        fig.legend(handles, [L(l) for _, _, l in entries], loc="center", bbox_to_anchor=(xp, 0.035),
                   title=L(grp), ncol=1, frameon=False, fontsize=16, title_fontsize=17, handlelength=2.2,
                   handletextpad=0.7, labelspacing=0.5, alignment="left")
    path = OUT / "fig1_dose_response_two_corpora.pdf"
    fig.savefig(path, metadata={"CreationDate": None}); plt.close(fig); print("  ->", Path(path).relative_to(REPO))


ROWS = [
    ("L1", "anthropic", MODEL_LABEL["anthropic"], "Single LLM, free-form"),
    ("L1", "openai",    MODEL_LABEL["openai"],    "Single LLM, free-form"),
    ("L1", "gemini",    MODEL_LABEL["gemini"],    "Single LLM, free-form"),
    ("L2", "anthropic", MODEL_LABEL["anthropic"], "Single LLM, structured"),
    ("L2", "openai",    MODEL_LABEL["openai"],    "Single LLM, structured"),
    ("L2", "gemini",    MODEL_LABEL["gemini"],    "Single LLM, structured"),
    ("CL1", "", "Claude Code, free-form",  "Agentic"),
    ("C",   "", "Claude Code, structured", "Agentic"),
    ("L3",  "", "Skepthical",              "Agentic"),
    ("B1",  "", "Stanford",                "External baselines"),
    ("B2",  "", "OpenReviewer-8B",         "External baselines"),
    ("B3",  "deep", "PaperAudit-R (Deep)", "External baselines"),
]


def fp_counts(cp):
    out = []
    for system, tag, label, group in ROWS:
        flags = cp.fp.get((system, tag), [])
        n_total = len(flags)
        n_fp = sum(1 for _, v in flags if v == "FP")
        n_amb = sum(1 for _, v in flags if v == "AMBIGUOUS")
        by_paper = defaultdict(list)
        for paper, v in flags:
            by_paper[paper].append(1 if v == "FP" else 0)
        if by_paper:
            _, lo, hi, _ = boot_metric(by_paper, n_boot=2000, ci=0.95); lo, hi = 100 * lo, 100 * hi
        else:
            lo = hi = 0.0
        out.append(dict(label=label, group=group, fp=n_fp, amb=n_amb, total=n_total,
                        fp_rate=(100 * n_fp / n_total if n_total else 0.0), ci_lo=lo, ci_hi=hi,
                        run=cp.was_run(system, tag), sys=NAME[(system, tag)]))
    return out


def figure2():
    print("\nFigure 2: flags on clean copies (FP, ambiguous, all adjudicated) and FP rate [95% interval]")
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 5.4), sharey=True)
    fig.subplots_adjust(left=0.335, right=0.985, top=0.93, bottom=0.19, wspace=0.08)
    handles = None
    for ax, (title, cp) in zip(axs, BLOCKS):
        counts = fp_counts(cp); n = len(counts); y = np.arange(n)[::-1]
        print(f"  [{title.split(' (')[0]}]")
        for c in counts:
            if c["run"]:
                print(f"    {c['sys']:<29} FP={c['fp']:>3} ambiguous={c['amb']:>3} total={c['total']:>4}  "
                      f"FP rate {c['fp_rate']:.1f}% [{c['ci_lo']:.1f}, {c['ci_hi']:.1f}]")
            else:
                print(f"    {c['sys']:<29} not run on this corpus")
        fp_w = np.array([c["fp"] for c in counts]); amb_w = np.array([c["amb"] for c in counts])
        h1 = ax.barh(y, fp_w, color="#c0392b", edgecolor="white", linewidth=0.6)
        h2 = ax.barh(y, amb_w, left=fp_w, color="#d6a847", edgecolor="white", linewidth=0.6)
        handles = handles or (h1, h2)
        for yi, fp, amb in zip(y, fp_w, amb_w):
            if fp > 0:
                ax.text(fp / 2, yi, str(int(fp)), ha="center", va="center", color="white", fontsize=13, fontweight="bold")
            if amb > 0:
                ax.text(fp + amb / 2, yi, str(int(amb)), ha="center", va="center", color="white", fontsize=13, fontweight="bold")
        x_max = max((fp_w + amb_w).max(), 1); off = x_max * 0.025
        for yi, fp, amb, c in zip(y, fp_w, amb_w, counts):
            if not c["run"]:
                ax.text(off, yi, r"\textit{not run on this corpus}", ha="left", va="center", fontsize=12, color="#777")
            else:
                ax.text(fp + amb + off, yi + 0.04, rf"\textbf{{{c['fp_rate']:.1f}\%}}", ha="left", va="bottom",
                        fontsize=12, color="#222")
                ax.text(fp + amb + off, yi - 0.04, rf"[{c['ci_lo']:.1f}, {c['ci_hi']:.1f}]", ha="left", va="top",
                        fontsize=11, color="#222")
        ax.set_xlim(0, x_max * 1.5); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="x", labelsize=12); ax.grid(axis="x", alpha=0.3, linestyle=":"); ax.set_axisbelow(True)
        ax.set_title(rf"\textbf{{{L(title.split(' (')[0])}}}", fontsize=13, pad=6)
        prev = None
        for yi, c in zip(y, counts):
            if prev is not None and c["group"] != prev:
                ax.axhline(yi + 0.5, color="#aaaaaa", linewidth=0.5, linestyle="--", alpha=0.6)
            prev = c["group"]
        if ax is axs[0]:
            ax.set_yticks(y); ax.set_yticklabels([L(c["label"]) for c in counts], fontsize=12, fontweight="bold")
            seen = {}
            for yi, c in zip(y, counts):
                seen.setdefault(c["group"], []).append(yi)
            for grp, rows in seen.items():
                two_line = grp.replace(", ", ",\n").replace("External baselines", "External\nbaselines")
                ax.annotate(L(two_line), xy=(-0.70, np.mean(rows)), xycoords=("axes fraction", "data"),
                            fontsize=12.5, color="#222", fontweight="bold", ha="right", va="center", linespacing=1.1)
                ax.annotate("", xy=(-0.665, max(rows) + 0.42), xytext=(-0.665, min(rows) - 0.42),
                            xycoords=("axes fraction", "data"), arrowprops=dict(arrowstyle="-", color="#888", linewidth=0.8))
    xc = (0.335 + 0.985) / 2
    fig.text(xc, 0.085, r"Flags on $n=15$ clean papers per corpus --- FP \(+\) Ambig.\ (count)",
             ha="center", va="center", fontsize=13)
    fig.legend(handles, ["False positive (clearly wrong flag)", "Ambiguous adjudication"], loc="center",
               bbox_to_anchor=(xc, 0.03), ncol=2, fontsize=11, frameon=False, handlelength=1.6, columnspacing=1.8)
    path = OUT / "fig2_fp_rates_two_corpora.pdf"
    fig.savefig(path, metadata={"CreationDate": None}); plt.close(fig); print("  ->", Path(path).relative_to(REPO))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
