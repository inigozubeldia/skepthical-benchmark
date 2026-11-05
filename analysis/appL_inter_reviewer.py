#!/usr/bin/env python3
"""Appendix L: inter-reviewer agreement on the astrophysics corpus (Figure 5).

Per-injection pairwise Pearson r (and Cohen's kappa) between six reviewer systems,
with single-LLM entries detected when any provider detected; paper-level correlation
of detection rates; and a chi-square dispersion ratio per system (near 1 for
independent detections within a paper).
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import MODS, OUTPUT, REPO, SYSTEMS as TABLE2_SYSTEMS, corpus, load_detections
from pathlib import Path

OUT = OUTPUT / "figures"
PAPERS = sorted(corpus("astro"))
PROVIDERS = ("anthropic", "openai", "gemini")

SYSTEMS = ("L1", "L2", "L3", "C", "CL1", "B1")
RUN_TAGS = {"L1": PROVIDERS, "L2": PROVIDERS, "L3": ("",), "C": ("",), "CL1": ("",), "B1": ("",)}
SHORT_FULL = {"L1": "LLM, free-form", "L2": "LLM, structured", "L3": "Skepthical",
              "C": "Claude Code, structured", "CL1": "Claude Code, free-form", "B1": "Stanford"}
Y_FULL = {"L1": "LLM,\nfree-form", "L2": "LLM,\nstructured", "L3": "Skepthical",
          "C": "Claude Code,\nstructured", "CL1": "Claude Code,\nfree-form", "B1": "Stanford"}
SHORT = {"L1": "LLM-ff", "L2": "LLM-st", "L3": "Skep", "C": "CC-st", "CL1": "CC-ff", "B1": "SAR"}
LEGEND = ("LLM-ff / LLM-st = single LLM (free-form / structured), Skep = Skepthical, "
          "CC-st / CC-ff = Claude Code (structured / free-form), SAR = Stanford Agentic Reviewer")
NAME = {(system, tag): name for system, tag, name in TABLE2_SYSTEMS}
TEN = [("L1", p) for p in PROVIDERS] + [("L2", p) for p in PROVIDERS] + [("L3", ""), ("C", ""), ("CL1", ""), ("B1", "")]

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


def pearson_b(a, b):
    a = [int(x) for x in a]; b = [int(x) for x in b]; n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return float("nan")
    return num / math.sqrt(va * vb)


def kappa_b(a, b):
    a = [int(x) for x in a]; b = [int(x) for x in b]; n = len(a)
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p1 = sum(a) / n; p2 = sum(b) / n
    p_e = p1 * p2 + (1 - p1) * (1 - p2)
    if p_e in (0.0, 1.0):
        return float("nan")
    return (p_o - p_e) / (1 - p_e)


D, INJ = load_detections(PAPERS)

det = defaultdict(dict)
for s in SYSTEMS:
    for tag in RUN_TAGS[s]:
        for key, d in D.get((s, tag), {}).items():
            det[key][s] = det[key].get(s, False) or d
full_keys = [k for k in sorted(det) if all(s in det[k] for s in SYSTEMS)]
keys_by_module = {m: [k for k in full_keys if k[2] == m] for m in MODS}


def matrix(fn, keys):
    n = len(SYSTEMS); Mx = np.full((n, n), np.nan)
    for i, s1 in enumerate(SYSTEMS):
        for j, s2 in enumerate(SYSTEMS):
            Mx[i, j] = 1.0 if i == j else fn([det[k][s1] for k in keys], [det[k][s2] for k in keys])
    return Mx


def rate(s, keys):
    return sum(det[k][s] for k in keys) / len(keys)


def per_injection_report():
    print("Per-injection agreement (astrophysics corpus, six systems)")
    print("  " + LEGEND)
    sets = [("overall", full_keys)] + [(m, keys_by_module[m]) for m in MODS]
    for name, keys in sets:
        R, K = matrix(pearson_b, keys), matrix(kappa_b, keys)
        print(f"\n  [{name}]  n = {len(keys)} injected errors")
        print("    rates: " + ", ".join(f"{SHORT[s]} {rate(s, keys):.3f}" for s in SYSTEMS))
        for i, j in combinations(range(len(SYSTEMS)), 2):
            s1, s2 = SYSTEMS[i], SYSTEMS[j]
            print(f"    {SHORT[s1]:>6}-{SHORT[s2]:<6} r={R[i, j]:+.3f}  kappa={K[i, j]:+.3f}  kappa-r={K[i, j] - R[i, j]:+.3f}"
                  f"  rates {rate(s1, keys):.3f}/{rate(s2, keys):.3f}")
        off = [R[i, j] for i, j in combinations(range(len(SYSTEMS)), 2)]
        print(f"    off-diagonal r range: {min(off):+.2f} to {max(off):+.2f}")
        if name == "overall":
            pairs = list(combinations(range(len(SYSTEMS)), 2))
            by_r = sorted(pairs, key=lambda p: -R[p]); by_k = sorted(pairs, key=lambda p: -K[p])
            print(f"    pairs ranked by r and by kappa in the same order: {by_r == by_k}")
            if by_r != by_k:
                print("      by r:     " + " ".join(f"{SHORT[SYSTEMS[a]]}-{SHORT[SYSTEMS[b]]}" for a, b in by_r))
                print("      by kappa: " + " ".join(f"{SHORT[SYSTEMS[a]]}-{SHORT[SYSTEMS[b]]}" for a, b in by_k))
            tops = sorted(pairs, key=lambda p: -R[p])[:2]
            print("    highest pairs: " + "; ".join(f"{SHORT[SYSTEMS[a]]}-{SHORT[SYSTEMS[b]]} {R[a, b]:.2f}" for a, b in tops))
            dk = [K[p] - R[p] for p in pairs]
            print(f"    kappa - r over all pairs: {min(dk):+.3f} to {max(dk):+.3f}")

    keys = keys_by_module["numerics"]
    pa, pb = rate("L3", keys), rate("C", keys)
    agree = sum(det[k]["L3"] == det[k]["C"] for k in keys) / len(keys)
    print(f"\n  numerics, Skepthical vs Claude Code (structured): rates {pa:.3f} and {pb:.3f}; "
          f"same verdict expected from the rates alone {pa * pb + (1 - pa) * (1 - pb):.3f}, observed {agree:.3f}")
    print(f"  two systems that each detect 80%: same verdict expected from the rates alone {0.8 * 0.8 + 0.2 * 0.2:.2f}")
    return matrix(pearson_b, full_keys), {m: matrix(pearson_b, keys_by_module[m]) for m in MODS}


def paper_level_report():
    print("\nPaper-level agreement and dispersion (astrophysics corpus, ten systems)")
    n_inj = {p: sum(1 for k in INJ if k[0] == p) for p in PAPERS}
    print("  injected errors per paper: " + ", ".join(f"{n}x{sorted(n_inj.values()).count(n)}"
                                                    for n in sorted(set(n_inj.values()))))
    k = {s: np.array([sum(v for key, v in D.get(s, {}).items() if key[0] == p) for p in PAPERS]) for s in TEN}
    rates = np.array([[k[s][i] / n_inj[p] for s in TEN] for i, p in enumerate(PAPERS)])
    R = np.corrcoef(rates, rowvar=False)
    upper = R[np.triu_indices_from(R, k=1)]
    print(f"  mean pairwise Pearson r between system columns: {np.nanmean(upper):+.3f} "
          f"(n = {np.isfinite(upper).sum()} pairs, {len(TEN)} systems)")

    ratios = []
    for s in TEN:
        N = np.array([sum(1 for key in D.get(s, {}) if key[0] == p) for p in PAPERS])
        p_hat = k[s].sum() / N.sum()
        chi2 = np.sum((k[s] - N * p_hat) ** 2 / (N * p_hat * (1 - p_hat)))
        ratios.append(chi2 / (len(PAPERS) - 1))
        print(f"    {NAME[s]:<29} p={p_hat:.3f}  chi2={chi2:6.2f}  ratio={ratios[-1]:.2f}")
    print(f"  mean dispersion ratio over the ten systems: {np.mean(ratios):.2f}")


def render(M_overall, M_module):
    n_sys = len(SYSTEMS)
    fig = plt.figure(figsize=(11.0, 8.5))
    gs = fig.add_gridspec(4, 3, hspace=0.85, wspace=0.30, left=0.05, right=0.98, top=0.88, bottom=0.05)
    ax_overall = fig.add_subplot(gs[1:3, 0])
    ax_cit = fig.add_subplot(gs[0:2, 1]); ax_num = fig.add_subplot(gs[0:2, 2])
    ax_mat = fig.add_subplot(gs[2:4, 1]); ax_gen = fig.add_subplot(gs[2:4, 2])
    cmap = mpl.colormaps.get_cmap("RdBu_r")
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)

    def panel(ax, Mx, title, n, title_x=0.5):
        Ms = Mx.copy()
        Ms[np.triu_indices(n_sys, k=1)] = np.nan
        im = ax.imshow(Ms, cmap=cmap, norm=norm, aspect="equal")
        for i in range(n_sys):
            for j in range(n_sys):
                v = Ms[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}" if i != j else "1", ha="center", va="center", fontsize=11,
                        color="white" if abs(v) > 0.6 else "black", fontweight="bold")
        ax.set_xticks(range(n_sys)); ax.set_xticklabels([L(SHORT_FULL[s]) for s in SYSTEMS], rotation=35, ha="right", fontsize=11)
        ax.set_yticks(range(n_sys)); ax.set_yticklabels([L(Y_FULL[s]) for s in SYSTEMS], fontsize=11)
        ax.set_title(rf"{L(title)}" "\n" rf"($n={n}$ cells)", fontsize=11.5, pad=4, x=title_x)
        return im

    for ax, Mx, ttl, n, tx in [
            (ax_overall, M_overall, "Overall (all 4 error types)", len(full_keys), 0.5),
            (ax_cit, M_module["citations"], "Citations", len(keys_by_module["citations"]), 0.5),
            (ax_num, M_module["numerics"], "Numerics", len(keys_by_module["numerics"]), 0.5),
            (ax_mat, M_module["mathematics"], "Mathematics", len(keys_by_module["mathematics"]), 0.62),
            (ax_gen, M_module["general"], "General", len(keys_by_module["general"]), 0.62)]:
        im = panel(ax, Mx, ttl, n, tx)
    cbar_ax = fig.add_axes([0.72, 0.945, 0.24, 0.022])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Pearson $r$", fontsize=11, labelpad=4)
    cbar.set_ticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); cbar.ax.set_xlim(-0.2, 1.0)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.xaxis.set_label_position("top"); cbar.ax.xaxis.set_ticks_position("bottom")
    path = OUT / "fig_correlation_reviewers.pdf"
    fig.savefig(path, metadata={"CreationDate": None}); plt.close(fig); print("\n  ->", Path(path).relative_to(REPO))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    M_overall, M_module = per_injection_report()
    paper_level_report()
    render(M_overall, M_module)
