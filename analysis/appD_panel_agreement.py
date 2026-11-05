#!/usr/bin/env python3
"""Appendix D: pairwise agreement between judges within each adjudication panel (Figure 4).

Pearson r and Cohen's kappa on binary votes for the detection-matching panel
(first-round votes only) and the general-issues false-positive panel
(CONFIRMED_ISSUE vs. not). The ML-minus-astrophysics difference uses a
paper-level bootstrap (2,000 draws, seed 20260427).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, FP_RE, OUTPUT, REPO, VER_RE, corpus

OUT = OUTPUT / "figures"
JUDGES = ("anthropic", "openai", "gemini")
LABELS_MATCH = {"anthropic": "Claude Sonnet 4.6", "openai": "GPT-5.4", "gemini": "Gemini 3 Flash"}
LABELS_FP = {"anthropic": "Claude Sonnet 4.6", "openai": "GPT-5.4", "gemini": "Gemini 3.1 Pro"}
PAIRS = [(ja, jb) for i, ja in enumerate(JUDGES) for jb in JUDGES[i + 1:]]
N_BOOT, SEED = 2000, 20260427

mpl.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": ["Times", "Times New Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{textcomp}",
    "font.size": 10, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def table2_system(system, tag):
    if system in ("L1", "L2"):
        return tag in ("anthropic", "openai", "gemini")
    if system in ("L3", "C", "CL1", "B1", "B2"):
        return tag == ""
    if system == "B3":
        return tag in ("", "deep")
    return False


def stats(pairs):
    """(Pearson r, Cohen's kappa, n) for a list of (vote_a, vote_b) booleans."""
    a = [int(x) for x, _ in pairs]; b = [int(y) for _, y in pairs]; n = len(a)
    if n == 0:
        return float("nan"), float("nan"), 0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    r = num / math.sqrt(va * vb) if va and vb else float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n; pe = ma * mb + (1 - ma) * (1 - mb)
    k = (po - pe) / (1 - pe) if pe not in (0.0, 1.0) else float("nan")
    return r, k, n


def add_pairs(out, paper, voted):
    for ja, jb in PAIRS:
        if ja in voted and jb in voted:
            out[(ja, jb)][paper].append((voted[ja], voted[jb]))


def matching_pairs(papers):
    """{(judge_a, judge_b): {paper: [(vote_a, vote_b), ...]}}, first-round votes."""
    out = defaultdict(lambda: defaultdict(list))
    for p in papers:
        for jf in sorted((DATA / p / "verifications").glob("*/v*.json")):
            mm = VER_RE.match(jf.stem)
            if not mm or not table2_system(jf.parent.name, (mm.group(2) or "").lower()):
                continue
            for v in json.loads(jf.read_text()).get("verdicts", []):
                panel = v.get("judge_panel")
                if not isinstance(panel, list):
                    continue
                voted = {}
                for e in panel:
                    if not isinstance(e, dict) or e.get("escalated_with_pdf"):
                        continue
                    prov = (e.get("judge_provider") or "").lower(); det = e.get("detected")
                    if prov in JUDGES and det is not None:
                        voted[prov] = bool(det)
                add_pairs(out, p, voted)
    return out


def fp_general_pairs(papers):
    """{(judge_a, judge_b): {paper: [(confirmed_a, confirmed_b), ...]}} on general-issue flags."""
    out = defaultdict(lambda: defaultdict(list))
    for p in papers:
        for jf in sorted((DATA / p / "fp").glob("*_v0.json")):
            mm = FP_RE.match(jf.stem)
            if not mm or not table2_system(mm.group(1), (mm.group(2) or "").lower()):
                continue
            for flag in (json.loads(jf.read_text()).get("per_module_verdicts") or {}).get("general") or []:
                panel = flag.get("judge_panel")
                if not isinstance(panel, list):
                    continue
                voted = {}
                for e in panel:
                    if not isinstance(e, dict):
                        continue
                    prov = (e.get("judge_provider") or "").lower(); vd = (e.get("verdict") or "").upper()
                    if prov in JUDGES and vd:
                        voted[prov] = (vd == "CONFIRMED_ISSUE")
                add_pairs(out, p, voted)
    return out


def pooled(by_paper):
    return [x for p in sorted(by_paper) for x in by_paper[p]]


def matrix(pairs):
    Mx = np.full((3, 3), np.nan); per_pair = {}
    for i, ja in enumerate(JUDGES):
        for j, jb in enumerate(JUDGES):
            if i == j:
                Mx[i, j] = 1.0; continue
            key = (ja, jb) if i < j else (jb, ja)
            r, k, n = stats(pooled(pairs.get(key, {})))
            Mx[i, j] = r; per_pair[key] = (r, k, n)
    ns = sorted({per_pair[k][2] for k in PAIRS})
    return Mx, (f"$n={ns[0]}$" if len(ns) == 1 else rf"$n \in [{ns[0]},{ns[-1]}]$"), per_pair


def boot_r(by_paper, papers, rng):
    sample = rng.choice(papers, size=len(papers), replace=True)
    return stats([x for p in sample for x in by_paper.get(p, [])])[0]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data, blocks = {}, []
    for cname, title in (("astro", "Astrophysics corpus"), ("ml", "Machine-learning corpus")):
        papers = sorted(corpus(cname))
        data[cname] = (papers, matching_pairs(papers), fp_general_pairs(papers))
        _, dp, fp = data[cname]
        dM, dn, dstat = matrix(dp)
        fM, fn, fstat = matrix(fp)
        blocks.append((title, dM, dn, fM, fn))
        print(f"{title}")
        for panel, st in (("detection-matching panel (first-round votes)", dstat),
                          ("general-issues false-positive panel", fstat)):
            print(f"  {panel}")
            for ja, jb in PAIRS:
                r, k, n = st[(ja, jb)]
                print(f"    {ja:>9}-{jb:<7} r={r:.3f}  kappa={k:.3f}  |kappa-r|={abs(k - r):.3f}  n={n}")
            rs = [st[p][0] for p in PAIRS]; ns = [st[p][2] for p in PAIRS]
            print(f"    r range {min(rs):.2f}-{max(rs):.2f}; n range {min(ns)}-{max(ns)}; "
                  f"max |kappa-r| = {max(abs(st[p][1] - st[p][0]) for p in PAIRS):.3f}")

    print(f"\nML - astrophysics difference in r (paper-level bootstrap, {N_BOOT} draws, seed {SEED}, 95% interval)")
    for idx, panel in ((1, "detection-matching"), (2, "general-issues FP")):
        for ja, jb in PAIRS:
            (pa, *ra), (pm, *rm) = data["astro"], data["ml"]
            a_pairs, m_pairs = ra[idx - 1][(ja, jb)], rm[idx - 1][(ja, jb)]
            diff = stats(pooled(m_pairs))[0] - stats(pooled(a_pairs))[0]
            rng = np.random.default_rng(SEED)
            boots = [boot_r(m_pairs, pm, rng) - boot_r(a_pairs, pa, rng) for _ in range(N_BOOT)]
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            flag = "  <- excludes zero" if lo > 0 or hi < 0 else ""
            print(f"  {panel:<19} {ja:>9}-{jb:<7} diff={diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.4))
    cmap = mpl.colormaps.get_cmap("Reds"); norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    for r_, (name, dM, dn, fM, fn) in enumerate(blocks):
        for c_, (Mx, ttl, n_str, labels) in enumerate([
                (dM, "Detection-matching panel", dn, LABELS_MATCH),
                (fM, "FP general-issues panel", fn, LABELS_FP)]):
            ax = axes[r_, c_]
            im = ax.imshow(Mx, cmap=cmap, norm=norm, aspect="equal")
            for i in range(3):
                for j in range(3):
                    v = Mx[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, "1" if i == j else f"{v:.2f}", ha="center", va="center", fontsize=12,
                            color="white" if v > 0.65 else "black", fontweight="bold")
            ax.set_xticks(range(3)); ax.set_xticklabels([labels[j] for j in JUDGES], rotation=20, ha="right", fontsize=10)
            ax.set_yticks(range(3)); ax.set_yticklabels([labels[j] for j in JUDGES], fontsize=10)
            ax.set_title(f"{ttl} ({n_str} pair-verdicts)", fontsize=10)
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_visible(False)
        fig.text(0.5, 0.975 - 0.485 * r_, rf"\textbf{{{name}}}", ha="center", va="bottom", fontsize=12)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.93, bottom=0.08, hspace=0.62, wspace=0.55)
    cax = fig.add_axes([0.905, 0.36, 0.014, 0.28])
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"Pearson $r$"); cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    path = OUT / "fig_correlation_panels_two_corpora.pdf"
    fig.savefig(path, metadata={"CreationDate": None}); print("\n  ->", Path(path).relative_to(REPO))
