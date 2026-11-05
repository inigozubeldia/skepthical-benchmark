#!/usr/bin/env python3
"""Appendix K: does detecting an error depend on the severity of the other errors in the same copy?

For each verdict, "worst neighbour" is the highest severity among the other three
errors of the copy. Prints detection rates by (own severity x worst neighbour) with
paper-level bootstrap intervals (2000 resamples, seed 20260427), and a logistic model
per corpus with paper-clustered errors and a paper-level bootstrap of the coefficient
(400 resamples).

Usage:  python analysis/appK_error_coupling.py [--ci 0.95]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONCISE, CORPORA, DATA, SYSTEMS  # noqa: E402

SEV = {"S1": 1, "S2": 2, "S3": 3}
VER_RE = re.compile(r"^(v[1-3])(?:__(.+))?$")
SEV_LABEL = {1: "subtle (S1)", 2: "clear (S2)", 3: "blatant (S3)"}
POOLED_SYSTEMS = {(s, t) for s, t, _ in SYSTEMS + CONCISE}


def term_name(term):
    if term == "co_max":
        return "worst neighbour"
    return "worst neighbour x own severity S" + term.split("[T.")[1].rstrip("]")


def build(corpus: str) -> pd.DataFrame:
    """One row per (paper, copy, error type, system) verdict; co_max is the worst-neighbour severity."""
    papers = CORPORA[corpus]
    sev_of: dict[tuple[str, str, str], int] = {}
    copy_sevs: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for p in papers:
        for c in json.loads((DATA / p / "manifest.json").read_text()).get("copies", []):
            for inj in c.get("injections", []):
                s = SEV.get(inj["severity"])
                if s is None:
                    continue
                sev_of[(p, c["copy_id"], inj["module"])] = s
                copy_sevs[(p, c["copy_id"])][inj["module"]] = s

    rows = []
    for p in papers:
        vd = DATA / p / "verifications"
        if not vd.is_dir():
            continue
        for sysd in sorted(vd.iterdir()):
            for jf in sorted(sysd.glob("*.json")):
                mm = VER_RE.match(jf.stem)
                if not mm:
                    continue
                copy, tag = mm.group(1), (mm.group(2) or "").lower()
                system = sysd.name
                if (system, tag) not in POOLED_SYSTEMS:
                    continue
                d = json.loads(jf.read_text())
                for v in d.get("verdicts", []):
                    mod = v.get("module")
                    key = (p, copy, mod)
                    if key not in sev_of:
                        continue
                    others = [s for m, s in copy_sevs[(p, copy)].items() if m != mod]
                    if not others:
                        continue
                    rows.append(dict(
                        paper=p, copy=copy, module=mod,
                        system=f"{system}:{tag}" if tag else system,
                        sev=sev_of[key],
                        co_max=max(others),
                        detected=int(bool(v.get("detected"))),
                    ))
    return pd.DataFrame(rows)


def boot_cell(sub: pd.DataFrame, all_papers, n_boot=2000, seed=20260427, ci=0.95):
    """Paper-level bootstrap of a cell's detection rate (papers resampled, not verdicts)."""
    if sub.empty:
        return np.nan, np.nan, np.nan, 0, 0, 0
    rate = float(sub.detected.mean())
    by_paper = {p: g.detected.to_numpy() for p, g in sub.groupby("paper")}
    rng = np.random.default_rng(seed)
    papers = np.array(all_papers)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(papers, size=len(papers), replace=True)
        vals = [by_paper[q] for q in samp if q in by_paper]
        if vals:
            boots.append(np.concatenate(vals).mean())
    a = (1 - ci) / 2
    lo, hi = (np.percentile(boots, [a * 100, (1 - a) * 100]) if boots else (np.nan, np.nan))
    n_inj = len(sub.drop_duplicates(["paper", "copy", "module"]))
    return rate, lo, hi, len(sub), n_inj, len(by_paper)


def matrix(df: pd.DataFrame, title: str, ci: float, diagnostics: bool) -> None:
    all_papers = sorted(df.paper.unique())
    print(f"\n{'='*88}")
    print(f"{title}   ({len(df)} verdicts · "
          f"{len(df.drop_duplicates(['paper','copy','module']))} injections · "
          f"{len(all_papers)} papers)")
    print(f"detection rate [{int(ci*100)}% paper-level bootstrap CI] "
          f"(n verdicts · n distinct injections · n papers)")
    print(f"\n  {'':<18}" + "".join(f"{'worst neighbour = '+str(c):>22}" for c in (1, 2, 3)))
    for s in (1, 2, 3):
        row = f"  {SEV_LABEL[s]:<18}"
        for c in (1, 2, 3):
            r, lo, hi, n, ni, npap = boot_cell(df[(df.sev == s) & (df.co_max == c)],
                                               all_papers, ci=ci)
            # Fewer than 3 papers give no usable between-paper spread.
            cell = ("-" if np.isnan(r) else
                    f"{100*r:.0f}% [n/a]" if npap < 3 else
                    f"{100*r:.0f}% [{100*lo:.0f},{100*hi:.0f}]")
            row += f"{cell:>14}{f'({n}·{ni}·{npap})':>8}"
        print(row)

    print(f"\n  {'ALL severities':<18}", end="")
    for c in (1, 2, 3):
        r, lo, hi, n, ni, npap = boot_cell(df[df.co_max == c], all_papers, ci=ci)
        cell = "-" if np.isnan(r) else f"{100*r:.0f}% [{100*lo:.0f},{100*hi:.0f}]"
        print(f"{cell:>14}{f'({n}·{ni}·{npap})':>8}", end="")
    print()

    print("\n  CONTRAST  worst neighbour 3 minus 2 (the comparison with enough errors behind it):")
    rng = np.random.default_rng(20260427)
    papers = np.array(all_papers)
    for s in (1, 2, 3, None):
        d = df if s is None else df[df.sev == s]
        a2, a3 = d[d.co_max == 2], d[d.co_max == 3]
        if a2.empty or a3.empty:
            continue
        obs = a3.detected.mean() - a2.detected.mean()
        g2 = {p: g.detected.to_numpy() for p, g in a2.groupby("paper")}
        g3 = {p: g.detected.to_numpy() for p, g in a3.groupby("paper")}
        bs = []
        for _ in range(2000):
            samp = rng.choice(papers, size=len(papers), replace=True)
            v2 = [g2[q] for q in samp if q in g2]
            v3 = [g3[q] for q in samp if q in g3]
            if v2 and v3:
                bs.append(np.concatenate(v3).mean() - np.concatenate(v2).mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lab = "all targets" if s is None else SEV_LABEL[s]
        flag = "EXCLUDES 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"    {lab:<16} {100*obs:+6.1f} pp   95% CI [{100*lo:+6.1f}, {100*hi:+6.1f}]   {flag}")

    if diagnostics:
        print("\n  Error-type mix per worst-neighbour column:")
        u = df.drop_duplicates(["paper", "copy", "module"])
        t = pd.crosstab(u.module, u.co_max)
        print("    " + t.to_string().replace("\n", "\n    "))


def fit(df: pd.DataFrame, label: str) -> None:
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    print(f"\n  --- logit, clustered by paper : {label} ---")
    f = "detected ~ co_max + C(sev) + C(module) + C(system) + C(paper)"
    try:
        groups = df["paper"].astype("category").cat.codes.to_numpy()
        m = smf.logit(f, data=df).fit(disp=False, maxiter=200, cov_type="cluster",
                                      cov_kwds={"groups": groups})
        b, se = m.params["co_max"], m.bse["co_max"]
        z = b / se
        pv = 2 * (1 - norm.cdf(abs(z)))
        print(f"    worst-neighbour coefficient (log-odds) : {b:+.4f}  SE {se:.4f}  z {z:+.2f}  p {pv:.3f}")
        print(f"    odds ratio per +1 severity             : {np.exp(b):.3f} "
              f"[{np.exp(b-1.96*se):.3f}, {np.exp(b+1.96*se):.3f}]")
    except Exception as e:                                   # noqa: BLE001
        print(f"    logit failed: {type(e).__name__}: {e}")
        return

    rng = np.random.default_rng(20260427)
    papers = df.paper.unique()
    boots = []
    for _ in range(400):
        samp = rng.choice(papers, size=len(papers), replace=True)
        d = pd.concat([df[df.paper == q].assign(paper=f"{q}_{k}")
                       for k, q in enumerate(samp)], ignore_index=True)
        try:
            mm = smf.logit(f, data=d).fit(disp=False, maxiter=200)
            boots.append(mm.params["co_max"])
        except Exception:                                    # noqa: BLE001
            continue
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"    paper-level bootstrap 95% CI           : [{lo:+.4f}, {hi:+.4f}] "
              f"({len(boots)} successful fits)  -> "
              f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")


def fit_interaction(df, label: str) -> None:
    import statsmodels.formula.api as smf
    from scipy.stats import norm
    print(f"\n  --- INTERACTION worst neighbour x own severity : {label} ---")
    f = "detected ~ co_max * C(sev) + C(module) + C(system) + C(paper)"
    try:
        g = df["paper"].astype("category").cat.codes.to_numpy()
        m = smf.logit(f, data=df).fit(disp=False, maxiter=300, cov_type="cluster",
                                      cov_kwds={"groups": g})
        for term in [t for t in m.params.index if "co_max" in t]:
            b, se = m.params[term], m.bse[term]
            p = 2 * (1 - norm.cdf(abs(b / se)))
            print(f"    {term_name(term):<40} {b:+.4f}  SE {se:.4f}  p {p:.3f}")
    except Exception as e:                                    # noqa: BLE001
        print(f"    interaction fit failed: {type(e).__name__}: {e}")

    print(f"\n  --- WITHIN EACH OWN SEVERITY : {label} ---")
    for s in (1, 2, 3):
        sub = df[df.sev == s]
        if sub.detected.nunique() < 2 or len(sub) < 50:
            print(f"    S{s}: too few rows / no variation")
            continue
        try:
            g = sub["paper"].astype("category").cat.codes.to_numpy()
            mm = smf.logit("detected ~ co_max + C(module) + C(system) + C(paper)",
                           data=sub).fit(disp=False, maxiter=300, cov_type="cluster",
                                         cov_kwds={"groups": g})
            b, se = mm.params["co_max"], mm.bse["co_max"]
            p = 2 * (1 - norm.cdf(abs(b / se)))
            rng = np.random.default_rng(20260427)
            ps = sub.paper.unique()
            bs = []
            for _ in range(300):
                samp = rng.choice(ps, size=len(ps), replace=True)
                d = pd.concat([sub[sub.paper == q].assign(paper=f"{q}_{k}")
                               for k, q in enumerate(samp)], ignore_index=True)
                try:
                    bs.append(smf.logit("detected ~ co_max + C(module) + C(system) + C(paper)",
                                        data=d).fit(disp=False, maxiter=300).params["co_max"])
                except Exception:                             # noqa: BLE001
                    continue
            ci = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
            print(f"    S{s} targets (n={len(sub)}): worst neighbour {b:+.4f} SE {se:.4f} "
                  f"p {p:.3f}  OR {np.exp(b):.3f}  boot95 [{ci[0]:+.3f}, {ci[1]:+.3f}] "
                  f"-> {'EXCLUDES 0' if (ci[0] > 0 or ci[1] < 0) else 'includes 0'}")
        except Exception as e:                                # noqa: BLE001
            print(f"    S{s}: fit failed: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--no-logit", action="store_true", help="skip the logistic models (slow)")
    a = ap.parse_args()

    frames = {c: build(c) for c in ("astro", "ml")}

    astro = frames["astro"].assign(paper="A:" + frames["astro"].paper)
    ml = frames["ml"].assign(paper="M:" + frames["ml"].paper)
    pooled = pd.concat([astro, ml], ignore_index=True)
    matrix(pooled, "POOLED — astrophysics + ML", a.ci, a.diagnostics)
    matrix(astro, "astrophysics only", a.ci, False)
    matrix(ml, "ML only", a.ci, False)

    if not a.no_logit:
        import statsmodels.formula.api  # noqa: F401  (import first: it installs its own filter)
        warnings.filterwarnings("ignore")
        for corpus in ("astro", "ml"):
            df = frames[corpus]
            print(f"\n{'='*74}")
            print(f"{corpus.upper()} — {len(df)} verdicts, {df.paper.nunique()} papers, "
                  f"{df.system.nunique()} systems")
            print(f"  overall detection {100*df.detected.mean():.1f}%")
            fit(df, corpus)
            fit_interaction(df, corpus)
    print("\nRow = one adjudicated verdict. Worst neighbour = highest severity of the other three")
    print("errors in the same copy. Controls: own severity, error type, system, paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
