#!/usr/bin/env python3
"""Appendix J and Section 3.1: citation detection across the two corpora.

Reads the group of each citation error from analysis/derived/citation_error_groups.csv
(field_knowledge, needs_cited_paper or either) and prints the group composition,
the pooled and re-weighted structured single-LLM rates, Skepthical's rates, and the
per-provider and per-system citation rates.

Usage:  python analysis/appJ_citations_transfer.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPORA, DATA, PROVIDER_MODEL, PROVIDERS, REPO, SYSTEMS, iter_verdicts  # noqa: E402

GROUPS_CSV = REPO / "analysis" / "derived" / "citation_error_groups.csv"
POOLED_NAME = {"L1": "Single LLM (free-form)", "L2": "Single LLM (structured)",
               "L25": "Single LLM (concise structured)", "L3": "Skepthical",
               "C": "Claude Code (structured)", "CL25": "Claude Code (concise structured)",
               "B2": "OpenReviewer-8B", "B3": "PaperAudit-Bench Deep"}
NW = max(map(len, POOLED_NAME.values())) + 2
GROUPS = ("field_knowledge", "either", "needs_cited_paper")
SHORT = {"field_knowledge": "FIELD", "either": "EITHER", "needs_cited_paper": "CITED"}


def load_groups():
    """(paper, copy) -> group, and (paper, copy) -> mutation kind of the citation error."""
    grp = {}
    with open(GROUPS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            grp[(r["paper"], r["copy"])] = r["group"]
    mut = {}
    for name in ("astro", "ml"):
        for p in CORPORA[name]:
            for c in ("v1", "v2", "v3"):
                inj = json.loads((DATA / p / c / "injections.json").read_text())
                inj = inj if isinstance(inj, list) else inj.get("injections", [])
                cit = [r for r in inj if r.get("module") == "citations"]
                assert len(cit) == 1, (p, c)
                mut[(p, c)] = (cit[0].get("truth") or {}).get("mutation_kind", "?")
                assert (p, c) in grp, f"no group for {(p, c)}"
    return grp, mut


def citation_verdicts(name):
    for s, tag, p, c, v in iter_verdicts(CORPORA[name]):
        if s == "L3" and tag:
            continue
        if (v.get("injection_ref") or "").startswith("citations"):
            yield s, tag, p, c, bool(v.get("detected"))


def pct(a, n):
    return f"{100*a/n:5.1f}% ({a}/{n})" if n else "     -"


def main() -> int:
    grp, mut = load_groups()

    bycls = collections.defaultdict(lambda: [0, 0])
    bymut = collections.defaultdict(lambda: [0, 0])
    bysys = collections.defaultdict(lambda: [0, 0])
    for name in ("astro", "ml"):
        for s, tag, p, c, det in citation_verdicts(name):
            k = (p, c)
            for d, key in ((bycls, (name, s, grp[k])), (bymut, (name, s, mut[k])),
                           (bysys, (name, s, tag))):
                d[key][0] += det
                d[key][1] += 1

    print("=" * 78)
    print("1. COMPOSITION - citation errors per group (FIELD = field knowledge suffices,")
    print("   CITED = the cited paper is needed, EITHER = could fall either way)")
    weights = {}
    for name in ("astro", "ml"):
        c = collections.Counter(grp[(p, cp)] for p in CORPORA[name] for cp in ("v1", "v2", "v3"))
        n = sum(c.values())
        weights[name] = {g: c[g] / n for g in GROUPS}
        print(f"   {name:<6} FIELD {c['field_knowledge']:>3}/{n}  "
              f"({100*c['field_knowledge']/n:4.0f}%)   EITHER {c['either']:>3}   "
              f"CITED {c['needs_cited_paper']:>3}")

    print("\n" + "=" * 78)
    print("2. DETECTION BY GROUP")
    print(f"   {'corpus':<7}{'system':<{NW}}{'FIELD':>17}{'EITHER':>17}{'CITED':>17}")
    for name in ("astro", "ml"):
        for s in ("L1", "L2", "L25", "L3", "C", "CL25", "B2", "B3"):
            if not any(bycls[(name, s, g)][1] for g in GROUPS):
                continue
            row = f"   {name:<7}{POOLED_NAME[s]:<{NW}}"
            for g in GROUPS:
                row += f"{pct(*bycls[(name, s, g)]):>17}"
            print(row)

    print("\n" + "=" * 78)
    print("3. RE-WEIGHTING - ML rates within each group, weighted by the astrophysics")
    print("   group proportions")
    wA = weights["astro"]
    for s in ("L2", "L3"):
        num_a = sum(bycls[("astro", s, g)][0] for g in GROUPS)
        den_a = sum(bycls[("astro", s, g)][1] for g in GROUPS)
        num_m = sum(bycls[("ml", s, g)][0] for g in GROUPS)
        den_m = sum(bycls[("ml", s, g)][1] for g in GROUPS)
        if not den_a or not den_m:
            continue
        obs_a, obs_m = num_a / den_a, num_m / den_m
        std = sum(wA[g] * (bycls[("ml", s, g)][0] / bycls[("ml", s, g)][1])
                  for g in GROUPS if bycls[("ml", s, g)][1])
        print(f"   {POOLED_NAME[s]}: astro {100*obs_a:5.1f}%  ->  ML {100*obs_m:5.1f}%   "
              f"(ML re-weighted to astro mix: {100*std:5.1f}%)")
        print(f"       composition {100*(obs_m-std):+5.1f} pp | "
              f"within-group {100*(std-obs_a):+5.1f} pp | "
              f"composition share of the change {100*(obs_m-std)/(obs_m-obs_a):4.1f}%")
        for g in GROUPS:
            a, m = bycls[("astro", s, g)], bycls[("ml", s, g)]
            ra, rm = a[0] / a[1], m[0] / m[1]
            print(f"       {SHORT[g]:<7} astro {100*ra:5.1f}% -> ML {100*rm:5.1f}%  "
                  f"({100*(rm-ra):+5.1f} pp; weighted contribution "
                  f"{100*wA[g]*(rm-ra):+5.1f} pp)")

    print("\n" + "=" * 78)
    print("4. MUTATION KIND - same mutation type in both corpora (does not use the groups)")
    for s in ("L2", "L3"):
        print(f"\n   --- {POOLED_NAME[s]} ---")
        print(f"   {'mutation kind':<46}{'astro':>15}{'ml':>15}")
        kinds = sorted({k[2] for k in bymut if k[1] == s})
        for kd in kinds:
            print(f"   {kd:<46}{pct(*bymut[('astro', s, kd)]):>15}"
                  f"{pct(*bymut[('ml', s, kd)]):>15}")

    print("\n" + "=" * 78)
    print("5. CITATION DETECTION RATE PER SYSTEM (Table 2 systems)")
    print(f"   {'system':<40}{'astro':>17}{'ml':>17}")
    for s, tag, label in SYSTEMS:
        print(f"   {label:<40}{pct(*bysys[('astro', s, tag)]):>17}"
              f"{pct(*bysys[('ml', s, tag)]):>17}")
    print("\n   structured single LLMs, per provider:")
    for prov in PROVIDERS:
        a, m = bysys[("astro", "L2", prov)], bysys[("ml", "L2", prov)]
        print(f"   {PROVIDER_MODEL[prov]:<40}{pct(*a):>17}{pct(*m):>17}"
              f"   {'higher on ML' if m[0]/m[1] > a[0]/a[1] else 'not higher on ML'}")
    others = [(bysys[("astro", s, t)][0] / bysys[("astro", s, t)][1], lab)
              for s, t, lab in SYSTEMS if s != "L3" and bysys[("astro", s, t)][1]]
    best = max(others)
    print(f"\n   astro, highest system other than Skepthical: {100*best[0]:.1f}% ({best[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
