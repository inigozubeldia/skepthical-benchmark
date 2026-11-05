"""Appendix E: adjudication-panel family-pairing test (Tables 6-8) and judge removal.

Uses first-round votes only. Intervals are 95 % paper-level bootstrap intervals
(10,000 resamples, seed 20260506; judge removal uses seed 20260507).

Delta_s = r_self - mean(r_f of the other two judges), where r_f is judge family f's
detection rate on system s; Delta_tilde_s subtracts each judge's pooled rate first.
Judge removal: majority of the two remaining votes; a 1-1 tie is resolved by the
escalated Anthropic vote when available, else counted as not detected ("strict")
or detected ("lenient").

Usage: python analysis/appE_family_pairing.py [astro|ml]   (default: both)
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys

from common import CORPORA, CORPUS_NAMES, DATA

BOOT_B = 10_000
BOOT_SEED = 20260506
BOOT_ALPHA = 0.05

FAMILIES = ("anthropic", "openai", "gemini")
FAM_SHORT = {"anthropic": "A", "openai": "O", "gemini": "G"}
FAM_NAME = {"anthropic": "Anthropic", "openai": "OpenAI", "gemini": "Google"}
MODULES = ("citations", "numerics", "mathematics", "general")
COPIES = ("v1", "v2", "v3")

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

# Single-family systems; Skepthical and the external reviewers are excluded from Delta.
SELF_FAMILY = {
    "Claude Opus 4.7 (free-form)": "anthropic",
    "GPT-5.5 (free-form)": "openai",
    "Gemini 3.1 Pro (free-form)": "gemini",
    "Claude Opus 4.7 (structured)": "anthropic",
    "GPT-5.5 (structured)": "openai",
    "Gemini 3.1 Pro (structured)": "gemini",
    "Claude Code (structured)": "anthropic",
    "Claude Code (free-form)": "anthropic",
}

REGIMES = [f"drop_{d}_{t}" for d in FAMILIES for t in ("strict", "lenient")]


def regime_name(r):
    _, fam, tie = r.split("_")
    return f"no {FAM_NAME[fam]} judge, {tie}"


def verdicts(paper, system, tag, copy):
    """Verdict list of one adjudicated review, or None if the file is absent."""
    name = f"{copy}__{tag}.json" if tag else f"{copy}.json"
    f = DATA / paper / "verifications" / system / name
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("verdicts", [])


def first_round_votes(panel):
    """{family: detected} from the first-round (non-escalated) votes."""
    out = {}
    for j in panel:
        if j.get("escalated_with_pdf"):
            continue
        fam = j.get("judge_provider")
        if fam in FAMILIES and fam not in out:
            out[fam] = bool(j.get("detected"))
    return out


def escalated_vote(panel):
    for j in panel:
        if j.get("escalated_with_pdf"):
            return bool(j.get("detected"))
    return None


def verdict_without(panel, drop, strict_tie):
    """Majority of the two first-round votes left after removing family `drop`."""
    fr = first_round_votes(panel)
    keep = [v for f, v in fr.items() if f != drop]
    if len(keep) < 2:
        return None
    n_det = sum(keep)
    if n_det in (0, 2):
        return bool(n_det)
    if drop != "anthropic":
        ev = escalated_vote(panel)
        if ev is not None:
            return ev
    return not strict_tie


def percentile_ci(values, alpha=BOOT_ALPHA):
    s = sorted(v for v in values if v == v)
    n = len(s)
    if n == 0:
        return (float("nan"), float("nan"))
    lo_i = max(0, int(round(alpha / 2 * n)) - 1)
    hi_i = min(n - 1, int(round((1 - alpha / 2) * n)) - 1)
    return (s[lo_i], s[hi_i])


def z(x):
    """Print a floating-point residue such as -1e-15 as zero."""
    return 0.0 if abs(x) < 1e-9 else x


def binom_le(k, n, p=0.5):
    """One-sided exact binomial P(X <= k)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def collect_removal(papers):
    """{label: {"rows": [...], "n", "baseline", "removed": {regime: rate}, ...}}."""
    data = {}
    for system, tag, label in SYSTEMS:
        rows = []
        for p in papers:
            for copy in COPIES:
                vs = verdicts(p, system, tag, copy)
                if vs is None:
                    continue
                for v in vs:
                    jp = v.get("judge_panel") or []
                    row = {"paper": p, "baseline": bool(v.get("detected"))}
                    for d in FAMILIES:
                        row[f"drop_{d}_strict"] = verdict_without(jp, d, strict_tie=True)
                        row[f"drop_{d}_lenient"] = verdict_without(jp, d, strict_tie=False)
                    rows.append(row)
        if not rows:
            continue

        def rate(key):
            xs = [r[key] for r in rows if r[key] is not None]
            return sum(xs) / len(xs) * 100.0 if xs else float("nan")

        rec = {"rows": rows, "n": len(rows), "baseline": rate("baseline"),
               "removed": {k: rate(k) for k in REGIMES}}
        for t in ("strict", "lenient"):
            rec[f"max_change_{t}"] = max(abs(rec["removed"][f"drop_{d}_{t}"] - rec["baseline"])
                                         for d in FAMILIES)
        data[label] = rec
    return data


def ranking(data, regime):
    get = (lambda r: r["baseline"]) if regime == "baseline" else (lambda r: r["removed"][regime])
    return sorted(((lab, get(rec)) for lab, rec in data.items()), key=lambda kv: -kv[1])


def bootstrap_removal(data, papers, B=BOOT_B, seed=BOOT_SEED + 1):
    keys = ["baseline"] + REGIMES
    agg = {}
    for label, rec in data.items():
        by_p = {}
        for r in rec["rows"]:
            by_p.setdefault(r["paper"], []).append(r)
        agg[label] = {p: {k: (sum(1 for r in by_p.get(p, []) if r[k] is not None),
                              sum(1 for r in by_p.get(p, []) if r[k]))
                          for k in keys} for p in papers}
    rng = random.Random(seed)
    n_p = len(papers)
    boot = {label: {k: [] for k in keys + ["max_change_strict", "max_change_lenient"]}
            for label in data}
    for _ in range(B):
        sample = [papers[rng.randrange(n_p)] for _ in range(n_p)]
        for label in data:
            rates = {}
            for k in keys:
                ne = sum(agg[label][p][k][0] for p in sample)
                ds = sum(agg[label][p][k][1] for p in sample)
                rates[k] = 100.0 * ds / ne if ne else float("nan")
                boot[label][k].append(rates[k])
            base = rates["baseline"]
            for t in ("strict", "lenient"):
                boot[label][f"max_change_{t}"].append(
                    max(abs(rates[f"drop_{d}_{t}"] - base) for d in FAMILIES)
                    if base == base else float("nan"))
    return {label: {k: percentile_ci(v) for k, v in b.items()} for label, b in boot.items()}


def paper_summaries(papers):
    summ = {}
    for system, tag, label in SYSTEMS:
        sf = SELF_FAMILY.get(label)
        sf_i = FAMILIES.index(sf) if sf is not None else None
        summ[label] = {}
        for p in papers:
            trips = []
            for copy in COPIES:
                for v in verdicts(p, system, tag, copy) or []:
                    fr = first_round_votes(v.get("judge_panel") or [])
                    if all(f in fr for f in FAMILIES):
                        trips.append(tuple(fr[f] for f in FAMILIES))
            s = {"n": len(trips), "det": [sum(t[i] for t in trips) for i in range(3)],
                 "n_split": 0, "same_det": 0, "same_nondet": 0,
                 "other_det": 0, "other_nondet": 0}
            if sf_i is not None:
                for t in trips:
                    k = sum(t)
                    if k not in (1, 2):
                        continue
                    s["n_split"] += 1
                    dissenter = t.index(False) if k == 2 else t.index(True)
                    kind = "det" if k == 2 else "nondet"
                    who = "same" if dissenter == sf_i else "other"
                    s[f"{who}_{kind}"] += 1
            summ[label][p] = s
    return summ


def pairing_stats(summ, sample):
    tot_n, tot_det, per_lab = 0, [0, 0, 0], {}
    for label, by_p in summ.items():
        n = sum(by_p[p]["n"] for p in sample)
        det = [sum(by_p[p]["det"][i] for p in sample) for i in range(3)]
        per_lab[label] = (n, det)
        tot_n += n
        tot_det = [a + b for a, b in zip(tot_det, det)]
    pooled = {f: (100.0 * tot_det[i] / tot_n if tot_n else float("nan"))
              for i, f in enumerate(FAMILIES)}
    out = {"pooled": pooled, "pooled_n": tot_n, "per_system": {}, "raw": [], "adj": [],
           "split": 0, "same": 0, "split_det": 0, "same_det": 0,
           "split_nondet": 0, "same_nondet": 0}
    for label, sf in SELF_FAMILY.items():
        n, det = per_lab[label]
        if n == 0:
            continue
        rates = {f: 100.0 * det[i] / n for i, f in enumerate(FAMILIES)}
        others = [f for f in FAMILIES if f != sf]
        raw = rates[sf] - sum(rates[f] for f in others) / 2.0
        adj = (rates[sf] - pooled[sf]) - sum(rates[f] - pooled[f] for f in others) / 2.0
        out["per_system"][label] = {"n": n, "rates": rates, "raw": raw, "adj": adj}
        out["raw"].append(raw)
        out["adj"].append(adj)
        c = {k: sum(summ[label][p][k] for p in sample)
             for k in ("n_split", "same_det", "same_nondet", "other_det", "other_nondet")}
        out["split"] += c["n_split"]
        out["same"] += c["same_det"] + c["same_nondet"]
        out["split_det"] += c["same_det"] + c["other_det"]
        out["same_det"] += c["same_det"]
        out["split_nondet"] += c["same_nondet"] + c["other_nondet"]
        out["same_nondet"] += c["same_nondet"]
    for suffix in ("", "_det", "_nondet"):
        d = out["split" + suffix]
        out["share" + suffix] = 100.0 * out["same" + suffix] / d if d else float("nan")
    return out


def bootstrap_pairing(summ, papers, B=BOOT_B, seed=BOOT_SEED):
    rng = random.Random(seed)
    n_p = len(papers)
    point = pairing_stats(summ, papers)
    point["mean_raw"] = statistics.mean(point["raw"])
    point["mean_adj"] = statistics.mean(point["adj"])
    point["n_pos_raw"] = sum(d > 0 for d in point["raw"])
    point["n_pos_adj"] = sum(d > 0 for d in point["adj"])
    boot = {"pooled": {f: [] for f in FAMILIES},
            "raw": {lab: [] for lab in SELF_FAMILY}, "adj": {lab: [] for lab in SELF_FAMILY},
            "mean_raw": [], "mean_adj": [], "share": [], "share_det": [], "share_nondet": []}
    for _ in range(B):
        sample = [papers[rng.randrange(n_p)] for _ in range(n_p)]
        r = pairing_stats(summ, sample)
        for f in FAMILIES:
            boot["pooled"][f].append(r["pooled"][f])
        for lab in SELF_FAMILY:
            ps = r["per_system"].get(lab)
            boot["raw"][lab].append(ps["raw"] if ps else float("nan"))
            boot["adj"][lab].append(ps["adj"] if ps else float("nan"))
        if r["raw"]:
            boot["mean_raw"].append(statistics.mean(r["raw"]))
        if r["adj"]:
            boot["mean_adj"].append(statistics.mean(r["adj"]))
        for k in ("share", "share_det", "share_nondet"):
            boot[k].append(r[k])
    cis = {"pooled": {f: percentile_ci(boot["pooled"][f]) for f in FAMILIES},
           "raw": {lab: percentile_ci(v) for lab, v in boot["raw"].items()},
           "adj": {lab: percentile_ci(v) for lab, v in boot["adj"].items()}}
    for k in ("mean_raw", "mean_adj", "share", "share_det", "share_nondet"):
        cis[k] = percentile_ci(boot[k])
    return point, cis


def report(which):
    papers = list(CORPORA[which])
    print(f"\n{'=' * 78}\n{CORPUS_NAMES[which]} ({len(papers)} papers)\n{'=' * 78}")

    summ = paper_summaries(papers)
    pt, ci = bootstrap_pairing(summ, papers)
    print(f"\nTable 6: pooled first-round detection rate by judge "
          f"({pt['pooled_n']:,} injections with all three first-round votes, all reviewer systems)")
    for f in FAMILIES:
        lo, hi = ci["pooled"][f]
        print(f"  {FAM_NAME[f]:<10} {pt['pooled'][f]:6.2f} %  [{lo:.2f}, {hi:.2f}]")
    spread = max(pt["pooled"].values()) - min(pt["pooled"].values())
    print(f"  spread between judges: {spread:.2f} pp")

    print("\nTable 7: Delta (raw) and Delta-tilde (adjusted), pp; the letter is the reviewer's own "
          "family (A = Anthropic, O = OpenAI, G = Google)")
    for lab, sf in SELF_FAMILY.items():
        ps = pt["per_system"].get(lab)
        if ps is None:
            continue
        (rl, rh), (al, ah) = ci["raw"][lab], ci["adj"][lab]
        print(f"  {lab:<30} {FAM_SHORT[sf]}  raw {z(ps['raw']):+6.2f} [{z(rl):+.2f}, {z(rh):+.2f}]"
              f"  adj {z(ps['adj']):+6.2f} [{z(al):+.2f}, {z(ah):+.2f}]  n={ps['n']}")
    (rl, rh), (al, ah) = ci["mean_raw"], ci["mean_adj"]
    print(f"  {'Mean across the eight systems':<30}    raw {pt['mean_raw']:+6.2f} [{rl:+.2f}, {rh:+.2f}]"
          f"  adj {pt['mean_adj']:+6.2f} [{al:+.2f}, {ah:+.2f}]")
    n_excl = sum(1 for lab in SELF_FAMILY if lab in pt["per_system"]
                 and (z(ci["raw"][lab][0]) > 0 or z(ci["raw"][lab][1]) < 0))
    print(f"  per-system raw intervals excluding zero: {n_excl} of {len(pt['per_system'])}")
    print(f"  systems with Delta > 0: raw {pt['n_pos_raw']}/8 (one-sided binomial P = "
          f"{binom_le(pt['n_pos_raw'], 8):.3f}), adjusted {pt['n_pos_adj']}/8 (P = "
          f"{binom_le(pt['n_pos_adj'], 8):.3f})")

    print("\nTable 8: share of 2-1 splits whose lone dissenter is the same-family judge")
    for name, k, nk in (("all 2-1 splits", "share", "split"),
                        ("2-1 detected splits", "share_det", "split_det"),
                        ("1-2 not-detected splits", "share_nondet", "split_nondet")):
        lo, hi = ci[k]
        print(f"  {name:<26} {pt[k]:5.2f} %  [{lo:.2f}, {hi:.2f}]  n={pt[nk]}")

    data = collect_removal(papers)
    rci = bootstrap_removal(data, papers)
    print("\nJudge removal: detection rate (%) with one judge family removed from the first-round panel "
          "(A, O, G = the Anthropic, OpenAI, Google judge removed)")
    head = "".join(f"{FAM_SHORT[r.split('_')[1]] + ' ' + r.split('_')[2]:>11}" for r in REGIMES)
    print(f"  {'Reviewer':<30}{'n':>5}{'base':>7}{head}   largest change, strict [CI]")
    for lab, rec in data.items():
        lo, hi = rci[lab]["max_change_strict"]
        print(f"  {lab:<30}{rec['n']:>5}{rec['baseline']:>7.1f}"
              + "".join(f"{rec['removed'][r]:>11.1f}" for r in REGIMES)
              + f"   {rec['max_change_strict']:.2f} [{lo:.2f}, {hi:.2f}]")
    base_top = [lab for lab, _ in ranking(data, "baseline")[:3]]
    print(f"\n  baseline top 3: {base_top}")
    for r in REGIMES:
        top = [lab for lab, _ in ranking(data, r)[:3]]
        verdict = ("same systems, same order" if top == base_top else
                   "same systems, reordered" if set(top) == set(base_top) else "different systems")
        print(f"  {regime_name(r):<30} top 3: {top}  ({verdict})")
    sk = data["Skepthical"]
    print(f"\n  Skepthical: baseline {sk['baseline']:.2f} %")
    for r in REGIMES:
        print(f"    {regime_name(r):<30} {sk['removed'][r]:6.2f} %  change {sk['removed'][r] - sk['baseline']:+.2f} pp")
    lo, hi = rci["Skepthical"]["max_change_strict"]
    print(f"    largest change, strict ties:  {sk['max_change_strict']:.2f} pp  [{lo:.2f}, {hi:.2f}]")
    print(f"    largest change, lenient ties: {sk['max_change_lenient']:.2f} pp")
    for other in ("GPT-5.5 (structured)", "Claude Code (structured)"):
        o = data[other]
        leads = [f"baseline {sk['baseline'] - o['baseline']:+.2f}"] + [
            f"{regime_name(r)} {sk['removed'][r] - o['removed'][r]:+.2f}" for r in REGIMES]
        print(f"  Skepthical lead over {other} (pp): " + ", ".join(leads))


if __name__ == "__main__":
    for which in (sys.argv[1:] or ["astro", "ml"]):
        report(which)
