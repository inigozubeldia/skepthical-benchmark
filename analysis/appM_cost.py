"""Appendix M: Tables 12 and 13, from the per-call cost ledgers data/<id>/costs.jsonl.

Systems without per-call API charges (Claude Code, Stanford, OpenReviewer-8B,
PaperAudit-Bench) do not appear.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict

from common import CORPORA, DATA, REPO

STAGES = [("injection", "Error injection"), ("reviewer", "Reviewer execution"),
          ("det", "Detection adjudication"), ("fp", "False-positive adjudication"),
          ("other", "Other (eligibility scan, re-reading)")]


def stage(agent: str) -> str:
    a = agent.lower()
    if a.startswith("injector"):
        return "injection"
    if a.startswith("reviewer."):
        return "reviewer"
    if a.startswith("verifier"):
        return "det"
    if a.startswith("fp_") or a.startswith("reference_resolver"):
        return "fp"
    return "other"


def ledger(papers):
    rows = []
    for p in papers:
        f = DATA / p / "costs.jsonl"
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def table13():
    print("Table 13: complete logged API cost (USD) by pipeline stage")
    totals = {}
    by = {}
    for name in ("astro", "ml"):
        s = defaultdict(float)
        for r in ledger(CORPORA[name]):
            s[stage(r["agent_name"])] += r.get("cost_usd") or 0.0
        by[name] = s
        totals[name] = sum(s.values())
    print(f"  {'Stage':<38}{'Astrophysics':>14}{'ML':>10}")
    for key, label in STAGES:
        print(f"  {label:<38}{by['astro'][key]:>14.2f}{by['ml'][key]:>10.2f}")
    print(f"  {'Total':<38}{totals['astro']:>14.2f}{totals['ml']:>10.2f}")
    print(f"  {'Per paper':<38}{totals['astro'] / 15:>14.2f}{totals['ml'] / 15:>10.2f}")


def table12():
    print("\nTable 12 (astrophysics, hosted-API rows): reviewer-side cost")
    rows = [r for r in ledger(CORPORA["astro"]) if r["agent_name"].startswith("reviewer.")]
    l3 = [r["cost_usd"] for r in rows if r["agent_name"].split(".")[1] == "L3"]
    mean_l3 = sum(l3) / len(l3)
    print(f"  Skepthical: {mean_l3:.2f} USD per paper version -> corpus cost {60 * mean_l3:.0f}")
    for code, label in (("L2", "Single-LLM structured"), ("L1", "Single-LLM free-form")):
        total = sum(r["cost_usd"] for r in rows if r["agent_name"].split(".")[1] == code)
        per_provider = total / 3
        print(f"  {label}: three providers {total:.2f} USD; provider average {per_provider:.2f} "
              f"-> corpus cost {per_provider:.0f}, USD / paper {per_provider / 60:.2f}")


def wall_clock():
    print("\nTable 12 (astrophysics): median wall-clock per review")
    by = defaultdict(list)
    with open(REPO / "analysis" / "derived" / "review_wall_seconds_astro.csv") as f:
        for r in csv.DictReader(f):
            by[r["system"]].append(float(r["wall_seconds"]))
    labels = {"L3": "Skepthical", "L2": "Single-LLM structured", "L1": "Single-LLM free-form",
              "C": "Claude Code structured", "CL1": "Claude Code free-form",
              "B2": "OpenReviewer-8B", "B3": "PaperAudit-Bench detector"}
    for code, label in labels.items():
        med = statistics.median(by[code])
        print(f"  {label:<28} n={len(by[code]):3d}  median {med:7.1f} s  ({med / 60:.1f} min)")


if __name__ == "__main__":
    table13()
    table12()
    wall_clock()
