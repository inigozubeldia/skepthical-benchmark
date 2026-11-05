#!/usr/bin/env python3
"""Appendix I: which section of Skepthical's astrophysics reviews identifies each
injected citation, numerics and mathematics error.

Reads analysis/derived/module_attribution_astro.csv (one row per injected error).
A section counts only when its value is "yes". The panel verdicts in the file are
checked against the released Skepthical verdicts.

Usage:  python analysis/appI_attribution.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPORA, REPO, load_detections  # noqa: E402

CSV = REPO / "analysis" / "derived" / "module_attribution_astro.csv"
SECTION = {"citations": "citations_section", "mathematics": "maths_section",
           "numerics": "numerics_section"}


def main() -> int:
    rows = list(csv.DictReader(open(CSV, newline="")))
    assert len(rows) == 177, len(rows)

    D, inj = load_detections(CORPORA["astro"])
    l3 = D[("L3", "")]
    mism = [r for r in rows
            if l3.get((r["paper"], r["copy"], r["error_type"])) != (r["panel_detected"] == "yes")]
    print(f"rows: {len(rows)}; panel verdicts disagreeing with released Skepthical verdicts: {len(mism)}")
    tot = sum(l3.values())
    print(f"Skepthical astrophysics detections (released verdicts): {tot}/{len(l3)} "
          f"= {100*tot/len(l3):.1f}%\n")

    print(f"{'error type':<13}{'n':>4}{'review finds':>14}{'own section':>13}"
          f"{'own section, of review finds':>31}")
    for t in ("citations", "numerics", "mathematics"):
        R = [r for r in rows if r["error_type"] == t]
        found = [r for r in R if r["panel_detected"] == "yes"]
        n_rel = sum(v for (p, c, m), v in l3.items() if m == t)
        assert n_rel == len(found), (t, n_rel, len(found))
        sec = [r for r in R if r[SECTION[t]] == "yes"]
        both = [r for r in sec if r["panel_detected"] == "yes"]
        print(f"{t:<13}{len(R):>4}{len(found):>14}{len(sec):>13}"
              f"{f'{len(both)} of {len(found)} ({100*len(both)/len(found):.0f}%)':>31}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
