"""Section 2.4: agreement statistics of the detection-matching panel, per corpus and pooled.

    python analysis/adjudication_stats.py

Twelve systems of Table 2 only. Splits use first-round votes; a fourth vote in
`judge_panel` is the PDF-attached escalation judge, whose vote replaces the majority.
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

from common import CORPORA, DATA, PROVIDERS, VER_RE


def keep(system, tag):
    return ((system in ("L1", "L2") and tag in PROVIDERS) or (system == "L3" and tag == "")
            or (system == "B3" and tag == "deep") or (system in ("C", "CL1", "B1", "B2") and tag == ""))


def det(papers):
    c = Counter()
    models = Counter()
    cells = 0
    for p in papers:
        for f in glob.glob(str(DATA / p / "verifications" / "*" / "*.json")):
            system = Path(f).parent.name
            mm = VER_RE.match(Path(f).stem)
            if not mm or not keep(system, (mm.group(2) or "").lower()):
                continue
            cells += 1
            for v in json.loads(Path(f).read_text()).get("verdicts", []):
                jp = v.get("judge_panel") or []
                if len(jp) < 3:
                    c["short_panel"] += 1
                    continue
                first = [bool(j["detected"]) for j in jp[:3]]
                maj = sum(first) >= 2
                for j in jp:
                    models[(j["judge_provider"], j["judge_model"], bool(j.get("escalated_with_pdf")))] += 1
                c["n"] += 1
                c["unan3"] += (len(set(first)) == 1)
                esc = len(jp) == 4
                c["esc"] += esc
                c["unan_all"] += (len({bool(j["detected"]) for j in jp}) == 1)
                if esc:
                    e = jp[3]
                    conf = (bool(e["detected"]) == maj)
                    c["esc_confirm"] += conf
                    c["esc_high"] += (e.get("confidence") == "high")
                    c["reversed"] += (not conf)
                c["final_low"] += (v.get("confidence") == "low")
    return cells, c, models


if __name__ == "__main__":
    A, M = CORPORA["astro"], CORPORA["ml"]
    for name, papers in (("astro", A), ("ML", M), ("pooled", A + M)):
        cells, c, models = det(papers)
        n = c["n"]
        e = c["esc"]
        print(f"\n=== {name}: adjudicated reviews={cells}, adjudications={n} (short panels skipped: {c['short_panel']}) ===")
        print(f"unanimous 3-0 first round {100*c['unan3']/n:.1f}%  | 2-1 splits {100*(n-c['unan3'])/n:.1f}%  | escalation fires {100*e/n:.1f}% ({e})")
        print(f"of escalated: confirm majority {100*c['esc_confirm']/e:.1f}%, high confidence {100*c['esc_high']/e:.1f}%  | reversed overall {100*c['reversed']/n:.1f}%  -> accepted-or-confirmed {100*(n-c['reversed'])/n:.1f}%")
        print(f"unanimous across every judge that voted {100*c['unan_all']/n:.1f}%  | low-confidence final verdict {100*c['final_low']/n:.1f}%")
        print("judge models (provider, model, escalated):", dict(models))
