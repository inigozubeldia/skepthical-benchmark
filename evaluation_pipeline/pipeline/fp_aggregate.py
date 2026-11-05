"""Aggregate per-paper false-positive reports into FP rates, judge agreement and cost."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_all_reports(data_dir: Path) -> List[Dict[str, Any]]:
    reports = []
    for fp_path in sorted(data_dir.glob("*/fp/*_v0.json")):
        try:
            with open(fp_path, encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception as exc:
            print(f"[fp_aggregate] could not load {fp_path}: {exc}")
    return reports


def _cell_key(report: Dict[str, Any]) -> str:
    rs = report.get("reviewer_system", "unknown")
    tag = report.get("provider_tag") or None
    return f"{rs}@{tag}" if tag else rs


def aggregate_fp_rates(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-(reviewer system x provider, module) FP rate across all papers.

        Returns {cell: {module: {"fp_count", "total", "fp_rate"}}}, with cell
        `<system>@<provider>` for panel reports and `<system>` otherwise.
    """
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"fp_count": 0, "total": 0, "unadj_count": 0})
    )

    for report in reports:
        cell = _cell_key(report)
        for mod, verdicts in report.get("per_module_verdicts", {}).items():
            for v in verdicts:
                verdict = v.get("verdict", "UNADJUDICATABLE")
                counts[cell][mod]["total"] += 1
                if verdict == "FP":
                    counts[cell][mod]["fp_count"] += 1
                if verdict == "UNADJUDICATABLE":
                    counts[cell][mod]["unadj_count"] += 1

    result: Dict[str, Any] = {}
    for cell, modules in counts.items():
        result[cell] = {}
        for mod, c in modules.items():
            adjudicated = c["total"] - c["unadj_count"]
            result[cell][mod] = {
                "fp_count": c["fp_count"],
                "confirmed_count": adjudicated - c["fp_count"],
                "unadj_count": c["unadj_count"],
                "total": c["total"],
                "fp_rate": c["fp_count"] / adjudicated if adjudicated > 0 else None,
            }
    return result


def _cohen_kappa(a: List[str], b: List[str]) -> Optional[float]:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    categories = list(set(a) | set(b))
    observed_agree = sum(1 for x, y in zip(a, b) if x == y) / n
    p_e = sum(
        (a.count(cat) / n) * (b.count(cat) / n)
        for cat in categories
    )
    if p_e == 1.0:
        return 1.0
    return (observed_agree - p_e) / (1.0 - p_e)


def compute_general_kappa(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pairwise Cohen's kappa between judges on general-issue flags."""
    judge_verdicts: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    flag_order: List[str] = []

    for report in reports:
        for mod, verdicts in report.get("per_module_verdicts", {}).items():
            if mod != "general":
                continue
            for v in verdicts:
                fid = v.get("flag_id", "")
                flag_order.append(fid)
                for sv in v.get("judge_panel", []):
                    key = (sv.get("judge_provider", ""), sv.get("judge_model", ""))
                    judge_verdicts[key].append(sv.get("verdict", "AMBIGUOUS"))

    judges = list(judge_verdicts.keys())
    kappa_table: Dict[str, float] = {}
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            v1 = judge_verdicts[j1]
            v2 = judge_verdicts[j2]
            min_len = min(len(v1), len(v2))
            if min_len == 0:
                continue
            k = _cohen_kappa(v1[:min_len], v2[:min_len])
            label = f"{j1[0]}:{j1[1]} vs {j2[0]}:{j2[1]}"
            kappa_table[label] = round(k, 4) if k is not None else None

    return {"inter_panel_kappa": kappa_table, "n_flags": len(flag_order)}


def aggregate_cost(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_run: List[Dict[str, Any]] = []
    grand_total = 0.0
    grand_extractor = 0.0
    grand_per_module: Dict[str, float] = defaultdict(float)
    grand_per_cell: Dict[str, float] = defaultdict(float)
    for report in reports:
        total = float(report.get("total_cost_usd", 0.0))
        extractor = float(report.get("extractor_cost_usd", 0.0))
        per_module = {k: float(v) for k, v in report.get("per_module_cost_usd", {}).items()}
        cell = _cell_key(report)
        per_run.append({
            "paper_id": report.get("paper_id", ""),
            "reviewer_system": report.get("reviewer_system", ""),
            "provider_tag": report.get("provider_tag"),
            "total_cost_usd": total,
            "extractor_cost_usd": extractor,
            "per_module_cost_usd": per_module,
        })
        grand_total += total
        grand_extractor += extractor
        grand_per_cell[cell] += total
        for mod, c in per_module.items():
            grand_per_module[mod] += c
    return {
        "grand_total_cost_usd": grand_total,
        "grand_extractor_cost_usd": grand_extractor,
        "grand_per_module_cost_usd": dict(grand_per_module),
        "grand_per_cell_cost_usd": dict(grand_per_cell),
        "per_run": per_run,
    }


def build_summary_table(fp_rates: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for rs, modules in sorted(fp_rates.items()):
        for mod, stats in sorted(modules.items()):
            rows.append({
                "reviewer_system": rs,
                "module": mod,
                **stats,
            })
    return rows


def run_aggregate(
    data_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = load_all_reports(data_dir)

    if not reports:
        print("[fp_aggregate] no FP reports found under", data_dir)
        return {}

    fp_rates = aggregate_fp_rates(reports)
    summary = build_summary_table(fp_rates)
    kappa = compute_general_kappa(reports)
    cost = aggregate_cost(reports)

    (output_dir / "fp_rates.json").write_text(
        json.dumps(fp_rates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "fp_summary_table.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "general_kappa.json").write_text(
        json.dumps(kappa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "fp_cost.json").write_text(
        json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[fp_aggregate] wrote results to {output_dir} — "
        f"grand total cost: ${cost['grand_total_cost_usd']:.4f}"
    )
    return {"fp_rates": fp_rates, "kappa": kappa, "cost": cost, "n_reports": len(reports)}
