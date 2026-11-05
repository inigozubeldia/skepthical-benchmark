#!/usr/bin/env python3
"""Run the false-positive pipeline on one paper.

Usage
-----
python fp_verify.py --paper-id <arxiv_id> --system <code>
python fp_verify.py --paper-id <arxiv_id> --system L1 L2 --review-providers anthropic openai gemini
python fp_verify.py --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline.fp_aggregate import run_aggregate
from pipeline.fp_orchestrator import _load_paper_pdf, run_fp_pipeline
from pipeline.fp_schema import AdjudicationVerdict, ReviewerSystem
from pipeline.llm_client import delete_pdf_remote


_DATA_DIR = _ROOT / "data"
_FP_RESULTS_DIR = _ROOT / "fp_results"

_PANEL_SYSTEMS = {"L1", "L2", "L25"}
_MODE_SYSTEMS = {"B3"}
_PANEL_TAGS = {"anthropic", "openai", "gemini"}
_MODE_TAGS = {"fast", "standard", "deep"}


def _tags_for_system(system: str, requested_tags: list | None) -> list:
    if not requested_tags:
        return [None]
    if system in _PANEL_SYSTEMS:
        return [t for t in requested_tags if t in _PANEL_TAGS] or [None]
    if system in _MODE_SYSTEMS:
        return [t for t in requested_tags if t in _MODE_TAGS] or [None]
    return [None]


_DEFAULT_PARAMS: dict = {
    "models": {
        "maths_extractor": "gemini-3.1-pro-preview",
        "maths_transcription_checker": "gemini-3.1-pro-preview",
        "maths_interpreter": "claude-sonnet-4-6",
        "maths_aggregator": "gemini-3.1-pro-preview",
        "numerics_rereader": "gpt-4.1",
        "paper_verification_agent": "gpt-5",
        "reference_resolver": "gpt-5-mini",
    },
    "maths_parallel_workers": 10,
    "maths_wolfram_persistent": True,
}


def _get_api_keys() -> dict:
    return {
        "OPENAI": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC": os.environ.get("ANTHROPIC_API_KEY", ""),
        "GEMINI": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FP detection pipeline on a single paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--paper-id", help="arXiv ID of the paper (e.g. 2504.10230)")
    parser.add_argument(
        "--system", nargs="+",
        choices=["L1", "L2", "L25", "L3", "B1", "B2", "B3", "C", "CL1", "CL25"],
        help="Reviewer system(s) to adjudicate.",
    )
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR,
                        help="Root data directory (default: ../data/)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip per-paper runs; only aggregate existing results.")
    parser.add_argument("--params-json", type=Path,
                        help="Optional JSON file overriding default params_skepthical.")
    parser.add_argument(
        "--review-providers", nargs="+", default=None,
        help="Panel mode: adjudicate one review per provider tag. Reads "
             "reviews/<system>/v0__<provider>.md and writes "
             "fp/<system>__<provider>_v0.json. Standard panel values are "
             "'anthropic'/'openai'/'gemini'; B3 uses mode names like 'deep'.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help=("Re-run cells: delete-then-attempt.  Existing "
              "fp/<system>[__<tag>]_v0.json for the cell is removed "
              "BEFORE the new attempt, so a partial-fail re-run can "
              "never leave a stale-version mix on disk."),
    )
    args = parser.parse_args()

    if args.aggregate_only:
        result = run_aggregate(data_dir=args.data_dir, output_dir=_FP_RESULTS_DIR)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if not args.paper_id:
        parser.error("--paper-id is required unless --aggregate-only is set.")
    if not args.system:
        parser.error("--system is required unless --aggregate-only is set.")

    paper_dir = args.data_dir / args.paper_id
    if not paper_dir.exists():
        print(f"[fp_verify_paper] paper directory not found: {paper_dir}", file=sys.stderr)
        return 1

    params = dict(_DEFAULT_PARAMS)
    if args.params_json:
        with open(args.params_json, encoding="utf-8") as f:
            params.update(json.load(f))

    api_keys = _get_api_keys()

    general_cache: dict[str, AdjudicationVerdict] = {}
    citations_cache: dict[str, AdjudicationVerdict] = {}
    corpus_csv = _FP_RESULTS_DIR / "maths_manual_review_queue.csv"

    requested_tags = list(args.review_providers) if args.review_providers else None

    fp_dir = paper_dir / "fp"
    for system in args.system:
        for provider_tag in _tags_for_system(system, requested_tags):
            cell_label = f"{system}/{provider_tag}" if provider_tag else system

            review_md = paper_dir / "reviews" / system / (
                f"v0__{provider_tag}.md" if provider_tag else "v0.md"
            )
            if not review_md.is_file():
                print(f"=== FP pipeline: {args.paper_id} / {cell_label} skip "
                      f"(no review MD: {review_md.relative_to(paper_dir)}) ===",
                      flush=True)
                continue

            target = fp_dir / (
                f"{system}__{provider_tag}_v0.json" if provider_tag
                else f"{system}_v0.json"
            )
            if not args.force and target.is_file() and target.stat().st_size > 0:
                print(f"=== FP pipeline: {args.paper_id} / {cell_label} skip "
                      f"(exists: {target.relative_to(paper_dir)}) ===", flush=True)
                continue
            if args.force:
                target.unlink(missing_ok=True)
            print(f"\n=== FP pipeline: {args.paper_id} / {cell_label} ===", flush=True)
            report = run_fp_pipeline(
                paper_id=args.paper_id,
                paper_dir=paper_dir,
                reviewer_system=system,  # type: ignore[arg-type]
                params_skepthical=params,
                api_keys=api_keys,
                general_dedup_cache=general_cache,
                citations_dedup_cache=citations_cache,
                corpus_csv=corpus_csv,
                provider_tag=provider_tag,
            )
            if report is not None:
                print(json.dumps(report.as_dict(), indent=2, default=str))

    pdf_path = _load_paper_pdf(paper_dir)
    if pdf_path is not None:
        result = delete_pdf_remote(pdf_path, api_keys=api_keys)
        print(
            f"[fp_verify_paper] cleaned up remote PDF uploads — "
            f"openai={result['openai']}, gemini={result['gemini']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
