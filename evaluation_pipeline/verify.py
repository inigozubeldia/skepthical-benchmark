#!/usr/bin/env python3
"""Score reviewer flags against the injected errors.

For each (paper, injected copy, reviewer system) cell with a review on disk,
write one VerificationReport with a verdict per injection, judged by a
cross-family panel.

Review files may carry a `__<tag>` suffix: the panel provider for L1/L2
(anthropic / openai / gemini) or the run mode for B3 (e.g. `B3:deep`).
Outputs go to `verifications/<system>/<copy>[__<tag>].json`.

Usage
-----
    python verify.py <arxiv_id> --systems <code>
    python verify.py <arxiv_id> --copies v1 --systems L2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import cost as cost_mod  # noqa: E402
from pipeline.llm_client import delete_pdf_remote  # noqa: E402
from pipeline.verifier import verify_review, write_verification  # noqa: E402


DEFAULT_SYSTEMS = "L1,L2,L3,B2,B3:deep"
DEFAULT_PANEL_PROVIDERS = "anthropic,openai,gemini"
_PANEL_SYSTEMS = {"L1", "L2", "L25"}


def _parse_systems(spec: str) -> List[Tuple[str, Optional[str]]]:
    """L1,L2,B3:deep -> [(L1, None), (L2, None), (B3, deep)]."""
    out: List[Tuple[str, Optional[str]]] = []
    for cell in spec.split(","):
        cell = cell.strip()
        if not cell:
            continue
        if ":" in cell:
            sys_, sub = cell.split(":", 1)
            out.append((sys_.strip(), sub.strip()))
        else:
            out.append((cell, None))
    return out


def _expand_cells(
    systems: List[Tuple[str, Optional[str]]],
    panel_providers: Optional[List[str]],
) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    for sys_, sub in systems:
        if sys_ in _PANEL_SYSTEMS and panel_providers and sub is None:
            for prov in panel_providers:
                out.append((sys_, prov))
        else:
            out.append((sys_, sub))
    return out


def _verification_path(paper_dir: Path, system: str, copy_id: str,
                       tag: Optional[str]) -> Path:
    stem = copy_id if not tag else f"{copy_id}__{tag}"
    return paper_dir / "verifications" / system / f"{stem}.json"


def _review_md_path(paper_dir: Path, system: str, copy_id: str,
                    tag: Optional[str]) -> Path:
    stem = copy_id if not tag else f"{copy_id}__{tag}"
    return paper_dir / "reviews" / system / f"{stem}.md"


def run(
    *,
    arxiv_id: str,
    copies: List[str],
    cells: List[Tuple[str, Optional[str]]],
    uniform_n_judges: Optional[int],
    escalate_with_pdf: bool,
    force: bool,
) -> int:
    paper_dir = _ROOT / "data" / arxiv_id
    if not paper_dir.is_dir():
        print(f"[verify] paper dir not found: {paper_dir}", file=sys.stderr)
        return 2

    cost_mod.reset()
    successes: list = []
    failures: list = []
    pdfs_seen: List[Path] = []

    for copy_id in copies:
        copy_pdf = paper_dir / copy_id / f"{copy_id}.pdf"
        for system, tag in cells:
            label = f"{system}{':' + tag if tag else ''}"
            target = _verification_path(paper_dir, system, copy_id, tag)
            review_md = _review_md_path(paper_dir, system, copy_id, tag)

            if not force and target.is_file() and target.stat().st_size > 0:
                print(f"[verify] {label} × {copy_id} skip "
                      f"(exists: {target.relative_to(paper_dir)})")
                successes.append((label, copy_id, "skip"))
                continue

            if force:
                target.unlink(missing_ok=True)

            if not review_md.is_file():
                print(f"[verify] SKIP {label} × {copy_id}: no review at "
                      f"{review_md.relative_to(_ROOT)}", file=sys.stderr)
                failures.append((label, copy_id, "no review MD"))
                continue

            try:
                print(f"[verify] {label} × {copy_id} — "
                      f"{'uniform-' + str(uniform_n_judges) if uniform_n_judges else 'panel'} judges, "
                      f"{'PDF-escalate' if escalate_with_pdf else 'no escalate'}")
                report = verify_review(
                    paper_id=arxiv_id, copy_id=copy_id,
                    reviewer_system=system, reviewer_tag=tag,
                    paper_dir=paper_dir,
                    uniform_n_judges=uniform_n_judges,
                    escalate_with_pdf=escalate_with_pdf,
                )
                out_path = write_verification(paper_dir, report)
                d = report.detection_count(); a = report.ambiguous_count(); e = report.escalation_count
                print(f"[verify] {label} × {copy_id} ok — {d}/4 detected, "
                      f"{a} ambiguous, {e} escalations → {out_path.relative_to(_ROOT)}")
                successes.append((label, copy_id, str(out_path.relative_to(_ROOT))))
                if copy_pdf.is_file() and copy_pdf not in pdfs_seen:
                    pdfs_seen.append(copy_pdf)
            except Exception as ex:
                print(f"[verify] {label} × {copy_id} FAILED: {ex}", file=sys.stderr)
                failures.append((label, copy_id, str(ex)))

        if copy_pdf.is_file() and copy_pdf in pdfs_seen:
            try:
                res = delete_pdf_remote(copy_pdf)
                print(f"[verify] cleaned up remote uploads for "
                      f"{copy_pdf.relative_to(_ROOT)} — "
                      f"openai={res['openai']}, gemini={res['gemini']}")
            except Exception as e:
                print(f"[verify] cleanup failed for {copy_pdf.relative_to(_ROOT)}: {e}",
                      file=sys.stderr)

    s = cost_mod.summary()
    print()
    print(f"[verify] done — {len(successes)} ok/skip, {len(failures)} failed")
    print(f"[cost] calls={s['n_calls']} in_tok={s['total_input_tokens']:,} "
          f"out_tok={s['total_output_tokens']:,} total=${s['total_cost_usd']:.4f}")
    if failures:
        print("[verify] failures:")
        for lbl, cp, reason in failures:
            print(f"  - {lbl} {cp}: {reason}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("arxiv_id")
    ap.add_argument("--copies", default="v1,v2,v3",
                    help="injected copies (v0 is FP-only, not processed here)")
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS,
                    help=f"Default: {DEFAULT_SYSTEMS}. Use B3:<mode> for B3 sub-modes.")
    ap.add_argument("--review-providers", default=DEFAULT_PANEL_PROVIDERS,
                    help=f"Panel providers for L1/L2/L3 (default: {DEFAULT_PANEL_PROVIDERS}).")
    ap.add_argument("--no-panel", action="store_true",
                    help="Disable the panel; for L1/L2/L3 read the legacy "
                         "<copy>.md (no provider tag).")
    ap.add_argument("--uniform-n-judges", type=int, default=None,
                    help="override: same n_judges for every module (1 / 2 / 3)")
    ap.add_argument("--no-escalate", action="store_true",
                    help="skip the PDF-attached ambiguity escalation pass")
    ap.add_argument("--force", action="store_true",
                    help="re-run cells: delete-then-attempt.  Existing "
                         "verifications/<SYS>/<copy>[__<tag>].json for the "
                         "cell is removed BEFORE the new attempt, so a "
                         "partial-fail re-run can never leave a stale-version "
                         "mix on disk.")
    args = ap.parse_args()

    panel_providers = (None if args.no_panel
                       else [p.strip() for p in args.review_providers.split(",")
                             if p.strip()])
    systems_raw = _parse_systems(args.systems)
    cells = _expand_cells(systems_raw, panel_providers)

    return run(
        arxiv_id=args.arxiv_id,
        copies=[c.strip() for c in args.copies.split(",") if c.strip()],
        cells=cells,
        uniform_n_judges=args.uniform_n_judges,
        escalate_with_pdf=not args.no_escalate,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
