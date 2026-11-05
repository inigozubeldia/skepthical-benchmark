#!/usr/bin/env bash
# Reproduce every table, figure and appendix result of the paper from the released files.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p output/figures
PY=${PYTHON:-python3}
run() { echo "== $1"; "$PY" "$1" > "output/${1%.py}.txt"; }
run table2_detection.py          # Table 2
run adjudication_stats.py        # Section 2.4
run section3_numbers.py          # Sections 3.1 and 3.2
run fig1_fig2.py                 # Figures 1 and 2
run appD_panel_agreement.py      # Appendix D, Figure 4
run appE_family_pairing.py       # Appendix E, Tables 6-8
run appG_significance.py         # Appendix G, Tables 9-10
run appH_concise.py              # Appendix H, Table 11
run appI_attribution.py          # Appendix I
run appJ_citations_transfer.py   # Appendix J
run appK_error_coupling.py       # Appendix K
run appL_inter_reviewer.py       # Appendix L, Figure 5
run appM_cost.py                 # Appendix M, Tables 12-13
echo "done: outputs in analysis/output/"
