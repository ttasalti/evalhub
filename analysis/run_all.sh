#!/usr/bin/env bash
# Regenerate every table and figure that depends only on the published data
# layer. Run from the repository root:  bash analysis/run_all.sh
# Scripts that need the unpublished raw outputs (build_data.py,
# code_switching.py, judge_language*.py, judge_cost.py, panel_three_judges.py,
# tab_error_rules.py) are not called here; see analysis/README.md.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="analysis/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
PY="${PYTHON:-python}"

tables=(selection_rule language_vs_benchmark mcnemar_exact bootstrap_panel edit_size
        code_switching_tables tab_judge_matrix tab_conditions tab_ladder tab_language)
figures=(fig_error_by_benchmark fig_gap_endpoints fig_edit_size fig_language_by_experiment
         fig_k_ladder fig_bootstrap fig_acceptance fig_saturation fig_code_switching fig_judge_capping)

for script in "${tables[@]}" "${figures[@]}"; do
    echo "== ${script}"
    "${PY}" "analysis/src/${script}.py" > /dev/null
done
echo "done: analysis/tables and analysis/figures refreshed"
