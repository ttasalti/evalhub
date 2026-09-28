"""Smoke tests for the analysis package: the scripts that run from the published
data layer must execute and reproduce the tables shipped in analysis/tables.

Figures are regenerated but compared only by presence and size class, because
the shipped PNGs were rendered with an older Matplotlib and antialiasing differs
between versions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "analysis" / "src"
SHIPPED_TABLES = REPO / "analysis" / "tables"

TABLE_SCRIPTS = {
    "mcnemar_exact": ["mcnemar_exact.txt"],
    "bootstrap_panel": ["bootstrap_paired.txt"],
    "language_vs_benchmark": ["language_vs_benchmark.txt"],
    "selection_rule": ["selection_rule.txt"],
    "edit_size": ["edit_size.txt", "tab_edit_size.tex"],
    "code_switching_tables": ["code_switching.txt", "tab_code_switching.tex"],
    "tab_judge_matrix": ["tab_judge_matrix.tex"],
    "tab_conditions": ["tab_conditions.tex"],
    "tab_ladder": ["tab_ladder.tex"],
    "tab_language": ["tab_language.tex"],
}
FIGURE_SCRIPTS = {
    "fig_error_by_benchmark": ["fig_error_by_benchmark.png"],
    "fig_gap_endpoints": ["fig_gap_endpoints.png"],
    "fig_k_ladder": ["fig_k_ladder.png"],
    "fig_bootstrap": ["fig_bootstrap.png"],
    "fig_edit_size": ["fig_edit_size.png"],
    "fig_acceptance": ["fig_acceptance.png"],
    "fig_saturation": ["fig_saturation.png"],
    "fig_code_switching": ["fig_code_switching.png"],
    "fig_judge_capping": ["fig_judge_capping.png"],
    "fig_language_by_experiment": ["fig_language_exp1.png", "fig_language_exp2.png", "fig_language_exp3.png"],
}


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("analysis_out")


def run(script: str, out_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, EVALHUB_ANALYSIS_OUT=str(out_dir), PYTHONPATH=str(SRC), MPLBACKEND="Agg")
    return subprocess.run([sys.executable, str(SRC / f"{script}.py")], env=env, capture_output=True, text=True)


@pytest.mark.parametrize("script", sorted(TABLE_SCRIPTS))
def test_table_script_reproduces_shipped_output(script: str, out_dir: Path) -> None:
    result = run(script, out_dir)
    assert result.returncode == 0, result.stderr[-2000:]
    for name in TABLE_SCRIPTS[script]:
        produced = (out_dir / "tables" / name).read_text()
        shipped = (SHIPPED_TABLES / name).read_text()
        assert produced == shipped, f"{name} differs from analysis/tables/{name}"


@pytest.mark.parametrize("script", sorted(FIGURE_SCRIPTS))
def test_figure_script_writes_png(script: str, out_dir: Path) -> None:
    result = run(script, out_dir)
    assert result.returncode == 0, result.stderr[-2000:]
    for name in FIGURE_SCRIPTS[script]:
        png = out_dir / "figures" / name
        assert png.is_file() and png.stat().st_size > 20_000, name


def test_headline_numbers_match_the_paper(out_dir: Path) -> None:
    """The averaged differences quoted in the abstract and Section 5.2 come from
    bootstrap_questions.py; the shipped diff_bootstrap_ci.csv must carry them."""
    result = run("fig_k_ladder", out_dir)
    assert result.returncode == 0, result.stderr[-2000:]
    rows = (REPO / "analysis" / "data" / "diff_bootstrap_ci.csv").read_text().splitlines()
    assert "generation,earlier,all4,19.7,16.4,23.0" in rows
    assert "generation,current,all4,4.1,2.7,5.5" in rows
    assert any(r.startswith("earlier_minus_current,") and ",15.6," in r for r in rows)
