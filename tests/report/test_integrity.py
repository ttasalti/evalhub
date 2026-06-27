"""Report-side integrity guard + exclusion/cap behaviour.

Mirrors ``scripts/audit_integrity.py`` but at the summary level: every judged
row must stay under its No-Judge reference (monotone) and account for exactly the
base-correct generations (count). Also covers the published-report filters added
to ``aggregate_results``: Ministral/Mistral exclusion and the first-64 K cap.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalhub.report.aggregate import aggregate_results, check_report_integrity
from evalhub.report.scan import scan_results


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _cell(
    root: Path,
    *,
    base_pass: dict,
    base_true: int,
    cot_pass: dict,
    cot_true: int,
    cot_false: int,
    model: str = "qwen-mini",
    judge: str = "qwen-judge",
    state: str = "base",
    bench: str = "gsm8k",
) -> None:
    """Write a base summary + a sibling judged summary in the V4 layout."""
    target = root / state / f"{model}__t0.6__max2048__n64"
    _write(
        target / bench / f"{bench}_summary.json",
        {"pass_at_k": base_pass, "cons_at_k": 0.55, "total_tasks": 1,
         "total_generations": 64, "true_count": base_true, "false_count": 0,
         "cot_false_count": 0, "invalid_format_count": 0},
    )
    jdir = target / "judged_by" / f"{judge}__state-think__t0.6__max16384__n3" / bench
    _write(
        jdir / f"{bench}_cot_summary.json",
        {"pass_at_k": cot_pass, "cons_at_k": 0.50, "total_tasks": 1,
         "total_generations": 64, "true_count": cot_true, "false_count": 0,
         "cot_false_count": cot_false, "invalid_format_count": 0},
    )


def test_clean_tree_has_no_violations(v3_results_root: Path) -> None:
    records = list(scan_results(v3_results_root))
    assert check_report_integrity(records) == []


def test_monotone_violation_detected(tmp_path: Path) -> None:
    root = tmp_path / "results"
    # cot pass@1 (0.60) exceeds No-Judge (0.50) -> impossible under the veto.
    _cell(root, base_pass={"1": 0.50}, base_true=32,
          cot_pass={"1": 0.60}, cot_true=32, cot_false=0)
    violations = check_report_integrity(list(scan_results(root)))
    assert any("pass@1" in v and ">" in v for v in violations), violations


def test_count_mismatch_detected(tmp_path: Path) -> None:
    root = tmp_path / "results"
    # base_true=32 but cot_true+cot_false=20+5=25 -> 7 generations unaccounted.
    _cell(root, base_pass={"1": 0.50}, base_true=32,
          cot_pass={"1": 0.40}, cot_true=20, cot_false=5)
    violations = check_report_integrity(list(scan_results(root)))
    assert any("count" in v for v in violations), violations


def test_clean_cell_passes_both(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _cell(root, base_pass={"1": 0.50, "2": 0.65}, base_true=32,
          cot_pass={"1": 0.40, "2": 0.55}, cot_true=28, cot_false=4)
    assert check_report_integrity(list(scan_results(root))) == []


def test_aggregate_does_not_block_on_violation(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _cell(root, base_pass={"1": 0.50}, base_true=32,
          cot_pass={"1": 0.90}, cot_true=32, cot_false=0)
    # The guard detects the violation ...
    assert check_report_integrity(list(scan_results(root)))
    # ... but aggregate still writes the CSV (warn, never block).
    out = tmp_path / "report.csv"
    df = aggregate_results(root, out)
    assert out.exists() and len(df) == 2


def test_ministral_mistral_excluded(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _cell(root, base_pass={"1": 0.5}, base_true=10, cot_pass={"1": 0.4},
          cot_true=8, cot_false=2, model="qwen-mini")
    _cell(root, base_pass={"1": 0.5}, base_true=10, cot_pass={"1": 0.4},
          cot_true=8, cot_false=2, model="Ministral-3-8B-Base")
    # Mistral as a *judge* must also drop the judged row.
    _cell(root, base_pass={"1": 0.5}, base_true=10, cot_pass={"1": 0.4},
          cot_true=8, cot_false=2, model="qwen-mini", judge="Mistral-judge",
          bench="aime2026")
    df = aggregate_results(root, tmp_path / "report.csv")
    joined = (df["model"].astype(str) + "|" + df["judge_model"].astype(str)).str.lower()
    assert not joined.str.contains("ministral|mistral").any()


def test_max_k_cap_drops_beyond_64(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _cell(root, base_pass={"1": 0.5, "64": 0.8, "128": 0.85}, base_true=40,
          cot_pass={"1": 0.4, "64": 0.7, "128": 0.72}, cot_true=36, cot_false=4)
    df = aggregate_results(root, tmp_path / "report.csv", max_k=64)
    assert "pass@64" in df.columns
    assert "pass@128" not in df.columns
