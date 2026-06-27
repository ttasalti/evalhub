#!/usr/bin/env python3
"""Render pass@k / cot-pass@k tables with RL progression columns.

Columns: Base (pretrained) | s120 | s240 | RL Final
One PNG per benchmark written to:
  results/RL train/combined_report/report_plots/custom_tables/
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("results/RL train/combined_report/report.csv")
OUT_DIR = Path("results/RL train/combined_report/report_plots/custom_tables")

K_VALUES = [1, 2, 4, 8, 16, 32, 64]
BENCHMARKS = ["aime2026", "aime2026_tr", "aime2026_pt", "tubitak_math2026"]
JUDGES = ["Qwen3.6-35B-A3B", "gemma-4-26B-A4B-it"]

# (model_short, display_label) — left to right column order
MODEL_ORDER = [
    ("Q-2B·Base",                        "Base"),
    ("DAPO-EN-Q-2B-t16g48·Base·s120",    "s120"),
    ("DAPO-EN-Q-2B-t16g48·Base·s240",    "s240"),
    ("DAPO-EN-Q-2B-t16g48·Base",         "RL Final"),
]


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def find_row(rows, model_short, benchmark, judged, judge_model):
    for row in rows:
        if (
            row["model_short"] == model_short
            and row["benchmark"] == benchmark
            and row["judged"] == str(judged)
            and (row["judge_model"] == judge_model if judged else True)
        ):
            return row
    return None


def fmt(row, k):
    if row is None:
        return "–"
    val = row.get(f"pass@{k}")
    if val in (None, ""):
        return "–"
    return f"{float(val) * 100:.1f}"


def build_table(rows, benchmark):
    col_labels = [""] + [label for _, label in MODEL_ORDER]
    table_rows = []

    # No-Judge block
    table_rows.append(["pass@k (No-Judge)"] + [""] * len(MODEL_ORDER))
    for k in K_VALUES:
        vals = [fmt(find_row(rows, ms, benchmark, False, None), k) for ms, _ in MODEL_ORDER]
        table_rows.append([f"  k={k}"] + vals)

    # CoT blocks (one per judge)
    for judge in JUDGES:
        short_judge = judge.replace("Qwen3.6-35B-A3B", "Qwen3.6-35B").replace("gemma-4-26B-A4B-it", "Gemma4-26B")
        table_rows.append([f"cot-pass@k  ({short_judge})"] + [""] * len(MODEL_ORDER))
        for k in K_VALUES:
            vals = [fmt(find_row(rows, ms, benchmark, True, judge), k) for ms, _ in MODEL_ORDER]
            table_rows.append([f"  k={k}"] + vals)

    return col_labels, table_rows


def render(benchmark, col_labels, table_rows, out_path):
    n_rows = len(table_rows)
    n_cols = len(col_labels)
    fig_h = 0.3 * n_rows + 0.6
    fig_w = 1.5 * n_cols + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(f"{benchmark} — pass@k progression (Base → s120 → s240 → RL Final)", fontsize=11, pad=6)

    col_widths = [0.32] + [0.17] * (n_cols - 1)
    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.3)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dddddd")
        elif r > 0 and table_rows[r - 1][0] and not table_rows[r - 1][0].startswith("  "):
            # Section header row
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold", fontsize=8)
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
            cell.PAD = 0.02

    fig.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    rows = load_rows()
    for benchmark in BENCHMARKS:
        col_labels, table_rows = build_table(rows, benchmark)
        render(benchmark, col_labels, table_rows, OUT_DIR / f"{benchmark}__passk_cotpassk.png")


if __name__ == "__main__":
    main()
