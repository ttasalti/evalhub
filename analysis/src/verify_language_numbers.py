"""Print every number quoted from the code-switching tables in Appendix C.5, Sections 4.1 and 5.1, the
Discussion and the captions, so that each can be checked against the published data layer.

Everything is read from analysis/data (code_switching_main.csv, code_switching_panel.csv,
code_switching_ptexams.csv, code_switching_budget.csv, judge_language.csv, judge_language_v4flash.csv) and
printed to stdout; nothing is written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import data

NON_ENGLISH = ["aime2026_pt", "aime2026_tr", "tubitak_math2026"]
PANEL_BENCHMARKS = ["aime2026_pt", "aime2026_tr", "tubitak_math2026", "pt_exams_math"]
CLASSES = ["target", "mixed", "english"]
JUDGE_COLUMNS = ["approved_Qwen3.6", "approved_Gemma4"]
MIN_LETTERS = 20
MIN_CELL = 20


def classify(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    df = df[df.n_letters.fillna(0) >= MIN_LETTERS].copy()
    df["cls"] = np.where(
        df.share_target >= threshold, "target", np.where(df.share_eng >= threshold, "english", "mixed")
    )
    return df


def load(name: str, threshold: float = 0.8) -> pd.DataFrame | None:
    path = data(name)
    if not path.exists():
        return None
    return classify(pd.read_csv(path), threshold)


def shares(s: pd.DataFrame) -> str:
    v = s.cls.value_counts(normalize=True) * 100
    return f"t={v.get('target', 0):5.1f} m={v.get('mixed', 0):5.1f} e={v.get('english', 0):5.1f} n={len(s)}"


def main_corpus_numbers(main: pd.DataFrame) -> None:
    print("=== Section 4.1: per-benchmark current-generation pooled non-target share; per solver non-target ===")
    for b in NON_ENGLISH:
        s = main[(main.benchmark == b) & (main.generation == "current")]
        print(b, shares(s))
    print("--- per solver, pooled over the three non-English benchmarks (Table 5 rows) ---")
    for solver, g in main[main.benchmark.isin(NON_ENGLISH)].groupby("solver", sort=False):
        line = f"{solver:14s} {shares(g)}"
        for j in JUDGE_COLUMNS:
            for c in CLASSES:
                x = g[g.cls == c][j].dropna().astype(float)
                if len(x) >= MIN_CELL:
                    line += f" | {j[9:]}-{c[:1]} {100 * x.mean():5.1f}(n{len(x)})"
                else:
                    line += f" | {j[9:]}-{c[:1]}   --   "
        print(line)
    print("--- base solvers per benchmark (77-90% English claim) ---")
    for solver in ["Q3.5-4B-Base", "Q3.5-9B-Base", "Q2.5-7B", "Q2.5-32B"]:
        for b in NON_ENGLISH:
            print(solver, b, shares(main[(main.benchmark == b) & (main.solver == solver)]))
    print("--- Q2.5-32B tubitak acceptance English vs Turkish (Qwen3.6) ---")
    g = main[(main.benchmark == "tubitak_math2026") & (main.solver == "Q2.5-32B")]
    for c in ["english", "target"]:
        x = g[g.cls == c]["approved_Qwen3.6"].dropna().astype(float)
        print(c, round(100 * x.mean(), 1), len(x))
    print("--- Qwen3.5 base cells: max |english - mixed| acceptance ---")
    max_gap = 0
    for solver in ["Q3.5-4B-Base", "Q3.5-9B-Base"]:
        for b in NON_ENGLISH:
            g = main[(main.benchmark == b) & (main.solver == solver)]
            for j in JUDGE_COLUMNS:
                e = g[g.cls == "english"][j].dropna().astype(float)
                m = g[g.cls == "mixed"][j].dropna().astype(float)
                if len(e) >= MIN_CELL and len(m) >= MIN_CELL:
                    max_gap = max(max_gap, abs(e.mean() - m.mean()) * 100)
    print("max gap", round(max_gap, 1))


def threshold_numbers(main: pd.DataFrame, panel: pd.DataFrame) -> None:
    print("--- other letters share, thresholds ---")
    for name, d in [("main", main), ("panel", panel)]:
        print(
            name,
            "mean share_other %",
            round(100 * d.share_other.mean(), 2),
            "max per bench",
            round(100 * d.groupby("benchmark").share_other.mean().max(), 2),
        )
        for threshold in (0.7, 0.9):
            d2 = load(f"code_switching_{name}.csv", threshold)
            for b in NON_ENGLISH:
                a = 100 * (d[d.benchmark == b].cls == "target").mean()
                c = 100 * (d2[d2.benchmark == b].cls == "target").mean()
                print(f"  thr {threshold} {b} target {a:.1f} -> {c:.1f} (delta {c - a:+.1f})")


def masklid_numbers(main: pd.DataFrame, panel: pd.DataFrame) -> None:
    print("--- MaskLID coverage and agreement ---")
    m = main[main.masklid_n.notna()]
    print(
        "coverage current",
        round(100 * main[main.generation == "current"].masklid_n.notna().mean(), 1),
        "earlier",
        round(100 * main[main.generation == "earlier"].masklid_n.notna().mean(), 1),
    )
    agree = (m.masklid_n >= 2) == (m.cls == "mixed")
    print(
        "agreement all",
        round(100 * agree.mean(), 1),
        "non-English",
        round(100 * agree[m.benchmark.isin(NON_ENGLISH)].mean(), 1),
        "n",
        len(m),
        int(m.benchmark.isin(NON_ENGLISH).sum()),
    )
    mp = panel[panel.masklid_n.notna()]
    print("panel agreement", round(100 * ((mp.masklid_n >= 2) == (mp.cls == "mixed")).mean(), 1), len(mp))


def panel_numbers(panel: pd.DataFrame) -> None:
    print("=== panel shares (54-73 English; 89 mixed PT exams) ===")
    for b in PANEL_BENCHMARKS:
        print(b, shares(panel[panel.benchmark == b]))
    print("=== panel acceptance by class (clean/final x judge) ===")
    columns = [c for c in panel.columns if c.startswith("clean_") or c.startswith("final_")]
    for b in PANEL_BENCHMARKS:
        for c in columns:
            g = panel[panel.benchmark == b]
            e = g[g.cls == "english"][c].dropna().astype(float)
            mm = g[g.cls == "mixed"][c].dropna().astype(float)
            print(
                f"{b:18s} {c:20s} mixed {100 * mm.mean():5.1f}(n{len(mm)})  english {100 * e.mean():5.1f}(n{len(e)})"
                f"  gap {100 * (mm.mean() - e.mean()):+5.1f}"
            )


def ptexams_numbers(ptexams: pd.DataFrame) -> None:
    print("=== PT exams block ===")
    judge_columns = [c for c in ptexams.columns if c.startswith("approved_")]
    for solver, g in ptexams.groupby("solver", sort=False):
        line = f"{solver:14s} {shares(g)}"
        min_acceptance = 100
        for j in judge_columns:
            for c in CLASSES:
                x = g[g.cls == c][j].dropna().astype(float)
                if len(x) >= MIN_CELL:
                    min_acceptance = min(min_acceptance, 100 * x.mean())
        print(line, "min acc", round(min_acceptance, 1))


def budget_numbers(budget: pd.DataFrame, main: pd.DataFrame) -> None:
    print("=== exp3 (budget) ===")
    for b in NON_ENGLISH:
        for th in ["Q3.5-4B-Think", "Q3.5-9B-Think", "G4-E2B-Think", "G4-E4B-Think"]:
            for bm in sorted(budget.budget.unique()):
                s = budget[(budget.benchmark == b) & (budget.solver == th) & (budget.budget == bm)]
                if len(s) >= MIN_CELL:
                    print(b, th, bm, shares(s))
    for b in NON_ENGLISH:
        for nt in ["Q3.5-4B", "Q3.5-9B", "G4-E2B", "G4-E4B"]:
            print("non-think", b, nt, shares(main[(main.benchmark == b) & (main.solver == nt)]))


def judge_language_numbers() -> None:
    for name in ["judge_language.csv", "judge_language_v4flash.csv"]:
        j = load(name)
        if j is None:
            continue
        print("===", name, "===")
        keys = [k for k in ["source", "judge", "benchmark", "condition"] if k in j.columns]
        t = pd.crosstab([j[k] for k in keys], j.cls, normalize="index").mul(100).round(1)
        t["n"] = j.groupby(keys).size()
        print(t.to_string())
        if "verdict" in j.columns:
            j["yes"] = (j.verdict == "yes").astype(float)
            print(j.groupby(keys + ["cls"]).yes.agg(["mean", "size"]).round(3).to_string())


def main() -> None:
    main_corpus = load("code_switching_main.csv")
    panel = load("code_switching_panel.csv")
    ptexams = load("code_switching_ptexams.csv")
    budget = load("code_switching_budget.csv")

    main_corpus_numbers(main_corpus)
    threshold_numbers(main_corpus, panel)
    masklid_numbers(main_corpus, panel)
    panel_numbers(panel)
    if ptexams is not None:
        ptexams_numbers(ptexams)
    if budget is not None:
        budget_numbers(budget, main_corpus)
    judge_language_numbers()


if __name__ == "__main__":
    main()
