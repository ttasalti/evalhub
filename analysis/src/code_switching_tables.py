"""Language-class summaries and the code-switching table from the per-chain language measurements
(Table tab:code-switching, tab_code_switching.tex, and the numbers quoted in Appendix C.5).

The per-chain measurements are code_switching_main.csv, code_switching_panel.csv and code_switching_ptexams.csv
in analysis/data. The script writes code_switching.txt (the summaries) and tab_code_switching.tex (the table)
to analysis/tables.

Class (letter shares from the segment layer): target >= T -> "target", English >= T -> "english",
otherwise "mixed"; T = 0.8 in the table, 0.7 and 0.9 in the sensitivity block. Chains with fewer than 20
letters are dropped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from common import TAB_DIR, data, ensure_output_dirs

BENCHMARK_NAME = {
    "aime2026": "AIME (EN)",
    "aime2026_pt": "AIME (PT)",
    "aime2026_tr": "AIME (TR)",
    "tubitak_math2026": "TÜB\\.ITAK (TR)",
    "pt_exams_math": "PT exams",
}
ORDER = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026", "pt_exams_math"]
CLASSES = ["target", "mixed", "english"]
SOLVERS = [
    "Q2.5-7B",
    "Q2.5-32B",
    "Q2.5-7B-Inst",
    "Q2.5-32B-Inst",
    "Q3.5-4B-Base",
    "Q3.5-9B-Base",
    "Q3.5-4B",
    "Q3.5-9B",
    "G4-E2B",
    "G4-E4B",
]
PTEXAMS_SOLVERS = [
    "Q2.5-7B",
    "Q2.5-32B",
    "Q2.5-7B-Inst",
    "Q2.5-32B-Inst",
    "Q3.5-4B-Base",
    "Q3.5-9B-Base",
    "Q3.5-4B-Think",
    "Q3.5-9B-Think",
    "G4-E2B-Think",
    "G4-E4B-Think",
]
MIN_LETTERS = 20
MIN_CELL = 20


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def classify(df: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    cls = np.where(df.share_target >= threshold, "target", np.where(df.share_eng >= threshold, "english", "mixed"))
    cls = np.where(df.n_letters.fillna(0) < MIN_LETTERS, "empty", cls)
    return pd.Series(cls, index=df.index)


def corpus_summary(name: str, df: pd.DataFrame) -> list[str]:
    """Class shares at three thresholds, acceptance by class and the top non-target/non-English labels."""
    out = []
    df = df[df.n_letters.fillna(0) >= MIN_LETTERS].copy()
    group_keys = ["benchmark"] + (["generation"] if name == "main" else [])
    for threshold in (0.8, 0.7, 0.9):
        df["cls"] = classify(df, threshold)
        out.append(f"\n=== {name} corpus, class shares (%) at T={threshold}, n chains ===")
        for key, s in df.groupby(group_keys):
            label = key[0] if len(key) == 1 else key  # a single grouping key is printed bare
            n = len(s)
            shares = s.cls.value_counts(normalize=True) * 100
            masklid_multi = 100 * (s.masklid_n[s.masklid_n.notna()] >= 2).mean()
            out.append(
                f"{str(label):40s} n={n:5d} target={shares.get('target', 0):5.1f} mixed={shares.get('mixed', 0):5.1f} "
                f"english={shares.get('english', 0):5.1f} "
                f"masklid>=2 langs={masklid_multi:5.1f} (run {int(s.masklid_n.notna().sum())}) "
                f"mean share_eng={100 * s.share_eng.mean():5.1f} other={100 * s.share_other.mean():4.1f} "
                f"q-segs dropped/chain={s.n_question_segments.mean():.1f}"
            )
    df["cls"] = classify(df, 0.8)
    out.append(f"\n=== {name} corpus, acceptance (%) by class at T=0.8 [Wilson 95%] ===")
    if name == "main":
        judges = [c for c in df.columns if c.startswith("approved_")]
    else:
        judges = [c for c in df.columns if c.startswith("clean_") or c.startswith("final_")]
    for bench in ORDER:
        s0 = df[df.benchmark == bench]
        if not len(s0):
            continue
        generations = sorted(s0.generation.unique()) if name == "main" else ["current"]
        for gen in generations:
            s1 = s0[s0.generation == gen] if name == "main" else s0
            for judge in judges:
                line = f"{BENCHMARK_NAME[bench]:14s} {gen:8s} {judge:18s}"
                for cls in CLASSES:
                    s = s1[(s1.cls == cls) & s1[judge].notna()]
                    n = len(s)
                    k = int(s[judge].astype(float).sum())
                    low, high = wilson(k, n)
                    rate = 100 * k / n if n else float("nan")
                    line += f"  {cls:7s} {rate:5.1f} [{low:4.1f},{high:5.1f}] n={n:5d}"
                out.append(line)
    out.append(f"\n=== {name}: GlotLID 'other' top labels (letters) ===")
    other_top = df.dropna(subset=["other_top"]).groupby("other_top").other_top_letters.sum()
    out.append(other_top.sort_values(ascending=False).head(8).to_string())
    return out


def acceptance_cell(s: pd.DataFrame, judge: str, cls: str) -> str:
    x = s[(s.cls == cls) & s[judge].notna()]
    return "--" if len(x) < MIN_CELL else f"{100 * x[judge].astype(float).mean():.0f}"


def solver_row(solver: str, s: pd.DataFrame) -> str:
    shares = s.cls.value_counts(normalize=True) * 100
    return (
        f"{solver} & {shares.get('target', 0):.0f} & {shares.get('mixed', 0):.0f} & {shares.get('english', 0):.0f} & "
        + " & ".join(acceptance_cell(s, "approved_Qwen3.6", c) for c in CLASSES)
        + " & "
        + " & ".join(acceptance_cell(s, "approved_Gemma4", c) for c in CLASSES)
        + f" & {len(s)} \\\\"
    )


def ptexams_summary(pt: pd.DataFrame) -> list[str]:
    out = ["\n=== Portuguese exams corpus, class shares and acceptance (T=0.8) ==="]
    for solver, s in pt.groupby("solver"):
        shares = s.cls.value_counts(normalize=True) * 100
        out.append(
            f"{solver:14s} n={len(s):5d} target={shares.get('target', 0):5.1f} mixed={shares.get('mixed', 0):5.1f} "
            f"english={shares.get('english', 0):5.1f} "
            + " ".join(
                f"{j}[{c}]={acceptance_cell(s, j, c)}" for j in ["approved_Qwen3.6", "approved_Gemma4"] for c in CLASSES
            )
        )
    return out


def main() -> None:
    ensure_output_dirs()
    main_corpus = pd.read_csv(data("code_switching_main.csv"))
    panel = pd.read_csv(data("code_switching_panel.csv"))

    out = corpus_summary("main", main_corpus) + corpus_summary("panel", panel)

    # The table pools the three non-English benchmarks of the main corpus (AIME PT, AIME TR, TUBITAK) at T=0.8,
    # one row per solver: class shares, then acceptance by class under both judges.
    m = main_corpus[(main_corpus.n_letters.fillna(0) >= MIN_LETTERS) & (main_corpus.benchmark != "aime2026")].copy()
    m["cls"] = classify(m, 0.8)
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\crnew{\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{chains (\%)} & \multicolumn{3}{c}{accepted, Qwen3.6 (\%)} & "
        r"\multicolumn{3}{c}{accepted, Gemma4 (\%)} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"Solver & target & mixed & English & target & mixed & English & target & mixed & English & $n$ \\",
        r"\midrule",
    ]
    for k, solver in enumerate(SOLVERS):
        s = m[m.solver == solver]
        if not len(s):
            continue
        if k == 4:
            lines.append(r"\midrule")
        lines.append(solver_row(solver, s))

    # Portuguese exams block (separate corpus file; thinking runs stand in for the missing non-thinking ones)
    ptexams_path = data("code_switching_ptexams.csv")
    if ptexams_path.exists():
        pt = pd.read_csv(ptexams_path)
        pt = pt[pt.n_letters.fillna(0) >= MIN_LETTERS].copy()
        pt["cls"] = classify(pt, 0.8)
        lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{11}{l}{\textit{Portuguese exams (16 generations per question; "
            r"thinking runs where no non-thinking run exists)}} \\"
        )
        for solver in PTEXAMS_SOLVERS:
            s = pt[pt.solver == solver]
            if not len(s):
                continue
            lines.append(solver_row(solver, s))
        out += ptexams_summary(pt)

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{\crnew{Language of the reasoning chains and acceptance by language, for the answer-correct "
        r"generations of the ten solvers of Section~\ref{sec:results-coincide} at the 16k budget on the three "
        r"non-English benchmarks (AIME PT, AIME TR and the Turkish olympiad, pooled), and below them on the "
        r"Portuguese exams. A chain is \emph{target} when at least 80\% of its letters, mathematics removed, are in "
        r"the benchmark's language, \emph{English} when at least 80\% are English, \emph{mixed} otherwise. Acceptance "
        r"is the majority verdict; a cell with fewer than 20 chains is left blank. $n$ is the number of chains. "
        r"Q2.5/Q3.5 = Qwen2.5/Qwen3.5 (-Inst instruct, -Base base, -Think thinking mode), G4 = Gemma-4-it.}}",
        r"\label{tab:code-switching}",
        r"\end{table*}",
    ]

    text = "\n".join(out)
    print(text)
    (TAB_DIR / "code_switching.txt").write_text(text + "\n")
    (TAB_DIR / "tab_code_switching.tex").write_text("\n".join(lines) + "\n")
    print("\nwrote tab_code_switching.tex")


if __name__ == "__main__":
    main()
