"""Token cost of judging relative to solving; feeds the judge-cost numbers quoted in the appendix
(tables/judge_cost.txt).

Solver completions are read from results/<solver>/<leaf>/<benchmark>_raw.jsonl and judge completions from
results/<solver>/judged_by/<tag>/<leaf>/cot_judge*_raw.jsonl; the last block of the summary also reads the
error-injection caches cache/judge_*consolidated*.parquet where they store token counts. None of these are part
of the repository. The script writes judge_cost.csv (one row per judge x solver x benchmark cell) and
judge_cost.txt (the summary) to analysis/tables.

Scope: the judge matrix of Section 5.2 (ten solvers x four 64-sample benchmarks, 16k solver budget)
under the four judges (Qwen3.6, Gemma4 at 16k; V4-Flash at 20,480; R1-distill at 16k, all thinking),
plus the error-injection caches where token counts were stored.

Per judgment: response.usage.completion_tokens (thinking tokens included) and finish_reason from the
judge raw files (exact file names only, grouped by task_id). Per solver generation:
usage.completion_tokens from <leaf>/<benchmark>_raw.jsonl (line order = generation index).

Ratios reported per (judge, solver, benchmark) and pooled:
  judge tokens per judged solution      = sum(judge completion) / n distinct judged generations (3 calls)
  judge / solver, judged generations    = sum(judge completion) / sum(solver completion of the judged gens)
  judge / solver, whole cell            = sum(judge completion) / sum(solver completion of all generations)
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from common import CACHE_DIR, RESULTS_ROOT, TAB_DIR, ensure_output_dirs, require

# (results dir relative to results/, label, generation)
CELLS = [
    ("Qwen2.5/base/Qwen2.5-7B", "Q2.5-7B", "earlier"),
    ("Qwen2.5/base/Qwen2.5-32B", "Q2.5-32B", "earlier"),
    ("Qwen2.5/non-think/Qwen2.5-7B-Instruct", "Q2.5-7B-Inst", "earlier"),
    ("Qwen2.5/non-think/Qwen2.5-32B-Instruct", "Q2.5-32B-Inst", "earlier"),
    ("base/Qwen3.5-4B-Base", "Q3.5-4B-Base", "current"),
    ("base/Qwen3.5-9B-Base", "Q3.5-9B-Base", "current"),
    ("non-think/Qwen3.5-4B", "Q3.5-4B", "current"),
    ("non-think/Qwen3.5-9B", "Q3.5-9B", "current"),
    ("non-think/gemma-4-E2B-it", "G4-E2B", "current"),
    ("non-think/gemma-4-E4B-it", "G4-E4B", "current"),
]
BENCH4 = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
JUDGE = {
    "Qwen3.6-35B-A3B": "Qwen3.6",
    "gemma-4-26B-A4B-it": "Gemma4",
    "deepseek-v4-flash": "V4-Flash",
    "DeepSeek-R1-0528-Qwen3-8B": "R1-distill",
}
TAG = re.compile(r"^(?P<judge>.+?)__state-(?P<state>think|non-think)__t0\.6__max(?P<jmax>\d+)__basemax(?P<bmax>\d+)")
RAWNAMES = ["cot_judge_raw.jsonl", "cot_judge_tr_raw.jsonl", "cot_judge_pt_raw.jsonl"]


def solver_tokens(leaf, bench):
    toks = {}
    with open(os.path.join(leaf, f"{bench}_raw.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            toks.setdefault(d["task_id"], []).append(int(d["response"]["usage"]["completion_tokens"]))
    return toks


def judge_tokens(files):
    """(completion_tokens, finish_reason) per judgment, grouped by judged generation id."""
    per = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                r = d["response"]
                per.setdefault(d["task_id"], []).append(
                    (int(r["usage"]["completion_tokens"]), r["choices"][0]["finish_reason"])
                )
    return per


def cell_rows():
    rows = []
    for rel, label, gen in CELLS:
        for bench in BENCH4:
            leafname = f"{bench}__t0.6__max16384__n64"
            stoks = solver_tokens(RESULTS_ROOT / rel / leafname, bench)
            all_solver = sum(sum(v) for v in stoks.values())
            for tagdir in sorted(glob.glob(str(RESULTS_ROOT / rel / "judged_by" / "*"))):
                m = TAG.match(os.path.basename(tagdir))
                if not m or m["judge"] not in JUDGE or m["state"] != "think" or m["bmax"] != "16384":
                    continue
                js = JUDGE[m["judge"]]
                if int(m["jmax"]) != (20480 if js == "V4-Flash" else 16384):
                    continue
                jl = os.path.join(tagdir, leafname)
                files = [os.path.join(jl, n) for n in RAWNAMES if os.path.exists(os.path.join(jl, n))]
                if not files:
                    continue
                per = judge_tokens(files)
                jt = [t for v in per.values() for t, _ in v]
                fl = [fr for v in per.values() for _, fr in v]
                judged_solver = 0
                missing = 0
                for gid in per:
                    t, i = gid.rsplit("_gen_", 1)
                    if t in stoks and int(i) < len(stoks[t]):
                        judged_solver += stoks[t][int(i)]
                    else:
                        missing += 1
                rows.append(
                    {
                        "judge": js,
                        "judge_max": int(m["jmax"]),
                        "solver": label,
                        "generation": gen,
                        "benchmark": bench,
                        "n_judgments": len(jt),
                        "n_solutions": len(per),
                        "judgments_per_solution": len(jt) / len(per),
                        "judge_tokens_mean": np.mean(jt),
                        "judge_tokens_median": np.median(jt),
                        "share_length": np.mean([x == "length" for x in fl]),
                        "judge_tokens_sum": sum(jt),
                        "solver_tokens_judged": judged_solver,
                        "solver_tokens_all": all_solver,
                        "ratio_judged": sum(jt) / judged_solver if judged_solver else np.nan,
                        "ratio_all": sum(jt) / all_solver if all_solver else np.nan,
                        "unmatched": missing,
                    }
                )
    return pd.DataFrame(rows)


def summary(df):
    out = ["=== per judge, pooled over the judge-matrix cells present (sums of tokens) ==="]
    for js, g in df.groupby("judge"):
        out.append(
            f"{js:10s} cells={len(g):2d} judgments={g.n_judgments.sum():6d} solutions={g.n_solutions.sum():6d} "
            f"judge tok/judgment mean={g.judge_tokens_sum.sum() / g.n_judgments.sum():7.0f} "
            f"share length={np.average(g.share_length, weights=g.n_judgments):.3f} "
            f"judge/solver(judged gens)={g.judge_tokens_sum.sum() / g.solver_tokens_judged.sum():.2f} "
            f"judge/solver(all gens)={g.judge_tokens_sum.sum() / g.solver_tokens_all.sum():.2f} "
            f"unmatched={g.unmatched.sum()}"
        )
    out.append("\n=== per judge x generation ===")
    for (js, gen), g in df.groupby(["judge", "generation"]):
        out.append(
            f"{js:10s} {gen:8s} cells={len(g):2d} tok/judgment={g.judge_tokens_sum.sum() / g.n_judgments.sum():7.0f} "
            f"length={np.average(g.share_length, weights=g.n_judgments):.3f} "
            f"j/s judged={g.judge_tokens_sum.sum() / g.solver_tokens_judged.sum():.2f} "
            f"j/s all={g.judge_tokens_sum.sum() / g.solver_tokens_all.sum():.2f}"
        )
    out.append("\n=== per judge x benchmark ===")
    for (js, b), g in df.groupby(["judge", "benchmark"]):
        out.append(
            f"{js:10s} {b:18s} cells={len(g):2d} tok/judgment={g.judge_tokens_sum.sum() / g.n_judgments.sum():7.0f} "
            f"length={np.average(g.share_length, weights=g.n_judgments):.3f} "
            f"j/s judged={g.judge_tokens_sum.sum() / g.solver_tokens_judged.sum():.2f} "
            f"j/s all={g.judge_tokens_sum.sum() / g.solver_tokens_all.sum():.2f}"
        )
    out.append("\n=== range over cells: judge/solver (judged gens) ===")
    for js, g in df.groupby("judge"):
        out.append(
            f"{js:10s} min={g.ratio_judged.min():.2f} max={g.ratio_judged.max():.2f}  "
            f"ratio_all min={g.ratio_all.min():.2f} max={g.ratio_all.max():.2f}"
        )

    out.append("\n=== error-injection study (cache parquet with judge_completion_tokens) ===")
    for f in sorted(glob.glob(str(CACHE_DIR / "judge_*consolidated*.parquet"))):
        c = pd.read_parquet(f)
        if "judge_completion_tokens" not in c.columns:
            continue
        c = c.dropna(subset=["judge_completion_tokens"])
        out.append(
            f"{os.path.basename(f):45s} judgments={len(c):6d} judge tok mean={c.judge_completion_tokens.mean():7.0f} "
            f"median={c.judge_completion_tokens.median():7.0f} solver tok mean={c.solution_len_tokens.mean():7.0f} "
            f"ratio(sum)={c.judge_completion_tokens.sum() / c.solution_len_tokens.sum():.2f}"
        )
    return "\n".join(out)


def main():
    require(RESULTS_ROOT, what="the raw experiment outputs (results/)")
    ensure_output_dirs()
    df = cell_rows()
    df.to_csv(TAB_DIR / "judge_cost.csv", index=False)
    txt = summary(df)
    print(txt)
    with open(TAB_DIR / "judge_cost.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
