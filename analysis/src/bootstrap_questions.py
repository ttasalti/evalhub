"""Question-level bootstrap intervals for Pass@64 - CoT-Pass@64, written to the data layer as diff_bootstrap_ci.csv.

Reads verdicts.csv and gap_curve.csv from analysis/data and writes diff_bootstrap_ci.csv back to analysis/data;
a different output path may be given as the first argument.

The difference is a paired difference of two proportions over the same
questions, so questions are resampled (10,000 replicates, fixed seed). Per
task, pass = 1 if at least one solution is correct and cot = 1 if at least one
correct solution is approved by the majority verdict; tasks with no correct
solution contribute (0, 0). The observed cell differences are asserted against
gap_curve.csv so the task-level reconstruction matches the published curves.

Rows of the output: one per solver x benchmark cell, one question-weighted
average per solver, one mean per generation, one earlier-minus-current
contrast per benchmark and one overall contrast. Only question-sampling
uncertainty is captured; cells are resampled independently. All-zero cells get
a degenerate [0, 0] interval.
"""

import sys

import numpy as np
import pandas as pd
from common import DATA_DIR, data

VERDICTS = data("verdicts.csv")
GAP_CURVE = data("gap_curve.csv")
OUT = DATA_DIR / "diff_bootstrap_ci.csv"

BM = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
ROWS = [
    ("Qwen2.5-7B", "base", "earlier"),
    ("Qwen2.5-32B", "base", "earlier"),
    ("Qwen2.5-7B-Instruct", "non-think", "earlier"),
    ("Qwen2.5-32B-Instruct", "non-think", "earlier"),
    ("Qwen3.5-4B-Base", "base", "current"),
    ("Qwen3.5-9B-Base", "base", "current"),
    ("Qwen3.5-4B", "non-think", "current"),
    ("Qwen3.5-9B", "non-think", "current"),
    ("gemma-4-E2B-it", "non-think", "current"),
    ("gemma-4-E4B-it", "non-think", "current"),
]
REPS = 10_000
SEED = 0


def load_cells():
    """Per solver x benchmark cell, the 0/1 vector 'task has correct solutions but none approved'."""
    v = pd.read_csv(VERDICTS)
    v = v[
        (v.judge_short == "Qwen3.6")
        & (v.judge_state == "think")
        & (v.solver_max_tokens == 16384)
        & (v.judge_max_tokens == 16384)
        & (v.benchmark.isin(BM))
        & (v.answer_correct)
    ]
    g = pd.read_csv(GAP_CURVE)
    g = g[
        (g.judge_short == "Qwen3.6")
        & (g.judge_state == "think")
        & (g.solver_max_tokens == 16384)
        & (g.judge_max_tokens == 16384)
        & (g.k == 64)
    ]

    cells = {}
    for m, s, _gen in ROWS:
        for b in BM:
            gg = g[(g.solver_model == m) & (g.state == s) & (g.benchmark == b)]
            n_tasks = int(gg.n_tasks.iloc[0])
            ref = 100 * (gg.pass_at_k.iloc[0] - gg.cot_pass_at_k.iloc[0])
            d = v[(v.solver_model == m) & (v.state == s) & (v.benchmark == b)]
            per_task = d.groupby("task_id").approved_maj.any()
            diff_ind = np.zeros(n_tasks, int)
            diff_ind[: len(per_task)] = (~per_task.to_numpy()).astype(int)
            obs = 100 * diff_ind.mean()
            assert abs(obs - ref) < 0.06, (m, b, obs, ref)
            cells[(m, b)] = diff_ind
    return cells


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else OUT
    cells = load_cells()
    rng = np.random.default_rng(SEED)

    def boot(ind):
        idx = rng.integers(0, len(ind), size=(REPS, len(ind)))
        return 100 * ind[idx].mean(1)

    rows = []
    model_boot = {}
    for m, _s, _gen in ROWS:
        per_b = []
        w = []
        obs_b = []
        for b in BM:
            ind = cells[(m, b)]
            bs = boot(ind)
            per_b.append(bs)
            w.append(len(ind))
            obs_b.append(100 * ind.mean())
            rows.append(
                {
                    "level": "cell",
                    "model": m,
                    "benchmark": b,
                    "obs": 100 * ind.mean(),
                    "lo": np.percentile(bs, 2.5),
                    "hi": np.percentile(bs, 97.5),
                }
            )
        w = np.array(w, float)
        avg = np.average(np.stack(per_b), axis=0, weights=w)
        model_boot[m] = avg
        rows.append(
            {
                "level": "model_avg",
                "model": m,
                "benchmark": "all4",
                "obs": np.average(obs_b, weights=w),
                "lo": np.percentile(avg, 2.5),
                "hi": np.percentile(avg, 97.5),
            }
        )

    for gen in ["earlier", "current"]:
        ms = [m for m, _s, gg in ROWS if gg == gen]
        stack = np.stack([model_boot[m] for m in ms]).mean(0)
        obs = np.mean([r["obs"] for r in rows if r["level"] == "model_avg" and r["model"] in ms])
        rows.append(
            {
                "level": "generation",
                "model": gen,
                "benchmark": "all4",
                "obs": obs,
                "lo": np.percentile(stack, 2.5),
                "hi": np.percentile(stack, 97.5),
            }
        )

    for b in BM:
        eb = np.stack([boot(cells[(m, b)]) for m, _s, gg in ROWS if gg == "earlier"]).mean(0)
        cb = np.stack([boot(cells[(m, b)]) for m, _s, gg in ROWS if gg == "current"]).mean(0)
        obs_e = np.mean([100 * cells[(m, b)].mean() for m, _s, gg in ROWS if gg == "earlier"])
        obs_c = np.mean([100 * cells[(m, b)].mean() for m, _s, gg in ROWS if gg == "current"])
        rows.append(
            {
                "level": "benchmark_contrast",
                "model": b,
                "benchmark": b,
                "obs": obs_e - obs_c,
                "lo": np.percentile(eb - cb, 2.5),
                "hi": np.percentile(eb - cb, 97.5),
            }
        )

    e = np.stack([model_boot[m] for m, _s, gg in ROWS if gg == "earlier"]).mean(0)
    c = np.stack([model_boot[m] for m, _s, gg in ROWS if gg == "current"]).mean(0)
    gen_obs = {r["model"]: r["obs"] for r in rows if r["level"] == "generation"}
    rows.append(
        {
            "level": "earlier_minus_current",
            "model": "contrast",
            "benchmark": "all4",
            "obs": gen_obs["earlier"] - gen_obs["current"],
            "lo": np.percentile(e - c, 2.5),
            "hi": np.percentile(e - c, 97.5),
        }
    )

    df = pd.DataFrame(rows).round(1)
    df.to_csv(out, index=False)
    print(df[df.level != "cell"].to_string(index=False))
    print("\ncell CIs excluding 0:", int((df[df.level == "cell"].lo > 0).sum()), "/ 40")
    print("model_avg CIs excluding 0:", int((df[df.level == "model_avg"].lo > 0).sum()), "/ 10")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
