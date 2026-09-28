"""Tiny live demo that runs the REAL evalhub judge (evalhub gen + eval) on a few
base solutions, writing exactly what the normal judge writes, and measures the
DeepSeek think-mode cost + prompt-cache behaviour from the raw files, then
extrapolates the full 7115-item x 3-gen run.

For each (model, benchmark, variant) it writes to
  error_study/output/judge_demo/<model>/<benchmark>/<variant>/
    <task>_cot_judge_input.jsonl   (the judge input, 6-field schema)
    <task>_raw.jsonl               (RAW judge responses: {"task_id","response"})
    <task>_results.jsonl / _per_task.csv / cot_judge*.jsonl   (evalhub eval)
i.e. the same artifacts a normal judge run produces (raw + parsed verdicts).

Caching is maximised: per group `clean` is generated + committed first, then the
4 corruptions reuse its prefix; then all variants gen-2/gen-3 re-send (~99% hit).
Cost/cache are measured from the raw `usage` (prompt_cache_hit/miss, completion).

Needs JUDGE_API_KEY (DeepSeek) in the env; never stored. Spends ~cents.
  JUDGE_API_KEY=sk-... PYTHONPATH=/repo python -m error_study.run_demo --bases 4
  ... --dry           # build inputs only, no API
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd

from error_study import config
from error_study.build_judge_inputs import judge_line
from error_study.estimate_cost import _chars_per_token, _est_input_tokens, _judge_df

_VARIANT_ORDER = ["clean", "truncated", "boxed_only", "intermediate_error", "consistent_error"]
EVALHUB = os.environ.get("EVALHUB_BIN", "evalhub")
DEMO_ROOT = os.path.join(config.OUTPUT_DIR, "judge_demo")


def pick_bases(n_bases: int) -> list[str]:
    """Base row_ids whose 4 corruptions are all applied, spread across the
    solution-length distribution (gentle percentiles to avoid context-limit
    outliers)."""
    df = _judge_df()
    applied = df[(df.error_type != "clean") & (df.corruption_applied)]
    full = applied.groupby("row_id").error_type.nunique()
    full_ids = set(full[full == 4].index)
    base = df[(df.error_type == "clean") & (df.row_id.isin(full_ids))].copy()
    base["sol_chars"] = base.corrupted_solution.str.len()
    base = base[base.sol_chars <= 130_000].sort_values("sol_chars").reset_index(drop=True)
    n = len(base)
    pcts = [0.15, 0.55, 0.85, 0.30, 0.70, 0.45][:n_bases]
    idxs = sorted({int(p * (n - 1)) for p in pcts})
    return base.loc[idxs, "row_id"].tolist()


def build_inputs(base_ids: list[str]):
    """Write per-(model,benchmark,variant) judge-input files. Returns the list of
    cells: dict(model, benchmark, variant, task, dir, input, n_tasks)."""
    df = _judge_df()
    demo = df[df.row_id.isin(base_ids)]
    cells = []
    for (model, bench, variant), sub in demo.groupby(["model", "benchmark", "error_type"]):
        task = config.JUDGE_TASK[bench]
        d = os.path.join(DEMO_ROOT, model, bench, variant)
        os.makedirs(d, exist_ok=True)
        inp = os.path.join(d, f"{task}_cot_judge_input.jsonl")
        with open(inp, "wb") as f:
            for r in sub.to_dict("records"):
                f.write(judge_line(r) + b"\n")
        cells.append(
            {
                "model": model,
                "benchmark": bench,
                "variant": variant,
                "task": task,
                "dir": d,
                "input": inp,
                "n_tasks": len(sub),
            }
        )
    return cells


def _gen(cell, n, resume, key):
    args = [
        EVALHUB,
        "gen",
        "--model",
        f"hosted_vllm/{config.JUDGE_MODEL}",
        "--tasks",
        cell["task"],
        "--model-state",
        "think",
        "--temperature",
        config.JUDGE_TEMPERATURE,
        "--top-p",
        config.JUDGE_TOP_P,
        "--n-samples",
        str(n),
        "--num-workers",
        "8",
        "--max-completion-tokens",
        str(config.JUDGE_MAX_COMPLETION_TOKENS),
        "--reasoning-effort",
        config.JUDGE_REASONING_EFFORT,
        "--extra-body",
        '{"thinking": {"type": "enabled"}}',
        "--output-dir",
        cell["dir"],
        "--override-args",
        json.dumps({"file_path": cell["input"]}),
    ]
    if resume:
        args.append("--resume")
    env = dict(os.environ, HOSTED_VLLM_API_BASE=config.JUDGE_API_BASE, HOSTED_VLLM_API_KEY=key)
    subprocess.run(args, env=env, check=True, stdout=subprocess.DEVNULL)


def run(n_bases: int, commit_wait: int, dry: bool) -> None:
    base_ids = pick_bases(n_bases)
    cells = build_inputs(base_ids)
    total_calls = sum(c["n_tasks"] for c in cells) * config.JUDGE_N_SAMPLES
    print(
        f"demo: {len(base_ids)} base solutions -> {len(cells)} (group x variant) cells, "
        f"{total_calls} calls (x{config.JUDGE_N_SAMPLES}-gen)"
    )
    for rid in base_ids:
        r = _judge_df().query("row_id == @rid and error_type == 'clean'").iloc[0]
        print(f"  {rid}  ({r['benchmark']}, ~{len(r['corrupted_solution'])} chars)")
    print(f"outputs -> {DEMO_ROOT}/<model>/<benchmark>/<variant>/  (raw + eval, per normal judge)")
    if dry:
        print("[dry] inputs written; no API calls.")
        return

    key = os.environ.get("JUDGE_API_KEY")
    if not key:
        raise SystemExit("set JUDGE_API_KEY=sk-... (DeepSeek); it is never stored")

    # group cells by (model,benchmark); clean first per group (round-major staging)
    groups = defaultdict(list)
    for c in cells:
        groups[(c["model"], c["benchmark"])].append(c)
    for (model, bench), gcells in groups.items():
        gcells.sort(key=lambda c: _VARIANT_ORDER.index(c["variant"]))
        clean = [c for c in gcells if c["variant"] == "clean"]
        corr = [c for c in gcells if c["variant"] != "clean"]
        print(f"\n=== {model}/{bench}: gen (round-major, clean-first)")
        for c in clean:
            _gen(c, 1, False, key)  # clean gen-1 (cold)
        print(f"  [commit-wait {commit_wait}s]")
        time.sleep(commit_wait)
        for c in corr:
            _gen(c, 1, False, key)  # corruptions gen-1 (reuse clean)
        for n in range(2, config.JUDGE_N_SAMPLES + 1):
            print(f"  [commit-wait {commit_wait}s]")
            time.sleep(commit_wait)
            for c in gcells:
                _gen(c, n, True, key)  # gen-2/3 re-send (~99% hit)
    # verdicts are parsed from the raw responses by consolidate() below, so no
    # separate evalhub-eval step is needed.

    # merge all per-cell raw files into ONE table next to the input metadata
    from error_study.consolidate import consolidate

    out_base = os.path.join(config.OUTPUT_DIR, "judge_demo_output")
    df = consolidate(DEMO_ROOT, out_base)
    if not df.empty:
        measure(df)


def measure(df: pd.DataFrame) -> None:
    m = df.rename(
        columns={"judge_completion_tokens": "comp", "judge_cache_hit_tokens": "hit", "judge_cache_miss_tokens": "miss"}
    ).copy()
    m["phase"] = [
        "clean_g1" if (et == "clean" and gi == 0) else "corr_g1" if (et != "clean" and gi == 0) else f"g{gi + 1}"
        for et, gi in zip(m.error_type, m.gen_index, strict=True)
    ]
    actual = (
        m.miss.sum() * config.PRICE_INPUT_MISS
        + m.hit.sum() * config.PRICE_INPUT_HIT
        + m.comp.sum() * config.PRICE_OUTPUT
    )

    def hr(sub):
        t = sub.hit.sum() + sub.miss.sum()
        return 100 * sub.hit.sum() / t if t else 0.0

    print(f"\n=== MEASURED ({len(m)} calls, actual demo spend ${actual:.3f}) ===")
    print(f"mean output tokens/call: {m.comp.mean():.0f}")
    print("\ncache hit-rate by phase (the caching levers):")
    for ph, lab in [
        ("clean_g1", "clean gen-1 (cold, expect ~0%)"),
        ("corr_g1", "corruption gen-1 (cross-variant reuse)"),
        ("g2", "gen-2 re-send (3-gen reuse)"),
        ("g3", "gen-3 re-send (3-gen reuse)"),
    ]:
        s = m[m.phase == ph]
        if len(s):
            print(f"  {lab:42} hit={hr(s):5.1f}%   out/call={s.comp.mean():.0f}")
    print("\noutput tokens/call by benchmark:")
    print(m.groupby("benchmark").comp.mean().round(0).to_string())

    # extrapolate full run
    base = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "base_sample.parquet"))
    cpt = _chars_per_token(base)
    df = _judge_df().copy()
    df["in_tok"] = _est_input_tokens(df, cpt)
    ng = config.JUDGE_N_SAMPLES
    ob = m.groupby("benchmark").comp.mean().to_dict()
    default_out = m.comp.mean()
    total_out = sum(len(sub) * ng * ob.get(b, default_out) for b, sub in df.groupby("benchmark"))
    hr_corr = hr(m[m.phase == "corr_g1"]) / 100 if len(m[m.phase == "corr_g1"]) else 0.6
    hr_g23 = hr(m[m.phase.isin(["g2", "g3"])]) / 100 if len(m[m.phase.isin(["g2", "g3"])]) else 0.98
    in_clean = df[df.error_type == "clean"].in_tok.sum()
    in_corr = df[df.error_type != "clean"].in_tok.sum()
    miss = in_clean + in_corr * (1 - hr_corr) + (in_clean + in_corr) * (ng - 1) * (1 - hr_g23)
    hit = in_corr * hr_corr + (in_clean + in_corr) * (ng - 1) * hr_g23
    in_cost = miss * config.PRICE_INPUT_MISS + hit * config.PRICE_INPUT_HIT
    out_cost = total_out * config.PRICE_OUTPUT
    print(f"\n=== EXTRAPOLATED full run (7115 items x {ng}-gen = {7115 * ng} calls) ===")
    print(
        f"  measured: corr-gen1 hit {hr_corr * 100:.0f}%, gen2/3 hit {hr_g23 * 100:.0f}%, out/call ~{default_out:.0f}"
    )
    print(f"  INPUT  ~${in_cost:,.2f}   OUTPUT ~${out_cost:,.2f}   =>  TOTAL ~${in_cost + out_cost:,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", type=int, default=4)
    ap.add_argument("--commit-wait", type=int, default=60)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run(a.bases, a.commit_wait, a.dry)


if __name__ == "__main__":
    main()
