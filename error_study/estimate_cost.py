"""Stage 3b, DeepSeek think-mode judge cost.

Three modes:

  analytic  (default)  Pre-estimate from measured solution lengths. Input side is
                       grounded; the OUTPUT side (think reasoning length) is only an
                       ASSUMPTION: that is exactly what the pilot measures.
  build-pilot          Write a small stratified pilot judge-input (benchmark x
                       error_type x length percentiles) to pilot/pilot_input.jsonl.
                       Run it with run_judge_study.sh --pilot (needs JUDGE_API_KEY).
  measure  <raw.jsonl> Read real DeepSeek usage from a judged raw file and
                       extrapolate the full-run cost from the measured output tokens
                       per call and cache hit/miss.

Pricing constants match scripts/run_judge_3gen_all.sh: miss $0.14/1M, hit
$0.0028/1M, output $0.28/1M.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import orjson
import pandas as pd

from error_study import config

TEMPLATE_TOKENS = 420  # judge prompt scaffold (measured from prompts.py)
# Assumed staged-caching hit fraction of the *input* for gen-1 of each variant,
# given the clean variant is sent + committed first (prefix reuse). gen-2/3 of any
# item re-send an identical prompt => ~0.98 hit. These are ASSUMPTIONS for the
# analytic pre-estimate only; `measure` replaces them with real numbers.
GEN1_HIT = {"clean": 0.0, "truncated": 0.72, "boxed_only": 0.97, "intermediate_error": 0.5, "consistent_error": 0.45}
GEN23_HIT = 0.98


def _chars_per_token(base: pd.DataFrame) -> float:
    return float((base["solution_len_chars"] / base["solution_len_tokens"]).mean())


def _judge_df():
    df = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "error_dataset.parquet"))
    return df[df["judge_include"] == True].copy()  # noqa: E712


def _est_input_tokens(df: pd.DataFrame, cpt: float) -> pd.Series:
    """Estimated judge INPUT tokens per item = (solution + question)/cpt + template."""
    sol_chars = df["corrupted_solution"].str.len().fillna(0)
    q_chars = df["question_text"].str.len().fillna(0)
    return (sol_chars + q_chars) / cpt + TEMPLATE_TOKENS


def analytic() -> None:
    base = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "base_sample.parquet"))
    cpt = _chars_per_token(base)
    df = _judge_df()
    df["in_tok"] = _est_input_tokens(df, cpt)
    n_gen = config.JUDGE_N_SAMPLES

    # input cost (staged caching)
    miss = hit = 0.0
    for et, sub in df.groupby("error_type"):
        it = sub["in_tok"].sum()
        h1 = GEN1_HIT.get(et, 0.0)
        # gen 1
        miss += it * (1 - h1)
        hit += it * h1
        # gens 2..n
        miss += it * (n_gen - 1) * (1 - GEN23_HIT)
        hit += it * (n_gen - 1) * GEN23_HIT
    input_cost = miss * config.PRICE_INPUT_MISS + hit * config.PRICE_INPUT_HIT

    n_calls = len(df) * n_gen
    print(f"chars/token (Qwen completions) ≈ {cpt:.2f}")
    print(f"judge items: {len(df)}   calls (x{n_gen}-gen): {n_calls}")
    print(f"est. input tokens (1 gen): {df['in_tok'].sum() / 1e6:.1f}M   per-item mean {df['in_tok'].mean():.0f}")
    print(
        f"\nINPUT cost (staged caching, {n_gen}-gen): ${input_cost:,.2f}  "
        f"[miss {miss / 1e6:.1f}M, hit {hit / 1e6:.1f}M]"
    )

    print("\nOUTPUT cost, think mode, ASSUMPTION (measure with --build-pilot + --measure):")
    for out_tok in (3000, 4500, 6000):
        oc = n_calls * out_tok * config.PRICE_OUTPUT
        print(f"  @ {out_tok:>5} out-tok/call: output ${oc:,.2f}   => TOTAL ${input_cost + oc:,.2f}")
    print("\n(aime2026 solutions ~23-24k tokens dominate the input side; per-benchmark below)")
    per = df.groupby("benchmark")["in_tok"].agg(["count", "sum", "mean"])
    per["sum_M"] = (per["sum"] / 1e6).round(1)
    print(per[["count", "sum_M", "mean"]].to_string())


def build_pilot(per_cell: int = 2) -> None:
    """Stratified pilot: per (benchmark, error_type) take `per_cell` rows spanning
    short/median/long solution lengths."""
    import evalhub.benchmarks  # noqa: F401
    from evalhub.benchmarks.math.verifier import extract_answer

    df = _judge_df()
    df["sol_chars"] = df["corrupted_solution"].str.len()
    pilot_rows = []
    for _, sub in df.groupby(["benchmark", "error_type"]):
        sub = sub.sort_values("sol_chars")
        n = len(sub)
        if n == 0:
            continue
        # pick spread indices (short / median / long ...) then dedup
        idxs = sorted({int(round(q * (n - 1))) for q in ([0.1, 0.5, 0.9] if per_cell >= 3 else [0.2, 0.8])[:per_cell]})
        pilot_rows.append(sub.iloc[idxs])
    pilot = pd.concat(pilot_rows).drop_duplicates("row_id" if False else None)
    outdir = os.path.join(config.OUTPUT_DIR, "pilot")
    os.makedirs(outdir, exist_ok=True)
    # one JSONL per judge_task language (the judge dataset is language-specific)
    files = defaultdict(list)
    for r in pilot.to_dict(orient="records"):
        task = config.JUDGE_TASK[r["benchmark"]]
        files[task].append(r)
    meta = []
    for task, rows in files.items():
        path = os.path.join(outdir, f"pilot_{task}.jsonl")
        with open(path, "wb") as f:
            for r in rows:
                content = r["corrupted_solution"]
                f.write(
                    orjson.dumps(
                        {
                            "task_id": f"{r['row_id']}::{r['error_type']}",
                            "original_task_id": r["task_id"],
                            "generation_idx": 0,
                            "ground_truth": r["ground_truth"],
                            "generated_answer": extract_answer(content) or "",
                            "raw_response": {
                                "choices": [
                                    {
                                        "index": 0,
                                        "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": content},
                                    }
                                ]
                            },
                        }
                    )
                    + b"\n"
                )
        for r in rows:
            meta.append(
                {
                    "task_id": f"{r['row_id']}::{r['error_type']}",
                    "judge_task": task,
                    "benchmark": r["benchmark"],
                    "error_type": r["error_type"],
                    "sol_chars": r["sol_chars"],
                }
            )
    pd.DataFrame(meta).to_csv(os.path.join(outdir, "pilot_meta.csv"), index=False)
    print(f"pilot: {len(pilot)} items across {len(files)} judge tasks -> {outdir}")
    print("run with:  scripts/run_judge_study.sh --pilot   (needs JUDGE_API_KEY)")
    print("then:      python -m error_study.estimate_cost measure <pilot raw.jsonl ...>")


def measure(raw_paths: list[str]) -> None:
    """Read DeepSeek usage from judged pilot raw files; extrapolate full-run cost."""
    meta_path = os.path.join(config.OUTPUT_DIR, "pilot", "pilot_meta.csv")
    meta = pd.read_csv(meta_path).set_index("task_id") if os.path.exists(meta_path) else None
    out_by_bench = defaultdict(list)
    hit = miss = out = 0
    n = 0
    for p in raw_paths:
        for line in open(p, "rb"):
            line = line.strip()
            if not line:
                continue
            rec = orjson.loads(line)
            u = rec.get("response", {}).get("usage", {}) or {}
            o = u.get("completion_tokens", 0) or 0
            hit += u.get("prompt_cache_hit_tokens", 0) or 0
            miss += u.get("prompt_cache_miss_tokens", 0) or 0
            out += o
            n += 1
            tid = rec.get("task_id", "")
            bench = meta.loc[tid, "benchmark"] if (meta is not None and tid in meta.index) else "?"
            out_by_bench[bench].append(o)

    if n == 0:
        print("no usage rows found")
        return
    mean_out = out / n
    print(f"pilot calls measured: {n}")
    print(f"MEASURED output tokens/call: mean {mean_out:.0f}")
    for b, xs in sorted(out_by_bench.items()):
        print(f"   {b:18} n={len(xs):3} mean_out={sum(xs) / len(xs):.0f}")
    cache_hit_rate = 100 * hit / (hit + miss) if (hit + miss) else 0
    print(f"cache hit-rate in pilot: {cache_hit_rate:.1f}%  (hit {hit / 1e6:.2f}M / miss {miss / 1e6:.2f}M)")

    # extrapolate: per-benchmark measured output mean x full call counts; input from analytic
    base = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "base_sample.parquet"))
    cpt = _chars_per_token(base)
    df = _judge_df()
    df["in_tok"] = _est_input_tokens(df, cpt)
    n_gen = config.JUDGE_N_SAMPLES
    bench_out = {b: (sum(xs) / len(xs)) for b, xs in out_by_bench.items() if xs}
    default_out = mean_out
    total_out = 0.0
    for b, sub in df.groupby("benchmark"):
        om = bench_out.get(b, default_out)
        total_out += len(sub) * n_gen * om
    # input: reuse analytic staged model but rescale hit-rate to measured overall rate
    in_tok_total = df["in_tok"].sum() * n_gen
    hr = (hit / (hit + miss)) if (hit + miss) else 0.6
    in_miss = in_tok_total * (1 - hr)
    in_hit = in_tok_total * hr
    input_cost = in_miss * config.PRICE_INPUT_MISS + in_hit * config.PRICE_INPUT_HIT
    output_cost = total_out * config.PRICE_OUTPUT
    print(f"\nEXTRAPOLATED full run ({len(df)} items x {n_gen}-gen = {len(df) * n_gen} calls):")
    print(f"  input  ~${input_cost:,.2f}  (assuming measured {hr * 100:.0f}% hit)")
    print(f"  output ~${output_cost:,.2f}  ({total_out / 1e6:.1f}M tokens)")
    print(f"  TOTAL  ~${input_cost + output_cost:,.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("analytic")
    bp = sub.add_parser("build-pilot")
    bp.add_argument("--per-cell", type=int, default=2)
    mp = sub.add_parser("measure")
    mp.add_argument("raw", nargs="+")
    args = ap.parse_args()
    if args.mode == "build-pilot":
        build_pilot(args.per_cell)
    elif args.mode == "measure":
        measure(args.raw)
    else:
        analytic()


if __name__ == "__main__":
    main()
