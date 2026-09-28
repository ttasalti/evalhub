"""Stage 3a, write judge-input JSONL files for the DeepSeek think-mode judge.

One file per (model, benchmark, error_type), containing only the rows that should
be judged (clean control + successfully-applied corruptions; skipped corruptions
are excluded). Each line uses evalhub's exact 6-field CoT-judge-input schema, with
the (possibly corrupted) solution placed at
``raw_response.choices[0].message.content``: the field CoTJudgeDataset reads. The
question is re-resolved by the judge from ``original_task_id`` at run time.

A manifest.json records every file, its judge task (language), row count, and the
recommended run order that maximises DeepSeek prefix-cache reuse (clean first, then
decreasing prefix overlap: type4, type1, type2, type3).

Pure data, no API calls.
"""

from __future__ import annotations

import json
import os

import orjson
import pandas as pd

import evalhub.benchmarks  # noqa: F401
from error_study import config
from evalhub.benchmarks.math.verifier import extract_answer

# Per-group send order: clean warms the shared solution prefix; then corruptions
# in decreasing prefix-overlap order so each reuses as much cached prefix as
# possible. (type1 = boxed at the very end; type4 shares the first 75%; type2
# diverges at 40-70%; type3 diverges at the first answer occurrence.)
RUN_ORDER = ["clean", "truncated", "boxed_only", "intermediate_error", "consistent_error"]


def judge_line(row: dict) -> bytes:
    content = row["corrupted_solution"]
    return orjson.dumps(
        {
            "task_id": f"{row['row_id']}::{row['error_type']}",
            "original_task_id": row["task_id"],
            "generation_idx": 0,
            "ground_truth": row["ground_truth"],
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


def main() -> None:
    df = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "error_dataset.parquet"))
    df = df[df["judge_include"] == True]  # noqa: E712
    root = os.path.join(config.OUTPUT_DIR, "judge_inputs")
    os.makedirs(root, exist_ok=True)

    manifest = []
    total = 0
    for model in config.MODELS:
        for benchmark in config.BENCHMARKS:
            for et in RUN_ORDER:
                sub = df[(df["model"] == model) & (df["benchmark"] == benchmark) & (df["error_type"] == et)]
                if len(sub) == 0:
                    continue
                rel = os.path.join(model, benchmark, f"{et}.jsonl")
                path = os.path.join(root, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    for r in sub.to_dict(orient="records"):
                        f.write(judge_line(r) + b"\n")
                manifest.append(
                    {
                        "model": model,
                        "benchmark": benchmark,
                        "error_type": et,
                        "judge_task": config.JUDGE_TASK[benchmark],
                        "file": rel,
                        "n_rows": len(sub),
                        "run_order_index": RUN_ORDER.index(et),
                    }
                )
                total += len(sub)

    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(
            {
                "judge_model": config.JUDGE_MODEL,
                "judge_state": config.JUDGE_STATE,
                "n_samples": config.JUDGE_N_SAMPLES,
                "max_completion_tokens": config.JUDGE_MAX_COMPLETION_TOKENS,
                "total_judge_items": total,
                "total_judge_calls_3gen": total * config.JUDGE_N_SAMPLES,
                "files": manifest,
            },
            f,
            indent=2,
        )
    print(
        f"wrote {len(manifest)} judge-input files, {total} judge items "
        f"({total * config.JUDGE_N_SAMPLES} calls at {config.JUDGE_N_SAMPLES}-gen)"
    )
    print(f"manifest: {os.path.join(root, 'manifest.json')}")


if __name__ == "__main__":
    main()
