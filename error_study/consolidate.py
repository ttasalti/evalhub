"""Consolidate the scattered per-cell judge raw files into ONE table.

`evalhub gen` must write one <task>_raw.jsonl per (model, benchmark, variant)
directory. This merges all of them into a single file where every judge
generation sits on one row next to the EXACT metadata of the input it judged
(model / benchmark / error_type / question / old->new number / etc.), joined from
error_dataset.parquet by (row_id, error_type).

Outputs (like the dataset: parquet is the full record, csv is the eyeball view):
  judge_output.parquet: full: all input metadata + judge verdict + raw judge
                          reasoning text + per-call usage (tokens, cache hit/miss).
  judge_output.csv: metadata + verdict + usage (no huge text columns).

Usage:  PYTHONPATH=/repo python -m error_study.consolidate \
          --root error_study/output/judge_demo --out error_study/output/judge_demo_output
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import orjson
import pandas as pd

from error_study import config
from evalhub.benchmarks.math.verifier.rllm import extract_boxed_answer

# input metadata carried onto every judge-output row
_META_COLS = [
    "row_id",
    "model",
    "state",
    "benchmark",
    "group",
    "task_id",
    "question_text",
    "ground_truth",
    "answer_type",
    "error_type",
    "corruption_applied",
    "skip_reason",
    "old_number",
    "new_number",
    "wrong_number_menu",
    "change_location_char",
    "change_location_pct",
    "intermediate_menu",
    "num_occurrences_replaced",
    "solution_len_tokens",
    "solution_len_chars",
    "corrupted_solution",
]
_TEXT_COLS = ["question_text", "corrupted_solution", "judge_reasoning", "judge_think"]  # dropped from csv


def _verdict(content: str) -> str:
    if content and "</think>" in content:
        content = content.split("</think>")[-1]
    b = extract_boxed_answer(content or "")
    if not b:
        return "invalid"
    b = b.strip().lower().replace("\\text{", "").replace("}", "")
    return "yes" if "yes" in b else "no" if "no" in b else "invalid"


def consolidate(root: str, out_base: str, judge_meta: dict | None = None) -> pd.DataFrame:
    meta = pd.read_parquet(os.path.join(config.OUTPUT_DIR, "error_dataset.parquet"))
    meta = meta.set_index(["row_id", "error_type"])

    rows = []
    for raw in glob.glob(os.path.join(root, "*", "*", "*", "*_raw.jsonl")):
        task = os.path.basename(raw).replace("_raw.jsonl", "")
        gen_seen: Counter = Counter()
        for line in open(raw, "rb"):
            line = line.strip()
            if not line:
                continue
            d = orjson.loads(line)
            jid = d["task_id"]  # "<row_id>::<error_type>"
            row_id, _, error_type = jid.partition("::")
            resp = d.get("response", {}) or {}
            choice = (resp.get("choices") or [{}])[0]
            msg = choice.get("message", {}) or {}
            content = msg.get("content") or ""
            # the FULL think block lives in reasoning_content (DeepSeek think mode);
            # capture it so the single file carries the judge's real reasoning.
            think = (
                msg.get("reasoning_content")
                or (msg.get("provider_specific_fields") or {}).get("reasoning_content")
                or ""
            )
            # Local vLLM judges are served WITHOUT --reasoning-parser, so the think
            # block stays inline in `content`. Split it back out so these columns
            # mean the same thing as in the API run. Two completion shapes occur:
            # "<think>...</think>ans" (R1, emits its own opening tag) and
            # "...</think>ans" (Qwen3.x, whose chat template prefills "<think>\n").
            fr = choice.get("finish_reason")
            if not think and content:
                if "</think>" in content:
                    head, _, content = content.partition("</think>")
                    think = head.split("<think>")[-1].strip()
                    content = content.lstrip("\n")
                elif fr == "length":
                    # hit the token cap before closing </think>: it is all reasoning
                    # and there is no answer, the shape the API returns for a
                    # truncated think (reasoning_content full, content empty).
                    think, content = content.split("<think>")[-1].strip(), ""
            u = resp.get("usage", {}) or {}
            gi = gen_seen[jid]
            gen_seen[jid] += 1
            rows.append(
                {
                    "row_id": row_id,
                    "error_type": error_type,
                    "judge_task": task,
                    "gen_index": gi,
                    "judge_verdict": _verdict(content),
                    "judge_think": think,
                    "judge_reasoning": content,
                    # per-call outcome: distinguishes "verdict missing because the
                    # judge hit its token cap" from "judge stopped without \boxed{}"
                    "finish_reason": fr or "",
                    "judge_capped": fr == "length",
                    "judge_completion_tokens": u.get("completion_tokens", 0) or 0,
                    "judge_prompt_tokens": u.get("prompt_tokens", 0) or 0,
                    "judge_cache_hit_tokens": u.get("prompt_cache_hit_tokens", 0) or 0,
                    "judge_cache_miss_tokens": u.get("prompt_cache_miss_tokens", 0) or 0,
                }
            )
    if not rows:
        print(f"no <task>_raw.jsonl found under {root}")
        return pd.DataFrame()

    jdf = pd.DataFrame(rows)
    # constant judge-identity columns (which judge produced this table), the
    # consolidated schema otherwise carries no judge identity at all.
    for k, v in (judge_meta or {}).items():
        jdf[k] = v
    # majority verdict per judged solution-variant (across its generations)
    maj = (
        jdf.groupby(["row_id", "error_type"])["judge_verdict"]
        .agg(
            lambda s: (
                "no"
                if (s == "no").sum() > (s == "yes").sum()
                else "yes"
                if (s == "yes").sum() > (s == "no").sum()
                else "invalid"
            )
        )
        .rename("majority_verdict")
    )
    jdf = jdf.join(maj, on=["row_id", "error_type"])

    # attach the exact input metadata
    keep = [c for c in _META_COLS if c not in ("row_id", "error_type")]
    jdf = jdf.join(meta[keep], on=["row_id", "error_type"])

    ordered = _META_COLS + [
        "judge_task",
        "gen_index",
        "judge_verdict",
        "majority_verdict",
        "judge_reasoning",
        "judge_think",
        "finish_reason",
        "judge_capped",
        "judge_model",
        "judge_state",
        "judge_max_tokens",
        "judge_completion_tokens",
        "judge_prompt_tokens",
        "judge_cache_hit_tokens",
        "judge_cache_miss_tokens",
    ]
    jdf = (
        jdf[[c for c in ordered if c in jdf.columns]]
        .sort_values(["model", "benchmark", "error_type", "row_id", "gen_index"])
        .reset_index(drop=True)
    )

    jdf.to_parquet(f"{out_base}.parquet", index=False)
    jdf.drop(columns=[c for c in _TEXT_COLS if c in jdf.columns]).to_csv(f"{out_base}.csv", index=False)
    print(
        f"consolidated {len(jdf)} judge generations "
        f"({jdf.groupby(['row_id', 'error_type']).ngroups} solution-variants) -> "
        f"{out_base}.parquet (+ .csv without text columns)"
    )
    return jdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(config.OUTPUT_DIR, "judge_demo"))
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "judge_demo_output"))
    ap.add_argument("--clean", action="store_true", help="delete the per-cell folders after writing the single file")
    ap.add_argument("--judge-model", default=None, help="constant judge_model column stamped onto every row")
    ap.add_argument("--judge-state", default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=None)
    a = ap.parse_args()
    jm = {
        k: v
        for k, v in [
            ("judge_model", a.judge_model),
            ("judge_state", a.judge_state),
            ("judge_max_tokens", a.judge_max_tokens),
        ]
        if v is not None
    }
    df = consolidate(a.root, a.out, judge_meta=jm or None)
    if a.clean and not df.empty:
        import shutil

        shutil.rmtree(a.root, ignore_errors=True)
        print(f"removed intermediate per-cell folders: {a.root}")


if __name__ == "__main__":
    main()
