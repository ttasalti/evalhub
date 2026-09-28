"""Stage 1, sample 1500 correct, long think-mode solutions.

For each of the 10 (model, benchmark) groups: read ONLY the highest max-token
variant's raw file, keep generations that are (a) correct via the authoritative
boxed grade, (b) finish_reason == "stop", (c) completion_tokens > MIN_TOKENS,
then round-robin sample PER_GROUP rows with <= MAX_SOLS_PER_Q per question.

Strictly read-only against results/. Writes only base_sample.parquet.
"""

from __future__ import annotations

# Offline HF so dataset question/GT loads come from the local cache.
import os

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import random
from collections import defaultdict

import orjson
import pandas as pd

import evalhub.benchmarks  # noqa: F401, registers datasets
from error_study import config
from error_study.common import classify_answer_type
from evalhub.benchmarks.registry import DATASET_MAP

_DATASET_CACHE: dict = {}


def get_dataset(benchmark: str):
    if benchmark not in _DATASET_CACHE:
        ds = DATASET_MAP[benchmark](benchmark)
        ds.load_tasks()
        _DATASET_CACHE[benchmark] = ds
    return _DATASET_CACHE[benchmark]


def _group_rng(group: str) -> random.Random:
    h = hashlib.md5(f"{config.SEED}:{group}".encode()).hexdigest()[:8]
    return random.Random(int(h, 16))


def load_results_gt(leaf: str, benchmark: str) -> dict[str, str]:
    gt: dict[str, str] = {}
    path = os.path.join(leaf, f"{benchmark}_results.jsonl")
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = orjson.loads(line)
            gt[d["task_id"]] = str(d.get("ground_truth", ""))
    return gt


def eligible_generations(model: str, benchmark: str):
    """Yield (task_id, ordinal, content, tokens, gt) for correct/stop/>600 gens."""
    leaf = config.leaf_dir(model, benchmark)
    ds = get_dataset(benchmark)
    results_gt = load_results_gt(leaf, benchmark)
    ordinal = defaultdict(int)
    raw_path = os.path.join(leaf, f"{benchmark}_raw.jsonl")
    with open(raw_path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = orjson.loads(line)
            tid = d["task_id"]
            ordn = ordinal[tid]
            ordinal[tid] += 1
            resp = d["response"]
            ch = resp["choices"][0]
            content = ch["message"].get("content") or ""
            if ch.get("finish_reason") != "stop":
                continue
            tokens = (resp.get("usage") or {}).get("completion_tokens", 0) or 0
            if tokens <= config.MIN_TOKENS:
                continue
            gt = ds.groundtruth[tid].answer if tid in ds.groundtruth else results_gt.get(tid, "")
            if not gt:
                continue
            ext = ds.extract_solution(tid, content)
            # authoritative grader = grade_answer OR the benchmark's patch() (e.g.
            # pt_exams_math's percent/decimal tolerance), same check evalhub's
            # base eval used to mark `correct`.
            if not ds.check_correct(ext, gt, tid):
                continue
            yield tid, ordn, content, tokens, gt


def round_robin_sample(by_task: dict[str, list], rng: random.Random) -> list:
    """<= MAX_SOLS_PER_Q per task, >=1 per included task, spread across tasks."""
    tasks = list(by_task.keys())
    rng.shuffle(tasks)
    pools = {t: rng.sample(sols, len(sols)) for t, sols in by_task.items()}
    taken = defaultdict(int)
    chosen = []
    for _ in range(config.MAX_SOLS_PER_Q):
        if len(chosen) >= config.PER_GROUP:
            break
        progressed = False
        for t in tasks:
            if len(chosen) >= config.PER_GROUP:
                break
            cap = min(config.MAX_SOLS_PER_Q, len(pools[t]))
            if taken[t] < cap:
                chosen.append(pools[t][taken[t]])
                taken[t] += 1
                progressed = True
        if not progressed:
            break
    return chosen


def build() -> pd.DataFrame:
    rows = []
    for model in config.MODELS:
        for benchmark in config.BENCHMARKS:
            group = config.group_name(model, benchmark)
            ds = get_dataset(benchmark)
            by_task: dict[str, list] = defaultdict(list)
            for tid, ordn, content, tokens, gt in eligible_generations(model, benchmark):
                by_task[tid].append((tid, ordn, content, tokens, gt))
            rng = _group_rng(group)
            chosen = round_robin_sample(by_task, rng)
            distinct_q = len({c[0] for c in chosen})
            status = "OK" if len(chosen) >= config.PER_GROUP else f"SHORT({len(chosen)})"
            print(
                f"{group:32} eligible={sum(len(v) for v in by_task.values()):5} "
                f"tasks={len(by_task):4} chosen={len(chosen):4} distinctQ={distinct_q:4} {status}"
            )
            mx = config.HIGHEST_VARIANT[benchmark]
            for tid, ordn, content, tokens, gt in chosen:
                q = ds.tasks[tid].prompt if tid in ds.tasks else ""
                rows.append(
                    {
                        "row_id": f"{model}|{benchmark}|{tid}|g{ordn}",
                        "model": model,
                        "state": "think",
                        "benchmark": benchmark,
                        "group": group,
                        "task_id": tid,
                        "question_text": q,
                        "ground_truth": gt,
                        "answer_type": classify_answer_type(gt),
                        "source_variant": mx,
                        "source_gen_ordinal": ordn,
                        "solution_text": content,
                        "solution_len_tokens": tokens,
                        "solution_len_chars": len(content),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df = build()
    out = os.path.join(config.OUTPUT_DIR, "base_sample.parquet")
    df.to_parquet(out, index=False)
    out_csv = out.replace(".parquet", ".csv")
    df.to_csv(out_csv, index=False)  # QUOTE_MINIMAL: long/multiline LaTeX fields get quoted
    print(f"\nTOTAL base rows: {len(df)}  (expected {config.PER_GROUP * len(config.MODELS) * len(config.BENCHMARKS)})")
    print(f"answer_type mix: {df['answer_type'].value_counts().to_dict()}")
    print(f"wrote {out}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
