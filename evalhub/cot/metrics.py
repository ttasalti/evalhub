"""Apply CoT judge verdicts to base results and recompute Pass@K / Cons@K.

The judge approves/vetoes each answer-correct generation via ``judge_n_samples``
verdicts (``yes``/``no``/``invalid``, counted in ``<bench>_cot_majority.jsonl``).
"correct" = judge boxed ``yes``. A generation survives the veto under three
**approval thresholds** of those judge samples:

  * ``majority``: survives iff ``yes > no`` (ties vetoed). This is the historical
    behaviour; its metrics keep the unprefixed summary keys / CSV columns.
  * ``any``: survives iff ``yes >= 1`` (vetoed only if no sample said yes).
    Loosest -> most generations survive -> HIGHEST cot metrics.
  * ``all``: survives iff unanimous yes (no ``no`` and no ``invalid``).
    Strictest -> fewest survive -> LOWEST cot metrics.

The full Pass@K / Cons@K / G-Pass@K / mG-Pass@K suite is recomputed for every
threshold; ``any`` and ``all`` results are written with ``_any`` / ``_all`` suffixes
alongside the (unsuffixed) majority ones. Monotonicity: ``all <= majority <= any``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from os import PathLike
from pathlib import Path
from typing import Any

import orjson

from evalhub.cot.ids import encode as encode_generation_id
from evalhub.utils.logger import logger
from evalhub.utils.metrics import aggregate_g_pass, compute_pass_at_k

DEFAULT_KS: list[int] = [2**i for i in range(11)]
COT_FALSE_LABEL = "cot_false"
# extra approval thresholds beyond the primary (unsuffixed) majority one
EXTRA_THRESHOLDS = ("any", "all")


def _verdict_class(token: object) -> str:
    """Classify one raw judge verdict as ``yes`` / ``no`` / ``invalid``.

    Mirrors ``evalhub.cot.aggregate._classify`` so counts derived here match how
    ``majority_correct`` was originally computed (anything but a bare yes/no is invalid,
    e.g. ``invalid_format`` or a stray boxed number)."""
    norm = (str(token) if token is not None else "").strip().lower()
    return norm if norm in ("yes", "no") else "invalid"


def _load_judge_counts(
    majority_path: Path,
) -> tuple[dict[str, bool], dict[str, tuple[int, int, int]]]:
    """Load per-generation judge verdicts from ``<bench>_cot_majority.jsonl``.

    Returns ``(majority_map, counts)`` where ``majority_map[gen_id]`` is the stored
    ``majority_correct`` boolean (used verbatim so the majority path stays byte-for-byte
    identical to the historical output) and ``counts[gen_id] = (yes, no, invalid)`` feeds
    the ``any`` / ``all`` thresholds.

    Two on-disk schemas exist: the current one carries ``yes_count`` / ``no_count`` /
    ``invalid_count`` directly; the older one carries only the raw verdict list
    ``solutions`` (e.g. ``["yes", "no", "invalid_format"]``), from which the counts are
    re-derived via :func:`_verdict_class`. Both reproduce the stored ``majority_correct``.
    """
    majority_map: dict[str, bool] = {}
    counts: dict[str, tuple[int, int, int]] = {}
    with majority_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            gid = record["task_id"]
            majority_map[gid] = bool(record["majority_correct"])
            if "yes_count" in record:
                counts[gid] = (
                    int(record.get("yes_count", 0)),
                    int(record.get("no_count", 0)),
                    int(record.get("invalid_count", 0)),
                )
            else:  # older schema: raw verdicts in `solutions`
                sols = record.get("solutions") or []
                classes = [_verdict_class(s) for s in sols]
                yes = classes.count("yes")
                no = classes.count("no")
                counts[gid] = (yes, no, len(classes) - yes - no)
    return majority_map, counts


def _survives(yes: int, no: int, invalid: int, threshold: str) -> bool:
    """Does an answer-correct generation survive the judge veto under ``threshold``?"""
    if threshold == "any":
        return yes >= 1
    if threshold == "all":
        return no == 0 and invalid == 0  # unanimous yes
    return yes > no  # majority (ties vetoed)


def _ks_from_record(record: dict[str, Any], n_generations: int) -> list[int]:
    declared = record.get("pass_at_k") or {}
    if declared:
        return sorted({int(k) for k in declared.keys() if int(k) <= n_generations})
    return [k for k in DEFAULT_KS if k <= n_generations]


def apply_cot_metrics(
    base_results_path: PathLike,
    majority_path: PathLike,
    output_results_path: PathLike,
    summary_path: PathLike,
    stats_path: PathLike | None = None,
) -> dict[str, Any]:
    """Re-evaluate base results under the CoT veto, for all three approval thresholds.

    Each generation the base evaluator marked correct is downgraded to ``"cot_false"``
    if the judge does not approve it under the threshold. Pass@K / Cons@K / G-Pass@K /
    mG-Pass@K are recomputed against the surviving true count per threshold. The
    ``majority`` threshold keeps the historical (unsuffixed) keys; ``any`` / ``all`` add
    ``_any`` / ``_all`` suffixed keys.
    """
    base_results_path = Path(base_results_path)
    majority_path = Path(majority_path)
    output_results_path = Path(output_results_path)
    summary_path = Path(summary_path)
    for p in (output_results_path, summary_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    if not base_results_path.exists():
        raise FileNotFoundError(f"Base results file missing: {base_results_path}")
    if not majority_path.exists():
        raise FileNotFoundError(f"Majority file missing: {majority_path}")

    majority_map, counts = _load_judge_counts(majority_path)

    # primary (majority) accumulators, unchanged from historical behaviour
    sum_pass_at_k: dict[str, float] = defaultdict(float)
    sum_cons_at_k = 0.0
    total_tasks = 0
    stats = {
        "total_tasks": 0,
        "total_generations": 0,
        "true_count": 0,
        "false_count": 0,
        "cot_false_count": 0,
        "invalid_count": 0,
    }
    csv_rows: list[dict[str, Any]] = []
    gpass_cot: list[tuple[int, int]] = []

    # extra thresholds (any / all) accumulators
    extra = {
        t: {
            "sum_pass": defaultdict(float),
            "sum_cons": 0.0,
            "gpass": [],
            "true": 0,
            "cot_false": 0,
        }
        for t in EXTRA_THRESHOLDS
    }

    with base_results_path.open("rb") as f_in, output_results_path.open("wb") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = orjson.loads(line)
            task_id = record["task_id"]
            correct_base: list[Any] = list(record.get("correct", []) or [])
            solutions: list[Any] = list(record.get("solutions", []) or [])
            n_generations = len(correct_base)
            ks = _ks_from_record(record, n_generations)

            sol_strs = ["" if s is None else str(s) for s in solutions] if solutions else []
            majority_answer = Counter(sol_strs).most_common(1)[0][0] if sol_strs else None

            # majority threshold (historical; drives results.jsonl + summary primary keys)
            correct_arr = list(correct_base)
            for i in range(n_generations):
                gen_id = encode_generation_id(task_id, i)
                if correct_arr[i] is True and gen_id in majority_map and not majority_map[gen_id]:
                    correct_arr[i] = COT_FALSE_LABEL

            true_count = sum(1 for x in correct_arr if x is True)
            false_count = sum(1 for x in correct_arr if x is False)
            cot_false_count = sum(1 for x in correct_arr if x == COT_FALSE_LABEL)
            invalid_count = n_generations - true_count - false_count - cot_false_count

            stats["total_tasks"] += 1
            stats["total_generations"] += n_generations
            stats["true_count"] += true_count
            stats["false_count"] += false_count
            stats["cot_false_count"] += cot_false_count
            stats["invalid_count"] += invalid_count

            gpass_cot.append((n_generations, true_count))

            new_pass_at_k: dict[str, float] = {}
            for k in ks:
                value = compute_pass_at_k(n_generations, true_count, k)
                new_pass_at_k[str(k)] = value
                sum_pass_at_k[str(k)] += value

            is_consensus_correct = False
            if sol_strs:
                is_consensus_correct = any(
                    correct_arr[i] is True for i, sol in enumerate(sol_strs) if sol == majority_answer
                )
                if is_consensus_correct:
                    sum_cons_at_k += 1.0

            total_tasks += 1
            record["correct"] = correct_arr
            record["pass_at_k"] = new_pass_at_k
            record["per_task_counts"] = {
                "true": true_count,
                "false": false_count,
                "cot_false": cot_false_count,
                "invalid_format": invalid_count,
            }
            record["is_correct_majority"] = is_consensus_correct

            # extra thresholds (any / all)
            for t in EXTRA_THRESHOLDS:
                arr = list(correct_base)
                for i in range(n_generations):
                    gen_id = encode_generation_id(task_id, i)
                    if arr[i] is True and gen_id in counts and not _survives(*counts[gen_id], t):
                        arr[i] = COT_FALSE_LABEL
                t_true = sum(1 for x in arr if x is True)
                t_cot_false = sum(1 for x in arr if x == COT_FALSE_LABEL)
                extra[t]["true"] += t_true
                extra[t]["cot_false"] += t_cot_false
                extra[t]["gpass"].append((n_generations, t_true))
                for k in ks:
                    extra[t]["sum_pass"][str(k)] += compute_pass_at_k(n_generations, t_true, k)
                if sol_strs and any(arr[i] is True for i, sol in enumerate(sol_strs) if sol == majority_answer):
                    extra[t]["sum_cons"] += 1.0
                record[f"true_{t}"] = t_true
                record[f"cot_false_{t}"] = t_cot_false

            f_out.write(orjson.dumps(record) + b"\n")
            csv_rows.append(record)

    if total_tasks == 0:
        raise ValueError(f"Base results file produced no records: {base_results_path}")

    pass_at_k_summary = {k: v / total_tasks for k, v in sum_pass_at_k.items()}
    cons_at_k = sum_cons_at_k / total_tasks
    ks = sorted({int(k) for k in pass_at_k_summary})
    g_pass_cot, mg_pass_cot = aggregate_g_pass(gpass_cot, ks)

    summary = {
        "pass_at_k": pass_at_k_summary,
        "g_pass_at_k": g_pass_cot,
        "mg_pass_at_k": mg_pass_cot,
        "cons_at_k": cons_at_k,
        "total_tasks": stats["total_tasks"],
        "total_generations": stats["total_generations"],
        "true_count": stats["true_count"],
        "false_count": stats["false_count"],
        "cot_false_count": stats["cot_false_count"],
        "invalid_format_count": stats["invalid_count"],
    }

    # any / all threshold blocks (false_count / invalid are threshold-invariant)
    for t in EXTRA_THRESHOLDS:
        e = extra[t]
        pk = {k: v / total_tasks for k, v in e["sum_pass"].items()}
        gp, mgp = aggregate_g_pass(e["gpass"], ks)
        summary[f"pass_at_k_{t}"] = pk
        summary[f"g_pass_at_k_{t}"] = gp
        summary[f"mg_pass_at_k_{t}"] = mgp
        summary[f"cons_at_k_{t}"] = e["sum_cons"] / total_tasks
        summary[f"true_count_{t}"] = e["true"]
        summary[f"cot_false_count_{t}"] = e["cot_false"]

    with summary_path.open("wb") as f_sum:
        f_sum.write(orjson.dumps(summary))

    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("wb") as f_stats:
            f_stats.write(orjson.dumps(stats))

    # Per-task CSV alongside the JSONL. Threshold count columns (true_any/cot_false_any/
    # true_all/cot_false_all) are appended so report_tasks.csv can surface all three.
    from evalhub.benchmarks.math.base import write_per_task_csv

    csv_path = output_results_path.with_name(output_results_path.stem.replace("_results", "_per_task") + ".csv")
    write_per_task_csv(
        csv_rows,
        csv_path,
        has_cot=True,
        extra_fields=["true_any", "cot_false_any", "true_all", "cot_false_all"],
    )
    logger.info(f"Per-task CSV saved to {csv_path}")

    for k, value in pass_at_k_summary.items():
        logger.info(f"CoT-Pass@{k}: {value:.4f}")
    logger.info(f"CoT-Cons@K:   {cons_at_k:.4f} over {total_tasks} tasks")

    return summary
