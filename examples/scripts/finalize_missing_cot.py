#!/usr/bin/env python
"""Re-finalize CoT cells whose judge votes are intact but metrics are missing/broken.

Some cells have valid, aligned judge verdicts on disk but no (or an empty/stale)
``*_cot_summary.json`` / ``*_cot_results.jsonl`` — e.g. ``cot finalize`` was never
run, was interrupted (empty ``{}`` summary), or the per-generation results file
was deleted in a space cleanup. The judge inference already happened, so this is
recoverable **without a GPU**: re-run the majority vote + CoT metrics from the
existing ``cot_judge*.jsonl`` solutions.

A cell is only touched when its judged generation ids EXACTLY match the
base-correct ids (so we never finalize a cross-wired or truncated run). Cells with
no votes (judge never produced output) or misaligned votes are reported and
skipped — those need a GPU re-judge.

    python scripts/finalize_missing_cot.py                 # dry-run
    python scripts/finalize_missing_cot.py --execute       # apply (with backup)
    python scripts/finalize_missing_cot.py --exclude n128   # skip a target class
    python scripts/finalize_missing_cot.py --only think     # restrict by substring
"""

from __future__ import annotations

import argparse
import ast
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evalhub.cot.aggregate import aggregate_judge_votes  # noqa: E402
from evalhub.cot.ids import encode as encode_generation_id  # noqa: E402
from evalhub.cot.metrics import apply_cot_metrics  # noqa: E402

ROOT = Path("results")
BACKUP_DIR = Path(".migration_backup")
_AUX_SUFFIXES = ("_raw.jsonl", "_results.jsonl", "_generation_stats.jsonl")


def is_solution_file(name: str) -> bool:
    return name.startswith("cot_judge") and name.endswith(".jsonl") and not name.endswith(_AUX_SUFFIXES)


def _iter(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def base_facts(base_results: Path) -> tuple[set[str], int]:
    ids: set[str] = set()
    true = 0
    for r in _iter(base_results):
        tid = r["task_id"]
        for i, v in enumerate(r.get("correct") or []):
            if v is True:
                ids.add(encode_generation_id(tid, i))
                true += 1
    return ids, true


def judged_counts(judge_sol: Path) -> Counter:
    c: Counter = Counter()
    for r in _iter(judge_sol):
        tid = r.get("task_id")
        if tid:
            c[tid] += 1
    return c


def _content(resp):
    if isinstance(resp, str):
        try:
            resp = ast.literal_eval(resp)
        except Exception:
            return resp
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return str(resp)
    return str(resp)


def _base_text_map(base_raw: Path) -> dict[tuple[str, int], str]:
    m: dict[tuple[str, int], str] = {}
    idxc: Counter = Counter()
    for r in _iter(base_raw):
        tid = r["task_id"]
        i = idxc[tid]
        idxc[tid] += 1
        m[(tid, i)] = _content(r["response"]).strip()
    return m


def judge_input_text_mismatch(cell: Path, bench: str) -> int:
    """Count judged generations whose text != the own-state base generation.

    The id-set guard alone can't catch a cross-wire whose vote ids were rewritten
    to look aligned (the ``cot_judge_input.jsonl`` still proves which generations
    were actually scored). Returns -1 if there is no judge_input/base_raw to check.
    """
    inp = cell / f"{bench}_cot_judge_input.jsonl"
    base_raw = cell.parents[2] / bench / f"{bench}_raw.jsonl"
    if not inp.exists() or not base_raw.exists():
        return -1
    bm = _base_text_map(base_raw)
    bad = 0
    for r in _iter(inp):
        key = (r["original_task_id"], int(r["generation_idx"]))
        if bm.get(key) != _content(r["raw_response"]).strip():
            bad += 1
    return bad


def find_cells() -> list[Path]:
    cells: set[Path] = set()
    for sol in ROOT.rglob("cot_judge*.jsonl"):
        if not is_solution_file(sol.name):
            continue
        d = sol.parent
        if d.parent.parent.name == "judged_by":
            cells.add(d)
    return sorted(cells)


def needs_repair(cot_results: Path, cot_summary: Path, base_true: int) -> bool:
    if not cot_results.exists():
        return True
    if not cot_summary.exists():
        return True
    s = orjson.loads(cot_summary.read_bytes())
    if not s:  # empty {} summary
        return True
    derived = (s.get("true_count") or 0) + (s.get("cot_false_count") or 0)
    return derived != base_true


def backup(cell: Path) -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    try:
        slug = cell.relative_to(ROOT)
    except ValueError:
        slug = cell
    slug = str(slug).replace("/", "__")
    tar = BACKUP_DIR / f"finalize_{slug}_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(cell, arcname=str(cell))
    return tar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="apply (default: dry-run)")
    ap.add_argument("--only", default=None, help="only cells whose path contains this substring")
    ap.add_argument("--exclude", default=None, help="skip cells whose path contains this substring")
    args = ap.parse_args()

    cells = find_cells()
    if args.only:
        cells = [c for c in cells if args.only in str(c)]
    if args.exclude:
        cells = [c for c in cells if args.exclude not in str(c)]

    repaired = empty_votes = misaligned = healthy = 0
    for cell in cells:
        bench = cell.name
        base_dir = cell.parents[2] / bench
        base_results = base_dir / f"{bench}_results.jsonl"
        cot_results = cell / f"{bench}_cot_results.jsonl"
        cot_summary = cell / f"{bench}_cot_summary.json"
        majority = cell / f"{bench}_cot_majority.jsonl"
        stats = cell / f"{bench}_cot_stats.json"
        sols = [p for p in cell.glob("cot_judge*.jsonl") if is_solution_file(p.name)]
        label = str(cell).replace(f"{ROOT}/", "").replace("/judged_by", "")

        if not base_results.exists():
            print(f"[NO-BASE] {label}")
            continue

        base_ids, base_true = base_facts(base_results)
        if not needs_repair(cot_results, cot_summary, base_true):
            healthy += 1
            continue

        judged = judged_counts(sols[0]) if sols else Counter()
        jids = set(judged)

        if not jids:
            print(f"[EMPTY-VOTES] {label}  base_true={base_true} -> judge produced 0 verdicts; GPU re-judge needed")
            empty_votes += 1
            continue
        if jids != base_ids:
            print(f"[MISALIGNED]  {label}  judged={len(jids)} base_correct={len(base_ids)} "
                  f"(extra={len(jids - base_ids)}, missing={len(base_ids - jids)}); GPU re-judge needed")
            misaligned += 1
            continue

        # Text-level guard: even with aligned ids, the judge may have scored a
        # different state's generations (cross-wire). Never re-finalize those.
        text_bad = judge_input_text_mismatch(cell, bench)
        if text_bad > 0:
            print(f"[CROSS-WIRE]  {label}  {text_bad} judged generation(s) text != own-state base; "
                  f"GPU re-judge needed (do NOT re-finalize)")
            misaligned += 1
            continue

        dist = dict(sorted(Counter(judged.values()).items()))
        print(f"[{'FIX' if args.execute else 'WILL-FIX'}] {label}  votes/id={dist}  base_true={base_true}")
        if not args.execute:
            repaired += 1
            continue

        backup(cell)
        aggregate_judge_votes(sols[0], majority)
        summary = apply_cot_metrics(base_results, majority, cot_results, cot_summary, stats)
        print(f"    -> true={summary['true_count']} cot_false={summary['cot_false_count']} "
              f"pass@1={summary['pass_at_k'].get('1')}")
        repaired += 1

    print(f"\n{'Applied' if args.execute else 'Dry-run'}: {repaired} repairable, "
          f"{empty_votes} empty-votes, {misaligned} misaligned, {healthy} already-healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
