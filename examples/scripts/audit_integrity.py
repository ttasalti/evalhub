#!/usr/bin/env python
"""Read-only integrity sweep over every CoT judge cell in a results tree.

Runs the five invariants the CoT-Pass@K pipeline must satisfy, on EVERY
``judged_by/<judge>/<benchmark>/`` cell, and reports any violation. Touches no
files. Use it after a data fix (e.g. ``trim_judge_votes.py``) or before
publishing ``report.csv`` to prove the join between base eval, judge votes, and
CoT metrics is sound.

Invariants (per cell):
  1. content   — cot_results.solutions == base_results.solutions, task by task.
                 The CoT stage only relabels ``correct``; it must never alter the
                 generations themselves. A mismatch means the judge read a
                 different base run (cross-wire).
  2. count     — base_true == cot.true_count + cot.cot_false_count. Every base
                 correct generation is either still true or vetoed; none vanish.
  3. votes     — every base-correct generation id has EXACTLY ``--votes`` (3)
                 judge verdicts; total verdicts == base_true * votes. Catches the
                 append-duplication bug (6 votes) and partial judge runs.
  4. monotone  — for every metric/k, CoT value <= No-Judge value. The veto can
                 only remove correct answers, so CoT-Pass@K can never exceed
                 plain Pass@K.
  5. ids       — set(judged generation ids) == set(base-correct generation ids).
                 The judge must score exactly the base-correct set — no extra ids
                 (cross-wire / wrong state) and none missing (truncated run).

    python scripts/audit_integrity.py                       # sweep results/
    python scripts/audit_integrity.py --results-root results
    python scripts/audit_integrity.py --only think          # filter by path substring
    python scripts/audit_integrity.py -v                    # list every PASS too
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evalhub.cot.ids import encode as encode_generation_id  # noqa: E402

EXPECT_VOTES = 3
TOL = 1e-9
TESTS = ("content", "count", "votes", "monotone", "ids", "text")

# Aux jsonl files that sit next to the real judge-solutions file and also match
# the ``cot_judge*.jsonl`` glob. The solutions file is e.g. ``cot_judge.jsonl`` /
# ``cot_judge_tr.jsonl``; these must never be mistaken for it.
_AUX_SUFFIXES = ("_raw.jsonl", "_results.jsonl", "_generation_stats.jsonl")


def is_solution_file(name: str) -> bool:
    return name.startswith("cot_judge") and name.endswith(".jsonl") and not name.endswith(_AUX_SUFFIXES)


def _iter_records(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def base_facts(base_results: Path) -> tuple[dict[str, list], set[str], int]:
    """Return (solutions_by_task, correct_gen_ids, base_true_count)."""
    sols: dict[str, list] = {}
    correct_ids: set[str] = set()
    base_true = 0
    for r in _iter_records(base_results):
        tid = r["task_id"]
        sols[tid] = list(r.get("solutions") or [])
        for i, v in enumerate(r.get("correct") or []):
            if v is True:
                correct_ids.add(encode_generation_id(tid, i))
                base_true += 1
    return sols, correct_ids, base_true


def cot_solutions(cot_results: Path) -> dict[str, list]:
    return {r["task_id"]: list(r.get("solutions") or []) for r in _iter_records(cot_results)}


def judged_vote_counts(judge_sol: Path) -> Counter:
    c: Counter = Counter()
    for r in _iter_records(judge_sol):
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


def judge_input_text_mismatch(cell: Path, bench: str) -> int:
    """How many judged generations have text != the own-state base generation.

    Returns -1 when there is no ``cot_judge_input.jsonl`` / base raw to verify
    against (the file is often pruned). This is the only check that detects a
    cross-wire whose vote ids were rewritten to *look* aligned, since the input
    records preserve the actual generation text that was scored.
    """
    inp = cell / f"{bench}_cot_judge_input.jsonl"
    base_raw = cell.parents[2] / bench / f"{bench}_raw.jsonl"
    if not inp.exists() or not base_raw.exists():
        return -1
    bm: dict[tuple[str, int], str] = {}
    idxc: Counter = Counter()
    for r in _iter_records(base_raw):
        tid = r["task_id"]
        bm[(tid, idxc[tid])] = _content(r["response"]).strip()
        idxc[tid] += 1
    bad = 0
    for r in _iter_records(inp):
        key = (r["original_task_id"], int(r["generation_idx"]))
        if bm.get(key) != _content(r["raw_response"]).strip():
            bad += 1
    return bad


def find_judge_cells(root: Path) -> list[Path]:
    cells: set[Path] = set()
    for sol in root.rglob("cot_judge*.jsonl"):
        if not is_solution_file(sol.name):
            continue
        d = sol.parent
        if d.parent.parent.name == "judged_by":
            cells.add(d)
    return sorted(cells)


def _metric_violations(base_sum: dict, cot_sum: dict) -> list[str]:
    """Any k/metric where CoT > No-Judge (within tolerance).

    ``pass_at_k`` and ``mg_pass_at_k`` are flat ``{k: value}``; ``g_pass_at_k`` is
    nested ``{k: {threshold: value}}``.
    """
    bad: list[str] = []
    for block in ("pass_at_k", "mg_pass_at_k"):
        b = base_sum.get(block) or {}
        c = cot_sum.get(block) or {}
        for k in sorted(set(b) & set(c), key=lambda x: int(x)):
            if c[k] > b[k] + TOL:
                bad.append(f"{block}@{k}: cot={c[k]:.4f} > base={b[k]:.4f}")
    bg = base_sum.get("g_pass_at_k") or {}
    cg = cot_sum.get("g_pass_at_k") or {}
    for k in sorted(set(bg) & set(cg), key=lambda x: int(x)):
        bt, ct = bg[k] or {}, cg[k] or {}
        for thr in sorted(set(bt) & set(ct), key=lambda x: float(x)):
            if ct[thr] > bt[thr] + TOL:
                bad.append(f"g_pass_at_k@{k}/{thr}: cot={ct[thr]:.4f} > base={bt[thr]:.4f}")
    bc, cc = base_sum.get("cons_at_k"), cot_sum.get("cons_at_k")
    if bc is not None and cc is not None and cc > bc + TOL:
        bad.append(f"cons_at_k: cot={cc:.4f} > base={bc:.4f}")
    return bad


def audit_cell(cell: Path, expect_votes: int = EXPECT_VOTES) -> tuple[dict[str, bool], list[str]]:
    """Return (test -> passed, detail messages). Missing inputs => that test fails."""
    bench = cell.name
    base_dir = cell.parents[2] / bench
    base_results = base_dir / f"{bench}_results.jsonl"
    base_summary = base_dir / f"{bench}_summary.json"
    cot_results = cell / f"{bench}_cot_results.jsonl"
    cot_summary = cell / f"{bench}_cot_summary.json"
    judge_sols = [p for p in cell.glob("cot_judge*.jsonl") if is_solution_file(p.name)]

    res = dict.fromkeys(TESTS, True)
    notes: list[str] = []

    if not base_results.exists():
        return dict.fromkeys(TESTS, False), [f"base results missing: {base_results}"]

    base_sols, base_correct_ids, base_true = base_facts(base_results)
    # A cell with no base-correct generations has nothing to judge: CoT-Pass@K=0
    # by definition, the pipeline takes its empty-summary early-exit, and the
    # per-generation cot_results.jsonl is legitimately never written. So a missing
    # results file / empty-or-zero summary is CORRECT here, not a failure.
    empty_cell = base_true == 0

    # 1. content
    if not cot_results.exists():
        if not empty_cell:
            res["content"] = False
            notes.append("cot_results.jsonl missing")
    else:
        c_sols = cot_solutions(cot_results)
        mism = [t for t in base_sols if t in c_sols and c_sols[t] != base_sols[t]]
        missing = set(base_sols) - set(c_sols)
        if mism or missing:
            res["content"] = False
            notes.append(f"solutions mismatch: {len(mism)} task(s) differ, {len(missing)} missing")

    # 2. count-identity + 4. monotone (both need the cot summary)
    cs = orjson.loads(cot_summary.read_bytes()) if cot_summary.exists() else None
    if not cs:  # missing or empty {} summary
        if not empty_cell:
            res["count"] = res["monotone"] = False
            notes.append("cot_summary.json missing/empty")
    else:
        derived = (cs.get("true_count") or 0) + (cs.get("cot_false_count") or 0)
        if derived != base_true:
            res["count"] = False
            notes.append(f"count: base_true={base_true} != cot_true+cot_false={derived}")
        if not base_summary.exists():
            res["monotone"] = False
            notes.append("base_summary.json missing")
        else:
            bs = orjson.loads(base_summary.read_bytes())
            viol = _metric_violations(bs, cs)
            if viol:
                res["monotone"] = False
                notes.append("cot>nojudge: " + "; ".join(viol[:3]) + (" ..." if len(viol) > 3 else ""))

    # 3. votes + 5. ids (both need the judge solutions file)
    if not judge_sols:
        if base_true == 0:
            pass  # no correct generations => empty judge set is correct
        else:
            res["votes"] = res["ids"] = False
            notes.append("judge solutions file missing")
    else:
        counts = judged_vote_counts(judge_sols[0])
        judged_ids = set(counts)
        if judged_ids != base_correct_ids:
            res["ids"] = False
            res["votes"] = False  # vote-count check is meaningless if the set is wrong
            extra = len(judged_ids - base_correct_ids)
            miss = len(base_correct_ids - judged_ids)
            notes.append(f"ids: judged={len(judged_ids)} base_correct={len(base_correct_ids)} "
                         f"(extra={extra}, missing={miss})")
        else:
            dist = dict(sorted(Counter(counts.values()).items()))
            off = {v: n for v, n in dist.items() if v != expect_votes}
            if off:
                res["votes"] = False
                notes.append(f"votes!={expect_votes}: dist={dist}")

    # 6. text — judged generations must be the OWN-state base generations.
    # Only checkable where cot_judge_input.jsonl survives (-1 => skipped).
    text_bad = judge_input_text_mismatch(cell, bench)
    if text_bad > 0:
        res["text"] = False
        notes.append(f"text: {text_bad} judged generation(s) != own-state base (CROSS-WIRE)")

    return res, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--votes", type=int, default=EXPECT_VOTES)
    ap.add_argument("--only", default=None, help="only cells whose path contains this substring")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also audit Ministral/Mistral cells (excluded from the report by default)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print passing cells")
    args = ap.parse_args()

    root = Path(args.results_root)
    cells = find_judge_cells(root)
    if args.only:
        cells = [c for c in cells if args.only in str(c)]
    if not args.include_excluded:
        cells = [c for c in cells if not any(p in str(c).lower() for p in ("ministral", "mistral"))]
    if not cells:
        print(f"No judge cells under {root}")
        return 0

    fail_tally = dict.fromkeys(TESTS, 0)
    failing_cells = 0
    for cell in cells:
        res, notes = audit_cell(cell, args.votes)
        ok = all(res.values())
        if not ok:
            failing_cells += 1
            for t in TESTS:
                if not res[t]:
                    fail_tally[t] += 1
        if ok and not args.verbose:
            continue
        label = str(cell).replace(f"{root}/", "").replace("/judged_by", "")
        tag = "PASS" if ok else "FAIL[" + ",".join(t for t in TESTS if not res[t]) + "]"
        print(f"[{tag}] {label}")
        for n in notes:
            print(f"     - {n}")

    print(f"\n=== {len(cells)} cell(s): {len(cells) - failing_cells} pass, {failing_cells} fail ===")
    for t in TESTS:
        mark = "OK" if fail_tally[t] == 0 else f"{fail_tally[t]} FAIL"
        print(f"   {t:9s}: {mark}")
    return 1 if failing_cells else 0


if __name__ == "__main__":
    raise SystemExit(main())
