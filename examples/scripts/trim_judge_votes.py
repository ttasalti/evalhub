#!/usr/bin/env python
"""Trim over-judged CoT cells down to a uniform 3 votes per correct generation.

Root cause: ``evalhub gen`` appended on a re-run (``base.py`` opened the output
files in ``"ab"``), so a judge stage run twice produced 6 votes per generation
(or a mix of 3/4/5/6 when interrupted). This keeps the **first 3** votes per
generation id and re-derives the CoT majority + metrics from them, so every
correct answer ends up with exactly 3 cot votes — consistent with every other
cell.

Only cells where some generation id has **>3** votes are touched. Cells already
uniform at 3 (incl. the separately re-judged cross-wired cell) are left alone. A
``.tar.gz`` backup of each touched cell is taken first. The original judge
verdicts are reused — no model inference (no GPU) is performed.

    python scripts/trim_judge_votes.py             # dry-run (report only)
    python scripts/trim_judge_votes.py --execute   # apply (with backup)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evalhub.cot.aggregate import aggregate_judge_votes  # noqa: E402
from evalhub.cot.metrics import apply_cot_metrics  # noqa: E402

ROOT = Path("results")
BACKUP_DIR = Path(".migration_backup")
KEEP = 3
TID = re.compile(rb'"task_id"\s*:\s*"([^"]+)"')


def per_id_counts(path: Path) -> Counter:
    c: Counter = Counter()
    for line in open(path, "rb"):
        m = TID.search(line)
        if m:
            c[m.group(1).decode()] += 1
    return c


def base_correct_ids(base_path: Path) -> set[str]:
    """Generation ids (``<task_id>_gen_<idx>``) the base evaluator marked correct."""
    ids: set[str] = set()
    for line in open(base_path, "rb"):
        if not line.strip():
            continue
        r = orjson.loads(line)
        tid = r["task_id"]
        for i, v in enumerate(r.get("correct") or []):
            if v is True:
                ids.add(f"{tid}_gen_{i}")
    return ids


# Aux jsonl files that also match the ``cot_judge*.jsonl`` glob but are NOT the
# solutions file (which is e.g. ``cot_judge.jsonl`` / ``cot_judge_tr.jsonl``).
_AUX_SUFFIXES = ("_raw.jsonl", "_results.jsonl", "_generation_stats.jsonl")


def find_solution_files() -> list[Path]:
    out = []
    for sol in ROOT.rglob("cot_judge*.jsonl"):
        if sol.name.endswith(_AUX_SUFFIXES):
            continue
        # must live in a judge benchmark dir: .../judged_by/<judge>/<benchmark>/
        d = sol.parent
        if d.parent.parent.name != "judged_by":
            continue
        out.append(sol)
    return sorted(out)


def majority_map(sol_path: Path) -> dict[str, bool]:
    """Run the real majority-vote aggregator on ``sol_path`` -> {gen_id: approved}."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    try:
        aggregate_judge_votes(sol_path, tmp.name)
        return {
            r["task_id"]: (str(r["majority_correct"]).lower() == "true")
            for r in (orjson.loads(line) for line in open(tmp.name, "rb") if line.strip())
        }
    finally:
        os.unlink(tmp.name)


def first_k_lines(path: Path, keep: int = KEEP) -> bytes:
    seen: Counter = Counter()
    out: list[bytes] = []
    for line in open(path, "rb"):
        m = TID.search(line)
        if not m:
            continue
        tid = m.group(1).decode()
        if seen[tid] < keep:
            out.append(line if line.endswith(b"\n") else line + b"\n")
            seen[tid] += 1
    return b"".join(out)


def backup(cell_dir: Path) -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    # Slug from the full cell path (not just the benchmark name): different cells
    # share a benchmark name (e.g. many ``aime2026_tr``), so a name+second token
    # collides and silently overwrites a sibling's backup.
    try:
        slug = cell_dir.relative_to(ROOT)
    except ValueError:
        slug = cell_dir
    slug = str(slug).replace("/", "__")
    tar = BACKUP_DIR / f"trim_votes_{slug}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(cell_dir, arcname=str(cell_dir))
    return tar


def _summary_pass(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    s = orjson.loads(summary_path.read_bytes())
    pk = s.get("pass_at_k", {})
    return {"pass@1": pk.get("1"), "pass@64": pk.get("64"), "true_count": s.get("true_count")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="apply (default: dry-run)")
    args = ap.parse_args()

    sols = find_solution_files()
    targets = []
    for sol in sols:
        cnt = per_id_counts(sol)
        if cnt and max(cnt.values()) > KEEP:
            targets.append((sol, cnt))

    print(f"{len(sols)} judge cell(s); {len(targets)} with >3 votes/generation (to trim)\n")
    if not targets:
        print("Nothing to trim.")
        return 0

    fixed = skipped = 0
    for sol, cnt in targets:
        d = sol.parent
        bench = d.name
        base = d.parents[2] / bench / f"{bench}_results.jsonl"
        maj = d / f"{bench}_cot_majority.jsonl"
        res = d / f"{bench}_cot_results.jsonl"
        summ = d / f"{bench}_cot_summary.json"
        stats = d / f"{bench}_cot_stats.json"
        dist = dict(sorted(Counter(cnt.values()).items()))
        label = str(d).replace("results/", "").replace("/judged_by", "")

        # guard 1: every generation must have >= KEEP votes to keep KEEP
        min_votes = min(cnt.values())
        if min_votes < KEEP:
            print(f"[SKIP] {label}\n   votes/id dağılımı={dist} -> bir id'de <{KEEP} oy ({min_votes}); ATLANDI")
            skipped += 1
            continue

        # guard 2: judged generation ids must == base-correct ids (alignment).
        # Catches cross-wired / misaligned judge runs so we never finalize them.
        if not base.exists():
            print(f"[SKIP] {label}\n   base results yok: {base}; ATLANDI")
            skipped += 1
            continue
        sol_ids = set(cnt.keys())
        bc_ids = base_correct_ids(base)
        if sol_ids != bc_ids:
            extra = len(sol_ids - bc_ids)
            missing = len(bc_ids - sol_ids)
            print(f"[SKIP] {label}\n   votes/id={dist} | judge_ids={len(sol_ids)} base_correct_ids={len(bc_ids)} "
                  f"(judge-only={extra}, base-only={missing}) -> HİZASIZ (cross-wire?), ATLANDI")
            skipped += 1
            continue

        # verify the over-voted run is self-consistent: full vs first-3 majority
        agree_note = ""
        try:
            full_maj = majority_map(sol)
            tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
            tmp.write(first_k_lines(sol))
            tmp.close()
            k3_maj = majority_map(Path(tmp.name))
            os.unlink(tmp.name)
            common = set(full_maj) & set(k3_maj)
            agree = sum(1 for t in common if full_maj[t] == k3_maj[t])
            agree_note = f"majority uyum (tam vs ilk-3): {agree}/{len(common)}"
        except Exception as e:  # pragma: no cover
            agree_note = f"(uyum hesaplanamadı: {e})"

        before = _summary_pass(summ)
        print(f"[{'TRIM' if args.execute else 'DRY'}] {label}")
        print(f"   votes/id dağılımı={dist}  ids={len(cnt)}  {agree_note}")

        if not args.execute:
            print(f"   önce: {before}")
            continue

        backup(d)
        # trim solutions + raw
        sol.write_bytes(first_k_lines(sol))
        for raw in d.glob("cot_judge*_raw.jsonl"):
            raw.write_bytes(first_k_lines(raw))
        # re-derive majority + metrics from the trimmed 3 votes
        aggregate_judge_votes(sol, maj)
        apply_cot_metrics(base, maj, res, summ, stats)
        after = _summary_pass(summ)
        # sanity: every id now exactly 3
        post = per_id_counts(sol)
        ok = all(v == KEEP for v in post.values())
        print(f"   önce: {before}\n   sonra:{after}  | her id=3 mi: {ok}")
        fixed += 1

    print(f"\n{'Uygulandı' if args.execute else 'Dry-run'}: {fixed} hücre trim, {skipped} atlandı.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
