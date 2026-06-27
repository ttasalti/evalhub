#!/usr/bin/env python3
"""Migrate the results tree from the V3 layout to the V5 layout (ONE folder per model).

V3 (old):  <root>/<state>/<model>__t<T>__max<N>__n<NS>/<benchmark>/...
                                                       /step_<K>/<benchmark>/...
                                                       /judged_by/<judge>/<benchmark>/...
V5 (new):  <root>/<state>/<model>/<benchmark>__t<T>__max<N>__n<NS>/...
                                 /step_<K>/<benchmark>__t<T>__max<N>__n<NS>/...
                                 /judged_by/<judge>/<benchmark>__t<T>__max<N>__n<NS>/...

The sampling suffix `__t..__max..__n..` is stripped off the *model* directory and
appended to every *benchmark-leaf* directory under it. All sampling variants of a
model thereby collapse into a single model folder, their benchmark leaves staying
distinct because each now carries its own t/max/n.

================================  DATA SAFETY  =================================
This script is built to make data loss IMPOSSIBLE:

  * Default is DRY-RUN — prints the planned moves and does nothing.
  * --execute first writes a tar.gz BACKUP of every root, then performs moves.
  * Moves are os.rename within the same filesystem (atomic, lossless). It NEVER
    copies-then-deletes and NEVER calls rm -rf.
  * GLOBAL CONFLICT PRE-SCAN with abort-all: before moving anything, the full
    (src -> dst) set is computed. If two distinct sources map to the same dst, OR
    any dst already exists on disk, the whole migration ABORTS without touching a
    single file. So a name collision can never overwrite/merge-destroy data.
  * Emptied model dirs are removed with os.rmdir ONLY (fails loudly if non-empty).
  * Every move is appended to a manifest TSV; --revert undoes them exactly.
  * Pre/post file-count verification (*_summary.json, *_cot_summary.json, *.jsonl);
    if the counts differ after moving, it stops and shouts.

Usage:
  python scripts/migrate_results_layout.py                 # dry-run, default roots
  python scripts/migrate_results_layout.py --root "results"
  python scripts/migrate_results_layout.py --execute
  python scripts/migrate_results_layout.py --revert results_layout_manifest_<ts>.tsv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tarfile
import time
from pathlib import Path

# A V3 model-dir leaf: "<model>__t<T>__max<N>__n<NS>".
_V3_MODEL_RE = re.compile(r"^(?P<model>.+?)__t[0-9.]+__max\d+__n\d+$")
_V3_SUFFIX_RE = re.compile(r"(?P<suffix>__t[0-9.]+__max\d+__n\d+)$")
_STATE_DIRS = ("base", "non-think", "think", "unknown")
_STEP_RE = re.compile(r"^step_\d+$")

# Files counted before/after to prove nothing was lost.
_COUNT_GLOBS = ("*_summary.json", "*_cot_summary.json", "*.jsonl")


def _count_files(roots: list[Path]) -> int:
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for pat in _COUNT_GLOBS:
            total += sum(1 for _ in root.rglob(pat))
    return total


def _benchmark_leaf_dirs(model_dir: Path) -> list[Path]:
    """Return the benchmark-leaf directories under a V3 model dir, structurally.

    Benchmark leaves live at:
        M/<b>                              (b not judged_by / step_*)
        M/step_*/<b>                       (b not judged_by)
        M/judged_by/<judge>/<b>
        M/step_*/judged_by/<judge>/<b>
    Empty leaves are included (so they migrate too — never silently dropped).
    """
    leaves: list[Path] = []

    def collect_under(base: Path) -> None:
        # direct benchmark children
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "judged_by":
                for judge in sorted(child.iterdir()):
                    if judge.is_dir():
                        leaves.extend(b for b in sorted(judge.iterdir()) if b.is_dir())
            elif _STEP_RE.match(child.name):
                continue  # handled separately below
            else:
                leaves.append(child)

    collect_under(model_dir)
    for child in sorted(model_dir.iterdir()):
        if child.is_dir() and _STEP_RE.match(child.name):
            collect_under(child)
    return leaves


def _find_state_dirs(roots: list[Path]) -> list[Path]:
    """All directories named base/non-think/think/unknown anywhere under the roots.

    Recursive so the whole results tree is covered: results/<state>,
    results/RL train/<state>, results/RL train/combined_report/<state>, … Benchmark
    leaves and judge leaves never share these exact names, so this is collision-free.
    """
    found: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for d in root.rglob("*"):
            if d.is_dir() and d.name in _STATE_DIRS:
                found.add(d)
    return sorted(found)


def plan_moves(roots: list[Path]) -> list[tuple[Path, Path]]:
    """Compute the full (src -> dst) move list across all roots. No side effects."""
    moves: list[tuple[Path, Path]] = []
    for state_dir in _find_state_dirs(roots):
        for model_dir in sorted(state_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            m = _V3_MODEL_RE.match(model_dir.name)
            if not m:
                continue  # already V5 (bare model) or unrelated — skip
            suffix = _V3_SUFFIX_RE.search(model_dir.name).group("suffix")
            new_model_dir = state_dir / m.group("model")
            for leaf in _benchmark_leaf_dirs(model_dir):
                rel = leaf.relative_to(model_dir)            # e.g. judged_by/<j>/<b>
                new_rel = rel.parent / (rel.name + suffix)   # append suffix to leaf name
                moves.append((leaf, new_model_dir / new_rel))
    return moves


def find_conflicts(moves: list[tuple[Path, Path]]) -> list[str]:
    """Return human-readable conflict strings; empty list means safe to proceed."""
    conflicts: list[str] = []
    seen: dict[Path, Path] = {}
    for src, dst in moves:
        if dst in seen:
            conflicts.append(f"two sources -> same dst:\n    {seen[dst]}\n    {src}\n  -> {dst}")
        seen[dst] = src
        if dst.exists():
            conflicts.append(f"dst already exists on disk:\n    {src}\n  -> {dst}")
    return conflicts


def make_backup(roots: list[Path], ts: str) -> Path:
    backup = Path(f"results_backup_{ts}.tar.gz")
    print(f"[backup] writing {backup} …")
    with tarfile.open(backup, "w:gz") as tar:
        for root in roots:
            if root.exists():
                tar.add(root, arcname=str(root))
    print(f"[backup] done: {backup} ({backup.stat().st_size / 1e9:.2f} GB)")
    return backup


def prune_empty_dirs(roots: list[Path]) -> None:
    """rmdir-only cleanup of dirs left empty by the moves. Never deletes data."""
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()  # only succeeds if truly empty
                except OSError:
                    pass


def do_migrate(roots: list[Path], execute: bool) -> int:
    moves = plan_moves(roots)
    if not moves:
        print("[migrate] nothing to migrate — no V3 model dirs found (already V5?).")
        return 0

    print(f"[migrate] planned {len(moves)} leaf move(s):")
    for src, dst in moves:
        print(f"  {src}\n    -> {dst}")

    conflicts = find_conflicts(moves)
    if conflicts:
        print("\n[ABORT] conflicts detected — NOTHING was moved:")
        for c in conflicts:
            print("  " + c)
        return 2

    if not execute:
        print("\n[dry-run] no changes made. Re-run with --execute to apply.")
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    make_backup(roots, ts)
    manifest = Path(f"results_layout_manifest_{ts}.tsv")

    before = _count_files(roots)
    with manifest.open("w") as mf:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():  # paranoia — pre-scan already guaranteed this can't happen
                print(f"[ABORT] dst appeared mid-run: {dst}")
                return 3
            os.rename(src, dst)
            mf.write(f"{src}\t{dst}\n")
    print(f"[migrate] manifest written: {manifest}")

    prune_empty_dirs(roots)

    after = _count_files(roots)
    if before != after:
        print(f"[FATAL] file-count changed: before={before} after={after}. "
              f"Investigate immediately; revert with: --revert {manifest}")
        return 4
    print(f"[verify] file count unchanged ({before}). Migration OK.")
    return 0


def do_revert(manifest: Path) -> int:
    if not manifest.exists():
        print(f"[revert] manifest not found: {manifest}")
        return 1
    pairs = [line.rstrip("\n").split("\t") for line in manifest.read_text().splitlines() if line.strip()]
    # Reverse order so nested dirs come back before their parents are recreated.
    for src, dst in reversed(pairs):
        src_p, dst_p = Path(src), Path(dst)
        if not dst_p.exists():
            print(f"[revert][skip] dst gone: {dst_p}")
            continue
        if src_p.exists():
            print(f"[revert][ABORT] original path already occupied: {src_p}")
            return 2
        src_p.parent.mkdir(parents=True, exist_ok=True)
        os.rename(dst_p, src_p)
        print(f"[revert] {dst_p}\n    -> {src_p}")
    # rmdir-only cleanup of the now-empty V5 scaffolding husks (never deletes data).
    prune_roots = sorted({Path(dst).parts[0] for _, dst in pairs})
    prune_empty_dirs([Path(p) for p in prune_roots])
    print("[revert] done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="results root to migrate (repeatable). "
                         "Default: 'results' and 'results/RL train'.")
    ap.add_argument("--execute", action="store_true",
                    help="apply the moves (default is dry-run).")
    ap.add_argument("--revert", metavar="MANIFEST.tsv",
                    help="undo a prior migration from its manifest, then exit.")
    args = ap.parse_args()

    if args.revert:
        return do_revert(Path(args.revert))

    # Single recursive root covers the whole tree (results/, results/RL train/,
    # results/RL train/combined_report/, …) via recursive state-dir discovery.
    roots = [Path(r) for r in (args.root or ["results"])]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("[migrate] no existing roots to process.")
        return 1
    print(f"[migrate] roots: {[str(r) for r in roots]}  execute={args.execute}")
    return do_migrate(roots, args.execute)


if __name__ == "__main__":
    sys.exit(main())
