#!/usr/bin/env python3
"""V3 -> V4 results layout migration.

Phase 1: strip `_think-{true|false}` suffix from instruct target dir names so
that every (state, model) pair lives under a single directory.

Phase 2: strip `__n<JUDGE_N_SAMPLES>` suffix from judge directory names so that
every (target, judge, judge_state, t, max) tuple is one directory holding all
benchmarks side by side.

Phase 3 (manual): update pipeline_common.sh + scan.py to emit/parse V4 paths.

Safety guarantees per merge:
  1. Pre-merge sha256 conflict check (abort on differing content).
  2. tar.gz backup under .migration_backup/.
  3. rsync --ignore-existing --checksum (never overwrites).
  4. Post-merge sha256 verify (every source file matched in dest).
  5. Source removed only after verify passes.

Usage:
    python scripts/migrate_legacy_paths.py                 # dry-run both phases
    python scripts/migrate_legacy_paths.py --execute       # phase 1 + 2
    python scripts/migrate_legacy_paths.py --phase 1       # dry-run phase 1
    python scripts/migrate_legacy_paths.py --phase 2 --execute
    python scripts/migrate_legacy_paths.py --report-json out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path("results")
BACKUP_DIR = Path(".migration_backup")

TARGET_SUFFIX_RE = re.compile(
    r"^(?P<base>.+?)_think-(?:true|false)__(?P<rest>t.+)$"
)
JUDGE_N_RE = re.compile(
    r"^(?P<head>.+?__state-(?:base|non-think|think|unknown)__t[0-9.]+__max\d+)"
    r"__n\d+$"
)


def sha256_of(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def relative_files(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}


@dataclass
class MergeOp:
    src: str
    dst: str
    phase: int
    identical_skip: list[str] = field(default_factory=list)
    differ_conflict: list[str] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)


def plan_merge(src: Path, dst: Path, phase: int) -> MergeOp:
    op = MergeOp(src=str(src), dst=str(dst), phase=phase)
    src_files = relative_files(src)
    dst_files = relative_files(dst) if dst.exists() else {}
    for rel, sp in src_files.items():
        tp = dst_files.get(rel)
        if tp is None:
            op.new_files.append(rel)
        elif sha256_of(sp) == sha256_of(tp):
            op.identical_skip.append(rel)
        else:
            op.differ_conflict.append(rel)
    return op


def enumerate_phase1() -> list[MergeOp]:
    ops: list[MergeOp] = []
    for state_dir in sorted(ROOT.iterdir()):
        if not state_dir.is_dir():
            continue
        if state_dir.name not in ("base", "non-think", "think"):
            continue
        for legacy in sorted(state_dir.iterdir()):
            if not legacy.is_dir():
                continue
            m = TARGET_SUFFIX_RE.match(legacy.name)
            if not m:
                continue
            dst = state_dir / f"{m['base']}__{m['rest']}"
            ops.append(plan_merge(legacy, dst, phase=1))
    return ops


def enumerate_phase2() -> list[MergeOp]:
    ops: list[MergeOp] = []
    for judge_dir in sorted(ROOT.rglob("judged_by/*")):
        if not judge_dir.is_dir():
            continue
        m = JUDGE_N_RE.match(judge_dir.name)
        if not m:
            continue
        dst = judge_dir.parent / m["head"]
        ops.append(plan_merge(judge_dir, dst, phase=2))
    return ops


def backup(src: Path, phase: int) -> Path:
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = str(src.relative_to(ROOT)).replace("/", "__")
    tar_path = BACKUP_DIR / f"faz{phase}_{safe}_{ts}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src, arcname=str(src.relative_to(ROOT)))
    return tar_path


def rsync_merge(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        return subprocess.call([
            "rsync", "-a", "--ignore-existing", "--checksum",
            f"{src}/", f"{dst}/",
        ])
    for rel, sp in relative_files(src).items():
        tp = dst / rel
        if tp.exists():
            continue
        tp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, tp)
    return 0


def verify_merge(src: Path, dst: Path) -> tuple[bool, str | None]:
    for rel, sp in relative_files(src).items():
        tp = dst / rel
        if not tp.exists():
            return False, f"missing in dst: {rel}"
        if sha256_of(sp) != sha256_of(tp):
            return False, f"sha256 mismatch: {rel}"
    return True, None


def fmt_report(ops: list[MergeOp]) -> str:
    lines = [f"\nMigration plan: {len(ops)} operation(s)", "=" * 70]
    for op in ops:
        lines.append(f"\n[Faz {op.phase}] {op.src}")
        lines.append(f"        -> {op.dst}")
        lines.append(
            f"        new={len(op.new_files):<5} "
            f"identical-skip={len(op.identical_skip):<5} "
            f"DIFFER={len(op.differ_conflict)}"
        )
        for rel in op.differ_conflict[:3]:
            lines.append(f"           !! {rel}")
        if len(op.differ_conflict) > 3:
            lines.append(f"           ... ({len(op.differ_conflict) - 3} more)")
    return "\n".join(lines)


def run_phase(ops: list[MergeOp], phase: int) -> None:
    for op in ops:
        src = Path(op.src)
        dst = Path(op.dst)
        tar = backup(src, phase=phase)
        print(f"[P{phase}] backup -> {tar}")
        rc = rsync_merge(src, dst)
        if rc != 0:
            sys.exit(f"[P{phase}] rsync failed for {src} (rc={rc})")
        ok, why = verify_merge(src, dst)
        if not ok:
            sys.exit(
                f"[P{phase}] verify FAILED for {src}: {why} "
                f"(source preserved; tar backup at {tar})"
            )
        shutil.rmtree(src)
        print(f"[P{phase}] merged + removed {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument(
        "--phase", type=int, choices=[1, 2], default=None,
        help="Run only this phase (default: both)",
    )
    ap.add_argument("--report-json", type=Path)
    args = ap.parse_args()

    if not ROOT.exists():
        sys.exit(f"ROOT not found: {ROOT.resolve()}")

    phases = (args.phase,) if args.phase else (1, 2)

    if not args.execute:
        all_ops: list[MergeOp] = []
        if 1 in phases:
            all_ops.extend(enumerate_phase1())
        if 2 in phases:
            all_ops.extend(enumerate_phase2())
        print(fmt_report(all_ops))
        if args.report_json:
            args.report_json.write_text(
                json.dumps([asdict(o) for o in all_ops], indent=2)
            )
            print(f"\nReport written: {args.report_json}")
        differ = sum(len(o.differ_conflict) for o in all_ops)
        if differ:
            print(f"\n*** {differ} differing-content conflicts — would ABORT ***")
            sys.exit(2)
        print("\nDry-run OK. Run with --execute to apply.")
        return

    # Execute phase 1 first
    if 1 in phases:
        ops1 = enumerate_phase1()
        differ = sum(len(o.differ_conflict) for o in ops1)
        if differ:
            print(fmt_report(ops1))
            sys.exit(f"P1 ABORT: {differ} differing-content conflicts")
        print(f"[P1] {len(ops1)} merge op(s)")
        run_phase(ops1, phase=1)

    # Re-enumerate phase 2 since phase 1 may have moved dirs into existence
    if 2 in phases:
        ops2 = enumerate_phase2()
        differ = sum(len(o.differ_conflict) for o in ops2)
        if differ:
            print(fmt_report(ops2))
            sys.exit(f"P2 ABORT: {differ} differing-content conflicts")
        print(f"\n[P2] {len(ops2)} merge op(s)")
        run_phase(ops2, phase=2)

    print("\nAll phases done. Apply Faz 3 (code changes) next.")


if __name__ == "__main__":
    main()
