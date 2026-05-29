#!/usr/bin/env python3
"""Migrate V0/V1 results directories to the V2 layout.

The V2 layout encodes 5 distinguishing parameters per run (model, state,
temperature, max_tokens, n_samples) so two sweeps with different hyperparams
never overwrite each other:

    <ROOT>/<model>__state-<s>__t<T>__max<N>__n<NS>/<benchmark>/
    <ROOT>/<target>__state-...__n<NS>/judged_by/<judge>__state-...__n<jNS>/<benchmark>/

This script walks a results root, parses every V0/V1 run it finds, derives the
missing metadata (state from path/model name, n_samples from results.jsonl),
augments stale summary.json files with aggregate count fields, generates
<benchmark>_per_task.csv from the existing _results.jsonl, removes obsolete
artefacts (only cot_judge_summary.json + legacy base-file duplicates living
inside judgment dirs — raw / judge.jsonl files are NEVER deleted), then moves
the directory to its V2 path.

Default mode is dry-run; pass --execute to apply.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- V1 path parsing -------------------------------------------------------

_V1_BASE_RE = re.compile(r"^(?P<model>.+?)_t(?P<temp>[0-9.]+)_max(?P<max>\d+)$")
_V1_JUDGE_RE = re.compile(r"^(?P<target>.+?)_evaluated_by_(?P<judge>.+?)_(?P<max>\d+)$")
_V1_BENCH_TEMP_RE = re.compile(r"^(?P<benchmark>.+?)_t(?P<temp>[0-9.]+)$")
_V2_BASE_RE = re.compile(r"^.+__state-(base|non-think|think|unknown)__t[0-9.]+__max\d+__n\d+$")

# Legacy instruct/reasoning dirs encode the "think" flag in the model segment:
# "<model>_think-true_t..." or "<model>_think-false_t...". Strip when present.
_THINK_FLAG_RE = re.compile(r"^(?P<model>.+?)_think-(?P<flag>true|false)$")

# Older pipeline versions named judge artefacts "math_judge_*" instead of
# "cot_judge_*". Treat the two prefixes as equivalent during migration; the
# files get renamed to the canonical cot_judge form as part of the move so
# the migrated layout looks like a current-pipeline output.
LEGACY_JUDGE_PREFIX = "math_judge"
CANONICAL_JUDGE_PREFIX = "cot_judge"


def state_from_class_dir(class_name: str) -> str | None:
    """Map the legacy class prefix (base/instruct/reasoning) to a state."""
    return {"base": "base", "instruct": "non-think", "reasoning": "think"}.get(class_name)


def state_from_model_name(name: str) -> str:
    """Heuristic when class prefix is absent."""
    low = name.lower()
    if "reasoning" in low or "think" in low:
        return "think"
    if "instruct" in low or low.endswith("-it") or "-it_" in low or "_it_" in low:
        return "non-think"
    if "base" in low:
        return "base"
    return "unknown"


# --- Metadata derivation ---------------------------------------------------


def n_samples_from_results(path: Path) -> int | None:
    """Read first record's correct[] length. Returns None if file missing/empty."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("rb") as f:
            line = f.readline()
        if not line:
            return None
        rec = json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    correct = rec.get("correct")
    if isinstance(correct, list) and correct:
        return len(correct)
    sols = rec.get("solutions")
    if isinstance(sols, list) and sols:
        return len(sols)
    return None


# --- Augmentation: summary.json + per_task.csv -----------------------------


def _per_task_counts_from_correct(correct: list[Any]) -> dict[str, int]:
    """Derive per_task_counts when an older result record lacks the field."""
    return {
        "true": sum(1 for x in correct if x is True),
        "false": sum(1 for x in correct if x is False),
        "cot_false": sum(1 for x in correct if x == "cot_false"),
        "invalid_format": sum(1 for x in correct if x not in (True, False, "cot_false")),
    }


def augment_summary_and_csv(
    results_path: Path,
    summary_path: Path,
    csv_path: Path,
    is_cot: bool,
    stats_path: Path | None = None,
) -> tuple[bool, bool]:
    """Add aggregate fields to summary.json + write per_task.csv from results.jsonl.

    Returns (summary_changed, csv_written).
    """
    if not results_path.exists():
        return (False, False)

    records: list[dict[str, Any]] = []
    with results_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        return (False, False)

    # Backfill per_task_counts when missing.
    for r in records:
        if "per_task_counts" not in r:
            r["per_task_counts"] = _per_task_counts_from_correct(r.get("correct", []))

    n_gen = len(records[0].get("correct") or records[0].get("solutions") or [])
    total_tasks = len(records)
    tc = sum(r["per_task_counts"]["true"] for r in records)
    fc = sum(r["per_task_counts"]["false"] for r in records)
    cfc = sum(r["per_task_counts"]["cot_false"] for r in records)
    ifc = sum(r["per_task_counts"]["invalid_format"] for r in records)

    # Augment summary if it lacks aggregate fields.
    summary_changed = False
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            with summary_path.open("rb") as f:
                summary = json.loads(f.read())
        except json.JSONDecodeError:
            summary = {}
    if stats_path is not None and stats_path.exists():
        try:
            with stats_path.open("rb") as f:
                stats = json.loads(f.read())
            # Prefer stats.json values if present.
            tc = stats.get("true_count", tc)
            fc = stats.get("false_count", fc)
            cfc = stats.get("cot_false_count", cfc)
            ifc = stats.get("invalid_count", ifc)
            total_tasks = stats.get("total_tasks", total_tasks)
            n_gen_total = stats.get("total_generations", total_tasks * n_gen)
        except json.JSONDecodeError:
            n_gen_total = total_tasks * n_gen
    else:
        n_gen_total = total_tasks * n_gen

    wanted = {
        "total_tasks": total_tasks,
        "total_generations": n_gen_total,
        "true_count": tc,
        "false_count": fc,
        "cot_false_count": cfc,
        "invalid_format_count": ifc,
    }
    for k, v in wanted.items():
        if summary.get(k) != v:
            summary[k] = v
            summary_changed = True

    # Per-task CSV.
    k_keys = sorted((records[0].get("pass_at_k") or {}).keys(), key=lambda s: int(s))
    fieldnames = (
        ["task_id", "true", "false", "cot_false", "invalid_format"]
        + [f"pass@{k}" for k in k_keys]
        + ["ground_truth", "majority_vote", "is_correct_majority"]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            counts = r["per_task_counts"]
            row = {
                "task_id": r["task_id"],
                "true": counts["true"],
                "false": counts["false"],
                "cot_false": counts["cot_false"],
                "invalid_format": counts["invalid_format"],
                "ground_truth": r.get("ground_truth", ""),
                "majority_vote": r.get("majority_vote", ""),
                "is_correct_majority": r.get("is_correct_majority", ""),
            }
            for k in k_keys:
                row[f"pass@{k}"] = (r.get("pass_at_k") or {}).get(k, "")
            writer.writerow(row)

    if summary_changed:
        with summary_path.open("wb") as f:
            f.write(json.dumps(summary).encode("utf-8"))
    return (summary_changed, True)


# --- Plan + execution ------------------------------------------------------


@dataclass
class MoveOp:
    src: Path
    dst: Path
    kind: str  # "base" or "judge"
    obsolete: list[Path] = field(default_factory=list)
    augment: list[tuple[Path, Path, Path, bool, Path | None]] = field(default_factory=list)
    # In-place file renames inside src before the dir is moved. Used to turn
    # legacy "math_judge_*" filenames into the canonical "cot_judge_*" form so
    # the migrated dir looks like a current-pipeline output.
    renames: list[tuple[Path, Path]] = field(default_factory=list)


@dataclass
class Plan:
    moves: list[MoveOp] = field(default_factory=list)
    top_deletes: list[Path] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    ambiguous: list[tuple[Path, str]] = field(default_factory=list)

    def has_ambiguous(self) -> bool:
        return bool(self.ambiguous)


OBSOLETE_FILENAMES = {"cot_judge_summary.json"}


def _enumerate_base_dirs(root: Path) -> list[tuple[Path, str | None]]:
    """Yield (dir, class_hint). class_hint is "base"/"instruct"/"reasoning" or None."""
    out: list[tuple[Path, str | None]] = []
    # Layout: <root>/<class>/<model>_t<T>_max<N>/...
    for cls in ("base", "instruct", "reasoning"):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for child in sorted(cls_dir.iterdir()):
            if child.name == "judgments":
                continue
            if child.is_dir() and _V1_BASE_RE.match(child.name):
                out.append((child, cls))
    # Layout fallback: <root>/<model>_t<T>_max<N>/... (no class prefix)
    for child in sorted(root.iterdir()):
        if child.is_dir() and _V1_BASE_RE.match(child.name) and not _V2_BASE_RE.match(child.name):
            out.append((child, None))
    return out


def _enumerate_judge_dirs(root: Path) -> list[tuple[Path, str | None]]:
    out: list[tuple[Path, str | None]] = []
    for cls in ("base", "instruct", "reasoning"):
        jroot = root / cls / "judgments"
        if not jroot.is_dir():
            continue
        for child in sorted(jroot.iterdir()):
            if child.is_dir() and _V1_JUDGE_RE.match(child.name):
                out.append((child, cls))
    # Fallback: <root>/judgments/...
    jroot = root / "judgments"
    if jroot.is_dir():
        for child in sorted(jroot.iterdir()):
            if child.is_dir() and _V1_JUDGE_RE.match(child.name):
                out.append((child, None))
    return out


def build_plan(root: Path, manifest: dict[str, Any] | None = None) -> Plan:
    plan = Plan()
    manifest = manifest or {}
    model_state_overrides: dict[str, str] = manifest.get("missing_state_for_model", {}) or {}

    # Track derived target metadata for judge dir migration. Keyed by
    # (model, state, temp, max) so the same model trained / served in two
    # states (e.g. instruct vs reasoning) keeps separate entries.
    target_index: dict[tuple[str, str, str, int], dict[str, Any]] = {}

    # --- BASE DIRS ----------------------------------------------------------
    for base_dir, cls in _enumerate_base_dirs(root):
        m = _V1_BASE_RE.match(base_dir.name)
        if not m:
            continue
        model = m.group("model")
        temp = m.group("temp")
        max_tok = int(m.group("max"))
        # Strip "_think-(true|false)" suffix that older instruct/reasoning runs
        # embedded in the model name. The class dir already supplies the state.
        think_match = _THINK_FLAG_RE.match(model)
        if think_match:
            model = think_match.group("model")
        state = state_from_class_dir(cls) if cls else None
        if state is None:
            override = model_state_overrides.get(model)
            if override and override != "?":
                state = override
            else:
                state = state_from_model_name(model)
        if state == "unknown" and model not in model_state_overrides:
            plan.ambiguous.append((base_dir, f"unresolved state for model={model!r}"))

        # n_samples from any benchmark's results.jsonl
        n_samples = None
        for bench_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            results = bench_dir / f"{bench_dir.name}_results.jsonl"
            n_samples = n_samples_from_results(results)
            if n_samples is not None:
                break
        if n_samples is None:
            plan.skipped.append((base_dir, "no readable results.jsonl to derive n_samples"))
            continue

        v2_name = f"{model}__state-{state}__t{temp}__max{max_tok}__n{n_samples}"
        dst = root / v2_name

        # Per-benchmark augment ops.
        augments: list[tuple[Path, Path, Path, bool, Path | None]] = []
        for bench_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            bench = bench_dir.name
            results = bench_dir / f"{bench}_results.jsonl"
            summary = bench_dir / f"{bench}_summary.json"
            csv_path = bench_dir / f"{bench}_per_task.csv"
            if results.exists():
                augments.append((results, summary, csv_path, False, None))

        plan.moves.append(MoveOp(src=base_dir, dst=dst, kind="base", augment=augments))
        target_index[(model, state, temp, max_tok)] = {
            "n_samples": n_samples,
            "state": state,
            "v2_dir": dst,
        }

    # --- JUDGE DIRS ---------------------------------------------------------
    for judge_dir, cls in _enumerate_judge_dirs(root):
        m = _V1_JUDGE_RE.match(judge_dir.name)
        if not m:
            continue
        target_model = m.group("target")
        judge_model = m.group("judge")
        judge_max = int(m.group("max"))
        # Same "_think-(true|false)" suffix may live on the target side too.
        think_match = _THINK_FLAG_RE.match(target_model)
        if think_match:
            target_model = think_match.group("model")

        # Determine which target this judge run belongs to. The judge dir name
        # carries the target model; the class dir (base/instruct/reasoning)
        # supplies the target state. We then look up (model, state, *, *) in
        # the index and expect exactly one match.
        target_state_from_cls = state_from_class_dir(cls) if cls else None
        if target_state_from_cls is not None:
            matches = [k for k in target_index if k[0] == target_model and k[1] == target_state_from_cls]
        else:
            matches = [k for k in target_index if k[0] == target_model]
        if not matches:
            plan.skipped.append((judge_dir, f"no migrated target dir for model={target_model!r}"))
            continue
        if len(matches) > 1:
            plan.ambiguous.append(
                (judge_dir, f"multiple target candidates for {target_model!r}: {matches}")
            )
            continue
        ti = target_index[matches[0]]
        target_temp = matches[0][2]
        target_max = matches[0][3]
        target_n = ti["n_samples"]
        target_state = ti["state"]
        target_v2 = ti["v2_dir"]

        # Per-benchmark loop — each subdir is "<benchmark>_t<JT>" (or just "<benchmark>").
        judge_state = model_state_overrides.get(judge_model)
        if judge_state is None or judge_state == "?":
            judge_state = state_from_model_name(judge_model)
        if judge_state == "unknown" and judge_model not in model_state_overrides:
            plan.ambiguous.append(
                (judge_dir, f"unresolved state for judge model={judge_model!r}")
            )

        # First pass: pick a representative judge_n from any benchmark subdir
        # under this judge dir whose results file is non-empty. Used as the
        # fallback for stub/empty benchmark subdirs (those are early-exit cases
        # where there were no base-correct samples, so no judge ever ran).
        judge_n_fallback: int | None = None
        for bench_subdir in (p for p in judge_dir.iterdir() if p.is_dir()):
            for fname in ("cot_judge_results.jsonl", "math_judge_results.jsonl"):
                got = n_samples_from_results(bench_subdir / fname)
                if got is not None:
                    judge_n_fallback = got
                    break
            if judge_n_fallback is not None:
                break

        for bench_subdir in sorted(p for p in judge_dir.iterdir() if p.is_dir()):
            bm_match = _V1_BENCH_TEMP_RE.match(bench_subdir.name)
            if bm_match:
                benchmark = bm_match.group("benchmark")
                judge_temp = bm_match.group("temp")
            else:
                benchmark = bench_subdir.name
                judge_temp = target_temp  # best-effort fallback

            # Derive judge_n from cot_judge_results.jsonl correct[] length.
            # Older runs used "math_judge_*"; treat as a fallback.
            judge_results = bench_subdir / "cot_judge_results.jsonl"
            if not judge_results.exists():
                judge_results = bench_subdir / "math_judge_results.jsonl"
            judge_n = n_samples_from_results(judge_results)
            if judge_n is None:
                # Empty stub: no base-correct samples → judge never ran.
                # Fall back to the sibling-derived n so the V2 path still
                # encodes a reasonable value; the stub files come along intact.
                if judge_n_fallback is not None:
                    judge_n = judge_n_fallback
                else:
                    # No sibling worked either — every benchmark under this
                    # judge dir was an empty stub. Fall back to the target's
                    # n_samples; it's a best-effort placeholder that keeps the
                    # V2 path valid. Log so the user can audit.
                    judge_n = target_n
                    plan.skipped.append(
                        (bench_subdir, f"all judge files empty; using target_n={target_n} as placeholder")
                    )

            judge_leaf = (
                f"{judge_model}__state-{judge_state}__t{judge_temp}"
                f"__max{judge_max}__n{judge_n}"
            )
            dst_bm = target_v2 / "judged_by" / judge_leaf / benchmark

            obsolete = []
            for fname in OBSOLETE_FILENAMES:
                p = bench_subdir / fname
                if p.exists():
                    obsolete.append(p)
            # Legacy base-file duplicates living inside judgments dir. Only
            # deleted at execute time when the corresponding base dir was
            # migrated successfully (so the data still exists somewhere).
            for suffix in (f"{benchmark}_results.jsonl",
                           f"{benchmark}_summary.json",
                           f"{benchmark}_generation_stats.json",
                           f"{benchmark}_generation_stats.jsonl"):
                p = bench_subdir / suffix
                if p.exists():
                    obsolete.append(p)

            # Rename legacy math_judge_* files to match the current pipeline.
            # Most become cot_judge_*; majority files take the benchmark-
            # prefixed canonical name <benchmark>_cot_majority.jsonl.
            # math_judge_summary.json is dropped (it's the misleading raw
            # judge yes-rate, same problem as cot_judge_summary.json).
            renames: list[tuple[Path, Path]] = []
            for p in bench_subdir.iterdir():
                if not p.is_file():
                    continue
                is_legacy = (
                    p.name.startswith(LEGACY_JUDGE_PREFIX + "_")
                    or p.name == LEGACY_JUDGE_PREFIX + ".jsonl"
                )
                if not is_legacy:
                    continue
                if p.name == f"{LEGACY_JUDGE_PREFIX}_summary.json":
                    # Will become a misleading cot_judge_summary.json — delete
                    # outright instead of renaming.
                    obsolete.append(p)
                    continue
                if p.name == f"{LEGACY_JUDGE_PREFIX}_majority.jsonl":
                    new_name = f"{benchmark}_cot_majority.jsonl"
                else:
                    new_name = CANONICAL_JUDGE_PREFIX + p.name[len(LEGACY_JUDGE_PREFIX):]
                renames.append((p, bench_subdir / new_name))

            # Augment cot summary + write cot per_task.csv.
            cot_results = bench_subdir / f"{benchmark}_cot_results.jsonl"
            cot_summary = bench_subdir / f"{benchmark}_cot_summary.json"
            cot_stats = bench_subdir / f"{benchmark}_cot_stats.json"
            cot_csv = bench_subdir / f"{benchmark}_cot_per_task.csv"
            augments = []
            if cot_results.exists():
                augments.append((cot_results, cot_summary, cot_csv, True,
                                 cot_stats if cot_stats.exists() else None))

            plan.moves.append(MoveOp(
                src=bench_subdir, dst=dst_bm, kind="judge",
                obsolete=obsolete, augment=augments, renames=renames,
            ))

    # --- Top-level deletes --------------------------------------------------
    legacy_csv = root / "all_results.csv"
    if legacy_csv.exists():
        plan.top_deletes.append(legacy_csv)

    return plan


def print_plan(plan: Plan) -> None:
    print(f"\n=== Migration plan ===")
    print(f"Moves:         {len(plan.moves)}")
    print(f"Top deletes:   {len(plan.top_deletes)}")
    print(f"Skipped:       {len(plan.skipped)}")
    print(f"Ambiguous:     {len(plan.ambiguous)}")
    print()
    for op in plan.moves:
        marker = "[base]" if op.kind == "base" else "[judge]"
        print(f"{marker} {op.src}")
        print(f"     → {op.dst}")
        for o in op.obsolete:
            print(f"     DEL {o.relative_to(op.src)} (duplicate, base copy preserved)")
        for old, new in op.renames:
            print(f"     REN {old.name} → {new.name}")
        for (results, summary, csv_path, is_cot, _) in op.augment:
            print(f"     AUG {summary.name} + emit {csv_path.name}")
    for d in plan.top_deletes:
        print(f"[top-del] {d}")
    for src, reason in plan.skipped:
        print(f"[skip]    {src} — {reason}")
    for src, reason in plan.ambiguous:
        print(f"[AMBIG]   {src} — {reason}")


def finalize_legacy_judge_dirs(plan: Plan) -> None:
    """Run apply_cot_metrics for each migrated judge dir that lacks the cot
    artefacts (_cot_results.jsonl / _cot_summary.json / _cot_per_task.csv).

    These come from the legacy pipeline that only ran judge gen+eval without
    ever calling `evalhub cot finalize`. We synthesise the finalize step in
    place so the dst dir contains the full set of V2 files.
    """
    from evalhub.cot.metrics import apply_cot_metrics

    for op in plan.moves:
        if op.kind != "judge":
            continue
        benchmark = op.dst.name
        majority = op.dst / f"{benchmark}_cot_majority.jsonl"
        cot_summary = op.dst / f"{benchmark}_cot_summary.json"
        cot_results = op.dst / f"{benchmark}_cot_results.jsonl"
        cot_stats = op.dst / f"{benchmark}_cot_stats.json"
        if not majority.exists() or majority.stat().st_size == 0:
            continue
        if cot_summary.exists() and cot_results.exists():
            continue  # already finalised by the source pipeline
        # base_results path: <target_v2> / <benchmark> / <benchmark>_results.jsonl
        # dst = <target_v2>/judged_by/<judge_leaf>/<benchmark>
        target_v2 = op.dst.parent.parent.parent
        base_results = target_v2 / benchmark / f"{benchmark}_results.jsonl"
        if not base_results.exists():
            continue
        try:
            apply_cot_metrics(
                base_results_path=base_results,
                majority_path=majority,
                output_results_path=cot_results,
                summary_path=cot_summary,
                stats_path=cot_stats,
            )
            print(f"finalised cot artefacts at {op.dst}")
        except Exception as exc:  # pragma: no cover — diagnostic only
            print(f"  WARN: cot finalize failed at {op.dst}: {exc}")


def execute_plan(plan: Plan) -> None:
    # 1. Per-op: obsolete deletes + renames + augments + collision-aware move.
    for op in plan.moves:
        # Renames first so augment + subsequent inspections see canonical names.
        for old, new in op.renames:
            if new.exists():
                raise SystemExit(f"RENAME COLLISION: {new} already exists")
            os.rename(old, new)
        for o in op.obsolete:
            if o.exists():
                o.unlink()
        for (results, summary, csv_path, is_cot, stats) in op.augment:
            augment_summary_and_csv(results, summary, csv_path, is_cot=is_cot, stats_path=stats)
        if op.dst.exists():
            # Collision: bail unless contents identical (no overwrite policy).
            raise SystemExit(f"COLLISION: {op.dst} already exists. Inspect manually.")
        op.dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(op.src, op.dst)
        print(f"moved {op.src} → {op.dst}")
    # 2. Top-level deletes.
    for d in plan.top_deletes:
        d.unlink()
        print(f"deleted {d}")
    # 2b. Finalise legacy judge dirs (synthesise _cot_results / _cot_summary
    # / _cot_per_task.csv when the source pipeline never ran cot finalize).
    finalize_legacy_judge_dirs(plan)
    # 3. Empty parent cleanup.
    for parent in {"base", "instruct", "reasoning"}:
        for sub in ("judgments", ""):
            p = plan.moves[0].src.parents[-1] if plan.moves else None  # placeholder
            # Walk known top dirs:
        pass
    # Simpler: walk a fixed set.
    candidates = []
    if plan.moves:
        root_guess = plan.moves[0].dst.parents[0]
        for sub in ("base", "instruct", "reasoning"):
            candidates.append(root_guess / sub / "judgments")
            candidates.append(root_guess / sub)
    for c in candidates:
        if c.is_dir() and not any(c.iterdir()):
            c.rmdir()
            print(f"rmdir {c}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Results root to migrate (e.g. results, results_demo)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the migration (default: dry-run)")
    parser.add_argument("--manifest", type=Path,
                        help="JSON file with overrides for ambiguous metadata")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"--root does not exist: {args.root}", file=sys.stderr)
        return 2

    manifest = None
    if args.manifest and args.manifest.exists():
        with args.manifest.open() as f:
            manifest = json.load(f)

    plan = build_plan(args.root, manifest=manifest)

    # Pre-flight: two source dirs mapping to the same destination = data loss
    # risk. Fail loudly before we move a single byte.
    seen_dsts: dict[Path, Path] = {}
    duplicate_dsts: list[tuple[Path, Path, Path]] = []
    for op in plan.moves:
        prior = seen_dsts.get(op.dst)
        if prior is not None:
            duplicate_dsts.append((op.dst, prior, op.src))
        else:
            seen_dsts[op.dst] = op.src
    if duplicate_dsts:
        print("\nFATAL: multiple source dirs map to the same V2 destination:")
        for dst, prior_src, new_src in duplicate_dsts:
            print(f"  dst:   {dst}")
            print(f"  src 1: {prior_src}")
            print(f"  src 2: {new_src}")
        return 3

    print_plan(plan)

    if plan.has_ambiguous():
        template = {
            "missing_state_for_model": {},
            "missing_n_samples_for_dir": {},
        }
        for src, reason in plan.ambiguous:
            if "state for" in reason and "model=" in reason:
                mod = reason.split("model=")[1].strip("'\"")
                template["missing_state_for_model"][mod] = "?"
            else:
                template["missing_n_samples_for_dir"][str(src)] = "?"
        out_manifest = args.root.parent / f"{args.root.name}.manifest.json"
        with out_manifest.open("w") as f:
            json.dump(template, f, indent=2)
        print(f"\nAmbiguous entries dumped to {out_manifest}.")
        print("Fill in the '?' values and re-run with --manifest", out_manifest)
        return 1

    if not args.execute:
        print("\n(dry-run; rerun with --execute to apply)")
        return 0

    execute_plan(plan)
    print("\nMigration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
