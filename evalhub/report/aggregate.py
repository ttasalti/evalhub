"""Wide master-CSV writer: one row per evaluated (model, mode, benchmark, judge).

Each :class:`RunRecord` (one ``*_summary.json`` / ``*_cot_summary.json``) becomes
exactly **one row** carrying every metric at every K and τ:

* ``pass@{k}``            — Pass@k
* ``gpass@{k}_t{tau}``    — G-Pass@k at threshold τ
* ``mgpass@{k}``          — mG-Pass@k

The **``judge_model`` column is the discriminator**: when it is empty the row is
the **No-Judge** reference and those columns mean pass / g-pass / mg-pass; when a
judge is set the same columns mean **cot-pass / cot-g-pass / cot-mg-pass**. There
are no separate ``cot_*`` columns.

Two entry points share the same schema:

* :func:`aggregate_results` — full rebuild from a results root.
* :func:`upsert_record` / :func:`upsert_summary` — append-or-replace a single
  row keyed by ``(model, state, benchmark, judge_model, judge_state)``, so a
  pipeline can call it once per finished evaluation and the CSV grows row by row.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from evalhub.report import labels
from evalhub.report.scan import RunRecord, record_from_summary, scan_results
from evalhub.utils.logger import logger

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


# G-Pass thresholds, exactly as keyed in the summary JSON.
TAUS: tuple[str, ...] = ("0.25", "0.5", "0.75", "1.0")
# Fallback K axis used only when a brand-new CSV has no data to infer from.
CANONICAL_KS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Row identity. judge_* are empty on No-Judge rows.
KEY_COLUMNS: tuple[str, ...] = ("model", "state", "benchmark", "judge_model", "judge_state")

# Identity / metadata columns, in order. Metric columns are appended after these.
META_COLUMNS: list[str] = [
    "model", "model_short", "model_family", "model_size_b", "is_base",
    "state", "mode", "benchmark", "language",
    "judged", "series",
    "judge_model", "judge_state",
    "n_samples", "temperature", "max_tokens",
    "judge_n_samples", "judge_temperature", "judge_max_tokens",
    "cons_at_k", "total_tasks", "total_generations",
    "true_count", "false_count", "cot_false_count", "invalid_count",
    "run_dir", "summary_path",
]


def _pass_col(k: int) -> str:
    return f"pass@{k}"


def _gpass_col(k: int, tau: str) -> str:
    return f"gpass@{k}_t{tau}"


def _mgpass_col(k: int) -> str:
    return f"mgpass@{k}"


def metric_columns(ks: Iterable[int]) -> list[str]:
    """All metric columns for the given K axis, grouped per K (pass, g-pass τ…, mg)."""
    cols: list[str] = []
    for k in ks:
        cols.append(_pass_col(k))
        for tau in TAUS:
            cols.append(_gpass_col(k, tau))
        cols.append(_mgpass_col(k))
    return cols


def _ks_from_records(records: list[RunRecord]) -> list[int]:
    ks: set[int] = set()
    for r in records:
        ks.update(int(k) for k in (r.pass_at_k or {}))
        ks.update(int(k) for k in (r.g_pass_at_k or {}))
        ks.update(int(k) for k in (r.mg_pass_at_k or {}))
    return sorted(ks) if ks else list(CANONICAL_KS)


def wide_row_from_record(record: RunRecord) -> dict:
    """Flatten one :class:`RunRecord` into a single wide row dict."""
    jm, js = record.judge_model, record.judge_state
    stats = record.stats or {}
    row: dict = {
        "model": record.model,
        "model_short": labels.short_model(record.model),
        "model_family": labels.model_family(record.model),
        "model_size_b": labels.model_size_b(record.model),
        "is_base": labels.is_base_model(record.model),
        "state": record.state,
        "mode": labels.mode_label(record.state),
        "benchmark": record.benchmark,
        "language": labels.language(record.benchmark),
        "judged": bool(jm),
        "series": labels.series_label(jm, js),
        "judge_model": jm,
        "judge_state": js,
        "n_samples": record.n_samples,
        "temperature": record.temperature,
        "max_tokens": record.max_completion_tokens,
        "judge_n_samples": record.judge_n_samples,
        "judge_temperature": record.judge_temperature,
        "judge_max_tokens": record.judge_max_completion_tokens,
        "cons_at_k": record.cons_at_k,
        "total_tasks": stats.get("total_tasks"),
        "total_generations": stats.get("total_generations"),
        "true_count": stats.get("true_count"),
        "false_count": stats.get("false_count"),
        "cot_false_count": stats.get("cot_false_count"),
        "invalid_count": stats.get("invalid_count"),
        "run_dir": str(record.run_dir),
        "summary_path": str(record.summary_path),
    }
    for k, v in (record.pass_at_k or {}).items():
        row[_pass_col(int(k))] = v
    for k, taud in (record.g_pass_at_k or {}).items():
        for tau, v in (taud or {}).items():
            row[_gpass_col(int(k), str(tau))] = v
    for k, v in (record.mg_pass_at_k or {}).items():
        row[_mgpass_col(int(k))] = v
    return row


def row_key(record: RunRecord) -> tuple[str, str, str, str, str]:
    return (
        record.model or "", record.state or "", record.benchmark or "",
        record.judge_model or "", record.judge_state or "",
    )


def _metric_sort_key(col: str):
    """Order metric columns by (k, type, tau): pass, then g-pass τ…, then mg."""
    head, rest = col.split("@", 1)
    if "_t" in rest:
        kpart, tpart = rest.split("_t", 1)
        return (int(kpart), 1, float(tpart))
    typ = {"pass": 0, "mgpass": 2}.get(head, 3)
    return (int(rest), typ, 0.0)


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    meta = [c for c in META_COLUMNS if c in df.columns]
    metric = sorted((c for c in df.columns if c not in meta), key=_metric_sort_key)
    return df[meta + metric]


def build_wide_dataframe(records: Iterable[RunRecord]) -> pd.DataFrame:
    """Explode records into the wide one-row-per-run DataFrame."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - explicit, actionable error
        raise ImportError(
            "evalhub.report.aggregate requires pandas. Install with "
            "`pip install evalhub[report]` or `pip install pandas`."
        ) from e

    records = list(records)
    rows = [wide_row_from_record(r) for r in records]
    columns = META_COLUMNS + metric_columns(_ks_from_records(records))
    df = pd.DataFrame(rows)
    # Guarantee the full, ordered schema even when some metric cols are absent.
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    return _order_columns(df)


def write_csv(df: pd.DataFrame, output_csv: Path | str) -> Path:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df)} row(s) × {df.shape[1]} col(s) -> {out}")
    return out


# Model families excluded from the published report (target OR judge). Matched
# case-insensitively as a substring of the model name.
EXCLUDED_MODEL_PATTERNS: tuple[str, ...] = ("ministral", "mistral")
# Cap the reported K axis. Every target is n=64 except one n=128 run, which we
# evaluate on its first 64 samples for parity — so drop any K beyond this.
REPORT_MAX_K: int = 64


def _excluded(record: RunRecord, patterns: tuple[str, ...]) -> bool:
    names = [record.model or "", record.judge_model or ""]
    return any(p in n.lower() for n in names for p in patterns)


def _cap_k(record: RunRecord, max_k: int) -> RunRecord:
    """Return a copy of ``record`` with K > ``max_k`` dropped from metric blocks.

    ``RunRecord`` is frozen, so we rebuild via :func:`dataclasses.replace`.
    """
    from dataclasses import replace

    return replace(
        record,
        pass_at_k={k: v for k, v in (record.pass_at_k or {}).items() if int(k) <= max_k},
        g_pass_at_k=(
            {k: v for k, v in record.g_pass_at_k.items() if int(k) <= max_k}
            if record.g_pass_at_k else record.g_pass_at_k
        ),
        mg_pass_at_k=(
            {k: v for k, v in record.mg_pass_at_k.items() if int(k) <= max_k}
            if record.mg_pass_at_k else record.mg_pass_at_k
        ),
    )


# Tolerance for the cot ≤ No-Judge metric comparison.
_INTEGRITY_TOL = 1e-9


def check_report_integrity(records: list[RunRecord]) -> list[str]:
    """Cross-check every judged record against its No-Judge reference.

    The CoT veto can only *remove* correct answers, so for the same
    (model, state, benchmark) a judged row must satisfy two invariants:

    * **monotone** — CoT-Pass@K ≤ Pass@K (and CoT-Cons ≤ Cons) at every K.
    * **count**    — ``true_count + cot_false_count == base true_count`` (the
      judged population is exactly the base-correct generations).

    Returns a list of human-readable violation strings (empty == clean). This is
    the in-report mirror of ``scripts/audit_integrity.py``; it runs on summaries
    so it is cheap enough to gate every ``report aggregate``.
    """
    base: dict[tuple[str, str, str], RunRecord] = {
        (r.model, r.state, r.benchmark): r for r in records if not r.judge_model
    }
    violations: list[str] = []
    for r in records:
        if not r.judge_model:
            continue
        ref = base.get((r.model, r.state, r.benchmark))
        tag = f"{r.state}/{r.model}/{r.benchmark} · judge={r.judge_model}"
        if ref is None:
            violations.append(f"{tag}: judged row has no No-Judge reference")
            continue
        for k, v in (r.pass_at_k or {}).items():
            bv = (ref.pass_at_k or {}).get(k)
            if bv is not None and v > bv + _INTEGRITY_TOL:
                violations.append(f"{tag}: cot pass@{k}={v:.4f} > No-Judge {bv:.4f}")
        if r.cons_at_k is not None and ref.cons_at_k is not None and r.cons_at_k > ref.cons_at_k + _INTEGRITY_TOL:
            violations.append(f"{tag}: cot cons={r.cons_at_k:.4f} > No-Judge {ref.cons_at_k:.4f}")
        if r.stats and ref.stats:
            derived = (r.stats.get("true_count") or 0) + (r.stats.get("cot_false_count") or 0)
            base_true = ref.stats.get("true_count")
            if base_true is not None and derived != base_true:
                violations.append(
                    f"{tag}: count cot_true+cot_false={derived} != base_true={base_true}"
                )
    return violations


def aggregate_results(
    results_root: Path | str,
    output_csv: Path | str,
    exclude_patterns: tuple[str, ...] = EXCLUDED_MODEL_PATTERNS,
    max_k: int | None = REPORT_MAX_K,
) -> pd.DataFrame:
    """Scan ``results_root``, build the wide DataFrame, and write the CSV.

    Records whose target or judge model matches ``exclude_patterns`` are dropped,
    and metrics beyond ``max_k`` are trimmed, so the published report is uniform.
    Integrity violations (cot > No-Judge / count mismatch) are logged as warnings
    but never block the write — the report still builds so the issue is visible.
    """
    records = []
    dropped = 0
    for rec in scan_results(results_root):
        if _excluded(rec, exclude_patterns):
            dropped += 1
            continue
        if max_k is not None:
            rec = _cap_k(rec, max_k)
        records.append(rec)
    if dropped:
        logger.info(f"Excluded {dropped} record(s) matching {exclude_patterns}")
    violations = check_report_integrity(records)
    if violations:
        logger.warning(f"Integrity: {len(violations)} violation(s) in report data:")
        for v in violations:
            logger.warning(f"  - {v}")
    df = build_wide_dataframe(records)
    write_csv(df, output_csv)
    return df


# --- Incremental upsert -----------------------------------------------------


def _series_key(row) -> tuple[str, str, str, str, str]:
    import pandas as pd

    def s(x) -> str:
        return "" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x)

    return tuple(s(row.get(c)) for c in KEY_COLUMNS)  # type: ignore[return-value]


def upsert_record(record: RunRecord, csv_path: Path | str) -> Path:
    """Append-or-replace the single row for ``record`` (keyed by KEY_COLUMNS).

    Re-running for the same key replaces that row (idempotent); a new key adds a
    row; a K not yet seen grows the schema (existing rows get NA for it).
    """
    import pandas as pd

    csv_path = Path(csv_path)
    new_row = wide_row_from_record(record)
    new_key = row_key(record)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df):
            keep = [i for i, r in df.iterrows() if _series_key(r) != new_key]
            df = df.loc[keep]
    else:
        df = pd.DataFrame(columns=META_COLUMNS + metric_columns(CANONICAL_KS))

    for c in new_row:
        if c not in df.columns:
            df[c] = pd.NA
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = _order_columns(df)
    write_csv(df, csv_path)
    return csv_path


def upsert_summary(
    summary_path: Path | str, csv_path: Path | str,
    results_root: Path | str | None = None,
) -> Path:
    """Parse one summary file and upsert it into ``csv_path``."""
    record = record_from_summary(summary_path, results_root)
    if record is None:
        raise ValueError(
            f"Could not parse a run record from {summary_path} — its directory "
            "names don't match a recognised layout."
        )
    return upsert_record(record, csv_path)
