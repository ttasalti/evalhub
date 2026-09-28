"""Wide master-CSV writer: one row per evaluated (model, mode, benchmark, judge).

Each :class:`RunRecord` (one ``*_summary.json`` / ``*_cot_summary.json``) becomes
exactly **one row** carrying every metric at every K and τ:

* ``pass@{k}``: Pass@k
* ``gpass@{k}_t{tau}``: G-Pass@k at threshold τ
* ``mgpass@{k}``: mG-Pass@k

The **``judge_model`` column is the discriminator**: when it is empty the row is
the **No-Judge** reference and those columns mean pass / g-pass / mg-pass; when a
judge is set the same columns mean **cot-pass / cot-g-pass / cot-mg-pass**. There
are no separate ``cot_*`` columns.

Two entry points share the same schema:

* :func:`aggregate_results`: full rebuild from a results root.
* :func:`upsert_record` / :func:`upsert_summary`, append-or-replace a single
  row keyed by ``(model, state, benchmark, max_tokens, judge_model,
  judge_state, judge_max_tokens)``, so a pipeline can call it once per
  finished evaluation and the CSV grows row by row.
"""

from __future__ import annotations

import gzip
from collections import Counter
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
# Row identity. judge_* are empty on No-Judge rows. max_tokens/judge_max_tokens
# are part of the key (not just max_k/pass@k metadata) because two runs of the
# same model/state/benchmark/judge at different token caps (e.g. a 16k and a
# 32k think-mode re-run) are genuinely different runs, not the same run
# re-measured: without them here, upserting one silently deletes the other's
# row (and, via evalhub.report.tasks reusing this same key, its task rows too).
KEY_COLUMNS: tuple[str, ...] = (
    "model",
    "state",
    "benchmark",
    "max_tokens",
    "judge_model",
    "judge_state",
    "judge_max_tokens",
)

# Identity / metadata columns, in order. Metric columns are appended after these.
META_COLUMNS: list[str] = [
    "model",
    "model_short",
    "model_family",
    "model_size_b",
    "is_base",
    "state",
    "mode",
    "benchmark",
    "language",
    "judged",
    "series",
    "judge_model",
    "judge_state",
    "n_samples",
    "temperature",
    "max_tokens",
    "judge_n_samples",
    "judge_temperature",
    "judge_max_tokens",
    "reasoning_effort",
    "extra_body",
    "cons_at_k",
    "total_tasks",
    "total_generations",
    "true_count",
    "false_count",
    "cot_false_count",
    "invalid_count",
    # trunc_count/trunc_total/trunc_ratio used to be one column shared by two
    # meanings (generation truncation on a No-Judge row, judge-verdict
    # truncation on a judged row); split by row type so each name means one
    # thing. Only the pair matching the row's own type is populated.
    "gen_trunc_count",
    "gen_text_total",
    "gen_trunc_ratio",
    "loop_gzip_n",
    "loop_ngram_n",
    "loop_tail_n",
    "judge_verdict_trunc_count",
    "judge_verdict_total",
    "judge_selftrunc_ratio",
    # cot_false "why no?" breakdown (judge rows only; blank on No-Judge rows).
    # Mirrors COT_FALSE_COLUMNS produced by cot_false_invalid_stats().
    "cotfalse_total",
    "cot_false_complete_verdict",
    "cot_false_invalid1",
    "cot_false_invalid2",
    "cot_false_invalid3",
    "cot_false_complete_verdict_share",
    "cot_false_invalid1_share",
    "cot_false_invalid2_share",
    "cot_false_invalid3_share",
    "cot_false_trunc_induced",
    "trunc_induced_share",
    "cotfalse_tasks",
    "cotfalse_tasks_ratio",
    "cot_false_invalid_by_trunc",
    "cot_false_invalid_by_other",
    "cot_false_invalid_by_trunc_share",
    "run_dir",
    "summary_path",
]


def _json_loads(line: bytes):
    """orjson if present (raw files are large), else stdlib json."""
    try:
        import orjson

        return orjson.loads(line)
    except ImportError:
        import json

        return json.loads(line)


_LOOP_NGRAM_WINDOW = 20


def _dominant_ngram_share(words: list[str]) -> float:
    """Share of ``words`` covered by the most-repeated contiguous 20-gram.

    Overlapping occurrences let the raw share exceed 1.0, so it's clipped.
    Texts shorter than the window can't contain a repeated 20-gram at all.
    """
    n = len(words)
    if n < _LOOP_NGRAM_WINDOW:
        return 0.0
    counts = Counter(tuple(words[i : i + _LOOP_NGRAM_WINDOW]) for i in range(n - _LOOP_NGRAM_WINDOW + 1))
    top = counts.most_common(1)[0][1]
    return min(1.0, top * _LOOP_NGRAM_WINDOW / n)


def _is_gzip_loop(text: str) -> bool:
    """True when ``text`` compresses far more aggressively than natural prose.

    Repeated phrasing (a stuck reasoning loop) lets gzip encode it almost for
    free; natural, non-repetitive reasoning compresses far less.
    """
    raw = text.encode("utf-8")
    if not raw:
        return False
    return len(gzip.compress(raw)) / len(raw) < 0.15


def truncation_stats(summary_path: Path | str, judged: bool = False) -> dict:
    """Count generations that hit the max-token cap, plus repetition-loop signals.

    A generation is *truncated* when ``response.choices[0].finish_reason`` is
    ``"length"`` (vs. ``"stop"`` for a natural end). The raw file lives next to
    the summary: the model's ``<bench>_raw.jsonl`` for a No-Judge row, or the
    judge's ``cot_judge*_raw.jsonl`` for a judged row, so each row reports
    truncation of *its own* generations. The count is reported into one of two
    disjoint column pairs depending on the row's own type, so a single column
    never carries two different meanings: ``gen_trunc_count``/``gen_text_total``/
    ``gen_trunc_ratio`` on a No-Judge row (the model's own generations), or
    ``judge_verdict_trunc_count``/``judge_verdict_total``/``judge_selftrunc_ratio``
    on a judged row (the judge's own verdict-generation calls, note
    ``judge_verdict_total`` counts *verdicts*, which can exceed the target's own
    generation count). The non-applicable pair is always ``None``. Blank when no
    raw is found.

    On No-Judge rows (``judged=False``), each truncated generation's text is
    additionally scored, in the same read pass, for three independent
    repetition-loop signals, ``loop_gzip_n`` (gzip ratio < 0.15),
    ``loop_ngram_n`` (dominant 20-gram share > 0.30), ``loop_tail_n`` (dominant
    20-gram share in the last 25% of the text > 0.50). The three counters are
    independent (one generation can trip more than one). On judged rows the raw
    holds the judge's own short verdict text, which these signals aren't
    meaningful for, so they're always ``None`` there.
    """
    leaf = Path(summary_path).parent
    raws = sorted(leaf.glob("*_raw.jsonl"))
    empty_loop = {"loop_gzip_n": None, "loop_ngram_n": None, "loop_tail_n": None}

    def _split(count, tot, ratio) -> dict:
        gen = {"gen_trunc_count": None, "gen_text_total": None, "gen_trunc_ratio": None}
        jud = {
            "judge_verdict_trunc_count": None,
            "judge_verdict_total": None,
            "judge_selftrunc_ratio": None,
        }
        if judged:
            jud = {
                "judge_verdict_trunc_count": count,
                "judge_verdict_total": tot,
                "judge_selftrunc_ratio": ratio,
            }
        else:
            gen = {"gen_trunc_count": count, "gen_text_total": tot, "gen_trunc_ratio": ratio}
        return {**gen, **jud}

    if not raws:
        return {**_split(None, None, None), **empty_loop}
    total = trunc = 0
    loop_gzip_n = loop_ngram_n = loop_tail_n = 0
    for raw in raws:
        try:
            fh = open(raw, "rb")
        except OSError:
            continue
        with fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    obj = _json_loads(line)
                    choice = obj["response"]["choices"][0]
                    fr = choice.get("finish_reason")
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                total += 1
                if fr != "length":
                    continue
                trunc += 1
                if judged:
                    continue
                text = (choice.get("message") or {}).get("content") or ""
                if _is_gzip_loop(text):
                    loop_gzip_n += 1
                words = text.split()
                if _dominant_ngram_share(words) > 0.30:
                    loop_ngram_n += 1
                tail_words = words[int(len(words) * 0.75) :]
                if _dominant_ngram_share(tail_words) > 0.50:
                    loop_tail_n += 1
    loop_cols = (
        empty_loop if judged else {"loop_gzip_n": loop_gzip_n, "loop_ngram_n": loop_ngram_n, "loop_tail_n": loop_tail_n}
    )
    if total == 0:
        return {**_split(0, 0, None), **loop_cols}
    return {**_split(trunc, total, round(trunc / total, 6)), **loop_cols}


# Columns produced by cot_false_invalid_stats, declared here so the writer can
# blank them for No-Judge rows and keep a stable, ordered schema.
COT_FALSE_COLUMNS: tuple[str, ...] = (
    "cotfalse_total",
    "cot_false_complete_verdict",
    "cot_false_invalid1",
    "cot_false_invalid2",
    "cot_false_invalid3",
    "cot_false_complete_verdict_share",
    "cot_false_invalid1_share",
    "cot_false_invalid2_share",
    "cot_false_invalid3_share",
    "cot_false_trunc_induced",
    "trunc_induced_share",
    "cotfalse_tasks",
    "cotfalse_tasks_ratio",
    "cot_false_invalid_by_trunc",
    "cot_false_invalid_by_other",
    "cot_false_invalid_by_trunc_share",
)


def _verdict_class(token: str) -> str:
    """yes / no / invalid, mirrors evalhub.cot.aggregate._classify.

    Replicated here (3 lines) to keep the report layer decoupled from the
    benchmark/cot import graph; the rule is identical (anything that is not a
    bare ``yes``/``no`` extraction abstains as ``invalid``).
    """
    norm = (token or "").strip().lower()
    if norm in ("yes", "no"):
        return norm
    return "invalid"


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def cot_false_invalid_stats(summary_path: Path | str) -> dict:
    """Explain *why* a judge row's cot_false (vetoed) generations went "no".

    A base-correct generation is vetoed when ``yes_count <= no_count``
    (``majority_correct`` False in ``<bench>_cot_majority.jsonl``). For each such
    generation this reports, over three different denominators:

    * **generation level** (denom ``cotfalse_total`` == ``cot_false_count``):
      ``cot_false_complete_verdict`` (0 invalid verdicts) and
      ``cot_false_invalid{1..3}`` (+ ``_share``), how many of the judge's
      samples were *invalid* (no boxed yes/no, usually truncated). A pure "no"
      has 0 invalid; an all-invalid veto has 3. ``cot_false_trunc_induced``
      (+ ``trunc_induced_share``), vetoes that would have flipped to
      *approved* had the invalids been "yes" (``yes_count + invalid_count >
      no_count``): the veto was caused by invalidity, not a genuine "no".
    * **task level** (denom ``total_tasks``): ``cotfalse_tasks`` (+ ratio),
      distinct source questions with >=1 vetoed generation.
    * **verdict level** (denom = invalid verdicts inside cot_false generations):
      ``cot_false_invalid_by_trunc`` / ``cot_false_invalid_by_other``
      (+ ``_share`` on the trunc side), split the invalid samples into
      truncated (``finish_reason == "length"``) vs. other, via a positional
      ``zip`` of the line-aligned ``cot_judge*.jsonl`` (verdict class) and
      ``cot_judge*_raw.jsonl`` (finish_reason).

    Blank (all ``None``) for No-Judge rows, which have no majority file.
    """
    blank = dict.fromkeys(COT_FALSE_COLUMNS, None)
    leaf = Path(summary_path).parent
    majs = sorted(leaf.glob("*_cot_majority.jsonl"))
    if not majs:
        return blank

    cotfalse_ids: set[str] = set()
    affected_tasks: set[str] = set()
    buckets = {0: 0, 1: 0, 2: 0, 3: 0}
    flip = 0
    for maj in majs:
        with open(maj, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = _json_loads(line)
                except ValueError:
                    continue
                if r.get("majority_correct"):
                    continue
                # Prefer the precomputed counts (current majority schema); fall
                # back to the raw per-verdict "solutions" list, since legacy
                # majority files predate the yes/no/invalid_count fields.
                if r.get("invalid_count") is not None:
                    yc = int(r.get("yes_count", 0))
                    nc = int(r.get("no_count", 0))
                    ic = int(r.get("invalid_count", 0))
                else:
                    classes = [_verdict_class(str(s)) for s in (r.get("solutions") or [])]
                    yc = classes.count("yes")
                    nc = classes.count("no")
                    ic = classes.count("invalid")
                buckets[min(ic, 3)] += 1
                tid = str(r.get("task_id", ""))
                cotfalse_ids.add(tid)
                affected_tasks.add(tid.rsplit("_gen_", 1)[0])
                if yc + ic > nc:
                    flip += 1

    cf = sum(buckets.values())

    # total_tasks for the per-question ratio, read from this row's cot summary.
    try:
        total_tasks = int(_json_loads(Path(summary_path).read_bytes()).get("total_tasks") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        total_tasks = 0

    # Verdict-level truncation split, restricted to cot_false generations. The
    # per-verdict solution file and its raw counterpart are line-aligned (same
    # order, same task_id); zip pairs each verdict class with its finish_reason.
    # cf == 0 => no invalid verdicts to split (counts 0). When the raw/sol files
    # are missing or unaligned the split is left blank (None) rather than wrong.
    split_trunc: int | None = 0
    split_other: int | None = 0
    if cf:
        raws = sorted(leaf.glob("cot_judge*_raw.jsonl"))
        raw_path = raws[0] if raws else None
        sol_path = raw_path.with_name(raw_path.name[: -len("_raw.jsonl")] + ".jsonl") if raw_path is not None else None
        if raw_path is not None and sol_path is not None and sol_path.exists():
            t = o = 0
            aligned = True
            with open(sol_path, "rb") as fs, open(raw_path, "rb") as fr:
                for sl, rl in zip(fs, fr, strict=False):
                    if not sl.strip() or not rl.strip():
                        continue
                    try:
                        s = _json_loads(sl)
                        rr = _json_loads(rl)
                    except ValueError:
                        continue
                    if s.get("task_id") != rr.get("task_id"):
                        aligned = False
                        break
                    if s.get("task_id") not in cotfalse_ids:
                        continue
                    if _verdict_class(str(s.get("solution", ""))) != "invalid":
                        continue
                    try:
                        fin = rr["response"]["choices"][0].get("finish_reason")
                    except (KeyError, IndexError, TypeError):
                        fin = None
                    if fin == "length":
                        t += 1
                    else:
                        o += 1
            if aligned:
                split_trunc, split_other = t, o
            else:
                split_trunc = split_other = None
        else:
            split_trunc = split_other = None

    inv_total = (split_trunc + split_other) if split_trunc is not None else None

    return {
        "cotfalse_total": cf,
        "cot_false_complete_verdict": buckets[0],
        "cot_false_invalid1": buckets[1],
        "cot_false_invalid2": buckets[2],
        "cot_false_invalid3": buckets[3],
        "cot_false_complete_verdict_share": _ratio(buckets[0], cf),
        "cot_false_invalid1_share": _ratio(buckets[1], cf),
        "cot_false_invalid2_share": _ratio(buckets[2], cf),
        "cot_false_invalid3_share": _ratio(buckets[3], cf),
        "cot_false_trunc_induced": flip,
        "trunc_induced_share": _ratio(flip, cf),
        "cotfalse_tasks": len(affected_tasks),
        "cotfalse_tasks_ratio": _ratio(len(affected_tasks), total_tasks),
        "cot_false_invalid_by_trunc": split_trunc,
        "cot_false_invalid_by_other": split_other,
        "cot_false_invalid_by_trunc_share": _ratio(split_trunc, inv_total) if inv_total else None,
    }


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
        "reasoning_effort": record.reasoning_effort,
        "extra_body": record.extra_body,
        "cons_at_k": record.cons_at_k,
        "total_tasks": stats.get("total_tasks"),
        "total_generations": stats.get("total_generations"),
        "true_count": stats.get("true_count"),
        "false_count": stats.get("false_count"),
        "cot_false_count": stats.get("cot_false_count"),
        "invalid_count": stats.get("invalid_count"),
        **truncation_stats(record.summary_path, judged=bool(jm)),
        **cot_false_invalid_stats(record.summary_path),
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
    # any/all judge-approval-threshold blocks (judged rows only). The whole cot metric
    # suite is duplicated per threshold with a __any / __all suffix; majority stays
    # unsuffixed. Monotone: __all <= (unsuffixed majority) <= __any.
    for t, tm in (record.threshold_metrics or {}).items():
        for k, v in (tm.get("pass_at_k") or {}).items():
            row[f"{_pass_col(int(k))}__{t}"] = v
        for k, taud in (tm.get("g_pass_at_k") or {}).items():
            for tau, v in (taud or {}).items():
                row[f"{_gpass_col(int(k), str(tau))}__{t}"] = v
        for k, v in (tm.get("mg_pass_at_k") or {}).items():
            row[f"{_mgpass_col(int(k))}__{t}"] = v
        row[f"cons_at_k__{t}"] = tm.get("cons_at_k")
        row[f"true_count__{t}"] = tm.get("true_count")
        row[f"cot_false_count__{t}"] = tm.get("cot_false_count")
    return row


def _key_str(x) -> str:
    """Normalize one key field to a stable string.

    Handles two sources of drift between a freshly-built key (from a
    ``RunRecord``, plain Python ints/None) and one read back from a
    CSV-round-tripped DataFrame (numpy floats, since a NaN anywhere in the
    column: e.g. ``judge_max_tokens`` on No-Judge rows, upcasts the whole
    column to float64, turning ``16384`` into ``16384.0``): NaN/None both
    become ``""``, and a whole-number float is coerced back to int form so
    ``16384`` and ``16384.0`` compare equal.
    """
    import pandas as pd

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def row_key(record: RunRecord) -> tuple[str, ...]:
    return (
        record.model or "",
        record.state or "",
        record.benchmark or "",
        _key_str(record.max_completion_tokens),
        record.judge_model or "",
        record.judge_state or "",
        _key_str(record.judge_max_completion_tokens),
    )


def _metric_sort_key(col: str):
    """Order metric columns by (k, type, tau): pass, then g-pass τ…, then mg.

    Threshold-expanded cot columns (``pass@k__any``, ``cons_at_k__all``, …) and any
    other non-standard metric column can't be parsed on the K axis; they sort into a
    trailing block ordered by name, so they never disturb the standard K-axis order.
    """
    try:
        head, rest = col.split("@", 1)
        if "_t" in rest:
            kpart, tpart = rest.split("_t", 1)
            return (0, int(kpart), 1, float(tpart), "")
        typ = {"pass": 0, "mgpass": 2}.get(head, 3)
        return (0, int(rest), typ, 0.0, "")
    except (ValueError, IndexError):
        return (1, 0, 0, 0.0, col)


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


# Model families excluded from the published report (target OR judge), matched
# case-insensitively as a substring of the model name. Empty by default: every
# model (Mistral, Ministral, gemma, …) appears in report.csv. Callers can still
# pass `exclude_patterns=(...)` to aggregate_results for an ad-hoc filtered view.
EXCLUDED_MODEL_PATTERNS: tuple[str, ...] = ()
# Cap the reported K axis. Every target is n=64 except one n=128 run, which we
# evaluate on its first 64 samples for parity, so drop any K beyond this.
REPORT_MAX_K: int = 64


def _excluded(record: RunRecord, patterns: tuple[str, ...]) -> bool:
    names = [record.model or "", record.judge_model or ""]
    return any(p in n.lower() for n in names for p in patterns)


def _cap_k(record: RunRecord, max_k: int) -> RunRecord:
    """Return a copy of ``record`` with K > ``max_k`` dropped from metric blocks.

    ``RunRecord`` is frozen, so we rebuild via :func:`dataclasses.replace`.
    """
    from dataclasses import replace

    def _cap_block(block: dict | None) -> dict | None:
        return {k: v for k, v in block.items() if int(k) <= max_k} if block else block

    capped_thresholds = None
    if record.threshold_metrics:
        capped_thresholds = {
            t: {
                **tm,
                "pass_at_k": _cap_block(tm.get("pass_at_k")),
                "g_pass_at_k": _cap_block(tm.get("g_pass_at_k")),
                "mg_pass_at_k": _cap_block(tm.get("mg_pass_at_k")),
            }
            for t, tm in record.threshold_metrics.items()
        }

    return replace(
        record,
        pass_at_k={k: v for k, v in (record.pass_at_k or {}).items() if int(k) <= max_k},
        g_pass_at_k=(
            {k: v for k, v in record.g_pass_at_k.items() if int(k) <= max_k}
            if record.g_pass_at_k
            else record.g_pass_at_k
        ),
        mg_pass_at_k=(
            {k: v for k, v in record.mg_pass_at_k.items() if int(k) <= max_k}
            if record.mg_pass_at_k
            else record.mg_pass_at_k
        ),
        threshold_metrics=capped_thresholds,
    )


# Tolerance for the cot ≤ No-Judge metric comparison.
_INTEGRITY_TOL = 1e-9


def check_report_integrity(records: list[RunRecord]) -> list[str]:
    """Cross-check every judged record against its No-Judge reference.

    The CoT veto can only *remove* correct answers, so for the same
    (model, state, benchmark, max_tokens) a judged row must satisfy two
    invariants:

    * **monotone**: CoT-Pass@K ≤ Pass@K (and CoT-Cons ≤ Cons) at every K.
    * **count**: ``true_count + cot_false_count == base true_count`` (the
      judged population is exactly the base-correct generations).

    ``max_tokens`` is part of the reference lookup (not just model/state/
    benchmark) because two runs of the same model at different token caps
    (e.g. a 16k and a 32k re-run) are different runs with different true
    counts: without it, one variant's base row silently overwrites the
    other's in the lookup, comparing some judged rows against the wrong
    reference and reporting false "count mismatch" violations.

    Returns a list of human-readable violation strings (empty == clean). This is
    the in-report mirror of ``scripts/audit_integrity.py``; it runs on summaries
    so it is cheap enough to gate every ``report aggregate``.
    """
    base: dict[tuple[str, str, str, int], RunRecord] = {
        (r.model, r.state, r.benchmark, r.max_completion_tokens): r for r in records if not r.judge_model
    }
    violations: list[str] = []
    for r in records:
        if not r.judge_model:
            continue
        ref = base.get((r.model, r.state, r.benchmark, r.max_completion_tokens))
        tag = f"{r.state}/{r.model}/{r.benchmark} · judge={r.judge_model} · max_tokens={r.max_completion_tokens}"
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
                violations.append(f"{tag}: count cot_true+cot_false={derived} != base_true={base_true}")
        # any/all threshold blocks: __all <= majority <= __any, each <= No-Judge base,
        # and true_count + cot_false_count == base_true within every threshold.
        maj_pass, base_pass = (r.pass_at_k or {}), (ref.pass_at_k or {})
        for t, blk in (r.threshold_metrics or {}).items():
            for k, v in (blk.get("pass_at_k") or {}).items():
                ik = int(k)
                bv = base_pass.get(ik)
                if bv is not None and v > bv + _INTEGRITY_TOL:
                    violations.append(f"{tag}: cot pass@{k} [{t}]={v:.4f} > No-Judge {bv:.4f}")
                mv = maj_pass.get(ik)
                if mv is not None and t == "any" and v + _INTEGRITY_TOL < mv:
                    violations.append(f"{tag}: cot pass@{k} any={v:.4f} < majority {mv:.4f}")
                if mv is not None and t == "all" and v > mv + _INTEGRITY_TOL:
                    violations.append(f"{tag}: cot pass@{k} all={v:.4f} > majority {mv:.4f}")
            if ref.stats:
                base_true = ref.stats.get("true_count")
                tt, tcf = blk.get("true_count"), blk.get("cot_false_count")
                if base_true is not None and tt is not None and tcf is not None and tt + tcf != base_true:
                    violations.append(f"{tag}: count [{t}] true+cot_false={tt + tcf} != base_true={base_true}")
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
    but never block the write, the report still builds so the issue is visible.
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

    from evalhub.report.tasks import TASK_CSV_NAME, aggregate_task_rows

    aggregate_task_rows(records, Path(output_csv).parent / TASK_CSV_NAME)

    return df


# Incremental upsert


def _series_key(row) -> tuple[str, ...]:
    return tuple(_key_str(row.get(c)) for c in KEY_COLUMNS)


def upsert_record(
    record: RunRecord,
    csv_path: Path | str,
    exclude_patterns: tuple[str, ...] = EXCLUDED_MODEL_PATTERNS,
    max_k: int | None = REPORT_MAX_K,
) -> Path | None:
    """Append-or-replace the single row for ``record`` (keyed by KEY_COLUMNS).

    Re-running for the same key replaces that row (idempotent); a new key adds a
    row; a K not yet seen grows the schema (existing rows get NA for it).

    Applies the same ``exclude_patterns``/``max_k`` trimming as
    :func:`aggregate_results`, so a CSV built incrementally via this function
    matches one built via a full rescan for the same inputs. Returns ``None``
    (no-op, CSV untouched) if ``record`` is excluded.
    """
    import pandas as pd

    if _excluded(record, exclude_patterns):
        logger.info(f"upsert: skipped (excluded by {exclude_patterns}): {record.model}")
        return None
    if max_k is not None:
        record = _cap_k(record, max_k)

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

    from evalhub.report.tasks import TASK_CSV_NAME, upsert_task_rows

    upsert_task_rows(record, csv_path.parent / TASK_CSV_NAME)

    return csv_path


def upsert_summary(
    summary_path: Path | str,
    csv_path: Path | str,
    results_root: Path | str | None = None,
    exclude_patterns: tuple[str, ...] = EXCLUDED_MODEL_PATTERNS,
    max_k: int | None = REPORT_MAX_K,
) -> Path | None:
    """Parse one summary file and upsert it into ``csv_path``."""
    record = record_from_summary(summary_path, results_root)
    if record is None:
        raise ValueError(
            f"Could not parse a run record from {summary_path}, its directory names don't match a recognised layout."
        )
    return upsert_record(record, csv_path, exclude_patterns=exclude_patterns, max_k=max_k)
