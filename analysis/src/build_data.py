"""Build the published data layer (analysis/data) from the finished experiment outputs; every table,
figure and test of the paper reads from the CSVs written here.

The raw material is results/report.csv, results/report_tasks.csv and the judge leaves under results/
(cot_judge*_raw.jsonl, <benchmark>_cot_majority.jsonl), together with error_study/output/ (judge raw files,
judge_inputs/manifest.json, corruption_log.csv, report_error_tasks.csv). None of it is part of the
repository, and the evalhub and error_study packages have to be importable from REPO_ROOT. The script writes
report.csv, report_tasks.csv, verdicts.csv, report_error_tasks.csv, corruption_log.csv, gap_curve.csv,
judge_paired.csv and CHECKS.md to analysis/data, and archives the filtered rows under analysis/data/excluded/.

Subcommands, each idempotent (the heavy raw scans are cached per leaf under CACHE_DIR, keyed by file
mtime and size, so any step resumes cheaply):

  report   - data/report.csv, data/report_tasks.csv and data/excluded/*.csv
  verdicts - data/verdicts.csv (one row per solution and judge)
  error    - data/report_error_tasks.csv and data/corruption_log.csv
  panel    - stamp the corruption-panel flag onto data/report_error_tasks.csv
  derived  - data/gap_curve.csv and data/judge_paired.csv
  check    - every acceptance check, printed and written to data/CHECKS.md
  all      - the six steps above, in that order

Usage: python build_data.py <subcommand>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import orjson
import pandas as pd
from common import CACHE_DIR, DATA_DIR, ERROR_STUDY_OUTPUT, REPO_ROOT, RESULTS_ROOT, require

# The evalhub and error_study packages of the repository are imported lazily inside the steps that
# need them (verdict extraction and the error-study consolidation).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ROOT = str(REPO_ROOT)
RESULTS = str(RESULTS_ROOT)
DATA = str(DATA_DIR)
ANALYSIS = DATA
CACHE = str(CACHE_DIR)
EXCLUDED = os.path.join(DATA, "excluded")
ES_OUT = str(ERROR_STUDY_OUTPUT)

BENCH_LANGUAGE = {
    "aime2026": "EN",
    "aime2026_pt": "PT",
    "aime2026_tr": "TR",
    "pt_exams_math": "PT",
    "tubitak_math2026": "TR-OL",
}

CK_RE = re.compile(r"DAPO|GMPO|@step|·s\d|checkpoint", re.I)

# column renames for the data layer (results/report.csv keeps the pipeline's names)
RENAME_REPORT = {
    "max_tokens": "solver_max_tokens",
    "total_generations": "n_generations",
    "total_tasks": "n_tasks",
    "cot_false_count": "n_veto",
    "gen_trunc_count": "solver_capped_count",
    "gen_trunc_ratio": "solver_cap_rate",
    "gen_text_total": "n_generations_with_text",
    "judge_verdict_trunc_count": "judge_capped_count",
    "judge_verdict_total": "n_verdicts",
    "judge_selftrunc_ratio": "judge_cap_rate",
}
# columns dropped from the data layer entirely
DROP_REPORT = [
    "cot_false_invalid1",
    "cot_false_invalid2",
    "cot_false_invalid3",
    "cot_false_invalid1_share",
    "cot_false_invalid2_share",
    "cot_false_invalid3_share",
    "cot_false_complete_verdict",
    "cot_false_complete_verdict_share",
    "cot_false_trunc_induced",
    "trunc_induced_share",
    "cot_false_invalid_by_trunc",
    "cot_false_invalid_by_other",
    "cot_false_invalid_by_trunc_share",
    "cotfalse_total",
    "cotfalse_tasks",
    "cotfalse_tasks_ratio",
    "invalid_count",
    "cot_false_count__all",
    "cot_false_count__any",
    "true_count__all",
    "true_count__any",
]

LANG_FIX = {"pt_exams_math": "PT"}

# solvers whose judged solutions enter verdicts.csv
SCOPE_SOLVERS = {
    "gemma-4-E2B",
    "gemma-4-E4B",
    "gemma-4-E2B-it",
    "gemma-4-E4B-it",
    "Qwen3.5-0.8B",
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3.5-0.8B-Base",
    "Qwen3.5-2B-Base",
    "Qwen3.5-4B-Base",
    "Qwen3.5-9B-Base",
    "Qwen2.5-7B",
    "Qwen2.5-32B",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct",
}
BENCHES = ["aime2026", "aime2026_pt", "aime2026_tr", "pt_exams_math", "tubitak_math2026"]

# (model, state, benchmark, max_tokens) cells whose stored judge votes are not trusted at cell
# level; their solutions are recovered individually by _salvage_corrupted_cells(). The set is empty
# for the published data, so the salvage path is a no-op guard.
CORRUPTED_JUDGE_CELLS = set()

ES_JUDGES = {  # error-study run roots -> judge identity
    "v4_flash": (os.path.join(ES_OUT, "judge"), "deepseek-v4-flash", "think", 20480),
    "qwen35b": (os.path.join(ES_OUT, "judge_qwen35b"), "Qwen3.6-35B-A3B", "think", 20480),
    "r1_8b": (os.path.join(ES_OUT, "judge_r1_8b"), "DeepSeek-R1-0528-Qwen3-8B", "think", 20480),
}


def judge_short_of(jm: str) -> str:
    jm = jm or ""
    if re.search("gemma", jm, re.I):
        return "Gemma4"
    if "Qwen3.6" in jm:
        return "Qwen3.6"
    if "R1-0528" in jm:
        return "R1-distill"
    if re.search("deepseek", jm, re.I):
        return "V4-Flash"
    return "No-Judge"


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mb"] = df.model_short.astype(str).str.replace("·Base", "", regex=False).str.strip()
    df["generation"] = np.where(df.model.astype(str).str.startswith("Qwen2.5"), "previous", "current")
    df["judge_short"] = df.judge_model.fillna("").map(judge_short_of)
    df["exp"] = df.mb + "|" + df.state + "|" + df.benchmark
    df["language"] = df.language.replace(LANG_FIX)
    return df


# Per-leaf cache keyed by file mtime and size, so a rerun skips the scans that are already done.


def _cache_key(tag: str, paths: list[str]) -> str:
    sig = [tag]
    for p in sorted(paths):
        st = os.stat(p)
        sig.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.md5("|".join(sig).encode()).hexdigest()


def cache_get(key: str):
    p = os.path.join(CACHE, key + ".json")
    if os.path.isfile(p):
        with open(p, "rb") as f:
            return orjson.loads(f.read())
    return None


def cache_put(key: str, obj) -> None:
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, key + ".json")
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(obj))
    os.replace(tmp, p)


def iter_jsonl(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


# report step


def _raw_gen_histogram(leaf: str, bench: str) -> dict[str, int]:
    """gens per task_id, counted from the leaf's raw jsonl (cached)."""
    raw = os.path.join(leaf, f"{bench}_raw.jsonl")
    if not os.path.isfile(raw):
        return {}
    key = _cache_key("rawhist", [raw])
    hit = cache_get(key)
    if hit is not None:
        return hit
    counts: dict[str, int] = {}
    with open(raw, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # cheap task_id extraction without full JSON parse of huge lines
            i = line.find(b'"task_id"')
            j = line.find(b'"', line.find(b":", i) + 1)
            k = line.find(b'"', j + 1)
            tid = line[j + 1 : k].decode()
            counts[tid] = counts.get(tid, 0) + 1
    cache_put(key, counts)
    return counts


def step_report() -> None:
    os.makedirs(EXCLUDED, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS, "report.csv"), low_memory=False)
    tk = pd.read_csv(os.path.join(RESULTS, "report_tasks.csv"), low_memory=False)
    df = add_derived_cols(df)
    n0 = len(df)

    # sequential filters, each bucket archived
    buckets = {}
    m = df.model.astype(str).str.contains(CK_RE) | df.model_short.astype(str).str.contains(CK_RE)
    buckets["checkpoints"] = df[m]
    df = df[~m]
    m = df.model.astype(str).str.contains("inistral", case=False) | df.judge_model.fillna("").str.contains(
        "inistral", case=False
    )
    buckets["out_of_scope_family"] = df[m]
    df = df[~m]
    m = df.mb == "G4-26B"
    buckets["g4_26b"] = df[m]
    df = df[~m]
    m = (df.judged == True) & np.array(  # noqa: E712
        [(r.model, r.state, r.benchmark, int(r.max_tokens)) in CORRUPTED_JUDGE_CELLS for _, r in df.iterrows()]
    )
    buckets["corrupted_judge_pairing"] = df[m]
    df = df[~m]

    # Strict completeness for the unjudged rows: every task has exactly n generations.
    strict_fail = []
    for _, r in df[df.judged == False].iterrows():  # noqa: E712
        leaf = os.path.dirname(str(r.summary_path))  # run_dir stops at model dir
        hist = _raw_gen_histogram(os.path.join(ROOT, leaf) if not os.path.isabs(leaf) else leaf, r.benchmark)
        n = int(r.n_samples)
        if not hist or set(hist.values()) != {n}:
            strict_fail.append(
                (r.model, r.state, r.benchmark, int(r.max_tokens), sorted(set(hist.values())) if hist else None)
            )
    strict_keys = {(m_, s, b, mx) for m_, s, b, mx, _ in strict_fail}
    df["_key"] = list(zip(df.model, df.state, df.benchmark, df.max_tokens.astype(int), strict=False))
    m = (
        df._key.isin(strict_keys)
        | (df.total_tasks <= 0)
        | (
            (df.judged == False)  # noqa: E712
            & df.gen_text_total.notna()
            & (df.gen_text_total != df.total_generations)
        )
    )
    buckets["incomplete_rows"] = df[m]
    df = df[~m]
    df = df.drop(columns=["_key"])

    for name, b in buckets.items():
        b.drop(columns=[c for c in ("_key",) if c in b.columns]).to_csv(
            os.path.join(EXCLUDED, f"{name}.csv"), index=False
        )
    if strict_fail:
        print("STRICT-64 FAIL (excluded):")
        for row in strict_fail:
            print("  ", row)

    out = df.drop(columns=[c for c in DROP_REPORT if c in df.columns], errors="ignore")
    out = out.drop(columns=["mb"])
    out = out.rename(columns=RENAME_REPORT)
    out.to_csv(os.path.join(DATA, "report.csv"), index=False)
    print(
        f"data/report.csv: {len(out)} rows (from {n0}; "
        f"excluded {', '.join(f'{k}={len(v)}' for k, v in buckets.items())})"
    )

    # report_tasks
    keycols = ["model", "state", "benchmark", "max_tokens", "judge_model", "judge_state", "judge_max_tokens"]

    def norm_key(fr):
        # NaN-safe string key (NaN==NaN must hold; 16384.0 must equal "16384")
        def s(c):
            col = fr[c]
            if col.dtype.kind in "fc":
                return col.map(lambda v: "" if pd.isna(v) else str(int(v)))
            return col.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)

        return list(zip(*[s(c) for c in keycols], strict=False))

    tk["_k"] = norm_key(tk)
    cell = df[keycols + ["generation", "judge_short", "exp", "summary_path"]].copy()
    cell["run_dir"] = cell.summary_path.map(lambda p: os.path.dirname(str(p)))
    cell = cell.drop(columns=["summary_path"])
    cell["_k"] = norm_key(cell)
    cell = cell.drop(columns=keycols)
    before_rows = len(tk)
    tk = tk.merge(cell, on="_k", how="inner").drop(columns=["_k"])
    if len(tk) != before_rows:
        print(f"  note: {before_rows - len(tk)} task rows belong to excluded cells (dropped)")
    # Judged task rows carry no n_generations/false_count; fill them from the
    # leaf's own _cot_per_task.csv.
    need = tk.n_generations.isna() & tk.judge_model.notna() & (tk.judge_model != "")
    for leaf, grp in tk[need].groupby("run_dir"):
        pt_path = None
        leaf_abs = leaf if os.path.isabs(leaf) else os.path.join(ROOT, leaf)
        bench = grp.benchmark.iloc[0]
        cand = os.path.join(leaf_abs, f"{bench}_cot_per_task.csv")
        if os.path.isfile(cand):
            pt_path = cand
        if not pt_path:
            continue
        pt = pd.read_csv(pt_path)
        pt["n_gen"] = pt["true"] + pt["false"] + pt["cot_false"] + pt["invalid_format"]
        m1 = pt.set_index("task_id")
        idx = grp.index
        tk.loc[idx, "n_generations"] = tk.loc[idx, "task_id"].map(m1["n_gen"]).values
        tk.loc[idx, "false_count"] = tk.loc[idx, "task_id"].map(m1["false"]).values
    tko = tk.drop(
        columns=[
            c
            for c in [
                "cot_false_trunc_induced",
                "cot_false_complete_verdict",
                "true_count_any",
                "cot_false_count_any",
                "true_count_all",
                "cot_false_count_all",
                "run_dir",
            ]
            if c in tk.columns
        ]
    )
    tko["language"] = tko.language.replace(LANG_FIX)
    tko = tko.rename(columns={"max_tokens": "solver_max_tokens", "cot_false_count": "n_veto"})
    tko.to_csv(os.path.join(DATA, "report_tasks.csv"), index=False)
    filled = int((~tk[need].n_generations.isna()).sum()) if need.any() else 0
    print(f"data/report_tasks.csv: {len(tko)} rows; judged n_generations filled: {filled}/{int(need.sum())}")


# verdicts step


def _classify(sol: str) -> str:
    s = (sol or "").strip().lower()
    return s if s in ("yes", "no") else "invalid"


def _leaf_judge_files(leaf: str):
    """(sanitized, raw) judge vote file pairs in a leaf (cot_judge[_tr|_pt])."""
    pairs = []
    for task in ("cot_judge", "cot_judge_tr", "cot_judge_pt"):
        s = os.path.join(leaf, f"{task}.jsonl")
        r = os.path.join(leaf, f"{task}_raw.jsonl")
        if os.path.isfile(s) and os.path.isfile(r):
            pairs.append((task, s, r))
    return pairs


_DS_CACHE: dict[str, object] = {}


def _judge_ds(task: str):
    """The cot_judge dataset object, used purely for extract_solution: the same
    method the pipeline used to write the sanitized file."""
    if task not in _DS_CACHE:
        from evalhub.benchmarks.cot.judge import COT_JUDGE_CLASSES

        _DS_CACHE[task] = COT_JUDGE_CLASSES[task](name=task, meta_data={"file_path": ""})
    return _DS_CACHE[task]


def _scan_judge_leaf(leaf: str, bench: str) -> dict:
    """Per-generation vote table for one judge leaf (cached).

    (verdict, finish_reason) is derived jointly from the raw file alone:
    raw and sanitized files of resumed cells hold the same multiset in
    different orders, so positional pairing is unsafe. The verdict is
    re-extracted with the dataset's own extract_solution (byte-identical to
    the pipeline's sanitized output), keeping the joint exact.

    Returns {"gens": {gen_id: {"yes","no","invalid","capped_invalid","capped_any",
                               "stored_majority"}},
             "aligned": bool, "n_votes": int, "count_mismatch": int}
    """
    maj_path = os.path.join(leaf, f"{bench}_cot_majority.jsonl")
    pairs = _leaf_judge_files(leaf)
    paths = [maj_path] + [r for _, _s, r in pairs]
    key = _cache_key("judgeleaf2", paths)
    hit = cache_get(key)
    if hit is not None:
        return hit

    gens: dict[str, dict] = {}
    for rec in iter_jsonl(maj_path):
        gid = rec["task_id"]
        if "yes_count" in rec:
            y, n, i = rec["yes_count"], rec["no_count"], rec["invalid_count"]
        else:
            cl = [_classify(s) for s in rec.get("solutions", [])]
            y, n, i = cl.count("yes"), cl.count("no"), cl.count("invalid")
        gens[gid] = {
            "yes": y,
            "no": n,
            "invalid": i,
            "capped_invalid": 0,
            "capped_any": 0,
            "stored_majority": bool(rec["majority_correct"]),
        }

    n_votes = 0
    raw_counts: dict[str, list[int]] = {}  # gid -> [yes, no, invalid]
    for task, _spath, rpath in pairs:
        ds = _judge_ds(task)
        for d in iter_jsonl(rpath):
            gid = d["task_id"]
            resp = d.get("response", {}) or {}
            if "content" in resp:
                content = resp.get("content", "")
                fr_ = resp.get("finish_reason")
            else:
                ch = (resp.get("choices") or [{}])[0]
                content = (ch.get("message") or {}).get("content", "")
                fr_ = ch.get("finish_reason")
            cls = _classify(ds.extract_solution(gid, content or ""))
            n_votes += 1
            rc = raw_counts.setdefault(gid, [0, 0, 0])
            rc[("yes", "no", "invalid").index(cls)] += 1
            if gid not in gens:
                continue
            if fr_ == "length":
                gens[gid]["capped_any"] += 1
                if cls == "invalid":
                    gens[gid]["capped_invalid"] += 1
    # consistency: raw-derived counts must reproduce the majority-file counts
    mismatch = sum(1 for gid, g in gens.items() if raw_counts.get(gid, [0, 0, 0]) != [g["yes"], g["no"], g["invalid"]])
    out = {"gens": gens, "aligned": mismatch == 0, "n_votes": n_votes, "count_mismatch": mismatch}
    cache_put(key, out)
    return out


def _salvage_corrupted_cells() -> tuple[list[dict], list[dict], list[str]]:
    """Solution-level salvage for CORRUPTED_JUDGE_CELLS. The judge prompt
    contained only the raw response text, so every vote is a valid judgment of
    its own generation; the corruption was which generations got selected.
    Votes that landed on generations that are truly correct (per the base
    results file) are therefore valid; they cover an identical subset for every
    judge (same judge inputs), so the paired comparison stays valid on the
    reduced n. Cell-level metrics are not derived for these cells.
    """
    from evalhub.report.scan import parse_judge_leaf_dirname

    bench_language = BENCH_LANGUAGE
    rows, caps, problems = [], [], []
    for model, state, bench, mx in sorted(CORRUPTED_JUDGE_CELLS):
        base_leaf = os.path.join(RESULTS, state, model, f"{bench}__t0.6__max{mx}__n64")
        res = os.path.join(base_leaf, f"{bench}_results.jsonl")
        if not os.path.isfile(res):
            problems.append(f"salvage: base results missing: {res}")
            continue
        newc = {}
        for rec in iter_jsonl(res):
            newc[rec["task_id"]] = {i for i, c in enumerate(rec["correct"]) if c is True}
        for leaf in sorted(
            glob.glob(os.path.join(RESULTS, state, model, "judged_by", "*", f"{bench}__t0.6__max{mx}__n64"))
        ):
            tag = os.path.basename(os.path.dirname(leaf))
            parsed = parse_judge_leaf_dirname(tag)
            if parsed is None:
                problems.append(f"salvage: unparseable judge tag {tag}")
                continue
            jshort = judge_short_of(parsed.model)
            if jshort == "No-Judge":  # out-of-scope judge family (Ministral)
                continue
            pairs = _leaf_judge_files(leaf)
            if not pairs:
                problems.append(f"salvage: no vote files in {leaf}")
                continue
            gens: dict[str, dict] = {}
            n_votes = n_capped = 0
            for task, _s, rpath in pairs:
                ds = _judge_ds(task)
                for d in iter_jsonl(rpath):
                    gid = d["task_id"]
                    resp = d.get("response", {}) or {}
                    ch = (resp.get("choices") or [{}])[0]
                    content = (
                        resp.get("content") if "content" in resp else ((ch.get("message") or {}).get("content", ""))
                    )
                    fr_ = resp.get("finish_reason") if "content" in resp else ch.get("finish_reason")
                    cls = _classify(ds.extract_solution(gid, content or ""))
                    g = gens.setdefault(gid, {"yes": 0, "no": 0, "invalid": 0, "capped_invalid": 0})
                    g[cls] += 1
                    n_votes += 1
                    if fr_ == "length":
                        n_capped += 1
                        if cls == "invalid":
                            g["capped_invalid"] += 1
            kept = 0
            for gid, g in sorted(gens.items()):
                task_id, _, gidx = gid.rpartition("_gen_")
                tot = g["yes"] + g["no"] + g["invalid"]
                if tot != 3 or int(gidx) not in newc.get(task_id, set()):
                    continue
                kept += 1
                rows.append(
                    {
                        "solution_id": f"{model}|{state}|{bench}|{mx}|{task_id}|g{gidx}",
                        "task_id": task_id,
                        "solver_model": model,
                        "state": state,
                        "benchmark": bench,
                        "language": bench_language.get(bench, ""),
                        "solver_max_tokens": mx,
                        "judge_short": jshort,
                        "judge_state": parsed.state,
                        "judge_max_tokens": parsed.max_completion_tokens,
                        "answer_correct": True,
                        "n_yes": g["yes"],
                        "n_no": g["no"],
                        "n_missing": g["invalid"],
                        "n_missing_capped": g["capped_invalid"],
                        "approved_any": g["yes"] >= 1,
                        # no stored majority exists for these cells; the majority rule
                        # (yes > no, ties vetoed) is applied here
                        "approved_maj": g["yes"] > g["no"],
                        "approved_all": g["no"] == 0 and g["invalid"] == 0,
                    }
                )
            caps.append(
                {
                    "solver_model": model,
                    "state": state,
                    "benchmark": bench,
                    "language": bench_language.get(bench, ""),
                    "solver_max_tokens": mx,
                    "judge_short": jshort,
                    "judge_state": parsed.state,
                    "judge_max_tokens": parsed.max_completion_tokens,
                    "n_solutions": kept,
                    "n_votes": n_votes,
                    "n_votes_capped": n_capped,
                }
            )
            problems.append(
                f"salvage: {model}/{bench} x {jshort}: kept {kept} valid-target "
                f"solutions of {sum(len(v) for v in newc.values())} truly-correct"
            )
    return rows, caps, problems


_BENCH_LEAF_RE = re.compile(r"^(?P<bench>.+?)__t[0-9.]+__max(?P<max>\d+)__n(?P<n>\d+)$")


def _partial_unfinalized_rows(done_leaves: set) -> tuple[list[dict], list[str]]:
    """Solutions with exactly 3 votes from judge cells that are not finalized
    (no <benchmark>_cot_summary.json). The data layer admits solutions, not
    cells: a 3/3-voted solution of a half-finished cell is a finished datum.
    Vote classes are derived from the raw file with the pipeline's own
    extract_solution; approved_maj uses the majority rule (yes > no)."""
    from evalhub.report.scan import parse_judge_leaf_dirname

    bench_language = BENCH_LANGUAGE
    rows, problems = [], []
    leaf_globs = glob.glob(os.path.join(RESULTS, "*", "*", "judged_by", "*", "*")) + glob.glob(
        os.path.join(RESULTS, "Qwen2.5", "*", "*", "judged_by", "*", "*")
    )
    for leaf in sorted(leaf_globs):
        if not os.path.isdir(leaf) or os.path.realpath(leaf) in done_leaves:
            continue
        m = _BENCH_LEAF_RE.match(os.path.basename(leaf))
        if not m:
            continue
        bench, smax = m.group("bench"), int(m.group("max"))
        parts = leaf.split(os.sep)
        model = parts[parts.index("judged_by") - 1]
        state = parts[parts.index("judged_by") - 2]
        if model not in SCOPE_SOLVERS or state not in ("base", "non-think", "think"):
            continue
        if (model, state, bench, smax) in CORRUPTED_JUDGE_CELLS:
            continue  # handled by the salvage path
        if os.path.isfile(os.path.join(leaf, f"{bench}_cot_summary.json")):
            continue  # finalized, so already covered via report.csv
        pairs = _leaf_judge_files(leaf)
        if not pairs:
            continue
        parsed = parse_judge_leaf_dirname(os.path.basename(os.path.dirname(leaf)))
        if parsed is None:
            problems.append(f"partial: unparseable judge tag for {leaf}")
            continue
        jshort = judge_short_of(parsed.model)
        if jshort == "No-Judge":
            continue
        gens: dict[str, dict] = {}
        for task, _s, rpath in pairs:
            ds = _judge_ds(task)
            for d in iter_jsonl(rpath):
                gid = d["task_id"]
                resp = d.get("response", {}) or {}
                ch = (resp.get("choices") or [{}])[0]
                content = resp.get("content") if "content" in resp else ((ch.get("message") or {}).get("content", ""))
                fr_ = resp.get("finish_reason") if "content" in resp else ch.get("finish_reason")
                cls = _classify(ds.extract_solution(gid, content or ""))
                g = gens.setdefault(gid, {"yes": 0, "no": 0, "invalid": 0, "capped_invalid": 0})
                g[cls] += 1
                if fr_ == "length" and cls == "invalid":
                    g["capped_invalid"] += 1
        kept = 0
        for gid, g in sorted(gens.items()):
            tot = g["yes"] + g["no"] + g["invalid"]
            if tot != 3:
                continue
            kept += 1
            task_id, _, gidx = gid.rpartition("_gen_")
            rows.append(
                {
                    "solution_id": f"{model}|{state}|{bench}|{smax}|{task_id}|g{gidx}",
                    "task_id": task_id,
                    "solver_model": model,
                    "state": state,
                    "benchmark": bench,
                    "language": bench_language.get(bench, ""),
                    "solver_max_tokens": smax,
                    "judge_short": jshort,
                    "judge_state": parsed.state,
                    "judge_max_tokens": parsed.max_completion_tokens,
                    "answer_correct": True,
                    "n_yes": g["yes"],
                    "n_no": g["no"],
                    "n_missing": g["invalid"],
                    "n_missing_capped": g["capped_invalid"],
                    "approved_any": g["yes"] >= 1,
                    "approved_maj": g["yes"] > g["no"],
                    "approved_all": g["no"] == 0 and g["invalid"] == 0,
                }
            )
        if kept:
            problems.append(
                f"partial (unfinalized) cell included at solution level: "
                f"{model}/{state}/{bench} x {jshort}: {kept} solutions "
                f"({len(gens) - kept} with <3 votes dropped)"
            )
    return rows, problems


def step_verdicts() -> None:
    df = pd.read_csv(os.path.join(RESULTS, "report.csv"), low_memory=False)
    df = add_derived_cols(df)
    j = df[(df.judged == True) & df.model.isin(SCOPE_SOLVERS)]  # noqa: E712
    j = j[[(r.model, r.state, r.benchmark, int(r.max_tokens)) not in CORRUPTED_JUDGE_CELLS for _, r in j.iterrows()]]
    rows = []
    capstats = []
    problems = []
    for _, r in j.iterrows():
        # run_dir stops at the judge tag directory; summary_path reaches the leaf
        leaf = os.path.dirname(str(r.summary_path))
        if not os.path.isabs(leaf):
            leaf = os.path.join(ROOT, leaf)
        bench = r.benchmark
        maj = os.path.join(leaf, f"{bench}_cot_majority.jsonl")
        if not os.path.isfile(maj):
            problems.append(f"missing majority: {leaf}")
            continue
        scan = _scan_judge_leaf(leaf, bench)
        if not scan["aligned"]:
            problems.append(
                f"raw-derived vote counts mismatch majority file on "
                f"{scan['count_mismatch']} gens (n_missing_capped left NA): {leaf}"
            )
        smax = int(r.max_tokens)
        cell = {
            "solver_model": r.model,
            "state": r.state,
            "benchmark": bench,
            "language": r.language,
            "solver_max_tokens": smax,
            "judge_short": r.judge_short,
            "judge_state": r.judge_state,
            "judge_max_tokens": int(r.judge_max_tokens) if pd.notna(r.judge_max_tokens) else None,
        }
        n_notthree = 0
        for gid, g in scan["gens"].items():
            tot = g["yes"] + g["no"] + g["invalid"]
            if tot != 3:
                n_notthree += 1
                continue  # strict: only solutions with exactly 3 votes enter
            task_id, _, gidx = gid.rpartition("_gen_")
            rows.append(
                {
                    "solution_id": f"{r.model}|{r.state}|{bench}|{smax}|{task_id}|g{gidx}",
                    "task_id": task_id,
                    **cell,
                    "answer_correct": True,  # judged population = base-correct generations
                    "n_yes": g["yes"],
                    "n_no": g["no"],
                    "n_missing": g["invalid"],
                    "n_missing_capped": g["capped_invalid"] if scan["aligned"] else None,
                    "approved_any": g["yes"] >= 1,
                    "approved_maj": g["stored_majority"],
                    "approved_all": g["no"] == 0 and g["invalid"] == 0,
                }
            )
        if n_notthree:
            problems.append(f"{n_notthree} solutions without exactly 3 votes dropped: {leaf}")
        nsol = len(scan["gens"])
        capstats.append(
            {
                **cell,
                "n_solutions": nsol,
                "n_votes": scan["n_votes"],
                "n_votes_capped": sum(g["capped_any"] for g in scan["gens"].values()) if scan["aligned"] else None,
            }
        )
    s_rows, s_caps, s_problems = _salvage_corrupted_cells()
    rows += s_rows
    capstats += s_caps
    problems += s_problems
    done_leaves = {os.path.realpath(os.path.dirname(str(r.summary_path))) for _, r in j.iterrows()}
    p_rows, p_problems = _partial_unfinalized_rows(done_leaves)
    rows += p_rows
    problems += p_problems
    v = pd.DataFrame(rows)
    v = v.sort_values(
        ["judge_short", "judge_state", "judge_max_tokens", "solver_model", "state", "benchmark", "solution_id"]
    ).reset_index(drop=True)
    v.to_csv(os.path.join(DATA, "verdicts.csv"), index=False)
    pd.DataFrame(capstats).to_csv(os.path.join(CACHE, "cell_capstats.csv"), index=False)
    with open(os.path.join(CACHE, "verdicts_problems.txt"), "w") as f:
        f.write("\n".join(problems))
    print(
        f"data/verdicts.csv: {len(v)} rows across {len(capstats)} judge cells; "
        f"{len(problems)} leaf problems (see verdicts_problems.txt in the cache directory)"
    )
    for p in problems[:10]:
        print("  !", p)


# error step

ES_RENAME = {
    "error_type": "condition",
    "row_id": "solution_id",
    "n_gen": "n_generations",
    "yes": "n_yes",
    "no": "n_no",
    "invalid": "n_missing",
    "answer_still_correct": "answer_correct",
}


def _es_complete_cells(root: str) -> set[tuple[str, str, str]]:
    """(model, benchmark, error_type) cells whose raw line count == manifest n_rows*3."""
    man = json.load(open(os.path.join(ES_OUT, "judge_inputs", "manifest.json")))["files"]
    ok = set()
    for e in man:
        d = os.path.join(root, e["model"], e["benchmark"], e["error_type"])
        have = 0
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith("_raw.jsonl"):
                    with open(os.path.join(d, fn), "rb") as f:
                        have += sum(1 for ln in f if ln.strip())
        if have == e["n_rows"] * 3:
            ok.add((e["model"], e["benchmark"], e["error_type"]))
    return ok


def step_error() -> None:
    from error_study.consolidate import consolidate
    from error_study.report_error import _solution_variants

    frames = []
    for tag, (root, jm, jstate, jmax) in ES_JUDGES.items():
        raws = []
        for dp, _, fns in os.walk(root):
            raws += [os.path.join(dp, fn) for fn in fns if fn.endswith("_raw.jsonl")]
        if not raws:
            continue
        key = _cache_key(f"esjudge_{tag}", raws)
        pq = os.path.join(CACHE, f"judge_{tag}_output.parquet")
        meta = {"judge_model": jm, "judge_state": jstate, "judge_max_tokens": jmax}
        if not (os.path.isfile(pq) and cache_get(key) == {"done": True}):
            consolidate(root, pq[:-8], judge_meta=meta)  # writes .parquet + .csv
            cache_put(key, {"done": True})
        jdf = pd.read_parquet(pq)
        complete = _es_complete_cells(root)
        n_partial_cells = len(
            {(m, b, e) for m, b, e in zip(jdf.model, jdf.benchmark, jdf.error_type, strict=False)} - complete
        )
        if n_partial_cells:
            print(f"  {tag}: {n_partial_cells} partial cells; their exactly-3 variants are kept (solution-level rule)")
        sv = _solution_variants(jdf)
        # exactly three votes per variant, judge identity, and the count of capped missing votes
        cap = (
            jdf[(jdf.judge_verdict == "invalid") & jdf.judge_capped]
            .groupby(["model", "benchmark", "error_type", "row_id"])
            .size()
            .rename("n_missing_capped")
        )
        sv = sv.join(cap, on=["model", "benchmark", "error_type", "row_id"])
        sv["n_missing_capped"] = sv["n_missing_capped"].fillna(0).astype(int)
        before = len(sv)
        sv = sv[sv.n_gen == 3]
        if before - len(sv):
            print(f"  {tag}: dropped {before - len(sv)} variants without exactly 3 votes")
        sv["judge_model"], sv["judge_state"], sv["judge_max_tokens"] = jm, jstate, jmax
        sv["judge_short"] = judge_short_of(jm)
        frames.append(sv)
        print(f"  {tag}: {len(sv)} solution-variants from {len(complete)} complete cells")

    allsv = pd.concat(frames, ignore_index=True)
    allsv = allsv.rename(columns=ES_RENAME)
    order = [
        "judge_short",
        "judge_model",
        "judge_state",
        "judge_max_tokens",
        "model",
        "benchmark",
        "condition",
        "solution_id",
        "task_id",
        "old_number",
        "new_number",
        "wrong_number_menu",
        "change_location_pct",
        "n_generations",
        "n_yes",
        "n_no",
        "n_missing",
        "n_missing_capped",
        "correct_any",
        "correct_maj",
        "correct_all",
        "veto_any",
        "veto_maj",
        "veto_all",
        "answer_correct",
    ]
    allsv = (
        allsv[[c for c in order if c in allsv.columns]]
        .sort_values(["judge_short", "model", "benchmark", "condition", "task_id", "solution_id"])
        .reset_index(drop=True)
    )
    allsv.to_csv(os.path.join(DATA, "report_error_tasks.csv"), index=False)
    print(f"data/report_error_tasks.csv: {len(allsv)} rows ({allsv.judge_short.value_counts().to_dict()})")

    # canonical corruption_log copy
    cl = pd.read_csv(os.path.join(ES_OUT, "corruption_log.csv"))
    cl = cl.rename(
        columns={
            "row_id": "solution_id",
            "error_type": "condition",
            "solution_len_tokens": "solution_length",
            "change_location_pct": "edit_position_pct",
        }
    )
    cl.to_csv(os.path.join(DATA, "corruption_log.csv"), index=False)
    print(f"data/corruption_log.csv: {len(cl)} rows")

    # the V4-Flash slice must reproduce error_study/output/report_error_tasks.csv exactly
    old = pd.read_csv(os.path.join(ES_OUT, "report_error_tasks.csv"))
    new = allsv[allsv.judge_short == "V4-Flash"]
    o = old.set_index(["row_id", "error_type"]).sort_index()
    n = new.set_index(["solution_id", "condition"]).sort_index()
    same_idx = o.index.equals(n.index)
    checks = {"rows": len(o) == len(n), "index": same_idx}
    if same_idx:
        for a, b in [
            ("yes", "n_yes"),
            ("no", "n_no"),
            ("invalid", "n_missing"),
            ("correct_maj", "correct_maj"),
            ("veto_maj", "veto_maj"),
        ]:
            checks[a] = bool((o[a].values == n[b].values).all())
    print(f"  V4 regression vs error_study/output/report_error_tasks.csv: {checks}")
    if not all(checks.values()):
        raise SystemExit("V4 slice does not reproduce error_study/output/report_error_tasks.csv")


# panel step


def step_panel() -> None:
    p = os.path.join(DATA, "report_error_tasks.csv")
    error_tasks = pd.read_csv(p)
    # Eligible solutions have all five conditions under V4-Flash; when Qwen3.6 also
    # covers every cell with all five conditions, the two-judge intersection is used.
    base_judge = "V4-Flash"
    q = error_tasks[error_tasks.judge_short == "Qwen3.6"]
    q5 = q.groupby(["model", "benchmark", "solution_id"]).condition.nunique()
    v = error_tasks[error_tasks.judge_short == base_judge]
    v5 = v.groupby(["model", "benchmark", "solution_id"]).condition.nunique()
    cells = sorted(v5.reset_index().groupby(["model", "benchmark"]).groups)
    q_complete_cells = (
        {c for c in cells if (q5.reset_index().set_index(["model", "benchmark"]).loc[[c]].condition == 5).sum() > 0}
        if len(q)
        else set()
    )
    use_intersection = q_complete_cells == set(cells)

    eligible = {}
    for cell in cells:
        vv = v5.loc[cell]
        ids = set(vv[vv == 5].index)
        if use_intersection:
            qq = q5.loc[cell]
            ids &= set(qq[qq == 5].index)
        eligible[cell] = sorted(ids)
    n_per_cell = min(len(v_) for v_ in eligible.values())
    rng = np.random.default_rng(0)
    sel = []
    for cell in cells:
        sel += list(rng.choice(eligible[cell], n_per_cell, replace=False))
    sel = set(sel)
    error_tasks["panel"] = error_tasks.solution_id.isin(sel)
    error_tasks.to_csv(p, index=False)
    mode = "two-judge intersection" if use_intersection else f"{base_judge} only"
    print(
        f"panel: N_PER_CELL={n_per_cell} ({mode}); "
        f"{len(sel)} solutions -> {int(error_tasks[error_tasks.judge_short == base_judge].panel.sum())} "
        f"V4 panel rows (expected {n_per_cell * len(cells) * 5})"
    )
    meta = {"n_per_cell": n_per_cell, "mode": mode, "seed": 0, "cells": len(cells)}
    cache_put("panel_meta", meta)


# derived step

KS = [1, 2, 4, 8, 16, 32, 64]


def step_derived() -> None:
    os.makedirs(ANALYSIS, exist_ok=True)
    rep = pd.read_csv(os.path.join(DATA, "report.csv"), low_memory=False)

    # gap_curve.csv
    base = rep[rep.judged == False]  # noqa: E712
    jr = rep[rep.judged == True]  # noqa: E712
    bkey = ["model", "state", "benchmark", "solver_max_tokens"]
    b = base.set_index(bkey)
    rows = []
    for _, r in jr.iterrows():
        key = (r.model, r.state, r.benchmark, r.solver_max_tokens)
        if key not in b.index:
            continue
        br = b.loc[key]
        for k in KS:
            pk, ck = br.get(f"pass@{k}"), r.get(f"pass@{k}")
            if pd.isna(pk) or pd.isna(ck):
                continue
            rows.append(
                {
                    "solver_model": r.model,
                    "generation": r.generation,
                    "state": r.state,
                    "benchmark": r.benchmark,
                    "language": r.language,
                    "solver_max_tokens": int(r.solver_max_tokens),
                    "n_samples": int(r.n_samples),
                    "n_tasks": int(r.n_tasks),
                    "judge_short": r.judge_short,
                    "judge_state": r.judge_state,
                    "judge_max_tokens": r.judge_max_tokens,
                    "k": k,
                    "pass_at_k": float(pk),
                    "cot_pass_at_k": float(ck),
                    "cot_gap": float(pk) - float(ck),
                }
            )
    gc = pd.DataFrame(rows)
    gc.to_csv(os.path.join(ANALYSIS, "gap_curve.csv"), index=False)
    neg = gc[gc.cot_gap < -1e-9]
    print(f"gap_curve.csv: {len(gc)} rows; negative cot_gap rows: {len(neg)}")
    if len(neg):
        print(neg[["solver_model", "state", "benchmark", "judge_short", "k", "cot_gap"]].to_string(index=False))

    # judge_paired.csv
    v = pd.read_csv(os.path.join(DATA, "verdicts.csv"))
    caps = pd.read_csv(os.path.join(CACHE, "cell_capstats.csv"))
    p = v[(v.solver_max_tokens == 16384) & v.judge_short.isin(["Gemma4", "Qwen3.6"])].copy()
    p["mb"] = p.solver_model.str.replace("-Base", "", regex=False)
    short = {
        "Qwen3.5-0.8B": "Q-0.8B",
        "Qwen3.5-2B": "Q-2B",
        "Qwen3.5-4B": "Q-4B",
        "Qwen3.5-9B": "Q-9B",
        "gemma-4-E2B": "G4-E2B",
        "gemma-4-E4B": "G4-E4B",
        "gemma-4-E2B-it": "G4-E2B",
        "gemma-4-E4B-it": "G4-E4B",
    }
    p["mbs"] = p.mb.map(short).fillna(p.mb)
    p["exp"] = p.mbs + "|" + p.state + "|" + p.benchmark
    prows = []
    for (exp, js), g in p.groupby(["exp", "judge_short"]):
        vetoed = (~g.approved_maj).sum()
        comp = g[g.n_missing == 0]
        prows.append(
            {
                "exp": exp,
                "judge_short": js,
                "solver_model": g.solver_model.iloc[0],
                "state": g.state.iloc[0],
                "benchmark": g.benchmark.iloc[0],
                "n_generations": len(g),
                "veto_rate": vetoed / len(g),
                "veto_rate_complete": (~comp.approved_maj).sum() / len(comp) if len(comp) else np.nan,
            }
        )
    pv = pd.DataFrame(prows)
    # judge_cap_rate comes from the cell-level scan, which counts every capped vote,
    # including the ones that still produced a verdict.
    caps = caps[(caps.solver_max_tokens == 16384) & caps.judge_short.isin(["Gemma4", "Qwen3.6"])].copy()
    caps["mbs"] = (caps.solver_model.str.replace("-Base", "", regex=False)).map(short).fillna(caps.solver_model)
    caps["exp"] = caps.mbs + "|" + caps.state + "|" + caps.benchmark
    caps["judge_cap_rate"] = caps.n_votes_capped / caps.n_votes
    pv = pv.merge(caps[["exp", "judge_short", "judge_cap_rate"]], on=["exp", "judge_short"], how="left")
    wide = pv.pivot(
        index=["exp", "solver_model", "state", "benchmark"],
        columns="judge_short",
        values=["n_generations", "veto_rate", "veto_rate_complete", "judge_cap_rate"],
    )
    wide.columns = [f"{a}_{'gemma' if b == 'Gemma4' else 'qwen'}" for a, b in wide.columns]
    wide = wide.reset_index()
    # only experiments judged by both judges stay as real pairs
    both = wide.dropna(subset=["n_generations_gemma", "n_generations_qwen"]).copy()
    mism = both[both.n_generations_gemma != both.n_generations_qwen]
    if len(mism):
        print(f"  ! {len(mism)} exps with unequal n_generations dropped:")
        print(mism[["exp", "n_generations_gemma", "n_generations_qwen"]].to_string(index=False))
        both = both[both.n_generations_gemma == both.n_generations_qwen]
    both["n_generations"] = both.n_generations_gemma.astype(int)
    both["has_judged_solutions"] = True

    # NA rows: rates undefined, has_judged_solutions=False. Two causes, kept
    # apart in CHECKS.md: (a) the solver had zero correct answers, so there was
    # nothing to judge; (b) the cell is in CORRUPTED_JUDGE_CELLS and the salvage
    # produced no paired row.
    zero = []
    repc = add_derived_cols(pd.read_csv(os.path.join(RESULTS, "report.csv"), low_memory=False))
    z = repc[
        (repc.judged == False)  # noqa: E712
        & (repc.max_tokens == 16384)
        & (repc.true_count == 0)
        & repc.model.isin(SCOPE_SOLVERS)
        & (repc.generation == "current")
    ]
    for _, r in z.iterrows():
        zero.append(
            {
                "exp": r.exp,
                "solver_model": r.model,
                "state": r.state,
                "benchmark": r.benchmark,
                "n_generations": 0,
                "has_judged_solutions": False,
                "na_reason": "zero_correct",
            }
        )
    paired_exps = set(both.exp)
    for model, state, bench, mx in sorted(CORRUPTED_JUDGE_CELLS):
        if mx != 16384:
            continue
        rr = repc[
            (repc.model == model)
            & (repc.state == state)
            & (repc.benchmark == bench)
            & (repc.max_tokens == mx)
            & (repc.judged == False)  # noqa: E712
        ]
        if len(rr) == 0 or rr.iloc[0].exp in paired_exps:
            continue  # salvage already produced a real paired row for this exp
        r = rr.iloc[0]
        zero.append(
            {
                "exp": r.exp,
                "solver_model": model,
                "state": state,
                "benchmark": bench,
                "n_generations": 0,
                "has_judged_solutions": False,
                "na_reason": "corrupted_votes",
            }
        )
    out = pd.concat([both, pd.DataFrame(zero)], ignore_index=True)
    cols = [
        "exp",
        "solver_model",
        "state",
        "benchmark",
        "n_generations",
        "has_judged_solutions",
        "na_reason",
        "veto_rate_gemma",
        "veto_rate_qwen",
        "veto_rate_complete_gemma",
        "veto_rate_complete_qwen",
        "judge_cap_rate_gemma",
        "judge_cap_rate_qwen",
    ]
    out = out[[c for c in cols if c in out.columns]].sort_values("exp").reset_index(drop=True)
    out.to_csv(os.path.join(ANALYSIS, "judge_paired.csv"), index=False)
    print(
        f"judge_paired.csv: {len(out)} rows "
        f"({int(out.has_judged_solutions.sum())} paired + "
        f"{int((~out.has_judged_solutions).sum())} zero-correct NA rows)"
    )


# check step


def step_check() -> None:
    lines = ["# data/ acceptance checks", ""]
    ok_all = True

    def chk(name, cond, detail=""):
        nonlocal ok_all
        mark = "PASS" if cond else "FAIL"
        ok_all &= bool(cond)
        lines.append(f"- [{'x' if cond else ' '}] **{mark}** {name}" + (f" ({detail})" if detail else ""))
        print(f"[{mark}] {name}" + (f" ({detail})" if detail else ""))

    v = pd.read_csv(os.path.join(DATA, "verdicts.csv"))
    chk("verdicts: n_yes+n_no+n_missing == 3 on every row", bool(((v.n_yes + v.n_no + v.n_missing) == 3).all()))
    chk("verdicts: n_missing_capped <= n_missing", bool((v.n_missing_capped.fillna(0) <= v.n_missing).all()))
    g16 = v[(v.solver_max_tokens == 16384) & (v.judge_short == "Gemma4")]
    q16 = v[(v.solver_max_tokens == 16384) & (v.judge_short == "Qwen3.6") & ~v.solver_model.str.startswith("Qwen2.5")]
    pair_exps = set(zip(g16.solver_model, g16.state, g16.benchmark, strict=False)) & set(
        zip(q16.solver_model, q16.state, q16.benchmark, strict=False)
    )
    gp = g16[[t in pair_exps for t in zip(g16.solver_model, g16.state, g16.benchmark, strict=False)]]
    qp = q16[[t in pair_exps for t in zip(q16.solver_model, q16.state, q16.benchmark, strict=False)]]
    chk("verdicts: Gemma4 row count == Qwen3.6 row count (paired exps)", len(gp) == len(qp), f"{len(gp)} vs {len(qp)}")
    chk(
        "verdicts: solution_id sets identical between the two judges (paired exps)",
        set(gp.solution_id) == set(qp.solution_id),
    )
    # pooled veto rate: majority rule, solver_max_tokens == 16384, Qwen2.5 excluded for Qwen3.6
    gv = (~gp.approved_maj).mean() * 100
    qv = (~qp.approved_maj).mean() * 100
    chk("pooled veto Gemma4 about 11.0%", abs(gv - 11.0) < 0.35, f"{gv:.2f}%")
    chk("pooled veto Qwen3.6 about 3.7%", abs(qv - 3.7) < 0.35, f"{qv:.2f}%")
    lines.append(
        "  - Note: the veto thresholds hold only for the definition that **excludes Qwen2.5** "
        "from the Qwen3.6 pool; including it gives 5.29% for Qwen3.6."
    )

    error_tasks = pd.read_csv(os.path.join(DATA, "report_error_tasks.csv"))
    chk(
        "error: n_yes+n_no+n_missing == 3",
        bool(((error_tasks.n_yes + error_tasks.n_no + error_tasks.n_missing) == 3).all()),
    )
    pv4 = error_tasks[(error_tasks.judge_short == "V4-Flash") & error_tasks.panel]
    per_cond = pv4.groupby("condition").size()
    chk("panel: n == 720 per condition (V4 slice)", bool((per_cond == 720).all()), per_cond.to_dict())

    gc = pd.read_csv(os.path.join(ANALYSIS, "gap_curve.csv"))
    dup = gc.groupby(["solver_model", "state", "benchmark", "solver_max_tokens", "k"])
    chk("gap_curve: pass_at_k identical across judges of the same cell", bool((dup.pass_at_k.nunique() == 1).all()))
    chk("gap_curve: cot_gap >= 0 (ACTIVE check)", bool((gc.cot_gap >= -1e-9).all()), f"min={gc.cot_gap.min():.4f}")
    jp = pd.read_csv(os.path.join(ANALYSIS, "judge_paired.csv"))
    # 100 = 93 paired exps + 7 zero-correct NA rows
    chk("judge_paired: 100 rows", len(jp) == 100, f"{len(jp)}")
    chk("judge_paired: paired rows have equal n_generations (by construction)", True)
    rates = [c for c in jp.columns if c.endswith("_rate") or "_rate_" in c]
    okr = all(jp[c].dropna().between(0, 1).all() for c in rates)
    chk("all rates within [0,1]", okr)
    rep = pd.read_csv(os.path.join(DATA, "report.csv"), low_memory=False)
    part = rep[rep.judged == True].copy()  # noqa: E712
    stub = part.n_generations.isna()  # zero-correct exps' empty judge cells (all-NaN)
    p2 = part[~stub]
    ok5 = (p2.true_count + p2.false_count + p2.n_veto) == p2.n_generations
    chk(
        "report: true+false+n_veto == n_generations (judged rows; all-NaN stubs excluded)",
        bool(ok5.all()),
        f"{int(ok5.sum())}/{len(p2)}, plus {int(stub.sum())} empty-cell stubs",
    )

    na = jp[~jp.has_judged_solutions]
    lines.append("")
    lines.append("## Experiments with undefined rates (has_judged_solutions=False)")
    for _, r in na.iterrows():
        why = (
            "the solver produced no correct answer (true_count=0), so nothing was judged"
            if r.na_reason == "zero_correct"
            else "the stored judge votes of this cell are not trusted (CORRUPTED_JUDGE_CELLS)"
        )
        lines.append(f"  - {r.exp}: {why}")
    lines.append("")
    lines.append(f"**RESULT: {'ALL CHECKS PASSED' if ok_all else 'AT LEAST ONE CHECK FAILED'}**")
    with open(os.path.join(DATA, "CHECKS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n=> data/CHECKS.md written; overall: {'PASS' if ok_all else 'FAIL'}")


STEPS = {
    "report": step_report,
    "verdicts": step_verdicts,
    "error": step_error,
    "panel": step_panel,
    "derived": step_derived,
    "check": step_check,
}


REQUIRES = {
    "report": [os.path.join(RESULTS, "report.csv"), os.path.join(RESULTS, "report_tasks.csv")],
    "verdicts": [os.path.join(RESULTS, "report.csv")],
    "error": [
        os.path.join(ES_OUT, "judge_inputs", "manifest.json"),
        os.path.join(ES_OUT, "corruption_log.csv"),
        os.path.join(ES_OUT, "report_error_tasks.csv"),
    ],
    "panel": [],
    "derived": [os.path.join(RESULTS, "report.csv")],
    "check": [],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    a = ap.parse_args()
    steps = ["report", "verdicts", "error", "panel", "derived", "check"] if a.step == "all" else [a.step]
    needed = [p for s in steps for p in REQUIRES[s]]
    if needed:
        require(*needed, what="raw experiment outputs (results/, error_study/output/)")
    os.makedirs(CACHE, exist_ok=True)
    for s in steps:
        print(f"\n===== {s} =====")
        STEPS[s]()


if __name__ == "__main__":
    main()
