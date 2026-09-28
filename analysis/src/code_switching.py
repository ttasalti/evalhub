"""Language identification of the reasoning chains (code switching); feeds Table "code_switching"
(code_switching_tables.py), Figure fig_code_switching.png and the language analyses of Section 5.

The chains are read from the raw solver outputs under results/, that is
results/<solver>/<benchmark>__t0.6__max*__n*/{<benchmark>_results.jsonl,<benchmark>_raw.jsonl}, and for the
panel corpus from error_study/output/base_sample.parquet; neither is part of the repository, and the GlotLID v3
model (GLOTLID_MODEL) is needed as well. From the repository it uses verdicts.csv and
error_injection_panel_3judges.csv in analysis/data, the benchmark question files under evalhub/benchmarks/math/
and vendor/flores200_glotlid_labels.txt. By default it writes code_switching_main.csv and
code_switching_panel.csv to analysis/data, one row per chain; with --corpus it writes
code_switching_ptexams.csv or code_switching_budget.csv instead.

Corpora
  main     the ten solvers of the judge matrix x four 64-sample benchmarks, 16k budget, every
           answer-correct generation, joined to the majority verdicts of Qwen3.6 and Gemma4
           (verdicts.csv, judge budget 16k, thinking).
  panel    the 720 original solutions of the error-injection study (base_sample.parquet), joined to
           the clean-control and final-answer verdicts of the three judges
           (error_injection_panel_3judges.csv).
  ptexams  the Portuguese exams (n=16, 16k) for the solvers that have them; segment layer only.
  budget   the thinking runs of Section 5.3 (16k for four solvers; 32k, 65k on English AIME, for
           Qwen3.5-4B/9B); segment layer only.

Language identification, two layers, both on the cleaned chain (see clean()):
  segment layer  GlotLID v3 (Kargaran et al. 2023), full label set, one label per segment (line or
                 sentence with >= MIN_LETTERS letters); the chain's share of letters in the benchmark
                 language, in English and in any other label.
  chain layer    MaskLID (Kargaran et al. 2024) on the first MASKLID_WORDS words of the cleaned chain,
                 FLORES-200 label set, README parameters; the set of labels it returns.
Cleaning: display and inline math, \\boxed{}, LaTeX commands, code blocks, markdown marks, digits and
operator symbols are removed, so numbers and formulas never receive a language. On the Turkish and
Portuguese benchmarks a segment that restates the question (>= QUESTION_OVERLAP of its words of
>= 4 letters occur in the question) is dropped, so quoting the problem in its own language does not
count as reasoning in that language.
Chain class (letter shares): benchmark language >= 0.8 is "target"; English >= 0.8 is "english";
otherwise "mixed". The sensitivity at 0.7 and 0.9 is reported by code_switching_tables.py.

Usage: python code_switching.py [--workers 8] [--limit N] [--corpus main|panel|both|ptexams|budget]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import pandas as pd
from common import ANALYSIS_ROOT, DATA_DIR, ERROR_STUDY_OUTPUT, GLOTLID_MODEL, REPO_ROOT, RESULTS_ROOT, data, require

VENDOR_DIR = ANALYSIS_ROOT / "src" / "vendor"
BENCHMARK_FILES = REPO_ROOT / "evalhub" / "benchmarks" / "math"
BASE_SAMPLE = ERROR_STUDY_OUTPUT / "base_sample.parquet"

MIN_LETTERS = 20
MASKLID_WORDS = 1000
QUESTION_OVERLAP = 0.7
# MaskLID (the slow chain layer, about 2 ms per word in fastText's Python binding) runs on every
# earlier-generation chain and on a seeded 20% sample of the current generation's chains in the main
# corpus, and on every panel chain; the segment layer runs on every chain.
MASKLID_SHARE = 0.2
MASK_RNG = np.random.default_rng(0)
# GlotLID near-variant labels folded into the language they are confused with on mathematical prose
# (Middle English / Scots to English, Galician to Portuguese, Latin-script Turkic neighbours to
# Turkish); reported in the appendix.
FOLD = {"enm": "eng", "sco": "eng", "glg": "por", "azj": "tur", "crh": "tur"}
# GlotLID's undetermined / no-linguistic-content labels (undelimited maths) leave the denominator.
NONLING = {"und", "zxx"}
LANG = {
    "aime2026": "eng",
    "aime2026_pt": "por",
    "aime2026_tr": "tur",
    "tubitak_math2026": "tur",
    "pt_exams_math": "por",
}

# (results dir relative to results/, solver_model in verdicts.csv, state, label)
CELLS = [
    ("Qwen2.5/base/Qwen2.5-7B", "Qwen2.5-7B", "base", "Q2.5-7B"),
    ("Qwen2.5/base/Qwen2.5-32B", "Qwen2.5-32B", "base", "Q2.5-32B"),
    ("Qwen2.5/non-think/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "non-think", "Q2.5-7B-Inst"),
    ("Qwen2.5/non-think/Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct", "non-think", "Q2.5-32B-Inst"),
    ("base/Qwen3.5-4B-Base", "Qwen3.5-4B-Base", "base", "Q3.5-4B-Base"),
    ("base/Qwen3.5-9B-Base", "Qwen3.5-9B-Base", "base", "Q3.5-9B-Base"),
    ("non-think/Qwen3.5-4B", "Qwen3.5-4B", "non-think", "Q3.5-4B"),
    ("non-think/Qwen3.5-9B", "Qwen3.5-9B", "non-think", "Q3.5-9B"),
    ("non-think/gemma-4-E2B-it", "gemma-4-E2B-it", "non-think", "G4-E2B"),
    ("non-think/gemma-4-E4B-it", "gemma-4-E4B-it", "non-think", "G4-E4B"),
]
GEN = {"Q2.5-7B": "earlier", "Q2.5-32B": "earlier", "Q2.5-7B-Inst": "earlier", "Q2.5-32B-Inst": "earlier"}
BENCH4 = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]

# The Portuguese exams have no non-thinking runs of the current generation.
PTEX_CELLS = [
    ("Qwen2.5/base/Qwen2.5-7B", "Qwen2.5-7B", "base", "Q2.5-7B"),
    ("Qwen2.5/base/Qwen2.5-32B", "Qwen2.5-32B", "base", "Q2.5-32B"),
    ("Qwen2.5/non-think/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", "non-think", "Q2.5-7B-Inst"),
    ("Qwen2.5/non-think/Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct", "non-think", "Q2.5-32B-Inst"),
    ("base/Qwen3.5-4B-Base", "Qwen3.5-4B-Base", "base", "Q3.5-4B-Base"),
    ("base/Qwen3.5-9B-Base", "Qwen3.5-9B-Base", "base", "Q3.5-9B-Base"),
    ("think/Qwen3.5-4B", "Qwen3.5-4B", "think", "Q3.5-4B-Think"),
    ("think/Qwen3.5-9B", "Qwen3.5-9B", "think", "Q3.5-9B-Think"),
    ("think/gemma-4-E2B-it", "gemma-4-E2B-it", "think", "G4-E2B-Think"),
    ("think/gemma-4-E4B-it", "gemma-4-E4B-it", "think", "G4-E4B-Think"),
]

# Thinking runs at 16k (four solvers, four benchmarks) plus the budget ladder (Qwen3.5 4B/9B).
BUDGET_CELLS = [
    ("think/Qwen3.5-4B", "Qwen3.5-4B", "Q3.5-4B-Think"),
    ("think/Qwen3.5-9B", "Qwen3.5-9B", "Q3.5-9B-Think"),
    ("think/gemma-4-E2B-it", "gemma-4-E2B-it", "G4-E2B-Think"),
    ("think/gemma-4-E4B-it", "gemma-4-E4B-it", "G4-E4B-Think"),
]

# ----------------------------------------------------------------- cleaning
RX = [
    (re.compile(r"```.*?```", re.S), " "),
    (re.compile(r"\$\$.*?\$\$", re.S), " "),
    (re.compile(r"\\\[.*?\\\]", re.S), " "),
    (re.compile(r"\\\(.*?\\\)", re.S), " "),
    (re.compile(r"\$[^$\n]{0,400}\$"), " "),
    (re.compile(r"\\boxed\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"), " "),
    (re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})*"), " "),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"[*#`_|~>]+"), " "),
    (re.compile(r"[0-9]+(?:[.,][0-9]+)*"), " "),
    (re.compile(r"[+\-−–—×÷·=<>≤≥≠≈^/\\(){}\[\]%&]+"), " "),
]
SENT = re.compile(r"(?<=[.!?;:])\s+")


def flores_labels() -> list[str]:
    with open(VENDOR_DIR / "flores200_glotlid_labels.txt") as f:
        return [line.strip() for line in f if line.strip()]


def clean(text):
    for rx, rep in RX:
        text = rx.sub(rep, text)
    return text


def letters(s):
    return sum(1 for ch in s if unicodedata.category(ch).startswith("L"))


def words(s):
    return [w for w in re.findall(r"[^\W\d_]+", s.lower()) if len(w) >= 4]


def segments(text):
    out = []
    for line in clean(text).split("\n"):
        for seg in SENT.split(line.strip()):
            seg = re.sub(r"\s+", " ", seg).strip()
            if letters(seg) >= MIN_LETTERS:
                out.append(seg)
    return out


# ---------------------------------------------------------------- models (loaded once, inherited by the forked workers)
_glot = None
_mask = None


def load_models():
    global _glot, _mask
    try:
        from vendor.masklid import MaskLID
    except ImportError:
        raise SystemExit(
            "code_switching.py needs the fasttext package (pip install fasttext-wheel) and the GlotLID model; "
            "see analysis/README.md"
        ) from None

    # One fastText model: MaskLID restricts it to the FLORES-200 labels for the chain layer, while
    # predict() on the underlying model keeps the full label set for the segment layer.
    _mask = MaskLID(str(GLOTLID_MODEL), languages=flores_labels())
    _glot = _mask.model


def _code(label):
    code = label.replace("__label__", "").split("_")[0]
    return FOLD.get(code, code)


def lid_chain(text, lang, question, do_mask=True):
    segs = segments(text)
    qw = set(words(question)) if question else set()
    if qw:
        keep = []
        for s in segs:
            w = words(s)
            if w and sum(1 for x in w if x in qw) / len(w) >= QUESTION_OVERLAP:
                continue
            keep.append(s)
        n_q = len(segs) - len(keep)
        segs = keep
    else:
        n_q = 0
    tot = {"target": 0, "eng": 0, "other": 0}
    n_low = 0
    labels = {}
    n_nonling = 0
    for s in segs:
        lab, prob = _glot.predict(s.replace("\n", " "), k=1)
        code = _code(lab[0])
        p = float(prob[0])
        seg_letters = letters(s)
        if code in NONLING:
            n_nonling += seg_letters
            continue
        key = "eng" if code == "eng" else ("target" if code == lang else "other")
        if lang == "eng" and code == "eng":
            key = "target"
        tot[key] += seg_letters
        labels[code] = labels.get(code, 0) + seg_letters
        if p < 0.5:
            n_low += 1
    n = sum(tot.values())
    top_other = max(((k, v) for k, v in labels.items() if k not in ("eng", lang)), key=lambda x: x[1], default=("", 0))
    # chain layer
    ctext = " ".join(clean(text).split()[:MASKLID_WORDS])
    try:
        if do_mask and letters(ctext) >= MIN_LETTERS:
            m = _mask.predict_codeswitch(
                ctext,
                beta=20,
                alpha=3,
                max_lambda=3,
                min_length=10,
                min_prob=0.90,
                max_retry=3,
                alpha_step_increase=3,
                beta_step_increase=5,
            )
        else:
            m = {}
    except Exception:
        m = {}
    mlabs = {} if do_mask else None
    for k, v in m.items() if do_mask else []:
        c = _code(k)
        mlabs[c] = mlabs.get(c, 0) + letters(v)
    return {
        "n_segments": len(segs),
        "n_question_segments": n_q,
        "n_letters": n,
        "n_letters_nonling": n_nonling,
        "share_target": tot["target"] / n if n else np.nan,
        "share_eng": tot["eng"] / n if n else np.nan,
        "share_other": tot["other"] / n if n else np.nan,
        "other_top": top_other[0],
        "other_top_letters": top_other[1],
        "share_lowconf": n_low / len(segs) if segs else np.nan,
        "masklid_labels": "+".join(sorted(mlabs, key=lambda k: -mlabs[k])) if mlabs is not None else np.nan,
        "masklid_n": len(mlabs) if mlabs is not None else np.nan,
        "masklid_letters": json.dumps(mlabs) if mlabs is not None else np.nan,
    }


def work(args):
    meta, text, lang, question = args
    r = lid_chain(text, lang, question, do_mask=bool(meta.get("masklid_run", True)))
    r.update(meta)
    return r


# ----------------------------------------------------------------- inputs
def read_leaf(leaf, bench):
    """Per-task results record and the list of raw completions (choices[0]) of one results leaf."""
    with open(leaf / f"{bench}_results.jsonl") as f:
        res = {}
        for line in f:
            rec = json.loads(line)
            res[rec["task_id"]] = rec
    raw = {}
    with open(leaf / f"{bench}_raw.jsonl") as f:
        for line in f:
            d = json.loads(line)
            raw.setdefault(d["task_id"], []).append(d["response"]["choices"][0])
    return res, raw


def questions():
    """Question texts keyed by (benchmark, index); task ids differ in prefix across benchmarks
    (AIME2026-PT/1, aime2026_tr/1, tubitak_math2026/7), so the numeric index is the join key."""
    q = {}
    tr = pd.read_parquet(BENCHMARK_FILES / "aime2026_tr" / "aime2026_tr.parquet")
    for _, r in tr.iterrows():
        q[("aime2026_tr", int(r.problem_idx))] = r.problem
    pt = pd.read_parquet(BENCHMARK_FILES / "aime2026_pt" / "aime2026_pt.parquet")
    for _, r in pt.iterrows():
        q[("aime2026_pt", int(r.problem_idx))] = r.problem
    tb = pd.read_csv(BENCHMARK_FILES / "tubitak_math2026" / "tubitak_math2026.csv")
    for _, r in tb.iterrows():
        q[("tubitak_math2026", int(r.Question_Number))] = r.Question_Text
    return q


def main_corpus(limit):
    v = pd.read_csv(data("verdicts.csv"))
    v = v[
        (v.solver_max_tokens == 16384)
        & (v.judge_state == "think")
        & (v.judge_max_tokens == 16384)
        & (v.judge_short.isin(["Qwen3.6", "Gemma4"]))
    ]
    ver = {
        (r.solution_id, r.judge_short): (bool(r.approved_maj), int(r.n_yes), int(r.n_no), int(r.n_missing))
        for r in v.itertuples()
    }
    question_by_task = questions()
    jobs = []
    for rel, sm, state, label in CELLS:
        for bench in BENCH4:
            leaf = RESULTS_ROOT / rel / f"{bench}__t0.6__max16384__n64"
            res, raw = read_leaf(leaf, bench)
            for tid, r in res.items():
                assert len(raw[tid]) == len(r["correct"]), (leaf, tid, len(raw[tid]), len(r["correct"]))
                for i, ok in enumerate(r["correct"]):
                    if not ok:
                        continue
                    sid = f"{sm}|{state}|{bench}|16384|{tid}|g{i}"
                    meta = {
                        "corpus": "main",
                        "solver": label,
                        "generation": GEN.get(label, "current"),
                        "state": state,
                        "benchmark": bench,
                        "task_id": tid,
                        "gen": i,
                        "solution_id": sid,
                        "finish_reason": raw[tid][i]["finish_reason"],
                    }
                    for j in ["Qwen3.6", "Gemma4"]:
                        a = ver.get((sid, j))
                        meta[f"approved_{j}"] = a[0] if a else np.nan
                        meta[f"nyes_{j}"] = a[1] if a else np.nan
                    meta["masklid_run"] = (GEN.get(label, "current") == "earlier") or (
                        MASK_RNG.random() < MASKLID_SHARE
                    )
                    q = question_by_task.get((bench, int(tid.rsplit("/", 1)[-1])), "") if LANG[bench] != "eng" else ""
                    if LANG[bench] != "eng":
                        assert q, tid
                    jobs.append((meta, raw[tid][i]["message"]["content"], LANG[bench], q))
            if limit and len(jobs) >= limit:
                return jobs[:limit]
    return jobs


def ptexams_corpus(limit):
    """Portuguese exams (n=16, 16k) for the solvers that have them; segment layer only (masklid_run False)."""
    v = pd.read_csv(data("verdicts.csv"))
    v = v[
        (v.benchmark == "pt_exams_math")
        & (v.solver_max_tokens == 16384)
        & (v.judge_state == "think")
        & (v.judge_max_tokens == 16384)
        & (v.judge_short.isin(["Qwen3.6", "Gemma4"]))
    ]
    ver = {(r.solution_id, r.judge_short): (bool(r.approved_maj), int(r.n_yes)) for r in v.itertuples()}
    bench = "pt_exams_math"
    jobs = []
    b = pd.read_parquet(BASE_SAMPLE)
    question_by_task = b[b.benchmark == bench].groupby("task_id").question_text.first().to_dict()
    pt = pd.read_csv(BENCHMARK_FILES / "pt_exams_math" / "pt_exams_math.csv")
    for rel, sm, state, label in PTEX_CELLS:
        leaf = RESULTS_ROOT / rel / f"{bench}__t0.6__max16384__n16"
        if not leaf.is_dir():
            print("no dir", leaf, flush=True)
            continue
        res, raw = read_leaf(leaf, bench)
        for tid, r in res.items():
            if len(raw.get(tid, [])) != len(r["correct"]):
                continue
            for i, ok in enumerate(r["correct"]):
                if not ok:
                    continue
                sid = f"{sm}|{state}|{bench}|16384|{tid}|g{i}"
                if (sid, "Qwen3.6") not in ver and (sid, "Gemma4") not in ver:
                    continue
                meta = {
                    "corpus": "ptexams",
                    "solver": label,
                    "generation": GEN.get(label.replace("-Think", ""), "current"),
                    "state": state,
                    "benchmark": bench,
                    "task_id": tid,
                    "gen": i,
                    "solution_id": sid,
                    "finish_reason": raw[tid][i]["finish_reason"],
                    "masklid_run": False,
                }
                for j in ["Qwen3.6", "Gemma4"]:
                    a = ver.get((sid, j))
                    meta[f"approved_{j}"] = a[0] if a else np.nan
                    meta[f"nyes_{j}"] = a[1] if a else np.nan
                # PT-exams task ids are 0-based row indices of the benchmark csv; the panel's
                # question_text is the fallback when the index does not resolve.
                idx = int(tid.rsplit("/", 1)[-1])
                q = str(pt.iloc[idx].question) if 0 <= idx < len(pt) else ""
                if not q:
                    q = question_by_task.get(tid, "")
                jobs.append((meta, raw[tid][i]["message"]["content"], "por", q))
    return jobs[:limit] if limit else jobs


def budget_corpus(limit):
    """Thinking runs of Section 5.3: 16k for four solvers, and 32k (65k on English AIME) for Qwen3.5-4B/9B.
    Segment layer only. Verdicts: Qwen3.6 (judge 16k at solver 16k; judge 32k at the raised budgets),
    Gemma4 at 16k."""
    v = pd.read_csv(data("verdicts.csv"))
    v = v[(v.state == "think") & (v.judge_state == "think") & (v.judge_short.isin(["Qwen3.6", "Gemma4"]))]
    ver = {(r.solution_id, r.judge_short, int(r.judge_max_tokens)): bool(r.approved_maj) for r in v.itertuples()}
    question_by_task = questions()
    jobs = []
    for rel, sm, label in BUDGET_CELLS:
        for bench in BENCH4:
            budgets = [16384] + ([65536 if bench == "aime2026" else 32768] if label.startswith("Q3.5") else [])
            for bmax in budgets:
                leaf = RESULTS_ROOT / rel / f"{bench}__t0.6__max{bmax}__n64"
                if not leaf.is_dir():
                    print("no dir", leaf, flush=True)
                    continue
                res, raw = read_leaf(leaf, bench)
                jb = 16384 if bmax == 16384 else 32768
                for tid, r in res.items():
                    if len(raw.get(tid, [])) != len(r["correct"]):
                        continue
                    for i, ok in enumerate(r["correct"]):
                        if not ok:
                            continue
                        sid = f"{sm}|think|{bench}|{bmax}|{tid}|g{i}"
                        meta = {
                            "corpus": "budget",
                            "solver": label,
                            "generation": "current",
                            "state": "think",
                            "budget": bmax,
                            "benchmark": bench,
                            "task_id": tid,
                            "gen": i,
                            "solution_id": sid,
                            "finish_reason": raw[tid][i]["finish_reason"],
                            "masklid_run": False,
                        }
                        meta["approved_Qwen3.6"] = ver.get((sid, "Qwen3.6", jb), np.nan)
                        meta["approved_Gemma4"] = ver.get((sid, "Gemma4", 16384), np.nan) if bmax == 16384 else np.nan
                        q = (
                            question_by_task.get((bench, int(tid.rsplit("/", 1)[-1])), "")
                            if LANG[bench] != "eng"
                            else ""
                        )
                        jobs.append((meta, raw[tid][i]["message"]["content"], LANG[bench], q))
    return jobs[:limit] if limit else jobs


def panel_corpus(limit):
    b = pd.read_parquet(BASE_SAMPLE)
    pan = pd.read_csv(data("error_injection_panel_3judges.csv"))
    ids = set(pan.solution_id)
    b = b[b.row_id.isin(ids)]
    assert len(b) == 720
    acc = pan.pivot_table(index="solution_id", columns=["judge_short", "condition"], values="correct_maj")
    jobs = []
    for r in b.itertuples():
        meta = {
            "corpus": "panel",
            "solver": r.model,
            "generation": "current",
            "state": "think",
            "benchmark": r.benchmark,
            "task_id": r.task_id,
            "gen": int(r.source_gen_ordinal),
            "solution_id": r.row_id,
            "finish_reason": "stop",
        }
        for j in ["V4-Flash", "Qwen3.6", "R1-distill"]:
            meta[f"clean_{j}"] = int(acc.loc[r.row_id, (j, "clean")])
            meta[f"final_{j}"] = int(acc.loc[r.row_id, (j, "boxed_only")])
        q = r.question_text if LANG[r.benchmark] != "eng" else ""
        jobs.append((meta, r.solution_text, LANG[r.benchmark], q))
    return jobs[:limit] if limit else jobs


CORPORA = [("main", main_corpus), ("panel", panel_corpus), ("ptexams", ptexams_corpus), ("budget", budget_corpus)]


def run_corpus(name, fn, workers, limit):
    jobs = fn(limit)
    print(name, len(jobs), "chains", flush=True)
    out = DATA_DIR / f"code_switching_{name}{'_test' if limit else ''}.csv"
    part = out.with_name(out.name + ".part")
    rows = []
    if part.exists():
        # resume: keep the finished chains of an interrupted run and skip them
        prev = pd.read_csv(part)
        done = set(prev.solution_id)
        rows = prev.to_dict("records")
        jobs = [j for j in jobs if j[0]["solution_id"] not in done]
        print(name, "resume:", len(done), "done,", len(jobs), "left", flush=True)
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(work, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 2000 == 0:
                # checkpoint so an interrupted run keeps its work
                pd.DataFrame(rows).to_csv(part, index=False)
                print(name, i, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("wrote", out, df.shape, flush=True)
    if part.exists():
        os.remove(part)
    print(df.groupby("benchmark")[["share_target", "share_eng", "share_other"]].mean().round(3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corpus", default="both", help="main | panel | both (main and panel) | ptexams | budget")
    a = ap.parse_args()
    require(RESULTS_ROOT, what="the raw experiment outputs (results/)")
    if a.corpus in ("both", "panel", "ptexams"):
        require(BASE_SAMPLE, what="the error-injection study output (error_study/output/)")
    require(GLOTLID_MODEL, what="the GlotLID v3 model (set GLOTLID_MODEL)")
    load_models()
    for name, fn in CORPORA:
        if a.corpus not in ("both", name):
            continue
        run_corpus(name, fn, a.workers, a.limit)


if __name__ == "__main__":
    main()
