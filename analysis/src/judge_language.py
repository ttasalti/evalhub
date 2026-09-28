"""Language of the judges' own output (thinking plus visible answer); feeds the judge-language
summary quoted in Section 5 (tables/judge_language.txt) and tab_language.py.

Panel judgments are read from the consolidated caches cache/judge_{qwen35b,r1_8b}_consolidated.parquet and
judge-matrix judgments from results/<solver>/judged_by/<tag>/<leaf>/cot_judge*_raw.jsonl; both live outside
the repository, as does the GlotLID v3 model. The script writes judge_language.csv (one row per judgment) to
analysis/data and judge_language.txt (the summary) to analysis/tables.

Segment layer only: GlotLID on the mathematics-stripped text, with the same cleaning, segmentation,
label folding and class rule as code_switching.py.
  (a) error-injection panel: columns judge_think (thinking) and judge_reasoning (visible answer) of the
      consolidated caches; SAMPLE_PANEL judgments per judge x benchmark x {clean, boxed_only}, seed 0.
  (b) judge matrix: the judgment text of the four judges over the ten solvers (thinking included;
      V4-Flash returns its thinking in reasoning_content, which is prepended to the visible answer);
      SAMPLE_MAIN judgments per judge x benchmark (PT, TR, TUBITAK; EN as control), seed 0.
"""

from __future__ import annotations

import glob
import json
import os
import re
from multiprocessing import Pool

import code_switching as cs
import numpy as np
import pandas as pd
from common import CACHE_DIR, DATA_DIR, GLOTLID_MODEL, RESULTS_ROOT, TAB_DIR, ensure_output_dirs, require

SAMPLE_PANEL = 300
SAMPLE_MAIN = 300
JUDGE = {
    "Qwen3.6-35B-A3B": "Qwen3.6",
    "gemma-4-26B-A4B-it": "Gemma4",
    "deepseek-v4-flash": "V4-Flash",
    "DeepSeek-R1-0528-Qwen3-8B": "R1-distill",
}
TAG = re.compile(r"^(?P<judge>.+?)__state-(?P<state>think|non-think)__t0\.6__max(?P<jmax>\d+)__basemax(?P<bmax>\d+)")
RAWNAMES = ["cot_judge_raw.jsonl", "cot_judge_tr_raw.jsonl", "cot_judge_pt_raw.jsonl"]
BENCH = ["aime2026", "aime2026_pt", "aime2026_tr", "tubitak_math2026"]
PANEL_CACHES = [("qwen35b", "Qwen3.6"), ("r1_8b", "R1-distill")]
rng = np.random.default_rng(0)


def work(args):
    meta, text, lang = args
    r = cs.lid_chain(text, lang, "", do_mask=False)
    r.update(meta)
    return r


def verdict_of(text):
    if "boxed{yes}" in text:
        return "yes"
    if "boxed{no}" in text:
        return "no"
    return "none"


def panel_jobs():
    jobs = []
    for tag, js in PANEL_CACHES:
        c = pd.read_parquet(
            CACHE_DIR / f"judge_{tag}_consolidated.parquet",
            columns=["row_id", "benchmark", "error_type", "judge_think", "judge_reasoning", "judge_verdict"],
        )
        c = c[c.error_type.isin(["clean", "boxed_only"])]
        for (bench, et), g in c.groupby(["benchmark", "error_type"]):
            take = g.iloc[rng.choice(len(g), min(SAMPLE_PANEL, len(g)), replace=False)]
            for r in take.itertuples():
                think = str(r.judge_think) if pd.notna(r.judge_think) else ""
                answer = str(r.judge_reasoning) if pd.notna(r.judge_reasoning) else ""
                meta = {
                    "source": "panel",
                    "judge": js,
                    "benchmark": bench,
                    "condition": et,
                    "verdict": str(r.judge_verdict),
                    "solution_id": r.row_id,
                }
                jobs.append((meta, think + "\n" + answer, cs.LANG[bench]))
    return jobs


def main_jobs():
    pool = {}
    for rel, _sm, _state, label in cs.CELLS:
        for bench in BENCH:
            leafname = f"{bench}__t0.6__max16384__n64"
            for tagdir in sorted(glob.glob(str(RESULTS_ROOT / rel / "judged_by" / "*"))):
                m = TAG.match(os.path.basename(tagdir))
                if not m or m["judge"] not in JUDGE or m["state"] != "think" or m["bmax"] != "16384":
                    continue
                js = JUDGE[m["judge"]]
                if int(m["jmax"]) != (20480 if js == "V4-Flash" else 16384):
                    continue
                for n in RAWNAMES:
                    f = os.path.join(tagdir, leafname, n)
                    if os.path.exists(f):
                        pool.setdefault((js, bench), []).append((f, label))
    jobs = []
    for (js, bench), files in pool.items():
        recs = []
        for f, label in files:
            with open(f) as fh:
                for line in fh:
                    d = json.loads(line)
                    r = d["response"]
                    msg = r["choices"][0]["message"]
                    # V4-Flash keeps its thinking in reasoning_content; the other judges inline it in content
                    text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
                    recs.append((label, d["task_id"], text, r["choices"][0]["finish_reason"]))
        take = [recs[i] for i in rng.choice(len(recs), min(SAMPLE_MAIN, len(recs)), replace=False)]
        for label, gid, text, fr in take:
            meta = {
                "source": "main",
                "judge": js,
                "benchmark": bench,
                "condition": "correct",
                "verdict": verdict_of(text),
                "solver": label,
                "solution_id": gid,
                "finish_reason": fr,
            }
            jobs.append((meta, text, cs.LANG[bench]))
    return jobs


def summary(df):
    df = df[df.n_letters.fillna(0) >= 20].copy()
    df["cls"] = np.where(df.share_target >= 0.8, "target", np.where(df.share_eng >= 0.8, "english", "mixed"))
    out = ["=== judge output language, class shares (%) and mean English share ==="]
    for (src, js, bench, cond), g in df.groupby(["source", "judge", "benchmark", "condition"]):
        sh = g.cls.value_counts(normalize=True) * 100
        out.append(
            f"{src:5s} {js:10s} {bench:18s} {cond:10s} n={len(g):4d} target={sh.get('target', 0):5.1f} "
            f"mixed={sh.get('mixed', 0):5.1f} english={sh.get('english', 0):5.1f} "
            f"mean_eng={100 * g.share_eng.mean():5.1f} mean_target={100 * g.share_target.mean():5.1f}"
        )
    out.append("\n=== verdict by judge-output class (main corpus, correct solutions; panel clean/boxed) ===")
    for (src, js, bench, cond), g in df.groupby(["source", "judge", "benchmark", "condition"]):
        line = f"{src:5s} {js:10s} {bench:18s} {cond:10s}"
        for cls in ["target", "mixed", "english"]:
            s = g[g.cls == cls]
            yes_rate = 100 * (s.verdict == "yes").mean() if len(s) else float("nan")
            line += f"  {cls}: yes {yes_rate:5.1f}% n={len(s):3d}"
        out.append(line)
    return "\n".join(out)


def main():
    require(RESULTS_ROOT, what="the raw experiment outputs (results/)")
    require(
        *[CACHE_DIR / f"judge_{tag}_consolidated.parquet" for tag, _ in PANEL_CACHES],
        what="the consolidated error-injection judge caches",
    )
    require(GLOTLID_MODEL, what="the GlotLID v3 model (set GLOTLID_MODEL)")
    ensure_output_dirs()
    jobs = panel_jobs()
    print("panel judgments sampled:", len(jobs), flush=True)
    main_part = main_jobs()
    print("main judgments sampled:", len(main_part), flush=True)
    jobs += main_part

    cs.load_models()
    with Pool(int(os.environ.get("WORKERS", "8"))) as p:
        rows = p.map(work, jobs, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "judge_language.csv", index=False)
    txt = summary(df)
    print(txt)
    with open(TAB_DIR / "judge_language.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
