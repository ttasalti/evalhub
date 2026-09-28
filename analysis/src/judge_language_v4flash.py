"""Language of the V4-Flash judge's output on the error-injection study (all five conditions, every
judgment); feeds the judge-language summary of Section 5 (tables/judge_language_v4flash.txt).

The raw judgments are read from error_study/output/judge/<model>/<benchmark>/<condition>/cot_judge*_raw.jsonl,
which is not part of the repository, and the GlotLID v3 model has to be available; the script stops with a
message when either is missing. It writes judge_language_v4flash.csv (one row per judgment) to analysis/data
and judge_language_v4flash.txt to analysis/tables.

Segment layer only (GlotLID, same cleaning and class rule as code_switching.py). The judged text is
the thinking (reasoning_content) followed by the visible answer.
"""

from __future__ import annotations

import glob
import json
import os
from multiprocessing import Pool

import code_switching as cs
import numpy as np
import pandas as pd
from common import DATA_DIR, ERROR_STUDY_OUTPUT, GLOTLID_MODEL, TAB_DIR, ensure_output_dirs, require

JUDGE_ROOT = ERROR_STUDY_OUTPUT / "judge"


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


def jobs_from_raw():
    jobs = []
    for f in sorted(glob.glob(str(JUDGE_ROOT / "*" / "*" / "*" / "cot_judge*_raw.jsonl"))):
        parts = f.split(os.sep)
        model, bench, cond = parts[-4], parts[-3], parts[-2]
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                r = d["response"]
                if "v4-flash" not in str(r.get("model", "")):
                    continue
                msg = r["choices"][0]["message"]
                text = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
                meta = {
                    "source": "panel",
                    "judge": "V4-Flash",
                    "model": model,
                    "benchmark": bench,
                    "condition": cond,
                    "task_id": d["task_id"],
                    "verdict": verdict_of(text),
                }
                jobs.append((meta, text, cs.LANG[bench]))
    return jobs


def main():
    require(JUDGE_ROOT, what="the error-injection judge outputs (error_study/output/judge/)")
    require(GLOTLID_MODEL, what="the GlotLID v3 model (set GLOTLID_MODEL)")
    ensure_output_dirs()
    jobs = jobs_from_raw()
    print("judgments", len(jobs), flush=True)
    cs.load_models()
    with Pool(int(os.environ.get("WORKERS", "4"))) as p:
        rows = p.map(work, jobs, chunksize=32)
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "judge_language_v4flash.csv", index=False)
    d = df[df.n_letters.fillna(0) >= 20].copy()
    d["cls"] = np.where(d.share_target >= 0.8, "target", np.where(d.share_eng >= 0.8, "english", "mixed"))
    t = pd.crosstab([d.benchmark, d.condition], d.cls, normalize="index").mul(100).round(1)
    t["n"] = d.groupby(["benchmark", "condition"]).size()
    txt = t.to_string()
    print(txt)
    with open(TAB_DIR / "judge_language_v4flash.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
