"""Cell selection rule and per-generation acceptance by benchmark language, quoted in Section 5.

A cell is a solver (model and state). A cell is balanced for a judge when it was
judged on all four of AIME (EN), AIME (TR), AIME (PT) and TUBITAK (TR) at the
16k solver budget. The script reports how many cells are balanced per judge,
the acceptance rate per correct generation in the balanced set, and paired
within-cell Wilcoxon tests between languages. It reads report_tasks.csv from
analysis/data and writes selection_rule.txt to analysis/tables.
"""

import numpy as np
import pandas as pd
from common import TAB_DIR, data, ensure_output_dirs
from scipy.stats import wilcoxon

pd.set_option("display.width", 210)

LANGUAGE = {"aime2026": "EN", "aime2026_tr": "TR", "aime2026_pt": "PT", "tubitak_math2026": "TUB"}
FOUR = ["EN", "TR", "PT", "TUB"]
ALL_JUDGES = ["No-Judge", "Qwen3.6", "Gemma4", "R1-distill", "V4-Flash"]
JUDGES = ["Qwen3.6", "Gemma4", "R1-distill", "V4-Flash"]
RULE = "=" * 78


def load_base():
    df = pd.read_csv(data("report_tasks.csv"))
    key = ["model", "state", "benchmark", "solver_max_tokens", "task_id"]
    ng = df.dropna(subset=["n_generations"]).groupby(key)["n_generations"].max()
    df["n_gen"] = df["n_generations"].fillna(pd.Series(df.set_index(key).index.map(ng), index=df.index))
    df["b"] = df.benchmark.map(LANGUAGE)
    df["cell"] = df.model_short + "|" + df.state
    base = df[(df.solver_max_tokens == 16384) & (df.b.notna())].copy()
    # c = correct generations (approved plus vetoed); cc = approved correct generations.
    base["c"] = np.where(base.judge_short == "No-Judge", base.true_count, base.true_count + base.n_veto.fillna(0))
    base["c"] = base["c"].astype(int)
    base["cc"] = base.true_count.astype(int)
    return base


def balanced(sub):
    """Rows of the cells that cover all four benchmark languages, and the sorted cell names."""
    cov = sub.groupby("cell")["b"].apply(lambda x: set(x.dropna()))
    keep = {c for c, s in cov.items() if set(FOUR) <= s}
    return sub[sub.cell.isin(keep)], sorted(keep)


def main():
    ensure_output_dirs()
    base = load_base()
    lines = []

    lines += [RULE, "SELECTION RULE: solver budget 16384 and all four of EN/TR/PT/TUB present", RULE]
    for jd in ALL_JUDGES:
        s = base[base.judge_short == jd]
        k, cells = balanced(s)
        share = 100 * len(cells) / max(s.cell.nunique(), 1)
        lines.append(f"{jd:11s}  {len(cells):2d} of {s.cell.nunique():2d} cells balanced  ({share:.0f}%)")

    lines += ["", RULE, "1) ACCEPTANCE PER CORRECT GENERATION IN THE BALANCED SET (%)", RULE]
    rows = []
    for jd in JUDGES:
        k, cells = balanced(base[base.judge_short == jd])
        if not cells:
            continue
        r = [jd, len(cells)] + [100 * k[k.b == b].cc.sum() / k[k.b == b].c.sum() for b in FOUR]
        r.append(k[k.b == "EN"].c.sum())
        rows.append(r)
    table = pd.DataFrame(rows, columns=["judge", "balanced cells"] + FOUR + ["EN denominator"])
    lines.append(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    lines += ["", RULE, "PAIRED DIFFERENCE: balanced cells, within cell (Wilcoxon)", RULE]
    for jd in JUDGES:
        k, cells = balanced(base[base.judge_short == jd])
        if len(cells) < 4:
            lines.append(f"{jd}: {len(cells)} balanced cells, too few for the test, skipped")
            continue
        rr = []
        for c in cells:
            s = k[k.cell == c]
            n = {b: s[s.b == b].c.sum() for b in FOUR}
            if min(n.values()) < 50:
                continue
            v = {b: 100 * s[s.b == b].cc.sum() / n[b] for b in FOUR}
            rr.append([v[b] for b in FOUR])
        if len(rr) < 5:
            lines.append(f"{jd}: {len(rr)} cells with denominator >= 50, too few")
            continue
        rates = np.array(rr)
        en, tr, pt, tub = rates[:, 0], rates[:, 1], rates[:, 2], rates[:, 3]
        lines.append(
            f"{jd:11s} (cells with denominator >= 50: {len(rr)})   "
            f"TR-EN {tr.mean() - en.mean():+5.1f} p={wilcoxon(tr, en).pvalue:.3f}   "
            f"PT-EN {pt.mean() - en.mean():+5.1f} p={wilcoxon(pt, en).pvalue:.3f}   "
            f"TUB-TR {tub.mean() - tr.mean():+5.1f} p={wilcoxon(tub, tr).pvalue:.4f}"
        )

    text = "\n".join(lines) + "\n"
    out = TAB_DIR / "selection_rule.txt"
    out.write_text(text)
    print(text, end="")
    print("wrote", out)


if __name__ == "__main__":
    main()
