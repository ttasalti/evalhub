"""Paired solution-level bootstrap intervals for the acceptance differences between error-injection conditions,
behind the "within X points" statements of Section 5.1.

The input is error_injection_panel_3judges.csv in analysis/data (720 solutions x 5 conditions x 3 judges, with
correct_maj as 0/1); the output is bootstrap_paired.txt in analysis/tables.

For each judge and each condition pair (A, B), d_i = accept_B(i) - accept_A(i) in {-1, 0, +1} for solution i.
20,000 resamples of the 720 solutions with replacement give mean(d) * 100; the 95% interval is the 2.5 and 97.5
percentiles. The seed is fixed, and the same resamples are used for every pair. "Within X points" in the text
means max(|low|, |high|) below X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import TAB_DIR, data, ensure_output_dirs

N_BOOT = 20000
SEED = 0
JUDGES = ["V4-Flash", "Qwen3.6", "R1-distill"]
PAIRS = [
    ("clean", "intermediate_error"),
    ("clean", "truncated"),
    ("clean", "boxed_only"),
    ("boxed_only", "consistent_error"),
]


def main() -> None:
    ensure_output_dirs()
    panel = pd.read_csv(data("error_injection_panel_3judges.csv"))
    assert len(panel) == 10800

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 720, size=(N_BOOT, 720))
    lines = []
    for judge in JUDGES:
        w = panel[panel.judge_short == judge].pivot(index="solution_id", columns="condition", values="correct_maj")
        assert w.shape[0] == 720 and w.notna().all().all()
        for a, b in PAIRS:
            d = (w[b] - w[a]).to_numpy(dtype=float)
            observed = 100 * d.mean()
            boot = 100 * d[idx].mean(axis=1)
            low, high = np.percentile(boot, [2.5, 97.5])
            lines.append(
                f"{judge:10s} {a:>10s} -> {b:18s}  diff {observed:+5.1f}  95% [{low:+5.1f}, {high:+5.1f}]"
                f"  |max| {max(abs(low), abs(high)):.1f}"
            )
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (TAB_DIR / "bootstrap_paired.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
