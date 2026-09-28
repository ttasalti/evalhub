"""Exact McNemar tests between error-injection conditions for each judge (the tests quoted in Section 5.1).

The panel is error_injection_panel_3judges.csv in analysis/data (720 solutions x 5 conditions x 3 judges,
correct_maj as 0/1); the results are written to mcnemar_exact.txt in analysis/tables.

Each comparison pairs two conditions on the same 720 solutions. The 2x2 table counts
n11 (accepted under both), n10 (accepted under A only), n01 (accepted under B only) and n00 (rejected under
both). Under the null hypothesis that the judge treats the two conditions alike, a discordant pair falls to
either side with probability 1/2, so n10 ~ Binomial(n10 + n01, 1/2). The exact two-sided p-value is the total
probability of all outcomes no more likely than the observed one (scipy's "two-sided" definition), computed
with exact fractions and cross-checked against scipy. The doubled one-tail p-value and the continuity-corrected
chi-square p-value are printed alongside for comparison.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pandas as pd
from common import TAB_DIR, data, ensure_output_dirs
from scipy.stats import binom, binomtest

JUDGES = ["V4-Flash", "Qwen3.6", "R1-distill"]
COMPARISONS = [  # (label, condition A, condition B)
    ("clean vs intermediate", "clean", "intermediate_error"),
    ("clean vs truncation", "clean", "truncated"),
    ("clean vs final-answer", "clean", "boxed_only"),
    ("final-answer vs consistent", "boxed_only", "consistent_error"),
]

panel = pd.read_csv(data("error_injection_panel_3judges.csv"))
assert len(panel) == 10800 and panel.correct_maj.isin([0, 1]).all()

lines: list[str] = []


def emit(text: str = "") -> None:
    print(text)
    lines.append(text)


def exact_two_sided(k: int, n: int) -> Fraction:
    """P(outcome at most as likely as the observed one | Binomial(n, 1/2)), as an exact fraction."""
    pmf = [Fraction(math.comb(n, i), 2**n) for i in range(n + 1)]
    observed = pmf[k]
    return sum(p for p in pmf if p <= observed)


def two_tail_doubled(k: int, n: int) -> Fraction:
    m = min(k, n - k)
    return min(Fraction(1), 2 * sum(Fraction(math.comb(n, i), 2**n) for i in range(m + 1)))


def chi2_corrected(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    chi = (abs(n10 - n01) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi / 2))


def table(judge: str, a: str, b: str, bench: str | None = None):
    s = panel[panel.judge_short == judge]
    if bench is not None:
        s = s[s.benchmark == bench]
    w = s.pivot(index="solution_id", columns="condition", values="correct_maj")
    x = w[a].astype(int)
    y = w[b].astype(int)
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    assert n11 + n10 + n01 + n00 == len(w)
    return len(w), n11, n10, n01, n00, 100 * x.mean(), 100 * y.mean()


def report(judge: str, label: str, a: str, b: str, bench: str | None = None) -> float:
    n, n11, n10, n01, n00, rate_a, rate_b = table(judge, a, b, bench)
    d = n10 + n01
    p_exact = exact_two_sided(n10, d) if d else Fraction(1)
    p_dbl = two_tail_doubled(n10, d) if d else Fraction(1)
    p_scipy = binomtest(n10, d, 0.5).pvalue if d else 1.0
    p_scipy_dbl = min(1.0, 2 * binom.cdf(min(n10, n01), d, 0.5)) if d else 1.0
    p_chi = chi2_corrected(n10, n01)
    # the exact fraction has to agree with scipy to 1e-9 relative, and the doubled tail with scipy's doubled cdf
    assert abs(float(p_exact) - p_scipy) <= 1e-9 * max(p_scipy, 1e-300), (judge, label, float(p_exact), p_scipy)
    assert abs(float(p_dbl) - p_scipy_dbl) <= 1e-9 * max(p_scipy_dbl, 1e-300), (judge, label, float(p_dbl), p_scipy_dbl)
    tag = f"{judge:10s} {label:28s}" + (f" [{bench}]" if bench else "")
    emit(
        f"{tag}\n   n={n}  A accept {rate_a:.1f}%  B accept {rate_b:.1f}%  diff {rate_b - rate_a:+.1f}"
        f"\n   2x2: both accept {n11}  A-accept/B-reject {n10}  A-reject/B-accept {n01}  both reject {n00}"
        f"\n   exact p = {float(p_exact):.3g}   (doubled tail {float(p_dbl):.3g}, scipy {p_scipy:.3g}, "
        f"scipy doubled tail {p_scipy_dbl:.3g}, chi-square corrected {p_chi:.3g})"
    )
    return float(p_exact)


def main() -> None:
    ensure_output_dirs()
    emit("=== POOLED (720 solutions) ===")
    for judge in JUDGES:
        for label, a, b in COMPARISONS:
            report(judge, label, a, b)
        emit()

    emit("=== R1-distill, per benchmark clean vs final-answer (144 solutions) ===")
    worst = 0.0
    for bench in sorted(panel.benchmark.unique()):
        worst = max(worst, report("R1-distill", "clean vs final-answer", "clean", "boxed_only", bench))
    emit(f"\nR1 largest per-benchmark exact p = {worst:.3g}")
    (TAB_DIR / "mcnemar_exact.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
