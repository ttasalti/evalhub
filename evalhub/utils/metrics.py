from collections import Counter
from typing import Any

import numpy as np

# G-Pass@k thresholds (tau). Fraction of the k sampled generations that must be
# correct for the draw to count as a "pass". Matches the user-requested set.
DEFAULT_TAUS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    r"""Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _compute_g_pass_at_k(n: int, c: int, k: int, m: int) -> float:
    r"""P(at least ``m`` of ``k`` sampled generations are correct).

    Ported verbatim from open-compass/GPassK ``lighteval_metric.py``. Models the
    draw of ``k`` generations (without replacement) out of ``n`` total of which
    ``c`` are correct as a hypergeometric variable X; returns ``P(X >= m)`` via
    the survival function ``sf(m - 1)``.
    """
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    from scipy.stats import hypergeom

    return float(hypergeom.sf(m - 1, n, c, k))


def compute_g_pass_at_k(n: int, c: int, k: int, t: float) -> float:
    r"""G-Pass@k at threshold ``t``: pass iff >= ceil(k*t) of k draws are correct."""
    m = max(int(np.ceil(k * t)), 1)
    return _compute_g_pass_at_k(n, c, k, m)


def compute_mg_pass_at_k(n: int, c: int, k: int) -> float:
    r"""mG-Pass@k: interpolated area of G-Pass@k over thresholds in (0.5, 1.0].

    Ported verbatim from open-compass/GPassK. Sums G-Pass@k at integer
    correct-count cutoffs from ceil(k*0.5)+1 up to k, scaled by 2/k. Note
    ``mG-Pass@1 == 0`` by construction (empty range).
    """
    lo, hi = int(np.ceil(k * 0.5)), k
    acc = 0.0
    for m in range(lo + 1, hi + 1):
        acc += _compute_g_pass_at_k(n, c, k, m)
    return 2.0 * acc / k


def aggregate_g_pass(
    per_task: list[tuple[int, int]],
    ks: list[int],
    taus: tuple[float, ...] = DEFAULT_TAUS,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    r"""Average G-Pass@k / mG-Pass@k over tasks.

    ``per_task`` is a list of ``(n, c)`` pairs — one per task — where ``n`` is
    the number of generations and ``c`` the number of correct ones. For each
    ``k`` in ``ks`` the per-task values are averaged (kept as a fraction in
    [0, 1], matching ``pass_at_k``).

    Returns ``(g_pass_at_k, mg_pass_at_k)`` where::

        g_pass_at_k  = {"<k>": {"<tau>": value, ...}, ...}
        mg_pass_at_k = {"<k>": value, ...}
    """
    g_pass_at_k: dict[str, dict[str, float]] = {}
    mg_pass_at_k: dict[str, float] = {}
    n_tasks = len(per_task)
    if n_tasks == 0:
        return g_pass_at_k, mg_pass_at_k
    for k in ks:
        per_tau_sums = {t: 0.0 for t in taus}
        mg_sum = 0.0
        for n, c in per_task:
            for t in taus:
                per_tau_sums[t] += compute_g_pass_at_k(n, c, k, t)
            mg_sum += compute_mg_pass_at_k(n, c, k)
        g_pass_at_k[str(k)] = {
            _tau_key(t): per_tau_sums[t] / n_tasks for t in taus
        }
        mg_pass_at_k[str(k)] = mg_sum / n_tasks
    return g_pass_at_k, mg_pass_at_k


def _tau_key(t: float) -> str:
    r"""Stable JSON key for a threshold: 0.25 -> "0.25", 0.5 -> "0.5", 1.0 -> "1.0"."""
    return f"{t:g}.0" if float(t).is_integer() else f"{t:g}"


def get_majority_vote(predictions: list[Any]) -> Any:
    filtered = [p for p in predictions if p is not None]
    if not filtered:
        return None

    counter = Counter(filtered)
    return counter.most_common(1)[0][0]
