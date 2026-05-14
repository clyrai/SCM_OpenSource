"""
ALB statistical methods.

Implements the methodology defined in SPEC.md §5:
  - Bootstrap 95% CIs (10,000 resamples) on every reported metric.
  - Paired t-test for SCM vs each baseline on each metric.
  - Wilcoxon signed-rank as non-parametric companion.
  - Cohen's d (paired) for effect size.
  - Holm-Bonferroni correction for multi-comparison families.
  - Minimum reportable effect threshold (Cohen's d >= 0.2).

We avoid scipy.stats here because the SCM project already keeps its
test deps minimal; bootstrapping and the paired t-test are simple
enough to implement directly. Wilcoxon uses a clean ranks-based form.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


# ─── Bootstrap CI ──────────────────────────────────────────────────────────


@dataclass
class BootstrapCI:
    mean: float
    low: float
    high: float
    n: int
    n_resamples: int
    alpha: float


def bootstrap_ci(
    scores: Sequence[float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> BootstrapCI:
    """
    Percentile bootstrap CI on the mean.

    Pure-Python so we don't pull numpy as a hard dependency for the
    benchmark runner.
    """
    n = len(scores)
    if n == 0:
        return BootstrapCI(mean=0.0, low=0.0, high=0.0, n=0,
                           n_resamples=n_resamples, alpha=alpha)
    if n == 1:
        return BootstrapCI(mean=float(scores[0]), low=float(scores[0]),
                           high=float(scores[0]), n=1,
                           n_resamples=n_resamples, alpha=alpha)

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample_mean = sum(rng.choice(scores) for _ in range(n)) / n
        means.append(sample_mean)
    means.sort()

    lo_idx = int(math.floor(n_resamples * alpha / 2))
    hi_idx = int(math.ceil(n_resamples * (1 - alpha / 2))) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))

    return BootstrapCI(
        mean=sum(scores) / n,
        low=means[lo_idx],
        high=means[hi_idx],
        n=n,
        n_resamples=n_resamples,
        alpha=alpha,
    )


# ─── Paired t-test ─────────────────────────────────────────────────────────


@dataclass
class PairedTResult:
    """Output of a paired t-test."""
    n: int
    mean_diff: float
    sd_diff: float
    se_diff: float
    t: float
    df: int
    p_two_tailed: float
    ci_low: float
    ci_high: float


def _t_cdf(t: float, df: int) -> float:
    """
    Two-sided p-value approximation for the t-distribution.

    Uses the regularized incomplete beta. We avoid scipy by implementing
    Lentz's continued fraction for the incomplete beta. Accurate to
    ~1e-6 for reasonable df, more than enough for paper reporting.
    """
    # p = 1 - I_{x}(df/2, 1/2) where x = df / (df + t^2)
    if df < 1:
        return 1.0
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    # Use symmetry / continued fraction for I_x(a, b).
    ibeta = _regularized_incomplete_beta(x, a, b)
    p = ibeta  # one-sided p in the upper tail of |t|
    return min(1.0, max(0.0, p))


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """
    I_x(a, b) — regularized incomplete beta function.

    Lentz's continued fraction expansion. From Numerical Recipes 3e §6.4
    (rewritten in Python).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # log of the front factor: ln(B(a,b) * x^a * (1-x)^b)
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_front = lbeta + a * math.log(x) + b * math.log(1.0 - x)
    front = math.exp(log_front)

    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    else:
        return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued fraction for the incomplete beta. Lentz's algorithm."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def paired_t_test(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> PairedTResult:
    """
    Paired t-test of (a - b). a and b must be paired by index.

    Returns PairedTResult with mean diff, t, df, two-sided p, and CI on
    the mean difference.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return PairedTResult(
            n=n, mean_diff=0.0, sd_diff=0.0, se_diff=0.0,
            t=0.0, df=max(0, n - 1), p_two_tailed=1.0,
            ci_low=0.0, ci_high=0.0,
        )

    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(max(0.0, var_d))
    se_d = sd_d / math.sqrt(n) if n > 0 else 0.0
    t = mean_d / se_d if se_d > 0 else 0.0
    df = n - 1
    p = _t_cdf(abs(t), df)

    # CI on mean diff using normal approximation (df is usually >= 30 for ALB)
    z_or_t = _t_critical(df, alpha)
    ci_low = mean_d - z_or_t * se_d
    ci_high = mean_d + z_or_t * se_d

    return PairedTResult(
        n=n, mean_diff=mean_d, sd_diff=sd_d, se_diff=se_d,
        t=t, df=df, p_two_tailed=p,
        ci_low=ci_low, ci_high=ci_high,
    )


def _t_critical(df: int, alpha: float = 0.05) -> float:
    """
    Two-sided t critical value for given df.

    Approximation using the inverse incomplete beta. For df >= 30 this
    is essentially the normal critical value. For ALB n=100, df=99,
    the value is very close to z_{0.975} = 1.96.
    """
    if df >= 30:
        # Normal approximation good enough.
        return _norm_inv(1 - alpha / 2)
    # Crude bisection on the t CDF for small df.
    lo, hi = 0.0, 100.0
    target = alpha
    for _ in range(60):
        mid = (lo + hi) / 2
        p = _t_cdf(mid, df)
        if p > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _norm_inv(p: float) -> float:
    """
    Inverse standard normal CDF. Beasley-Springer-Moro approximation.
    Good to ~1e-9 in the body of the distribution.
    """
    if p < 0.5:
        return -_norm_inv(1 - p)
    # Constants from Moro (1995)
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187,
    ]
    u = p - 0.5
    if abs(u) < 0.42:
        r = u * u
        num = ((a[3] * r + a[2]) * r + a[1]) * r + a[0]
        den = (((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0
        return u * num / den
    r = 1 - p
    r = math.log(-math.log(r))
    x = c[0]
    x = c[1] + r * x
    for i in range(2, 9):
        x = c[i] + r * x
    return x


# ─── Wilcoxon signed-rank ──────────────────────────────────────────────────


@dataclass
class WilcoxonResult:
    n: int
    w_plus: float
    w_minus: float
    p_two_tailed: float


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> WilcoxonResult:
    """
    Paired Wilcoxon signed-rank test, normal-approximation form.

    Used as a non-parametric companion to the paired t-test, especially
    useful when scores are bounded [0, 1] and heavily skewed.
    """
    if len(a) != len(b):
        raise ValueError("length mismatch")
    diffs = [float(x) - float(y) for x, y in zip(a, b) if (x - y) != 0]
    n = len(diffs)
    if n == 0:
        return WilcoxonResult(n=0, w_plus=0.0, w_minus=0.0, p_two_tailed=1.0)

    abs_diffs = sorted(((abs(d), d) for d in diffs), key=lambda z: z[0])
    # Assign average ranks for ties on |d|
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2  # 1-based ranks
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
    w_minus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)

    # Normal approximation
    mu = n * (n + 1) / 4
    sigma2 = n * (n + 1) * (2 * n + 1) / 24
    sigma = math.sqrt(sigma2) if sigma2 > 0 else 1.0
    z = (w_plus - mu) / sigma
    p_two_tailed = 2 * (1 - _norm_cdf(abs(z)))

    return WilcoxonResult(
        n=n, w_plus=w_plus, w_minus=w_minus,
        p_two_tailed=min(1.0, max(0.0, p_two_tailed)),
    )


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ─── Cohen's d (paired) ────────────────────────────────────────────────────


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Paired Cohen's d: mean(a - b) / sd(a - b).

    Conventional thresholds:
      |d| < 0.2  → trivial
      0.2 ≤ |d| < 0.5 → small
      0.5 ≤ |d| < 0.8 → medium
      |d| ≥ 0.8 → large

    SPEC.md §5.5: effects with |d| < 0.2 are reported as "not meaningful"
    even when statistically significant.
    """
    if len(a) != len(b):
        raise ValueError("length mismatch")
    n = len(a)
    if n < 2:
        return 0.0
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(max(0.0, var_d))
    if sd_d == 0:
        return 0.0
    return mean_d / sd_d


# ─── Holm-Bonferroni correction ────────────────────────────────────────────


@dataclass
class HolmDecision:
    """One row of the Holm-Bonferroni decision table."""
    label: str
    p: float
    rank: int
    threshold: float
    reject_null: bool


def holm_bonferroni(
    p_values: Sequence[Tuple[str, float]],
    alpha: float = 0.05,
) -> List[HolmDecision]:
    """
    Apply Holm-Bonferroni step-down correction to a family of p-values.

    Args:
        p_values: list of (label, p) tuples.
        alpha: family-wise error rate (default 0.05).

    Returns:
        Decisions in the original order.

    Holm-Bonferroni:
        Sort p-values ascending. The k-th smallest p (1-indexed) is
        compared to alpha / (m - k + 1) where m = total comparisons.
        Reject null in step-down order; once one fails, all higher fail.
    """
    m = len(p_values)
    if m == 0:
        return []
    # Sort by p ascending while keeping original index for restore.
    indexed = sorted(enumerate(p_values), key=lambda z: z[1][1])
    decisions = [None] * m
    rejected_so_far = True
    for k, (orig_i, (label, p)) in enumerate(indexed, start=1):
        threshold = alpha / (m - k + 1)
        if rejected_so_far and p < threshold:
            decision = True
        else:
            decision = False
            rejected_so_far = False
        decisions[orig_i] = HolmDecision(
            label=label, p=p, rank=k, threshold=threshold,
            reject_null=decision,
        )
    return decisions  # type: ignore[return-value]
