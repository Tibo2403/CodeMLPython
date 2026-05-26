"""Automatic statistics calculator for management/statistics exercises.

The module can be imported from notebooks or used from the command line:

    python statistics_calculator.py describe 12 15 18 20
    python statistics_calculator.py ci-mean 12 15 18 20 --confidence 0.95
    python statistics_calculator.py test-mean 12 15 18 20 --mu0 13 --sigma 4

It intentionally uses only Python's standard library, so it works in small
classroom environments without SciPy.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Iterable, Sequence


TAILS = {"two-sided", "left", "right"}


@dataclass(frozen=True)
class ConfidenceInterval:
    """A confidence interval with its center and margin of error."""

    center: float
    margin: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class TestResult:
    """Result of a hypothesis test."""

    statistic: float
    p_value: float
    reject_h0: bool
    alpha: float
    distribution: str
    degrees_freedom: float | None = None


@dataclass(frozen=True)
class RegressionResult:
    """Simple linear regression and its main inference quantities."""

    intercept: float
    slope: float
    r: float
    r_squared: float
    residual_variance: float
    slope_interval: ConfidenceInterval
    intercept_interval: ConfidenceInterval


def parse_values(values: Iterable[str | float | int]) -> list[float]:
    """Parse numbers from CLI values or comma/semicolon separated strings."""
    parsed: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            parsed.append(float(value))
            continue
        normalized = value.replace(";", ",")
        for part in normalized.split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    if not parsed:
        raise ValueError("At least one numeric value is required.")
    return parsed


def _sample_variance(values: Sequence[float]) -> float:
    data = parse_values(values)
    if len(data) < 2:
        raise ValueError("At least two values are required.")
    avg = mean(data)
    return sum((x - avg) ** 2 for x in data) / (len(data) - 1)


def _tail_p_value(statistic: float, cdf_value: float, tail: str) -> float:
    if tail == "left":
        return cdf_value
    if tail == "right":
        return 1.0 - cdf_value
    if tail == "two-sided":
        return min(1.0, 2.0 * min(cdf_value, 1.0 - cdf_value))
    raise ValueError(f"tail must be one of: {', '.join(sorted(TAILS))}.")


def descriptive_stats(values: Sequence[float]) -> dict[str, float]:
    """Return common descriptive statistics for a sample."""
    data = parse_values(values)
    n = len(data)
    avg = mean(data)
    variance_population = sum((x - avg) ** 2 for x in data) / n
    variance_sample = sum((x - avg) ** 2 for x in data) / (n - 1) if n > 1 else math.nan
    sample_stdev = math.sqrt(variance_sample) if n > 1 else math.nan
    return {
        "n": float(n),
        "mean": avg,
        "median": median(data),
        "minimum": min(data),
        "maximum": max(data),
        "range": max(data) - min(data),
        "variance_population": variance_population,
        "variance_sample": variance_sample,
        "stdev_sample": sample_stdev,
        "coefficient_variation": sample_stdev / avg if n > 1 and avg != 0 else math.nan,
    }


def grouped_descriptive_stats(
    centers: Sequence[float],
    counts: Sequence[float],
) -> dict[str, float]:
    """Mean and variance for a distribution grouped by class centers."""
    x = parse_values(centers)
    n_i = parse_values(counts)
    if len(x) != len(n_i):
        raise ValueError("centers and counts must have the same length.")
    total = sum(n_i)
    if total <= 0:
        raise ValueError("The total count must be positive.")
    avg = sum(count * value for count, value in zip(n_i, x)) / total
    variance = sum(count * value**2 for count, value in zip(n_i, x)) / total - avg**2
    return {"n": total, "mean": avg, "variance_population": variance, "stdev_population": math.sqrt(variance)}


def covariance(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    x = parse_values(values_x)
    y = parse_values(values_y)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have the same length and at least two values.")
    return sum((xi - mean(x)) * (yi - mean(y)) for xi, yi in zip(x, y)) / len(x)


def correlation(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    """Population-style correlation coefficient used in the course formula."""
    x = parse_values(values_x)
    y = parse_values(values_y)
    sx = math.sqrt(sum((xi - mean(x)) ** 2 for xi in x) / len(x))
    sy = math.sqrt(sum((yi - mean(y)) ** 2 for yi in y) / len(y))
    if sx == 0 or sy == 0:
        raise ValueError("x and y must both vary.")
    return covariance(x, y) / (sx * sy)


def normal_cdf(z: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_probability(
    lower: float | None = None,
    upper: float | None = None,
    mean_value: float = 0.0,
    std: float = 1.0,
) -> float:
    """Probability P(lower <= X <= upper) for X following N(mean, std)."""
    if std <= 0:
        raise ValueError("std must be positive.")
    lower_cdf = 0.0 if lower is None else normal_cdf((lower - mean_value) / std)
    upper_cdf = 1.0 if upper is None else normal_cdf((upper - mean_value) / std)
    return upper_cdf - lower_cdf


def inverse_normal_cdf(probability: float) -> float:
    """Approximate inverse CDF for the standard normal distribution."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1.")
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    low = 0.02425
    high = 1.0 - low
    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if probability <= high:
        q = probability - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _regularized_gamma_p(a: float, x: float) -> float:
    if a <= 0 or x < 0:
        raise ValueError("Invalid gamma parameters.")
    if x == 0:
        return 0.0
    eps = 1e-12
    max_iter = 200
    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(max_iter):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))

    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return max(0.0, min(1.0, 1.0 - q))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    eps = 3e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_beta(a: float, b: float, x: float) -> float:
    if a <= 0 or b <= 0 or not 0 <= x <= 1:
        raise ValueError("Invalid beta parameters.")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t_value: float, degrees_freedom: float) -> float:
    """Student t cumulative distribution function."""
    if degrees_freedom <= 0:
        raise ValueError("degrees_freedom must be positive.")
    x = degrees_freedom / (degrees_freedom + t_value * t_value)
    ibeta = _regularized_beta(degrees_freedom / 2.0, 0.5, x)
    return 1.0 - 0.5 * ibeta if t_value >= 0 else 0.5 * ibeta


def chi_square_cdf(value: float, degrees_freedom: float) -> float:
    if value < 0 or degrees_freedom <= 0:
        raise ValueError("Invalid chi-square parameters.")
    return _regularized_gamma_p(degrees_freedom / 2.0, value / 2.0)


def f_cdf(value: float, df1: float, df2: float) -> float:
    if value < 0 or df1 <= 0 or df2 <= 0:
        raise ValueError("Invalid F parameters.")
    x = (df1 * value) / (df1 * value + df2)
    return _regularized_beta(df1 / 2.0, df2 / 2.0, x)


def _inverse_cdf(probability: float, cdf, low: float = 0.0, high: float = 1.0) -> float:
    if not 0 < probability < 1:
        raise ValueError("probability must be between 0 and 1.")
    while cdf(high) < probability:
        high *= 2.0
        if high > 1e10:
            raise RuntimeError("Could not bracket the quantile.")
    for _ in range(100):
        mid = (low + high) / 2.0
        if cdf(mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def z_critical(confidence: float) -> float:
    return inverse_normal_cdf(0.5 + confidence / 2.0)


def t_critical(confidence: float, degrees_freedom: float) -> float:
    return _inverse_cdf(0.5 + confidence / 2.0, lambda x: student_t_cdf(x, degrees_freedom))


def chi_square_critical(probability: float, degrees_freedom: float) -> float:
    return _inverse_cdf(probability, lambda x: chi_square_cdf(x, degrees_freedom))


def f_critical(probability: float, df1: float, df2: float) -> float:
    return _inverse_cdf(probability, lambda x: f_cdf(x, df1, df2))


def confidence_interval_mean(
    values: Sequence[float],
    confidence: float = 0.95,
    population_std: float | None = None,
    large_sample: bool = False,
) -> ConfidenceInterval:
    """IC for a mean: known variance uses z, unknown normal uses t, large n uses z."""
    data = parse_values(values)
    n = len(data)
    if n < 2 and population_std is None:
        raise ValueError("At least two values are required without a population std.")
    avg = mean(data)
    spread = population_std if population_std is not None else stdev(data)
    critical = z_critical(confidence) if population_std is not None or large_sample else t_critical(confidence, n - 1)
    margin = critical * spread / math.sqrt(n)
    return ConfidenceInterval(avg, margin, avg - margin, avg + margin, confidence)


def confidence_interval_mean_summary(
    sample_mean: float,
    std: float,
    n: int,
    confidence: float = 0.95,
    population_std_known: bool = False,
    large_sample: bool = False,
) -> ConfidenceInterval:
    if std <= 0 or n <= 0:
        raise ValueError("std and n must be positive.")
    critical = z_critical(confidence) if population_std_known or large_sample else t_critical(confidence, n - 1)
    margin = critical * std / math.sqrt(n)
    return ConfidenceInterval(sample_mean, margin, sample_mean - margin, sample_mean + margin, confidence)


def confidence_interval_proportion(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """IC for one proportion, large-sample normal approximation."""
    if not 0 <= successes <= n or n <= 0:
        raise ValueError("successes must be between 0 and n, and n must be positive.")
    p_hat = successes / n
    margin = z_critical(confidence) * math.sqrt(p_hat * (1.0 - p_hat) / n)
    return ConfidenceInterval(p_hat, margin, max(0.0, p_hat - margin), min(1.0, p_hat + margin), confidence)


def confidence_interval_variance(values: Sequence[float], confidence: float = 0.95) -> ConfidenceInterval:
    """IC for a normal population variance."""
    data = parse_values(values)
    variance = _sample_variance(data)
    return confidence_interval_variance_summary(variance, len(data), confidence)


def confidence_interval_variance_summary(sample_variance: float, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    if sample_variance <= 0 or n < 2:
        raise ValueError("sample_variance must be positive and n must be at least 2.")
    alpha = 1.0 - confidence
    df = n - 1
    lower = df * sample_variance / chi_square_critical(1.0 - alpha / 2.0, df)
    upper = df * sample_variance / chi_square_critical(alpha / 2.0, df)
    return ConfidenceInterval(sample_variance, math.nan, lower, upper, confidence)


def confidence_interval_two_means_known(
    mean1: float,
    mean2: float,
    sigma1: float,
    sigma2: float,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    center = mean1 - mean2
    margin = z_critical(confidence) * math.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    return ConfidenceInterval(center, margin, center - margin, center + margin, confidence)


def confidence_interval_two_means_equal_variance(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    center = mean1 - mean2
    df = n1 + n2 - 2
    pooled = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / df)
    margin = t_critical(confidence, df) * pooled * math.sqrt(1.0 / n1 + 1.0 / n2)
    return ConfidenceInterval(center, margin, center - margin, center + margin, confidence)


def confidence_interval_two_means_large(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    center = mean1 - mean2
    margin = z_critical(confidence) * math.sqrt(std1**2 / n1 + std2**2 / n2)
    return ConfidenceInterval(center, margin, center - margin, center + margin, confidence)


def confidence_interval_paired_mean(
    before: Sequence[float],
    after: Sequence[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    x = parse_values(before)
    y = parse_values(after)
    if len(x) != len(y):
        raise ValueError("before and after must have the same length.")
    return confidence_interval_mean([xi - yi for xi, yi in zip(x, y)], confidence)


def confidence_interval_two_proportions(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    p1 = successes1 / n1
    p2 = successes2 / n2
    center = p1 - p2
    margin = z_critical(confidence) * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return ConfidenceInterval(center, margin, center - margin, center + margin, confidence)


def confidence_interval_variance_ratio(
    variance1: float,
    variance2: float,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if variance1 <= 0 or variance2 <= 0 or n1 < 2 or n2 < 2:
        raise ValueError("Variances must be positive and sample sizes at least 2.")
    alpha = 1.0 - confidence
    ratio = variance1 / variance2
    df1 = n1 - 1
    df2 = n2 - 1
    lower = ratio / f_critical(1.0 - alpha / 2.0, df1, df2)
    upper = ratio / f_critical(alpha / 2.0, df1, df2)
    return ConfidenceInterval(ratio, math.nan, lower, upper, confidence)


def z_test_mean(
    sample_mean: float,
    hypothesized_mean: float,
    std: float,
    n: int,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    if std <= 0 or n <= 0:
        raise ValueError("std and n must be positive.")
    z = (sample_mean - hypothesized_mean) / (std / math.sqrt(n))
    p_value = _tail_p_value(z, normal_cdf(z), tail)
    return TestResult(z, p_value, p_value < alpha, alpha, "N(0,1)")


def t_test_mean(
    values: Sequence[float],
    hypothesized_mean: float,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    data = parse_values(values)
    df = len(data) - 1
    t_value = (mean(data) - hypothesized_mean) / (stdev(data) / math.sqrt(len(data)))
    p_value = _tail_p_value(t_value, student_t_cdf(t_value, df), tail)
    return TestResult(t_value, p_value, p_value < alpha, alpha, f"t({df})", df)


def test_mean(
    values: Sequence[float],
    hypothesized_mean: float,
    alpha: float = 0.05,
    tail: str = "two-sided",
    population_std: float | None = None,
    large_sample: bool = False,
) -> TestResult:
    data = parse_values(values)
    if population_std is not None:
        return z_test_mean(mean(data), hypothesized_mean, population_std, len(data), alpha, tail)
    if large_sample:
        return z_test_mean(mean(data), hypothesized_mean, stdev(data), len(data), alpha, tail)
    return t_test_mean(data, hypothesized_mean, alpha, tail)


def z_test_proportion(
    successes: int,
    n: int,
    hypothesized_p: float,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    if not 0 < hypothesized_p < 1 or not 0 <= successes <= n or n <= 0:
        raise ValueError("Use 0 < p0 < 1 and 0 <= successes <= n.")
    p_hat = successes / n
    z = (p_hat - hypothesized_p) / math.sqrt(hypothesized_p * (1 - hypothesized_p) / n)
    p_value = _tail_p_value(z, normal_cdf(z), tail)
    return TestResult(z, p_value, p_value < alpha, alpha, "N(0,1)")


def z_test_two_means_known(
    mean1: float,
    mean2: float,
    sigma1: float,
    sigma2: float,
    n1: int,
    n2: int,
    hypothesized_diff: float = 0.0,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    z = ((mean1 - mean2) - hypothesized_diff) / math.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
    p_value = _tail_p_value(z, normal_cdf(z), tail)
    return TestResult(z, p_value, p_value < alpha, alpha, "N(0,1)")


def t_test_two_means_equal_variance(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
    hypothesized_diff: float = 0.0,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    df = n1 + n2 - 2
    pooled = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / df)
    t_value = ((mean1 - mean2) - hypothesized_diff) / (pooled * math.sqrt(1 / n1 + 1 / n2))
    p_value = _tail_p_value(t_value, student_t_cdf(t_value, df), tail)
    return TestResult(t_value, p_value, p_value < alpha, alpha, f"t({df})", df)


def z_test_two_means_large(
    mean1: float,
    mean2: float,
    std1: float,
    std2: float,
    n1: int,
    n2: int,
    hypothesized_diff: float = 0.0,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    z = ((mean1 - mean2) - hypothesized_diff) / math.sqrt(std1**2 / n1 + std2**2 / n2)
    p_value = _tail_p_value(z, normal_cdf(z), tail)
    return TestResult(z, p_value, p_value < alpha, alpha, "N(0,1)")


def paired_t_test(
    before: Sequence[float],
    after: Sequence[float],
    hypothesized_mean_diff: float = 0.0,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    x = parse_values(before)
    y = parse_values(after)
    if len(x) != len(y):
        raise ValueError("before and after must have the same length.")
    return t_test_mean([xi - yi for xi, yi in zip(x, y)], hypothesized_mean_diff, alpha, tail)


def chi_square_test_variance(
    sample_variance: float,
    n: int,
    hypothesized_variance: float,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    if sample_variance <= 0 or hypothesized_variance <= 0 or n < 2:
        raise ValueError("Variances must be positive and n must be at least 2.")
    df = n - 1
    statistic = df * sample_variance / hypothesized_variance
    p_value = _tail_p_value(statistic, chi_square_cdf(statistic, df), tail)
    return TestResult(statistic, p_value, p_value < alpha, alpha, f"chi-square({df})", df)


def f_test_variances(
    variance1: float,
    variance2: float,
    n1: int,
    n2: int,
    hypothesized_ratio: float = 1.0,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    if variance1 <= 0 or variance2 <= 0 or hypothesized_ratio <= 0 or n1 < 2 or n2 < 2:
        raise ValueError("Variances and hypothesized ratio must be positive; n1 and n2 at least 2.")
    df1 = n1 - 1
    df2 = n2 - 1
    statistic = (variance1 / variance2) / hypothesized_ratio
    p_value = _tail_p_value(statistic, f_cdf(statistic, df1, df2), tail)
    return TestResult(statistic, p_value, p_value < alpha, alpha, f"F({df1},{df2})", None)


def z_test_two_proportions(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    hypothesized_diff: float = 0.0,
    alpha: float = 0.05,
    tail: str = "two-sided",
) -> TestResult:
    if min(n1, n2) <= 0 or not 0 <= successes1 <= n1 or not 0 <= successes2 <= n2:
        raise ValueError("Use valid successes and sample sizes.")
    p1 = successes1 / n1
    p2 = successes2 / n2
    if hypothesized_diff == 0:
        pooled = (successes1 + successes2) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    else:
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = ((p1 - p2) - hypothesized_diff) / se
    p_value = _tail_p_value(z, normal_cdf(z), tail)
    return TestResult(z, p_value, p_value < alpha, alpha, "N(0,1)")


def chi_square_goodness_of_fit(
    observed: Sequence[float],
    expected_probabilities: Sequence[float],
    alpha: float = 0.05,
    estimated_parameters: int = 0,
) -> TestResult:
    obs = parse_values(observed)
    probs = parse_values(expected_probabilities)
    if len(obs) != len(probs) or any(p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("observed and probabilities must match, and probabilities must sum to 1.")
    total = sum(obs)
    expected = [total * p for p in probs]
    statistic = sum((o - e) ** 2 / e for o, e in zip(obs, expected) if e > 0)
    df = len(obs) - 1 - estimated_parameters
    p_value = 1.0 - chi_square_cdf(statistic, df)
    return TestResult(statistic, p_value, p_value < alpha, alpha, f"chi-square({df})", df)


def chi_square_independence(table: Sequence[Sequence[float]], alpha: float = 0.05) -> TestResult:
    if len(table) < 2 or any(len(row) != len(table[0]) for row in table) or len(table[0]) < 2:
        raise ValueError("The contingency table must be rectangular with at least 2 rows and 2 columns.")
    row_totals = [sum(row) for row in table]
    col_totals = [sum(row[j] for row in table) for j in range(len(table[0]))]
    total = sum(row_totals)
    statistic = 0.0
    for i, row in enumerate(table):
        for j, observed in enumerate(row):
            expected = row_totals[i] * col_totals[j] / total
            statistic += (observed - expected) ** 2 / expected
    df = (len(table) - 1) * (len(table[0]) - 1)
    p_value = 1.0 - chi_square_cdf(statistic, df)
    return TestResult(statistic, p_value, p_value < alpha, alpha, f"chi-square({df})", df)


def simple_linear_regression(
    values_x: Sequence[float],
    values_y: Sequence[float],
    confidence: float = 0.95,
) -> RegressionResult:
    x = parse_values(values_x)
    y = parse_values(values_y)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("x and y must have the same length and at least three values.")
    n = len(x)
    x_bar = mean(x)
    y_bar = mean(y)
    sxx = sum((xi - x_bar) ** 2 for xi in x)
    sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    syy = sum((yi - y_bar) ** 2 for yi in y)
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    residual_variance = sum(e * e for e in residuals) / (n - 2)
    critical = t_critical(confidence, n - 2)
    slope_se = math.sqrt(residual_variance / sxx)
    intercept_se = math.sqrt(residual_variance * (1 / n + x_bar**2 / sxx))
    slope_interval = ConfidenceInterval(slope, critical * slope_se, slope - critical * slope_se, slope + critical * slope_se, confidence)
    intercept_interval = ConfidenceInterval(intercept, critical * intercept_se, intercept - critical * intercept_se, intercept + critical * intercept_se, confidence)
    r = sxy / math.sqrt(sxx * syy)
    return RegressionResult(intercept, slope, r, r * r, residual_variance, slope_interval, intercept_interval)


def binomial_probability(n: int, k: int, probability: float) -> float:
    if n < 0 or not 0 <= k <= n or not 0 <= probability <= 1:
        raise ValueError("Use n >= 0, 0 <= k <= n and 0 <= probability <= 1.")
    return math.comb(n, k) * probability**k * (1.0 - probability) ** (n - k)


def poisson_probability(lam: float, k: int) -> float:
    if lam <= 0 or k < 0:
        raise ValueError("lambda must be positive and k must be non-negative.")
    return math.exp(-lam) * lam**k / math.factorial(k)


def print_mapping(values: dict[str, float]) -> None:
    for key, value in values.items():
        print(f"{key}: {value:.6g}")


def print_interval(interval: ConfidenceInterval) -> None:
    print(f"center: {interval.center:.6g}")
    if not math.isnan(interval.margin):
        print(f"margin: {interval.margin:.6g}")
    print(f"lower: {interval.lower:.6g}")
    print(f"upper: {interval.upper:.6g}")
    print(f"confidence: {interval.confidence:.2%}")


def print_test(result: TestResult) -> None:
    print(f"statistic: {result.statistic:.6g}")
    print(f"distribution: {result.distribution}")
    print(f"p_value: {result.p_value:.6g}")
    print(f"reject_h0: {result.reject_h0}")
    print(f"alpha: {result.alpha:.6g}")


def _add_summary_two_means_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mean1", type=float, required=True)
    parser.add_argument("--mean2", type=float, required=True)
    parser.add_argument("--std1", type=float, required=True)
    parser.add_argument("--std2", type=float, required=True)
    parser.add_argument("--n1", type=int, required=True)
    parser.add_argument("--n2", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Automatic statistics calculator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Descriptive statistics.")
    describe.add_argument("values", nargs="+")

    grouped = subparsers.add_parser("describe-grouped", help="Grouped descriptive statistics.")
    grouped.add_argument("--centers", nargs="+", required=True)
    grouped.add_argument("--counts", nargs="+", required=True)

    normal = subparsers.add_parser("normal", help="Normal distribution probability.")
    normal.add_argument("--mean", type=float, default=0.0)
    normal.add_argument("--std", type=float, default=1.0)
    normal.add_argument("--lower", type=float)
    normal.add_argument("--upper", type=float)

    ci_mean = subparsers.add_parser("ci-mean", help="IC for one mean.")
    ci_mean.add_argument("values", nargs="+")
    ci_mean.add_argument("--confidence", type=float, default=0.95)
    ci_mean.add_argument("--population-std", type=float)
    ci_mean.add_argument("--large-sample", action="store_true")

    ci_prop = subparsers.add_parser("ci-proportion", help="IC for one proportion.")
    ci_prop.add_argument("--successes", type=int, required=True)
    ci_prop.add_argument("--n", type=int, required=True)
    ci_prop.add_argument("--confidence", type=float, default=0.95)

    ci_var = subparsers.add_parser("ci-variance", help="IC for one variance.")
    ci_var.add_argument("--sample-variance", type=float)
    ci_var.add_argument("--n", type=int)
    ci_var.add_argument("--values", nargs="+")
    ci_var.add_argument("--confidence", type=float, default=0.95)

    ci_2m = subparsers.add_parser("ci-two-means", help="IC for mu1 - mu2.")
    _add_summary_two_means_args(ci_2m)
    ci_2m.add_argument("--case", choices=["known", "equal-variance", "large"], default="large")
    ci_2m.add_argument("--confidence", type=float, default=0.95)

    ci_pair = subparsers.add_parser("ci-paired", help="Paired IC for mean differences.")
    ci_pair.add_argument("--before", nargs="+", required=True)
    ci_pair.add_argument("--after", nargs="+", required=True)
    ci_pair.add_argument("--confidence", type=float, default=0.95)

    ci_2p = subparsers.add_parser("ci-two-proportions", help="IC for p1 - p2.")
    ci_2p.add_argument("--successes1", type=int, required=True)
    ci_2p.add_argument("--n1", type=int, required=True)
    ci_2p.add_argument("--successes2", type=int, required=True)
    ci_2p.add_argument("--n2", type=int, required=True)
    ci_2p.add_argument("--confidence", type=float, default=0.95)

    ci_vr = subparsers.add_parser("ci-variance-ratio", help="IC for sigma1^2 / sigma2^2.")
    ci_vr.add_argument("--variance1", type=float, required=True)
    ci_vr.add_argument("--variance2", type=float, required=True)
    ci_vr.add_argument("--n1", type=int, required=True)
    ci_vr.add_argument("--n2", type=int, required=True)
    ci_vr.add_argument("--confidence", type=float, default=0.95)

    test_m = subparsers.add_parser("test-mean", help="Hypothesis test for one mean.")
    test_m.add_argument("values", nargs="+")
    test_m.add_argument("--mu0", type=float, required=True)
    test_m.add_argument("--sigma", type=float)
    test_m.add_argument("--large-sample", action="store_true")
    test_m.add_argument("--alpha", type=float, default=0.05)
    test_m.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    test_p = subparsers.add_parser("test-proportion", help="Hypothesis test for one proportion.")
    test_p.add_argument("--successes", type=int, required=True)
    test_p.add_argument("--n", type=int, required=True)
    test_p.add_argument("--p0", type=float, required=True)
    test_p.add_argument("--alpha", type=float, default=0.05)
    test_p.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    test_2m = subparsers.add_parser("test-two-means", help="Hypothesis test for mu1 - mu2.")
    _add_summary_two_means_args(test_2m)
    test_2m.add_argument("--case", choices=["known", "equal-variance", "large"], default="large")
    test_2m.add_argument("--diff0", type=float, default=0.0)
    test_2m.add_argument("--alpha", type=float, default=0.05)
    test_2m.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    test_pair = subparsers.add_parser("test-paired", help="Paired t test.")
    test_pair.add_argument("--before", nargs="+", required=True)
    test_pair.add_argument("--after", nargs="+", required=True)
    test_pair.add_argument("--diff0", type=float, default=0.0)
    test_pair.add_argument("--alpha", type=float, default=0.05)
    test_pair.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    test_var = subparsers.add_parser("test-variance", help="Chi-square test for one variance.")
    test_var.add_argument("--sample-variance", type=float, required=True)
    test_var.add_argument("--n", type=int, required=True)
    test_var.add_argument("--variance0", type=float, required=True)
    test_var.add_argument("--alpha", type=float, default=0.05)
    test_var.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    test_f = subparsers.add_parser("test-variances", help="F test for two variances.")
    test_f.add_argument("--variance1", type=float, required=True)
    test_f.add_argument("--variance2", type=float, required=True)
    test_f.add_argument("--n1", type=int, required=True)
    test_f.add_argument("--n2", type=int, required=True)
    test_f.add_argument("--ratio0", type=float, default=1.0)
    test_f.add_argument("--alpha", type=float, default=0.05)
    test_f.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    test_2p = subparsers.add_parser("test-two-proportions", help="Hypothesis test for p1 - p2.")
    test_2p.add_argument("--successes1", type=int, required=True)
    test_2p.add_argument("--n1", type=int, required=True)
    test_2p.add_argument("--successes2", type=int, required=True)
    test_2p.add_argument("--n2", type=int, required=True)
    test_2p.add_argument("--diff0", type=float, default=0.0)
    test_2p.add_argument("--alpha", type=float, default=0.05)
    test_2p.add_argument("--tail", choices=sorted(TAILS), default="two-sided")

    gof = subparsers.add_parser("chi-square-gof", help="Chi-square goodness-of-fit test.")
    gof.add_argument("--observed", nargs="+", required=True)
    gof.add_argument("--probabilities", nargs="+", required=True)
    gof.add_argument("--estimated-parameters", type=int, default=0)
    gof.add_argument("--alpha", type=float, default=0.05)

    indep = subparsers.add_parser("chi-square-independence", help="Chi-square independence test.")
    indep.add_argument("--rows", nargs="+", required=True, help="Rows like '10,20,30' '6,12,18'.")
    indep.add_argument("--alpha", type=float, default=0.05)

    reg = subparsers.add_parser("regression", help="Simple linear regression.")
    reg.add_argument("--x", nargs="+", required=True)
    reg.add_argument("--y", nargs="+", required=True)
    reg.add_argument("--confidence", type=float, default=0.95)

    binomial = subparsers.add_parser("binomial", help="Binomial probability P(X = k).")
    binomial.add_argument("--n", type=int, required=True)
    binomial.add_argument("--k", type=int, required=True)
    binomial.add_argument("--p", type=float, required=True)

    poisson = subparsers.add_parser("poisson", help="Poisson probability P(X = k).")
    poisson.add_argument("--lambda", dest="lam", type=float, required=True)
    poisson.add_argument("--k", type=int, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        print_mapping(descriptive_stats(parse_values(args.values)))
    elif args.command == "describe-grouped":
        print_mapping(grouped_descriptive_stats(parse_values(args.centers), parse_values(args.counts)))
    elif args.command == "normal":
        print(f"probability: {normal_probability(args.lower, args.upper, args.mean, args.std):.6g}")
    elif args.command == "ci-mean":
        print_interval(confidence_interval_mean(parse_values(args.values), args.confidence, args.population_std, args.large_sample))
    elif args.command == "ci-proportion":
        print_interval(confidence_interval_proportion(args.successes, args.n, args.confidence))
    elif args.command == "ci-variance":
        if args.values:
            print_interval(confidence_interval_variance(parse_values(args.values), args.confidence))
        else:
            print_interval(confidence_interval_variance_summary(args.sample_variance, args.n, args.confidence))
    elif args.command == "ci-two-means":
        if args.case == "known":
            print_interval(confidence_interval_two_means_known(args.mean1, args.mean2, args.std1, args.std2, args.n1, args.n2, args.confidence))
        elif args.case == "equal-variance":
            print_interval(confidence_interval_two_means_equal_variance(args.mean1, args.mean2, args.std1, args.std2, args.n1, args.n2, args.confidence))
        else:
            print_interval(confidence_interval_two_means_large(args.mean1, args.mean2, args.std1, args.std2, args.n1, args.n2, args.confidence))
    elif args.command == "ci-paired":
        print_interval(confidence_interval_paired_mean(parse_values(args.before), parse_values(args.after), args.confidence))
    elif args.command == "ci-two-proportions":
        print_interval(confidence_interval_two_proportions(args.successes1, args.n1, args.successes2, args.n2, args.confidence))
    elif args.command == "ci-variance-ratio":
        print_interval(confidence_interval_variance_ratio(args.variance1, args.variance2, args.n1, args.n2, args.confidence))
    elif args.command == "test-mean":
        print_test(test_mean(parse_values(args.values), args.mu0, args.alpha, args.tail, args.sigma, args.large_sample))
    elif args.command == "test-proportion":
        print_test(z_test_proportion(args.successes, args.n, args.p0, args.alpha, args.tail))
    elif args.command == "test-two-means":
        if args.case == "known":
            print_test(z_test_two_means_known(args.mean1, args.mean2, args.std1, args.std2, args.n1, args.n2, args.diff0, args.alpha, args.tail))
        elif args.case == "equal-variance":
            print_test(t_test_two_means_equal_variance(args.mean1, args.mean2, args.std1, args.std2, args.n1, args.n2, args.diff0, args.alpha, args.tail))
        else:
            print_test(z_test_two_means_large(args.mean1, args.mean2, args.std1, args.std2, args.n1, args.n2, args.diff0, args.alpha, args.tail))
    elif args.command == "test-paired":
        print_test(paired_t_test(parse_values(args.before), parse_values(args.after), args.diff0, args.alpha, args.tail))
    elif args.command == "test-variance":
        print_test(chi_square_test_variance(args.sample_variance, args.n, args.variance0, args.alpha, args.tail))
    elif args.command == "test-variances":
        print_test(f_test_variances(args.variance1, args.variance2, args.n1, args.n2, args.ratio0, args.alpha, args.tail))
    elif args.command == "test-two-proportions":
        print_test(z_test_two_proportions(args.successes1, args.n1, args.successes2, args.n2, args.diff0, args.alpha, args.tail))
    elif args.command == "chi-square-gof":
        print_test(chi_square_goodness_of_fit(parse_values(args.observed), parse_values(args.probabilities), args.alpha, args.estimated_parameters))
    elif args.command == "chi-square-independence":
        table = [parse_values([row]) for row in args.rows]
        print_test(chi_square_independence(table, args.alpha))
    elif args.command == "regression":
        result = simple_linear_regression(parse_values(args.x), parse_values(args.y), args.confidence)
        print(f"intercept: {result.intercept:.6g}")
        print(f"slope: {result.slope:.6g}")
        print(f"r: {result.r:.6g}")
        print(f"r_squared: {result.r_squared:.6g}")
        print(f"residual_variance: {result.residual_variance:.6g}")
        print("slope_interval:")
        print_interval(result.slope_interval)
        print("intercept_interval:")
        print_interval(result.intercept_interval)
    elif args.command == "binomial":
        print(f"probability: {binomial_probability(args.n, args.k, args.p):.6g}")
    elif args.command == "poisson":
        print(f"probability: {poisson_probability(args.lam, args.k):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
