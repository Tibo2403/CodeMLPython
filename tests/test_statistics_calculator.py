import math

from statistics_calculator import (
    binomial_probability,
    chi_square_goodness_of_fit,
    chi_square_independence,
    chi_square_test_variance,
    confidence_interval_mean,
    confidence_interval_proportion,
    confidence_interval_two_means_equal_variance,
    confidence_interval_two_proportions,
    confidence_interval_variance_summary,
    descriptive_stats,
    f_test_variances,
    normal_probability,
    poisson_probability,
    student_t_cdf,
    t_test_two_means_equal_variance,
    z_test_mean,
    z_test_proportion,
    z_test_two_proportions,
)


def test_descriptive_stats():
    stats = descriptive_stats([10, 12, 14, 16])
    assert stats["n"] == 4
    assert stats["mean"] == 13
    assert stats["median"] == 13
    assert stats["range"] == 6
    assert math.isclose(stats["variance_sample"], 6.6666666667)


def test_normal_probability_between_minus_one_and_one():
    probability = normal_probability(lower=-1, upper=1)
    assert math.isclose(probability, 0.682689, rel_tol=1e-5)


def test_confidence_interval_mean_uses_student_table():
    interval = confidence_interval_mean([10, 12, 14, 16], confidence=0.95)
    assert math.isclose(interval.center, 13)
    assert interval.lower < interval.center < interval.upper
    assert math.isclose(interval.margin, 4.10794, rel_tol=1e-3)


def test_confidence_interval_proportion():
    interval = confidence_interval_proportion(successes=45, n=100, confidence=0.95)
    assert interval.lower < 0.45 < interval.upper
    assert math.isclose(interval.margin, 0.09745, rel_tol=1e-3)


def test_z_test_mean_two_sided():
    result = z_test_mean(
        sample_mean=52,
        hypothesized_mean=50,
        std=5,
        n=25,
        alpha=0.05,
    )
    assert math.isclose(result.statistic, 2.0)
    assert math.isclose(result.p_value, 0.0455, rel_tol=1e-3)
    assert result.reject_h0 is True


def test_discrete_probabilities():
    assert math.isclose(binomial_probability(n=10, k=3, probability=0.2), 0.201326592)
    assert math.isclose(poisson_probability(lam=2, k=3), 0.1804470443)


def test_student_distribution_is_symmetric():
    assert math.isclose(student_t_cdf(0, 10), 0.5)
    assert math.isclose(student_t_cdf(1.812, 10), 0.95, rel_tol=2e-3)


def test_proportion_tests_and_interval():
    result = z_test_proportion(successes=55, n=100, hypothesized_p=0.50)
    assert math.isclose(result.statistic, 1.0)
    assert result.reject_h0 is False

    interval = confidence_interval_two_proportions(55, 100, 45, 100)
    assert interval.lower < 0.10 < interval.upper

    two_sample = z_test_two_proportions(55, 100, 45, 100)
    assert math.isclose(two_sample.statistic, 1.414213562, rel_tol=1e-6)


def test_two_mean_equal_variance_case():
    interval = confidence_interval_two_means_equal_variance(
        mean1=12,
        mean2=10,
        std1=2,
        std2=3,
        n1=15,
        n2=12,
    )
    assert interval.lower < 2 < interval.upper

    result = t_test_two_means_equal_variance(
        mean1=12,
        mean2=10,
        std1=2,
        std2=3,
        n1=15,
        n2=12,
    )
    assert result.distribution == "t(25)"


def test_variance_and_chi_square_cases():
    interval = confidence_interval_variance_summary(sample_variance=4, n=20)
    assert interval.lower < 4 < interval.upper

    variance_test = chi_square_test_variance(
        sample_variance=4,
        n=20,
        hypothesized_variance=4,
    )
    assert math.isclose(variance_test.statistic, 19)
    assert variance_test.reject_h0 is False

    f_test = f_test_variances(variance1=9, variance2=4, n1=12, n2=10)
    assert f_test.statistic == 2.25


def test_chi_square_tests():
    gof = chi_square_goodness_of_fit([25, 25, 50], [0.25, 0.25, 0.5])
    assert math.isclose(gof.statistic, 0)
    assert gof.reject_h0 is False

    independence = chi_square_independence([[10, 20], [20, 40]])
    assert math.isclose(independence.statistic, 0)
    assert independence.reject_h0 is False
