import pytest
import numpy as np

from scipy.stats import skew
from scipy.stats import kurtosis as scipy_kurtosis
from asset_pricing_lab.statistics import kurtosis

from asset_pricing_lab.statistics import (
    mean_return,
    variance,
    standard_deviation,
    covariance,
    correlation,
    skewness,
    covariance_matrix,
    correlation_matrix,
    adjusted_r_squared,
)

def test_mean_return():
    returns = np.array([0.02, 0.01, 0.04, 0.03])

    expected = 0.025

    assert np.isclose(mean_return(returns), expected)


def test_sample_variance():
    returns = np.array([1, 2, 3, 4, 5], dtype=float)

    expected = np.var(returns, ddof=1)

    assert np.isclose(variance(returns), expected)


def test_population_variance():
    returns = np.array([1, 2, 3, 4, 5], dtype=float)

    expected = np.var(returns, ddof=0)

    assert np.isclose(
        variance(returns, ddof=0),
        expected,
    )


def test_variance_empty_array():
    with pytest.raises(ValueError):
        variance([])


def test_variance_insufficient_observations():
    with pytest.raises(ValueError):
        variance([0.05], ddof=1)

def test_sample_standard_deviation():
    returns = np.array([1, 2, 3, 4, 5], dtype=float)

    expected = np.std(returns, ddof=1)

    assert np.isclose(
        standard_deviation(returns),
        expected,
    )


def test_population_standard_deviation():
    returns = np.array([1, 2, 3, 4, 5], dtype=float)

    expected = np.std(returns, ddof=0)

    assert np.isclose(
        standard_deviation(returns, ddof=0),
        expected,
    )

def test_sample_covariance():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    expected = np.cov(x, y, ddof=1)[0, 1]

    assert np.isclose(covariance(x, y), expected)


def test_population_covariance():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    expected = np.cov(x, y, ddof=0)[0, 1]

    assert np.isclose(
        covariance(x, y, ddof=0),
        expected,
    )


def test_covariance_different_lengths():
    with pytest.raises(ValueError):
        covariance([1, 2, 3], [1, 2])


def test_covariance_empty_array():
    with pytest.raises(ValueError):
        covariance([], [])


def test_covariance_insufficient_observations():
    with pytest.raises(ValueError):
        covariance([1], [2], ddof=1)

def test_sample_correlation():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)

    expected = np.corrcoef(x, y)[0, 1]

    assert np.isclose(correlation(x, y), expected)


def test_negative_correlation():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 4, 3, 2, 1], dtype=float)

    assert np.isclose(correlation(x, y), -1.0)


def test_zero_variance():
    with pytest.raises(ValueError):
        correlation([1, 1, 1], [2, 3, 4])

def test_skewness():
    returns = np.array([1, 2, 3, 4, 10], dtype=float)

    expected = skew(returns, bias=False)

    assert np.isclose(
        skewness(returns),
        expected,
    )

def test_skewness_insufficient_data():
    with pytest.raises(ValueError):
        skewness([1, 2])

def test_kurtosis():
    returns = np.array([1, 2, 3, 4, 10], dtype=float)

    expected = scipy_kurtosis(
        returns,
        bias=False,
        fisher=True,
    )

    assert np.isclose(
        kurtosis(returns),
        expected,
    )


def test_kurtosis_insufficient_data():
    with pytest.raises(ValueError):
        kurtosis([1, 2, 3])

def test_covariance_matrix():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    expected = np.cov(
        returns,
        rowvar=False,
        ddof=1,
    )

    assert np.allclose(
        covariance_matrix(returns),
        expected,
    )

def test_covariance_matrix():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    expected = np.cov(
        returns,
        rowvar=False,
        ddof=1,
    )

    assert np.allclose(
        covariance_matrix(returns),
        expected,
    )

def test_covariance_matrix_is_symmetric():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    cov = covariance_matrix(returns)

    assert np.allclose(cov, cov.T)

def test_covariance_matrix_diagonal():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    cov = covariance_matrix(returns)

    assert np.isclose(
        cov[0, 0],
        variance(returns[:, 0]),
    )

    assert np.isclose(
        cov[1, 1],
        variance(returns[:, 1]),
    )

def test_covariance_matrix_empty():
    with pytest.raises(ValueError):
        covariance_matrix([])

def test_covariance_matrix_one_dimensional():
    with pytest.raises(ValueError):
        covariance_matrix([1, 2, 3])

def test_covariance_matrix_insufficient_observations():
    with pytest.raises(ValueError):
        covariance_matrix([[1, 2]], ddof=1)

def test_correlation_matrix():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    expected = np.corrcoef(
        returns,
        rowvar=False,
    )

    assert np.allclose(
        correlation_matrix(returns),
        expected,
    )

def test_correlation_matrix_is_symmetric():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    corr = correlation_matrix(returns)

    assert np.allclose(corr, corr.T)

def test_correlation_matrix_diagonal():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    corr = correlation_matrix(returns)

    assert np.allclose(
        np.diag(corr),
        np.ones(corr.shape[0]),
    )

def test_correlation_matrix_range():
    returns = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
    ], dtype=float)

    corr = correlation_matrix(returns)

    assert np.all(corr <= 1.0)
    assert np.all(corr >= -1.0)

def test_correlation_matrix_empty():
    with pytest.raises(ValueError):
        correlation_matrix([])

def test_correlation_matrix_one_dimensional():
    with pytest.raises(ValueError):
        correlation_matrix([1, 2, 3])

from asset_pricing_lab.statistics import adjusted_r_squared

def test_adjusted_r_squared():
    result = adjusted_r_squared(
        r_squared=0.85,
        n_observations=100,
        n_predictors=3,
    )

    assert np.isclose(result, 0.8453125)

def test_adjusted_r_squared_perfect():
    result = adjusted_r_squared(
        r_squared=1.0,
        n_observations=50,
        n_predictors=3,
    )

    assert result == 1.0

def test_adjusted_r_squared_invalid_r2():
    with pytest.raises(ValueError):
        adjusted_r_squared(
            r_squared=1.2,
            n_observations=100,
            n_predictors=3,
        )

def test_adjusted_r_squared_invalid_sample_size():
    with pytest.raises(ValueError):
        adjusted_r_squared(
            r_squared=0.8,
            n_observations=4,
            n_predictors=3,
        )

