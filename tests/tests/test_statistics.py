import pytest
import numpy as np

from asset_pricing_lab.statistics import (
    mean_return,
    variance,
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