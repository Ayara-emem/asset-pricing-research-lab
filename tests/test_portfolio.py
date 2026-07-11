
import numpy as np

import pytest

from asset_pricing_lab.portfolio import (
    correlation_matrix,
    portfolio_expected_return,
    portfolio_variance,
    portfolio_volatility,
    covariance_matrix,
)



def test_portfolio_expected_return():
    weights = np.array([0.5, 0.3, 0.2])
    returns = np.array([0.10, 0.08, 0.12])

    result = portfolio_expected_return(
        weights,
        returns,
    )

    assert np.isclose(result, 0.098)


import pytest

def test_portfolio_expected_return_shape_error():
    with pytest.raises(ValueError):
        portfolio_expected_return(
            np.array([0.5, 0.5]),
            np.array([0.10]),
        )

def test_portfolio_variance():
    weights = np.array([0.5, 0.5])

    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = portfolio_variance(
        weights,
        covariance,
    )

    assert np.isclose(result, 0.0375)

def test_portfolio_volatility():
    weights = np.array([0.5, 0.5])

    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = portfolio_volatility(
        weights,
        covariance,
    )

    expected = np.sqrt(0.0375)

    assert np.isclose(result, expected)

def test_portfolio_variance_invalid_shape():
    weights = np.array([0.5, 0.5])

    covariance = np.array([
        [0.04],
        [0.01],
    ])

    with pytest.raises(ValueError):
        portfolio_variance(
            weights,
            covariance,
        )

def test_portfolio_variance_not_matrix():
    weights = np.array([0.5, 0.5])

    covariance = np.array([0.04, 0.09])

    with pytest.raises(ValueError):
        portfolio_variance(
            weights,
            covariance,
        )

def test_covariance_matrix_shape():
    returns = np.array([
        [0.01, 0.02],
        [0.03, 0.01],
        [0.02, 0.04],
    ])

    cov = covariance_matrix(returns)

    assert cov.shape == (2, 2)

def test_covariance_matrix_is_symmetric():
    returns = np.array([
        [0.01, 0.02],
        [0.03, 0.01],
        [0.02, 0.04],
    ])

    cov = covariance_matrix(returns)

    assert np.allclose(cov, cov.T)

def test_correlation_matrix_diagonal():
    returns = np.array([
        [0.01, 0.02],
        [0.03, 0.01],
        [0.02, 0.04],
    ])
    corr = correlation_matrix(returns)

    assert np.allclose(
        np.diag(corr),
        np.ones(2),
    )

import pytest

def test_covariance_matrix_not_2d():
    with pytest.raises(ValueError):
        covariance_matrix(
            np.array([1, 2, 3]),
        )