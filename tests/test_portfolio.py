
import numpy as np

import pytest

from asset_pricing_lab.portfolio import (
    correlation_matrix,
    maximum_sharpe_portfolio,
    portfolio_expected_return,
    portfolio_variance,
    portfolio_volatility,
    covariance_matrix,
    diversification_ratio,
    simulate_portfolios,
    global_minimum_variance_portfolio,
    efficient_frontier,
    project_weights,
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

def test_diversification_ratio():
    weights = np.array([0.5, 0.5])

    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = diversification_ratio(
        weights,
        covariance,
    )

    expected = 0.25 / np.sqrt(0.0375)

    assert np.isclose(result, expected)

def test_diversification_ratio_invalid_shape():
    with pytest.raises(ValueError):
        diversification_ratio(
            np.array([0.5, 0.5]),
            np.array([[0.04, 0.01]]),
        )

def test_diversification_ratio_zero_volatility():
    covariance = np.zeros((2, 2))

    with pytest.raises(ValueError):
        diversification_ratio(
            np.array([0.5, 0.5]),
            covariance,
        )

def test_simulate_portfolios_shapes():

    expected_returns = np.array(
        [0.08,0.10,0.12]
    )

    covariance = np.eye(3)

    result = simulate_portfolios(
        expected_returns,
        covariance,
        n_portfolios=100,
        random_state=42,
    )

    assert result["weights"].shape == (
        100,
        3,
    )

    assert result["returns"].shape == (
        100,
    )

    assert result["volatility"].shape == (
        100,
    )

def test_simulated_weights_sum_to_one():

    result = simulate_portfolios(
        np.array([0.08,0.10]),

        np.eye(2),

        n_portfolios=50,

        random_state=42,
    )

    assert np.allclose(
        result["weights"].sum(axis=1),
        np.ones(50),
    )

def test_simulate_portfolios_invalid_covariance():

    with pytest.raises(ValueError):

        simulate_portfolios(

            np.array([0.08,0.10]),

            np.eye(3),
        )

def test_simulate_portfolios_invalid_count():

    with pytest.raises(ValueError):

        simulate_portfolios(

            np.array([0.08]),

            np.eye(1),

            n_portfolios=0,
        )

def test_gmvp_weights_sum_to_one():
    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = global_minimum_variance_portfolio(
        covariance,
    )

    assert np.isclose(
        result["weights"].sum(),
        1.0,
    )

def test_gmvp_output_keys():
    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = global_minimum_variance_portfolio(
        covariance,
    )

    assert set(result.keys()) == {
        "weights",
        "variance",
        "volatility",
    }

import pytest

def test_gmvp_singular_matrix():
    covariance = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(ValueError):
        global_minimum_variance_portfolio(
            covariance,
        )

def test_gmvp_non_square():
    covariance = np.ones((2, 3))

    with pytest.raises(ValueError):
        global_minimum_variance_portfolio(
            covariance,
        )

def test_maximum_sharpe_weights_sum_to_one():
    expected_returns = np.array([0.08, 0.12])

    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = maximum_sharpe_portfolio(
        expected_returns,
        covariance,
    )

    assert np.isclose(
        result["weights"].sum(),
        1.0,
    )

def test_maximum_sharpe_output_keys():
    expected_returns = np.array([0.08, 0.12])

    covariance = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])

    result = maximum_sharpe_portfolio(
        expected_returns,
        covariance,
    )

    assert set(result.keys()) == {
        "weights",
        "expected_return",
        "variance",
        "volatility",
        "sharpe_ratio",
    }

def test_maximum_sharpe_singular_matrix():
    covariance = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(
            np.array([0.08, 0.10]),
            covariance,
        )

def test_maximum_sharpe_invalid_shape():
    with pytest.raises(ValueError):
        maximum_sharpe_portfolio(
            np.array([0.08, 0.10]),
            np.eye(3),
        )

def test_efficient_frontier_shapes():
    result = efficient_frontier(
        expected_returns=np.array([0.08, 0.10]),
        covariance_matrix=np.eye(2),
        n_portfolios=500,
        n_bins=20,
        random_state=42,
    )

    assert result["weights"].ndim == 2
    assert result["returns"].ndim == 1
    assert result["volatility"].ndim == 1

    assert (
        len(result["weights"])
        ==
        len(result["returns"])
        ==
        len(result["volatility"])
    )

def test_frontier_weights_sum_to_one():
    result = efficient_frontier(
        np.array([0.08, 0.10]),
        np.eye(2),
        n_portfolios=500,
        n_bins=20,
        random_state=42,
    )

    assert np.allclose(
        result["weights"].sum(axis=1),
        1.0,
    )

def test_efficient_frontier_invalid_bins():
    with pytest.raises(ValueError):
        efficient_frontier(
            np.array([0.08]),
            np.eye(1),
            n_bins=0,
        )

def test_efficient_frontier_reproducible():
    result1 = efficient_frontier(
        np.array([0.08, 0.10]),
        np.eye(2),
        random_state=123,
    )

    result2 = efficient_frontier(
        np.array([0.08, 0.10]),
        np.eye(2),
        random_state=123,
    )

    assert np.allclose(
        result1["returns"],
        result2["returns"],
    )

    assert np.allclose(
        result1["volatility"],
        result2["volatility"],
    )

def test_project_weights_long_only():
    weights = np.array([
        -0.2,
        0.5,
        0.7,
    ])

    result = project_weights(weights)

    assert np.all(result >= 0)

    assert np.isclose(
        result.sum(),
        1.0,
    )

def test_project_weights_already_feasible():
    weights = np.array([
        0.2,
        0.5,
        0.3,
    ])

    result = project_weights(weights)

    assert np.allclose(
        result,
        weights,
    )

def test_project_weights_upper_bound():
    weights = np.array([
        0.8,
        0.1,
        0.1,
    ])

    result = project_weights(
        weights,
        upper_bound=0.6,
    )

    assert np.all(result >= 0)

    assert np.isclose(
        result.sum(),
        1.0,
    )