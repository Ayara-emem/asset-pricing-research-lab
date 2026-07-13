import numpy as np

import pytest

from asset_pricing_lab.black_litterman import (
    implied_equilibrium_returns,
    validate_pick_matrix,
    validate_views,
    build_pick_matrix,
    validate_omega,
    default_omega,
    posterior_expected_returns,
    black_litterman_portfolio,
    black_litterman_report,
)

def test_implied_equilibrium_returns():
    weights = np.array([
        0.5,
        0.3,
        0.2,
    ])

    covariance = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.09, 0.02],
        [0.00, 0.02, 0.16],
    ])

    result = implied_equilibrium_returns(
        weights,
        covariance,
        risk_aversion=3.0,
    )

    expected = np.array([
        0.069,
        0.108,
        0.114,
    ])

    assert np.allclose(
        result,
        expected,
    )

def test_implied_equilibrium_returns_shape():
    with pytest.raises(ValueError):
        implied_equilibrium_returns(
            np.array([0.5, 0.5]),
            np.eye(3),
            3.0,
        )

def test_implied_equilibrium_returns_invalid_lambda():
    with pytest.raises(ValueError):
        implied_equilibrium_returns(
            np.array([1.0]),
            np.eye(1),
            0.0,
        )

def test_implied_equilibrium_returns_weight_dimension():
    with pytest.raises(ValueError):
        implied_equilibrium_returns(
            np.array([[1.0]]),
            np.eye(1),
            3.0,
        )


def test_validate_pick_matrix():
    P = np.array([
        [1, -1, 0],
        [0, 1, -1],
    ])

    result = validate_pick_matrix(
        P,
        n_assets=3,
    )

    assert np.allclose(result, P)

def test_validate_pick_matrix_invalid_columns():
    with pytest.raises(ValueError):
        validate_pick_matrix(
            np.ones((2,4)),
            n_assets=3,
        )

def test_validate_views():
    views = np.array([
        0.02,
        0.01,
    ])

    result = validate_views(
        views,
        n_views=2,
    )

    assert np.allclose(
        result,
        views,
    )

def test_validate_views_length():
    with pytest.raises(ValueError):
        validate_views(
            np.array([0.02]),
            n_views=2,
        )


def test_build_pick_matrix():
    P = build_pick_matrix(
        [
            [1,-1,0],
            [0,1,-1],
        ]
    )

    assert P.shape == (
        2,
        3,
    )

def test_default_omega():

    omega = default_omega(
        3
    )

    assert omega.shape == (
        3,
        3,
    )

    assert np.allclose(
        omega,
        omega.T,
    )

def test_default_omega_diagonal():

    omega = default_omega(
        4,
        uncertainty=0.10,
    )

    expected = (
        0.10
        * np.eye(4)
    )

    assert np.allclose(
        omega,
        expected,
    )

def test_validate_omega():

    omega = (
        0.05
        * np.eye(2)
    )

    result = validate_omega(
        omega,
        2,
    )

    assert np.allclose(
        result,
        omega,
    )

def test_validate_omega_shape():

    with pytest.raises(
        ValueError
    ):

        validate_omega(
            np.eye(3),
            2,
        )

def test_validate_omega_not_symmetric():

    omega = np.array([
        [1,2],
        [0,1],
    ])

    with pytest.raises(
        ValueError
    ):

        validate_omega(
            omega,
            2,
        )

def test_validate_omega_negative_diagonal():

    omega = np.array([
        [-1,0],
        [0,1],
    ])

    with pytest.raises(
        ValueError
    ):

        validate_omega(
            omega,
            2,
        )

def test_posterior_expected_returns_shape():

    pi = np.array([
        0.08,
        0.09,
    ])

    sigma = np.eye(2)

    P = np.array([
        [1,-1],
    ])

    Q = np.array([
        0.02,
    ])

    omega = (
        0.05
        * np.eye(1)
    )

    result = posterior_expected_returns(
        pi,
        sigma,
        P,
        Q,
        omega,
    )

    assert result.shape == (
        2,
    )

def test_posterior_equals_prior_when_views_match():

    pi = np.array([
        0.08,
        0.09,
    ])

    sigma = np.eye(2)

    P = np.array([
        [1,-1],
    ])

    Q = P @ pi

    omega = (
        0.05
        * np.eye(1)
    )

    result = posterior_expected_returns(
        pi,
        sigma,
        P,
        Q,
        omega,
    )

    assert np.allclose(
        result,
        pi,
    )

def test_posterior_invalid_tau():

    with pytest.raises(
        ValueError
    ):

        posterior_expected_returns(

            np.array([0.08]),

            np.eye(1),

            np.array([[1]]),

            np.array([0.08]),

            np.eye(1),

            tau=0,
        )

def test_posterior_covariance_shape():

    with pytest.raises(
        ValueError
    ):

        posterior_expected_returns(

            np.array([0.08,0.09]),

            np.eye(3),

            np.array([[1,-1]]),

            np.array([0.02]),

            np.eye(1),
        )

def test_black_litterman_portfolio_keys():

    covariance = np.eye(2)

    result = black_litterman_portfolio(

        market_weights=np.array([0.5,0.5]),

        covariance_matrix=covariance,

        risk_aversion=3,

        pick_matrix=np.array([[1,-1]]),

        views=np.array([0.02]),

        omega=0.05*np.eye(1),
    )

    assert set(result.keys()) == {

        "equilibrium_returns",

        "posterior_returns",

        "weights",

        "expected_return",

        "variance",

        "volatility",

        "sharpe_ratio",
    }

def test_black_litterman_weights_sum():

    covariance = np.eye(2)

    result = black_litterman_portfolio(

        np.array([0.5,0.5]),

        covariance,

        3,

        np.array([[1,-1]]),

        np.array([0.02]),

        0.05*np.eye(1),
    )

    assert np.isclose(

        result["weights"].sum(),

        1,
    )

def test_black_litterman_report():

    covariance = np.eye(2)

    result = black_litterman_portfolio(

        np.array([0.5,0.5]),

        covariance,

        3,

        np.array([[1,-1]]),

        np.array([0.02]),

        0.05*np.eye(1),
    )

    report = black_litterman_report(
        result
    )

    assert "Sharpe Ratio" in report

    assert "Weights" in report


