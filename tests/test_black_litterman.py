import numpy as np

import pytest

from asset_pricing_lab.black_litterman import (
    implied_equilibrium_returns,
    validate_pick_matrix,
    validate_views,
    build_pick_matrix,
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

