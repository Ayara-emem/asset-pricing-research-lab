import numpy as np

import pytest

from asset_pricing_lab.capm import (
    beta,
    alpha,
    capm_expected_return,
)


def test_beta():
    asset = np.array([
        0.03,
        0.02,
        -0.01,
        0.01,
    ])

    market = np.array([
        0.02,
        0.01,
        -0.02,
        0.00,
    ])

    b = beta(
        asset,
        market,
    )

    assert np.isfinite(b)

def test_beta_equals_one():
    market = np.array([
        0.01,
        0.03,
        -0.02,
        0.04,
    ])

    b = beta(
        market,
        market,
    )

    assert np.isclose(
        b,
        1.0,
    )

def test_beta_zero_market_variance():
    asset = np.array([
        0.01,
        0.02,
        0.03,
    ])

    market = np.array([
        0.02,
        0.02,
        0.02,
    ])

    b = beta(
        asset,
        market,
    )

    assert np.isnan(b)


def test_beta_shape_mismatch():
    asset = np.array([
        0.01,
        0.02,
    ])

    market = np.array([
        0.01,
    ])

    with pytest.raises(ValueError):
        beta(
            asset,
            market,
        )

def test_beta_empty():
    with pytest.raises(ValueError):
        beta([], [])

def test_beta_negative():
    market = np.array([
        0.01,
        0.02,
        -0.01,
        -0.02,
    ])

    asset = -market

    b = beta(
        asset,
        market,
    )

    assert np.isclose(
        b,
        -1.0,
    )

def test_alpha_positive():
    a = alpha(
        asset_return=0.12,
        market_return=0.08,
        risk_free_rate=0.02,
        beta=1.0,
    )

    assert np.isclose(
        a,
        0.04,
    )

def test_alpha_zero():
    a = alpha(
        asset_return=0.08,
        market_return=0.08,
        risk_free_rate=0.02,
        beta=1.0,
    )

    assert np.isclose(
        a,
        0.0,
    )

def test_alpha_negative():
    a = alpha(
        asset_return=0.05,
        market_return=0.08,
        risk_free_rate=0.02,
        beta=1.0,
    )

    assert a < 0

def test_alpha_beta_zero():
    a = alpha(
        asset_return=0.05,
        market_return=0.20,
        risk_free_rate=0.02,
        beta=0.0,
    )

    assert np.isclose(
        a,
        0.03,
    )

def test_alpha_negative_beta():
    a = alpha(
        asset_return=0.10,
        market_return=0.08,
        risk_free_rate=0.02,
        beta=-1.0,
    )

    expected = 0.10 - (
        0.02
        + (-1.0)
        * (0.08 - 0.02)
    )

    assert np.isclose(
        a,
        expected,
    )

def test_capm_expected_return():
    expected = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.5,
    )

    assert np.isclose(
        expected,
        0.11,
    )

def test_capm_expected_return_beta_zero():
    expected = capm_expected_return(
        risk_free_rate=0.03,
        market_return=0.12,
        beta=0,
    )

    assert np.isclose(
        expected,
        0.03,
    )

def test_capm_expected_return_beta_one():
    expected = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.09,
        beta=1,
    )

    assert np.isclose(
        expected,
        0.09,
    )

def test_capm_expected_return_negative_beta():
    expected = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        beta=-1,
    )

    assert np.isclose(
        expected,
        -0.04,
    )

def test_alpha_matches_capm_expected_return():
    expected = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.2,
    )

    a = alpha(
        asset_return=expected,
        market_return=0.08,
        risk_free_rate=0.02,
        beta=1.2,
    )

    assert np.isclose(
        a,
        0.0,
    )

