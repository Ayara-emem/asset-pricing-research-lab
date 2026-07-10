import numpy as np

import pytest

from asset_pricing_lab.capm import (
    beta,
    alpha,
    capm_expected_return,
    estimate_capm,
    r_squared,
    residuals,
    security_selection,
    rolling_beta,
    treynor_ratio,
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

def test_security_selection_undervalued():
    result = security_selection(
        asset_return=0.10,
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.0,
    )

    assert result == "undervalued"

def test_security_selection_fair():
    expected = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.2,
    )

    result = security_selection(
        asset_return=expected,
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.2,
    )

    assert result == "fairly valued"

def test_security_selection_overvalued():
    result = security_selection(
        asset_return=0.04,
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.0,
    )

    assert result == "overvalued"

def test_security_selection_tolerance():
    expected = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.0,
    )

    result = security_selection(
        asset_return=expected + 1e-14,
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.0,
    )

    assert result == "fairly valued"

def test_residuals():
    actual = np.array([
        0.05,
        0.06,
        0.04,
    ])

    predicted = np.array([
        0.04,
        0.05,
        0.05,
    ])

    r = residuals(
        actual,
        predicted,
    )

    expected = np.array([
        0.01,
        0.01,
        -0.01,
    ])

    assert np.allclose(
        r,
        expected,
    )

def test_residuals_zero():
    x = np.array([
        0.01,
        0.02,
    ])

    r = residuals(
        x,
        x,
    )

    assert np.allclose(
        r,
        0
    )

def test_residuals_empty():
    with pytest.raises(ValueError):
        residuals([], [])

def test_residuals_shape_mismatch():
    with pytest.raises(ValueError):
        residuals(
            [1,2],
            [1],
        )
def test_r_squared_perfect():
    x = np.array([
        0.01,
        0.02,
        0.03,
    ])

    r2 = r_squared(
        x,
        x,
    )

    assert np.isclose(
        r2,
        1.0,
    )

def test_r_squared():
    actual = np.array([
        1,
        2,
        3,
    ])

    predicted = np.array([
        1,
        2,
        2,
    ])

    r2 = r_squared(
        actual,
        predicted,
    )

    assert 0 <= r2 <= 1

def test_r_squared_zero_variance():
    actual = np.array([
        2,
        2,
        2,
    ])

    predicted = np.array([
        2,
        2,
        2,
    ])

    r2 = r_squared(
        actual,
        predicted,
    )

    assert np.isnan(r2)

def test_estimate_capm_keys():
    asset = np.array([
        0.03,
        0.01,
        -0.01,
        0.02,
    ])

    market = np.array([
        0.02,
        0.01,
        -0.02,
        0.01,
    ])

    result = estimate_capm(
        asset,
        market,
    )

    assert "alpha" in result
    assert "beta" in result

def test_estimate_capm_beta_one():
    market = np.array([
        0.01,
        0.03,
        -0.02,
        0.04,
    ])

    result = estimate_capm(
        market,
        market,
    )

    assert np.isclose(
        result["beta"],
        1.0,
    )

    assert np.isclose(
        result["alpha"],
        0.0,
    )

def test_estimate_capm_zero_market_variance():
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

    result = estimate_capm(
        asset,
        market,
    )

    assert np.isnan(
        result["beta"]
    )

    assert np.isnan(
        result["alpha"]
    )

def test_estimate_capm_shape_mismatch():
    with pytest.raises(ValueError):
        estimate_capm(
            [0.01, 0.02],
            [0.01],
        )

def test_estimate_capm_empty():
    with pytest.raises(ValueError):
        estimate_capm([], [])

def test_rolling_beta():
    asset = np.array([
        0.02,
        0.03,
        0.01,
        0.04,
        0.05,
    ])

    market = np.array([
        0.01,
        0.02,
        0.00,
        0.03,
        0.04,
    ])

    result = rolling_beta(
        asset,
        market,
        window=3,
    )

    assert len(result) == 3

def test_rolling_beta_perfect():
    market = np.array([
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ])

    result = rolling_beta(
        market,
        market,
        window=3,
    )

    assert np.allclose(
        result,
        1.0,
    )

def test_rolling_beta_invalid_window():
    asset = np.array([
        0.01,
        0.02,
    ])

    market = np.array([
        0.01,
        0.02,
    ])

    with pytest.raises(ValueError):
        rolling_beta(
            asset,
            market,
            window=0,
        )

def test_rolling_beta_window_too_large():
    asset = np.array([
        0.01,
        0.02,
    ])

    market = np.array([
        0.01,
        0.02,
    ])

    with pytest.raises(ValueError):
        rolling_beta(
            asset,
            market,
            window=5,
        )

def test_rolling_beta_shape_mismatch():
    with pytest.raises(ValueError):
        rolling_beta(
            [0.01, 0.02],
            [0.01],
            window=2,
        )

def test_treynor_ratio():
    ratio = treynor_ratio(
        portfolio_return=0.12,
        risk_free_rate=0.02,
        beta=1.25,
    )

    assert np.isclose(
        ratio,
        0.08,
    )

def test_treynor_ratio_negative_beta():
    ratio = treynor_ratio(
        portfolio_return=0.08,
        risk_free_rate=0.02,
        beta=-1.0,
    )

    assert np.isclose(
        ratio,
        -0.06,
    )

def test_treynor_ratio_zero_beta():
    ratio = treynor_ratio(
        portfolio_return=0.10,
        risk_free_rate=0.02,
        beta=0.0,
    )

    assert np.isnan(
        ratio
    )

def test_treynor_ratio_negative_excess_return():
    ratio = treynor_ratio(
        portfolio_return=0.01,
        risk_free_rate=0.02,
        beta=1.0,
    )

    assert ratio < 0

def test_treynor_ratio_zero():
    ratio = treynor_ratio(
        portfolio_return=0.03,
        risk_free_rate=0.03,
        beta=2.0,
    )

    assert np.isclose(
        ratio,
        0.0,
    )

