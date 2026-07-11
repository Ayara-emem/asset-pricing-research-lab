import numpy as np
import pytest

from asset_pricing_lab.fama_french import (
    fama_french_expected_return,
    estimate_fama_french,
    predicted_returns_fama_french,
    residuals_fama_french,
    r_squared_fama_french,
    rolling_fama_french,
    factor_attribution,
    factor_exposure_report,
)


def test_fama_french_expected_return():
    expected = fama_french_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        smb=0.02,
        hml=0.03,
        market_beta=1.1,
        smb_beta=0.4,
        hml_beta=-0.2,
    )

    assert np.isclose(
        expected,
        0.088,
    )

def test_fama_french_zero_betas():
    expected = fama_french_expected_return(
        risk_free_rate=0.03,
        market_return=0.10,
        smb=0.02,
        hml=0.01,
        market_beta=0.0,
        smb_beta=0.0,
        hml_beta=0.0,
    )

    assert np.isclose(
        expected,
        0.03,
    )

from asset_pricing_lab.capm import (
    capm_expected_return,
)


def test_fama_french_reduces_to_capm():
    ff = fama_french_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        smb=0.00,
        hml=0.00,
        market_beta=1.3,
        smb_beta=0.0,
        hml_beta=0.0,
    )

    capm = capm_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        beta=1.3,
    )

    assert np.isclose(
        ff,
        capm,
    )

def test_fama_french_negative_factor_loading():
    expected = fama_french_expected_return(
        risk_free_rate=0.02,
        market_return=0.08,
        smb=0.03,
        hml=0.02,
        market_beta=1.0,
        smb_beta=-0.5,
        hml_beta=0.5,
    )

    assert np.isclose(
        expected,
        0.075,
    )

def test_estimate_fama_french_keys():
    asset = np.array([0.05, 0.04, 0.06, 0.05])
    market = np.array([0.04, 0.03, 0.05, 0.04])
    smb = np.array([0.01, 0.00, -0.01, 0.02])
    hml = np.array([0.02, -0.01, 0.00, 0.01])

    result = estimate_fama_french(
        asset,
        market,
        smb,
        hml,
    )

    assert "alpha" in result
    assert "market_beta" in result
    assert "smb_beta" in result
    assert "hml_beta" in result

def test_estimate_fama_french_perfect():
    np.random.seed(42)

    n = 500

    market = np.random.normal(0.05, 0.02, n)
    smb = np.random.normal(0.00, 0.01, n)
    hml = np.random.normal(0.00, 0.01, n)

    alpha = 0.01
    bm = 1.2
    bs = 0.5
    bh = -0.3

    asset = (
        alpha
        + bm * market
        + bs * smb
        + bh * hml
    )

    result = estimate_fama_french(
        asset,
        market,
        smb,
        hml,
    )

    assert np.isclose(result["alpha"], alpha)
    assert np.isclose(result["market_beta"], bm)
    assert np.isclose(result["smb_beta"], bs)
    assert np.isclose(result["hml_beta"], bh)

def test_estimate_fama_french_shape_mismatch():
    with pytest.raises(ValueError):
        estimate_fama_french(
            [0.01, 0.02],
            [0.01],
            [0.0, 0.0],
            [0.0, 0.0],
        )

def test_estimate_fama_french_empty():
    with pytest.raises(ValueError):
        estimate_fama_french(
            [],
            [],
            [],
            [],
        )

def test_predicted_returns_fama_french():
    market = np.array([
        0.05,
        0.06,
    ])

    smb = np.array([
        0.01,
        0.02,
    ])

    hml = np.array([
        0.03,
        0.01,
    ])

    predicted = predicted_returns_fama_french(
        market_returns=market,
        smb=smb,
        hml=hml,
        alpha=0.01,
        market_beta=1,
        smb_beta=1,
        hml_beta=1,
    )

    expected = np.array([
        0.10,
        0.10,
    ])

    assert np.allclose(
        predicted,
        expected,
    )

def test_predicted_returns_zero_betas():
    market = np.array([
        0.02,
        0.03,
    ])

    smb = np.array([
        0.01,
        0.02,
    ])

    hml = np.array([
        0.02,
        0.03,
    ])

    predicted = predicted_returns_fama_french(
        market,
        smb,
        hml,
        alpha=0.05,
        market_beta=0,
        smb_beta=0,
        hml_beta=0,
    )

    assert np.allclose(
        predicted,
        0.05,
    )

def test_predicted_returns_negative_beta():
    market = np.array([
        0.04,
    ])

    smb = np.array([
        0.02,
    ])

    hml = np.array([
        0.01,
    ])

    predicted = predicted_returns_fama_french(
        market,
        smb,
        hml,
        alpha=0,
        market_beta=1,
        smb_beta=-1,
        hml_beta=0,
    )

    assert np.allclose(
        predicted,
        0.02,
    )

def test_predicted_returns_shape_mismatch():
    with pytest.raises(ValueError):
        predicted_returns_fama_french(
            [0.01],
            [0.01, 0.02],
            [0.01],
            0,
            1,
            1,
            1,
            1,
        )

def test_fama_french_residuals():
    actual = np.array([0.05, 0.06, 0.04])
    predicted = np.array([0.04, 0.05, 0.05])

    result = residuals_fama_french(actual, predicted)

    expected = np.array([0.01, 0.01, -0.01])

    assert np.allclose(result, expected)

def test_fama_french_residuals_shape_error():
    actual = np.array([0.05, 0.06])
    predicted = np.array([0.04])

    import pytest

    with pytest.raises(ValueError):
        residuals_fama_french(actual, predicted)

def test_fama_french_r_squared_perfect():
    actual = np.array([1, 2, 3, 4])

    result = r_squared_fama_french(actual, actual)

    assert np.isclose(result, 1.0)

actual = [1, 2, 3, 4]
predicted = [1.1, 1.9, 3.2, 3.8]

def test_fama_french_r_squared():
    actual = np.array([1, 2, 3, 4])
    predicted = np.array([1.1, 1.9, 3.2, 3.8])

    result = r_squared_fama_french(actual, predicted)

    assert np.isclose(result, 0.98)


def test_fama_french_r_squared_zero_variance():
    actual = np.array([2.0, 2.0, 2.0])
    predicted = np.array([2.0, 2.0, 2.0])

    result = r_squared_fama_french(actual, predicted)

    assert result == 1.0

def test_rolling_fama_french_output_length():
    import numpy as np

    n = 10
    window = 5

    asset = np.linspace(0.01, 0.10, n)
    market = np.linspace(0.02, 0.11, n)
    smb = np.linspace(-0.01, 0.01, n)
    hml = np.linspace(0.03, -0.02, n)

    result = rolling_fama_french(
        asset,
        market,
        smb,
        hml,
        window=window,
    )

    expected = n - window + 1

    assert len(result["alpha"]) == expected
    assert len(result["market_beta"]) == expected
    assert len(result["smb_beta"]) == expected
    assert len(result["hml_beta"]) == expected

import pytest

def test_rolling_fama_french_window_too_large():
    asset = np.arange(5.0)

    with pytest.raises(ValueError):
        rolling_fama_french(
            asset,
            asset,
            asset,
            asset,
            window=6,
        )

def test_rolling_fama_french_window_too_small():
    asset = np.arange(5.0)

    with pytest.raises(ValueError):
        rolling_fama_french(
            asset,
            asset,
            asset,
            asset,
            window=1,
        )

def test_rolling_fama_french_shape_error():
    import numpy as np
    import pytest

    asset = np.arange(6.0)
    market = np.arange(6.0)
    smb = np.arange(5.0)
    hml = np.arange(6.0)

    with pytest.raises(ValueError):
        rolling_fama_french(
            asset,
            market,
            smb,
            hml,
            window=3,
        )

def test_factor_attribution_values():
    result = factor_attribution(
        alpha=0.001,
        market_beta=1.2,
        smb_beta=0.5,
        hml_beta=-0.2,
        market_excess=np.array([0.02]),
        smb=np.array([0.01]),
        hml=np.array([-0.03]),
    )

    assert np.allclose(result["alpha"], [0.001])
    assert np.allclose(result["market"], [0.024])
    assert np.allclose(result["smb"], [0.005])
    assert np.allclose(result["hml"], [0.006])
    assert np.allclose(result["predicted"], [0.036])

def test_factor_attribution_shape_error():
    with pytest.raises(ValueError):
        factor_attribution(
            alpha=0.0,
            market_beta=1.0,
            smb_beta=1.0,
            hml_beta=1.0,
            market_excess=np.array([1.0, 2.0]),
            smb=np.array([1.0]),
            hml=np.array([1.0, 2.0]),
        )

def test_factor_attribution_keys():
    result = factor_attribution(
        alpha=0.0,
        market_beta=1.0,
        smb_beta=1.0,
        hml_beta=1.0,
        market_excess=np.array([1.0]),
        smb=np.array([2.0]),
        hml=np.array([3.0]),
    )

    assert set(result.keys()) == {
        "alpha",
        "market",
        "smb",
        "hml",
        "predicted",
    }


import numpy as np

def test_factor_exposure_report_keys():
    asset = np.array([0.02, 0.03, 0.01, 0.04, 0.05])
    market = np.array([0.01, 0.02, 0.00, 0.03, 0.04])
    smb = np.array([0.005, -0.002, 0.001, 0.004, -0.001])
    hml = np.array([-0.003, 0.002, 0.000, -0.001, 0.003])

    report = factor_exposure_report(
        asset,
        market,
        smb,
        hml,
    )

    assert set(report.keys()) == {
        "alpha",
        "market_beta",
        "smb_beta",
        "hml_beta",
        "r_squared",
        "predicted_returns",
        "residuals",
        "attribution",
    }

def test_factor_exposure_report_prediction_consistency():
    asset = np.array([0.02, 0.03, 0.01, 0.04, 0.05])
    market = np.array([0.01, 0.02, 0.00, 0.03, 0.04])
    smb = np.array([0.005, -0.002, 0.001, 0.004, -0.001])
    hml = np.array([-0.003, 0.002, 0.000, -0.001, 0.003])

    report = factor_exposure_report(
        asset,
        market,
        smb,
        hml,
    )

    assert np.allclose(
        report["predicted_returns"],
        report["attribution"]["predicted"],
    )

