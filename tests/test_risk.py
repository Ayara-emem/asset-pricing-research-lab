import numpy as np
import pytest
from asset_pricing_lab.returns import annualized_return
from asset_pricing_lab.statistics import (
    standard_deviation,
    mean_return,
    variance,
)
from asset_pricing_lab.risk import rolling_volatility
from asset_pricing_lab.risk import (
    historical_volatility,
    downside_deviation,
    sharpe_ratio,
    sortino_ratio,
    drawdown,
    maximum_drawdown,
    calmar_ratio,

)

def test_historical_volatility():
    returns = np.array(
        [0.01, 0.02, -0.01, 0.03, 0.00]
    )

    expected = standard_deviation(returns)

    assert np.isclose(
        historical_volatility(returns),
        expected,
    )

def test_rolling_volatility():
    returns = np.array(
        [0.01, 0.02, -0.01, 0.03, 0.00]
    )

    result = rolling_volatility(
        returns,
        window=3,
    )

    expected = np.array([
        historical_volatility(returns[:3]),
        historical_volatility(returns[1:4]),
        historical_volatility(returns[2:5]),
    ])

    assert np.allclose(result, expected)

def test_rolling_volatility_invalid_window():
    returns = np.array([0.01, 0.02])

    with pytest.raises(ValueError):
        rolling_volatility(
            returns,
            window=0,
        )

def test_rolling_volatility_invalid_window():
    returns = np.array([0.01, 0.02])

    with pytest.raises(ValueError):
        rolling_volatility(
            returns,
            window=0,
        )

def test_rolling_volatility_window_too_large():
    returns = np.array([0.01, 0.02])

    with pytest.raises(ValueError):
        rolling_volatility(
            returns,
            window=10,
        )

def test_downside_deviation():
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    downside = np.minimum(
        returns,
        0.0,
    )

    expected = np.sqrt(
        np.mean(
            downside ** 2
        )
    )

    assert np.isclose(
        downside_deviation(returns),
        expected,
    )

def test_downside_deviation_no_losses():
    returns = np.array(
        [0.01, 0.02, 0.03]
    )

    assert np.isclose(
        downside_deviation(returns),
        0.0,
    )

def test_sharpe_ratio():
    returns = np.array(
        [0.01, 0.02, -0.01, 0.03]
    )

    expected = (
        mean_return(returns)
        / historical_volatility(returns)
    )

    assert np.isclose(
        sharpe_ratio(returns),
        expected,
    )

def test_sharpe_ratio_with_risk_free_rate():
    returns = np.array(
        [0.01, 0.02, -0.01, 0.03]
    )

    rf = 0.005

    expected = (
        mean_return(returns) - rf
    ) / historical_volatility(returns)

    assert np.isclose(
        sharpe_ratio(
            returns,
            risk_free_rate=rf,
        ),
        expected,
    )

def test_sharpe_ratio_zero_volatility():
    returns = np.array(
        [0.01, 0.01, 0.01]
    )

    with pytest.raises(ValueError):
        sharpe_ratio(returns)

def test_sortino_ratio():
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    expected = (
        mean_return(returns)
        / downside_deviation(returns)
    )

    assert np.isclose(
        sortino_ratio(returns),
        expected,
    )

def test_sortino_ratio_with_risk_free_rate():
    returns = np.array(
        [0.01, -0.02, 0.03, -0.01]
    )

    rf = 0.005

    expected = (
        mean_return(returns) - rf
    ) / downside_deviation(returns)

    assert np.isclose(
        sortino_ratio(
            returns,
            risk_free_rate=rf,
        ),
        expected,
    )

def test_sortino_ratio_zero_downside():
    returns = np.array(
        [0.01, 0.02, 0.03]
    )

    with pytest.raises(ValueError):
        sortino_ratio(returns)

def test_drawdown():
    prices = np.array(
        [100, 120, 110, 130, 125],
        dtype=float,
    )

    expected = np.array([
        0.0,
        0.0,
        (110 - 120) / 120,
        0.0,
        (125 - 130) / 130,
    ])

    assert np.allclose(
        drawdown(prices),
        expected,
    )

def test_drawdown_empty():
    with pytest.raises(ValueError):
        drawdown([])

def test_drawdown_negative_prices():
    with pytest.raises(ValueError):
        drawdown([100, -50, 120])

def test_drawdown_monotonic_prices():
    prices = np.array(
        [100, 105, 110, 120],
        dtype=float,
    )

    expected = np.zeros_like(prices)

    assert np.allclose(
        drawdown(prices),
        expected,
    )

def test_maximum_drawdown():
    prices = np.array(
        [100, 120, 110, 130, 125],
        dtype=float,
    )

    expected = (110 - 120) / 120

    assert np.isclose(
        maximum_drawdown(prices),
        expected,
    )

def test_maximum_drawdown_no_loss():
    prices = np.array(
        [100, 105, 110, 120],
        dtype=float,
    )

    assert np.isclose(
        maximum_drawdown(prices),
        0.0,
    )

def test_maximum_drawdown_matches_drawdown():
    prices = np.array(
        [100, 120, 110, 130, 90],
        dtype=float,
    )

    assert np.isclose(
        maximum_drawdown(prices),
        np.min(drawdown(prices)),
    )

def test_calmar_ratio():
    prices = np.array(
        [100, 110, 105, 120, 115],
        dtype=float,
    )

    cumulative_return = (prices[-1] / prices[0]) - 1

    expected = (
    annualized_return(
        cumulative_return=cumulative_return,
        periods=len(prices) - 1,
        periods_per_year=252,
    )
    / abs(maximum_drawdown(prices))
)

    assert np.isclose(
    calmar_ratio(
        prices,
        periods=len(prices) - 1,
        periods_per_year=252,
    ),
    expected,
)

def test_calmar_ratio_zero_drawdown():
    prices = np.array(
        [100, 105, 110, 120],
        dtype=float,
    )

    periods = len(prices) - 1

    with pytest.raises(ValueError):
        calmar_ratio(
            prices,
            periods=periods,
            periods_per_year=252,
        )

def test_calmar_ratio_relationship():
    prices = np.array(
        [100, 120, 110, 130, 90],
        dtype=float,
    )

    cumulative_return = (prices[-1] / prices[0]) - 1

    expected = (
    annualized_return(
        cumulative_return=cumulative_return,
        periods=len(prices) - 1,
        periods_per_year=252,
    )
    / abs(maximum_drawdown(prices))
)

    assert np.isclose(
    calmar_ratio(
        prices,
        periods=len(prices) - 1,
    ),
    expected,
)