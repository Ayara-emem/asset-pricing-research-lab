import numpy as np
import sys
from pathlib import Path

# Make the src directory importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing_lab.returns import (
    arithmetic_returns,
    log_returns,
    cumulative_returns,
    annualized_return,
    annualized_volatility,
)


def test_arithmetic_returns():
    prices = np.array([100, 105, 110])

    expected = np.array([0.05, 0.04761905])

    result = arithmetic_returns(prices)

    assert np.allclose(result, expected)


def test_log_returns():
    prices = np.array([100, 105, 110])

    expected = np.log(np.array([105 / 100, 110 / 105]))

    result = log_returns(prices)

    assert np.allclose(result, expected)


def test_cumulative_returns():
    returns = np.array([0.10, 0.05, -0.03])

    expected = (1.10 * 1.05 * 0.97) - 1

    result = cumulative_returns(returns)

    assert np.isclose(result, expected)

def test_annualized_return():
    cumulative = 0.10
    periods = 252

    expected = 0.10

    result = annualized_return(cumulative, periods)

    assert np.isclose(result, expected)

def test_annualized_volatility():
    returns = np.array([0.01, 0.02, -0.01, 0.03])

    expected = np.std(returns, ddof=1) * np.sqrt(252)

    result = annualized_volatility(returns)

    assert np.isclose(result, expected)