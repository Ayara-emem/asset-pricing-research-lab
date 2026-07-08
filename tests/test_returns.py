import numpy as np
import sys
from pathlib import Path

# Make the src directory importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing_lab.returns import arithmetic_returns, log_returns


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