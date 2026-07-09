import numpy as np

from asset_pricing_lab.risk import historical_volatility
from asset_pricing_lab.statistics import standard_deviation


def test_historical_volatility():
    returns = np.array(
        [0.01, 0.02, -0.01, 0.03, 0.00]
    )

    expected = standard_deviation(returns)

    assert np.isclose(
        historical_volatility(returns),
        expected,
    )