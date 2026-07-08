import numpy as np

from asset_pricing_lab.statistics import mean_return


def test_mean_return():
    returns = np.array([0.02, 0.01, 0.04, 0.03])

    expected = 0.025

    assert np.isclose(mean_return(returns), expected)