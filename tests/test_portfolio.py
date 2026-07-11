
import numpy as np

import pytest

from asset_pricing_lab.portfolio import (
    portfolio_expected_return,
)



def test_portfolio_expected_return():
    weights = np.array([0.5, 0.3, 0.2])
    returns = np.array([0.10, 0.08, 0.12])

    result = portfolio_expected_return(
        weights,
        returns,
    )

    assert np.isclose(result, 0.098)


import pytest

def test_portfolio_expected_return_shape_error():
    with pytest.raises(ValueError):
        portfolio_expected_return(
            np.array([0.5, 0.5]),
            np.array([0.10]),
        )