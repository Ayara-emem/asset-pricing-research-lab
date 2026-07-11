import numpy as np

def portfolio_expected_return(
    weights,
    expected_returns,
):
    """
    Compute the expected return of a portfolio.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    expected_returns : array-like
        Expected return of each asset.

    Returns
    -------
    float
        Expected portfolio return.
    """
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)

    if weights.shape != expected_returns.shape:
        raise ValueError(
            "weights and expected_returns must have the same shape."
        )

    return float(np.dot(weights, expected_returns))