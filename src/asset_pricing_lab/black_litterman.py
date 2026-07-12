import numpy as np


def implied_equilibrium_returns(
    market_weights,
    covariance_matrix,
    risk_aversion,
):
    """
    Compute implied equilibrium returns.

    Parameters
    ----------
    market_weights : array-like
        Market-capitalization weights.

    covariance_matrix : array-like
        Covariance matrix.

    risk_aversion : float
        Market risk-aversion coefficient.

    Returns
    -------
    numpy.ndarray
        Implied equilibrium returns.
    """
    market_weights = np.asarray(
        market_weights,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if market_weights.ndim != 1:
        raise ValueError(
            "market_weights must be one-dimensional."
        )

    if covariance_matrix.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    n = len(market_weights)

    if covariance_matrix.shape != (n, n):
        raise ValueError(
            "covariance_matrix shape must match market_weights."
        )

    if risk_aversion <= 0:
        raise ValueError(
            "risk_aversion must be positive."
        )

    return (
        risk_aversion
        * covariance_matrix
        @ market_weights
    )