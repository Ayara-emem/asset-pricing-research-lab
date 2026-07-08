"""
Statistical functions for financial return series.
"""

from __future__ import annotations

import numpy as np


def mean_return(returns):
    """
    Compute the arithmetic mean of a return series.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.

    Returns
    -------
    float
        Mean return.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        raise ValueError("Input returns array cannot be empty.")

    return np.mean(returns)


def variance(returns, ddof=1):
    """
    Compute the variance of a return series.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.
    ddof : int, default=1
        Delta degrees of freedom.
        ddof=1 computes the sample variance.
        ddof=0 computes the population variance.

    Returns
    -------
    float
        Variance of the return series.

    Raises
    ------
    ValueError
        If the input array is empty or contains too few observations.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        raise ValueError("Input returns array cannot be empty.")

    if returns.size <= ddof:
        raise ValueError(
            "Number of observations must be greater than ddof."
        )

    return np.var(returns, ddof=ddof)