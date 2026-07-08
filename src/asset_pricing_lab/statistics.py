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