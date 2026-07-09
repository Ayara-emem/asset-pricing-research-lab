"""
Risk analytics functions for financial return series.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from .statistics import standard_deviation


def historical_volatility(
    returns: ArrayLike,
    ddof: int = 1,
) -> float:
    """
    Compute the historical volatility of a return series.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    float
        Historical volatility.
    """
    return standard_deviation(
        returns,
        ddof=ddof,
    )