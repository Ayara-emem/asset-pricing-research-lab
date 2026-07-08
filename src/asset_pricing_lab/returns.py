"""
Return calculations for financial assets.

This module implements return calculations from first principles.
"""

from __future__ import annotations

import numpy as np


def arithmetic_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute arithmetic returns from a price series.

    Parameters
    ----------
    prices : np.ndarray
        One-dimensional array of asset prices.

    Returns
    -------
    np.ndarray
        Arithmetic return series.
    """

    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be one-dimensional")

    if len(prices) < 2:
        raise ValueError("At least two prices are required.")

    return (prices[1:] - prices[:-1]) / prices[:-1]

def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute continuously compounded (log) returns.

    Parameters
    ----------
    prices : np.ndarray
        One-dimensional array of asset prices.

    Returns
    -------
    np.ndarray
        Log return series.

    Raises
    ------
    ValueError
        If prices are not one-dimensional, contain fewer than
        two observations, or contain non-positive values.
    """

    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 1:
        raise ValueError("prices must be one-dimensional")

    if len(prices) < 2:
        raise ValueError("At least two prices are required.")

    if np.any(prices <= 0):
        raise ValueError("All prices must be positive.")

    return np.log(prices[1:] / prices[:-1])