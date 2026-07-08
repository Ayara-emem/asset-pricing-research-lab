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

def cumulative_returns(returns: np.ndarray) -> float:
    """
    Compute the cumulative return from a series of arithmetic returns.

    Parameters
    ----------
    returns : np.ndarray
        One-dimensional array of arithmetic returns.

    Returns
    -------
    float
        Cumulative return over the period.

    Raises
    ------
    ValueError
        If the input is not one-dimensional.
    """

    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")

    return np.prod(1 + returns) - 1

def annualized_return(
    cumulative_return: float,
    periods: int,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualized return from a cumulative return.

    Parameters
    ----------
    cumulative_return : float
        Cumulative return over the investment period.

    periods : int
        Number of observed periods.

    periods_per_year : int, default=252
        Number of trading periods in one year.

    Returns
    -------
    float
        Annualized return.
    """

    if periods <= 0:
        raise ValueError("periods must be positive")

    return (1 + cumulative_return) ** (periods_per_year / periods) - 1

def annualized_volatility(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized volatility from periodic returns.

    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.

    periods_per_year : int, default=252
        Number of trading periods in one year.

    Returns
    -------
    float
        Annualized volatility.
    """

    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 1:
        raise ValueError("returns must be one-dimensional")

    if len(returns) < 2:
        raise ValueError("At least two returns are required.")

    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)