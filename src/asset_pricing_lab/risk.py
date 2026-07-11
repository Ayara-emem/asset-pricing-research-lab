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

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rolling_volatility(
    returns: ArrayLike,
    window: int = 20,
    ddof: int = 1,
) -> NDArray[np.float64]:
    """
    Compute rolling historical volatility.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.

    window : int, default=20
        Rolling window length.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Rolling volatility values.

    Raises
    ------
    ValueError
        If the window size is invalid.
    """
    returns = np.asarray(returns, dtype=float)

    if window <= 0:
        raise ValueError(
            "Window size must be positive."
        )

    if returns.size < window:
        raise ValueError(
            "Window size cannot exceed the number of observations."
        )

    rolling = []

    for i in range(window, len(returns) + 1):
        window_returns = returns[i - window:i]

        rolling.append(
            historical_volatility(
                window_returns,
                ddof=ddof,
            )
        )

    return np.asarray(rolling)

import numpy as np
from numpy.typing import ArrayLike


def downside_deviation(
    returns: ArrayLike,
    target: float = 0.0,
) -> float:
    """
    Compute the downside deviation of a return series.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.

    target : float, default=0.0
        Minimum acceptable return.

    Returns
    -------
    float
        Downside deviation.
    """
    returns = np.asarray(returns, dtype=float)

    downside = np.minimum(
        returns - target,
        0.0,
    )

    return np.sqrt(
        np.mean(
            downside ** 2
        )
    )

from .statistics import mean_return


def sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    ddof: int = 1,
) -> float:
    """
    Compute the Sharpe Ratio.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.

    risk_free_rate : float, default=0.0
        Risk-free return expressed in the same periodicity
        as the returns.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    float
        Sharpe Ratio.

    Raises
    ------
    ValueError
        If volatility is zero.
    """
    avg_return = mean_return(returns)

    volatility = historical_volatility(
        returns,
        ddof=ddof,
    )

    if np.isclose(volatility, 0.0):
        raise ValueError(
            "Sharpe Ratio is undefined when volatility is zero."
        )

    return (avg_return - risk_free_rate) / volatility

def sortino_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    target: float = 0.0,
) -> float:
    """
    Compute the Sortino Ratio.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.

    risk_free_rate : float, default=0.0
        Risk-free return expressed in the same periodicity
        as the returns.

    target : float, default=0.0
        Minimum acceptable return used to compute
        downside deviation.

    Returns
    -------
    float
        Sortino Ratio.

    Raises
    ------
    ValueError
        If downside deviation is zero.
    """
    avg_return = mean_return(returns)

    downside = downside_deviation(
        returns,
        target=target,
    )

    if np.isclose(downside, 0.0):
        raise ValueError(
            "Sortino Ratio is undefined when downside deviation is zero."
        )

    return (avg_return - risk_free_rate) / downside

def drawdown(
    prices: ArrayLike,
) -> NDArray[np.float64]:
    """
    Compute the drawdown series from a price series.

    Parameters
    ----------
    prices : array-like
        Portfolio or asset values.

    Returns
    -------
    numpy.ndarray
        Drawdown series.

    Raises
    ------
    ValueError
        If the input is empty or contains non-positive values.
    """
    prices = np.asarray(prices, dtype=float)

    if prices.size == 0:
        raise ValueError(
            "Price series cannot be empty."
        )

    if np.any(prices <= 0):
        raise ValueError(
            "Prices must be strictly positive."
        )

    running_max = np.maximum.accumulate(prices)

    return (prices - running_max) / running_max

def maximum_drawdown(
    prices: ArrayLike,
) -> float:
    """
    Compute the maximum drawdown of a price series.

    Parameters
    ----------
    prices : array-like
        Portfolio or asset values.

    Returns
    -------
    float
        Maximum drawdown.
    """
    return float(np.min(drawdown(prices)))

from .returns import annualized_return


def calmar_ratio(
    prices: ArrayLike,
    periods: int,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the Calmar Ratio.

    Parameters
    ----------
    prices : array-like
        Asset or portfolio values.

    periods : int
        Number of periods.

    periods_per_year : int, default=252
        Number of observations per year.

    Returns
    -------
    float
        Calmar Ratio.

    Raises
    ------
    ValueError
        If maximum drawdown is zero.
    """
    prices = np.asarray(prices, dtype=float)
    cumulative_return = (prices[-1] / prices[0]) - 1
    annual_return = annualized_return(
        cumulative_return=cumulative_return,
        periods=periods,
        periods_per_year=periods_per_year,
)

    mdd = abs(maximum_drawdown(prices))

    if np.isclose(mdd, 0.0):
        raise ValueError(
            "Calmar Ratio is undefined when maximum drawdown is zero."
        )

    return annual_return / mdd

def historical_var(
    returns,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Historical Value at Risk (VaR).

    Parameters
    ----------
    returns : array-like
        Periodic returns.

    confidence_level : float, default=0.95
        Confidence level.

    Returns
    -------
    float
        Historical Value at Risk reported as a positive loss.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        raise ValueError("returns must not be empty.")

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    percentile = np.percentile(
        returns,
        (1 - confidence_level) * 100,
    )

    return -float(percentile)

def expected_shortfall(
    returns,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Historical Expected Shortfall (Conditional VaR).

    Parameters
    ----------
    returns : array-like
        Periodic returns.

    confidence_level : float, default=0.95
        Confidence level.

    Returns
    -------
    float
        Expected Shortfall reported as a positive loss.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        raise ValueError("returns must not be empty.")

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    percentile = np.percentile(
        returns,
        (1 - confidence_level) * 100,
    )

    tail_losses = returns[returns <= percentile]

    return -float(np.mean(tail_losses))


def tracking_error(
    portfolio_returns,
    benchmark_returns,
    ddof: int = 1,
) -> float:
    """
    Compute the Tracking Error between a portfolio and its benchmark.

    Parameters
    ----------
    portfolio_returns : array-like
        Portfolio periodic returns.

    benchmark_returns : array-like
        Benchmark periodic returns.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    float
        Tracking Error.
    """
    portfolio_returns = np.asarray(
        portfolio_returns,
        dtype=float,
    )

    benchmark_returns = np.asarray(
        benchmark_returns,
        dtype=float,
    )

    if portfolio_returns.size == 0:
        raise ValueError(
            "portfolio_returns must not be empty."
        )

    if benchmark_returns.size == 0:
        raise ValueError(
            "benchmark_returns must not be empty."
        )

    if portfolio_returns.shape != benchmark_returns.shape:
        raise ValueError(
            "portfolio_returns and benchmark_returns must have the same shape."
        )

    active_returns = (
        portfolio_returns
        - benchmark_returns
    )

    return standard_deviation(
        active_returns,
        ddof=ddof,
    )

from .statistics import mean_return


import numpy as np

def information_ratio(
    portfolio_returns,
    benchmark_returns,
    ddof=1,
):
    """
    Compute the Information Ratio.

    Parameters
    ----------
    portfolio_returns : array-like
        Portfolio returns.
    benchmark_returns : array-like
        Benchmark returns.
    ddof : int, default=1
        Delta degrees of freedom used when computing
        the standard deviation of active returns.

    Returns
    -------
    float
        Information Ratio.
    """
    portfolio = np.asarray(portfolio_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)

    if portfolio.shape != benchmark.shape:
        raise ValueError(
            "portfolio_returns and benchmark_returns must have the same shape."
        )

    active = portfolio - benchmark

    tracking_error = np.std(active, ddof=ddof)

    if not np.isfinite(tracking_error) or tracking_error <= 0:
        raise ValueError("Tracking error must be positive.")
    return np.mean(active) / tracking_error


def appraisal_ratio(
    alpha,
    residuals,
    ddof=1,
):
    """
    Compute the Appraisal Ratio.

    Parameters
    ----------
    alpha : float
        Estimated alpha.
    residuals : array-like
        Regression residuals.
    ddof : int, default=1
        Delta degrees of freedom used when computing
        the residual standard deviation.

    Returns
    -------
    float
        Appraisal Ratio.
    """
    residuals = np.asarray(residuals, dtype=float)

    residual_std = np.std(residuals, ddof=ddof)

    if np.isclose(residual_std, 0.0):
        raise ValueError("Residual standard deviation is zero.")

    return alpha / residual_std
