"""
Asset Pricing Research Laboratory

An institutional-grade Python framework for asset pricing,
factor modeling, and quantitative investment research.
"""

from .version import __version__

from .returns import (
    arithmetic_returns,
    log_returns,
    cumulative_returns,
    annualized_return,
    annualized_volatility,
)

from .statistics import (
    mean_return,
    variance,
    standard_deviation,
    covariance,
    correlation,
    kurtosis,
    skewness,
    covariance_matrix,
    correlation_matrix,
)

from .risk import (
    historical_volatility,
    rolling_volatility,
    downside_deviation,
    sharpe_ratio,
    sortino_ratio,
    drawdown,
    maximum_drawdown,
    calmar_ratio,
    historical_var,
    expected_shortfall,
    tracking_error,
    information_ratio,
) 

__all__ = [
    "arithmetic_returns",
    "log_returns",
    "cumulative_returns",
    "annualized_return",
    "annualized_volatility",
    "mean_return",
    "variance",
    "standard_deviation",
    "kurtosis",
    "skewness",
    "correlation_matrix",
    "historical_volatility",
    "rolling_volatility",
    "downside_deviation",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown",
    "maximum_drawdown",
    "calmar_ratio",
    "historical_var",
    "expected_shortfall",
    "tracking_error",
    "information_ratio",
]