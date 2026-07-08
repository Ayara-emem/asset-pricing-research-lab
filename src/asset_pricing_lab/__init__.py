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
    covariance_matrix,
    correlation_matrix,
    
)
from scipy.stats import skew

__all__ = [
    "arithmetic_returns",
    "log_returns",
    "cumulative_returns",
    "annualized_return",
    "annualized_volatility",
    "mean_return",
    "variance",
    "kurtosis"
    "correlation_matrix",
]