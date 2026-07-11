"""
Statistical functions for financial return series.
"""

from __future__ import annotations

import numpy as np

from scipy.stats import skew


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

def standard_deviation(returns, ddof=1):
    """
    Compute the standard deviation (volatility) of a return series.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.
    ddof : int, default=1
        Delta degrees of freedom.
        ddof=1 computes the sample standard deviation.
        ddof=0 computes the population standard deviation.

    Returns
    -------
    float
        Standard deviation of the return series.
    """
    return np.sqrt(variance(returns, ddof=ddof))

def covariance(x, y, ddof=1):
    """
    Compute the covariance between two return series.

    Parameters
    ----------
    x : array-like
        First return series.
    y : array-like
        Second return series.
    ddof : int, default=1
        Delta degrees of freedom.
        ddof=1 computes the sample covariance.
        ddof=0 computes the population covariance.

    Returns
    -------
    float
        Covariance between the two return series.

    Raises
    ------
    ValueError
        If the inputs have different lengths, are empty,
        or contain too few observations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty.")

    if x.size != y.size:
        raise ValueError("Input arrays must have the same length.")

    if x.size <= ddof:
        raise ValueError(
            "Number of observations must be greater than ddof."
        )

    return np.cov(x, y, ddof=ddof)[0, 1]

def correlation(x, y, ddof=1):
    """
    Compute the Pearson correlation coefficient between two return series.

    Parameters
    ----------
    x : array-like
        First return series.
    y : array-like
        Second return series.
    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    float
        Pearson correlation coefficient.

    Raises
    ------
    ValueError
        If either series has zero standard deviation.
    """
    cov = covariance(x, y, ddof=ddof)
    std_x = standard_deviation(x, ddof=ddof)
    std_y = standard_deviation(y, ddof=ddof)

    if std_x == 0 or std_y == 0:
        raise ValueError(
            "Correlation is undefined when a series has zero variance."
        )

    return cov / (std_x * std_y)


from scipy.stats import skew

def skewness(returns, bias=False):
    returns = np.asarray(returns, dtype=float)

    if returns.size < 3:
        raise ValueError(
            "At least three observations are required."
        )

    return float(skew(returns, bias=bias))

from scipy.stats import kurtosis as scipy_kurtosis


def kurtosis(returns, bias=False, fisher=True):
    """
    Compute the kurtosis of a return series.

    Parameters
    ----------
    returns : array-like
        Sequence of returns.
    bias : bool, default=False
        If False, compute the bias-corrected estimator.
    fisher : bool, default=True
        If True, return excess kurtosis
        (normal distribution has kurtosis 0).
        If False, return Pearson kurtosis
        (normal distribution has kurtosis 3).

    Returns
    -------
    float
        Kurtosis of the return series.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size < 4:
        raise ValueError(
            "At least four observations are required."
        )

    return float(
        scipy_kurtosis(
            returns,
            bias=bias,
            fisher=fisher,
        )
    )

def covariance_matrix(returns, ddof=1):
    """
    Compute the covariance matrix for multiple return series.

    Parameters
    ----------
    returns : array-like of shape (n_observations, n_assets)
        Matrix of returns where each column represents an asset.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Covariance matrix.

    Raises
    ------
    ValueError
        If the input is empty, one-dimensional,
        or has too few observations.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        raise ValueError("Input returns array cannot be empty.")

    if returns.ndim != 2:
        raise ValueError(
            "Input must be a 2-dimensional array."
        )

    if returns.shape[0] <= ddof:
        raise ValueError(
            "Number of observations must be greater than ddof."
        )

    return np.cov(
        returns,
        rowvar=False,
        ddof=ddof,
    )
def correlation_matrix(returns):
    """
    Compute the correlation matrix for multiple return series.

    Parameters
    ----------
    returns : array-like of shape (n_observations, n_assets)
        Matrix of returns where each column represents an asset.

    Returns
    -------
    numpy.ndarray
        Correlation matrix.

    Raises
    ------
    ValueError
        If the input is empty or not two-dimensional.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.size == 0:
        raise ValueError("Input returns array cannot be empty.")

    if returns.ndim != 2:
        raise ValueError(
            "Input must be a 2-dimensional array."
        )

    return np.corrcoef(
        returns,
        rowvar=False,
    )

import numpy as np

def adjusted_r_squared(
    r_squared,
    n_observations,
    n_predictors,
):
    """
    Compute the adjusted R-squared statistic.

    Parameters
    ----------
    r_squared : float
        Ordinary R-squared.
    n_observations : int
        Number of observations.
    n_predictors : int
        Number of predictors (excluding the intercept).

    Returns
    -------
    float
        Adjusted R-squared.
    """
    if not 0.0 <= r_squared <= 1.0:
        raise ValueError("r_squared must be between 0 and 1.")

    if n_observations <= n_predictors + 1:
        raise ValueError(
            "n_observations must be greater than n_predictors + 1."
        )

    return (
        1.0
        - (1.0 - r_squared)
        * (n_observations - 1)
        / (n_observations - n_predictors - 1)
    )