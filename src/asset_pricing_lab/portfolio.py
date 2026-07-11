import numpy as np

def portfolio_expected_return(
    weights,
    expected_returns,
):
    """
    Compute the expected return of a portfolio.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    expected_returns : array-like
        Expected return of each asset.

    Returns
    -------
    float
        Expected portfolio return.
    """
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)

    if weights.shape != expected_returns.shape:
        raise ValueError(
            "weights and expected_returns must have the same shape."
        )

    return float(np.dot(weights, expected_returns))


def portfolio_variance(
    weights,
    covariance_matrix,
):
    """
    Compute portfolio variance.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    covariance_matrix : array-like
        Asset covariance matrix.

    Returns
    -------
    float
        Portfolio variance.
    """
    weights = np.asarray(weights, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)

    if covariance_matrix.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    n = len(weights)

    if covariance_matrix.shape != (n, n):
        raise ValueError(
            "covariance_matrix shape must match the number of weights."
        )

    return float(
        weights.T @ covariance_matrix @ weights
    )

def portfolio_volatility(
    weights,
    covariance_matrix,
):
    """
    Compute portfolio volatility.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    covariance_matrix : array-like
        Asset covariance matrix.

    Returns
    -------
    float
        Portfolio volatility.
    """
    variance = portfolio_variance(
        weights,
        covariance_matrix,
    )

    return float(np.sqrt(variance))

import numpy as np


def covariance_matrix(
    returns,
    ddof=1,
):
    """
    Compute the covariance matrix of asset returns.

    Parameters
    ----------
    returns : array-like
        Matrix of returns with rows representing observations
        and columns representing assets.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Covariance matrix.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 2:
        raise ValueError(
            "returns must be a two-dimensional array."
        )

    return np.cov(
        returns,
        rowvar=False,
        ddof=ddof,
    )

def correlation_matrix(
    returns,
):
    """
    Compute the correlation matrix of asset returns.

    Parameters
    ----------
    returns : array-like
        Matrix of asset returns.

    Returns
    -------
    numpy.ndarray
        Correlation matrix.
    """
    returns = np.asarray(returns, dtype=float)

    if returns.ndim != 2:
        raise ValueError(
            "returns must be a two-dimensional array."
        )

    return np.corrcoef(
        returns,
        rowvar=False,
    )