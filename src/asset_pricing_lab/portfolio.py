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

import numpy as np


def diversification_ratio(
    weights,
    covariance_matrix,
):
    """
    Compute the diversification ratio.

    Parameters
    ----------
    weights : array-like
        Portfolio weights.

    covariance_matrix : array-like
        Covariance matrix.

    Returns
    -------
    float
        Diversification ratio.
    """
    weights = np.asarray(weights, dtype=float)
    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if covariance_matrix.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    n = len(weights)

    if covariance_matrix.shape != (n, n):
        raise ValueError(
            "covariance_matrix shape must match weights."
        )

    asset_volatility = np.sqrt(
        np.diag(covariance_matrix)
    )

    weighted_average_volatility = np.dot(
        weights,
        asset_volatility,
    )

    portfolio_vol = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    if np.isclose(portfolio_vol, 0.0):
        raise ValueError(
            "Portfolio volatility is zero."
        )

    return (
        weighted_average_volatility
        / portfolio_vol
    )

import numpy as np


def simulate_portfolios(
    expected_returns,
    covariance_matrix,
    n_portfolios=10000,
    random_state=None,
):
    """
    Simulate random portfolios.

    Parameters
    ----------
    expected_returns : array-like
        Expected returns.

    covariance_matrix : array-like
        Covariance matrix.

    n_portfolios : int
        Number of portfolios.

    random_state : int or None
        Random seed.

    Returns
    -------
    dict
        Simulated portfolios.
    """
    expected_returns = np.asarray(
        expected_returns,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    n_assets = len(expected_returns)

    if covariance_matrix.shape != (
        n_assets,
        n_assets,
    ):
        raise ValueError(
            "covariance_matrix has incorrect shape."
        )

    if n_portfolios <= 0:
        raise ValueError(
            "n_portfolios must be positive."
        )

    rng = np.random.default_rng(
        random_state
    )

    weights = np.empty(
        (n_portfolios, n_assets)
    )

    portfolio_returns = np.empty(
        n_portfolios
    )

    portfolio_volatilities = np.empty(
    n_portfolios
    )

    for i in range(n_portfolios):

        w = rng.random(n_assets)

        w /= w.sum()

        weights[i] = w

        portfolio_returns[i] = (
            portfolio_expected_return(
                w,
                expected_returns,
            )
        )

        portfolio_volatilities[i] = portfolio_volatility(
            w,
            covariance_matrix,
            )

    return {
        "weights": weights,
        "returns": portfolio_returns,
        "volatility": portfolio_volatilities,
    }

import numpy as np


def global_minimum_variance_portfolio(
    covariance_matrix,
):
    """
    Compute the Global Minimum Variance Portfolio.

    Parameters
    ----------
    covariance_matrix : array-like
        Covariance matrix.

    Returns
    -------
    dict
        Portfolio weights, variance and volatility.
    """
    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if covariance_matrix.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    n, m = covariance_matrix.shape

    if n != m:
        raise ValueError(
            "covariance_matrix must be square."
        )

    ones = np.ones(n)

    try:
        inv_cov = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "covariance_matrix is singular."
        ) from exc

    weights = inv_cov @ ones
    weights /= ones @ weights

    variance = portfolio_variance(
        weights,
        covariance_matrix,
    )

    volatility = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    return {
        "weights": weights,
        "variance": variance,
        "volatility": volatility,
    }

import numpy as np


def maximum_sharpe_portfolio(
    expected_returns,
    covariance_matrix,
    risk_free_rate=0.0,
):
    """
    Compute the Maximum Sharpe (Tangency) Portfolio.

    Parameters
    ----------
    expected_returns : array-like
        Expected returns.

    covariance_matrix : array-like
        Covariance matrix.

    risk_free_rate : float, default=0.0
        Risk-free rate.

    Returns
    -------
    dict
        Portfolio statistics.
    """
    expected_returns = np.asarray(
        expected_returns,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if covariance_matrix.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    n = len(expected_returns)

    if covariance_matrix.shape != (n, n):
        raise ValueError(
            "covariance_matrix shape must match expected_returns."
        )

    ones = np.ones(n)

    excess_returns = (
        expected_returns
        - risk_free_rate
    )

    try:
        weights = np.linalg.solve(
            covariance_matrix,
            excess_returns,
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "covariance_matrix is singular."
        ) from exc

    weights /= ones @ weights

    expected_return = portfolio_expected_return(
        weights,
        expected_returns,
    )

    variance = portfolio_variance(
        weights,
        covariance_matrix,
    )

    volatility = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    if np.isclose(volatility, 0.0):
        raise ValueError(
            "Portfolio volatility is zero."
        )

    sharpe_ratio = (
        expected_return
        - risk_free_rate
    ) / volatility

    return {
        "weights": weights,
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }

def efficient_frontier(
    expected_returns,
    covariance_matrix,
    n_portfolios=10000,
    n_bins=100,
    random_state=None,
):
    """
    Approximate the Efficient Frontier using
    Monte Carlo simulation.

    Parameters
    ----------
    expected_returns : array-like
    covariance_matrix : array-like
    n_portfolios : int
    n_bins : int
    random_state : int or None

    Returns
    -------
    dict
        Approximate Efficient Frontier.
    """
    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive."
        )

    simulation = simulate_portfolios(
        expected_returns,
        covariance_matrix,
        n_portfolios=n_portfolios,
        random_state=random_state,
    )

    vol = simulation["volatility"]
    ret = simulation["returns"]
    weights = simulation["weights"]

    edges = np.linspace(
        vol.min(),
        vol.max(),
        n_bins + 1,
    )

    frontier_weights = []
    frontier_returns = []
    frontier_volatility = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (vol >= edges[i]) &
                (vol <= edges[i + 1])
            )
        else:
            mask = (
                (vol >= edges[i]) &
                (vol < edges[i + 1])
            )

        if not np.any(mask):
            continue

        local_returns = ret[mask]

        best = np.argmax(local_returns)

        frontier_weights.append(
            weights[mask][best]
        )

        frontier_returns.append(
            local_returns[best]
        )

        frontier_volatility.append(
            vol[mask][best]
        )

    return {
        "weights": np.asarray(frontier_weights),
        "returns": np.asarray(frontier_returns),
        "volatility": np.asarray(frontier_volatility),
    }