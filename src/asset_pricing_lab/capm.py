import numpy as np

from .statistics import (
    covariance,
    variance,
)

def beta(
    asset_returns,
    market_returns,
    ddof: int = 1,
) -> float:
    """
    Estimate the CAPM Beta.

    Parameters
    ----------
    asset_returns : array-like
        Asset periodic returns.

    market_returns : array-like
        Market benchmark periodic returns.

    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    float
        Estimated Beta. Returns np.nan if the
        market variance is zero.
    """
    asset_returns = np.asarray(
        asset_returns,
        dtype=float,
    )

    market_returns = np.asarray(
        market_returns,
        dtype=float,
    )

    if asset_returns.size == 0:
        raise ValueError(
            "asset_returns must not be empty."
        )

    if market_returns.size == 0:
        raise ValueError(
            "market_returns must not be empty."
        )

    if asset_returns.shape != market_returns.shape:
        raise ValueError(
            "asset_returns and market_returns must have the same shape."
        )

    market_variance = variance(
        market_returns,
        ddof=ddof,
    )

    if np.isclose(market_variance, 0.0):
        return np.nan

    return covariance(
        asset_returns,
        market_returns,
        ddof=ddof,
    ) / market_variance

def alpha(
    asset_return: float,
    market_return: float,
    risk_free_rate: float,
    beta: float,
) -> float:
    """
    Compute Jensen's Alpha.
    """
    expected_return = capm_expected_return(
        risk_free_rate=risk_free_rate,
        market_return=market_return,
        beta=beta,
    )

    return asset_return - expected_return

def capm_expected_return(
    risk_free_rate: float,
    market_return: float,
    beta: float,
) -> float:
    """
    Compute the CAPM expected return.

    Parameters
    ----------
    risk_free_rate : float
        Risk-free return.

    market_return : float
        Expected market return.

    beta : float
        Estimated CAPM Beta.

    Returns
    -------
    float
        CAPM expected return.
    """
    return (
        risk_free_rate
        + beta
        * (
            market_return
            - risk_free_rate
        )
    )

