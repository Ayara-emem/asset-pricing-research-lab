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

def security_selection(
    asset_return: float,
    risk_free_rate: float,
    market_return: float,
    beta: float,
    atol: float = 1e-12,
) -> str:
    """
    Classify an asset relative to the Security Market Line.

    Parameters
    ----------
    asset_return : float
        Observed or expected asset return.

    risk_free_rate : float
        Risk-free return.

    market_return : float
        Expected market return.

    beta : float
        CAPM Beta.

    atol : float, default=1e-12
        Absolute tolerance used when comparing returns.

    Returns
    -------
    str
        One of:
        - "undervalued"
        - "fairly valued"
        - "overvalued"
    """
    expected = capm_expected_return(
        risk_free_rate=risk_free_rate,
        market_return=market_return,
        beta=beta,
    )

    if np.isclose(
        asset_return,
        expected,
        atol=atol,
    ):
        return "fairly valued"

    if asset_return > expected:
        return "undervalued"

    return "overvalued"

def residuals(
    asset_returns,
    predicted_returns,
):
    """
    Compute regression residuals.

    Parameters
    ----------
    asset_returns : array-like
        Actual asset returns.

    predicted_returns : array-like
        Predicted asset returns.

    Returns
    -------
    numpy.ndarray
        Regression residuals.
    """
    asset_returns = np.asarray(
        asset_returns,
        dtype=float,
    )

    predicted_returns = np.asarray(
        predicted_returns,
        dtype=float,
    )

    if asset_returns.size == 0:
        raise ValueError(
            "asset_returns must not be empty."
        )

    if predicted_returns.size == 0:
        raise ValueError(
            "predicted_returns must not be empty."
        )

    if asset_returns.shape != predicted_returns.shape:
        raise ValueError(
            "asset_returns and predicted_returns must have the same shape."
        )

    return asset_returns - predicted_returns

def r_squared(
    asset_returns,
    predicted_returns,
):
    """
    Compute coefficient of determination.

    Parameters
    ----------
    asset_returns : array-like

    predicted_returns : array-like

    Returns
    -------
    float
        R² statistic.
    """
    asset_returns = np.asarray(
        asset_returns,
        dtype=float,
    )

    predicted_returns = np.asarray(
        predicted_returns,
        dtype=float,
    )

    res = residuals(
        asset_returns,
        predicted_returns,
    )

    ss_res = np.sum(
        res ** 2
    )

    ss_tot = np.sum(
        (
            asset_returns
            - np.mean(asset_returns)
        ) ** 2
    )

    if np.isclose(
        ss_tot,
        0.0,
    ):
        return np.nan

    return 1 - (
        ss_res
        / ss_tot
    )

from .statistics import mean_return


def estimate_capm(
    asset_returns,
    market_returns,
    ddof: int = 1,
) -> dict:
    """
    Estimate CAPM regression parameters.

    Parameters
    ----------
    asset_returns : array-like
        Asset periodic returns.

    market_returns : array-like
        Market periodic returns.

    ddof : int, default=1
        Delta degrees of freedom used in Beta estimation.

    Returns
    -------
    dict
        Dictionary containing the estimated
        alpha and beta.
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

    b = beta(
        asset_returns,
        market_returns,
        ddof=ddof,
    )

    if np.isnan(b):
        a = np.nan
    else:
        a = (
            mean_return(asset_returns)
            - b * mean_return(market_returns)
        )

    return {
        "alpha": a,
        "beta": b,
    }
