def fama_french_expected_return(
    risk_free_rate: float,
    market_return: float,
    smb: float,
    hml: float,
    market_beta: float,
    smb_beta: float,
    hml_beta: float,
) -> float:
    """
    Compute the expected return under the
    Fama–French Three-Factor Model.

    Parameters
    ----------
    risk_free_rate : float
        Risk-free return.

    market_return : float
        Expected market return.

    smb : float
        Small Minus Big factor return.

    hml : float
        High Minus Low factor return.

    market_beta : float
        Market factor loading.

    smb_beta : float
        SMB factor loading.

    hml_beta : float
        HML factor loading.

    Returns
    -------
    float
        Expected return implied by the
        Three-Factor Model.
    """
    market_premium = (
        market_return
        - risk_free_rate
    )

    return (
        risk_free_rate
        + market_beta * market_premium
        + smb_beta * smb
        + hml_beta * hml
    )

import numpy as np


def estimate_fama_french(
    asset_returns,
    market_returns,
    smb,
    hml,
    risk_free_rate: float = 0.0,
):
    """
    Estimate the Fama–French Three-Factor Model.

    Parameters
    ----------
    asset_returns : array-like
        Asset returns.

    market_returns : array-like
        Market returns.

    smb : array-like
        SMB factor returns.

    hml : array-like
        HML factor returns.

    risk_free_rate : float, default=0.0
        Constant risk-free rate.

    Returns
    -------
    dict
        Estimated coefficients.
    """
    asset_returns = np.asarray(asset_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    smb = np.asarray(smb, dtype=float)
    hml = np.asarray(hml, dtype=float)

    n = asset_returns.size

    if n == 0:
        raise ValueError("asset_returns must not be empty.")

    if (
        market_returns.size != n
        or smb.size != n
        or hml.size != n
    ):
        raise ValueError(
            "All inputs must have the same length."
        )

    market_excess = (
        market_returns
        - risk_free_rate
    )

    X = np.column_stack(
        (
            np.ones(n),
            market_excess,
            smb,
            hml,
        )
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        X,
        asset_returns,
        rcond=None,
    )

    return {
        "alpha": coefficients[0],
        "market_beta": coefficients[1],
        "smb_beta": coefficients[2],
        "hml_beta": coefficients[3],
    }

def predicted_returns_fama_french(
    market_returns,
    smb,
    hml,
    alpha: float,
    market_beta: float,
    smb_beta: float,
    hml_beta: float,
    risk_free_rate: float = 0.0,
):
    """
    Compute predicted returns from the
    Fama-French Three-Factor Model.
    """
    market_returns = np.asarray(
        market_returns,
        dtype=float,
    )

    smb = np.asarray(
        smb,
        dtype=float,
    )

    hml = np.asarray(
        hml,
        dtype=float,
    )

    if (
        market_returns.shape != smb.shape
        or market_returns.shape != hml.shape
    ):
        raise ValueError(
            "All factor arrays must have the same shape."
        )

    market_premium = (
        market_returns
        - risk_free_rate
    )

    return (
        alpha
        + market_beta * market_premium
        + smb_beta * smb
        + hml_beta * hml
    )

def residuals_fama_french(
    actual_returns,
    predicted_returns,
):
    """
    Compute residuals from a Fama-French model.

    Parameters
    ----------
    actual_returns : array-like
        Actual excess returns.

    predicted_returns : array-like
        Predicted excess returns.

    Returns
    -------
    numpy.ndarray
        Residuals.
    """
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual_returns and predicted_returns must have the same shape."
        )

    return actual - predicted

def r_squared_fama_french(
    actual_returns,
    predicted_returns,
):
    """
    Compute the coefficient of determination (R²)
    for a Fama-French model.

    Parameters
    ----------
    actual_returns : array-like
        Actual excess returns.

    predicted_returns : array-like
        Predicted excess returns.

    Returns
    -------
    float
        R² statistic.
    """
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual_returns and predicted_returns must have the same shape."
        )

    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)

    if np.isclose(ss_tot, 0.0):
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0

    return 1.0 - ss_res / ss_tot

import numpy as np

def rolling_fama_french(
    asset_returns,
    market_excess,
    smb,
    hml,
    window=60,
):
    """
    Estimate rolling Fama-French factor loadings.

    Parameters
    ----------
    asset_returns : array-like
    market_excess : array-like
    smb : array-like
    hml : array-like
    window : int, default=60

    Returns
    -------
    dict
        Dictionary containing NumPy arrays with rolling
        alpha, market_beta, smb_beta and hml_beta.
    """
    asset = np.asarray(asset_returns, dtype=float)
    market = np.asarray(market_excess, dtype=float)
    smb = np.asarray(smb, dtype=float)
    hml = np.asarray(hml, dtype=float)

    n = len(asset)

    if not (len(market) == len(smb) == len(hml) == n):
        raise ValueError("All input arrays must have the same length.")

    if window < 2:
        raise ValueError("window must be at least 2.")

    if window > n:
        raise ValueError("window cannot exceed the number of observations.")

    alpha = []
    market_beta = []
    smb_beta = []
    hml_beta = []

    for start in range(n - window + 1):
        stop = start + window

        result = estimate_fama_french(
            asset[start:stop],
            market[start:stop],
            smb[start:stop],
            hml[start:stop],
        )

        alpha.append(result["alpha"])
        market_beta.append(result["market_beta"])
        smb_beta.append(result["smb_beta"])
        hml_beta.append(result["hml_beta"])

    return {
        "alpha": np.asarray(alpha),
        "market_beta": np.asarray(market_beta),
        "smb_beta": np.asarray(smb_beta),
        "hml_beta": np.asarray(hml_beta),
    }

import numpy as np

def factor_attribution(
    alpha,
    market_beta,
    smb_beta,
    hml_beta,
    market_excess,
    smb,
    hml,
):
    """
    Compute Fama-French factor attribution.

    Parameters
    ----------
    alpha : float
    market_beta : float
    smb_beta : float
    hml_beta : float
    market_excess : array-like
    smb : array-like
    hml : array-like

    Returns
    -------
    dict
        Factor contributions and predicted returns.
    """
    market = np.asarray(market_excess, dtype=float)
    smb = np.asarray(smb, dtype=float)
    hml = np.asarray(hml, dtype=float)

    if not (market.shape == smb.shape == hml.shape):
        raise ValueError(
            "market_excess, smb, and hml must have the same shape."
        )

    alpha_contribution = np.full_like(market, alpha, dtype=float)
    market_contribution = market_beta * market
    smb_contribution = smb_beta * smb
    hml_contribution = hml_beta * hml

    predicted = (
        alpha_contribution
        + market_contribution
        + smb_contribution
        + hml_contribution
    )

    return {
        "alpha": alpha_contribution,
        "market": market_contribution,
        "smb": smb_contribution,
        "hml": hml_contribution,
        "predicted": predicted,
    }