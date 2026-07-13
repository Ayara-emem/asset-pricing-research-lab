import numpy as np


def implied_equilibrium_returns(
    market_weights,
    covariance_matrix,
    risk_aversion,
):
    """
    Compute implied equilibrium returns.

    Parameters
    ----------
    market_weights : array-like
        Market-capitalization weights.

    covariance_matrix : array-like
        Covariance matrix.

    risk_aversion : float
        Market risk-aversion coefficient.

    Returns
    -------
    numpy.ndarray
        Implied equilibrium returns.
    """
    market_weights = np.asarray(
        market_weights,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if market_weights.ndim != 1:
        raise ValueError(
            "market_weights must be one-dimensional."
        )

    if covariance_matrix.ndim != 2:
        raise ValueError(
            "covariance_matrix must be two-dimensional."
        )

    n = len(market_weights)

    if covariance_matrix.shape != (n, n):
        raise ValueError(
            "covariance_matrix shape must match market_weights."
        )

    if risk_aversion <= 0:
        raise ValueError(
            "risk_aversion must be positive."
        )

    return (
        risk_aversion
        * covariance_matrix
        @ market_weights
    )

import numpy as np


def validate_pick_matrix(
    pick_matrix,
    n_assets,
):
    """
    Validate a Black-Litterman pick matrix.

    Parameters
    ----------
    pick_matrix : array-like
        Pick matrix.

    n_assets : int
        Number of assets.

    Returns
    -------
    numpy.ndarray
        Validated pick matrix.
    """
    pick_matrix = np.asarray(
        pick_matrix,
        dtype=float,
    )

    if pick_matrix.ndim != 2:
        raise ValueError(
            "pick_matrix must be two-dimensional."
        )

    if pick_matrix.shape[1] != n_assets:
        raise ValueError(
            "pick_matrix has incorrect number of columns."
        )

    return pick_matrix

def validate_views(
    views,
    n_views,
):
    """
    Validate the view vector.
    """
    views = np.asarray(
        views,
        dtype=float,
    )

    if views.ndim != 1:
        raise ValueError(
            "views must be one-dimensional."
        )

    if len(views) != n_views:
        raise ValueError(
            "Number of views does not match pick matrix."
        )

    return views

def build_pick_matrix(
    rows,
):
    """
    Build a pick matrix from rows.

    Parameters
    ----------
    rows : sequence of array-like

    Returns
    -------
    numpy.ndarray
    """
    pick_matrix = np.asarray(
        rows,
        dtype=float,
    )

    if pick_matrix.ndim != 2:
        raise ValueError(
            "rows must define a two-dimensional matrix."
        )

    return pick_matrix

import numpy as np


def validate_omega(
    omega,
    n_views,
):
    """
    Validate the Black-Litterman Omega matrix.

    Parameters
    ----------
    omega : array-like

    n_views : int

    Returns
    -------
    numpy.ndarray
    """
    omega = np.asarray(
        omega,
        dtype=float,
    )

    if omega.ndim != 2:
        raise ValueError(
            "omega must be two-dimensional."
        )

    if omega.shape != (
        n_views,
        n_views,
    ):
        raise ValueError(
            "omega has incorrect shape."
        )

    if not np.allclose(
        omega,
        omega.T,
    ):
        raise ValueError(
            "omega must be symmetric."
        )

    diagonal = np.diag(
        omega
    )

    if np.any(
        diagonal <= 0
    ):
        raise ValueError(
            "omega must have positive diagonal."
        )

    return omega

def default_omega(
    n_views,
    uncertainty=0.05,
):
    """
    Construct a default diagonal
    Omega matrix.

    Parameters
    ----------
    n_views : int

    uncertainty : float

    Returns
    -------
    numpy.ndarray
    """
    if n_views <= 0:
        raise ValueError(
            "n_views must be positive."
        )

    if uncertainty <= 0:
        raise ValueError(
            "uncertainty must be positive."
        )

    return (
        uncertainty
        * np.eye(
            n_views,
            dtype=float,
        )
    )

import numpy as np


def posterior_expected_returns(
    equilibrium_returns,
    covariance_matrix,
    pick_matrix,
    views,
    omega,
    tau=0.025,
):
    """
    Compute Black-Litterman posterior expected returns.

    Parameters
    ----------
    equilibrium_returns : array-like
    covariance_matrix : array-like
    pick_matrix : array-like
    views : array-like
    omega : array-like
    tau : float

    Returns
    -------
    numpy.ndarray
    """
    pi = np.asarray(
        equilibrium_returns,
        dtype=float,
    )

    sigma = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    P = np.asarray(
        pick_matrix,
        dtype=float,
    )

    Q = np.asarray(
        views,
        dtype=float,
    )

    omega = np.asarray(
        omega,
        dtype=float,
    )

    n_assets = len(pi)

    validate_pick_matrix(
        P,
        n_assets,
    )

    validate_views(
        Q,
        P.shape[0],
    )

    validate_omega(
        omega,
        P.shape[0],
    )

    if sigma.shape != (
        n_assets,
        n_assets,
    ):
        raise ValueError(
            "covariance_matrix has incorrect shape."
        )

    if tau <= 0:
        raise ValueError(
            "tau must be positive."
        )

    A = (
    P @ (tau * sigma) @ P.T
    + omega
    )
    rhs = (
        Q
        - P @ pi
        )
    adjustment = (
    tau
    * sigma
    @ P.T
    @ np.linalg.solve(
        A,
        rhs,
    )
)

    return pi + adjustment