"""
Heston and Black--Scholes calibration to market option quotes.

The Heston objective uses the semi-analytical Fourier pricer from
:mod:`heston_analytical`. Optimisation is performed with Nelder--Mead as
specified in Issue #6.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from utils.black_scholes import black_scholes_call
from utils.heston_analytical import heston_call_price
from utils.market_data import OptionChain


@dataclass(frozen=True)
class HestonCalibrationResult:
    """Calibrated Heston parameters and in-sample fit quality."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    rmse: float
    mae: float
    max_abs_error: float
    feller_ratio: float
    success: bool
    message: str
    n_evaluations: int


@dataclass(frozen=True)
class BlackScholesCalibrationResult:
    """Flat volatility Black--Scholes fit to the same quote set."""

    sigma: float
    rmse: float
    mae: float
    max_abs_error: float


def _rmse(pred: np.ndarray, market: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - market) ** 2)))


def _error_stats(pred: np.ndarray, market: np.ndarray) -> tuple[float, float, float]:
    err = pred - market
    return (
        _rmse(pred, market),
        float(np.mean(np.abs(err))),
        float(np.max(np.abs(err))),
    )


def heston_model_prices(
    chain: OptionChain,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    *,
    n_quad: int = 128,
) -> np.ndarray:
    """Vector of Heston call prices for all strikes in ``chain``."""
    return np.array(
        [
            heston_call_price(
                chain.spot,
                K,
                chain.maturity,
                chain.rate,
                v0,
                kappa,
                theta,
                xi,
                rho,
                n_quad=n_quad,
            )
            for K in chain.strikes
        ],
        dtype=float,
    )


def heston_calibration_objective(
    params: np.ndarray,
    chain: OptionChain,
    *,
    n_quad: int = 128,
    feller_penalty: float = 0.0,
) -> float:
    """
    RMSE between market mids and Heston Fourier prices.

    Optional quadratic penalty when the Feller condition is violated.
    """
    v0, kappa, theta, xi, rho = params
    if v0 <= 0.0 or kappa <= 0.0 or theta <= 0.0 or xi <= 0.0:
        return 1e6
    if abs(rho) >= 1.0:
        return 1e6

    model = heston_model_prices(chain, v0, kappa, theta, xi, rho, n_quad=n_quad)
    loss = _rmse(model, chain.mid_prices)

    if feller_penalty > 0.0:
        feller = 2.0 * kappa * theta / (xi**2)
        if feller < 1.0:
            loss += feller_penalty * (1.0 - feller) ** 2
    return loss


def calibrate_heston(
    chain: OptionChain,
    *,
    x0: tuple[float, float, float, float, float] | None = None,
    n_quad: int = 128,
    feller_penalty: float = 1.0,
    maxiter: int = 800,
) -> HestonCalibrationResult:
    """
    Calibrate (v0, kappa, theta, xi, rho) with Nelder--Mead.

    Parameters
    ----------
    chain : OptionChain
        Market quotes.
    x0 : tuple, optional
        Initial guess; defaults to a literature-style SPY starting point.
    n_quad : int
        Fourier quadrature points inside the objective (lower for speed).
    feller_penalty : float
        Weight on Feller-condition violation (0 disables the penalty).
    maxiter : int
        Maximum Nelder--Mead iterations.
    """
    if x0 is None:
        x0 = (0.04, 2.0, 0.04, 0.4, -0.7)

    objective = partial(
        heston_calibration_objective,
        chain=chain,
        n_quad=n_quad,
        feller_penalty=feller_penalty,
    )
    result = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6},
    )

    v0, kappa, theta, xi, rho = result.x
    model = heston_model_prices(chain, v0, kappa, theta, xi, rho, n_quad=256)
    rmse, mae, max_abs = _error_stats(model, chain.mid_prices)
    feller = 2.0 * kappa * theta / (xi**2)

    return HestonCalibrationResult(
        v0=float(v0),
        kappa=float(kappa),
        theta=float(theta),
        xi=float(xi),
        rho=float(rho),
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs,
        feller_ratio=float(feller),
        success=bool(result.success),
        message=str(result.message),
        n_evaluations=int(result.nfev),
    )


def calibrate_black_scholes(chain: OptionChain) -> BlackScholesCalibrationResult:
    """Calibrate a single flat volatility to minimise RMSE on the quote set."""

    def objective(sigma: float) -> float:
        if sigma <= 0.0:
            return 1e6
        model = np.array(
            [
                black_scholes_call(chain.spot, K, chain.maturity, chain.rate, sigma)
                for K in chain.strikes
            ]
        )
        return _rmse(model, chain.mid_prices)

    opt = minimize_scalar(objective, bounds=(0.05, 1.50), method="bounded")
    sigma = float(opt.x)
    model = np.array(
        [
            black_scholes_call(chain.spot, K, chain.maturity, chain.rate, sigma)
            for K in chain.strikes
        ]
    )
    rmse, mae, max_abs = _error_stats(model, chain.mid_prices)
    return BlackScholesCalibrationResult(
        sigma=sigma, rmse=rmse, mae=mae, max_abs_error=max_abs
    )
