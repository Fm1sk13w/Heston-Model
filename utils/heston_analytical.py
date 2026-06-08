"""
Semi-analytical Heston option pricing via characteristic functions.

European call and put prices are obtained by Fourier inversion of the
affine characteristic function. The implementation follows the stable
complex-log formulation discussed by Albrecher et al. (2007) and uses
the single-integral representation of Lewis (2001).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from utils.black_scholes import black_scholes_price

OptionType = Literal["call", "put"]


def _heston_phi(
    k: complex,
    tau: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
) -> complex:
    """
    Heston characteristic function for the Lewis inversion.

    Parameters correspond to the standard CIR variance dynamics with
    mean-reversion speed kappa, long-run variance theta, vol-of-vol xi,
    correlation rho, and initial variance v0.

    Parameters
    ----------
    k : complex
        Fourier variable (the Lewis integrand evaluates phi at k + i/2).
    tau : float
        Time to maturity.
    v0, theta, kappa, xi, rho : float
        Heston model parameters.

    Returns
    -------
    complex
        Characteristic function value.
    """
    b = kappa + 1j * rho * xi * k
    d = np.sqrt(b**2 + xi**2 * k * (k - 1j))
    g = (b - d) / (b + d)

    term_exp = 1.0 - g * np.exp(-d * tau)
    term_log = 1.0 - g

    T_coef = (b - d) / xi**2
    T_part = T_coef * (1.0 - np.exp(-d * tau)) / term_exp
    W_part = kappa * theta * (
        tau * T_coef - 2.0 * np.log(term_exp / term_log) / xi**2
    )

    return np.exp(W_part + v0 * T_part)


def _lewis_transform(
    tau: float,
    log_forward_moneyness: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
    u_max: float = 120.0,
    n_quad: int = 256,
    method: Literal["gauss", "quad"] = "gauss",
) -> float:
    """
    Evaluate the Lewis Fourier integral for undiscounted call value.

    Parameters
    ----------
    tau : float
        Time to maturity.
    log_forward_moneyness : float
        ln(F/K) with forward F = S exp((r-q)tau); here q = 0.
    v0, theta, kappa, xi, rho : float
        Heston parameters.
    u_max : float
        Truncation of the semi-infinite integration domain.
    n_quad : int
        Gauss-Legendre nodes when method='gauss'.
    method : {'gauss', 'quad'}
        Integration scheme.

    Returns
    -------
    float
        Integral of the Lewis integrand on (0, u_max).
    """

    def integrand(u: float) -> float:
        if u == 0.0:
            return 0.0
        phi = _heston_phi(u + 0.5j, tau, v0, theta, kappa, xi, rho)
        return float(
            2.0
            * np.real(np.exp(-1j * u * log_forward_moneyness) * phi)
            / (u**2 + 0.25)
        )

    if method == "gauss":
        nodes, weights = leggauss(n_quad)
        u = 0.5 * u_max * (nodes + 1.0)
        w = 0.5 * u_max * weights
        values = np.array([integrand(ui) for ui in u])
        return float(np.dot(w, values))

    integral, _ = quad(
        integrand,
        0.0,
        u_max,
        limit=200,
        epsabs=1e-10,
        epsrel=1e-10,
    )
    return float(integral)


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    u_max: float = 120.0,
    n_quad: int = 256,
    method: Literal["gauss", "quad"] = "gauss",
) -> float:
    """
    Semi-analytical Heston call price via Lewis Fourier inversion.

    The undiscounted forward call is
        C_fwd = F - sqrt(FK)/(2 pi) * integral,
    and the spot call is obtained by discounting: C = exp(-rT) C_fwd.

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    T : float
        Maturity in years.
    r : float
        Risk-free rate (continuous compounding, zero dividends).
    v0, kappa, theta, xi, rho : float
        Heston model parameters.
    u_max : float
        Upper truncation of the Fourier integral.
    n_quad : int
        Number of Gauss-Legendre nodes.
    method : {'gauss', 'quad'}
        Integration rule.

    Returns
    -------
    float
        European call price.
    """
    if T <= 0.0 or S0 <= 0.0 or K <= 0.0:
        raise ValueError("S0, K, and T must be positive.")

    forward = S0 * math.exp(r * T)
    log_moneyness = math.log(forward / K)
    integral = _lewis_transform(
        T, log_moneyness, v0, theta, kappa, xi, rho, u_max, n_quad, method
    )
    undiscounted_call = forward - math.sqrt(forward * K) / (2.0 * math.pi) * integral
    return math.exp(-r * T) * max(undiscounted_call, 0.0)


def heston_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    **kwargs,
) -> float:
    """
    European put from the Fourier call formula and put-call parity.

    P = C - S0 + K exp(-rT).
    """
    call = heston_call_price(
        S0, K, T, r, v0, kappa, theta, xi, rho, **kwargs
    )
    return call - S0 + K * math.exp(-r * T)


def heston_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: OptionType = "call",
    **kwargs,
) -> float:
    """Dispatch to call or put semi-analytical Heston pricing."""
    if option_type == "call":
        return heston_call_price(
            S0, K, T, r, v0, kappa, theta, xi, rho, **kwargs
        )
    if option_type == "put":
        return heston_put_price(
            S0, K, T, r, v0, kappa, theta, xi, rho, **kwargs
        )
    raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_limit_error(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kappa: float,
    rho: float = 0.0,
    xi_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure |C_Heston - C_BS| as volatility-of-volatility xi tends to zero.

    With v0 = theta = sigma^2 and rho = 0, the Heston model converges to
    Black-Scholes with constant volatility sigma.

    Parameters
    ----------
    S0, K, T, r, sigma : float
        Market inputs; sigma is the BS volatility used in the limit.
    kappa : float
        Mean-reversion speed.
    rho : float
        Spot-variance correlation (0 for a clean limit).
    xi_values : ndarray, optional
        Grid of xi values.

    Returns
    -------
    xi_values, abs_errors
        Arrays of xi and corresponding absolute pricing errors.
    """
    v0 = theta = sigma**2
    if xi_values is None:
        xi_values = np.logspace(-4, 0, 25)

    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type="call")
    errors = np.array(
        [
            abs(
                heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho)
                - bs_price
            )
            for xi in xi_values
        ]
    )
    return xi_values, errors
