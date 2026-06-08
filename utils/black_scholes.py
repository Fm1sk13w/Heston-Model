"""
Black-Scholes analytical pricing and implied volatility inversion.

Provides the baseline reference model for the Heston PDE project. Under
constant volatility sigma, European option prices satisfy a one-dimensional
parabolic PDE that admits a closed-form solution.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


def norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard normal cumulative distribution function N(x).

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).

    Returns
    -------
    float or ndarray
        Probability P(Z <= x) for Z ~ N(0, 1).
    """
    return norm.cdf(x)


def norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """
    Standard normal probability density function.

    Parameters
    ----------
    x : float or ndarray
        Evaluation point(s).

    Returns
    -------
    float or ndarray
        Density phi(x) = (2 pi)^(-1/2) exp(-x^2 / 2).
    """
    return norm.pdf(x)


def _d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> tuple[float, float]:
    """
    Compute Black-Scholes d1 and d2 parameters.

    Parameters
    ----------
    S, K : float
        Spot price and strike.
    T : float
        Time to maturity in years (must be positive).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Constant volatility (must be positive).

    Returns
    -------
    tuple[float, float]
        (d1, d2) as defined in the Black-Scholes formulas.
    """
    if T <= 0.0:
        raise ValueError("Time to maturity T must be positive.")
    if sigma <= 0.0:
        raise ValueError("Volatility sigma must be positive.")
    if S <= 0.0 or K <= 0.0:
        raise ValueError("Spot S and strike K must be positive.")

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """
    Analytical Black-Scholes price of a European call or put.

    The price solves the backward Kolmogorov PDE
        V_t + (1/2) sigma^2 S^2 V_SS + r S V_S - r V = 0
    with terminal payoff max(±(S - K), 0).

    Parameters
    ----------
    S, K : float
        Spot price and strike.
    T : float
        Time to maturity in years.
    r : float
        Risk-free rate.
    sigma : float
        Constant volatility.
    option_type : {'call', 'put'}
        Option contract type.

    Returns
    -------
    float
        Present value of the European option.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    if option_type == "put":
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call price under Black-Scholes."""
    return black_scholes_price(S, K, T, r, sigma, option_type="call")


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price under Black-Scholes."""
    return black_scholes_price(S, K, T, r, sigma, option_type="put")


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes vega (sensitivity of call price to volatility).

    Vega is identical for calls and puts with the same parameters:
        Vega = S * phi(d1) * sqrt(T).

    Used as the derivative in Newton-Raphson implied-volatility inversion.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard Black-Scholes inputs.

    Returns
    -------
    float
        Partial derivative dC/dsigma.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm_pdf(d1) * math.sqrt(T)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    initial_guess: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Recover implied volatility from a market price via Newton-Raphson.

    Solves C_BS(sigma) = market_price (or the put analogue) for sigma,
    using the analytical vega as the derivative. A small floor on vega
    avoids division by zero in nearly degenerate cases.

    Parameters
    ----------
    market_price : float
        Observed option price.
    S, K, T, r : float
        Market inputs.
    option_type : {'call', 'put'}
        Contract type used for inversion.
    initial_guess : float
        Starting value for sigma.
    tol : float
        Convergence tolerance on the price residual.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    float
        Implied volatility sigma such that BS price matches market_price.

    Raises
    ------
    RuntimeError
        If Newton-Raphson fails to converge within max_iter steps.
    ValueError
        If market_price is non-positive or inputs are invalid.
    """
    if market_price <= 0.0:
        raise ValueError("market_price must be positive for implied-vol inversion.")
    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        raise ValueError("S, K, and T must be positive.")

    sigma = initial_guess

    for _ in range(max_iter):
        model_price = black_scholes_price(S, K, T, r, sigma, option_type)
        diff = model_price - market_price

        if abs(diff) < tol:
            return sigma

        vega = black_scholes_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            raise RuntimeError("Vega too small; implied volatility is ill-conditioned.")

        sigma -= diff / vega

        # Keep iterates in a reasonable range for numerical stability.
        sigma = float(np.clip(sigma, 1e-6, 5.0))

    raise RuntimeError(
        f"Implied volatility did not converge in {max_iter} iterations "
        f"(last residual: {diff:.2e})."
    )


def put_call_parity_check(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    tol: float = 1e-10,
) -> bool:
    """
    Verify put-call parity: C - P = S - K exp(-rT).

    Parameters
    ----------
    call_price, put_price : float
        Observed or model prices.
    S, K, T, r : float
        Market inputs.
    tol : float
        Acceptable absolute error.

    Returns
    -------
    bool
        True if parity holds within tolerance.
    """
    lhs = call_price - put_price
    rhs = S - K * math.exp(-r * T)
    return abs(lhs - rhs) < tol
