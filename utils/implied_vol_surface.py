"""
Implied-volatility surfaces from market quotes and model prices.

Black--Scholes implied volatilities are recovered by Newton--Raphson
(:func:`utils.black_scholes.implied_volatility`). Scattered quote clouds
are interpolated onto a regular $(K, T)$ grid for 3D visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from utils.black_scholes import black_scholes_call, implied_volatility
from utils.heston_adi_solver import HestonModelParams, prices_along_strikes, solve_heston_adi
from utils.heston_analytical import heston_call_price

# ADI defaults for IV surfaces (finer grid, maturity-scaled time stepping).
ADI_NX = 81
ADI_NV = 41
ADI_N_STEPS_MIN = 400
FOURIER_N_QUAD = 256


@dataclass(frozen=True)
class IVSurface:
    """Implied-volatility surface on a $(K, T)$ tensor-product grid."""

    strikes: np.ndarray
    maturities: np.ndarray
    iv: np.ndarray
    label: str

    @property
    def strike_grid(self) -> np.ndarray:
        """Meshgrid strikes, shape (n_T, n_K)."""
        k, _ = np.meshgrid(self.strikes, self.maturities)
        return k

    @property
    def maturity_grid(self) -> np.ndarray:
        """Meshgrid maturities, shape (n_T, n_K)."""
        _, t = np.meshgrid(self.strikes, self.maturities)
        return t


def implied_volatility_safe(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    *,
    initial_guess: float = 0.2,
) -> float:
    """Return BS implied vol or ``np.nan`` when inversion fails."""
    if price <= 0.0 or maturity <= 0.0:
        return float("nan")
    try:
        return implied_volatility(
            price, spot, strike, maturity, rate, option_type="call", initial_guess=initial_guess
        )
    except (RuntimeError, ValueError):
        return float("nan")


def iv_from_prices(
    prices: np.ndarray,
    spot: float,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    *,
    min_price: float = 0.05,
    initial_guess: float = 0.2,
) -> np.ndarray:
    """
    Vectorised implied-vol inversion along a strike vector at fixed maturity.

    Uses a chained initial guess along strikes and skips deep OTM quotes whose
    mids are below ``min_price`` (IV inversion is ill-conditioned there).
    """
    iv_out = np.empty(strikes.size, dtype=float)
    guess = initial_guess
    for i, (p, k) in enumerate(zip(prices, strikes)):
        if p < min_price:
            iv_out[i] = np.nan
            continue
        sigma = implied_volatility_safe(p, spot, k, maturity, rate, initial_guess=guess)
        iv_out[i] = sigma
        if np.isfinite(sigma):
            guess = sigma
    return iv_out


def quotes_to_iv_points(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute implied volatilities at each quote row.

    Expects columns ``mid``, ``strike``, ``T``, ``spot``, ``r``. Rows are
    processed maturity-by-maturity with strikes sorted for stable inversion.
    """
    strikes_list: list[float] = []
    maturities_list: list[float] = []
    iv_list: list[float] = []

    for T, group in df.groupby("T"):
        sub = group.sort_values("strike")
        guess = 0.2
        for row in sub.itertuples(index=False):
            sigma = implied_volatility_safe(
                float(row.mid),
                float(row.spot),
                float(row.strike),
                float(row.T),
                float(row.r),
                initial_guess=guess,
            )
            if np.isfinite(sigma):
                guess = sigma
            strikes_list.append(float(row.strike))
            maturities_list.append(float(T))
            iv_list.append(sigma)

    return (
        np.asarray(strikes_list, dtype=float),
        np.asarray(maturities_list, dtype=float),
        np.asarray(iv_list, dtype=float),
    )


def select_slice_maturity(df: pd.DataFrame, *, min_quotes: int = 20) -> float:
    """Pick a maturity with at least ``min_quotes`` OTM calls (most quotes wins)."""
    counts = df.groupby("T").size()
    eligible = counts[counts >= min_quotes]
    if eligible.empty:
        return float(counts.idxmax())
    return float(eligible.idxmax())


def market_iv_slice(df: pd.DataFrame, maturity: float, *, tol: float = 1e-6) -> pd.DataFrame:
    """Raw market quotes and inverted IV at a single expiry."""
    mask = np.isclose(df["T"].to_numpy(dtype=float), maturity, rtol=0.0, atol=tol)
    sub = df.loc[mask].sort_values("strike").copy()
    if sub.empty:
        raise ValueError(f"No quotes at maturity T={maturity}")

    spot = float(sub["spot"].iloc[0])
    rate = float(sub["r"].iloc[0])
    sub["iv_market"] = iv_from_prices(
        sub["mid"].to_numpy(dtype=float),
        spot,
        sub["strike"].to_numpy(dtype=float),
        maturity,
        rate,
    )
    return sub


def model_iv_at_strikes(
    spot: float,
    rate: float,
    strikes: np.ndarray,
    maturity: float,
    *,
    sigma_bs: float,
    heston_params: tuple[float, float, float, float, float],
    adi_params: HestonModelParams | None = None,
    include_adi: bool = False,
    min_price: float = 0.05,
) -> dict[str, np.ndarray]:
    """Model implied vols on the same strikes as a market smile slice."""
    v0, kappa, theta, xi, rho = heston_params

    fourier_prices = np.array(
        [
            heston_call_price(spot, K, maturity, rate, v0, kappa, theta, xi, rho, n_quad=FOURIER_N_QUAD)
            for K in strikes
        ]
    )
    out: dict[str, np.ndarray] = {
        "black_scholes": np.full(strikes.size, sigma_bs, dtype=float),
        "heston_fourier": iv_from_prices(
            fourier_prices, spot, strikes, maturity, rate, min_price=min_price
        ),
    }

    if include_adi and adi_params is not None:
        n_steps = max(ADI_N_STEPS_MIN, int(maturity * 8000))
        u, grid, _ = solve_heston_adi(
            adi_params,
            maturity,
            nx=ADI_NX,
            nv=ADI_NV,
            n_steps=n_steps,
            rannacher_steps=min(8, n_steps // 4),
        )
        adi_prices = prices_along_strikes(u, grid, strikes, v0, spot)
        out["heston_adi"] = iv_from_prices(
            adi_prices, spot, strikes, maturity, rate, min_price=min_price
        )

    return out


def interpolate_iv_surface(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv: np.ndarray,
    *,
    n_strikes: int = 30,
    n_maturities: int = 12,
    strike_pad: float = 0.0,
) -> IVSurface:
    """
    Interpolate scattered (K, T, sigma) points onto a regular grid.

    Parameters
    ----------
    strikes, maturities, iv : ndarray
        Scattered surface samples (``iv`` may contain NaNs which are dropped).
    n_strikes, n_maturities : int
        Output grid resolution.
    strike_pad : float
        Fractional padding of the strike range beyond observed min/max.
    """
    mask = np.isfinite(iv) & (iv > 0.0)
    if mask.sum() < 4:
        raise ValueError("Not enough valid IV points to interpolate a surface.")

    k_obs = strikes[mask]
    t_obs = maturities[mask]
    iv_obs = iv[mask]

    k_min, k_max = k_obs.min(), k_obs.max()
    pad = strike_pad * (k_max - k_min)
    strike_grid = np.linspace(k_min - pad, k_max + pad, n_strikes)
    mat_grid = np.linspace(t_obs.min(), t_obs.max(), n_maturities)
    kk, tt = np.meshgrid(strike_grid, mat_grid)

    iv_grid = griddata((k_obs, t_obs), iv_obs, (kk, tt), method="linear")
    return IVSurface(strikes=strike_grid, maturities=mat_grid, iv=iv_grid, label="interpolated")


def market_iv_surface(df: pd.DataFrame, **grid_kwargs) -> IVSurface:
    """Build the market implied-volatility surface from quote mids."""
    k, t, iv = quotes_to_iv_points(df)
    surface = interpolate_iv_surface(k, t, iv, **grid_kwargs)
    return IVSurface(
        strikes=surface.strikes,
        maturities=surface.maturities,
        iv=surface.iv,
        label="Market",
    )


def black_scholes_iv_surface(
    sigma: float,
    spot: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
) -> IVSurface:
    """Flat Black--Scholes implied-volatility plane at ``sigma``."""
    iv = np.full((maturities.size, strikes.size), sigma, dtype=float)
    return IVSurface(strikes=strikes, maturities=maturities, iv=iv, label="Black-Scholes")


def heston_fourier_iv_surface(
    spot: float,
    rate: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> IVSurface:
    """Heston Fourier prices inverted to BS implied vol on a $(K,T)$ grid."""
    iv = np.zeros((maturities.size, strikes.size))
    for i, T in enumerate(maturities):
        prices = np.array(
            [
                heston_call_price(spot, K, T, rate, v0, kappa, theta, xi, rho, n_quad=FOURIER_N_QUAD)
                for K in strikes
            ]
        )
        iv[i, :] = iv_from_prices(prices, spot, strikes, T, rate)
    return IVSurface(strikes=strikes, maturities=maturities, iv=iv, label="Heston Fourier")


def heston_adi_iv_surface(
    spot: float,
    rate: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    params: HestonModelParams,
    v0: float,
    *,
    nx: int = ADI_NX,
    nv: int = ADI_NV,
    n_steps: int | None = None,
) -> IVSurface:
    """Heston ADI prices inverted to BS implied vol (one PDE solve per maturity)."""
    iv = np.zeros((maturities.size, strikes.size))
    grid = None
    for i, T in enumerate(maturities):
        steps = n_steps if n_steps is not None else max(ADI_N_STEPS_MIN, int(T * 8000))
        u, grid, _ = solve_heston_adi(
            params,
            T,
            grid=grid,
            nx=nx if grid is None else grid.nx,
            nv=nv if grid is None else grid.nv,
            n_steps=steps,
            rannacher_steps=min(8, steps // 4),
        )
        prices = prices_along_strikes(u, grid, strikes, v0, spot)
        iv[i, :] = iv_from_prices(prices, spot, strikes, T, rate)
    return IVSurface(strikes=strikes, maturities=maturities, iv=iv, label="Heston ADI")


def aligned_model_surfaces(
    market_surface: IVSurface,
    *,
    spot: float,
    rate: float,
    sigma_bs: float,
    heston_params: tuple[float, float, float, float, float],
    adi_params: HestonModelParams | None = None,
    v0: float | None = None,
    include_adi: bool = True,
) -> dict[str, IVSurface]:
    """Build BS and Heston surfaces on the same grid as ``market_surface``."""
    k = market_surface.strikes
    t = market_surface.maturities
    v0, kappa, theta, xi, rho = heston_params
    if v0 is None:
        v0 = heston_params[0]

    surfaces = {
        "market": market_surface,
        "black_scholes": black_scholes_iv_surface(sigma_bs, spot, k, t),
        "heston_fourier": heston_fourier_iv_surface(
            spot, rate, k, t, v0, kappa, theta, xi, rho
        ),
    }
    if include_adi and adi_params is not None:
        surfaces["heston_adi"] = heston_adi_iv_surface(
            spot, rate, k, t, adi_params, v0
        )
    return surfaces
