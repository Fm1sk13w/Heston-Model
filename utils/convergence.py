"""
Convergence diagnostics for the Heston ADI solver.

Discrete L2 and L-infinity norms compare the PDE solution against the
Fourier benchmark on the same (x, v) grid. Empirical convergence orders
are estimated from successive refinements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.grid_builder import HestonSpatialGrid
from utils.heston_adi_solver import HestonModelParams, solve_heston_adi
from utils.heston_analytical import heston_call_price


@dataclass(frozen=True)
class ErrorNorms:
    """Discrete error norms on the interior of a (x, v) grid."""

    l2: float
    linf: float
    rmse: float
    n_nodes: int


def trapezoidal_weights(nodes: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a 1D non-uniform node set."""
    n = nodes.size
    if n < 2:
        raise ValueError("At least two nodes are required.")
    weights = np.empty(n)
    weights[0] = 0.5 * (nodes[1] - nodes[0])
    weights[-1] = 0.5 * (nodes[-1] - nodes[-2])
    if n > 2:
        weights[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    return weights


def grid_quadrature_weights(grid: HestonSpatialGrid) -> np.ndarray:
    """Outer product of 1D trapezoidal weights on the tensor-product grid."""
    wx = trapezoidal_weights(grid.x)
    wv = trapezoidal_weights(grid.v)
    return np.outer(wx, wv)


def interior_mask(
    grid: HestonSpatialGrid,
    *,
    exclude_x_boundaries: bool = True,
    exclude_v0: bool = True,
    exclude_v_max: bool = False,
) -> np.ndarray:
    """Boolean mask selecting interior nodes for error evaluation."""
    mask = np.ones((grid.nx, grid.nv), dtype=bool)
    if exclude_x_boundaries:
        mask[0, :] = False
        mask[-1, :] = False
    if exclude_v0:
        mask[:, 0] = False
    if exclude_v_max:
        mask[:, -1] = False
    return mask


def feller_ratio(kappa: float, theta: float, xi: float) -> float:
    """Return 2*kappa*theta / xi^2 (Feller condition holds when ratio >= 1)."""
    return 2.0 * kappa * theta / (xi**2)


def fourier_surface_on_grid(
    grid: HestonSpatialGrid,
    params: HestonModelParams,
    T: float,
    *,
    n_quad: int = 128,
) -> np.ndarray:
    """
    Evaluate the Fourier call price on every node of ``grid``.

    At node (x_i, v_j) the reference price uses spot S = K exp(x_i) and
    initial variance v_j.
    """
    surface = np.empty((grid.nx, grid.nv))
    K = params.K
    for i, x in enumerate(grid.x):
        spot = K * np.exp(x)
        for j, v in enumerate(grid.v):
            surface[i, j] = heston_call_price(
                spot,
                K,
                T,
                params.r,
                v,
                params.kappa,
                params.theta,
                params.xi,
                params.rho,
                n_quad=n_quad,
            )
    return surface


def error_norms(
    u_adi: np.ndarray,
    u_ref: np.ndarray,
    grid: HestonSpatialGrid,
    mask: np.ndarray | None = None,
) -> ErrorNorms:
    """
    Compute weighted discrete L2, L-infinity, and RMS errors.

    The L2 norm uses trapezoidal weights on the masked node set.
    """
    if mask is None:
        mask = interior_mask(grid)
    err = (u_adi - u_ref)[mask]
    if err.size == 0:
        raise ValueError("Error mask is empty.")

    weights = grid_quadrature_weights(grid)[mask]
    weight_sum = weights.sum()
    l2 = float(np.sqrt(np.sum(weights * err**2) / weight_sum))
    linf = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return ErrorNorms(l2=l2, linf=linf, rmse=rmse, n_nodes=int(err.size))


def characteristic_mesh_size(grid: HestonSpatialGrid) -> float:
    """Mesh-size proxy h = sqrt(h_x * h_v) with mean spacings."""
    hx = float(np.mean(grid.dx_forward))
    hv = float(np.mean(grid.dv_forward))
    return float(np.sqrt(hx * hv))


def empirical_orders(
    errors: np.ndarray,
    mesh_sizes: np.ndarray,
) -> np.ndarray:
    """
    Estimate convergence order p between successive levels:

        p_k = log(e_k / e_{k+1}) / log(h_k / h_{k+1}).
    """
    errors = np.asarray(errors, dtype=float)
    mesh_sizes = np.asarray(mesh_sizes, dtype=float)
    if errors.size != mesh_sizes.size:
        raise ValueError("errors and mesh_sizes must have the same length.")
    if errors.size < 2:
        return np.array([])

    orders = np.full(errors.size - 1, np.nan)
    for k in range(errors.size - 1):
        if errors[k] <= 0.0 or errors[k + 1] <= 0.0:
            continue
        ratio_e = errors[k] / errors[k + 1]
        ratio_h = mesh_sizes[k] / mesh_sizes[k + 1]
        if ratio_e > 0.0 and ratio_h > 0.0:
            orders[k] = float(np.log(ratio_e) / np.log(ratio_h))
    return orders


def run_adi_with_reference(
    params: HestonModelParams,
    T: float,
    *,
    nx: int,
    nv: int,
    n_steps: int,
    rannacher_steps: int = 10,
    n_quad: int = 128,
    grid: HestonSpatialGrid | None = None,
) -> tuple[np.ndarray, np.ndarray, HestonSpatialGrid, ErrorNorms]:
    """
    Solve the ADI scheme and return the solution, Fourier reference, grid, and norms.
    """
    if grid is None:
        from utils.grid_builder import build_heston_grid

        grid = build_heston_grid(nx=nx, nv=nv)

    u_adi, grid, _ = solve_heston_adi(
        params,
        T,
        grid=grid,
        n_steps=n_steps,
        rannacher_steps=rannacher_steps,
    )
    u_ref = fourier_surface_on_grid(grid, params, T, n_quad=n_quad)
    norms = error_norms(u_adi, u_ref, grid)
    return u_adi, u_ref, grid, norms


def market_pricing_errors(
    strikes: np.ndarray,
    market_prices: np.ndarray,
    model_prices: np.ndarray,
) -> dict[str, float]:
    """Summary statistics for model-vs-market pricing residuals."""
    residuals = model_prices - market_prices
    abs_err = np.abs(residuals)
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
        "mean_signed": float(np.mean(residuals)),
    }
