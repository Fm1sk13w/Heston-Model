"""
Alternating Direction Implicit (ADI) finite-difference solver for the Heston PDE.

The log-price formulation is marched forward in reversed time tau = T - t using
the Hundsdorfer-Verwer ADI scheme. Spatial operators come from
:mod:`grid_builder`; the mixed derivative is treated explicitly in the
predictor step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded

from utils.grid_builder import (
    HestonSpatialGrid,
    build_heston_grid,
    build_operators_with_boundary_conditions,
)


@dataclass(frozen=True)
class HestonModelParams:
    """Heston model parameters for the ADI solver."""

    r: float
    kappa: float
    theta: float
    xi: float
    rho: float
    K: float = 100.0


def call_terminal_payoff(grid: HestonSpatialGrid, K: float) -> np.ndarray:
    """
    European call payoff on the (x, v) grid at tau = 0.

    With x = ln(S/K), the payoff is max(S - K, 0) = K max(e^x - 1, 0).
    """
    x = grid.x[:, np.newaxis]
    payoff_x = K * np.maximum(np.exp(x) - 1.0, 0.0)
    return np.broadcast_to(payoff_x, (grid.nx, grid.nv)).copy()


def x_boundary_values(
    grid: HestonSpatialGrid,
    params: HestonModelParams,
    tau: float,
    T: float,
) -> tuple[float, float]:
    """Dirichlet values at x_min and x_max for a European call."""
    remaining = max(T - tau, 0.0)
    u_min = 0.0
    u_max = params.K * np.exp(grid.x[-1]) - params.K * np.exp(-params.r * remaining)
    return u_min, u_max


def apply_x_boundary(
    u: np.ndarray,
    grid: HestonSpatialGrid,
    params: HestonModelParams,
    tau: float,
    T: float,
) -> np.ndarray:
    """Enforce Dirichlet conditions in the x-direction."""
    u = u.copy()
    u_min, u_max = x_boundary_values(grid, params, tau, T)
    u[0, :] = u_min
    u[-1, :] = u_max
    return u


def _apply_v_operator_row(
    row: np.ndarray,
    D_v: sparse.csr_matrix,
    D_vv: sparse.csr_matrix,
    grid: HestonSpatialGrid,
    params: HestonModelParams,
) -> np.ndarray:
    """v-direction part of the spatial operator at fixed log-moneyness."""
    row_vec = np.asarray(row, dtype=float).ravel()
    dv = D_v @ row_vec
    dvv = D_vv @ row_vec
    v = grid.v
    return params.kappa * (params.theta - v) * dv + 0.5 * params.xi**2 * v * dvv


def _apply_cross_derivative(
    u: np.ndarray,
    D_x: sparse.csr_matrix,
    D_v: sparse.csr_matrix,
) -> np.ndarray:
    """Mixed derivative u_xv via D_x @ (U @ D_v^T) on the tensor grid."""
    return D_x @ (u @ D_v.T)


def spatial_operator(
    u: np.ndarray,
    grid: HestonSpatialGrid,
    D_x: sparse.csr_matrix,
    D_v: sparse.csr_matrix,
    D_xx: sparse.csr_matrix,
    D_vv: sparse.csr_matrix,
    params: HestonModelParams,
) -> np.ndarray:
    """
    Evaluate A u for the semi-discrete Heston PDE in tau-time.

    du/dtau = A u with the log-price convection, diffusion, cross, and reaction
    terms from eq. (heston-pde-x) in the project report.
    """
    v = grid.v[np.newaxis, :]
    conv_x = (params.r - 0.5 * v) * (D_x @ u) + 0.5 * v * (D_xx @ u)
    conv_v = np.zeros_like(u)
    for i in range(u.shape[0]):
        conv_v[i, :] = _apply_v_operator_row(u[i, :], D_v, D_vv, grid, params)
    cross = (
        params.rho
        * params.xi
        * grid.v[np.newaxis, :]
        * _apply_cross_derivative(u, D_x, D_v)
    )
    return conv_x + conv_v + cross - params.r * u


def _tridiagonal_from_sparse(
    mat: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract lower, main, and upper diagonals from a tridiagonal sparse matrix."""
    n = mat.shape[0]
    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)
    for i in range(n):
        main[i] = mat[i, i]
        if i > 0:
            lower[i] = mat[i, i - 1]
        if i < n - 1:
            upper[i] = mat[i, i + 1]
    return lower, main, upper


def _solve_tridiagonal(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system with scipy.linalg.solve_banded."""
    n = main.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = main
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


def _impose_dirichlet_rows(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    left_value: float,
    right_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replace first and last rows with identity rows for Dirichlet endpoints."""
    lower = lower.copy()
    main = main.copy()
    upper = upper.copy()
    rhs = rhs.copy()

    if main.size == 1:
        main[0] = 1.0
        rhs[0] = left_value
        return lower, main, upper, rhs

    main[0] = 1.0
    lower[0] = 0.0
    upper[0] = 0.0
    rhs[0] = left_value

    n = main.size
    main[-1] = 1.0
    lower[-1] = 0.0
    upper[-1] = 0.0
    rhs[-1] = right_value

    return lower, main, upper, rhs


def _build_x_implicit_matrix(
    D_x: sparse.csr_matrix,
    D_xx: sparse.csr_matrix,
    v_j: float,
    params: HestonModelParams,
    dt: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build I - (theta/2) dt L_x for the HV implicit x-solve at variance v_j."""
    n = D_x.shape[0]
    Lx = (params.r - 0.5 * v_j) * D_x + 0.5 * v_j * D_xx - params.r * sparse.eye(n)
    lower, main, upper = _tridiagonal_from_sparse(Lx.tocsr())
    fac = theta * dt
    main = 1.0 - fac * main
    lower = -fac * lower
    upper = -fac * upper
    return lower, main, upper


def _build_v_implicit_matrix(
    D_v: sparse.csr_matrix,
    D_vv: sparse.csr_matrix,
    grid: HestonSpatialGrid,
    params: HestonModelParams,
    dt: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build I - (theta/2) dt L_v for the HV implicit v-solve."""
    n = D_v.shape[0]
    v = grid.v
    Lv = sparse.diags(params.kappa * (params.theta - v)) @ D_v + sparse.diags(
        0.5 * params.xi**2 * v
    ) @ D_vv
    lower, main, upper = _tridiagonal_from_sparse(Lv.tocsr())
    fac = theta * dt
    main = 1.0 - fac * main
    lower = -fac * lower
    upper = -fac * upper
    return lower, main, upper


def _x_direction_operator(
    u: np.ndarray,
    D_x: sparse.csr_matrix,
    D_xx: sparse.csr_matrix,
    grid: HestonSpatialGrid,
    params: HestonModelParams,
) -> np.ndarray:
    """Apply only x-direction terms of A."""
    v = grid.v[np.newaxis, :]
    return (params.r - 0.5 * v) * (D_x @ u) + 0.5 * v * (D_xx @ u) - params.r * u


def _v_direction_operator(
    u: np.ndarray,
    D_v: sparse.csr_matrix,
    D_vv: sparse.csr_matrix,
    grid: HestonSpatialGrid,
    params: HestonModelParams,
) -> np.ndarray:
    """Apply only v-direction terms of A."""
    out = np.zeros_like(u)
    for i in range(u.shape[0]):
        out[i, :] = _apply_v_operator_row(u[i, :], D_v, D_vv, grid, params)
    return out


def hv_adi_step(
    u: np.ndarray,
    grid: HestonSpatialGrid,
    D_x: sparse.csr_matrix,
    D_v: sparse.csr_matrix,
    D_xx: sparse.csr_matrix,
    D_vv: sparse.csr_matrix,
    params: HestonModelParams,
    dt: float,
    tau: float,
    T: float,
    theta: float = 0.5,
) -> np.ndarray:
    """
    Advance the solution one ADI time step with the Hundsdorfer-Verwer scheme.
    """
    u_left, u_right = x_boundary_values(grid, params, tau, T)

    a_u = spatial_operator(u, grid, D_x, D_v, D_xx, D_vv, params)
    phi0 = u + dt * a_u

    ax_u = _x_direction_operator(u, D_x, D_xx, grid, params)
    phi1 = phi0.copy()
    for j in range(grid.nv):
        lower, main, upper = _build_x_implicit_matrix(
            D_x, D_xx, grid.v[j], params, dt, theta
        )
        rhs = phi0[:, j] - theta * dt * ax_u[:, j]
        lower, main, upper, rhs = _impose_dirichlet_rows(
            lower, main, upper, rhs, u_left, u_right
        )
        phi1[:, j] = _solve_tridiagonal(lower, main, upper, rhs)

    av_u = _v_direction_operator(u, D_v, D_vv, grid, params)
    v_lower, v_main, v_upper = _build_v_implicit_matrix(
        D_v, D_vv, grid, params, dt, theta
    )
    u_new = phi1.copy()
    v_fac = theta * dt
    for i in range(grid.nx):
        rhs = phi1[i, :] - v_fac * av_u[i, :]
        u_new[i, :] = _solve_tridiagonal(v_lower, v_main, v_upper, rhs)

    return apply_x_boundary(u_new, grid, params, tau, T)


def solve_heston_adi(
    params: HestonModelParams,
    T: float,
    grid: HestonSpatialGrid | None = None,
    n_steps: int = 800,
    theta: float = 0.5,
    rannacher_steps: int = 8,
    nx: int = 81,
    nv: int = 41,
) -> tuple[np.ndarray, HestonSpatialGrid, np.ndarray]:
    """
    Solve the Heston PDE on a (x, v) grid and return the option-value surface.

    Parameters
    ----------
    params : HestonModelParams
        Model parameters including strike K and risk-free rate r.
    T : float
        Option maturity in years.
    grid : HestonSpatialGrid, optional
        Pre-built spatial grid; created with ``build_heston_grid`` if omitted.
    n_steps : int
        Number of uniform steps in reversed time tau in [0, T].
    theta : float
        HV ADI parameter (0.5 for standard Hundsdorfer-Verwer).
    rannacher_steps : int
        Number of initial steps with theta = 1 (Rannacher smoothing).
    nx, nv : int
        Grid sizes when ``grid`` is not supplied.

    Returns
    -------
    u_final : ndarray, shape (nx, nv)
        Option value u(x, v) at tau = T (i.e. physical time t = 0).
    grid : HestonSpatialGrid
        Spatial grid used in the solve.
    tau_grid : ndarray
        Reversed-time nodes used in the march.
    """
    if T <= 0.0:
        raise ValueError("T must be positive.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")

    if grid is None:
        grid = build_heston_grid(nx=nx, nv=nv)

    ops = build_operators_with_boundary_conditions(grid)
    dt = T / n_steps
    tau_grid = np.linspace(0.0, T, n_steps + 1)

    u = call_terminal_payoff(grid, params.K)
    u = apply_x_boundary(u, grid, params, tau=0.0, T=T)

    for step in range(n_steps):
        tau_next = tau_grid[step + 1]
        step_theta = 1.0 if step < rannacher_steps else theta
        u = hv_adi_step(
            u,
            grid,
            ops.D_x,
            ops.D_v,
            ops.D_xx,
            ops.D_vv,
            params,
            dt,
            tau_next,
            T,
            step_theta,
        )

    return u, grid, tau_grid


def price_at_point(
    u: np.ndarray,
    grid: HestonSpatialGrid,
    x0: float,
    v0: float,
) -> float:
    """
    Bilinear interpolation of the solution at (x0, v0).

    Parameters
    ----------
    u : ndarray, shape (nx, nv)
        Option-value surface from :func:`solve_heston_adi`.
    grid : HestonSpatialGrid
        Spatial grid.
    x0, v0 : float
        Log-moneyness ln(S/K) and initial variance.

    Returns
    -------
    float
        Interpolated option value.
    """
    interp = RegularGridInterpolator(
        (grid.x, grid.v),
        u,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return float(interp((x0, v0)))


def heston_adi_call_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    grid: HestonSpatialGrid | None = None,
    n_steps: int = 800,
    rannacher_steps: int = 8,
    **grid_kwargs,
) -> float:
    """
    Convenience wrapper: solve the PDE and return the call price at (S0, v0).
    """
    params = HestonModelParams(
        r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, K=K
    )
    if grid is None:
        grid = build_heston_grid(**grid_kwargs)
    u, grid, _ = solve_heston_adi(
        params,
        T,
        grid=grid,
        n_steps=n_steps,
        rannacher_steps=rannacher_steps,
    )
    x0 = np.log(S0 / K)
    return price_at_point(u, grid, x0, v0)


def prices_along_strikes(
    u: np.ndarray,
    grid: HestonSpatialGrid,
    strikes: np.ndarray,
    v0: float,
    S0: float,
) -> np.ndarray:
    """
    Interpolate ADI call prices for multiple strikes at fixed v0.

    Each strike K defines x = ln(S0/K).
    """
    return np.array(
        [price_at_point(u, grid, np.log(S0 / K), v0) for K in strikes],
        dtype=float,
    )
