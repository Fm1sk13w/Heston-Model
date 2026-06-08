"""
Non-uniform spatial grids and finite-difference operators for the Heston PDE.

The solver works in log-moneyness $x = \\ln(S/K)$ and variance $v$. This module
builds sinh-stretched tensor-product grids and assembles sparse derivative
operators with boundary rows compatible with a European call formulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class HestonSpatialGrid:
    """
    Two-dimensional spatial grid in (x, v) with non-uniform node spacing.

    Attributes
    ----------
    x : ndarray, shape (nx,)
        Log-moneyness nodes; ATM is near x = 0.
    v : ndarray, shape (nv,)
        Variance nodes on [0, v_max]; nodes cluster near v = 0.
    dx_forward : ndarray, shape (nx - 1,)
        Forward spacings dx_i = x_{i+1} - x_i.
    dv_forward : ndarray, shape (nv - 1,)
        Forward spacings in v.
    """

    x: np.ndarray
    v: np.ndarray
    dx_forward: np.ndarray
    dv_forward: np.ndarray

    @property
    def nx(self) -> int:
        return int(self.x.size)

    @property
    def nv(self) -> int:
        return int(self.v.size)

    @property
    def dx_backward(self) -> np.ndarray:
        """Backward spacings dx_{i-1} = x_i - x_{i-1}."""
        return self.dx_forward

    @property
    def dv_backward(self) -> np.ndarray:
        return self.dv_forward


def sinh_spaced_grid(
    n_points: int,
    x_min: float,
    x_max: float,
    concentration: float = 3.0,
) -> np.ndarray:
    """
    Sinh-transformed nodes on [x_min, x_max] clustered about the interval midpoint.

    For log-moneyness with x_min < 0 < x_max, the midpoint is the at-the-money
    region. The map is
        x(xi) = x_mid + (L/2) * sinh(mu * xi) / sinh(mu),  xi in [-1, 1],
    with L = x_max - x_min and mu = concentration.

    Parameters
    ----------
    n_points : int
        Number of grid nodes (>= 2).
    x_min, x_max : float
        Domain endpoints.
    concentration : float
        Stretching parameter mu >= 0; larger values pack more nodes near x_mid.

    Returns
    -------
    ndarray
        Strictly increasing node coordinates.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min.")

    xi = np.linspace(-1.0, 1.0, n_points)
    x_mid = 0.5 * (x_min + x_max)
    half_length = 0.5 * (x_max - x_min)

    if concentration < 1e-12:
        return np.linspace(x_min, x_max, n_points)

    scale = half_length / np.sinh(concentration)
    nodes = x_mid + scale * np.sinh(concentration * xi)

    # Enforce exact endpoints (corrects floating-point drift).
    nodes[0] = x_min
    nodes[-1] = x_max
    return nodes


def sinh_spaced_grid_from_zero(
    n_points: int,
    v_max: float,
    concentration: float = 4.0,
) -> np.ndarray:
    """
    Sinh-transformed nodes on [0, v_max] clustered near the degenerate boundary v = 0.

    Uses v(eta) = v_max * sinh(mu * eta) / sinh(mu) for eta in [0, 1].

    Parameters
    ----------
    n_points : int
        Number of variance nodes (>= 2).
    v_max : float
        Upper variance cutoff.
    concentration : float
        Stretching parameter; larger mu packs more nodes near v = 0.

    Returns
    -------
    ndarray
        Nodes with v[0] = 0 and v[-1] = v_max.
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if v_max <= 0.0:
        raise ValueError("v_max must be positive.")

    eta = np.linspace(0.0, 1.0, n_points)
    if concentration < 1e-12:
        return np.linspace(0.0, v_max, n_points)

    nodes = v_max * np.sinh(concentration * eta) / np.sinh(concentration)
    nodes[0] = 0.0
    nodes[-1] = v_max
    return nodes


def build_heston_grid(
    nx: int = 81,
    nv: int = 41,
    x_min: float = -2.0,
    x_max: float = 2.0,
    v_max: float = 0.5,
    concentration_x: float = 3.0,
    concentration_v: float = 4.0,
) -> HestonSpatialGrid:
    """
    Construct the full (x, v) tensor-product grid used by the ADI solver.

    Parameters
    ----------
    nx, nv : int
        Number of nodes in x and v.
    x_min, x_max : float
        Log-moneyness domain; ATM is x = 0.
    v_max : float
        Maximum variance.
    concentration_x, concentration_v : float
        Sinh stretching parameters.

    Returns
    -------
    HestonSpatialGrid
        Grid coordinates and spacings.
    """
    x = sinh_spaced_grid(nx, x_min, x_max, concentration_x)
    v = sinh_spaced_grid_from_zero(nv, v_max, concentration_v)
    return HestonSpatialGrid(
        x=x,
        v=v,
        dx_forward=np.diff(x),
        dv_forward=np.diff(v),
    )


def _first_derivative_matrix(nodes: np.ndarray, bc: str = "neumann") -> sparse.csr_matrix:
    """
    First-derivative matrix on a 1D non-uniform grid.

    Interior points use central differences. Boundary rows are either zero
    (Dirichlet values imposed by the solver) or second-order one-sided stencils
    (Neumann/outflow).
    """
    n = nodes.size
    h_fwd = np.diff(nodes)
    rows, cols, data = [], [], []

    if bc == "neumann" and n >= 3:
        h0 = h_fwd[0]
        rows.extend([0, 0, 0])
        cols.extend([0, 1, 2])
        data.extend([-3.0 / (2.0 * h0), 4.0 / (2.0 * h0), -1.0 / (2.0 * h0)])

    for i in range(1, n - 1):
        h_m = nodes[i] - nodes[i - 1]
        h_p = nodes[i + 1] - nodes[i]
        denom = h_m + h_p
        rows.extend([i, i])
        cols.extend([i - 1, i + 1])
        data.extend([-1.0 / denom, 1.0 / denom])

    if bc == "neumann" and n >= 3:
        hn = h_fwd[-1]
        rows.extend([n - 1, n - 1, n - 1])
        cols.extend([n - 3, n - 2, n - 1])
        data.extend([1.0 / (2.0 * hn), -4.0 / (2.0 * hn), 3.0 / (2.0 * hn)])

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    if bc == "dirichlet":
        mat = mat.tolil()
        mat[0, :] = 0.0
        mat[n - 1, :] = 0.0
        mat = mat.tocsr()
    return mat


def _second_derivative_matrix(nodes: np.ndarray) -> sparse.csr_matrix:
    """
    Second-derivative matrix on a 1D non-uniform grid.

    Interior stencil for non-uniform spacing:
        u''(x_i) ~ 2/(h_m + h_p) * ((u_{i+1}-u_i)/h_p - (u_i-u_{i-1})/h_m).

    Boundary rows use second-order one-sided formulas; at v = 0 the caller may
    replace the first row via `apply_v0_degenerate_bc`.
    """
    n = nodes.size
    rows, cols, data = [], [], []

    # Left boundary (x_min): one-sided three-point formula.
    if n >= 3:
        h0 = nodes[1] - nodes[0]
        h1 = nodes[2] - nodes[1]
        denom = h0 + h1
        rows.extend([0, 0, 0])
        cols.extend([0, 1, 2])
        data.extend(
            [
                2.0 / (h0 * denom),
                -2.0 / (h0 * h1),
                2.0 / (h1 * denom),
            ]
        )

    for i in range(1, n - 1):
        h_m = nodes[i] - nodes[i - 1]
        h_p = nodes[i + 1] - nodes[i]
        denom = h_m + h_p
        a = 2.0 / (h_m * denom)
        b = -2.0 / (h_m * h_p)
        c = 2.0 / (h_p * denom)
        rows.extend([i, i, i])
        cols.extend([i - 1, i, i + 1])
        data.extend([a, b, c])

    # Right boundary.
    if n >= 3:
        hm = nodes[-2] - nodes[-3]
        hp = nodes[-1] - nodes[-2]
        denom = hm + hp
        rows.extend([n - 1, n - 1, n - 1])
        cols.extend([n - 3, n - 2, n - 1])
        data.extend(
            [
                2.0 / (hm * denom),
                -2.0 / (hm * hp),
                2.0 / (hp * denom),
            ]
        )

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


@dataclass(frozen=True)
class FiniteDifferenceOperators:
    """
    Sparse 1D finite-difference operators for x and v directions.

    Attributes
    ----------
    D_x, D_v : csr_matrix
        First-derivative operators.
    D_xx, D_vv : csr_matrix
        Second-derivative operators.
    """

    D_x: sparse.csr_matrix
    D_v: sparse.csr_matrix
    D_xx: sparse.csr_matrix
    D_vv: sparse.csr_matrix


def build_fd_operators(
    grid: HestonSpatialGrid,
    x_bc: str = "dirichlet",
    v_bc: str = "neumann",
) -> FiniteDifferenceOperators:
    """
    Assemble derivative operators on the Heston spatial grid.

    Parameters
    ----------
    grid : HestonSpatialGrid
        Spatial grid.
    x_bc : {'dirichlet', 'neumann'}
        Boundary in x: Dirichlet rows for call payoff boundaries (enforced by
        the solver); Neumann for natural/outflow treatment.
    v_bc : {'neumann', 'dirichlet'}
        Boundary in v at v_max; v = 0 row is handled separately.

    Returns
    -------
    FiniteDifferenceOperators
        Sparse matrices for ADI splitting.
    """
    D_x = _first_derivative_matrix(grid.x, bc=x_bc)
    D_v = _first_derivative_matrix(grid.v, bc=v_bc)
    D_xx = _second_derivative_matrix(grid.x)
    D_vv = _second_derivative_matrix(grid.v)
    return FiniteDifferenceOperators(D_x=D_x, D_v=D_v, D_xx=D_xx, D_vv=D_vv)


def apply_v0_neumann_row(
    grid: HestonSpatialGrid,
    D_v: sparse.csr_matrix,
    D_vv: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Set row 0 of D_v and D_vv for the degenerate boundary v = 0.

    Parameters
    ----------
    grid : HestonSpatialGrid
        Grid providing dv spacing near v = 0.
    D_v, D_vv : csr_matrix
        Operators to modify in place (copied first).

    Returns
    -------
    D_v, D_vv
        Operators with updated first row.
    """
    D_v = D_v.tolil()
    D_vv = D_vv.tolil()

    h0 = grid.dv_forward[0]
    h1 = grid.dv_forward[1]

    D_v[0, :] = 0.0
    D_v[0, 0] = -3.0 / (2.0 * h0)
    D_v[0, 1] = 4.0 / (2.0 * h0)
    D_v[0, 2] = -1.0 / (2.0 * h0)

    # One-sided second derivative at v = 0 (consistent with non-uniform spacing).
    denom = h0 + h1
    D_vv[0, :] = 0.0
    D_vv[0, 0] = 2.0 / (h0 * denom)
    D_vv[0, 1] = -2.0 / (h0 * h1)
    D_vv[0, 2] = 2.0 / (h1 * denom)

    return D_v.tocsr(), D_vv.tocsr()


def apply_x_dirichlet_rows(
    D_x: sparse.csr_matrix,
    D_xx: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Zero out boundary rows of x-derivative operators (Dirichlet values imposed separately).

    For a European call: u(x_min, v) = 0 and u(x_max, v) is set from the
    asymptotic payoff; the derivative rows at i = 0 and i = nx-1 are not used.
    """
    D_x = D_x.tolil()
    D_xx = D_xx.tolil()
    n = D_x.shape[0]
    D_x[0, :] = 0.0
    D_x[n - 1, :] = 0.0
    D_xx[0, :] = 0.0
    D_xx[n - 1, :] = 0.0
    return D_x.tocsr(), D_xx.tocsr()


def build_operators_with_boundary_conditions(
    grid: HestonSpatialGrid,
) -> FiniteDifferenceOperators:
    """
    Build FD operators with Heston-consistent boundary treatment.

    - x: Dirichlet rows (call payoff enforced by the time-stepping solver).
    - v = 0: Neumann du/dv = 0 (degenerate boundary).
    - v = v_max: Neumann outflow.
    """
    ops = build_fd_operators(grid, x_bc="dirichlet", v_bc="neumann")
    D_v, D_vv = apply_v0_neumann_row(grid, ops.D_v, ops.D_vv)
    D_x, D_xx = apply_x_dirichlet_rows(ops.D_x, ops.D_xx)
    return FiniteDifferenceOperators(D_x=D_x, D_v=D_v, D_xx=D_xx, D_vv=D_vv)


def mesh_spacing_stats(spacing: np.ndarray) -> dict[str, float]:
    """
    Summary statistics of grid spacing (useful for convergence diagnostics).

    Returns
    -------
    dict
        min, max, ratio (max/min), and mean spacing.
    """
    s_min = float(np.min(spacing))
    s_max = float(np.max(spacing))
    return {
        "min": s_min,
        "max": s_max,
        "ratio": s_max / s_min,
        "mean": float(np.mean(spacing)),
    }
