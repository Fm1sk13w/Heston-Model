"""Persist calibrated model parameters for reuse in later notebook sections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.calibration import BlackScholesCalibrationResult, HestonCalibrationResult

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "calibrated_params.json"


def save_calibrated_params(
    bs_fit: BlackScholesCalibrationResult,
    heston_fit: HestonCalibrationResult,
    *,
    path: Path | str = DEFAULT_PATH,
) -> Path:
    """Write BS and Heston calibration results to JSON."""
    payload: dict[str, Any] = {
        "sigma_bs": bs_fit.sigma,
        "heston": {
            "v0": heston_fit.v0,
            "kappa": heston_fit.kappa,
            "theta": heston_fit.theta,
            "xi": heston_fit.xi,
            "rho": heston_fit.rho,
        },
        "fit_metrics": {
            "bs_rmse": bs_fit.rmse,
            "heston_rmse": heston_fit.rmse,
            "heston_feller": heston_fit.feller_ratio,
        },
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_calibrated_params(*, path: Path | str = DEFAULT_PATH) -> dict[str, Any]:
    """Load calibrated parameters written by :func:`save_calibrated_params`."""
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run notebook Section 6 (calibration) first."
        )
    return json.loads(src.read_text(encoding="utf-8"))
