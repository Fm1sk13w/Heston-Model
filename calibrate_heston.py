#!/usr/bin/env python3
"""
Calibrate Heston and Black--Scholes models to SPY option quotes.

Usage
-----
    python calibrate_heston.py                  # bundled CSV snapshot
    python calibrate_heston.py --live             # refresh from Yahoo Finance
    python calibrate_heston.py --csv path.csv     # custom quote file
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.calibration import calibrate_black_scholes, calibrate_heston
from utils.market_data import (
    DEFAULT_SNAPSHOT,
    fetch_spy_option_chain,
    load_snapshot_csv,
    save_option_chain_csv,
)


def _print_chain_summary(chain) -> None:
    print(f"Trade date:   {chain.trade_date}")
    print(f"Expiry:       {chain.expiry}")
    print(f"Spot S0:      {chain.spot:.2f}")
    print(f"Maturity T:   {chain.maturity:.4f} years")
    print(f"Rate r:       {chain.rate:.4f}")
    print(f"Quotes used:  {chain.n_quotes}")
    print(f"Strike range: [{chain.strikes.min():.1f}, {chain.strikes.max():.1f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Heston/BS to SPY options.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Path to option CSV (default: {DEFAULT_SNAPSHOT})",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Download fresh SPY quotes via yfinance instead of the snapshot.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the filtered chain as CSV.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=800,
        help="Nelder-Mead iteration cap for Heston calibration.",
    )
    args = parser.parse_args()

    if args.live:
        chain = fetch_spy_option_chain()
    else:
        chain = load_snapshot_csv(args.csv or DEFAULT_SNAPSHOT)

    if args.save is not None:
        save_option_chain_csv(chain, args.save)

    print("=== Market data ===")
    _print_chain_summary(chain)

    print("\n=== Black--Scholes calibration ===")
    bs = calibrate_black_scholes(chain)
    print(f"sigma_BS:  {bs.sigma:.4f} ({bs.sigma * 100:.2f}%)")
    print(f"RMSE:      {bs.rmse:.4f}")
    print(f"MAE:       {bs.mae:.4f}")
    print(f"Max error: {bs.max_abs_error:.4f}")

    print("\n=== Heston calibration (Nelder--Mead, Fourier pricer) ===")
    heston = calibrate_heston(chain, maxiter=args.maxiter)
    print(f"v0:        {heston.v0:.6f}")
    print(f"kappa:     {heston.kappa:.6f}")
    print(f"theta:     {heston.theta:.6f}")
    print(f"xi:        {heston.xi:.6f}")
    print(f"rho:       {heston.rho:.6f}")
    print(f"Feller:    {heston.feller_ratio:.4f}  (>= 1 satisfied: {heston.feller_ratio >= 1.0})")
    print(f"RMSE:      {heston.rmse:.4f}")
    print(f"MAE:       {heston.mae:.4f}")
    print(f"Max error: {heston.max_abs_error:.4f}")
    print(f"Success:   {heston.success} ({heston.message})")
    print(f"Func evals:{heston.n_evaluations}")


if __name__ == "__main__":
    main()
