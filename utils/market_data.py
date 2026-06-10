"""
Market option-chain loading and filtering for Heston calibration.

A frozen CSV snapshot (``data/spy_options_snapshot.csv``) ships with the
repository for offline reproducibility. Live quotes can be refreshed via
``yfinance`` when network access is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "spy_options_snapshot.csv"
DEFAULT_SURFACE_CSV = PROJECT_ROOT / "data" / "spy_options_surface.csv"


@dataclass(frozen=True)
class OptionChain:
    """Filtered European call quotes on a single expiry slice."""

    spot: float
    rate: float
    maturity: float
    trade_date: str
    expiry: str
    strikes: np.ndarray
    mid_prices: np.ndarray
    volumes: np.ndarray

    @property
    def n_quotes(self) -> int:
        return int(self.strikes.size)


def _mid_price(row: pd.Series) -> float:
    bid, ask, last = float(row["bid"]), float(row["ask"]), float(row["lastPrice"])
    if bid > 0.0 and ask > 0.0 and ask >= bid:
        return 0.5 * (bid + ask)
    return last


def filter_option_chain(
    df: pd.DataFrame,
    *,
    spot: float,
    otm_calls_only: bool = True,
    min_volume: float = 1.0,
    min_mid: float = 0.10,
    max_moneyness: float = 1.12,
    min_moneyness: float = 1.001,
) -> pd.DataFrame:
    """
    Apply standard liquidity and moneyness filters for calibration.

    Parameters
    ----------
    df : DataFrame
        Raw option rows with at least ``strike``, ``bid``, ``ask``,
        ``lastPrice``, and ``volume``.
    spot : float
        Underlying spot for moneyness filtering.
    otm_calls_only : bool
        Keep calls with K > S (out-of-the-money calls).
    min_volume : float
        Minimum reported volume.
    min_mid : float
        Minimum mid price in dollars (removes ultra-cheap wing quotes).
    max_moneyness, min_moneyness : float
        Keep strikes with K/S in [min_moneyness, max_moneyness].
    """
    work = df.copy()
    if "mid" not in work.columns:
        work["mid"] = work.apply(_mid_price, axis=1)

    moneyness = work["strike"] / spot
    mask = (
        (work["volume"] >= min_volume)
        & (work["mid"] >= min_mid)
        & (moneyness >= min_moneyness)
        & (moneyness <= max_moneyness)
    )
    if otm_calls_only:
        mask &= work["strike"] > spot

    filtered = work.loc[mask].sort_values("strike").reset_index(drop=True)
    return filtered


def load_snapshot_csv(path: Path | str = DEFAULT_SNAPSHOT) -> OptionChain:
    """Load the bundled SPY option snapshot and return a filtered chain."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Option snapshot not found: {path}")

    df = pd.read_csv(path)
    spot = float(df["spot"].iloc[0])
    rate = float(df["r"].iloc[0])
    maturity = float(df["T"].iloc[0])
    trade_date = str(df["trade_date"].iloc[0])
    expiry = str(df["expiry"].iloc[0])

    filtered = filter_option_chain(df, spot=spot)
    if filtered.empty:
        raise ValueError("No quotes remain after filtering the snapshot.")

    return OptionChain(
        spot=spot,
        rate=rate,
        maturity=maturity,
        trade_date=trade_date,
        expiry=expiry,
        strikes=filtered["strike"].to_numpy(dtype=float),
        mid_prices=filtered["mid"].to_numpy(dtype=float),
        volumes=filtered["volume"].to_numpy(dtype=float),
    )


def fetch_spy_option_chain(
    trade_date: datetime | None = None,
    *,
    min_days_to_expiry: int = 10,
    max_days_to_expiry: int = 60,
    ticker_symbol: str = "SPY",
) -> OptionChain:
    """
    Download SPY call options via ``yfinance`` and apply calibration filters.

    Parameters
    ----------
    trade_date : datetime, optional
        Reference date; defaults to today.
    min_days_to_expiry, max_days_to_expiry : int
        Expiry window measured from ``trade_date``.
    ticker_symbol : str
        Yahoo Finance underlying symbol.

    Returns
    -------
    OptionChain
        Filtered OTM call quotes on the first expiry in the window.
    """
    import yfinance as yf

    if trade_date is None:
        trade_date = datetime.now()

    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(start=trade_date.strftime("%Y-%m-%d"))
    if hist.empty:
        raise RuntimeError("Could not download spot price from Yahoo Finance.")
    spot = float(hist["Close"].iloc[-1])

    chosen_expiry = None
    chosen_T = None
    for exp in ticker.options:
        T = (pd.Timestamp(exp) - pd.Timestamp(trade_date)).days / 365.0
        days = (pd.Timestamp(exp) - pd.Timestamp(trade_date)).days
        if min_days_to_expiry <= days <= max_days_to_expiry:
            chosen_expiry = exp
            chosen_T = T
            break

    if chosen_expiry is None:
        raise RuntimeError("No option expiry found in the requested window.")

    calls = ticker.option_chain(chosen_expiry).calls
    calls = calls.copy()
    calls["mid"] = calls.apply(_mid_price, axis=1)
    calls["spot"] = spot
    calls["expiry"] = chosen_expiry
    calls["trade_date"] = trade_date.strftime("%Y-%m-%d")
    calls["T"] = chosen_T
    calls["r"] = 0.045

    filtered = filter_option_chain(calls, spot=spot)
    if filtered.empty:
        raise ValueError("No liquid OTM quotes after filtering live data.")

    return OptionChain(
        spot=spot,
        rate=float(calls["r"].iloc[0]),
        maturity=float(chosen_T),
        trade_date=trade_date.strftime("%Y-%m-%d"),
        expiry=chosen_expiry,
        strikes=filtered["strike"].to_numpy(dtype=float),
        mid_prices=filtered["mid"].to_numpy(dtype=float),
        volumes=filtered["volume"].to_numpy(dtype=float),
    )


def load_surface_csv(path: Path | str = DEFAULT_SURFACE_CSV) -> pd.DataFrame:
    """
    Load a multi-expiry SPY option surface snapshot.

    Returns a DataFrame with one row per (strike, expiry) quote after the
    same liquidity filters used for single-expiry calibration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Surface snapshot not found: {path}")

    raw = pd.read_csv(path)
    if "mid" not in raw.columns:
        raw["mid"] = raw.apply(_mid_price, axis=1)

    chunks: list[pd.DataFrame] = []
    for expiry, group in raw.groupby("expiry"):
        spot = float(group["spot"].iloc[0])
        filtered = filter_option_chain(group, spot=spot)
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        raise ValueError("No quotes remain after filtering the surface snapshot.")

    return pd.concat(chunks, ignore_index=True)


def pooled_option_chain(df: pd.DataFrame) -> OptionChain:
    """Collapse a multi-expiry DataFrame into a single :class:`OptionChain`."""
    return OptionChain(
        spot=float(df["spot"].iloc[0]),
        rate=float(df["r"].iloc[0]),
        maturity=float(df["T"].median()),
        trade_date=str(df["trade_date"].iloc[0]),
        expiry="pooled",
        strikes=df["strike"].to_numpy(dtype=float),
        mid_prices=df["mid"].to_numpy(dtype=float),
        volumes=df["volume"].to_numpy(dtype=float),
    )


def save_option_chain_csv(chain: OptionChain, path: Path | str) -> None:
    """Persist a filtered chain in the snapshot CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "strike": chain.strikes,
            "bid": np.nan,
            "ask": np.nan,
            "lastPrice": chain.mid_prices,
            "mid": chain.mid_prices,
            "volume": chain.volumes,
            "impliedVolatility": np.nan,
            "spot": chain.spot,
            "expiry": chain.expiry,
            "trade_date": chain.trade_date,
            "T": chain.maturity,
            "r": chain.rate,
        }
    )
    df.to_csv(path, index=False)


# #region agent log
try:
    import json
    import time

    _dbg_path = PROJECT_ROOT / "debug-bae89c.log"
    with open(_dbg_path, "a", encoding="utf-8") as _dbg_f:
        _dbg_f.write(
            json.dumps(
                {
                    "sessionId": "bae89c",
                    "hypothesisId": "A",
                    "location": "market_data.py:module_load",
                    "message": "market_data module loaded",
                    "data": {
                        "file": str(__file__),
                        "has_load_surface_csv": "load_surface_csv" in globals(),
                        "exports": sorted(
                            n
                            for n in globals()
                            if not n.startswith("_") and callable(globals()[n])
                        ),
                    },
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )
except Exception:
    pass
# #endregion
