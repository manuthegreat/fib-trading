"""Utilities for ranking hourly trade candidates by actionable risk/reward."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _safe_float(value: object) -> float:
    """Convert a scalar value to float, returning NaN for invalid values."""
    try:
        if value is None:
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def compute_rr_for_row(row: pd.Series) -> Tuple[float, float]:
    """Compute risk:reward ratio and remaining reward percentage for a row.

    Expected keys:
    - entry
    - stop
    - take_profit
    Optional keys:
    - side (defaults to "long")
    - current_price
    """
    entry = _safe_float(row.get("entry"))
    stop = _safe_float(row.get("stop"))
    take_profit = _safe_float(row.get("take_profit"))
    current_price = _safe_float(row.get("current_price"))
    side = str(row.get("side", "long")).strip().lower()

    if np.isnan(entry) or np.isnan(stop) or np.isnan(take_profit):
        return (np.nan, np.nan)

    if side == "short":
        risk = stop - entry
        reward = entry - take_profit
        if np.isnan(current_price):
            reward_from_current_pct = np.nan
        else:
            remaining_reward = max(current_price - take_profit, 0.0)
            reward_from_current_pct = (
                remaining_reward / reward if reward > 0 else np.nan
            )
    else:
        risk = entry - stop
        reward = take_profit - entry
        if np.isnan(current_price):
            reward_from_current_pct = np.nan
        else:
            remaining_reward = max(take_profit - current_price, 0.0)
            reward_from_current_pct = (
                remaining_reward / reward if reward > 0 else np.nan
            )

    if risk == 0 or risk < 0 or reward <= 0:
        return (np.nan, np.nan)

    rr = reward / risk
    return (rr, reward_from_current_pct)


def _safe_minmax_normalize(series: pd.Series) -> pd.Series:
    """Normalize values to [0, 1], handling NaN and constant values safely."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    min_val = valid.min()
    max_val = valid.max()
    if max_val == min_val:
        out = pd.Series(1.0, index=series.index, dtype=float)
        out[numeric.isna()] = np.nan
        return out

    return (numeric - min_val) / (max_val - min_val)


def rank_hourly_candidates(
    hourly_df: pd.DataFrame,
    current_price_col: str = "current_price",
) -> pd.DataFrame:
    """Rank hourly candidates using a composite score based on R:R and reward remaining.

    Returns a sorted copy with columns:
    - rr
    - reward_from_current_pct
    - rr_score
    """
    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame(columns=list(getattr(hourly_df, "columns", [])) + ["rr", "reward_from_current_pct", "rr_score"])

    ranked = hourly_df.copy()

    if "entry" not in ranked.columns and "entry_618" in ranked.columns:
        ranked["entry"] = ranked["entry_618"]

    if "side" not in ranked.columns:
        ranked["side"] = "long"

    if current_price_col != "current_price" and current_price_col in ranked.columns:
        ranked["current_price"] = ranked[current_price_col]
    elif "current_price" not in ranked.columns:
        ranked["current_price"] = np.nan

    rr_data = ranked.apply(compute_rr_for_row, axis=1, result_type="expand")
    rr_data.columns = ["rr", "reward_from_current_pct"]

    ranked["rr"] = rr_data["rr"]
    ranked["reward_from_current_pct"] = rr_data["reward_from_current_pct"]

    rr_norm = _safe_minmax_normalize(ranked["rr"])
    reward_norm = _safe_minmax_normalize(ranked["reward_from_current_pct"])
    ranked["rr_score"] = (0.7 * rr_norm.fillna(0)) + (0.3 * reward_norm.fillna(0))

    ranked = ranked.sort_values(by=["rr_score", "rr"], ascending=[False, False]).reset_index(drop=True)
    return ranked
