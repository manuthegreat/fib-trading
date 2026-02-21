import numpy as np
import pandas as pd

from updater import load_all_market_data, load_all_market_data_hourly


def _fib_levels(swing_low: float, swing_high: float):
    swing_range = swing_high - swing_low
    fib618 = swing_high - 0.618 * swing_range
    fib786 = swing_high - 0.786 * swing_range
    return fib618, fib786


def pivot_highs(highs, look: int = 5):
    pivots = []
    if len(highs) < (2 * look + 1):
        return pivots

    for i in range(look, len(highs) - look):
        window = highs[i - look : i + look + 1]
        if highs[i] == np.max(window):
            pivots.append(i)
    return pivots


def find_daily_wave_and_retr_low(group: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 300):
    window = group[
        (group["Date"] <= as_of)
        & (group["Date"] >= (as_of - pd.Timedelta(days=lookback_days)))
    ].copy()

    if len(window) < 11:
        return None

    window = window.sort_values("Date").reset_index(drop=True)
    highs = window["High"].to_numpy()
    lows = window["Low"].to_numpy()

    piv = pivot_highs(highs, look=5)
    if not piv:
        return None

    best_idx = max(piv, key=lambda idx: highs[idx])
    swing_high = float(highs[best_idx])
    swing_high_date = pd.to_datetime(window.loc[best_idx, "Date"])

    prior = window.iloc[: best_idx + 1]
    low_rel_idx = int(prior["Low"].to_numpy().argmin())
    swing_low = float(prior.iloc[low_rel_idx]["Low"])
    swing_low_date = pd.to_datetime(prior.iloc[low_rel_idx]["Date"])

    if swing_low >= swing_high:
        return None

    post = window.iloc[best_idx + 1 :]
    if post.empty:
        return None

    retr_rel_idx = int(post["Low"].to_numpy().argmin())
    retr_low = float(post.iloc[retr_rel_idx]["Low"])
    retr_low_date = pd.to_datetime(post.iloc[retr_rel_idx]["Date"])

    fib618, fib786 = _fib_levels(swing_low, swing_high)
    in_zone = fib786 <= retr_low <= fib618
    if not in_zone:
        return None

    return {
        "AsOf": pd.to_datetime(as_of),
        "SwingLowDate": swing_low_date,
        "SwingLow": swing_low,
        "SwingHighDate": swing_high_date,
        "SwingHigh": swing_high,
        "RetrLowDate": retr_low_date,
        "RetrLow": retr_low,
        "Fib618": float(fib618),
        "Fib786": float(fib786),
    }


def build_daily_list_last_30_days(df: pd.DataFrame, days: int = 30, lookback_days: int = 300):
    rows = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").copy()
        as_of_dates = group["Date"].drop_duplicates().sort_values().tail(days)

        latest_hit = None
        for as_of in sorted(as_of_dates.tolist(), reverse=True):
            hit = find_daily_wave_and_retr_low(group, pd.to_datetime(as_of), lookback_days=lookback_days)
            if hit is not None:
                latest_hit = {"Ticker": ticker, **hit}
                break

        if latest_hit is not None:
            rows.append(latest_hit)

    cols = [
        "Ticker",
        "AsOf",
        "SwingLowDate",
        "SwingLow",
        "SwingHighDate",
        "SwingHigh",
        "RetrLowDate",
        "RetrLow",
        "Fib618",
        "Fib786",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)[cols].sort_values(["AsOf", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_hourly_entry_list(
    df_hourly: pd.DataFrame,
    daily_list: pd.DataFrame,
    hours_lookback: int = 240,
    min_hh_count: int = 2,
    min_pullback_pct: float = 0.007,
):
    cols = [
        "Ticker",
        "Daily_AsOf",
        "Daily_RetrLowDate",
        "Daily_RetrLow",
        "Hourly_LocalHighTime",
        "Hourly_LocalHigh",
        "Entry_61_8",
        "Stop",
        "TakeProfit",
        "RR",
        "Triggered_LastBar",
        "LastClose",
    ]
    if daily_list.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    daily_lookup = daily_list.set_index("Ticker")

    for ticker in daily_lookup.index:
        group = df_hourly[df_hourly["Ticker"] == ticker].sort_values("DateTime")
        if group.empty:
            continue

        window = group.tail(hours_lookback).copy()
        if len(window) < 11:
            continue

        highs = window["High"].to_numpy()
        piv = pivot_highs(highs, look=5)
        if len(piv) < min_hh_count:
            continue

        last_piv = piv[-min_hh_count:]
        last_piv_values = [float(highs[i]) for i in last_piv]
        if not all(x < y for x, y in zip(last_piv_values, last_piv_values[1:])):
            continue

        local_idx = piv[-1]
        local_high = float(highs[local_idx])
        local_high_time = pd.to_datetime(window.iloc[local_idx]["DateTime"])

        last_close = float(window["Close"].iloc[-1])
        pullback_pct = (local_high - last_close) / local_high if local_high > 0 else 0.0
        if pullback_pct < min_pullback_pct:
            continue

        drow = daily_lookup.loc[ticker]
        fib_low = float(drow["RetrLow"])

        entry = local_high - 0.618 * (local_high - fib_low)
        stop = fib_low
        takeprofit = local_high
        denom = entry - stop
        rr = (takeprofit - entry) / denom if denom > 0 else np.nan

        last_bar = window.iloc[-1]
        triggered_last_bar = bool(last_bar["Low"] <= entry <= last_bar["High"])

        rows.append(
            {
                "Ticker": ticker,
                "Daily_AsOf": pd.to_datetime(drow["AsOf"]),
                "Daily_RetrLowDate": pd.to_datetime(drow["RetrLowDate"]),
                "Daily_RetrLow": fib_low,
                "Hourly_LocalHighTime": local_high_time,
                "Hourly_LocalHigh": local_high,
                "Entry_61_8": float(entry),
                "Stop": float(stop),
                "TakeProfit": float(takeprofit),
                "RR": float(rr) if not np.isnan(rr) else np.nan,
                "Triggered_LastBar": triggered_last_bar,
                "LastClose": last_close,
            }
        )

    if not rows:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)[cols].sort_values(["Triggered_LastBar", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def run_engine():
    df_daily = load_all_market_data()
    required_daily = {"Ticker", "Date", "High", "Low"}
    missing_daily = required_daily - set(df_daily.columns)
    if missing_daily:
        raise ValueError(
            f"load_all_market_data() missing required columns: {sorted(missing_daily)}. "
            f"Found columns: {sorted(df_daily.columns.tolist())}"
        )

    df_daily = df_daily.copy()
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    daily_list = build_daily_list_last_30_days(df_daily, days=30, lookback_days=300)

    df_hourly = load_all_market_data_hourly()
    required_hourly = {"Ticker", "DateTime", "High", "Low", "Close"}
    missing_hourly = required_hourly - set(df_hourly.columns)
    if missing_hourly:
        raise ValueError(
            f"load_all_market_data_hourly() missing required columns: {sorted(missing_hourly)}. "
            f"Found columns: {sorted(df_hourly.columns.tolist())}"
        )

    df_hourly = df_hourly.copy()
    df_hourly["DateTime"] = pd.to_datetime(df_hourly["DateTime"])

    hourly_list = build_hourly_entry_list(
        df_hourly,
        daily_list,
        hours_lookback=240,
        min_hh_count=2,
        min_pullback_pct=0.007,
    )

    return daily_list, hourly_list
