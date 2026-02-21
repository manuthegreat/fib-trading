import numpy as np
import pandas as pd

from updater import load_all_market_data, load_all_market_data_hourly


DAILY_LOOKBACK_DAYS = 300
DAILY_HISTORY_DAYS = 30
HOURS_LOOKBACK = 240
DAILY_PIVOT_LOOK = 5
HOURLY_PIVOT_LOOK = 5
MIN_PULLBACK_PCT = 0.007


def pivot_highs(highs: np.ndarray, look: int):
    pivots = []
    if len(highs) < (2 * look + 1):
        return pivots

    for i in range(look, len(highs) - look):
        if highs[i] == np.max(highs[i - look : i + look + 1]):
            pivots.append(i)
    return pivots


def _evaluate_daily_in_zone_event(group: pd.DataFrame, as_of: pd.Timestamp):
    window = group[
        (group["Date"] <= as_of)
        & (group["Date"] >= (as_of - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)))
    ].copy()

    if window.empty:
        return None

    window = window.sort_values("Date").reset_index(drop=True)
    highs = window["High"].to_numpy()

    piv = pivot_highs(highs, look=DAILY_PIVOT_LOOK)
    if not piv:
        return None

    best_rel_idx = max(piv, key=lambda idx: highs[idx])
    swing_high = float(highs[best_rel_idx])
    swing_high_date = pd.to_datetime(window.loc[best_rel_idx, "Date"])

    prior_segment = window.iloc[: best_rel_idx + 1]
    low_idx = prior_segment["Low"].idxmin()
    swing_low = float(prior_segment.loc[low_idx, "Low"])
    swing_low_date = pd.to_datetime(prior_segment.loc[low_idx, "Date"])

    fib618 = swing_high - 0.618 * (swing_high - swing_low)
    fib786 = swing_high - 0.786 * (swing_high - swing_low)

    correction = window[(window["Date"] > swing_high_date) & (window["Date"] <= as_of)]
    if correction.empty:
        return None

    retr_low = float(correction["Low"].min())
    retr_low_idx = correction["Low"].idxmin()
    retr_low_date = pd.to_datetime(correction.loc[retr_low_idx, "Date"])

    if not (fib786 <= retr_low <= fib618):
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


def build_daily_list(df_daily: pd.DataFrame):
    rows = []

    for ticker, group in df_daily.groupby("Ticker"):
        group = group.sort_values("Date").copy()
        as_of_dates = group["Date"].drop_duplicates().sort_values().tail(DAILY_HISTORY_DAYS)

        latest_event = None
        for as_of in as_of_dates.tolist():
            event = _evaluate_daily_in_zone_event(group, pd.to_datetime(as_of))
            if event is not None:
                latest_event = event

        if latest_event is not None:
            rows.append({"Ticker": ticker, **latest_event})

    columns = [
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
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)[columns].sort_values(["Ticker"]).reset_index(drop=True)


def build_hourly_list(df_hourly: pd.DataFrame, daily_list_df: pd.DataFrame):
    columns = [
        "Ticker",
        "Daily_AsOf",
        "Daily_RetrLowDate",
        "Daily_RetrLow",
        "Hourly_LocalHighTime",
        "Hourly_LocalHigh",
        "Entry_61_8",
        "Stop",
        "TakeProfit",
        "PullbackPct",
        "LastClose",
        "Triggered_LastBar",
    ]

    if daily_list_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    daily_lookup = daily_list_df.set_index("Ticker")

    for ticker in daily_lookup.index:
        drow = daily_lookup.loc[ticker]
        group = df_hourly[df_hourly["Ticker"] == ticker].sort_values("DateTime")
        if group.empty:
            continue

        window = group.tail(HOURS_LOOKBACK).copy()
        highs = window["High"].to_numpy()
        piv = pivot_highs(highs, look=HOURLY_PIVOT_LOOK)

        if len(piv) < 3:
            continue

        last_three_values = [float(highs[i]) for i in piv[-3:]]
        if not (last_three_values[0] < last_three_values[1] < last_three_values[2]):
            continue

        local_idx = piv[-1]
        local_high = float(highs[local_idx])
        local_high_time = pd.to_datetime(window.iloc[local_idx]["DateTime"])

        last_close = float(window["Close"].iloc[-1])
        pullback_pct = (local_high - last_close) / local_high if local_high > 0 else np.nan
        if pd.isna(pullback_pct) or pullback_pct < MIN_PULLBACK_PCT:
            continue

        fib_low = float(drow["RetrLow"])
        entry = local_high - 0.618 * (local_high - fib_low)
        stop = fib_low
        take_profit = local_high

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
                "TakeProfit": float(take_profit),
                "PullbackPct": float(pullback_pct),
                "LastClose": last_close,
                "Triggered_LastBar": triggered_last_bar,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)[columns].sort_values(["Ticker"]).reset_index(drop=True)


def run_engine():
    df_daily = load_all_market_data()
    required_daily = {"Ticker", "Date", "Open", "High", "Low", "Close"}
    missing_daily = required_daily - set(df_daily.columns)
    if missing_daily:
        raise ValueError(
            f"load_all_market_data() missing required columns: {sorted(missing_daily)}. "
            f"Found columns: {sorted(df_daily.columns.tolist())}"
        )

    df_daily = df_daily.copy()
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])

    daily_list_df = build_daily_list(df_daily)

    df_hourly = load_all_market_data_hourly()
    required_hourly = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing_hourly = required_hourly - set(df_hourly.columns)
    if missing_hourly:
        raise ValueError(
            f"load_all_market_data_hourly() missing required columns: {sorted(missing_hourly)}. "
            f"Found columns: {sorted(df_hourly.columns.tolist())}"
        )

    df_hourly = df_hourly.copy()
    df_hourly["DateTime"] = pd.to_datetime(df_hourly["DateTime"])

    hourly_list_df = build_hourly_list(df_hourly, daily_list_df)

    return daily_list_df, hourly_list_df
