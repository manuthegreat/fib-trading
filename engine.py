import numpy as np
import pandas as pd

from updater import load_all_market_data, load_all_market_data_hourly


LOOKBACK_DAYS = 300
DAILY_PIVOT_LOOK = 5
HOURLY_PIVOT_LOOK = 2


def _pivot_high_indices(values: np.ndarray, look: int) -> list[int]:
    pivots = []
    if len(values) < (2 * look + 1):
        return pivots

    for i in range(look, len(values) - look):
        if values[i] == np.max(values[i - look : i + look + 1]):
            pivots.append(i)
    return pivots


def _find_swing_as_of_quick(group: pd.DataFrame, current_date: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS):
    window = group[
        (group["Date"] <= current_date)
        & (group["Date"] >= (current_date - pd.Timedelta(days=lookback_days)))
    ]

    if len(window) < 10:
        return None

    window = window.sort_values("Date").reset_index(drop=True)
    highs = window["High"].to_numpy()

    pivots = _pivot_high_indices(highs, look=DAILY_PIVOT_LOOK)
    if not pivots:
        return None

    best_rel_idx = max(pivots, key=lambda idx: highs[idx])
    swing_high_price = float(highs[best_rel_idx])
    swing_high_date = pd.to_datetime(window.loc[best_rel_idx, "Date"])

    prior_segment = window.iloc[: best_rel_idx + 1]
    low_idx = prior_segment["Low"].idxmin()
    swing_low_price = float(prior_segment.loc[low_idx, "Low"])
    swing_low_date = pd.to_datetime(prior_segment.loc[low_idx, "Date"])

    if swing_low_price >= swing_high_price:
        return None

    swing_range = swing_high_price - swing_low_price

    return {
        "Swing Low Date": swing_low_date,
        "Swing Low Price": swing_low_price,
        "Swing High Date": swing_high_date,
        "Swing High Price": swing_high_price,
        "Retrace 50": swing_high_price - 0.50 * swing_range,
        "Retrace 61": swing_high_price - 0.618 * swing_range,
        "Stop Consider (78.6%)": swing_high_price - 0.786 * swing_range,
    }


def build_watchlist(df: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    rows = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date")

        latest_price = float(group["Close"].iloc[-1])
        latest_date = pd.to_datetime(group["Date"].iloc[-1])

        swing = _find_swing_as_of_quick(group, latest_date, lookback_days)
        if swing is None:
            continue

        post_high = group[(group["Date"] > swing["Swing High Date"]) & (group["Date"] <= latest_date)]
        if (not post_high.empty) and (post_high["Low"] < swing["Stop Consider (78.6%)"]).any():
            continue

        retracement = (swing["Swing High Price"] - latest_price) / (swing["Swing High Price"] - swing["Swing Low Price"])
        if retracement < 0.38:
            continue

        retr_low_idx = post_high["Low"].idxmin()
        retr_low_price = float(group.loc[retr_low_idx, "Low"])
        retr_low_date = pd.to_datetime(group.loc[retr_low_idx, "Date"])

        rows.append(
            {
                "Ticker": ticker,
                "Latest Date": latest_date,
                "Latest Price": latest_price,
                "Swing Low Date": swing["Swing Low Date"],
                "Swing Low Price": float(swing["Swing Low Price"]),
                "Swing High Date": swing["Swing High Date"],
                "Swing High Price": float(swing["Swing High Price"]),
                "Swing Range": float(swing["Swing High Price"] - swing["Swing Low Price"]),
                "Retracement": float(retracement),
                "Daily Retracement Low Date": retr_low_date,
                "Daily Retracement Low": retr_low_price,
            }
        )

    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True) if rows else pd.DataFrame()


def build_hourly_hh_retrace_list(df_hourly: pd.DataFrame, daily_watch: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Daily Retracement Low Date",
        "Daily Retracement Low",
        "Hourly Local High Time",
        "Hourly Local High",
        "Higher High Confirmed",
        "Retracing Now",
        "Fib 61.8",
        "Flagged 61.8%",
        "Latest Hourly Close",
    ]

    if daily_watch.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    lookup = daily_watch.set_index("Ticker")

    for ticker in lookup.index:
        drow = lookup.loc[ticker]
        retr_low_date = pd.to_datetime(drow["Daily Retracement Low Date"])
        daily_fib_low = float(drow["Daily Retracement Low"])

        g = df_hourly[(df_hourly["Ticker"] == ticker) & (df_hourly["DateTime"] >= retr_low_date)].sort_values("DateTime")
        if len(g) < 8:
            continue

        highs = g["High"].to_numpy()
        piv = _pivot_high_indices(highs, look=HOURLY_PIVOT_LOOK)
        if len(piv) < 2:
            continue

        hh_confirmed = any(highs[piv[i]] > highs[piv[i - 1]] for i in range(1, len(piv)))
        if not hh_confirmed:
            continue

        local_high_idx = piv[-1]
        local_high = float(highs[local_high_idx])
        local_high_time = pd.to_datetime(g.iloc[local_high_idx]["DateTime"])

        after_high = g.iloc[local_high_idx + 1 :].copy()
        if after_high.empty:
            retracing_now = False
            latest_close = float(g["Close"].iloc[-1])
            flagged_618 = False
        else:
            latest_close = float(after_high["Close"].iloc[-1])
            retracing_now = (after_high["High"].max() <= local_high) and (latest_close < local_high)

            fib_618 = local_high - 0.618 * (local_high - daily_fib_low)
            flagged_618 = bool((after_high["Low"] <= fib_618).any())

        fib_618 = local_high - 0.618 * (local_high - daily_fib_low)

        rows.append(
            {
                "Ticker": ticker,
                "Daily Retracement Low Date": retr_low_date,
                "Daily Retracement Low": daily_fib_low,
                "Hourly Local High Time": local_high_time,
                "Hourly Local High": local_high,
                "Higher High Confirmed": bool(hh_confirmed),
                "Retracing Now": bool(retracing_now),
                "Fib 61.8": float(fib_618),
                "Flagged 61.8%": bool(flagged_618),
                "Latest Hourly Close": latest_close,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)[columns].sort_values(["Flagged 61.8%", "Ticker"], ascending=[False, True]).reset_index(drop=True)


def run_engine():
    df_daily = load_all_market_data().copy()
    required_daily = {"Ticker", "Date", "Open", "High", "Low", "Close"}
    missing_daily = required_daily - set(df_daily.columns)
    if missing_daily:
        raise ValueError(f"load_all_market_data() missing required columns: {sorted(missing_daily)}")

    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    daily_list_df = build_watchlist(df_daily, lookback_days=LOOKBACK_DAYS)

    df_hourly = load_all_market_data_hourly().copy()
    required_hourly = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing_hourly = required_hourly - set(df_hourly.columns)
    if missing_hourly:
        raise ValueError(f"load_all_market_data_hourly() missing required columns: {sorted(missing_hourly)}")

    df_hourly["DateTime"] = pd.to_datetime(df_hourly["DateTime"])
    hourly_list_df = build_hourly_hh_retrace_list(df_hourly, daily_list_df)

    return daily_list_df, hourly_list_df
