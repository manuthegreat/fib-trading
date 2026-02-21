import numpy as np
import pandas as pd

from updater import load_all_market_data, load_all_market_data_hourly


DAILY_LOOKBACK_DAYS = 300
DAILY_HISTORY_DAYS = 30
HOURS_LOOKBACK = 240
MIN_HH_COUNT = 2
MIN_PULLBACK_PCT = 0.007
MIN_SWING_RANGE_PCT = 0.08
MIN_POST_SWING_BARS = 10


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


def _empty_daily_gate(as_of: pd.Timestamp, reason: str):
    return {
        "AsOf": pd.to_datetime(as_of),
        "Gate1_Pass": False,
        "Gate1_Reason": reason,
        "SwingLowDate": pd.NaT,
        "SwingLow": np.nan,
        "SwingHighDate": pd.NaT,
        "SwingHigh": np.nan,
        "RetrLowDate": pd.NaT,
        "RetrLow": np.nan,
        "Fib618": np.nan,
        "Fib786": np.nan,
        "SwingRangePct": np.nan,
        "PostSwingBars": np.nan,
    }


def evaluate_daily_gate(group: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = DAILY_LOOKBACK_DAYS):
    window = group[
        (group["Date"] <= as_of)
        & (group["Date"] >= (as_of - pd.Timedelta(days=lookback_days)))
    ].copy()

    if len(window) < 11:
        return _empty_daily_gate(as_of, "insufficient_daily_bars")

    window = window.sort_values("Date").reset_index(drop=True)
    highs = window["High"].to_numpy()

    piv = pivot_highs(highs, look=5)
    if not piv:
        return _empty_daily_gate(as_of, "no_pivot_high")

    best_idx = max(piv, key=lambda idx: highs[idx])
    swing_high = float(highs[best_idx])
    swing_high_date = pd.to_datetime(window.loc[best_idx, "Date"])

    prior = window.iloc[: best_idx + 1]
    low_rel_idx = int(prior["Low"].to_numpy().argmin())
    swing_low = float(prior.iloc[low_rel_idx]["Low"])
    swing_low_date = pd.to_datetime(prior.iloc[low_rel_idx]["Date"])

    if swing_low <= 0 or swing_low >= swing_high:
        return _empty_daily_gate(as_of, "invalid_swing")

    swing_range_pct = (swing_high - swing_low) / swing_low
    if swing_range_pct < MIN_SWING_RANGE_PCT:
        return {
            "AsOf": pd.to_datetime(as_of),
            "Gate1_Pass": False,
            "Gate1_Reason": "swing_range_below_8pct",
            "SwingLowDate": swing_low_date,
            "SwingLow": swing_low,
            "SwingHighDate": swing_high_date,
            "SwingHigh": swing_high,
            "RetrLowDate": pd.NaT,
            "RetrLow": np.nan,
            "Fib618": np.nan,
            "Fib786": np.nan,
            "SwingRangePct": swing_range_pct,
            "PostSwingBars": np.nan,
        }

    post = window.iloc[best_idx + 1 :]
    post_bars = int(len(post))
    if post_bars < MIN_POST_SWING_BARS:
        return {
            "AsOf": pd.to_datetime(as_of),
            "Gate1_Pass": False,
            "Gate1_Reason": "post_swing_window_lt_10_bars",
            "SwingLowDate": swing_low_date,
            "SwingLow": swing_low,
            "SwingHighDate": swing_high_date,
            "SwingHigh": swing_high,
            "RetrLowDate": pd.NaT,
            "RetrLow": np.nan,
            "Fib618": np.nan,
            "Fib786": np.nan,
            "SwingRangePct": swing_range_pct,
            "PostSwingBars": post_bars,
        }

    retr_rel_idx = int(post["Low"].to_numpy().argmin())
    retr_low = float(post.iloc[retr_rel_idx]["Low"])
    retr_low_date = pd.to_datetime(post.iloc[retr_rel_idx]["Date"])

    fib618, fib786 = _fib_levels(swing_low, swing_high)
    in_zone = fib786 <= retr_low <= fib618
    reason = "OK" if in_zone else "retr_low_not_in_fib_zone"

    return {
        "AsOf": pd.to_datetime(as_of),
        "Gate1_Pass": bool(in_zone),
        "Gate1_Reason": reason,
        "SwingLowDate": swing_low_date,
        "SwingLow": swing_low,
        "SwingHighDate": swing_high_date,
        "SwingHigh": swing_high,
        "RetrLowDate": retr_low_date,
        "RetrLow": retr_low,
        "Fib618": float(fib618),
        "Fib786": float(fib786),
        "SwingRangePct": swing_range_pct,
        "PostSwingBars": post_bars,
    }


def build_daily_audit_and_history(df: pd.DataFrame, days: int = DAILY_HISTORY_DAYS, lookback_days: int = DAILY_LOOKBACK_DAYS):
    audit_rows = []
    history_rows = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").copy()
        latest_as_of = pd.to_datetime(group["Date"].max())
        latest_eval = evaluate_daily_gate(group, latest_as_of, lookback_days=lookback_days)
        audit_rows.append({"Ticker": ticker, **latest_eval})

        as_of_dates = group["Date"].drop_duplicates().sort_values().tail(days)
        for as_of in as_of_dates.tolist():
            gate_eval = evaluate_daily_gate(group, pd.to_datetime(as_of), lookback_days=lookback_days)
            history_rows.append(
                {
                    "Ticker": ticker,
                    "AsOfDate": pd.to_datetime(as_of),
                    "InZone": bool(gate_eval["Gate1_Pass"]),
                    "SwingLowDate": gate_eval["SwingLowDate"],
                    "SwingHighDate": gate_eval["SwingHighDate"],
                    "RetrLowDate": gate_eval["RetrLowDate"],
                    "RetrLow": gate_eval["RetrLow"],
                    "Fib618": gate_eval["Fib618"],
                    "Fib786": gate_eval["Fib786"],
                    "Reason": gate_eval["Gate1_Reason"],
                }
            )

    daily_audit_df = pd.DataFrame(audit_rows)
    if daily_audit_df.empty:
        daily_audit_df = pd.DataFrame(
            columns=[
                "Ticker",
                "AsOf",
                "Gate1_Pass",
                "Gate1_Reason",
                "SwingLowDate",
                "SwingLow",
                "SwingHighDate",
                "SwingHigh",
                "RetrLowDate",
                "RetrLow",
                "Fib618",
                "Fib786",
                "SwingRangePct",
                "PostSwingBars",
            ]
        )

    daily_history_df = pd.DataFrame(history_rows)
    if daily_history_df.empty:
        daily_history_df = pd.DataFrame(
            columns=[
                "Ticker",
                "AsOfDate",
                "InZone",
                "SwingLowDate",
                "SwingHighDate",
                "RetrLowDate",
                "RetrLow",
                "Fib618",
                "Fib786",
                "Reason",
            ]
        )

    daily_audit_df = daily_audit_df.sort_values(["Ticker"]).reset_index(drop=True)
    daily_history_df = daily_history_df.sort_values(["Ticker", "AsOfDate"]).reset_index(drop=True)
    return daily_audit_df, daily_history_df


def build_hourly_list(df_hourly: pd.DataFrame, daily_list_df: pd.DataFrame):
    cols = [
        "Ticker",
        "Daily_AsOf",
        "Daily_RetrLowDate",
        "Daily_RetrLow",
        "local_high_time",
        "local_high",
        "entry",
        "stop",
        "take_profit",
        "triggered_last_bar",
        "last_close",
        "Gate2_Pass",
        "Gate2_Reason",
        "Gate3_Pass",
        "Gate3_Reason",
    ]
    if daily_list_df.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["Ticker", "Gate2_Reason", "Gate3_Reason"])

    rows = []
    rejections = []
    daily_lookup = daily_list_df.set_index("Ticker")

    for ticker in daily_lookup.index:
        drow = daily_lookup.loc[ticker]
        group = df_hourly[df_hourly["Ticker"] == ticker].sort_values("DateTime")
        if group.empty:
            rejections.append({"Ticker": ticker, "Gate2_Reason": "missing_hourly_data", "Gate3_Reason": "gate2_failed"})
            continue

        window = group.tail(HOURS_LOOKBACK).copy()
        if len(window) < 11:
            rejections.append({"Ticker": ticker, "Gate2_Reason": "insufficient_hourly_bars", "Gate3_Reason": "gate2_failed"})
            continue

        highs = window["High"].to_numpy()
        piv = pivot_highs(highs, look=5)
        if len(piv) < (MIN_HH_COUNT + 1):
            rejections.append({"Ticker": ticker, "Gate2_Reason": "not_enough_pivot_highs", "Gate3_Reason": "gate2_failed"})
            continue

        last_piv = piv[-MIN_HH_COUNT:]
        last_piv_values = [float(highs[i]) for i in last_piv]
        if not all(x < y for x, y in zip(last_piv_values, last_piv_values[1:])):
            rejections.append({"Ticker": ticker, "Gate2_Reason": "last_pivot_highs_not_strictly_rising", "Gate3_Reason": "gate2_failed"})
            continue

        local_idx = piv[-1]
        local_high = float(highs[local_idx])
        local_high_time = pd.to_datetime(window.iloc[local_idx]["DateTime"])

        if local_high_time <= pd.to_datetime(drow["RetrLowDate"]):
            rejections.append({"Ticker": ticker, "Gate2_Reason": "latest_pivot_not_post_retracement", "Gate3_Reason": "gate2_failed"})
            continue

        gate2_pass = True
        gate2_reason = "OK"

        last_close = float(window["Close"].iloc[-1])
        pullback_pct = (local_high - last_close) / local_high if local_high > 0 else 0.0
        if pullback_pct < MIN_PULLBACK_PCT:
            rejections.append({"Ticker": ticker, "Gate2_Reason": gate2_reason, "Gate3_Reason": "pullback_below_min"})
            continue

        fib_low = float(drow["RetrLow"])
        entry = local_high - 0.618 * (local_high - fib_low)
        stop = fib_low
        take_profit = local_high

        if not (stop < entry < local_high):
            rejections.append({"Ticker": ticker, "Gate2_Reason": gate2_reason, "Gate3_Reason": "entry_not_between_stop_and_local_high"})
            continue

        last_bar = window.iloc[-1]
        triggered_last_bar = bool(last_bar["Low"] <= entry <= last_bar["High"])
        near_price = abs(last_close - entry) / entry <= 0.02 if entry > 0 else False
        if not (near_price or triggered_last_bar):
            rejections.append({"Ticker": ticker, "Gate2_Reason": gate2_reason, "Gate3_Reason": "entry_too_far_from_last_close"})
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Daily_AsOf": pd.to_datetime(drow["AsOf"]),
                "Daily_RetrLowDate": pd.to_datetime(drow["RetrLowDate"]),
                "Daily_RetrLow": fib_low,
                "local_high_time": local_high_time,
                "local_high": local_high,
                "entry": float(entry),
                "stop": float(stop),
                "take_profit": float(take_profit),
                "triggered_last_bar": triggered_last_bar,
                "last_close": last_close,
                "Gate2_Pass": gate2_pass,
                "Gate2_Reason": gate2_reason,
                "Gate3_Pass": True,
                "Gate3_Reason": "OK",
            }
        )

    hourly_list_df = pd.DataFrame(rows)
    if hourly_list_df.empty:
        hourly_list_df = pd.DataFrame(columns=cols)
    else:
        hourly_list_df = hourly_list_df[cols].sort_values(["Ticker"]).reset_index(drop=True)

    reject_df = pd.DataFrame(rejections)
    if reject_df.empty:
        reject_df = pd.DataFrame(columns=["Ticker", "Gate2_Reason", "Gate3_Reason"])

    return hourly_list_df, reject_df


def _print_rejection_summary(daily_audit_df: pd.DataFrame, gate23_reject_df: pd.DataFrame):
    print("\\nGate1 rejection summary:")
    gate1_reasons = daily_audit_df.loc[~daily_audit_df["Gate1_Pass"], "Gate1_Reason"].value_counts()
    if gate1_reasons.empty:
        print("  none")
    else:
        for reason, count in gate1_reasons.items():
            print(f"  {reason}: {count}")

    print("Gate2 rejection summary:")
    gate2_reasons = gate23_reject_df[gate23_reject_df["Gate2_Reason"] != "OK"]["Gate2_Reason"].value_counts()
    if gate2_reasons.empty:
        print("  none")
    else:
        for reason, count in gate2_reasons.items():
            print(f"  {reason}: {count}")

    print("Gate3 rejection summary:")
    gate3_reasons = gate23_reject_df[
        (gate23_reject_df["Gate2_Reason"] == "OK") & (gate23_reject_df["Gate3_Reason"] != "OK")
    ]["Gate3_Reason"].value_counts()
    if gate3_reasons.empty:
        print("  none")
    else:
        for reason, count in gate3_reasons.items():
            print(f"  {reason}: {count}")


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

    daily_audit_df, daily_history_df = build_daily_audit_and_history(
        df_daily,
        days=DAILY_HISTORY_DAYS,
        lookback_days=DAILY_LOOKBACK_DAYS,
    )

    daily_list_df = (
        daily_audit_df[daily_audit_df["Gate1_Pass"]]
        .sort_values(["Ticker"])
        .reset_index(drop=True)
    )

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

    hourly_list_df, gate23_reject_df = build_hourly_list(df_hourly, daily_list_df)

    _print_rejection_summary(daily_audit_df, gate23_reject_df)

    # keep objects available for audit/debugging
    run_engine.daily_audit_df = daily_audit_df
    run_engine.daily_history_df = daily_history_df

    return daily_list_df, hourly_list_df
