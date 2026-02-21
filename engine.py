import pandas as pd
import numpy as np
import yfinance as yf  # still unused but kept in case you extend later
from datetime import datetime, timedelta
from updater import load_all_market_data, load_all_market_data_hourly


# Optional: console display settings (used only if you print from here)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", None)

# --- Parameters ---
LOOKBACK_DAYS = 300
BOUNCE_TOL = 0.01



# =========================================================
# 2. Swing Logic Using HIGH/LOW (Fibonacci)
# =========================================================
def find_swing_as_of_quick(group, current_date, lookback_days=LOOKBACK_DAYS):
    """
    Swing high based on HIGH prices.
    Swing low based on LOW prices.
    """

    window = group[
        (group["Date"] <= current_date)
        & (group["Date"] >= (current_date - pd.Timedelta(days=lookback_days)))
    ]

    if len(window) < 10:
        return None

    highs = window["High"].values
    lows = window["Low"].values
    dates = window["Date"].values

    look = 5
    pivots = []

    # Find local maxima using HIGH
    for i in range(look, len(highs) - look):
        if highs[i] == max(highs[i - look : i + look + 1]):
            pivots.append(i)

    if not pivots:
        return None

    # Most prominent swing high (highest high)
    best_rel_idx = max(pivots, key=lambda idx: highs[idx])
    swing_high_price = float(highs[best_rel_idx])
    swing_high_date = pd.to_datetime(dates[best_rel_idx])

    # Find swing low BEFORE that pivot based on LOW
    prior_segment = window.iloc[: best_rel_idx + 1]
    low_idx = prior_segment["Low"].idxmin()

    swing_low_price = float(group.loc[low_idx, "Low"])
    swing_low_date = pd.to_datetime(group.loc[low_idx, "Date"])

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


# =========================================================
# 3. Build Watchlist for All Tickers
# =========================================================
def build_watchlist(df, lookback_days=LOOKBACK_DAYS):
    rows = []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date")

        latest_price = group["Close"].iloc[-1]  # use Close for latest price
        latest_date = group["Date"].iloc[-1]

        swing = find_swing_as_of_quick(group, latest_date, lookback_days)
        if swing is None:
            continue

        # Check if price broke below 78.6%
        post_high = group[
            (group["Date"] > swing["Swing High Date"])
            & (group["Date"] <= latest_date)
        ]

        if (not post_high.empty) and (
            post_high["Low"] < swing["Stop Consider (78.6%)"]
        ).any():
            continue

        retracement = (swing["Swing High Price"] - latest_price) / (
            swing["Swing High Price"] - swing["Swing Low Price"]
        )

        if retracement >= 0.38:  # minimal valid retracement
            rows.append(
                {
                    "Ticker": ticker,
                    "Latest Date": latest_date,
                    "Latest Price": float(latest_price),
                    "Swing Low Date": swing["Swing Low Date"],
                    "Swing Low Price": float(swing["Swing Low Price"]),
                    "Swing High Date": swing["Swing High Date"],
                    "Swing High Price": float(swing["Swing High Price"]),
                    "Swing Range": float(
                        swing["Swing High Price"] - swing["Swing Low Price"]
                    ),
                    "Retracement": float(retracement),
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# 4. Correct Fibonacci retracement scoring
# =========================================================
def fib_retrace_score(r):
    if r < 38.2:
        return 0
    if r > 78.6:
        return 0
    if r <= 61.8:
        return (r - 38.2) / (61.8 - 38.2)
    return (78.6 - r) / (78.6 - 61.8)


# =========================================================
# 6. Confirmation Engine: Retracement Held + Uptrend Resumed
# =========================================================
def confirmation_engine(df, watch):
    """
    For each ticker in the watchlist:
    - Confirm retracement held at Fib support
    - Confirm structure break (BOS)
    - Confirm momentum resumption
    Returns a new dataframe with confirmation columns.
    """

    results = []

    for _, row in watch.iterrows():
        ticker = row["Ticker"]
        swing_low_date = row["Swing Low Date"]
        swing_high_date = row["Swing High Date"]

        # Full history for this ticker
        g = df[df["Ticker"] == ticker].sort_values("Date").copy()

        # ----------------------------
        # Pre-compute Fib levels for this swing
        # ----------------------------
        fib50 = row["Swing High Price"] - 0.50 * (row["Swing High Price"] - row["Swing Low Price"])
        fib786 = row["Swing High Price"] - 0.786 * (row["Swing High Price"] - row["Swing Low Price"])

        # --------------------------------------------------
        # Identify RETRACEMENT LEG (from swing high downward)
        # --------------------------------------------------
        correction = g[(g["Date"] > swing_high_date) &
                       (g["Date"] <= row["Latest Date"])].copy()
        if correction.empty:
            # if no data after swing high, skip
            continue

        # This is the "circled" bar: lowest low after swing high
        retr_idx = correction["Low"].idxmin()
        retr_low_price = correction.loc[retr_idx, "Low"]
        retr_low_date = correction.loc[retr_idx, "Date"]

        # Segment AFTER the retracement low (for HL, etc.)
        post = g[g["Date"] > retr_low_date].copy()

        # ----------------------------
        # CHECK 1 — Retracement Held at Fib Support
        # ----------------------------

        # (a) Did price find its low *inside* the Fib zone?
        retr_in_zone = (retr_low_price <= fib50) and (retr_low_price >= fib786)

        # (b) After that low, did price avoid making a lower low?
        if post.empty:
            no_lower_after = True
        else:
            no_lower_after = post["Low"].min() >= retr_low_price

        retracement_floor_respected = retr_in_zone and no_lower_after

        # ----------------------------
        # CHECK 2 — Higher Low formed (relative to retracement low)
        # ----------------------------
        higher_low_found = False
        hl_price = np.nan

        if len(post) >= 3:
            lows = post["Low"].values

            # pivot lows in the segment AFTER retracement
            pivot_lows = []
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                    pivot_lows.append(i)

            for idx in pivot_lows:
                pivot_low = lows[idx]

                # (1) must be above the retracement low
                if pivot_low <= retr_low_price:
                    continue

                # (2) later candles must not undercut this pivot low
                if post["Low"].iloc[idx + 1:].min() < pivot_low:
                    continue

                # (3) some upward follow-through
                has_green_follow_through = False
                if idx + 1 < len(post):
                    if post["Close"].iloc[idx + 1] > post["Open"].iloc[idx + 1]:
                        has_green_follow_through = True

                broke_minor_high = False
                if idx + 2 < len(post):
                    minor_high = max(post["High"].iloc[idx:idx + 2])
                    if post["High"].iloc[idx + 2:].max() > minor_high:
                        broke_minor_high = True

                if has_green_follow_through or broke_minor_high:
                    higher_low_found = True
                    hl_price = float(pivot_low)
                    break

        # ----------------------------
        # CHECK 3 — Bullish Reaction Candle at Fib zone (enhanced)
        # ----------------------------

        bullish_candle = False
        corr = correction.reset_index(drop=True)

        for i in range(2, len(corr)):
            # Current candle
            o  = corr["Open"].iloc[i]
            c  = corr["Close"].iloc[i]
            h  = corr["High"].iloc[i]
            l  = corr["Low"].iloc[i]
            body = abs(c - o)
            range_ = max(h - l, 1e-9)
            lower_wick = (o - l) if c >= o else (c - l)

            # Prior candles
            o1 = corr["Open"].iloc[i-1]
            c1 = corr["Close"].iloc[i-1]
            o2 = corr["Open"].iloc[i-2]
            c2 = corr["Close"].iloc[i-2]

            in_fib_zone = (l <= fib50) and (l >= fib786)

            # --- 1) Hammer / Pin bar ---
            hammer = (
                in_fib_zone and
                lower_wick > 0.6 * range_ and
                c >= o
            )

            # --- 2) Bullish Engulfing ---
            engulf = (
                in_fib_zone and
                (c > o1) and
                (o < c1) and
                (c1 < o1)    # previous candle bearish
            )

            # --- 3) Morning Star (3-candle pattern) ---
            morning_star = (
                in_fib_zone and
                (c1 < o1) and                        # candle 1 red
                (abs(c2 - o2) <= 0.3 * (corr["High"].iloc[i-2] - corr["Low"].iloc[i-2])) and  # small C2 body
                (c > (o1 + c1) / 2)                  # candle 3 closes above midpoint of candle1
            )

            # --- 4) Piercing Line ---
            piercing = (
                in_fib_zone and
                (c1 < o1) and                        # previous candle red
                (o < c1) and                         # gap continuation
                (c > (o1 + c1) / 2)                  # closes above midpoint of prior body
            )

            # --- 5) Tweezer Bottom ---
            tweezer = (
                abs(l - corr["Low"].iloc[i-1]) <= 0.2 * range_ and   # equal lows
                in_fib_zone and
                (c >= o)                                             # current candle bullish or neutral
            )

            # --- 6) Strong rejection wick ---
            strong_reversal = (
                in_fib_zone and
                c >= l + 0.6 * range_
            )

            if hammer or engulf or morning_star or piercing or tweezer or strong_reversal:
                bullish_candle = True
                break

        # ----------------------------
        # CHECK 4 — Proper BOS logic
        # ----------------------------

        # Only examine bars AFTER the retracement low but BEFORE the current bar
        corr = g[(g["Date"] > retr_low_date) & (g["Date"] < row["Latest Date"])].copy()

        if corr.empty or len(corr) < 3:
            last_local_high = np.nan
            bos = False
        else:
            highs = corr["High"].values

            # find pivot highs (3-bar or 5-bar pivot)
            pivot_highs = []
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                    highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                    pivot_highs.append(highs[i])

            # if no pivots, fallback to max high BEFORE breakout
            if pivot_highs:
                bos_level = max(pivot_highs)
            else:
                bos_level = corr["High"].max()

            last_local_high = float(bos_level)

            # BOS occurs if ANY candle AFTER retr low closed above bos_level
            post = g[g["Date"] > retr_low_date]
            bos = (post["Close"] > bos_level).any()

        # CHECK 5 — Momentum checks (unchanged)
        # ----------------------------
        gp = g.copy()

        gp["SMA10"] = gp["Close"].rolling(10).mean()

        delta = gp["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()
        rs = roll_up / roll_down
        gp["RSI"] = 100 - (100 / (1 + rs))

        ema12 = gp["Close"].ewm(span=12, adjust=False).mean()
        ema26 = gp["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        gp["MACDH"] = macd - signal

        last_row = gp.iloc[-1]
        close_now = last_row["Close"]
        sma10 = last_row["SMA10"]
        rsi_now = last_row["RSI"]
        macdh_now = last_row["MACDH"]

        # 2-of-3 rule
        cond1 = close_now > sma10
        cond2 = macdh_now > 0
        cond3 = rsi_now > 50
        two_of_three = (cond1 + cond2 + cond3) >= 2

        # Breakout momentum
        macd_line = macd.iloc[-1]
        macd_line_prev = macd.iloc[-2]
        macd_cross_up = macd_line > macd_line_prev

        rsi_strong = rsi_now > 55
        last3_high = gp["High"].iloc[-3:].max()
        price_breakout = close_now > last3_high

        breakout_momentum = macd_cross_up or (rsi_strong and price_breakout)

        momentum_ok = two_of_three or breakout_momentum

        # ---------------------------------
        # BOS Strength (0–100, diagnostic only)
        # ---------------------------------
        if not np.isnan(last_local_high):
            close_now_for_bos = row["Latest Price"]

            # 1) Break distance (0–40 points)
            break_dist = max(close_now_for_bos - last_local_high, 0)
            break_dist_pct = break_dist / last_local_high
            score_break = min(break_dist_pct / 0.02, 1.0) * 40
            # (2% breakout = full 40 pts)

            # 2) Candle strength (0–30 points)
            latest_bar = g.iloc[-1]
            high_ = latest_bar["High"]
            low_ = latest_bar["Low"]
            open_ = latest_bar["Open"]
            close_ = latest_bar["Close"]

            bar_range = max(high_ - low_, 1e-9)
            body = abs(close_ - open_)
            close_pos = (close_ - low_) / bar_range

            body_score = (body / bar_range)            # 1 = strong body
            close_score = close_pos                    # 1 = closes at high

            score_candle = 30 * min((0.6 * body_score + 0.4 * close_score), 1.0)

            # 3) Momentum alignment (0–30 points)
            momentum_flags = 0
            momentum_flags += 1 if macdh_now > 0 else 0
            momentum_flags += 1 if rsi_now > 55 else 0
            momentum_flags += 1 if close_ > sma10 else 0

            score_momentum = (momentum_flags / 3) * 30

            # Final BOS strength score
            bos_strength = score_break + score_candle + score_momentum

        else:
            bos_strength = 0.0

        shape = setup_shape(g, retr_low_date, last_local_high)

        # ----------------------------
        # FINAL SIGNAL
        # ----------------------------
        retracement_held = (
            retracement_floor_respected and
            higher_low_found and
            bullish_candle
        )
        uptrend_resumed = bos and momentum_ok

        final_signal = (
            "BUY"
            if retracement_held and uptrend_resumed
            else "WATCH"
            if retracement_held
            else "INVALID"
        )

        row_dict = {
            "Ticker": ticker,
            "Retr Held": retracement_held,
            "HL Formed": higher_low_found,
            "HL Price": hl_price,
            "Bullish Candle": bullish_candle,
            "BOS": bos,
            "Momentum OK": momentum_ok,
            "FINAL_SIGNAL": final_signal,
            "LastLocalHigh": last_local_high,
            "Latest Price": row["Latest Price"],
            "BOS_Strengh": bos_strength,
            "Shape": shape,
            "SwingRange": row["Swing Range"],
            "SwingLow": row["Swing Low Price"],
            "SwingHigh": row["Swing High Price"],
            "DailyRetrLowDate": pd.to_datetime(retr_low_date),
            "DailyRetrLowPrice": float(retr_low_price),
        }

        # Inject recent history for compression scoring
        row_dict["__gdata__"] = gp.tail(40).copy()

        results.append(row_dict)

    return pd.DataFrame(results)


def _hourly_pivot_high_indices(highs, look=5):
    pivots = []
    if len(highs) < 2 * look + 1:
        return pivots

    for i in range(look, len(highs) - look):
        if highs[i] == max(highs[i - look : i + look + 1]):
            pivots.append(i)

    return pivots


def build_hourly_entries(
    combined,
    hourly_df,
    pivot_look=5,
    min_hh_count=2,
    min_pullback_pct=0.002,
    max_pullback_pct=0.12,
    near_entry_tol=0.02,
    max_bars_after_low=320,
    min_bars_since_high=3,
):
    """
    Build hourly entry candidates from DAILY-qualified tickers (combined/List A).
    Returns:
    - hourly_entries_df: actionable hourly candidates
    - hourly_rejects_df: optional diagnostics for rejected tickers
    """
    if combined is None or combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    if hourly_df is None or hourly_df.empty:
        rejects = combined[["Ticker"]].copy()
        rejects["RejectReason"] = "missing_hourly_data"
        return pd.DataFrame(), rejects

    required_combined = {"Ticker", "DailyRetrLowDate", "DailyRetrLowPrice"}
    required_hourly = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}

    if required_combined.difference(combined.columns):
        raise RuntimeError(
            "combined dataframe missing required daily retracement columns for hourly scan"
        )
    if required_hourly.difference(hourly_df.columns):
        raise RuntimeError("hourly dataframe missing required OHLC columns")

    entries = []
    rejects = []

    # Kept for signature stability.
    _ = near_entry_tol
    _ = pivot_look
    _ = min_hh_count
    _ = min_pullback_pct
    _ = max_pullback_pct
    _ = min_bars_since_high
    # Kept for signature stability; hourly inclusion no longer depends on these checks.
    _ = near_entry_tol
    _ = pivot_look
    _ = min_hh_count

    hourly = hourly_df.copy()
    hourly["DateTime"] = pd.to_datetime(hourly["DateTime"], errors="coerce")
    hourly = hourly.dropna(subset=["DateTime"]).copy()

    if getattr(hourly["DateTime"].dt, "tz", None) is not None:
        hourly["DateTime"] = hourly["DateTime"].dt.tz_convert("UTC").dt.tz_localize(None)

    hourly = hourly.sort_values(["Ticker", "DateTime"])

    for _, row in combined.iterrows():
        ticker = row["Ticker"]
        retr_low_date = pd.to_datetime(row["DailyRetrLowDate"], errors="coerce")
        retr_low_price = float(row["DailyRetrLowPrice"])

        g = hourly[hourly["Ticker"] == ticker].copy()
        if g.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "missing_hourly_data"})
            continue

        if pd.isna(retr_low_date):
            rejects.append({"Ticker": ticker, "RejectReason": "invalid_daily_low_datetime"})
            continue

        if getattr(g["DateTime"].dt, "tz", None) is not None and retr_low_date.tzinfo is None:
            retr_low_date = retr_low_date.tz_localize("UTC").tz_localize(None)
        elif getattr(g["DateTime"].dt, "tz", None) is None and retr_low_date.tzinfo is not None:
            retr_low_date = retr_low_date.tz_convert("UTC").tz_localize(None)
        elif retr_low_date.tzinfo is not None:
            retr_low_date = retr_low_date.tz_convert("UTC").tz_localize(None)

        after_low = g[g["DateTime"] > retr_low_date].copy()
        if after_low.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "no_hourly_after_daily_low"})
            continue

        if len(after_low) > max_bars_after_low:
            after_low = after_low.tail(max_bars_after_low).copy()

        if len(after_low) < 20:
            rejects.append({"Ticker": ticker, "RejectReason": "insufficient_hourly_bars"})
            continue

        prev_close = after_low["Close"].shift(1)
        tr1 = after_low["High"] - after_low["Low"]
        tr2 = (after_low["High"] - prev_close).abs()
        tr3 = (after_low["Low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = true_range.rolling(14).mean()

        impulse_move = after_low["Close"] - after_low["Close"].shift(6)
        impulse_mask = impulse_move > (1.2 * atr14)

        impulse_indices = after_low.index[impulse_mask.fillna(False)]
        if len(impulse_indices) == 0:
            rejects.append({"Ticker": ticker, "RejectReason": "no_impulse_bar"})
            continue

        impulse_idx = impulse_indices[-1]
        impulse_pos = after_low.index.get_loc(impulse_idx)
        window_start = max(0, impulse_pos - 1)
        window_end = min(len(after_low), impulse_pos + 2)
        local_window = after_low.iloc[window_start:window_end]

        if local_window.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "invalid_impulse_window"})
            continue


        if local_window.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "invalid_impulse_window"})
            continue
        if len(after_low) <= min_bars_since_high:
            rejects.append({"Ticker": ticker, "RejectReason": "high_too_recent"})
            continue

        eligible = after_low.iloc[:-min_bars_since_high]
        if eligible.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "high_too_recent"})
            continue

        local_idx = eligible["High"].idxmax()
        local_high = float(eligible.loc[local_idx, "High"])
        local_high_time = pd.to_datetime(eligible.loc[local_idx, "DateTime"])
        local_idx = after_low["High"].idxmax()
        local_high = float(after_low.loc[local_idx, "High"])
        local_high_time = pd.to_datetime(after_low.loc[local_idx, "DateTime"])

        local_idx = local_window["High"].idxmax()
        local_high = float(local_window.loc[local_idx, "High"])
        local_high_time = pd.to_datetime(local_window.loc[local_idx, "DateTime"])

        last_bar = after_low.iloc[-1]
        last_close = float(last_bar["Close"])
        last_low = float(last_bar["Low"])
        last_high = float(last_bar["High"])

        if len(after_low) < 2:
            rejects.append({"Ticker": ticker, "RejectReason": "insufficient_recent_bars"})
            continue

        previous_close = float(after_low.iloc[-2]["Close"])
        currently_pulling_back = (last_close < local_high) and (last_close < previous_close)
        if not currently_pulling_back:
            rejects.append({"Ticker": ticker, "RejectReason": "not_currently_pulling_back"})
        bars_since_high = int((after_low["DateTime"] > local_high_time).sum())
        if bars_since_high < 3:
            rejects.append({"Ticker": ticker, "RejectReason": "high_too_recent"})
            continue

        retrace_from_high_pct = (local_high - last_close) / local_high
        retracing_now = (
            (last_close < local_high)
            and (retrace_from_high_pct >= min_pullback_pct)
            and (retrace_from_high_pct <= max_pullback_pct)
        )
        if not retracing_now:
            rejects.append({"Ticker": ticker, "RejectReason": "no_pullback"})
            continue

        fib_low = retr_low_price
        fib_high = local_high
        entry_382 = fib_high - 0.382 * (fib_high - fib_low)
        entry_50 = fib_high - 0.50 * (fib_high - fib_low)
        entry_618 = fib_high - 0.618 * (fib_high - fib_low)
        stop = fib_low
        take_profit = fib_high

        if not (stop < entry_618 < take_profit):
            rejects.append({"Ticker": ticker, "RejectReason": "invalid_trade_levels"})
            continue

        retrace_from_high_pct = (local_high - last_close) / local_high
        distance_to_entry_618_pct = (last_close - entry_618) / entry_618
        entry_618_hit = (last_low <= entry_618) and (entry_618 <= last_high)
        bars_since_high = int((after_low["DateTime"] > local_high_time).sum())
        distance_to_entry_618_pct = (last_close - entry_618) / entry_618
        entry_618_hit = (last_low <= entry_618) and (entry_618 <= last_high)

        entries.append(
            {
                "Ticker": ticker,
                "DailyRetrLowDate": retr_low_date,
                "DailyRetrLowPrice": fib_low,
                "local_high_time": local_high_time,
                "local_high": fib_high,
                "entry_382": entry_382,
                "entry_50": entry_50,
                "entry_618": entry_618,
                "stop": stop,
                "take_profit": take_profit,
                "last_close": last_close,
                "last_low": last_low,
                "last_high": last_high,
                "retrace_from_high_pct": retrace_from_high_pct,
                "bars_since_high": bars_since_high,
                "distance_to_entry_618_pct": distance_to_entry_618_pct,
                "entry_618_hit": bool(entry_618_hit),
            }
        )

    entries_df = pd.DataFrame(entries)
    rejects_df = pd.DataFrame(rejects)

    if not entries_df.empty:
        entries_df["abs_distance_to_entry_618_pct"] = entries_df["distance_to_entry_618_pct"].abs()
        entries_df = entries_df.sort_values(
            by=["entry_618_hit", "abs_distance_to_entry_618_pct", "retrace_from_high_pct"],
            ascending=[False, True, False],
        ).drop(columns=["abs_distance_to_entry_618_pct"]).reset_index(drop=True)

    return entries_df, rejects_df


def setup_shape(g, retr_low_date, last_local_high):
    """
    Labels the structural shape of price AFTER the retracement low.
    More robust and less noisy than previous version.
    """

    post = g[g["Date"] > retr_low_date].copy()
    if post.empty or len(post) < 6:
        return "insufficient data"

    closes = post["Close"].values
    highs = post["High"].values
    lows = post["Low"].values

    x = np.arange(len(closes))

    # --- Trend slope ---
    coeffs = np.polyfit(x, closes, 1)
    slope = coeffs[0]

    # --- Smoothness (noise level) ---
    fitted = np.polyval(coeffs, x)
    noise = np.std(closes - fitted)
    noise_ratio = noise / np.mean(closes)

    # --- % recovery from retr low to current close ---
    total_up = closes[-1] - closes[0]
    range_up = max(closes) - min(closes)
    if range_up == 0:
        recovery_pct = 0
    else:
        recovery_pct = total_up / range_up

    # --- BOS distance ---
    if not np.isnan(last_local_high):
        dist_to_bos = (last_local_high - closes[-1]) / last_local_high
    else:
        dist_to_bos = None

    # -----------------------------------------------------------
    # SHAPE CLASSIFICATION (ordered from tightest to loosest)
    # -----------------------------------------------------------

    # 1) CONSOLIDATION UNDER BOS
    if dist_to_bos is not None and dist_to_bos < 0.02 and noise_ratio < 0.008:
        return "consolidation under BOS"

    # 2) ROUNDED RECOVERY
    if slope > 0 and noise_ratio < 0.015 and recovery_pct > 0.60:
        return "rounded recovery"

    # 3) STRONG RECOVERY
    if slope > 0 and recovery_pct > 0.75:
        return "strong recovery"

    # 4) TRUE V-REVERSAL (sharp move, low noise)
    if slope > np.mean(closes) * 0.0008 and recovery_pct > 0.85 and noise_ratio < 0.02:
        return "V-reversal"

    # 5) WIDE PULLBACK / VOLATILE
    if noise_ratio > 0.03:
        return "volatile pullback"

    return "normal recovery"


def shape_priority(shape):
    order = {
        "consolidation under BOS": 1,
        "rounded recovery": 2,
        "strong recovery": 3,
        "normal recovery": 4,
        "V-reversal": 5,
        "volatile pullback": 6,
        "insufficient data": 7
    }
    return order.get(shape, 7)


# =========================================================
# NEXT ACTION
# =========================================================
def next_action(row):
    """
    Compact one-line trading instruction.
    BUY → "Limit: X"
    WATCH → "Trigger > X | Invalidate < Y"
    INVALID → "Not actionable"
    """
    bos = row["LastLocalHigh"]
    hl = row["HL Price"]

    # ---------- BUY ----------
    if row["FINAL_SIGNAL"] == "BUY":
        # Preferred entry = retest of BOS
        if not np.isnan(bos):
            return f"Limit: {bos:.2f}"
        else:
            return "Limit: N/A"

    # ---------- WATCH ----------
    if row["FINAL_SIGNAL"] == "WATCH":
        trigger = f"Trigger > {bos:.2f}" if not np.isnan(bos) else "Trigger > N/A"
        inval = f"Invalidate < {hl:.2f}" if row["HL Formed"] else "Invalidate < N/A"
        return f"{trigger} | {inval}"

    # ---------- INVALID ----------
    return "Not actionable"


# =========================================================
# 8. BUY QUALITY SCORE
# =========================================================
def compute_buy_quality(row):
    if row["FINAL_SIGNAL"] != "BUY":
        return None

    retr = 1 if row["Retr Held"] else 0
    hl = 1 if row["HL Formed"] else 0
    candle = 1 if row["Bullish Candle"] else 0
    bos = 1 if row["BOS"] else 0
    momentum = 1 if row["Momentum OK"] else 0

    score = 100 * (
        0.25 * bos
        + 0.25 * momentum
        + 0.20 * candle
        + 0.15 * hl
        + 0.15 * retr
    )
    return round(score, 2)


# =========================================================
# 9. WATCH READINESS SCORE
# =========================================================
def compute_watch_readiness(row):
    """
    How close the structure is to a valid BUY.
    Applies to ALL signals.
    BUY will get 100.
    """
    if row["FINAL_SIGNAL"] == "BUY":
        return 100.0

    # retracement
    retr = 1 if row["Retr Held"] else 0

    # HL formation
    hl = 1 if row["HL Formed"] else 0

    # candle
    candle = 1 if row["Bullish Candle"] else 0

    # momentum
    mom = 1 if row["Momentum OK"] else 0

    # BOS proximity
    bos = row["LastLocalHigh"]
    price = row["Latest Price"]
    if np.isnan(bos):
        bos_prox = 0
    else:
        bos_dist = bos - price
        raw = 1 - bos_dist / max(price, 1e-9)
        bos_prox = np.clip(raw, 0, 1)

    score = 100 * (
        0.25 * retr +
        0.20 * hl +
        0.15 * candle +
        0.20 * mom +
        0.20 * bos_prox
    )

    return round(np.clip(score, 0, 100), 2)


# =========================================================
# 10. PERFECT ENTRY, ENTRY BIAS & BREAKOUT PRESSURE
# =========================================================
def compute_perfect_entry(row):
    """
    Perfect Entry = clean fib retracement + valid HL + structure intact.
    BUY setups get a small bonus if they are fully confirmed.
    """

    price = row["Latest Price"]
    low = row["SwingLow"]
    high = row["SwingHigh"]
    hl = row["HL Price"]

    if np.isnan(low) or np.isnan(high):
        return None

    swing = high - low
    if swing <= 0:
        return None

    # ============================================
    # 1. RETRACEMENT QUALITY (core of PE)
    # ============================================
    retr = (high - hl) / swing  # HL relative position between low & high
    ideal = 0.56                # target ~56% depth
    dist = abs(retr - ideal)

    # hard reject if HL is in low-quality region
    if retr < 0.38 or retr > 0.78:
        base_score = 60
    else:
        if dist < 0.05:
            retr_score = 40        # excellent
        elif dist < 0.10:
            retr_score = 30        # good
        else:
            retr_score = 15        # acceptable

        # ============================================
        # 2. HL confirmation
        # ============================================
        hl_score = 30 if row["HL Formed"] else 0

        # ============================================
        # 3. Structure quality (shape)
        # ============================================
        pr = row.get("ShapePriority", 5)

        if pr == 1:            # consolidation under BOS
            shape_score = 20
        elif pr == 2:          # rounded recovery
            shape_score = 15
        elif pr == 3:          # strong recovery
            shape_score = 10
        else:
            shape_score = 0

        # ============================================
        # 4. Proximity to BOS (soft requirement)
        # ============================================
        bos = row["LastLocalHigh"]
        if not np.isnan(bos):
            dist_bos = (bos - price) / bos
            if dist_bos < 0.03:
                bos_score = 10
            elif dist_bos < 0.06:
                bos_score = 6
            else:
                bos_score = 0
        else:
            bos_score = 0

        base_score = retr_score + hl_score + shape_score + bos_score

    score = base_score

    # ============================================
    # 5. BONUS FOR FULLY CONFIRMED BUY
    # ============================================
    if (
        row["FINAL_SIGNAL"] == "BUY"
        and row["Retr Held"]
        and row["Momentum OK"]
        and row["READINESS_SCORE"] >= 95
    ):
        score += 15  # push best BUYs like NCLH into the 90s

    # cap for INVALID setups
    if row["FINAL_SIGNAL"] == "INVALID":
        score = min(score, 55)

    return round(min(score, 100), 2)


def compute_entry_bias(row):
    """
    Predict whether BUY should be:
    - RETEST LIKELY
    - BREAKOUT CONTINUATION LIKELY
    - NEUTRAL / RANGE
    """
    if row["FINAL_SIGNAL"] != "BUY":
        return None

    price = row["Latest Price"]
    bos = row["LastLocalHigh"]

    if np.isnan(bos) or bos == 0:
        return "NEUTRAL"

    # -----------------------------
    # 1. Price position relative to BOS
    # -----------------------------
    dist_pct = (price - bos) / bos

    # -----------------------------
    # 2. Shape bias (tight = retest, steep = breakout)
    # -----------------------------
    shape = row["Shape"]

    # -----------------------------
    # 3. Signal logic
    # -----------------------------
    # A) Strong continuation behaviour
    if dist_pct > 0.035 and shape in ("strong recovery", "V-reversal"):
        return "BREAKOUT CONTINUATION LIKELY"

    # B) Tight consolidation under BOS → retest usually happens
    if dist_pct < 0.015 and shape in ("consolidation under BOS", "rounded recovery"):
        return "RETEST LIKELY"

    # C) Low noise but no breakout yet → neutral base
    if shape == "normal recovery":
        return "BASE FORMATION"

    return "NEUTRAL"


def compute_breakout_pressure(row):
    """
    Breakout pressure for ALL signals:
    BUY  → continuation pressure after BOS
    WATCH → pressure under BOS
    INVALID → always low but computed
    """
    price = row["Latest Price"]
    bos = row["LastLocalHigh"]
    hl = row["HL Price"]

    # 1. BOS proximity (for all signals)
    if np.isnan(bos):
        bos_prox = 0
    else:
        raw = 1 - ((bos - price) / max(price, 1e-9))
        bos_prox = np.clip(raw, 0, 1)

    # 2. HL recovery (for all)
    if np.isnan(hl):
        hl_rec = 0
    else:
        hl_rec = np.clip((price - hl) / (0.35 * price), 0, 1)

    # 3. Momentum
    mom = 1 if row["Momentum OK"] else 0

    # 4. Compression (if available)
    g = row.get("__gdata__", None)
    if g is not None and len(g) >= 20:
        atr5 = g["High"].tail(5).max() - g["Low"].tail(5).min()
        atr20 = g["High"].tail(20).max() - g["Low"].tail(20).min()
        comp = np.clip(1 - atr5 / max(atr20, 1e-9), 0, 1)
    else:
        comp = 0

    # BUY → continuation pressure
    if row["FINAL_SIGNAL"] == "BUY" and not np.isnan(bos):
        cont = (price - bos) / (0.02 * bos)
        cont = np.clip(cont, 0, 1)
        bos_prox = max(bos_prox, cont)

    score = 100 * (
        0.40 * bos_prox +
        0.25 * hl_rec +
        0.20 * mom +
        0.15 * comp
    )

    if row["FINAL_SIGNAL"] == "INVALID":
        score = min(score, 40.0)   # invalid setups should not show pressure

    return round(np.clip(score, 0, 100), 2)


def generate_insight_tags(row):
    tags = []

    price = row["Latest Price"]
    bos = row["LastLocalHigh"]
    hl = row["HL Price"]
    g = row.get("__gdata__", None)

    # ======================================================
    # 0 — SAFETY GUARDS
    # ======================================================
    if g is None or len(g) < 5:
        return ""

    # -----------------------------
    # Precompute ATR% for volatility-adjusted logic
    # -----------------------------
    recent = g.tail(20)
    atr = (recent["High"] - recent["Low"]).rolling(5).mean().iloc[-1]
    atr_pct = atr / max(price, 1e-9)

    # ======================================================
    # 1 — PRIME WATCH
    # ======================================================
    if (
        row["FINAL_SIGNAL"] == "WATCH"
        and row["READINESS_SCORE"] >= 95
        and row["BREAKOUT_PRESSURE"] >= 60
    ):
        tags.append("🔥 PRIME")

    # ======================================================
    # 2 — BOS IMMINENT (volatility-aware)
    # ======================================================
    if not np.isnan(bos):
        dist = (bos - price) / bos
        if 0 <= dist < max(0.0075, 0.75 * atr_pct):
            tags.append("⚡ BOS_IMMINENT")

    # ======================================================
    # 3 — VOLATILITY SQUEEZE (tight + low momentum)
    # ======================================================
    if len(g) >= 20:
        atr5 = g["High"].tail(5).max() - g["Low"].tail(5).min()
        atr20 = g["High"].tail(20).max() - g["Low"].tail(20).min()
        comp = np.clip(1 - atr5 / max(atr20, 1e-9), 0, 1)

        mac_flat = abs(g["MACDH"].tail(5).mean()) < 0.15 * abs(g["MACDH"].tail(20).std() + 1e-9)

        if comp > 0.65 and mac_flat:
            tags.append("📉 SQUEEZE")

    # ======================================================
    # 4 — PERFECT ENTRY (BUY-biased)
    # ======================================================
    pe = row.get("PERFECT_ENTRY", None)

    if pe is not None:
        # Primary use-case: confirmed BUYs like NCLH
        if row["FINAL_SIGNAL"] == "BUY" and pe >= 80:
            tags.append("🎯 PERFECT_ENTRY")
        # Optional: ultra-clean WATCH, very close to flipping BUY
        elif (
            row["FINAL_SIGNAL"] == "WATCH"
            and pe >= 90
            and row["READINESS_SCORE"] >= 95
        ):
            tags.append("🎯 PERFECT_ENTRY")

    # ======================================================
    # 5 — STRUCTURE STRONG (now with quality filter)
    # ======================================================
    if (
        row["FINAL_SIGNAL"] != "BUY"
        and row["ShapePriority"] <= 3
        and row["READINESS_SCORE"] > 80
    ):
        # quality filter → avoid noisy structures
        if g["MACDH"].iloc[-1] > 0 or price > recent["Close"].mean():
            tags.append("🌀 STRUCTURE_STRONG")

    # ======================================================
    # 6 — EXTENDED BUY
    # ======================================================
    if row["FINAL_SIGNAL"] == "BUY" and not np.isnan(bos):
        if (price - bos) / bos > 0.05:
            tags.append("🛑 EXTENDED")

    # ======================================================
    # 8 — EARLY BOS (micro-breakout)
    # ======================================================
    if price > recent["High"].iloc[:-1].max():
        tags.append("📈 EARLY_BOS")

    # ======================================================
    # 10 — MACD THRUST (strong momentum expansion)
    # ======================================================
    if "MACDH" in g.columns:
        mac = g["MACDH"].tail(4).values
        if mac[-1] > 0 and mac[-1] > mac[-2] > mac[-3]:
            tags.append("💥 MACD_THRUST")

    # ======================================================
    # 11 — ENERGY BUILDUP (tight coil + rising momentum)
    # ======================================================
    rng_pct = (recent["High"].max() - recent["Low"].min()) / max(price, 1e-9)
    rsi_slope = g["RSI"].tail(5).diff().mean()

    if rng_pct < max(0.02, 0.6 * atr_pct) and rsi_slope > 0.15:
        tags.append("🔋 ENERGY_BUILDUP")

    # ======================================================
    # 12 — REVERSAL CONFIRM (stable version)
    # ======================================================
    if "MACDH" in g.columns and "RSI" in g.columns:
        mac = g["MACDH"].tail(3).values
        if mac[-1] > 0 and mac[-2] > 0 and mac[-3] < 0 and g["RSI"].iloc[-1] > 50:
            tags.append("🔄 REVERSAL_CONFIRM")

    return " | ".join(tags) if tags else ""


def generate_trading_summary(row):
    """
    Produces a human-readable trading summary for any ticker
    with at least one insight tag.
    """

    name = row["Ticker"]
    insights = row["INSIGHT_TAGS"]
    next_action_text = row["NEXT_ACTION"]
    final_signal = row["FINAL_SIGNAL"]

    # -----------------------
    # INTERPRETATION LOGIC
    # -----------------------
    interp_parts = []

    if "PRIME" in insights:
        interp_parts.append("extremely clean structure, very close to BUY")

    if "BOS_IMMINENT" in insights:
        interp_parts.append("price sitting just beneath breakout level")

    if "SQUEEZE" in insights:
        interp_parts.append("volatility compression suggests fast breakout behavior")

    if "STRUCTURE_STRONG" in insights:
        interp_parts.append("trend geometry is strong with smooth recovery")

    if "PERFECT_ENTRY" in insights:
        interp_parts.append("high-quality HL and clean retracement in place")

    if "MACD_THRUST" in insights:
        interp_parts.append("momentum expanding into BOS")

    if "EARLY_BOS" in insights:
        interp_parts.append("micro-break of minor highs signals early strength")

    interpretation = (
        " | ".join(interp_parts)
        if interp_parts else
        "setup is developing but not yet actionable"
    )

    # -----------------------
    # PRIMARY ENTRY
    # -----------------------
    if final_signal == "BUY":
        primary_entry = f"Enter on retest of BOS level ({row['LastLocalHigh']:.2f})."
    else:
        primary_entry = f"Trigger if price breaks above {row['LastLocalHigh']:.2f}."

    # -----------------------
    # ALTERNATE ENTRY
    # -----------------------
    if final_signal == "BUY":
        alternate_entry = "If no retest occurs, you may enter on continuation strength."
    else:
        alternate_entry = "If breakout fakes out, buy retest of coil or HL region."

    # -----------------------
    # NO-TRADE CONDITIONS
    # -----------------------
    no_trade = []

    if "SQUEEZE" in insights:
        no_trade.append("avoid if the squeeze breaks downward with momentum")

    if "PERFECT_ENTRY" not in insights and final_signal != "BUY":
        no_trade.append("avoid if HL breaks or structure becomes noisy")

    if "MACD_THRUST" in insights:
        no_trade.append("avoid if MACD flips red before breakout")

    if not no_trade:
        no_trade.append("avoid if the broader market is risk-off")

    no_trade_text = " | ".join(no_trade)

    summary = f"""
Name: {name}
Insight Tags: {insights}
Next Action: {next_action_text}

Interpretation:
{interpretation}

Your Trading Plan:
Primary Entry:
- {primary_entry}

Alternate Trade:
- {alternate_entry}

No-Trade Conditions:
- {no_trade_text}
""".strip()

    return summary


# =========================================================
# WRAPPER: run the whole engine and return dataframes
# =========================================================
def run_engine():
    """
    Runs your full pipeline and returns:
    - df_all: full OHLC dataframe
    - combined: final ranked dashboard dataframe
    - insight_df: subset of combined with non-empty INSIGHT_TAGS
    - hourly_entries_df: hourly entry candidates built from combined tickers
    - hourly_rejects_df: hourly rejects diagnostics
    - hourly_df: full hourly OHLC dataframe used for List B charting
    """
    df = load_all_market_data()
    watch = build_watchlist(df, lookback_days=LOOKBACK_DAYS)

    if watch.empty:
        combined = pd.DataFrame()
        insight_df = pd.DataFrame()
        return df, combined, insight_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Enhance watchlist
    watch["Retracement %"] = watch["Retracement"] * 100

    # Entry at 61.8% retracement (not above current price)
    watch["Entry Price"] = watch.apply(
        lambda row: min(
            row["Latest Price"],
            row["Swing High Price"]
            - 0.618 * (row["Swing High Price"] - row["Swing Low Price"]),
        ),
        axis=1,
    )

    # Stop below swing low
    watch["Stop Price"] = watch["Swing Low Price"] * 0.995

    # Target: swing high or 2:1 R:R
    watch["Target Price"] = watch.apply(
        lambda row: min(
            row["Swing High Price"],
            row["Entry Price"] + 2 * (row["Entry Price"] - row["Stop Price"]),
        ),
        axis=1,
    )

    # R:R & distances (needed before sorting)
    watch["R:R"] = (watch["Target Price"] - watch["Entry Price"]) / (
        watch["Entry Price"] - watch["Stop Price"]
    )

    watch["% to Entry"] = (
        (watch["Latest Price"] - watch["Entry Price"]) / watch["Entry Price"]
    ) * 100

    watch["% to Stop"] = (
        (watch["Latest Price"] - watch["Stop Price"]) / watch["Latest Price"]
    ) * 100

    watch["% to Target"] = (
        (watch["Target Price"] - watch["Latest Price"]) / watch["Latest Price"]
    ) * 100

    # ---------- SCORING ----------
    watch["RetraceScore"] = watch["Retracement %"].apply(fib_retrace_score)
    watch["StopDistanceScore"] = (
        (watch["Latest Price"] - watch["Stop Price"])
        / (watch["Swing High Price"] - watch["Stop Price"])
    ).clip(0, 1)
    watch["SwingRangeScore"] = watch["Swing Range"] / watch["Swing Range"].max()

    watch["ActionableScore"] = (
        0.5 * watch["RetraceScore"]
        + 0.1 * watch["StopDistanceScore"]
        + 0.4 * watch["SwingRangeScore"]
    )

    # Prime setups only
    watch["Prime Setup"] = watch["Retracement %"].between(50, 78.6)
    watch = watch[watch["Prime Setup"]]

    if watch.empty:
        combined = pd.DataFrame()
        insight_df = pd.DataFrame()
        return df, combined, insight_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # CONFIRMATION
    confirm = confirmation_engine(df, watch)

    # Add Shape classification ranking
    confirm["ShapePriority"] = confirm["Shape"].apply(shape_priority)

    # Scores & NEXT_ACTION
    confirm["BUY_QUALITY"] = confirm.apply(compute_buy_quality, axis=1)
    confirm["READINESS_SCORE"] = confirm.apply(compute_watch_readiness, axis=1)
    confirm["NEXT_ACTION"] = confirm.apply(next_action, axis=1)

    # Entry timing & pressure metrics
    confirm["PERFECT_ENTRY"] = confirm.apply(compute_perfect_entry, axis=1)
    confirm["ENTRY_BIAS"] = confirm.apply(compute_entry_bias, axis=1)
    confirm["BREAKOUT_PRESSURE"] = confirm.apply(compute_breakout_pressure, axis=1)

    # Sorting logic (same as your script)
    confirm["SignalPriority"] = confirm["Shape"].apply(shape_priority)
    confirm["WatchPriority"] = confirm.apply(
        lambda r: r["READINESS_SCORE"]
        if r["FINAL_SIGNAL"] == "WATCH"
        else -1,
        axis=1,
    )

    confirm["BOS_Distance"] = confirm.apply(
        lambda r: 0
        if r["BOS"]
        else (r["LastLocalHigh"] - r["Latest Price"])
        if not np.isnan(r["LastLocalHigh"])
        else 9999,
        axis=1,
    )

    confirm = confirm.sort_values(
        by=[
            "SignalPriority",
            "WatchPriority",
            "ShapePriority",
            "BOS_Distance",
            "Momentum OK",
            "Bullish Candle",
            "HL Formed",
            "Retr Held",
        ],
        ascending=[True, False, True, True, False, False, False, False]
    ).reset_index(drop=True)

    # Insight tags
    confirm["INSIGHT_TAGS"] = confirm.apply(generate_insight_tags, axis=1)

    combined = confirm.copy()
    combined = combined.sort_values(
        by=["READINESS_SCORE", "PERFECT_ENTRY", "BREAKOUT_PRESSURE"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    insight_df = combined[combined["INSIGHT_TAGS"] != ""].copy()

    try:
        hourly_df = load_all_market_data_hourly()
        hourly_entries_df, hourly_rejects_df = build_hourly_entries(combined, hourly_df)
    except Exception as exc:
        import traceback

        hourly_df = pd.DataFrame()
        hourly_entries_df = pd.DataFrame()
        hourly_rejects_df = pd.DataFrame(
            [
                {
                    "Ticker": "__ERROR__",
                    "RejectReason": "hourly_pipeline_exception",
                    "Error": str(exc),
                    "Traceback": traceback.format_exc(),
                }
            ]
        )

    return df, combined, insight_df, hourly_entries_df, hourly_rejects_df, hourly_df
