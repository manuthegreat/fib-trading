import pandas as pd
import numpy as np
import yfinance as yf  # still unused but kept in case you extend later
from datetime import datetime, timedelta


# Optional: console display settings (used only if you print from here)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 2000)
pd.set_option("display.max_columns", None)

# --- Parameters ---
LOOKBACK_DAYS = 300
PRE_HIGH_LOOKBACK = 365
BOUNCE_TOL = 0.01



# =========================================================
# 2. Swing Logic Using HIGH/LOW (Fibonacci)
# =========================================================
def find_swing_as_of_quick(group, current_date, lookback_days=LOOKBACK_DAYS):
    """
    Identifies the most recent meaningful swing high/low pair for Fibonacci analysis.

    SWING HIGH: the highest pivot high within LOOKBACK_DAYS of current_date.

    SWING LOW: the minimum Low price BETWEEN the most recent prior pivot high
               and the current swing high. This ensures the swing low belongs
               to the same impulse wave as the swing high, not an earlier wave.

    Example (7625.HK):
      Prior pivot high: Feb 2025 spike (~11.5)
      Current swing high: Jun 2025 (10.68)
      Correct swing low: trough between Feb–Jun 2025 (~7.85)  ← this fix
      Old (wrong) result: Aug 2024 absolute low (6.00) — different wave entirely

    Fallback: if no prior pivot high exists before the current swing high,
              falls back to min(Low) in the PRE_HIGH_LOOKBACK window (original behavior).
    """

    window = group[
        (group["Date"] <= current_date)
        & (group["Date"] >= (current_date - pd.Timedelta(days=lookback_days)))
    ]

    if len(window) < 10:
        return None

    highs = window["High"].values
    lows  = window["Low"].values
    dates = window["Date"].values

    look = 5
    pivot_indices = []

    # Find all local pivot highs using HIGH prices
    for i in range(look, len(highs) - look):
        if highs[i] == max(highs[i - look : i + look + 1]):
            pivot_indices.append(i)

    if not pivot_indices:
        return None

    # Current swing high = highest of all pivots
    best_rel_idx    = max(pivot_indices, key=lambda idx: highs[idx])
    swing_high_price = float(highs[best_rel_idx])
    swing_high_date  = pd.to_datetime(dates[best_rel_idx])

    # ------------------------------------------------------------------
    # SWING LOW: anchor the low to the current wave, not an earlier one.
    #
    # Algorithm:
    #   1. Find the candidate low = min Low in PRE_HIGH_LOOKBACK window
    #      before the swing high (same as original behaviour).
    #   2. Check if there is a pivot high BETWEEN the candidate low date
    #      and the current swing high date that is HIGHER than the
    #      candidate low (i.e. a real intervening higher high).
    #   3. Only if such an intervening high exists → the candidate low
    #      belongs to an earlier wave. Re-anchor: swing low = min Low
    #      AFTER that intervening high (and before the current swing high).
    #   4. If no intervening higher high → keep the candidate low as-is.
    #      (original behaviour preserved exactly)
    #
    # Example (7625.HK):
    #   Candidate low : 6.00  (Aug 2024)
    #   Intervening HH: ~11.5 (Feb 2025 spike)  ← exists → re-anchor
    #   Corrected low : ~7.85 (trough after Feb 2025 spike) ✓
    #
    #   Counter-example (stock making first-ever high, no prior HH):
    #   Candidate low : absolute trough (correct as-is, no re-anchor)
    # ------------------------------------------------------------------

    pre_high_start = swing_high_date - pd.Timedelta(days=PRE_HIGH_LOOKBACK)
    pre_high_segment = group[
        (group["Date"] >= pre_high_start) &
        (group["Date"] <= swing_high_date)
    ]

    if pre_high_segment.empty:
        return None

    # Step 1: candidate low in the full pre-high window
    candidate_low_idx   = pre_high_segment["Low"].idxmin()
    candidate_low_date  = pd.to_datetime(group.loc[candidate_low_idx, "Date"])

    # Step 2: look for an intervening higher high between candidate low
    # and current swing high
    between_highs = group[
        (group["Date"] > candidate_low_date) &
        (group["Date"] < swing_high_date)
    ]

    intervening_hh_date = None
    if not between_highs.empty:
        # Find pivot highs in this segment (5-bar pivot)
        bh_highs = between_highs["High"].values
        bh_dates = between_highs["Date"].values
        lk = min(5, (len(bh_highs) - 1) // 2)  # shrink look if segment is short

        for i in range(lk, len(bh_highs) - lk):
            if bh_highs[i] == max(bh_highs[max(0, i - lk): i + lk + 1]):
                # Only re-anchor if this intervening pivot is HIGHER than
                # the current swing high. If it's lower, it is just a
                # correction within the same impulse wave and the candidate
                # low (the absolute trough) is already correct.
                #
                # 7625.HK: intervening 11.5 > current 10.68 -> re-anchor
                # Baidu:   intervening 138  < current 161.2  -> keep 105
                # Alibaba: intervening 125  < current 186.2  -> keep 101
                # AMD:     intervening 185  < current 267    -> keep 148
                # DOW:     intervening 26   < current 34.77  -> keep 20
                if bh_highs[i] > swing_high_price:
                    intervening_hh_date = pd.to_datetime(bh_dates[i])
                    # Use most recent qualifying pivot, keep iterating

    # Step 3: re-anchor if an intervening HH was found
    if intervening_hh_date is not None:
        after_hh = group[
            (group["Date"] > intervening_hh_date) &
            (group["Date"] <= swing_high_date)
        ]
        between_segment = after_hh if not after_hh.empty else pre_high_segment
    else:
        # Step 4: no intervening HH — use original candidate (no change)
        between_segment = pre_high_segment

    if between_segment.empty:
        return None

    low_idx         = between_segment["Low"].idxmin()
    swing_low_price = float(group.loc[low_idx, "Low"])
    swing_low_date  = pd.to_datetime(group.loc[low_idx, "Date"])

    if swing_low_price >= swing_high_price:
        return None

    swing_range = swing_high_price - swing_low_price

    return {
        "Swing Low Date":          swing_low_date,
        "Swing Low Price":         swing_low_price,
        "Swing High Date":         swing_high_date,
        "Swing High Price":        swing_high_price,
        "Retrace 50":              swing_high_price - 0.50  * swing_range,
        "Retrace 61":              swing_high_price - 0.618 * swing_range,
        "Stop Consider (78.6%)":   swing_high_price - 0.786 * swing_range,
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
    Build hourly entry candidates from DAILY-qualified tickers.

    LOGIC (inverse of daily):
      Daily engine:  High first → find Low below → price retracing DOWN toward low
      Hourly engine: Low is GIVEN (daily retracement low) → find swing High above it
                     → price pulling back DOWN from that high → enter the pullback

    Step-by-step:
      1. Anchor low  = daily retracement low (already found, passed in via combined)
      2. Swing high  = most recent confirmed pivot high on hourly chart AFTER anchor low
                       (pivot = local max with at least min_bars_since_high bars after it)
      3. Fib levels  = measured bottom-up: from anchor_low → swing_high
                       entry_382 = swing_high - 0.382 * range
                       entry_50  = swing_high - 0.500 * range
                       entry_618 = swing_high - 0.618 * range  ← primary entry
      4. Stop        = anchor_low (daily low)
      5. Target      = swing_high
      6. Valid setup = current price has pulled back from high and is between
                       entry_382 and stop (i.e. inside the fib zone)

    Returns:
      hourly_entries_df  — actionable candidates sorted by proximity to entry
      hourly_rejects_df  — diagnostic reject log
    """
    if combined is None or combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    if hourly_df is None or hourly_df.empty:
        rejects = combined[["Ticker"]].copy()
        rejects["RejectReason"] = "missing_hourly_data"
        return pd.DataFrame(), rejects

    required_combined = {"Ticker", "DailyRetrLowDate", "DailyRetrLowPrice"}
    required_hourly   = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}

    if required_combined.difference(combined.columns):
        raise RuntimeError(
            "combined missing required daily retracement columns for hourly scan"
        )
    if required_hourly.difference(hourly_df.columns):
        raise RuntimeError("hourly_df missing required OHLC columns")

    entries = []
    rejects = []

    hourly = hourly_df.copy()
    hourly["DateTime"] = pd.to_datetime(hourly["DateTime"], errors="coerce")
    hourly = hourly.dropna(subset=["DateTime"]).copy()

    if getattr(hourly["DateTime"].dt, "tz", None) is not None:
        hourly["DateTime"] = (
            hourly["DateTime"].dt.tz_convert("UTC").dt.tz_localize(None)
        )

    hourly = hourly.sort_values(["Ticker", "DateTime"])

    for _, row in combined.iterrows():
        ticker         = row["Ticker"]
        anchor_low_px  = float(row["DailyRetrLowPrice"])
        anchor_low_dt  = pd.to_datetime(row["DailyRetrLowDate"], errors="coerce")

        if pd.isna(anchor_low_dt):
            rejects.append({"Ticker": ticker, "RejectReason": "invalid_daily_low_datetime"})
            continue

        # Normalise timezone
        if anchor_low_dt.tzinfo is not None:
            anchor_low_dt = anchor_low_dt.tz_convert("UTC").tz_localize(None)

        g = hourly[hourly["Ticker"] == ticker].copy()
        if g.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "missing_hourly_data"})
            continue

        # All hourly bars AFTER the daily anchor low
        after_low = g[g["DateTime"] > anchor_low_dt].copy().reset_index(drop=True)

        if after_low.empty:
            rejects.append({"Ticker": ticker, "RejectReason": "no_hourly_after_daily_low"})
            continue

        if len(after_low) > max_bars_after_low:
            after_low = after_low.tail(max_bars_after_low).copy().reset_index(drop=True)

        if len(after_low) < 10:
            rejects.append({"Ticker": ticker, "RejectReason": "insufficient_hourly_bars"})
            continue

        # ----------------------------------------------------------------
        # STEP 2: find the most recent confirmed swing HIGH after anchor low
        #
        # "Confirmed" means at least min_bars_since_high bars have formed
        # after the candidate high (so we know price has turned down from it).
        # We scan backwards from the end of the eligible segment so we always
        # pick the most recent meaningful pivot.
        # ----------------------------------------------------------------
        # Eligible = all bars except the last min_bars_since_high
        # (those are the "confirmation tail" — the pullback bars)
        n = len(after_low)
        if n <= min_bars_since_high + 2:
            rejects.append({"Ticker": ticker, "RejectReason": "insufficient_hourly_bars"})
            continue

        eligible = after_low.iloc[: n - min_bars_since_high]
        h_vals   = eligible["High"].values
        h_dts    = eligible["DateTime"].values

        # 5-bar pivot: h[i] is highest in a ±5 bar window
        lk = min(5, (len(h_vals) - 1) // 2)
        if lk < 1:
            rejects.append({"Ticker": ticker, "RejectReason": "insufficient_hourly_bars"})
            continue

        swing_high      = None
        swing_high_time = None

        # Iterate backwards to find the most recent pivot high
        for i in range(len(h_vals) - lk - 1, lk - 1, -1):
            window = h_vals[max(0, i - lk): i + lk + 1]
            if h_vals[i] == max(window) and h_vals[i] > anchor_low_px:
                swing_high      = float(h_vals[i])
                swing_high_time = pd.to_datetime(h_dts[i])
                break

        if swing_high is None:
            # No confirmed pivot — use absolute max of eligible segment as fallback
            best = eligible["High"].idxmax()
            swing_high      = float(eligible.loc[best, "High"])
            swing_high_time = pd.to_datetime(eligible.loc[best, "DateTime"])

        if swing_high <= anchor_low_px:
            rejects.append({"Ticker": ticker, "RejectReason": "swing_high_below_anchor_low"})
            continue

        # ----------------------------------------------------------------
        # STEP 3: Fibonacci levels (bottom-up: anchor_low → swing_high)
        # ----------------------------------------------------------------
        fib_range = swing_high - anchor_low_px
        entry_382 = swing_high - 0.382 * fib_range
        entry_50  = swing_high - 0.500 * fib_range
        entry_618 = swing_high - 0.618 * fib_range
        stop      = anchor_low_px          # stop = daily anchor low
        take_profit = swing_high           # target = back to swing high

        if not (stop < entry_618 < take_profit):
            rejects.append({"Ticker": ticker, "RejectReason": "invalid_trade_levels"})
            continue

        # ----------------------------------------------------------------
        # STEP 4: Check current price is pulling back from the swing high
        # ----------------------------------------------------------------
        last_bar   = after_low.iloc[-1]
        last_close = float(last_bar["Close"])
        last_low   = float(last_bar["Low"])
        last_high  = float(last_bar["High"])

        bars_since_high = int((after_low["DateTime"] > swing_high_time).sum())
        if bars_since_high < min_bars_since_high:
            rejects.append({"Ticker": ticker, "RejectReason": "high_too_recent"})
            continue

        retrace_from_high_pct = (swing_high - last_close) / swing_high

        # Must be pulling back (below swing high) and within fib zone
        # i.e. between entry_382 (shallow) and stop (deep)
        in_fib_zone = (last_close <= entry_382) and (last_close >= stop)
        pulling_back = last_close < swing_high

        if not (pulling_back and in_fib_zone):
            rejects.append({
                "Ticker": ticker,
                "RejectReason": (
                    "no_pullback" if not pulling_back
                    else f"outside_fib_zone (close={last_close:.2f}, "
                         f"entry_382={entry_382:.2f}, stop={stop:.2f})"
                ),
            })
            continue

        # ----------------------------------------------------------------
        # STEP 5: Entry hit detection and distance to primary entry (61.8%)
        # ----------------------------------------------------------------
        distance_to_entry_618_pct = (last_close - entry_618) / entry_618
        entry_618_hit = (last_low <= entry_618) and (entry_618 <= last_high)

        entries.append({
            "Ticker":                    ticker,
            "DailyRetrLowDate":          anchor_low_dt,
            "DailyRetrLowPrice":         anchor_low_px,
            "local_high_time":           swing_high_time,
            "local_high":                swing_high,
            "entry_382":                 entry_382,
            "entry_50":                  entry_50,
            "entry_618":                 entry_618,
            "stop":                      stop,
            "take_profit":               take_profit,
            "last_close":                last_close,
            "last_low":                  last_low,
            "last_high":                 last_high,
            "retrace_from_high_pct":     retrace_from_high_pct,
            "bars_since_high":           bars_since_high,
            "distance_to_entry_618_pct": distance_to_entry_618_pct,
            "entry_618_hit":             bool(entry_618_hit),
        })

    entries_df = pd.DataFrame(entries)
    rejects_df = pd.DataFrame(rejects)

    if not entries_df.empty:
        entries_df["abs_dist"] = entries_df["distance_to_entry_618_pct"].abs()
        entries_df = (
            entries_df
            .sort_values(
                by=["entry_618_hit", "abs_dist"],
                ascending=[False, True],
            )
            .drop(columns=["abs_dist"])
            .reset_index(drop=True)
        )

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
    bos = row["LastLocalHigh"]
    hl = row["HL Price"]

    if row["FINAL_SIGNAL"] == "BUY":
        if not np.isnan(bos):
            return f"Limit: {bos:.2f}"
        else:
            return "Limit: N/A"

    if row["FINAL_SIGNAL"] == "WATCH":
        trigger = f"Trigger > {bos:.2f}" if not np.isnan(bos) else "Trigger > N/A"
        inval = f"Invalidate < {hl:.2f}" if row["HL Formed"] else "Invalidate < N/A"
        return f"{trigger} | {inval}"

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
# 10. PERFECT ENTRY, ENTRY BIAS & BREAKOUT PRESSURE
# =========================================================
def compute_perfect_entry(row):
    price = row["Latest Price"]
    low = row["SwingLow"]
    high = row["SwingHigh"]
    hl = row["HL Price"]

    if np.isnan(low) or np.isnan(high):
        return None

    swing = high - low
    if swing <= 0:
        return None

    retr = (high - hl) / swing
    ideal = 0.56
    dist = abs(retr - ideal)

    if retr < 0.38 or retr > 0.78:
        base_score = 60
    else:
        if dist < 0.05:
            retr_score = 40
        elif dist < 0.10:
            retr_score = 30
        else:
            retr_score = 15

        hl_score = 30 if row["HL Formed"] else 0

        pr = row.get("ShapePriority", 5)
        if pr == 1:
            shape_score = 20
        elif pr == 2:
            shape_score = 15
        elif pr == 3:
            shape_score = 10
        else:
            shape_score = 0

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

    if (
        row["FINAL_SIGNAL"] == "BUY"
        and row["Retr Held"]
        and row["Momentum OK"]
    ):
        score += 15

    if row["FINAL_SIGNAL"] == "INVALID":
        score = min(score, 55)

    return round(min(score, 100), 2)


def compute_entry_bias(row):
    if row["FINAL_SIGNAL"] != "BUY":
        return None

    price = row["Latest Price"]
    bos = row["LastLocalHigh"]

    if np.isnan(bos) or bos == 0:
        return "NEUTRAL"

    dist_pct = (price - bos) / bos
    shape = row["Shape"]

    if dist_pct > 0.035 and shape in ("strong recovery", "V-reversal"):
        return "BREAKOUT CONTINUATION LIKELY"

    if dist_pct < 0.015 and shape in ("consolidation under BOS", "rounded recovery"):
        return "RETEST LIKELY"

    if shape == "normal recovery":
        return "BASE FORMATION"

    return "NEUTRAL"


def compute_breakout_pressure(row):
    price = row["Latest Price"]
    bos = row["LastLocalHigh"]
    hl = row["HL Price"]

    if np.isnan(bos):
        bos_prox = 0
    else:
        raw = 1 - ((bos - price) / max(price, 1e-9))
        bos_prox = np.clip(raw, 0, 1)

    if np.isnan(hl):
        hl_rec = 0
    else:
        hl_rec = np.clip((price - hl) / (0.35 * price), 0, 1)

    mom = 1 if row["Momentum OK"] else 0

    g = row.get("__gdata__", None)
    if g is not None and len(g) >= 20:
        atr5 = g["High"].tail(5).max() - g["Low"].tail(5).min()
        atr20 = g["High"].tail(20).max() - g["Low"].tail(20).min()
        comp = np.clip(1 - atr5 / max(atr20, 1e-9), 0, 1)
    else:
        comp = 0

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
        score = min(score, 40.0)

    return round(np.clip(score, 0, 100), 2)


def generate_insight_tags(row):
    tags = []

    price = row["Latest Price"]
    bos = row["LastLocalHigh"]
    hl = row["HL Price"]
    g = row.get("__gdata__", None)

    if g is None or len(g) < 5:
        return ""

    recent = g.tail(20)
    atr = (recent["High"] - recent["Low"]).rolling(5).mean().iloc[-1]
    atr_pct = atr / max(price, 1e-9)

    if (
        row["FINAL_SIGNAL"] == "WATCH"
        and row["BREAKOUT_PRESSURE"] >= 75
        and row["ShapePriority"] <= 2
    ):
        tags.append("🔥 PRIME")

    if not np.isnan(bos):
        dist = (bos - price) / bos
        if 0 <= dist < max(0.0075, 0.75 * atr_pct):
            tags.append("⚡ BOS_IMMINENT")

    if len(g) >= 20:
        atr5 = g["High"].tail(5).max() - g["Low"].tail(5).min()
        atr20 = g["High"].tail(20).max() - g["Low"].tail(20).min()
        comp = np.clip(1 - atr5 / max(atr20, 1e-9), 0, 1)
        mac_flat = abs(g["MACDH"].tail(5).mean()) < 0.15 * abs(g["MACDH"].tail(20).std() + 1e-9)
        if comp > 0.65 and mac_flat:
            tags.append("📉 SQUEEZE")

    pe = row.get("PERFECT_ENTRY", None)
    if pe is not None:
        if row["FINAL_SIGNAL"] == "BUY" and pe >= 80:
            tags.append("🎯 PERFECT_ENTRY")
        elif (
            row["FINAL_SIGNAL"] == "WATCH"
            and pe >= 90
            and row["BREAKOUT_PRESSURE"] >= 75
        ):
            tags.append("🎯 PERFECT_ENTRY")

    if (
        row["FINAL_SIGNAL"] != "BUY"
        and row["ShapePriority"] <= 3
        and row["BREAKOUT_PRESSURE"] > 55
    ):
        if g["MACDH"].iloc[-1] > 0 or price > recent["Close"].mean():
            tags.append("🌀 STRUCTURE_STRONG")

    if row["FINAL_SIGNAL"] == "BUY" and not np.isnan(bos):
        if (price - bos) / bos > 0.05:
            tags.append("🛑 EXTENDED")

    if price > recent["High"].iloc[:-1].max():
        tags.append("📈 EARLY_BOS")

    if "MACDH" in g.columns:
        mac = g["MACDH"].tail(4).values
        if mac[-1] > 0 and mac[-1] > mac[-2] > mac[-3]:
            tags.append("💥 MACD_THRUST")

    rng_pct = (recent["High"].max() - recent["Low"].min()) / max(price, 1e-9)
    rsi_slope = g["RSI"].tail(5).diff().mean()
    if rng_pct < max(0.02, 0.6 * atr_pct) and rsi_slope > 0.15:
        tags.append("🔋 ENERGY_BUILDUP")

    if "MACDH" in g.columns and "RSI" in g.columns:
        mac = g["MACDH"].tail(3).values
        if mac[-1] > 0 and mac[-2] > 0 and mac[-3] < 0 and g["RSI"].iloc[-1] > 50:
            tags.append("🔄 REVERSAL_CONFIRM")

    return " | ".join(tags) if tags else ""


def generate_trading_summary(row):
    name = row["Ticker"]
    insights = row["INSIGHT_TAGS"]
    next_action_text = row["NEXT_ACTION"]
    final_signal = row["FINAL_SIGNAL"]

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
        " | ".join(interp_parts) if interp_parts
        else "setup is developing but not yet actionable"
    )

    if final_signal == "BUY":
        primary_entry = f"Enter on retest of BOS level ({row['LastLocalHigh']:.2f})."
    else:
        primary_entry = f"Trigger if price breaks above {row['LastLocalHigh']:.2f}."

    if final_signal == "BUY":
        alternate_entry = "If no retest occurs, you may enter on continuation strength."
    else:
        alternate_entry = "If breakout fakes out, buy retest of coil or HL region."

    no_trade = []
    if "SQUEEZE" in insights:
        no_trade.append("avoid if the squeeze breaks downward with momentum")
    if "PERFECT_ENTRY" not in insights and final_signal != "BUY":
        no_trade.append("avoid if HL breaks or structure becomes noisy")
    if "MACD_THRUST" in insights:
        no_trade.append("avoid if MACD flips red before breakout")
    if not no_trade:
        no_trade.append("avoid if the broader market is risk-off")

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
- {" | ".join(no_trade)}
""".strip()

    return summary


# =========================================================
# WRAPPER: run the whole engine and return dataframes
# =========================================================
def run_engine():
    df = load_all_market_data()
    watch = build_watchlist(df, lookback_days=LOOKBACK_DAYS)

    if watch.empty:
        combined = pd.DataFrame()
        insight_df = pd.DataFrame()
        return df, combined, insight_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    watch["Retracement %"] = watch["Retracement"] * 100

    watch["Entry Price"] = watch.apply(
        lambda row: min(
            row["Latest Price"],
            row["Swing High Price"]
            - 0.618 * (row["Swing High Price"] - row["Swing Low Price"]),
        ),
        axis=1,
    )

    watch["Stop Price"] = watch["Swing Low Price"] * 0.995

    watch["Target Price"] = watch.apply(
        lambda row: min(
            row["Swing High Price"],
            row["Entry Price"] + 2 * (row["Entry Price"] - row["Stop Price"]),
        ),
        axis=1,
    )

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

    watch["Prime Setup"] = watch["Retracement %"].between(50, 78.6)
    watch = watch[watch["Prime Setup"]]

    if watch.empty:
        combined = pd.DataFrame()
        insight_df = pd.DataFrame()
        return df, combined, insight_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    confirm = confirmation_engine(df, watch)
    confirm["ShapePriority"] = confirm["Shape"].apply(shape_priority)
    confirm["BUY_QUALITY"] = confirm.apply(compute_buy_quality, axis=1)
    confirm["NEXT_ACTION"] = confirm.apply(next_action, axis=1)
    confirm["PERFECT_ENTRY"] = confirm.apply(compute_perfect_entry, axis=1)
    confirm["ENTRY_BIAS"] = confirm.apply(compute_entry_bias, axis=1)
    confirm["BREAKOUT_PRESSURE"] = confirm.apply(compute_breakout_pressure, axis=1)

    confirm["SignalPriority"] = confirm["Shape"].apply(shape_priority)
    confirm["WatchPriority"] = confirm.apply(
        lambda r: r.get("BREAKOUT_PRESSURE", 0)
        if r["FINAL_SIGNAL"] == "WATCH" else -1,
        axis=1,
    )
    confirm["BOS_Distance"] = confirm.apply(
        lambda r: 0 if r["BOS"]
        else (r["LastLocalHigh"] - r["Latest Price"]) if not np.isnan(r["LastLocalHigh"])
        else 9999,
        axis=1,
    )

    confirm = confirm.sort_values(
        by=["SignalPriority", "WatchPriority", "ShapePriority",
            "BOS_Distance", "Momentum OK", "Bullish Candle", "HL Formed", "Retr Held"],
        ascending=[True, False, True, True, False, False, False, False]
    ).reset_index(drop=True)

    confirm["INSIGHT_TAGS"] = confirm.apply(generate_insight_tags, axis=1)

    combined = confirm.copy()
    combined = combined.sort_values(
        by=["BREAKOUT_PRESSURE", "PERFECT_ENTRY", "BUY_QUALITY"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    insight_df = combined[combined["INSIGHT_TAGS"] != ""].copy()

    try:
        from updater import load_hourly_prices_for_tickers
        hourly_df = load_hourly_prices_for_tickers(combined["Ticker"].unique().tolist())
        hourly_entries_df, hourly_rejects_df = build_hourly_entries(combined, hourly_df)
    except Exception as exc:
        import traceback
        hourly_df = pd.DataFrame()
        hourly_entries_df = pd.DataFrame()
        hourly_rejects_df = pd.DataFrame([{
            "Ticker": "__ERROR__",
            "RejectReason": "hourly_pipeline_exception",
            "Error": str(exc),
            "Traceback": traceback.format_exc(),
        }])

    return df, combined, insight_df, hourly_entries_df, hourly_rejects_df, hourly_df
