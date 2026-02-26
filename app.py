import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from engine import run_engine, generate_trading_summary


# ==========================================================
# FOCUS SCORE
# Simplified: entry proximity + R:R only.
# BUY/WATCH/INVALID and Breakout Pressure removed — buy
# decisions are made purely from the hourly setup.
# ==========================================================

def _rr(r):
    entry  = float(r.get("entry_618",   np.nan))
    stop   = float(r.get("stop",        np.nan))
    tp     = float(r.get("take_profit", np.nan))
    if any(np.isnan(x) for x in [entry, stop, tp]):
        return np.nan
    risk   = entry - stop
    reward = tp    - entry
    return round(reward / risk, 2) if risk > 0 and reward > 0 else np.nan


def compute_focus_score(row: pd.Series) -> float:
    score = 0.0
    dist  = float(row.get("distance_to_entry_618_pct", 0.10))
    hit   = bool(row.get("entry_618_hit", False))

    if hit:
        proximity_score = 55.0
    elif dist > 0:
        proximity_score = 55.0 * max(0.0, 1.0 - dist / 0.10)
    else:
        proximity_score = 55.0 * max(0.0, 1.0 + dist / 0.05)
    score += proximity_score

    if hit:
        score += 15.0

    rr = float(row.get("rr", np.nan))
    if not np.isnan(rr) and rr > 0:
        score += 30.0 * min(rr / 5.0, 1.0)

    return round(float(np.clip(score, 0.0, 100.0)), 1)


def _focus_label(score: float) -> str:
    if score >= 85: return "🔴 ACT NOW"
    if score >= 65: return "🟠 HIGH"
    if score >= 45: return "🟡 MEDIUM"
    return "⚪ LOW"


def rank_hourly_candidates(hourly_df: pd.DataFrame) -> pd.DataFrame:
    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame()
    ranked = hourly_df.copy()
    ranked["rr"]          = ranked.apply(_rr, axis=1)
    ranked["FOCUS_SCORE"] = ranked.apply(compute_focus_score, axis=1)
    ranked["FOCUS"]       = ranked["FOCUS_SCORE"].apply(_focus_label)
    return ranked.sort_values(
        by=["entry_618_hit", "FOCUS_SCORE"],
        ascending=[False, False],
    ).reset_index(drop=True)


# ==========================================================
# PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Fib Dashboard", layout="wide")
st.title("📈 Fib Retracement Dashboard")

st.markdown("""
Scans **S&P 500 and Hang Seng (HSI)** for names where:
- Daily swing high → retracement low is identified
- Current price is in the **50%–78.6% Fibonacci zone**
- Hourly chart shows a new swing high forming after the daily low
- Price is pulling back toward the **61.8% Fibonacci entry**
""")

st.markdown("""
<style>button[kind="header"] {display: none !important;}</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.header("Settings")
lookback_days = st.sidebar.slider("Chart lookback (days)", 60, 500, 180, 10)
st.sidebar.write("---")
st.sidebar.write("Run after market close / before open.")

# ==========================================================
# ENGINE RUN (cached)
# ==========================================================
ENGINE_VERSION = "2026-02-26-clean-ui-v1"


@st.cache_data(show_spinner="Running engine… ~30-50s on first load.")
def compute_dashboard(engine_version):
    _ = engine_version
    return run_engine()


(df_all, combined, insight_df,
 hourly_entries_df, hourly_rejects_df, hourly_df) = compute_dashboard(ENGINE_VERSION)

if combined.empty:
    st.error("No names in watchlist. Check data or parameters.")
    st.stop()

df_view = combined.copy()
if df_view.empty:
    st.warning("No tickers match current filters.")
    st.stop()


# ==========================================================
# HOURLY ENTRY CANDIDATES
# ==========================================================
st.write("---")
st.write("### ⏱️ Hourly Entry Candidates")
st.markdown("""
Ranked by **Focus Score** — how actionable the setup is right now.  
`🔴 ACT NOW` ≥ 85  ·  `🟠 HIGH` ≥ 65  ·  `🟡 MEDIUM` ≥ 45  ·  `⚪ LOW` < 45
""")

ranked_hourly = pd.DataFrame()

if hourly_entries_df is not None and not hourly_entries_df.empty:

    ranked_hourly = rank_hourly_candidates(hourly_entries_df)

    top_n    = st.slider("Show top N setups", 5, 50, 15, 5)
    show_top = ranked_hourly.head(top_n).reset_index(drop=True)

    display_cols = {
        "Ticker":                    "Ticker",
        "FOCUS":                     "Focus",
        "FOCUS_SCORE":               "Score",
        "entry_618_hit":             "At Entry?",
        "entry_618":                 "Entry 61.8%",
        "stop":                      "Stop",
        "take_profit":               "Target",
        "rr":                        "R:R",
        "distance_to_entry_618_pct": "Dist to Entry",
        "retrace_from_high_pct":     "Retrace %",
        "bars_since_high":           "Bars Since High",
    }

    avail_cols  = {k: v for k, v in display_cols.items() if k in show_top.columns}
    hourly_view = show_top[list(avail_cols.keys())].rename(columns=avail_cols).copy()

    for col_raw, col_display in [
        ("distance_to_entry_618_pct", "Dist to Entry"),
        ("retrace_from_high_pct",     "Retrace %"),
    ]:
        if col_display in hourly_view.columns:
            hourly_view[col_display] = (
                pd.to_numeric(hourly_view[col_display], errors="coerce") * 100
            ).round(2).astype(str) + "%"

    if "R:R" in hourly_view.columns:
        hourly_view["R:R"] = pd.to_numeric(hourly_view["R:R"], errors="coerce").round(2)

    if "At Entry?" in hourly_view.columns:
        hourly_view["At Entry?"] = hourly_view["At Entry?"].map(
            {True: "✅ YES", False: "—", 1: "✅ YES", 0: "—"}
        ).fillna("—")

    hourly_event = st.dataframe(
        hourly_view,
        hide_index=True,
        use_container_width=True,
        key="hourly_ranked_df",
        on_select="rerun",
        selection_mode="single-row",
    )

    hourly_selected_rows = []
    if hourly_event is not None:
        if hasattr(hourly_event, "selection") and hasattr(hourly_event.selection, "rows"):
            hourly_selected_rows = hourly_event.selection.rows
        elif isinstance(hourly_event, dict):
            hourly_selected_rows = hourly_event.get("selection", {}).get("rows", [])

    if "hourly_selected_ticker" not in st.session_state:
        st.session_state.hourly_selected_ticker = (
            show_top.iloc[0]["Ticker"] if not show_top.empty else None
        )

    if hourly_selected_rows:
        idx = hourly_selected_rows[0]
        if 0 <= idx < len(show_top):
            st.session_state.hourly_selected_ticker = show_top.iloc[idx]["Ticker"]

    hourly_ticker_selected = st.session_state.hourly_selected_ticker

    # --- Hourly chart ---
    if hourly_ticker_selected and hourly_df is not None and not hourly_df.empty:
        sel_row_df    = ranked_hourly[ranked_hourly["Ticker"] == hourly_ticker_selected]
        ticker_hourly = hourly_df[hourly_df["Ticker"] == hourly_ticker_selected].copy()

        if not sel_row_df.empty and not ticker_hourly.empty:
            hrow = sel_row_df.iloc[0]

            ticker_hourly["DateTime"] = pd.to_datetime(
                ticker_hourly["DateTime"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            ticker_hourly = ticker_hourly.dropna(subset=["DateTime"]).sort_values("DateTime")

            high_dt = pd.to_datetime(hrow.get("local_high_time"),  errors="coerce", utc=True)
            retr_dt = pd.to_datetime(hrow.get("DailyRetrLowDate"), errors="coerce", utc=True)
            if pd.notna(high_dt): high_dt = high_dt.tz_convert(None)
            if pd.notna(retr_dt): retr_dt = retr_dt.tz_convert(None)

            max_dt    = ticker_hourly["DateTime"].max()
            win_start = (high_dt - pd.Timedelta(hours=72)) if pd.notna(high_dt) else (max_dt - pd.Timedelta(hours=72))
            if pd.notna(retr_dt):
                win_start = min(retr_dt, win_start)
            win_end = max_dt

            visible = ticker_hourly[
                (ticker_hourly["DateTime"] >= win_start) &
                (ticker_hourly["DateTime"] <= win_end)
            ]
            if len(visible) < 50:
                fallback = ticker_hourly.tail(240)
                if not fallback.empty:
                    win_start = fallback["DateTime"].min()
                    win_end   = fallback["DateTime"].max()

            st.markdown(
                f"**{hourly_ticker_selected}** &nbsp;|&nbsp; "
                f"{hrow.get('FOCUS', '')} (Score: {hrow.get('FOCUS_SCORE', '—')})"
            )

            fig_h = go.Figure(data=[go.Candlestick(
                x=ticker_hourly["DateTime"],
                open=ticker_hourly["Open"], high=ticker_hourly["High"],
                low=ticker_hourly["Low"],   close=ticker_hourly["Close"],
                name=hourly_ticker_selected,
            )])

            for col, label, color in [
                ("entry_382",   "Entry 38.2%", "#1f77b4"),
                ("entry_50",    "Entry 50%",   "#9467bd"),
                ("entry_618",   "Entry 61.8%", "#ff7f0e"),
                ("stop",        "Stop",        "#d62728"),
                ("take_profit", "Target",      "#2ca02c"),
            ]:
                val = hrow.get(col)
                if val is not None and pd.notna(val):
                    fig_h.add_hline(
                        y=float(val), line_dash="dash", line_color=color,
                        annotation_text=label, annotation_position="top left",
                    )

            fig_h.update_layout(
                title=f"{hourly_ticker_selected} — Hourly",
                xaxis_title="DateTime", yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                template="plotly_white", height=480,
            )
            fig_h.update_xaxes(range=[win_start, win_end])

            vis2 = ticker_hourly[
                (ticker_hourly["DateTime"] >= win_start) &
                (ticker_hourly["DateTime"] <= win_end)
            ]
            if not vis2.empty:
                y_lo = pd.to_numeric(vis2["Low"],  errors="coerce").min()
                y_hi = pd.to_numeric(vis2["High"], errors="coerce").max()
                if pd.notna(y_lo) and pd.notna(y_hi):
                    pad = max((y_hi - y_lo) * 0.08, y_hi * 0.002)
                    fig_h.update_yaxes(range=[y_lo - pad, y_hi + pad])

            st.plotly_chart(fig_h, use_container_width=True)

else:
    st.info("No hourly entry candidates found for current run.")


# ==========================================================
# SUMMARY METRICS
# ==========================================================
st.write("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Symbols in Fib zone", len(df_view))
with col2:
    n_hourly = len(hourly_entries_df) if hourly_entries_df is not None else 0
    st.metric("Hourly entry candidates", n_hourly)
with col3:
    n_at_entry = 0
    if hourly_entries_df is not None and not hourly_entries_df.empty and "entry_618_hit" in hourly_entries_df.columns:
        n_at_entry = int(hourly_entries_df["entry_618_hit"].sum())
    st.metric("At entry zone now", n_at_entry)


# ==========================================================
# DAILY WATCHLIST TABLE
# ==========================================================
st.write("### 📋 Daily Watchlist (Fib Zone)")

ranked_table = df_view[["Ticker", "SwingHigh", "SwingLow", "Latest Price"]].copy()
swing_range  = (ranked_table["SwingHigh"] - ranked_table["SwingLow"]).replace(0, pd.NA)
ranked_table["Retracement %"] = (
    (ranked_table["SwingHigh"] - ranked_table["Latest Price"]) / swing_range * 100
).round(1)
ranked_table = ranked_table.drop(columns=["Latest Price"])

event = st.dataframe(
    ranked_table,
    hide_index=True,
    use_container_width=True,
    key="ranked_df",
    on_select="rerun",
    selection_mode="single-row",
)

selected_rows = []
if event is not None:
    if hasattr(event, "selection") and hasattr(event.selection, "rows"):
        selected_rows = event.selection.rows
    elif isinstance(event, dict):
        selected_rows = event.get("selection", {}).get("rows", [])

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

if selected_rows:
    idx = selected_rows[0]
    if 0 <= idx < len(ranked_table):
        st.session_state.selected_ticker = ranked_table.iloc[idx]["Ticker"]

ticker_selected = st.session_state.selected_ticker


# ==========================================================
# DAILY CHART — candlestick + Fib levels only, no indicators
# ==========================================================
def plot_ticker_chart(df_all, row, lookback_days=180):
    ticker  = row["Ticker"]
    df_full = df_all[df_all["Ticker"] == ticker].sort_values("Date").copy()
    if df_full.empty:
        st.write("No price data found.")
        return

    df_t  = df_full.tail(lookback_days).copy()
    dates = df_t["Date"]

    fig = go.Figure(data=[go.Candlestick(
        x=dates,
        open=df_t["Open"], high=df_t["High"],
        low=df_t["Low"],   close=df_t["Close"],
        name=ticker,
    )])

    swing_low  = row.get("SwingLow")
    swing_high = row.get("SwingHigh")
    if pd.notna(swing_low) and pd.notna(swing_high):
        swing  = swing_high - swing_low
        x0, x1 = dates.iloc[0], dates.iloc[-1]
        for label, level, color in [
            ("100% (High)", swing_high,                "#e74c3c"),
            ("78.6%",       swing_high - 0.786 * swing, "#e67e22"),
            ("61.8%",       swing_high - 0.618 * swing, "#f1c40f"),
            ("50%",         swing_high - 0.500 * swing, "#2ecc71"),
            ("38.2%",       swing_high - 0.382 * swing, "#3498db"),
            ("0% (Low)",    swing_low,                  "#9b59b6"),
        ]:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=level, y1=level,
                line=dict(color=color, width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x1, y=level, text=label, showarrow=False,
                xanchor="left", font=dict(size=10, color=color),
            )

    fig.update_layout(
        title=f"{ticker} — Daily",
        yaxis_title="Price",
        height=500,
        showlegend=False,
        margin=dict(l=0, r=80, t=40, b=20),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# HOURLY ENTRY DETAIL CARD
# ==========================================================
def render_entry_card(hourly_row=None):
    st.markdown("### 📘 Trade Levels")

    if hourly_row is None or hourly_row.empty:
        st.info("No hourly entry found for this ticker yet.")
        return

    hr       = hourly_row.iloc[0]
    entry_v  = float(hr.get("entry_618",               np.nan))
    stop_v   = float(hr.get("stop",                    np.nan))
    tp_v     = float(hr.get("take_profit",              np.nan))
    pb_v     = float(hr.get("retrace_from_high_pct",    np.nan))
    dist_v   = float(hr.get("distance_to_entry_618_pct", np.nan))
    hit_v    = bool(hr.get("entry_618_hit", False))
    rr_v     = float(hr.get("rr", np.nan))
    focus_v  = hr.get("FOCUS_SCORE", "—")
    focus_l  = hr.get("FOCUS", "")

    rr_str   = f"{rr_v:.2f}" if not np.isnan(rr_v) else "N/A"
    pb_str   = f"{pb_v*100:.2f}%" if not np.isnan(pb_v) else "N/A"
    dist_str = f"{dist_v*100:.2f}%" if not np.isnan(dist_v) else "N/A"
    at_entry = '<span style="color:lime;font-weight:bold">  ✅ AT ENTRY NOW</span>' if hit_v else ""

    html = f"""
<div style="background:#1e1e2e;padding:20px;border-radius:10px;
            border:1px solid #444;font-size:15px;color:#eee;line-height:1.8;">
  <h3 style="color:#4CC9F0;margin:0 0 12px 0;">⏱️ Hourly Entry Plan</h3>
  <b>Focus:</b> {focus_l} ({focus_v}){at_entry}<br>
  <b>Entry (61.8%):</b> {entry_v:.4f}<br>
  <b>Stop:</b> {stop_v:.4f}<br>
  <b>Target:</b> {tp_v:.4f}<br>
  <b>R:R:</b> {rr_str}<br>
  <b>Pullback from high:</b> {pb_str}<br>
  <b>Distance to entry:</b> {dist_str}<br>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# TICKER DRILLDOWN
# ==========================================================
if ticker_selected:
    sel_df = df_view[df_view["Ticker"] == ticker_selected]
    if sel_df.empty:
        st.warning("Selected ticker no longer available.")
        st.stop()
    row_sel = sel_df.iloc[0]

    st.write(f"### 📌 {ticker_selected}")

    swing_h = row_sel.get("SwingHigh", np.nan)
    swing_l = row_sel.get("SwingLow",  np.nan)
    latest  = row_sel.get("Latest Price", np.nan)
    rng     = swing_h - swing_l if pd.notna(swing_h) and pd.notna(swing_l) else np.nan
    retr_pct = (swing_h - latest) / rng * 100 if rng else np.nan

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Swing High",  f"{swing_h:.2f}" if pd.notna(swing_h) else "N/A")
    with c2: st.metric("Swing Low",   f"{swing_l:.2f}" if pd.notna(swing_l) else "N/A")
    with c3: st.metric("Retracement", f"{retr_pct:.1f}%" if pd.notna(retr_pct) else "N/A")

    plot_ticker_chart(df_all, row_sel, lookback_days=lookback_days)

    hourly_sel = None
    if not ranked_hourly.empty:
        hourly_sel = ranked_hourly[ranked_hourly["Ticker"] == ticker_selected]
        if hourly_sel.empty:
            hourly_sel = None

    render_entry_card(hourly_row=hourly_sel)

else:
    st.info("Select a row from the hourly candidates or daily watchlist to see charts and trade levels.")


# ==========================================================
# LEGEND
# ==========================================================
st.write("---")
st.subheader("📘 How to read this dashboard")
st.markdown("""
### Strategy Flow
1. **Daily** — identify swing high and the retracement low of that wave
2. **Fib filter** — only symbols where price is in the **50%–78.6% zone** (meaningful support)
3. **Daily low held** — confirmation that the retracement low is holding as support
4. **Hourly** — after the daily low, identify the new swing high that forms
5. **Entry** — wait for price to pull back to the **61.8% Fibonacci level** of the hourly swing
6. **Stop** — daily retracement low. If price breaks below it, setup is **invalid**
7. **Target** — hourly swing high

---
### Focus Score
Ranks how actionable a setup is *right now*:
- **Entry proximity (55%)** — how close price is to the 61.8% entry
- **Entry hit bonus (+15 pts)** — price is at the zone right now
- **R:R (30%)** — reward vs risk, capped at 5:1
""")
