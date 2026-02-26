import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import run_engine, generate_trading_summary


# ==========================================================
# FOCUS SCORE — ranks hourly candidates by actionability
# ==========================================================

def compute_focus_score(row: pd.Series) -> float:
    """
    Composite score (0-100) answering: "what should I look at first?"

    Component                      Weight   Rationale
    ----------------------------   ------   ------------------------------------
    Entry proximity (61.8% dist)     30%    Closest to entry = most urgent
    Entry hit bonus                 +15pt   Price IS at the zone right now
    R:R ratio (capped at 5:1)        25%    Quality of the trade
    Daily signal (BUY/WATCH)         20%    Daily timeframe confirmation
    Breakout pressure (0-100)        15%    Structural energy from daily engine
    Shape quality (1-7 priority)     10%    Clean structure preferred
    """
    score = 0.0

    # --- 1. Entry proximity (30pts) ---
    # distance_to_entry_618_pct:
    #   ~0       = price AT entry         → act now
    #   positive = price above entry      → pulling back toward it
    #   negative = price below entry      → overshot / missed
    dist = float(row.get("distance_to_entry_618_pct", 0.10))
    hit  = bool(row.get("entry_618_hit", False))

    if hit:
        proximity_score = 30.0
    elif dist > 0:
        # Above entry: linear decay — 0% away=30pts, 10%+ away=0pts
        proximity_score = 30.0 * max(0.0, 1.0 - dist / 0.10)
    else:
        # Below entry (overshot): rapid decay — 0%=30pts, -5%=0pts
        proximity_score = 30.0 * max(0.0, 1.0 + dist / 0.05)

    score += proximity_score

    # --- 2. Entry hit bonus (+15pts) ---
    if hit:
        score += 15.0

    # --- 3. R:R (25pts, capped at 5:1) ---
    rr = float(row.get("rr", np.nan))
    if not np.isnan(rr) and rr > 0:
        score += 25.0 * min(rr / 5.0, 1.0)

    # --- 4. Daily signal quality (20pts) ---
    signal = str(row.get("FINAL_SIGNAL", "WATCH"))
    score += {"BUY": 20.0, "WATCH": 12.0, "INVALID": 0.0}.get(signal, 12.0)

    # --- 5. Breakout pressure from daily engine (15pts) ---
    bp = float(row.get("BREAKOUT_PRESSURE", 50.0))
    score += 15.0 * np.clip(bp / 100.0, 0.0, 1.0)

    # --- 6. Shape quality (10pts) — lower ShapePriority = better ---
    sp = float(row.get("ShapePriority", 4.0))
    score += 10.0 * max(0.0, (7.0 - sp) / 6.0)

    return round(float(np.clip(score, 0.0, 100.0)), 1)


def _focus_label(score: float) -> str:
    """Human-readable urgency label."""
    if score >= 85:
        return "🔴 ACT NOW"
    if score >= 65:
        return "🟠 HIGH"
    if score >= 45:
        return "🟡 MEDIUM"
    return "⚪ LOW"


def rank_hourly_candidates(
    hourly_df: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily context into hourly candidates, compute FOCUS_SCORE,
    and return sorted result.
    """
    if hourly_df is None or hourly_df.empty:
        return pd.DataFrame()

    ranked = hourly_df.copy()

    # --- Join daily context columns ---
    daily_cols = ["Ticker", "FINAL_SIGNAL", "BREAKOUT_PRESSURE",
                  "PERFECT_ENTRY", "INSIGHT_TAGS", "ShapePriority", "Shape"]
    if combined is not None and not combined.empty:
        avail = [c for c in daily_cols if c in combined.columns]
        ranked = ranked.merge(
            combined[avail].drop_duplicates("Ticker"),
            on="Ticker", how="left",
        )

    # Fill defaults for any missing daily columns
    ranked["FINAL_SIGNAL"]      = ranked.get("FINAL_SIGNAL",      pd.Series(["WATCH"] * len(ranked)))
    ranked["BREAKOUT_PRESSURE"] = ranked.get("BREAKOUT_PRESSURE", pd.Series([50.0]   * len(ranked)))
    ranked["ShapePriority"]     = ranked.get("ShapePriority",     pd.Series([4.0]    * len(ranked)))

    # --- Compute R:R ---
    def _rr(r):
        entry = float(r.get("entry_618", np.nan))
        stop  = float(r.get("stop",      np.nan))
        tp    = float(r.get("take_profit", np.nan))
        if any(np.isnan(x) for x in [entry, stop, tp]):
            return np.nan
        risk   = entry - stop
        reward = tp - entry
        return round(reward / risk, 2) if risk > 0 and reward > 0 else np.nan

    ranked["rr"] = ranked.apply(_rr, axis=1)

    # --- Compute FOCUS_SCORE ---
    ranked["FOCUS_SCORE"] = ranked.apply(compute_focus_score, axis=1)
    ranked["FOCUS"]       = ranked["FOCUS_SCORE"].apply(_focus_label)

    # Sort: entry_618_hit first, then by FOCUS_SCORE descending
    ranked = ranked.sort_values(
        by=["entry_618_hit", "FOCUS_SCORE"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return ranked


# ==========================================================
# PAGE SETUP
# ==========================================================
st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("📈 Fib Retracement Dashboard")

st.markdown("""
### What this app does
Scans **S&P 500, Hang Seng Index (HSI), and EURO STOXX 50** for names that:
- recently made a swing high
- are retracing into **Fibonacci support zones**
- show signs of **bullish structure, momentum, and rebound strength**

Hourly candidates are ranked by **Focus Score** — how actionable the setup is *right now*.
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
ENGINE_VERSION = "2026-02-26-focus-score-v1"   # bump to force cache refresh


@st.cache_data(show_spinner="Running engine… this takes ~2 minutes on first load.")
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
# HOURLY ENTRY CANDIDATES — LIST B
# ==========================================================
st.write("---")
st.write("### ⏱️ Hourly Entry Candidates")

st.markdown("""
**Focus Score** ranks candidates by actionability right now:
`🔴 ACT NOW` ≥ 85 · `🟠 HIGH` ≥ 65 · `🟡 MEDIUM` ≥ 45 · `⚪ LOW` < 45

Scoring weights: Entry proximity 30% · Entry-hit bonus +15pts · R:R 25% · Daily signal 20% · Breakout pressure 15% · Shape quality 10%
""")

if hourly_entries_df is not None and not hourly_entries_df.empty:

    ranked_hourly = rank_hourly_candidates(hourly_entries_df, combined)

    top_n = st.slider("Show top N setups", min_value=5, max_value=50, value=15, step=5)
    show_top = ranked_hourly.head(top_n).reset_index(drop=True)

    # Build display table
    display_cols = {
        "Ticker":                      "Ticker",
        "FOCUS":                       "Focus",
        "FOCUS_SCORE":                 "Score",
        "FINAL_SIGNAL":                "Daily Signal",
        "entry_618_hit":               "At Entry?",
        "entry_618":                   "Entry (61.8%)",
        "stop":                        "Stop",
        "take_profit":                 "Target",
        "rr":                          "R:R",
        "distance_to_entry_618_pct":   "Dist to Entry",
        "retrace_from_high_pct":       "Retrace %",
        "bars_since_high":             "Bars Since High",
        "BREAKOUT_PRESSURE":           "BP",
        "INSIGHT_TAGS":                "Tags",
    }

    avail_cols = {k: v for k, v in display_cols.items() if k in show_top.columns}
    hourly_view = show_top[list(avail_cols.keys())].rename(columns=avail_cols).copy()

    # Format percentages
    for col_raw, col_display in [
        ("distance_to_entry_618_pct", "Dist to Entry"),
        ("retrace_from_high_pct",     "Retrace %"),
    ]:
        if col_display in hourly_view.columns:
            hourly_view[col_display] = (
                pd.to_numeric(hourly_view[col_display], errors="coerce") * 100
            ).round(2).astype(str) + "%"

    # Format R:R
    if "R:R" in hourly_view.columns:
        hourly_view["R:R"] = pd.to_numeric(
            hourly_view["R:R"], errors="coerce"
        ).round(2)

    # Format At Entry? as Yes/No
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

    # --- Row selection ---
    hourly_selected_rows = []
    if hourly_event is not None:
        if hasattr(hourly_event, "selection") and hasattr(hourly_event.selection, "rows"):
            hourly_selected_rows = hourly_event.selection.rows
        elif isinstance(hourly_event, dict):
            hourly_selected_rows = (
                hourly_event.get("selection", {}).get("rows", [])
            )

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
        sel_row_df = ranked_hourly[ranked_hourly["Ticker"] == hourly_ticker_selected]
        ticker_hourly = hourly_df[hourly_df["Ticker"] == hourly_ticker_selected].copy()

        if not sel_row_df.empty and not ticker_hourly.empty:
            hrow = sel_row_df.iloc[0]

            ticker_hourly["DateTime"] = pd.to_datetime(
                ticker_hourly["DateTime"], errors="coerce", utc=True
            ).dt.tz_convert(None)
            ticker_hourly = ticker_hourly.dropna(subset=["DateTime"]).sort_values("DateTime")

            high_dt    = pd.to_datetime(hrow.get("local_high_time"), errors="coerce", utc=True)
            retr_dt    = pd.to_datetime(hrow.get("DailyRetrLowDate"), errors="coerce", utc=True)
            if pd.notna(high_dt):  high_dt = high_dt.tz_convert(None)
            if pd.notna(retr_dt):  retr_dt = retr_dt.tz_convert(None)

            max_dt = ticker_hourly["DateTime"].max()
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

            focus_score = hrow.get("FOCUS_SCORE", "—")
            focus_label = hrow.get("FOCUS", "")
            daily_sig   = hrow.get("FINAL_SIGNAL", "—")
            tags        = hrow.get("INSIGHT_TAGS", "")

            st.markdown(
                f"**{hourly_ticker_selected}** &nbsp;|&nbsp; "
                f"{focus_label} (Score: {focus_score}) &nbsp;|&nbsp; "
                f"Daily: **{daily_sig}** &nbsp;|&nbsp; {tags}"
            )

            fig_h = go.Figure(data=[go.Candlestick(
                x=ticker_hourly["DateTime"],
                open=ticker_hourly["Open"],
                high=ticker_hourly["High"],
                low=ticker_hourly["Low"],
                close=ticker_hourly["Close"],
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
                        y=float(val),
                        line_dash="dash",
                        line_color=color,
                        annotation_text=label,
                        annotation_position="top left",
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
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tickers scanned", len(df_view))
with col2:
    st.metric("BUY signals", int((df_view["FINAL_SIGNAL"] == "BUY").sum()))
with col3:
    st.metric("WATCH signals", int((df_view["FINAL_SIGNAL"] == "WATCH").sum()))
with col4:
    bp_mean = pd.to_numeric(df_view.get("BREAKOUT_PRESSURE"), errors="coerce").mean()
    st.metric("Avg Breakout Pressure", f"{bp_mean:.1f}")


# ==========================================================
# RANKED DAILY DASHBOARD
# ==========================================================
st.write("### 📋 Ranked Daily Dashboard")

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
# CHART FUNCTION
# ==========================================================
def plot_ticker_chart(df_all, row, lookback_days=180):
    ticker = row["Ticker"]
    df_full = df_all[df_all["Ticker"] == ticker].sort_values("Date").copy()
    if df_full.empty:
        st.write("No price data found.")
        return

    df_full["SMA10"]   = df_full["Close"].rolling(10).mean()
    df_full["EMA20"]   = df_full["Close"].ewm(span=20).mean()
    df_full["EMA50"]   = df_full["Close"].ewm(span=50).mean()
    df_full["EMA12"]   = df_full["Close"].ewm(span=12).mean()
    df_full["EMA26"]   = df_full["Close"].ewm(span=26).mean()
    df_full["MACD"]    = df_full["EMA12"] - df_full["EMA26"]
    df_full["Signal"]  = df_full["MACD"].ewm(span=9).mean()
    df_full["MACDH"]   = df_full["MACD"] - df_full["Signal"]

    delta    = df_full["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    df_full["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    df_t  = df_full.tail(lookback_days).copy()
    dates = df_t["Date"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=df_t["Open"], high=df_t["High"],
        low=df_t["Low"], close=df_t["Close"], name=ticker,
    ), row=1, col=1)

    for ma, color in [("SMA10", "orange"), ("EMA20", "cyan"), ("EMA50", "magenta")]:
        fig.add_trace(go.Scatter(x=dates, y=df_t[ma], name=ma,
                                  line=dict(color=color, width=1)), row=1, col=1)

    swing_low  = row.get("SwingLow")
    swing_high = row.get("SwingHigh")
    if pd.notna(swing_low) and pd.notna(swing_high):
        swing = swing_high - swing_low
        x0, x1 = dates.iloc[0], dates.iloc[-1]
        for label, level in [
            ("100%", swing_high),
            ("78.6%", swing_high - 0.786 * swing),
            ("61.8%", swing_high - 0.618 * swing),
            ("50%",   swing_high - 0.500 * swing),
            ("38.2%", swing_high - 0.382 * swing),
            ("0%",    swing_low),
        ]:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                          line=dict(color="green", width=1, dash="dot"), row=1, col=1)
            fig.add_annotation(x=x1, y=level, text=label, showarrow=False,
                                xanchor="left", font=dict(size=10, color="green"),
                                row=1, col=1)

    fig.add_hline(y=0, line=dict(color="white", width=1), row=2, col=1)
    fig.add_trace(go.Bar(
        x=dates, y=df_t["MACDH"],
        marker_color=df_t["MACDH"].apply(lambda v: "green" if v >= 0 else "red"),
        opacity=0.45, name="MACDH",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_t["MACD"],   name="MACD"),   row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_t["Signal"], name="Signal"), row=2, col=1)

    fig.add_trace(go.Scatter(x=dates, y=df_t["RSI"], name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="red",   dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD",  row=2, col=1)
    fig.update_yaxes(title_text="RSI",   row=3, col=1, range=[0, 100])
    fig.update_layout(height=760, showlegend=False,
                      margin=dict(l=0, r=0, t=20, b=20),
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# TRADING SUMMARY CARD
# ==========================================================
def format_section(text, start, end):
    try:
        section = text.split(start, 1)[1]
        if end:
            section = section.split(end, 1)[0]
        lines = [f"• {l.strip()}" for l in section.split("\n") if l.strip()]
        return "<br>".join(lines)
    except Exception:
        return "N/A"


def render_summary_card(row, hourly_row=None):
    summary = generate_trading_summary(row)
    st.markdown("### 📘 Trading Summary")

    hourly_html = ""
    if hourly_row is not None and not hourly_row.empty:
        hr = hourly_row.iloc[0]
        entry_v   = float(hr.get("entry_618",              np.nan))
        stop_v    = float(hr.get("stop",                   np.nan))
        tp_v      = float(hr.get("take_profit",            np.nan))
        pb_v      = float(hr.get("retrace_from_high_pct",  np.nan))
        dist_v    = float(hr.get("distance_to_entry_618_pct", np.nan))
        hit_v     = bool(hr.get("entry_618_hit", False))
        rr_v      = float(hr.get("rr", np.nan))
        focus_v   = hr.get("FOCUS_SCORE", "—")
        focus_l   = hr.get("FOCUS", "")

        hourly_html = f"""
<h3 style="color:#4CC9F0; margin-bottom:5px;">⏱️ Hourly Entry Plan</h3>
<b>Focus:</b> {focus_l} ({focus_v})<br>
<b>Entry (61.8%):</b> {entry_v:.4f} &nbsp; {'<span style="color:lime">✅ AT ENTRY NOW</span>' if hit_v else ''}<br>
<b>Stop:</b> {stop_v:.4f}<br>
<b>Target:</b> {tp_v:.4f}<br>
<b>R:R:</b> {rr_v:.2f if not np.isnan(rr_v) else "N/A"}<br>
<b>Pullback from high:</b> {pb_v*100:.2f}%<br>
<b>Distance to entry:</b> {dist_v*100:.2f}%<br><br>
"""

    html = f"""
<div style="background:#f8f9fa;padding:20px;border-radius:10px;
            border:1px solid #ddd;font-size:15px;">

<h3 style="color:#4CC9F0;margin-bottom:5px;">🎯 Overview</h3>
<b>Ticker:</b> {row['Ticker']}<br>
<b>Signal:</b> {row['FINAL_SIGNAL']}<br>
<b>Shape:</b> {row.get('Shape','—')}<br>
<b>Insights:</b> {row.get('INSIGHT_TAGS','—')}<br>
<b>Next Action:</b> {row.get('NEXT_ACTION','—')}<br><br>

<h3 style="color:#4CC9F0;margin-bottom:5px;">📈 Interpretation</h3>
{format_section(summary, "Interpretation:", "Your Trading Plan")}
<br>

<h3 style="color:#4CC9F0;margin-bottom:5px;">📝 Trade Plan</h3>
{format_section(summary, "Primary Entry:", "No-Trade Conditions:")}
<br>

<h3 style="color:#F72585;margin-bottom:5px;">⚠️ Risk Conditions</h3>
{format_section(summary, "No-Trade Conditions:", None)}
<br>
{hourly_html}
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
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Signal",   row_sel["FINAL_SIGNAL"])
    with c2: st.metric("Shape",    row_sel.get("Shape", "—"))
    with c3: st.metric("Breakout Pressure", f"{row_sel.get('BREAKOUT_PRESSURE', 0):.1f}")
    with c4:
        pe = row_sel.get("PERFECT_ENTRY")
        st.metric("Perfect Entry", f"{pe:.1f}" if pd.notna(pe) else "N/A")

    plot_ticker_chart(df_all, row_sel, lookback_days=lookback_days)

    hourly_sel = None
    if hourly_entries_df is not None and not hourly_entries_df.empty:
        # Use ranked version so FOCUS_SCORE is available
        try:
            hourly_sel = ranked_hourly[ranked_hourly["Ticker"] == ticker_selected]
            if hourly_sel.empty:
                hourly_sel = None
        except NameError:
            hourly_sel = None

    render_summary_card(row_sel, hourly_row=hourly_sel)
else:
    st.info("Click a row in the table above to see charts and trading summary.")


# ==========================================================
# LEGEND
# ==========================================================
st.write("---")
st.subheader("📘 How to read this dashboard")
st.markdown("""
### Focus Score (Hourly)
Ranks candidates by **how actionable they are right now**. Factors:
- **Entry proximity (30%)** — how close price is to the 61.8% Fibonacci entry
- **Entry hit bonus (+15pts)** — price is *at* the zone right now → act immediately
- **R:R (25%)** — reward vs risk, capped at 5:1
- **Daily signal (20%)** — BUY from daily engine scores higher than WATCH
- **Breakout pressure (15%)** — structural energy built up on the daily chart
- **Shape quality (10%)** — cleaner structure (consolidation, rounded) scores higher

---
### Insight Tags
- **🔥 PRIME** — clean structure, very close to BUY
- **⚡ BOS_IMMINENT** — price just below breakout level
- **💥 MACD_THRUST** — momentum expanding
- **📉 SQUEEZE** — volatility coil, likely to break fast
- **🔋 ENERGY_BUILDUP** — rising energy, narrowing range
- **🎯 PERFECT_ENTRY** — clean retracement and higher low

---
### Breakout Pressure
Measures structural energy: BOS proximity + higher-low strength + MACD/RSI thrust + compression.
Higher = stronger probability of upside continuation.
""")
