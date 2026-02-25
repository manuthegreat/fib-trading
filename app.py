import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import run_engine, generate_trading_summary
from ranking import rank_hourly_candidates


# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Momentum Dashboard",
    layout="wide",
)

st.title("📈 Fib Retracement Dashboard")

st.markdown("""
### What this app does
This dashboard scans **all S&P 500, Hang Seng Index (HSI), and EURO STOXX 50** companies and identifies names that:

- recently made a swing high  
- are currently **retracing into Fibonacci support zones**  
- show signs of **bullish structure, momentum, and rebound strength**  
- may be presenting **high-probability buying opportunities**

These setups are evaluated using structural signals, retracement depth, momentum confirmation, and pattern behavior.
""")

# Hide Streamlit sidebar toggle by default
hide_sidebar = """
<style>
    button[kind="header"] {display: none !important;}
</style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Settings")

lookback_days = st.sidebar.slider(
    "Chart lookback (days)",
    min_value=60,
    max_value=500,
    value=180,
    step=10,
)

st.sidebar.write("---")
st.sidebar.write("Run this after market close / before open.")


# -----------------------------
# Cache Engine Run
# -----------------------------
ENGINE_VERSION = "2026-02-21-ui-cleanup-v1"


@st.cache_data(show_spinner=True)
def compute_dashboard(engine_version):
    _ = engine_version
    (
        df_all,
        combined,
        insight_df,
        hourly_entries_df,
        hourly_rejects_df,
        hourly_df,
    ) = run_engine()
    return df_all, combined, insight_df, hourly_entries_df, hourly_rejects_df, hourly_df


df_all, combined, insight_df, hourly_entries_df, hourly_rejects_df, hourly_df = compute_dashboard(ENGINE_VERSION)

if combined.empty:
    st.error("No names in watchlist / combined. Check data or parameters.")
    st.stop()


# ---------------------------------------------------------
# Daily list uses combined output directly to preserve existing ticker selection/ranking.
# Removed only UI-level filters whose defaults were non-restrictive (0 / unchecked).
# ---------------------------------------------------------
df_view = combined.copy()

if df_view.empty:
    st.warning("No tickers match current filters.")
    st.stop()


# ---------------------------------------------------------
# Hourly Entry Candidates (List B)
# ---------------------------------------------------------
st.write("### Hourly Entry Candidates (List B)")

if hourly_entries_df is not None and not hourly_entries_df.empty:
    price_lookup = {}
    if combined is not None and not combined.empty and {"Ticker", "Latest Price"}.issubset(combined.columns):
        price_lookup = combined.set_index("Ticker")["Latest Price"].to_dict()

    hourly_rank_input = hourly_entries_df.copy()
    hourly_rank_input["entry"] = pd.to_numeric(hourly_rank_input.get("entry_618"), errors="coerce")
    hourly_rank_input["side"] = hourly_rank_input.get("side", "long")
    hourly_rank_input["current_price"] = hourly_rank_input["Ticker"].map(price_lookup)
    hourly_rank_input["current_price"] = hourly_rank_input["current_price"].fillna(
        pd.to_numeric(hourly_rank_input.get("last_close"), errors="coerce")
    )

    # readiness removed: ranking now uses R:R only.
    ranked_hourly = rank_hourly_candidates(hourly_rank_input, current_price_col="current_price")

    top_n = st.slider("Top hourly setups", min_value=3, max_value=50, value=15, step=1)

    hourly_view = ranked_hourly[[
        "Ticker",
        "side",
        "entry",
        "stop",
        "take_profit",
        "rr",
        "reward_from_current_pct",
        "DailyRetrLowDate",
        "DailyRetrLowPrice",
        "local_high_time",
        "local_high",
        "entry_382",
        "entry_50",
        "entry_618",
        "last_close",
        "retrace_from_high_pct",
        "bars_since_high",
        "distance_to_entry_618_pct",
        "entry_618_hit",
    ]].head(top_n).copy()

    hourly_view = hourly_view.rename(
        columns={
            "DailyRetrLowDate": "Daily Low Date",
            "DailyRetrLowPrice": "Daily Low",
            "local_high_time": "Hourly High Time",
            "local_high": "Hourly High",
            "entry": "Entry",
            "entry_382": "Entry 38.2%",
            "entry_50": "Entry 50%",
            "entry_618": "Entry 61.8%",
            "rr": "R:R",
            "reward_from_current_pct": "Reward Left %",
            "take_profit": "Take Profit",
            "last_close": "Last Close",
            "retrace_from_high_pct": "Retrace % From High",
            "bars_since_high": "Bars Since High",
            "distance_to_entry_618_pct": "Distance to 61.8%",
            "entry_618_hit": "61.8% Hit",
        }
    )

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
        elif hasattr(hourly_event, "rows"):
            hourly_selected_rows = hourly_event.rows
        elif isinstance(hourly_event, dict):
            if "selection" in hourly_event and isinstance(hourly_event["selection"], dict):
                hourly_selected_rows = hourly_event["selection"].get("rows", [])
            else:
                hourly_selected_rows = hourly_event.get("rows", [])

    if "hourly_selected_ticker" not in st.session_state:
        st.session_state.hourly_selected_ticker = None

    hourly_top = ranked_hourly.head(top_n).reset_index(drop=True)
    if hourly_selected_rows:
        selected_idx = hourly_selected_rows[0]
        if 0 <= selected_idx < len(hourly_top):
            st.session_state.hourly_selected_ticker = hourly_top.iloc[selected_idx]["Ticker"]

    if (
        st.session_state.hourly_selected_ticker is None
        and not hourly_top.empty
    ):
        st.session_state.hourly_selected_ticker = hourly_top.iloc[0]["Ticker"]

    hourly_ticker_selected = st.session_state.hourly_selected_ticker

    if hourly_ticker_selected and hourly_df is not None and not hourly_df.empty:
        selected_hourly_row = ranked_hourly[ranked_hourly["Ticker"] == hourly_ticker_selected]
        ticker_hourly = hourly_df[hourly_df["Ticker"] == hourly_ticker_selected].copy()

        if (not selected_hourly_row.empty) and (not ticker_hourly.empty):
            hrow = selected_hourly_row.iloc[0]
            ticker_hourly["DateTime"] = pd.to_datetime(ticker_hourly["DateTime"], errors="coerce", utc=True)
            ticker_hourly["DateTime"] = ticker_hourly["DateTime"].dt.tz_convert(None)
            ticker_hourly = ticker_hourly.dropna(subset=["DateTime"]).sort_values("DateTime")

            retr_low_dt = pd.to_datetime(hrow.get("DailyRetrLowDate"), errors="coerce", utc=True)
            high_dt = pd.to_datetime(hrow.get("local_high_time"), errors="coerce", utc=True)

            if pd.notna(retr_low_dt):
                retr_low_dt = retr_low_dt.tz_convert(None)
            if pd.notna(high_dt):
                high_dt = high_dt.tz_convert(None)

            max_dt = ticker_hourly["DateTime"].max()
            context_start = high_dt - pd.Timedelta(hours=72) if pd.notna(high_dt) else max_dt - pd.Timedelta(hours=72)

            if pd.notna(retr_low_dt):
                window_start = min(retr_low_dt, context_start)
            else:
                window_start = context_start

            if pd.notna(high_dt):
                window_end = max(max_dt, high_dt + pd.Timedelta(hours=6))
            else:
                window_end = max_dt

            default_window_hourly = ticker_hourly[
                (ticker_hourly["DateTime"] >= window_start) & (ticker_hourly["DateTime"] <= window_end)
            ]

            if len(default_window_hourly) < 50:
                fallback_window = ticker_hourly.tail(240)
                if not fallback_window.empty:
                    window_start = fallback_window["DateTime"].min()
                    window_end = fallback_window["DateTime"].max()

            fig_hourly = go.Figure(
                data=[
                    go.Candlestick(
                        x=ticker_hourly["DateTime"],
                        open=ticker_hourly["Open"],
                        high=ticker_hourly["High"],
                        low=ticker_hourly["Low"],
                        close=ticker_hourly["Close"],
                        name=hourly_ticker_selected,
                    )
                ]
            )

            level_specs = [
                ("entry_382", "Entry 38.2%", "#1f77b4"),
                ("entry_50", "Entry 50%", "#9467bd"),
                ("entry_618", "Entry 61.8%", "#ff7f0e"),
                ("stop", "Stop", "#d62728"),
                ("take_profit", "Take Profit", "#2ca02c"),
            ]

            for col, label, color in level_specs:
                if col in hrow and pd.notna(hrow[col]):
                    fig_hourly.add_hline(
                        y=float(hrow[col]),
                        line_dash="dash",
                        line_color=color,
                        annotation_text=label,
                        annotation_position="top left",
                    )

            fig_hourly.update_layout(
                title=f"{hourly_ticker_selected} – Hourly (List B)",
                xaxis_title="DateTime",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=500,
            )
            fig_hourly.update_xaxes(range=[window_start, window_end])

            visible_hourly = ticker_hourly[
                (ticker_hourly["DateTime"] >= window_start) & (ticker_hourly["DateTime"] <= window_end)
            ]
            if not visible_hourly.empty:
                y_low = pd.to_numeric(visible_hourly["Low"], errors="coerce").min()
                y_high = pd.to_numeric(visible_hourly["High"], errors="coerce").max()
                if pd.notna(y_low) and pd.notna(y_high):
                    y_span = y_high - y_low
                    y_padding = max(y_span * 0.08, y_high * 0.002)
                    fig_hourly.update_yaxes(range=[y_low - y_padding, y_high + y_padding])

            st.plotly_chart(fig_hourly, use_container_width=True)
else:
    st.info("No hourly entry candidates found for current run.")


# ---------------------------------------------------------
# Summary Metrics
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tickers (filtered)", len(df_view))
with col2:
    st.metric("BUY signals", (df_view["FINAL_SIGNAL"] == "BUY").sum())
with col3:
    st.metric("WATCH signals", (df_view["FINAL_SIGNAL"] == "WATCH").sum())
with col4:
    st.metric("Avg Breakout Pressure", f"{pd.to_numeric(df_view.get('BREAKOUT_PRESSURE'), errors='coerce').mean():.1f}")


# ---------------------------------------------------------
# Ranked Dashboard — Row-click selection using st.dataframe
# ---------------------------------------------------------
st.write("### Ranked Dashboard (Filtered)")

ranked_table = df_view[
    [
        "Ticker",
        "SwingHigh",
        "SwingLow",
        "Latest Price",
    ]
].reset_index(drop=True)

swing_range = (ranked_table["SwingHigh"] - ranked_table["SwingLow"]).replace(0, pd.NA)
ranked_table["Current Retracement %"] = (
    (ranked_table["SwingHigh"] - ranked_table["Latest Price"]) / swing_range
) * 100
ranked_table = ranked_table.drop(columns=["Latest Price"])

event = st.dataframe(
    ranked_table,
    hide_index=True,
    use_container_width=True,
    key="ranked_df",
    on_select="rerun",
    selection_mode="single-row",
)

# Handle selection for newer/older Streamlit selection payloads
selected_rows = []
if event is not None:
    if hasattr(event, "selection") and hasattr(event.selection, "rows"):
        selected_rows = event.selection.rows
    elif hasattr(event, "rows"):
        selected_rows = event.rows
    elif isinstance(event, dict):
        if "selection" in event and isinstance(event["selection"], dict):
            selected_rows = event["selection"].get("rows", [])
        else:
            selected_rows = event.get("rows", [])

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

if selected_rows:
    # DataframeSelectionState.rows are integer positions in the original df
    selected_idx = selected_rows[0]
    if 0 <= selected_idx < len(ranked_table):
        st.session_state.selected_ticker = ranked_table.iloc[selected_idx]["Ticker"]

ticker_selected = st.session_state.selected_ticker


# ---------------------------------------------------------
# Enhanced Chart Function (combined: price + MACD + RSI)
# ---------------------------------------------------------
def plot_ticker_chart(df_all, row, lookback_days=180):
    import numpy as np  # noqa: F401

    ticker = row["Ticker"]

    # 1. Load full history & compute indicators
    df_full = df_all[df_all["Ticker"] == ticker].sort_values("Date").copy()
    if df_full.empty:
        st.write("No price data found.")
        return

    # Moving averages
    df_full["SMA10"] = df_full["Close"].rolling(10).mean()
    df_full["EMA20"] = df_full["Close"].ewm(span=20).mean()
    df_full["EMA50"] = df_full["Close"].ewm(span=50).mean()

    # MACD
    df_full["EMA12"] = df_full["Close"].ewm(span=12).mean()
    df_full["EMA26"] = df_full["Close"].ewm(span=26).mean()
    df_full["MACD"] = df_full["EMA12"] - df_full["EMA26"]
    df_full["Signal"] = df_full["MACD"].ewm(span=9).mean()
    df_full["MACDH"] = df_full["MACD"] - df_full["Signal"]

    # RSI
    delta = df_full["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_full["RSI"] = 100 - (100 / (1 + rs))

    # Slice for display
    df_t = df_full.tail(lookback_days).copy()
    dates = df_t["Date"]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )

    # Price + MAs
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df_t["Open"],
            high=df_t["High"],
            low=df_t["Low"],
            close=df_t["Close"],
            name=ticker,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=dates, y=df_t["SMA10"], name="SMA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_t["EMA20"], name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df_t["EMA50"], name="EMA50"), row=1, col=1)

    swing_low = row["SwingLow"]
    swing_high = row["SwingHigh"]

    if pd.notna(swing_low) and pd.notna(swing_high):
        swing = swing_high - swing_low
        fib_levels = {
            "100%": swing_high,
            "78.6%": swing_high - 0.786 * swing,
            "61.8%": swing_high - 0.618 * swing,
            "50%": swing_high - 0.500 * swing,
            "38.2%": swing_high - 0.382 * swing,
            "0%": swing_low,
        }

        x0 = dates.iloc[0]
        x1 = dates.iloc[-1]

        for label, level in fib_levels.items():
            fig.add_shape(
                type="line",
                x0=x0,
                x1=x1,
                y0=level,
                y1=level,
                line=dict(color="green", width=1, dash="dot"),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=x1,
                y=level,
                text=label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=10, color="green"),
                row=1,
                col=1,
            )

    # MACD
    fig.add_hline(y=0, line=dict(color="white", width=1), row=2, col=1)
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df_t["MACDH"],
            marker_color=df_t["MACDH"].apply(
                lambda v: "green" if v >= 0 else "red"
            ),
            opacity=0.45,
            name="MACDH",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=dates, y=df_t["MACD"], name="MACD"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=dates, y=df_t["Signal"], name="Signal"), row=2, col=1
    )

    # RSI
    fig.add_trace(go.Scatter(x=dates, y=df_t["RSI"], name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    fig.update_layout(
        height=760,
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=20),
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Enhanced Trading Summary Card
# ---------------------------------------------------------
def format_section(summary_text, start, end):
    try:
        section = summary_text.split(start, 1)[1]
        if end:
            section = section.split(end, 1)[0]
        lines = [f"• {line.strip()}" for line in section.split("\n") if line.strip()]
        return "<br>".join(lines)
    except Exception:
        return "N/A"


def render_summary_card(row, hourly_row=None):
    summary = generate_trading_summary(row)

    st.markdown("### 📘 Trading Summary")

    hourly_plan_html = ""
    if hourly_row is not None and not hourly_row.empty:
        hr = hourly_row.iloc[0]
        entry_val = pd.to_numeric(pd.Series([hr.get("entry", hr.get("entry_618", np.nan))]), errors="coerce").iloc[0]
        stop_val = pd.to_numeric(pd.Series([hr.get("stop", np.nan)]), errors="coerce").iloc[0]
        tp_val = pd.to_numeric(pd.Series([hr.get("take_profit", np.nan)]), errors="coerce").iloc[0]
        pullback_val = pd.to_numeric(pd.Series([hr.get("pullback_pct", hr.get("retrace_from_high_pct", np.nan))]), errors="coerce").iloc[0]
        dist_val = pd.to_numeric(pd.Series([hr.get("distance_to_entry_pct", hr.get("distance_to_entry_618_pct", np.nan))]), errors="coerce").iloc[0]
        hit_val = hr.get("entry_hit", hr.get("entry_618_hit", False))

        hourly_plan_html = f"""
<h3 style=\"color:#4CC9F0; margin-bottom:5px;\">⏱️ Hourly Entry Plan</h3>
<b>Entry:</b> {entry_val:.4f}<br>
<b>Stop:</b> {stop_val:.4f}<br>
<b>Take Profit:</b> {tp_val:.4f}<br>
<b>Pullback:</b> {pullback_val*100:.2f}%<br>
<b>Distance to Entry:</b> {dist_val*100:.2f}%<br>
<b>Entry Hit:</b> {bool(hit_val)}<br><br>
"""

    html = f"""
<div style="background-color:#f8f9fa;padding:20px;border-radius:10px;border:1px solid #ddd; font-size:15px;">

<h3 style="color:#4CC9F0; margin-bottom:5px;">🎯 Overview</h3>
<b>Ticker:</b> {row['Ticker']}<br>
<b>Signal:</b> {row['FINAL_SIGNAL']}<br>
<b>Shape:</b> {row['Shape']}<br>
<b>Insights:</b> {row['INSIGHT_TAGS']}<br>
<b>Next Action:</b> {row['NEXT_ACTION']}<br><br>

<h3 style="color:#4CC9F0; margin-bottom:5px;">📈 Interpretation</h3>
{format_section(summary, "Interpretation:", "Your Trading Plan")}

<br>

<h3 style="color:#4CC9F0; margin-bottom:5px;">📝 Trade Plan</h3>
{format_section(summary, "Primary Entry:", "No-Trade Conditions:")}

<br>

<h3 style="color:#F72585; margin-bottom:5px;">⚠️ Risk Conditions</h3>
{format_section(summary, "No-Trade Conditions:", None)}

<br>
{hourly_plan_html}

</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------
# Ticker Drilldown
# ---------------------------------------------------------
if ticker_selected:
    selected_rows_df = df_view[df_view["Ticker"] == ticker_selected]
    if selected_rows_df.empty:
        st.warning("Selected ticker no longer available.")
        st.stop()
    row_sel = selected_rows_df.iloc[0]

    st.write(f"### 📌 Selected: **{ticker_selected}**")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Signal", row_sel["FINAL_SIGNAL"])
    with colB:
        st.metric("Shape", row_sel.get("Shape", "N/A"))
    with colC:
        st.metric("Breakout Pressure", f"{row_sel['BREAKOUT_PRESSURE']:.2f}")
    with colD:
        st.metric(
            "Perfect Entry",
            f"{row_sel['PERFECT_ENTRY']:.2f}"
            if pd.notna(row_sel["PERFECT_ENTRY"])
            else "N/A",
        )

    plot_ticker_chart(df_all, row_sel, lookback_days=lookback_days)
    hourly_sel = hourly_entries_df[hourly_entries_df["Ticker"] == ticker_selected] if hourly_entries_df is not None and not hourly_entries_df.empty else None
    render_summary_card(row_sel, hourly_row=hourly_sel)
else:
    st.info("Click a row in the table to display charts and trading summary.")


st.write("---")
st.subheader("📘 Indicator Explanations")

st.markdown("""
### **Insight Tags**
These are quick-glance labels that highlight strong structural or momentum characteristics:
- **🔥 PRIME** – very clean structure, very close to turning into a BUY  
- **⚡ BOS_IMMINENT** – price is sitting just below the breakout level  
- **💥 MACD_THRUST** – strong momentum expansion  
- **📉 SQUEEZE** – tight volatility coil, likely to explode  
- **🔋 ENERGY_BUILDUP** – rising energy with a narrowing range  
- **🎯 PERFECT_ENTRY** – exceptionally clean retracement & higher low  
- *and others…*

---

### **Ranking Note**
Hourly candidates are ranked by actionable **Risk:Reward (R:R)** and remaining reward from the current price.

---

### **Breakout Pressure**
Measures the “energy” pushing price upward:
- closeness to BOS  
- higher-low strength  
- MACD / RSI thrust  
- volatility compression  

Higher = stronger probability of breakout continuation.

---

### **Perfect Entry Score**
Evaluates the **quality of the retracement**, **cleanliness of the higher low**, **shape geometry**, and **proximity to BOS**.

Score > 80 normally signals an institution-grade entry structure.
""")
