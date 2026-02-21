import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import run_engine, generate_trading_summary


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
This dashboard scans **all S&P 500, Hang Seng Index (HSI), and Straits Times Index (STI)** companies and identifies names that:

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

min_readiness = st.sidebar.slider(
    "Min Readiness Score",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)

min_pressure = st.sidebar.slider(
    "Min Breakout Pressure",
    min_value=0,
    max_value=100,
    value=0,
    step=1,
)

show_only_insights = st.sidebar.checkbox(
    "Show only insight names (INSIGHT_TAGS != '')",
    value=False,
)

lookback_days = st.sidebar.slider(
    "Chart lookback (days)",
    min_value=60,
    max_value=500,
    value=180,
    step=10,
)

# -----------------------------
# Insight Tag Filter
# -----------------------------
st.sidebar.subheader("Filter by Insight Tags")

INSIGHT_OPTIONS = [
    "🔥 PRIME",
    "⚡ BOS_IMMINENT",
    "🎯 PERFECT_ENTRY",
    "🌀 STRUCTURE_STRONG",
    "📉 SQUEEZE",
    "💥 MACD_THRUST",
    "📈 EARLY_BOS",
    "🔋 ENERGY_BUILDUP",
    "🔄 REVERSAL_CONFIRM",
    "🛑 EXTENDED",
]

selected_insights = st.sidebar.multiselect(
    "Show tickers with any selected tags:",
    INSIGHT_OPTIONS,
    default=[],
)

st.sidebar.write("---")
st.sidebar.write("Run this daily after market close / before open.")


# -----------------------------
# Cache Engine Run
# -----------------------------
@st.cache_data(show_spinner=True)
def compute_dashboard():
    df_all, combined, insight_df, hourly_entries_df, hourly_rejects_df = run_engine()
    return df_all, combined, insight_df, hourly_entries_df, hourly_rejects_df


df_all, combined, insight_df, hourly_entries_df, hourly_rejects_df = compute_dashboard()

if combined.empty:
    st.error("No names in watchlist / combined. Check data or parameters.")
    st.stop()


# ---------------------------------------------------------
# Apply Basic Filters
# ---------------------------------------------------------
df_view = combined.copy()
df_view = df_view[df_view["READINESS_SCORE"] >= min_readiness]
df_view = df_view[df_view["BREAKOUT_PRESSURE"] >= min_pressure]

if show_only_insights:
    df_view = df_view[df_view["INSIGHT_TAGS"] != ""]

# Apply insight tag filtering
if selected_insights:
    df_view = df_view[
        df_view["INSIGHT_TAGS"].apply(
            lambda tags: any(tag in tags for tag in selected_insights)
        )
    ]

if df_view.empty:
    st.warning("No tickers match current filters.")
    st.stop()


# ---------------------------------------------------------
# Hourly Entry Candidates (List B)
# ---------------------------------------------------------
st.write("### Hourly Entry Candidates (List B)")

if hourly_entries_df is not None and not hourly_entries_df.empty:
    hourly_view = hourly_entries_df[[
        "Ticker",
        "DailyRetrLowDate",
        "DailyRetrLowPrice",
        "local_high_time",
        "local_high",
        "entry",
        "stop",
        "take_profit",
        "last_close",
        "pullback_pct",
        "triggered_last_bar",
        "near_entry",
    ]].copy()
    st.dataframe(hourly_view, hide_index=True, use_container_width=True)
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
    st.metric("Avg Readiness", f"{df_view['READINESS_SCORE'].mean():.1f}")


# ---------------------------------------------------------
# Ranked Dashboard — Row-click selection using st.dataframe
# ---------------------------------------------------------
st.write("### Ranked Dashboard (Filtered)")

ranked_table = df_view[
    [
        "Ticker",
        "FINAL_SIGNAL",
        "Shape",
        "BREAKOUT_PRESSURE",
        "PERFECT_ENTRY",
        "READINESS_SCORE",
        "INSIGHT_TAGS",
        "NEXT_ACTION",
    ]
].reset_index(drop=True)

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
        hourly_plan_html = f"""
<h3 style=\"color:#4CC9F0; margin-bottom:5px;\">⏱️ Hourly Entry Plan</h3>
<b>Entry:</b> {hr['entry']:.4f}<br>
<b>Stop:</b> {hr['stop']:.4f}<br>
<b>Take Profit:</b> {hr['take_profit']:.4f}<br>
<b>Pullback:</b> {hr['pullback_pct']*100:.2f}%<br>
<b>Triggered Last Bar:</b> {bool(hr['triggered_last_bar'])}<br>
<b>Near Entry:</b> {bool(hr['near_entry'])}<br><br>
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
    row_sel = df_view[df_view["Ticker"] == ticker_selected].iloc[0]

    st.write(f"### 📌 Selected: **{ticker_selected}**")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Signal", row_sel["FINAL_SIGNAL"])
    with colB:
        st.metric("Readiness", f"{row_sel['READINESS_SCORE']:.2f}")
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

### **Readiness Score**
How close a setup is to a confirmed BUY, combining:
- retracement depth  
- higher low confirmation  
- bullish reaction candle  
- momentum alignment  
- proximity to BOS  

**100 = ready to break out any moment.**

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

