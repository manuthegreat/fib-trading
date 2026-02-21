import pandas as pd
import streamlit as st

from updater import load_all_market_data


st.set_page_config(page_title="Fib Retracement Dashboard", layout="wide")
st.title("📈 Fib Retracement Dashboard")
st.write("List of available tickers with a direct chart link.")


@st.cache_data(show_spinner=True)
def get_ticker_table() -> pd.DataFrame:
    df_all = load_all_market_data()

    latest = (
        df_all.sort_values("Date")
        .groupby("Ticker", as_index=False)
        .tail(1)[["Ticker", "Date", "Close"]]
        .rename(columns={"Date": "Last Date", "Close": "Last Close"})
    )

    latest["Chart Link"] = latest["Ticker"].apply(
        lambda t: f"https://www.tradingview.com/chart/?symbol={t}"
    )

    return latest.sort_values("Ticker").reset_index(drop=True)


df_view = get_ticker_table()

if df_view.empty:
    st.warning("No ticker data available.")
    st.stop()

st.dataframe(
    df_view,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Last Date": st.column_config.DateColumn("Last Date"),
        "Last Close": st.column_config.NumberColumn("Last Close", format="%.2f"),
        "Chart Link": st.column_config.LinkColumn("Chart Link", display_text="Open chart"),
    },
)
