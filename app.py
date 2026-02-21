import pandas as pd
import streamlit as st

from engine import run_engine


st.set_page_config(page_title="Fib Trading Scanner", layout="wide")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_scan_results():
    return run_engine()


def _format_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted[col]):
            formatted[col] = formatted[col].dt.strftime("%Y-%m-%d %H:%M")
    return formatted


def main():
    st.title("Fib Trading Scanner")
    st.caption("Daily swing/retracement scan + hourly entry candidates")

    refresh = st.button("Refresh scan")
    if refresh:
        load_scan_results.clear()

    try:
        with st.spinner("Running market scan. This can take a minute..."):
            daily_list, hourly_list = load_scan_results()
    except Exception as exc:
        st.error(f"Unable to run market scan: {exc}")
        st.info("This app requires external network access to Wikipedia and Yahoo Finance.")
        return

    c1, c2 = st.columns(2)
    c1.metric("Daily matches", len(daily_list))
    c2.metric("Hourly entries", len(hourly_list))

    st.subheader("Hourly Entry List")
    if hourly_list.empty:
        st.info("No hourly entries found for the current scan.")
    else:
        st.dataframe(_format_datetime_columns(hourly_list), use_container_width=True)

    st.subheader("Daily List")
    if daily_list.empty:
        st.info("No daily matches found for the current scan.")
    else:
        st.dataframe(_format_datetime_columns(daily_list), use_container_width=True)


if __name__ == "__main__":
    main()
