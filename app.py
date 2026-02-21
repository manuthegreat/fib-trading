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

    if st.button("Refresh scan"):
        load_scan_results.clear()

    try:
        with st.spinner("Running market scan. This can take a minute..."):
            daily_list_df, hourly_list_df = load_scan_results()
    except Exception as exc:
        st.error(f"Unable to run market scan: {exc}")
        return

    st.subheader(f"Daily List ({len(daily_list_df)})")
    if daily_list_df.empty:
        st.info("No daily matches found for the current scan.")
    else:
        st.dataframe(_format_datetime_columns(daily_list_df), use_container_width=True)

    st.subheader(f"Hourly List ({len(hourly_list_df)})")
    if hourly_list_df.empty:
        st.info("No hourly entries found for the current scan.")
    else:
        st.dataframe(_format_datetime_columns(hourly_list_df), use_container_width=True)


if __name__ == "__main__":
    main()
