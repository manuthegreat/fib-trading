"""
updater.py  (simple, Streamlit-friendly version)

✓ Builds SP500, HSI, EURO STOXX 50 universes
✓ Downloads last 600 days of OHLC from Yahoo
✓ Batches requests with yf.download (no manual threads)
✓ Returns ONE CLEAN MERGED DATAFRAME
✓ No parquet saving, no disk IO
"""

import pandas as pd
import requests
from io import StringIO
import yfinance as yf


# ==========================================================
# Helpers
# ==========================================================
def _flatten_columns(cols) -> list[str]:
    """
    pd.read_html can return MultiIndex columns if the HTML table has multi-row headers.
    This flattens columns into readable strings.
    """
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x).strip() for x in c if str(x).strip() and str(x).strip().lower() != "nan"]
            out.append(" ".join(parts).strip())
        else:
            out.append(str(c).strip())
    return out


def _find_col(df: pd.DataFrame, contains: list[str]) -> str | None:
    """
    Find the first column whose lowercase name contains ALL substrings in `contains`.
    """
    cols = _flatten_columns(df.columns)
    for col in cols:
        low = col.lower()
        if all(s in low for s in contains):
            return col
    return None


def _read_html_tables(url: str, headers: dict, timeout: int = 15) -> list[pd.DataFrame]:
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


# ==========================================================
# 1. UNIVERSE BUILDERS
# ==========================================================
def get_sp500_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    tables = _read_html_tables(url, headers=headers, timeout=15)

    df = None
    for t in tables:
        cols = _flatten_columns(t.columns)
        if any(c.strip().lower() == "symbol" for c in cols):
            df = t.copy()
            df.columns = cols
            break

    if df is None:
        raise RuntimeError("Could not find S&P500 table on Wikipedia")

    df["Ticker"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
    df["Name"] = df["Security"].astype(str).str.strip()
    df["Sector"] = df["GICS Sector"].astype(str).str.strip()

    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe():
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}

    tables = _read_html_tables(url, headers=headers, timeout=15)

    df = None
    for t in tables:
        cols = [c.lower() for c in _flatten_columns(t.columns)]
        # try to find the constituents table
        if any("ticker" in c or "code" in c or "sehk" in c for c in cols) and any("name" in c or "constitu" in c for c in cols):
            df = t.copy()
            df.columns = _flatten_columns(df.columns)
            break

    if df is None:
        raise RuntimeError("Could not find HSI constituents table on Wikipedia")

    # find ticker-like column
    ticker_col = None
    for c in df.columns:
        cl = c.lower()
        if "sehk" in cl or "ticker" in cl or "code" in cl:
            ticker_col = c
            break
    if ticker_col is None:
        raise RuntimeError("Could not find ticker column for HSI")

    # find name-like column
    name_col = None
    for c in df.columns:
        cl = c.lower()
        if c != ticker_col and ("name" in cl or "constitu" in cl or "company" in cl):
            name_col = c
            break
    if name_col is None:
        # fallback: first non-ticker col
        name_col = [c for c in df.columns if c != ticker_col][0]

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )
    df["Name"] = df[name_col].astype(str).str.strip()

    # optional sector/sub-index if present
    sector_col = None
    for c in df.columns:
        cl = c.lower()
        if "sub-index" in cl or "industry" in cl or "sector" in cl:
            sector_col = c
            break
    df["Sector"] = df[sector_col].astype(str).str.strip() if sector_col else None

    df = df[df["Ticker"].str.len() > 3].reset_index(drop=True)
    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe():
    """
    Wikipedia already lists Yahoo-friendly tickers like ADS.DE, SAN.PA, ASML.AS, etc.
    The key fix is handling multi-row headers -> MultiIndex columns.
    """
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    headers = {"User-Agent": "Mozilla/5.0"}

    tables = _read_html_tables(url, headers=headers, timeout=15)

    table = None
    ticker_col = None
    name_col = None
    sector_col = None

    for t in tables:
        tmp = t.copy()
        tmp.columns = _flatten_columns(tmp.columns)

        # We want the "Composition" constituents table. It has Ticker + Name.
        tc = _find_col(tmp, ["ticker"])
        nc = _find_col(tmp, ["name"])
        if tc and nc:
            table = tmp
            ticker_col = tc
            name_col = nc
            sector_col = _find_col(tmp, ["sector"])
            break

    if table is None:
        raise RuntimeError("EURO STOXX 50 scrape failed: could not find table with Ticker + Name columns")

    tickers = table[ticker_col].astype(str).str.strip()
    names = table[name_col].astype(str).str.strip()

    out = pd.DataFrame(
        {
            "Ticker": tickers,
            "Name": names,
            "Sector": table[sector_col].astype(str).str.strip() if sector_col else None,
        }
    )

    # basic cleaning
    out = out.dropna(subset=["Ticker"])
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out = out[out["Ticker"] != ""].reset_index(drop=True)

    # guard: should mostly be exchange-suffixed tickers
    # (Wikipedia currently lists like ADS.DE, SAN.PA, etc.)
    return out[["Ticker", "Name", "Sector"]]


# ==========================================================
# 2. YAHOO DOWNLOADER (batched)
# ==========================================================
def _extract_one_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    yfinance returns a few different shapes depending on tickers/interval.
    This extracts OHLC for a single ticker robustly.
    """
    if data is None or data.empty:
        return None

    # Case A: group_by="ticker" => columns MultiIndex: (ticker, field)
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = set(map(str, data.columns.get_level_values(0)))
        lvl1 = set(map(str, data.columns.get_level_values(1)))

        if ticker in lvl0:
            df_t = data[ticker].copy()
        elif ticker in lvl1:
            # sometimes columns are (field, ticker)
            df_t = data.xs(ticker, axis=1, level=1, drop_level=True).copy()
        else:
            return None
    else:
        # Single ticker request sometimes returns flat columns already
        # but in our batching we still might see flat if only 1 valid ticker returned.
        cols = [c.lower() for c in data.columns]
        needed = {"open", "high", "low", "close"}
        if needed.issubset(set(cols)):
            df_t = data.copy()
            # normalize column names
            rename_map = {c: c.title() for c in data.columns}
            df_t = df_t.rename(columns=rename_map)
        else:
            return None

    # Normalize expected OHLC column capitalization
    col_map = {}
    for c in df_t.columns:
        cl = str(c).lower()
        if cl == "open":
            col_map[c] = "Open"
        elif cl == "high":
            col_map[c] = "High"
        elif cl == "low":
            col_map[c] = "Low"
        elif cl == "close":
            col_map[c] = "Close"
        elif cl == "adj close":
            col_map[c] = "Adj Close"
        elif cl == "volume":
            col_map[c] = "Volume"
    df_t = df_t.rename(columns=col_map)

    if not {"Open", "High", "Low", "Close"}.issubset(df_t.columns):
        return None

    df_t = df_t.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df_t.empty:
        return None

    return df_t


def download_yahoo_prices(tickers, label, period="600d", interval="1d"):
    """
    Downloads last `period` of OHLC for a list of tickers.
    Uses yf.download in batches of 40 tickers (no manual threads).
    Returns a list of clean DataFrames.
    """
    tickers = [str(t).strip() for t in (tickers or []) if str(t).strip()]
    if not tickers:
        return []

    batch_size = 40
    frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]

        try:
            data = yf.download(
                tickers=batch,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for t in batch:
            try:
                df_t = _extract_one_ticker_frame(data, t)
                if df_t is None or df_t.empty:
                    continue
                df_t = df_t.copy()
                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t.reset_index())
            except Exception:
                continue

    return frames


# ==========================================================
# 3. MASTER FUNCTIONS (engine/app call these)
# ==========================================================
def load_all_market_data():
    """
    Returns a merged OHLC dataframe for SP500 + HSI + EURO STOXX 50
    without saving anything to disk.
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    eurostoxx50 = get_eurostoxx50_universe()

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d", interval="1d")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d", interval="1d")
    euro = download_yahoo_prices(eurostoxx50["Ticker"].tolist(), "EUROSTOXX50", period="600d", interval="1d")

    if not (sp or hs or euro):
        raise RuntimeError("No OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    # yfinance uses either "Date" or "Datetime" index name depending on interval
    date_col = "Date" if "Date" in combined.columns else ("Datetime" if "Datetime" in combined.columns else None)
    if date_col is None:
        raise RuntimeError("Downloaded data missing Date/Datetime column")

    combined = combined.rename(columns={date_col: "Date"})
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return combined


def load_all_market_data_hourly():
    """
    Returns merged hourly OHLC dataframe for SP500 + HSI + EURO STOXX 50.
    Output columns include: Ticker, DateTime, Open, High, Low, Close.
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    eurostoxx50 = get_eurostoxx50_universe()

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="60d", interval="60m")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="60d", interval="60m")
    euro = download_yahoo_prices(eurostoxx50["Ticker"].tolist(), "EUROSTOXX50", period="60d", interval="60m")

    if not (sp or hs or euro):
        raise RuntimeError("No hourly OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    datetime_col = "Datetime" if "Datetime" in combined.columns else ("Date" if "Date" in combined.columns else None)
    if datetime_col is None:
        raise RuntimeError("Hourly data missing Datetime/Date column")

    combined = combined.rename(columns={datetime_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
    combined = combined.dropna(subset=["DateTime"])
    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(combined.columns)
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    return combined
