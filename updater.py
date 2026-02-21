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
# 1. UNIVERSE BUILDERS
# ==========================================================

def get_sp500_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break

    if df is None:
        raise RuntimeError("Could not find S&P500 table on Wikipedia")

    df["Ticker"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    df["Name"] = df["Security"].astype(str)
    df["Sector"] = df["GICS Sector"].astype(str)

    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe():
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index", "sub index", "code"]):
            df = t.copy()
            break

    if df is None:
        raise RuntimeError("Could not find HSI table on Wikipedia")

    df.columns = [str(c).lower() for c in df.columns]

    ticker_col = None
    for c in df.columns:
        if "sehk" in c or "ticker" in c or "code" in c:
            ticker_col = c
            break

    if ticker_col is None:
        raise RuntimeError("Could not find ticker column for HSI")

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

    if "name" in df.columns:
        name_col = "name"
    else:
        possible = [c for c in df.columns if c != ticker_col]
        name_col = possible[0] if possible else ticker_col

    df["Name"] = df[name_col].astype(str)
    df["Sector"] = df.get("sub-index", df.get("sub index", df.get("industry", None)))

    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe():
    """
    Scrape Euro Stoxx 50 constituents from Wikipedia and map them to Yahoo symbols
    by using Exchange/Country columns to append the correct Yahoo suffix.
    """
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))

    # Find a constituents table. Wikipedia formatting changes; be flexible.
    table = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        has_ticker = any(c in cols for c in ["ticker", "symbol"])
        has_name = any(c in cols for c in ["name", "company"])
        if has_ticker and has_name:
            table = t.copy()
            break

    if table is None:
        raise RuntimeError("EURO STOXX 50 scrape failed: could not find a constituents table with ticker + name/company")

    # Normalize columns (keep original casing for later, but build lookup)
    colmap = {str(c).strip().lower(): str(c).strip() for c in table.columns}
    ticker_col = colmap.get("ticker", colmap.get("symbol"))
    name_col = colmap.get("name", colmap.get("company"))

    # Optional columns that help mapping
    exchange_col = None
    country_col = None
    sector_col = None

    for key in ["exchange", "listing", "market"]:
        if key in colmap:
            exchange_col = colmap[key]
            break
    for key in ["country", "country (of incorporation)", "headquarters"]:
        if key in colmap:
            country_col = colmap[key]
            break
    for key in ["sector", "industry"]:
        if key in colmap:
            sector_col = colmap[key]
            break

    if ticker_col is None or name_col is None:
        raise RuntimeError("EURO STOXX 50 scrape failed: missing ticker/name columns after normalization")

    # Yahoo suffix mapping
    exchange_suffix = {
        # Germany
        "xetra": ".DE",
        "frankfurt": ".DE",
        "deutsche börse": ".DE",
        # France
        "euronext paris": ".PA",
        "paris": ".PA",
        # Netherlands
        "euronext amsterdam": ".AS",
        "amsterdam": ".AS",
        # Italy
        "borsa italiana": ".MI",
        "milan": ".MI",
        "milano": ".MI",
        # Spain
        "madrid": ".MC",
        "bolsas y mercados españoles": ".MC",
        "bme": ".MC",
        # Belgium
        "euronext brussels": ".BR",
        "brussels": ".BR",
        # Finland
        "helsinki": ".HE",
        "nasdaq helsinki": ".HE",
        # Ireland
        "irish stock exchange": ".IR",
        "dublin": ".IR",
        # Portugal
        "euronext lisbon": ".LS",
        "lisbon": ".LS",
        # Austria
        "vienna": ".VI",
        # Switzerland
        "swiss exchange": ".SW",
        "six swiss exchange": ".SW",
    }

    country_suffix = {
        "germany": ".DE",
        "france": ".PA",
        "netherlands": ".AS",
        "italy": ".MI",
        "spain": ".MC",
        "belgium": ".BR",
        "finland": ".HE",
        "ireland": ".IR",
        "portugal": ".LS",
        "austria": ".VI",
        "switzerland": ".SW",
    }

    # Small set of known special cases (kept, but should be rarely needed now)
    yahoo_overrides = {
        # sometimes Wikipedia has legacy/odd formats
        "NOKIA": "NOKIA.HE",
    }

    def _to_yahoo_symbol(raw_ticker: str, exchange: str | None, country: str | None) -> str:
        t = (raw_ticker or "").strip()
        if not t:
            return ""

        # Overrides first
        if t in yahoo_overrides:
            return yahoo_overrides[t]

        # Already a Yahoo symbol with suffix
        if "." in t:
            return t

        exch = (exchange or "").strip().lower()
        ctry = (country or "").strip().lower()

        # Best: exchange-based mapping
        for k, suf in exchange_suffix.items():
            if k in exch:
                return f"{t}{suf}"

        # Fallback: country-based mapping
        for k, suf in country_suffix.items():
            if k in ctry:
                return f"{t}{suf}"

        # Last resort: return as-is (may still work for a few names, but most won't)
        return t

    tickers_raw = table[ticker_col].astype(str).str.strip()
    exchanges = table[exchange_col].astype(str).str.strip() if exchange_col else pd.Series([None] * len(table))
    countries = table[country_col].astype(str).str.strip() if country_col else pd.Series([None] * len(table))

    yahoo_tickers = [
        _to_yahoo_symbol(t, e if exchange_col else None, c if country_col else None)
        for t, e, c in zip(tickers_raw.tolist(), exchanges.tolist(), countries.tolist())
    ]

    out = pd.DataFrame(
        {
            "Ticker": yahoo_tickers,
            "Name": table[name_col].astype(str).str.strip(),
            "Sector": table[sector_col].astype(str).str.strip() if sector_col else None,
        }
    )

    out = out.dropna(subset=["Ticker"])
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out = out[out["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    return out[["Ticker", "Name", "Sector"]]


# ==========================================================
# 2. YAHOO DOWNLOADER (600 days, batched)
# ==========================================================

def download_yahoo_prices(tickers, label, period="600d", interval="1d"):
    """
    Downloads last `period` of OHLC for a list of tickers.
    Uses yf.download in batches of 40 tickers (no manual threads).
    Returns a list of clean DataFrames.
    """
    if not tickers:
        return []

    # clean list
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    batch_size = 40
    frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        try:
            data = yf.download(
                batch,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        if data is None or getattr(data, "empty", True):
            continue

        is_multi = isinstance(data.columns, pd.MultiIndex)

        for t in batch:
            try:
                if is_multi:
                    if t not in data.columns.get_level_values(0):
                        continue
                    df_t = data[t].dropna().copy()
                else:
                    # yfinance returns non-multiindex for single ticker batches
                    if len(batch) != 1:
                        # unexpected shape; skip safely
                        continue
                    df_t = data.dropna().copy()

                if df_t.empty:
                    continue

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

    # Hard fail if EURO didn't load (otherwise you'll "think" it's working)
    if not euro:
        raise RuntimeError("EUROSTOXX50 download returned 0 tickers. Yahoo symbol mapping likely wrong / Wikipedia table changed.")

    if not (sp or hs or euro):
        raise RuntimeError("No OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    # Standardize date column name from reset_index()
    if "Date" not in combined.columns:
        # sometimes yfinance uses 'index' name variations, but reset_index() should create one
        # try to detect the first datetime-like column
        dt_candidates = [c for c in combined.columns if str(c).lower() in ("date", "datetime")]
        if dt_candidates:
            combined = combined.rename(columns={dt_candidates[0]: "Date"})
        else:
            raise RuntimeError("Daily data missing Date column after download")

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Safety checks
    if any(str(t).endswith(".SI") for t in combined["Ticker"].unique()):
        raise RuntimeError("Found .SI tickers; STI still present")

    europe_suffixes = (".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".IR", ".LS", ".VI", ".SW")
    if sum(str(t).endswith(europe_suffixes) for t in combined["Ticker"].unique()) < 10:
        raise RuntimeError("EURO STOXX tickers not present or Yahoo mapping failed")

    if "Index" not in combined.columns or not combined["Index"].isin(["SP500", "HSI", "EUROSTOXX50"]).any():
        raise RuntimeError("Index labels missing or EUROSTOXX50 not included")

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

    if not euro:
        raise RuntimeError("EUROSTOXX50 hourly download returned 0 tickers. Yahoo symbol mapping likely wrong / Wikipedia table changed.")

    if not (sp or hs or euro):
        raise RuntimeError("No hourly OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    datetime_col = "Datetime" if "Datetime" in combined.columns else ("Date" if "Date" in combined.columns else None)
    if datetime_col is None:
        raise RuntimeError("Hourly data missing Datetime/Date column")

    combined = combined.rename(columns={datetime_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
    combined = combined.dropna(subset=["DateTime"]).sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(combined.columns)
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    if any(str(t).endswith(".SI") for t in combined["Ticker"].unique()):
        raise RuntimeError("Found .SI tickers; STI still present")

    europe_suffixes = (".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".IR", ".LS", ".VI", ".SW")
    if sum(str(t).endswith(europe_suffixes) for t in combined["Ticker"].unique()) < 10:
        raise RuntimeError("EURO STOXX tickers not present or Yahoo mapping failed")

    if "Index" not in combined.columns or not combined["Index"].isin(["SP500", "HSI", "EUROSTOXX50"]).any():
        raise RuntimeError("Index labels missing or EUROSTOXX50 not included")

    return combined
