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

    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break

    if df is None:
        raise RuntimeError("Could not find S&P500 table on Wikipedia")

    df["Ticker"] = df["Symbol"].str.replace(".", "-", regex=False)
    df["Name"] = df["Security"]
    df["Sector"] = df["GICS Sector"]

    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe():
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)

    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index"]):
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
        name_col = possible[0]

    df["Name"] = df[name_col]
    df["Sector"] = df.get("sub-index", df.get("industry", None))

    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe():
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch EURO STOXX 50 Wikipedia page: {exc}") from exc

    try:
        tables = pd.read_html(StringIO(response.text))
    except ValueError as exc:
        raise RuntimeError("No HTML tables found on EURO STOXX 50 Wikipedia page") from exc

    table = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols and "name" in cols:
            table = t.copy()
            break

    if table is None:
        raise RuntimeError(
            "EURO STOXX 50 scrape failed: constituents table with 'Ticker' and 'Name' columns was not found"
        )

    table.columns = [str(c).strip() for c in table.columns]
    if "Ticker" not in table.columns:
        raise RuntimeError("EURO STOXX 50 scrape failed: missing 'Ticker' column in constituents table")
    if "Name" not in table.columns:
        raise RuntimeError("EURO STOXX 50 scrape failed: missing 'Name' column in constituents table")

    sector_col = "Sector" if "Sector" in table.columns else None

    yahoo_overrides = {
        "ADYEN": "ADYEN.AS",
        "ASML": "ASML.AS",
        "BAS": "BAS.DE",
        "BAYN": "BAYN.DE",
        "BBVA": "BBVA.MC",
        "BMW": "BMW.DE",
        "BN": "BN.PA",
        "BNP": "BNP.PA",
        "CS": "CS.PA",
        "DHL": "DHL.DE",
        "DG": "DG.PA",
        "ENEL": "ENEL.MI",
        "ENI": "ENI.MI",
        "IBE": "IBE.MC",
        "IFX": "IFX.DE",
        "INGA": "INGA.AS",
        "ISP": "ISP.MI",
        "ITX": "ITX.MC",
        "KER": "KER.PA",
        "MUV2": "MUV2.DE",
        "NOKIA": "NOKIA.HE",
        "OR": "OR.PA",
        "PHIA": "PHIA.AS",
        "RMS": "RMS.PA",
        "SAF": "SAF.PA",
        "SAN": "SAN.MC",
        "SAP": "SAP.DE",
        "SIE": "SIE.DE",
        "SU": "SU.PA",
        "TTE": "TTE.PA",
        "UCG": "UCG.MI",
        "VOW3": "VOW3.DE",
    }

    tickers = table["Ticker"].astype(str).str.strip()
    tickers = tickers.replace(yahoo_overrides)

    out = pd.DataFrame(
        {
            "Ticker": tickers,
            "Name": table["Name"].astype(str).str.strip(),
            "Sector": table[sector_col].astype(str).str.strip() if sector_col else None,
        }
    )

    out = out.dropna(subset=["Ticker"])
    out = out[out["Ticker"] != ""].reset_index(drop=True)

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

        for t in batch:
            try:
                df_t = data[t].dropna().copy()
                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t.reset_index())
            except Exception:
                continue

    return frames


# ==========================================================
# 3. MASTER FUNCTION (engine/app call this)
# ==========================================================

def load_all_market_data():
    """
    Returns a merged OHLC dataframe for SP500 + HSI + EURO STOXX 50
    without saving anything to disk.
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    eurostoxx50 = get_eurostoxx50_universe()

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d")
    euro = download_yahoo_prices(eurostoxx50["Ticker"].tolist(), "EUROSTOXX50", period="600d")

    if not (sp or hs or euro):
        raise RuntimeError("No OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if any(str(t).endswith(".SI") for t in combined["Ticker"].unique()):
        raise RuntimeError("Found .SI tickers; STI still present")

    europe_suffixes = (".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".IR", ".LS", ".VI", ".SW")
    if sum(str(t).endswith(europe_suffixes) for t in combined["Ticker"].unique()) < 10:
        raise RuntimeError("EURO STOXX tickers not present or Yahoo mapping failed")

    if not ("Index" in combined.columns and combined["Index"].isin(["SP500", "HSI", "EUROSTOXX50"]).any()):
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

    if not (sp or hs or euro):
        raise RuntimeError("No hourly OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    datetime_col = "Datetime" if "Datetime" in combined.columns else "Date"
    if datetime_col not in combined.columns:
        raise RuntimeError("Hourly data missing Datetime/Date column")

    combined = combined.rename(columns={datetime_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"])
    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(combined.columns)
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    if any(str(t).endswith(".SI") for t in combined["Ticker"].unique()):
        raise RuntimeError("Found .SI tickers; STI still present")

    europe_suffixes = (".DE", ".PA", ".AS", ".MI", ".MC", ".BR", ".HE", ".IR", ".LS", ".VI", ".SW")
    if sum(str(t).endswith(europe_suffixes) for t in combined["Ticker"].unique()) < 10:
        raise RuntimeError("EURO STOXX tickers not present or Yahoo mapping failed")

    if not ("Index" in combined.columns and combined["Index"].isin(["SP500", "HSI", "EUROSTOXX50"]).any()):
        raise RuntimeError("Index labels missing or EUROSTOXX50 not included")

    return combined
