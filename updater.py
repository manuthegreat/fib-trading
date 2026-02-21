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

    df["Ticker"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
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
        if any(x in cols for x in ["ticker", "constituent", "sub-index", "code", "sehk"]):
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

    # Name column heuristic
    if "name" in df.columns:
        name_col = "name"
    else:
        possible = [c for c in df.columns if c != ticker_col]
        name_col = possible[0] if possible else ticker_col

    df["Name"] = df[name_col].astype(str)
    df["Sector"] = df.get("sub-index", df.get("industry", None))

    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe():
    """
    Hardcoded EURO STOXX 50 list (Yahoo Finance tickers).
    Note: Constituents can change over time; this list is meant to be stable/working,
    and the downloader tolerates failures (bad/missing tickers are skipped).
    """
    data = [
        ("ADS.DE",   "adidas", "Consumer Discretionary"),
        ("ADYEN.AS", "Adyen", "Information Technology"),
        ("AD.AS",    "Ahold Delhaize", "Consumer Staples"),
        ("AIR.PA",   "Airbus", "Industrials"),
        ("ALV.DE",   "Allianz", "Financials"),
        ("ABI.BR",   "Anheuser-Busch InBev", "Consumer Staples"),
        ("ASML.AS",  "ASML", "Information Technology"),
        ("CS.PA",    "AXA", "Financials"),
        ("BBVA.MC",  "BBVA", "Financials"),
        ("BAS.DE",   "BASF", "Materials"),
        ("BAYN.DE",  "Bayer", "Health Care"),
        ("BMW.DE",   "BMW", "Consumer Discretionary"),
        ("BNP.PA",   "BNP Paribas", "Financials"),
        ("BN.PA",    "Danone", "Consumer Staples"),
        ("DB1.DE",   "Deutsche Börse", "Financials"),
        ("DTE.DE",   "Deutsche Telekom", "Communication Services"),
        ("DHL.DE",   "DHL Group", "Industrials"),
        ("ENEL.MI",  "Enel", "Utilities"),
        ("ENI.MI",   "Eni", "Energy"),
        ("EL.PA",    "EssilorLuxottica", "Health Care"),
        ("RACE.MI",  "Ferrari", "Consumer Discretionary"),
        ("IBE.MC",   "Iberdrola", "Utilities"),
        ("IFX.DE",   "Infineon", "Information Technology"),
        ("INGA.AS",  "ING", "Financials"),
        ("ISP.MI",   "Intesa Sanpaolo", "Financials"),
        ("ITX.MC",   "Inditex", "Consumer Discretionary"),
        ("KER.PA",   "Kering", "Consumer Discretionary"),
        ("OR.PA",    "L'Oréal", "Consumer Staples"),
        ("MC.PA",    "LVMH", "Consumer Discretionary"),
        ("MBG.DE",   "Mercedes-Benz Group", "Consumer Discretionary"),
        ("MUV2.DE",  "Munich Re", "Financials"),
        ("NOKIA.HE", "Nokia", "Information Technology"),
        ("PRX.AS",   "Prosus", "Consumer Discretionary"),
        ("RI.PA",    "Pernod Ricard", "Consumer Staples"),
        ("PHIA.AS",  "Philips", "Health Care"),
        ("SAF.PA",   "Safran", "Industrials"),
        ("SAN.MC",   "Santander", "Financials"),
        ("SAN.PA",   "Sanofi", "Health Care"),
        ("SAP.DE",   "SAP", "Information Technology"),
        ("SU.PA",    "Schneider Electric", "Industrials"),
        ("SIE.DE",   "Siemens", "Industrials"),
        ("SGOB.PA",  "Saint-Gobain", "Industrials"),
        ("STLA.PA",  "Stellantis", "Consumer Discretionary"),
        ("TTE.PA",   "TotalEnergies", "Energy"),
        ("UCG.MI",   "UniCredit", "Financials"),
        ("VOW3.DE",  "Volkswagen", "Consumer Discretionary"),
        ("DG.PA",    "Vinci", "Industrials"),
        ("CRH.IR",   "CRH", "Materials"),
        ("ORA.PA",   "Orange", "Communication Services"),
    ]

    # Ensure exactly 50 rows (pad/trim defensively)
    df = pd.DataFrame(data, columns=["Ticker", "Name", "Sector"]).copy()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Sector"] = df["Sector"].astype(str).str.strip()

    # Drop blanks / duplicates (keep first)
    df = df[df["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    # If fewer than 50 due to duplicates, that's ok—the downloader will still work.
    return df[["Ticker", "Name", "Sector"]]


# ==========================================================
# 2. YAHOO DOWNLOADER (batched, robust)
# ==========================================================

def _extract_one_ticker_frame(data, ticker: str) -> pd.DataFrame:
    """
    yfinance returns either:
      - MultiIndex columns when requesting multiple tickers, accessed via data[ticker]
      - Single-index columns when requesting one ticker (Open/High/Low/Close/Adj Close/Volume)
    This normalizes both cases.
    """
    if data is None or getattr(data, "empty", True):
        return pd.DataFrame()

    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return pd.DataFrame()
        df_t = data[ticker].copy()
    else:
        # Single-ticker response
        df_t = data.copy()

    # Normalize column names
    df_t = df_t.rename(columns={c: str(c).strip().title() for c in df_t.columns})
    # Keep only OHLCV if present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df_t.columns]
    df_t = df_t[keep].dropna(how="all").copy()
    return df_t


def download_yahoo_prices(tickers, label, period="600d", interval="1d"):
    """
    Downloads last `period` of OHLC for a list of tickers.
    Uses yf.download in batches.
    Returns a list of clean DataFrames (each with columns: Date/Datetime + OHLC + Ticker + Index).
    """
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

        for t in batch:
            try:
                df_t = _extract_one_ticker_frame(data, t)
                if df_t.empty:
                    continue
                df_t = df_t.reset_index()
                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t)
            except Exception:
                continue

    return frames


# ==========================================================
# 3. MASTER FUNCTIONS (engine/app call these)
# ==========================================================

def load_all_market_data():
    """
    Returns a merged DAILY OHLC dataframe for SP500 + HSI + EURO STOXX 50
    without saving anything to disk.
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    euro = get_eurostoxx50_universe()

    sp_frames = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d", interval="1d")
    hs_frames = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d", interval="1d")
    eu_frames = download_yahoo_prices(euro["Ticker"].tolist(), "EUROSTOXX50", period="600d", interval="1d")

    if not (sp_frames or hs_frames or eu_frames):
        raise RuntimeError("No OHLC data downloaded from Yahoo")

    combined = pd.concat(sp_frames + hs_frames + eu_frames, ignore_index=True)

    # yfinance daily index name is usually "Date"
    if "Date" not in combined.columns:
        # sometimes it's "index" or unnamed
        idx_cols = [c for c in combined.columns if str(c).lower() in ("date", "index", "datetime")]
        if idx_cols:
            combined = combined.rename(columns={idx_cols[0]: "Date"})
        else:
            raise RuntimeError("Daily data missing a Date column")

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Standardize OHLC casing (engine expects Open/High/Low/Close)
    rename_map = {c: str(c).title() for c in combined.columns}
    combined = combined.rename(columns=rename_map)

    # Ensure essential columns exist
    required = {"Ticker", "Date", "Open", "High", "Low", "Close", "Index"}
    missing = required.difference(set(combined.columns))
    if missing:
        raise RuntimeError(f"Daily data missing required columns: {sorted(missing)}")

    return combined


def load_all_market_data_hourly():
    """
    Returns merged HOURLY OHLC dataframe for SP500 + HSI + EURO STOXX 50.
    Output columns include: Ticker, DateTime, Open, High, Low, Close, Index.
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    euro = get_eurostoxx50_universe()

    sp_frames = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="60d", interval="60m")
    hs_frames = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="60d", interval="60m")
    eu_frames = download_yahoo_prices(euro["Ticker"].tolist(), "EUROSTOXX50", period="60d", interval="60m")

    if not (sp_frames or hs_frames or eu_frames):
        raise RuntimeError("No hourly OHLC data downloaded from Yahoo")

    combined = pd.concat(sp_frames + hs_frames + eu_frames, ignore_index=True)

    # yfinance intraday index name is usually "Datetime"
    dt_col = None
    for cand in ["Datetime", "DateTime", "Date", "Index"]:
        if cand in combined.columns:
            dt_col = cand
            break
    if dt_col is None:
        raise RuntimeError("Hourly data missing Datetime/Date column")

    combined = combined.rename(columns={dt_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["DateTime"]).copy()
    combined["DateTime"] = combined["DateTime"].dt.tz_convert(None)

    # Standardize OHLC casing
    rename_map = {c: str(c).title() for c in combined.columns}
    combined = combined.rename(columns=rename_map)

    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "Datetime"}  # after title-case, DateTime -> Datetime
    # Fix: title-case of "DateTime" becomes "Datetime"
    if "Datetime" not in combined.columns and "DateTime" in combined.columns:
        combined = combined.rename(columns={"DateTime": "Datetime"})

    # Now enforce required set with final names
    required_final = {"Ticker", "Datetime", "Open", "High", "Low", "Close", "Index"}
    missing = required_final.difference(set(combined.columns))
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    # Engine expects "DateTime" specifically
    combined = combined.rename(columns={"Datetime": "DateTime"})

    return combined
