"""
updater.py  (simple, Streamlit-friendly version)

✓ Builds SP500, HSI, EURO STOXX 50 universes
✓ Downloads daily + hourly OHLC from Yahoo (yfinance)
✓ Batches requests with yf.download (no manual threads)
✓ Retries on rate-limit / transient errors
✓ Returns ONE CLEAN MERGED DATAFRAME
✓ No parquet saving, no disk IO
"""

from __future__ import annotations

import time
from io import StringIO
from typing import Iterable, List, Optional

import pandas as pd
import requests
import yfinance as yf


# ==========================================================
# 0. SMALL HELPERS
# ==========================================================

def _chunks(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _normalize_yf_download_output(data: pd.DataFrame, tickers: List[str]) -> dict:
    """
    yfinance returns:
      - MultiIndex columns when multiple tickers
      - Single-index columns when one ticker
    Normalize to: {ticker: df_ticker}
    """
    out = {}

    if data is None or getattr(data, "empty", True):
        return out

    # Single ticker => columns like Open/High/Low/Close/Adj Close/Volume
    if not isinstance(data.columns, pd.MultiIndex):
        # Infer the only ticker
        if len(tickers) == 1:
            out[tickers[0]] = data
        else:
            # Unexpected shape; best effort: treat as empty
            return {}
        return out

    # Multi ticker => first level is field or ticker depending on group_by
    # With group_by="ticker": columns are (TICKER, Field)
    # So data[ticker] works.
    for t in tickers:
        try:
            out[t] = data[t]
        except Exception:
            continue

    return out


def _download_with_retries(
    tickers: List[str],
    period: str,
    interval: str,
    max_retries: int = 4,
    base_sleep: float = 1.5,
) -> pd.DataFrame:
    """
    Safer yfinance download:
    - threads=False to reduce rate-limit spikes
    - retries with exponential backoff
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return yf.download(
                tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=False,     # IMPORTANT: reduce 429/rate-limit issues
                progress=False,
            )
        except Exception as e:
            last_exc = e
            # exponential backoff
            sleep_s = base_sleep * (2 ** attempt)
            time.sleep(sleep_s)

    # If we got here, all retries failed
    raise last_exc


# ==========================================================
# 1. UNIVERSE BUILDERS
# ==========================================================

def get_sp500_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break
    if df is None:
        raise RuntimeError("Could not find S&P500 table on Wikipedia")

    df["Ticker"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
    df["Name"] = df["Security"].astype(str).str.strip()
    df["Sector"] = df["GICS Sector"].astype(str).str.strip()
    return df[["Ticker", "Name", "Sector"]]


def get_hsi_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index", "code"]):
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

    df["Name"] = df[name_col].astype(str).str.strip()
    df["Sector"] = df.get("sub-index", df.get("industry", "")).astype(str)
    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe() -> pd.DataFrame:
    """
    Hardcoded EURO STOXX 50 constituents (Yahoo-friendly tickers).
    This avoids scraping changes + fixes the "no Europe in output" problem.
    """
    data = [
        ("ADS.DE", "adidas", "Retail"),
        ("ADYEN.AS", "Adyen", "Technology"),
        ("AD.AS", "Ahold Delhaize", "Consumer staples"),
        ("AI.PA", "Air Liquide", "Basic materials"),
        ("AIR.PA", "Airbus", "Industrials"),
        ("ALV.DE", "Allianz", "Financials"),
        ("ABI.BR", "Anheuser-Busch InBev", "Consumer defensive"),
        ("ARGX.BR", "argenx", "Healthcare"),
        ("ASML.AS", "ASML Holding", "Technology"),
        ("CS.PA", "AXA", "Financials"),
        ("BAS.DE", "BASF", "Basic materials"),
        ("BAYN.DE", "Bayer", "Healthcare"),
        ("BBVA.MC", "BBVA", "Financials"),
        ("SAN.MC", "Banco Santander", "Financials"),
        ("BMW.DE", "BMW", "Consumer cyclical"),
        ("BNP.PA", "BNP Paribas", "Financials"),
        ("BN.PA", "Danone", "Consumer defensive"),
        ("DBK.DE", "Deutsche Bank", "Financials"),
        ("DB1.DE", "Deutsche Börse", "Financials"),
        ("DHL.DE", "DHL Group", "Industrials"),
        ("DTE.DE", "Deutsche Telekom", "Communication services"),
        ("ENEL.MI", "Enel", "Utilities"),
        ("ENI.MI", "Eni", "Energy"),
        ("EL.PA", "EssilorLuxottica", "Healthcare"),
        ("RACE.MI", "Ferrari", "Consumer cyclical"),
        ("RMS.PA", "Hermès", "Consumer cyclical"),
        ("IBE.MC", "Iberdrola", "Utilities"),
        ("ITX.MC", "Inditex", "Consumer cyclical"),
        ("IFX.DE", "Infineon", "Technology"),
        ("INGA.AS", "ING Groep", "Financials"),
        ("ISP.MI", "Intesa Sanpaolo", "Financials"),
        ("OR.PA", "L'Oréal", "Consumer defensive"),
        ("MC.PA", "LVMH", "Consumer cyclical"),
        ("MBG.DE", "Mercedes-Benz Group", "Consumer cyclical"),
        ("MUV2.DE", "Munich Re", "Financials"),
        ("NDA-FI.HE", "Nordea Bank", "Financials"),
        ("PRX.AS", "Prosus", "Consumer cyclical"),
        ("RHM.DE", "Rheinmetall", "Industrials"),
        ("SAF.PA", "Safran", "Industrials"),
        ("SGO.PA", "Saint-Gobain", "Industrials"),
        ("SAN.PA", "Sanofi", "Healthcare"),
        ("SAP.DE", "SAP", "Technology"),
        ("SU.PA", "Schneider Electric", "Industrials"),
        ("SIE.DE", "Siemens", "Industrials"),
        ("ENR.DE", "Siemens Energy", "Industrials"),
        ("TTE.PA", "TotalEnergies", "Energy"),
        ("DG.PA", "Vinci", "Industrials"),
        ("UCG.MI", "UniCredit", "Financials"),
        ("VOW.DE", "Volkswagen", "Consumer cyclical"),
        ("WKL.AS", "Wolters Kluwer", "Industrials"),
    ]
    return pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])


# ==========================================================
# 2. YAHOO DOWNLOADER (batched)
# ==========================================================

def download_yahoo_prices(
    tickers: Iterable[str],
    label: str,
    period: str = "600d",
    interval: str = "1d",
    batch_size: int = 25,
) -> List[pd.DataFrame]:
    """
    Returns list of per-ticker OHLC frames with columns:
      Date/Datetime index -> reset_index() -> Date/Datetime column
      + Ticker + Index label
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    frames: List[pd.DataFrame] = []

    for batch in _chunks(tickers, batch_size):
        try:
            data = _download_with_retries(batch, period=period, interval=interval)
        except Exception:
            continue

        per_ticker = _normalize_yf_download_output(data, batch)

        for t in batch:
            df_t = per_ticker.get(t)
            if df_t is None or df_t.empty:
                continue

            # Standardize
            df_t = df_t.dropna().copy()
            df_t["Ticker"] = t
            df_t["Index"] = label
            frames.append(df_t.reset_index())

        # small pause between batches to reduce throttling
        time.sleep(0.35)

    return frames


# ==========================================================
# 3. MASTER FUNCTIONS
# ==========================================================

def load_all_market_data() -> pd.DataFrame:
    """
    Returns merged DAILY OHLC dataframe for SP500 + HSI + EURO STOXX 50.
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    euro = get_eurostoxx50_universe()

    sp_frames = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d", interval="1d")
    hs_frames = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d", interval="1d")
    eu_frames = download_yahoo_prices(euro["Ticker"].tolist(), "EUROSTOXX50", period="600d", interval="1d")

    if not (sp_frames or hs_frames or eu_frames):
        raise RuntimeError("No DAILY OHLC data downloaded from Yahoo")

    combined = pd.concat(sp_frames + hs_frames + eu_frames, ignore_index=True)

    # yfinance daily uses 'Date'
    if "Date" not in combined.columns:
        # fallback if reset_index produced something else
        dt_col = "Datetime" if "Datetime" in combined.columns else None
        if dt_col is None:
            raise RuntimeError("Daily data missing Date column after download")
        combined = combined.rename(columns={dt_col: "Date"})

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return combined


def load_all_market_data_hourly(tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Returns merged HOURLY OHLC dataframe for SP500 + HSI + EURO STOXX 50.

    KEY FIX:
    - If you pass tickers=[...] it only downloads hourly for those names,
      preventing yfinance rate-limits and making your hourly entries work.
    """
    if tickers is None:
        # Default: full universes (NOT recommended; likely to rate-limit)
        sp500 = get_sp500_universe()
        hsi = get_hsi_universe()
        euro = get_eurostoxx50_universe()
        tickers_all = (
            sp500["Ticker"].tolist()
            + hsi["Ticker"].tolist()
            + euro["Ticker"].tolist()
        )
    else:
        tickers_all = [str(t).strip() for t in tickers if str(t).strip()]

    # Hourly: keep it smaller to reduce throttling
    frames = download_yahoo_prices(tickers_all, "MIXED", period="60d", interval="60m", batch_size=10)
    if not frames:
        raise RuntimeError("No HOURLY OHLC data downloaded from Yahoo (rate-limit or invalid tickers)")

    combined = pd.concat(frames, ignore_index=True)

    # yfinance hourly uses Datetime in many cases
    if "Datetime" in combined.columns:
        combined = combined.rename(columns={"Datetime": "DateTime"})
    elif "Date" in combined.columns:
        combined = combined.rename(columns={"Date": "DateTime"})
    else:
        raise RuntimeError("Hourly data missing Datetime/Date column")

    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
    combined = combined.dropna(subset=["DateTime"])
    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(combined.columns)
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    return combined
