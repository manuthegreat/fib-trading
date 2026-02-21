"""
updater.py  (simple, Streamlit-friendly version)

✓ Builds SP500, HSI, EURO STOXX 50 universes
✓ Downloads daily OHLC (600d) from Yahoo in batches
✓ Hourly data is PROVIDED VIA helper that downloads ONLY the tickers you pass in
  (critical to avoid Yahoo rate limits)
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

    # Name + sector are messy on wiki; keep best-effort
    name_col = "name" if "name" in df.columns else None
    if name_col is None:
        candidates = [c for c in df.columns if c != ticker_col]
        name_col = candidates[0] if candidates else ticker_col

    df["Name"] = df[name_col].astype(str).str.strip()
    df["Sector"] = df.get("sub-index", df.get("industry", ""))

    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe() -> pd.DataFrame:
    """
    Hardcoded EURO STOXX 50 tickers in Yahoo format.
    NOTE: Constituents change over time; if STOXX rebalances, update this list.
    """
    data = [
        # Germany
        ("ADS.DE", "Adidas", "Consumer Discretionary"),
        ("ALV.DE", "Allianz", "Financials"),
        ("BAS.DE", "BASF", "Materials"),
        ("BAYN.DE", "Bayer", "Health Care"),
        ("BMW.DE", "BMW", "Consumer Discretionary"),
        ("DB1.DE", "Deutsche Börse", "Financials"),
        ("DBK.DE", "Deutsche Bank", "Financials"),
        ("DHL.DE", "Deutsche Post", "Industrials"),
        ("DTE.DE", "Deutsche Telekom", "Communication"),
        ("IFX.DE", "Infineon Technologies", "Information Technology"),
        ("MBG.DE", "Mercedes-Benz Group", "Consumer Discretionary"),
        ("MUV2.DE", "Munich Re", "Financials"),
        ("RWE.DE", "RWE", "Utilities"),
        ("SAP.DE", "SAP", "Information Technology"),
        ("SIE.DE", "Siemens", "Industrials"),
        ("ENR.DE", "Siemens Energy", "Industrials"),
        ("VOW3.DE", "Volkswagen Group (Pref)", "Consumer Discretionary"),

        # France
        ("AI.PA", "Air Liquide", "Materials"),
        ("AIR.PA", "Airbus", "Industrials"),
        ("BNP.PA", "BNP Paribas", "Financials"),
        ("CS.PA", "AXA", "Financials"),
        ("DG.PA", "Vinci", "Industrials"),
        ("EL.PA", "EssilorLuxottica", "Health Care"),
        ("MC.PA", "LVMH", "Consumer Discretionary"),
        ("OR.PA", "L'Oréal", "Consumer Staples"),
        ("RMS.PA", "Hermès", "Consumer Discretionary"),
        ("SAF.PA", "Safran", "Industrials"),
        ("SAN.PA", "Sanofi", "Health Care"),
        ("SGO.PA", "Saint-Gobain", "Industrials"),
        ("SU.PA", "Schneider Electric", "Industrials"),
        ("TTE.PA", "TotalEnergies", "Energy"),
        ("VIV.PA", "Vivendi", "Communication"),

        # Netherlands
        ("AD.AS", "Ahold Delhaize", "Consumer Staples"),
        ("ADYEN.AS", "Adyen", "Financials"),
        ("ASML.AS", "ASML Holding", "Information Technology"),
        ("HEIA.AS", "Heineken", "Consumer Staples"),
        ("INGA.AS", "ING Group", "Financials"),
        ("PHIA.AS", "Philips", "Health Care"),
        ("PRX.AS", "Prosus", "Consumer Discretionary"),
        ("WKL.AS", "Wolters Kluwer", "Industrials"),

        # Spain
        ("BBVA.MC", "BBVA", "Financials"),
        ("IBE.MC", "Iberdrola", "Utilities"),
        ("ITX.MC", "Inditex", "Consumer Discretionary"),
        ("SAN.MC", "Banco Santander", "Financials"),

        # Italy
        ("ENEL.MI", "Enel", "Utilities"),
        ("ENI.MI", "Eni", "Energy"),
        ("ISP.MI", "Intesa Sanpaolo", "Financials"),
        ("UCG.MI", "UniCredit", "Financials"),
        ("STLAM.MI", "Stellantis", "Consumer Discretionary"),

        # Belgium
        ("ABI.BR", "Anheuser-Busch InBev", "Consumer Staples"),

        # Finland
        ("NDA-FI.HE", "Nordea", "Financials"),
    ]

    df = pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])
    # Ensure uniqueness + non-empty
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df


# ==========================================================
# 2. YAHOO DOWNLOADER (batched + retries)
# ==========================================================

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def download_yahoo_prices(
    tickers: List[str],
    label: str,
    period: str = "600d",
    interval: str = "1d",
    batch_size: int = 15,
    max_retries: int = 3,
    sleep_s: float = 1.0,
) -> List[pd.DataFrame]:
    """
    Downloads OHLC for a list of tickers.
    Uses small batches + retries to reduce YF rate-limits.
    Returns list of per-ticker DataFrames (with Date/Datetime index reset).
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    frames: List[pd.DataFrame] = []

    for batch in _chunk(tickers, batch_size):
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=False,        # IMPORTANT: threads=True triggers rate limits faster
                    progress=False,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                # Exponential-ish backoff
                time.sleep(sleep_s * attempt)

        if last_exc is not None:
            # Skip this batch; we keep going so the app still works.
            continue

        # yfinance output format differs for 1 ticker vs many tickers
        for t in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    df_t = data[t].dropna().copy()
                else:
                    # single-ticker download returns flat columns
                    df_t = data.dropna().copy()

                if df_t.empty:
                    continue

                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t.reset_index())
            except Exception:
                continue

        time.sleep(sleep_s)

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

    # Standardize datetime column name
    date_col = "Date" if "Date" in combined.columns else "Datetime"
    combined = combined.rename(columns={date_col: "Date"})
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])

    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return combined


def load_hourly_prices_for_tickers(tickers: Iterable[str]) -> pd.DataFrame:
    """
    Returns merged HOURLY OHLC dataframe ONLY for the tickers you pass in.
    This is the only safe way to avoid rate limits and actually get hourly entries.
    """
    tickers = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "DateTime", "Open", "High", "Low", "Close", "Index"])

    frames = download_yahoo_prices(
        tickers,
        label="HOURLY",
        period="60d",
        interval="60m",
        batch_size=10,
        max_retries=4,
        sleep_s=1.25,
    )

    if not frames:
        return pd.DataFrame(columns=["Ticker", "DateTime", "Open", "High", "Low", "Close", "Index"])

    combined = pd.concat(frames, ignore_index=True)

    dt_col = "Datetime" if "Datetime" in combined.columns else "Date"
    combined = combined.rename(columns={dt_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
    combined = combined.dropna(subset=["DateTime"])
    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(combined.columns)
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    return combined
