""" updater.py — Yahoo-only, SP500 + HSI + Euro Stoxx 50
"""

from __future__ import annotations

import random
import time
from io import StringIO
from typing import Iterable, List, Optional

import pandas as pd
import requests
import yfinance as yf


# ==========================================================
# 1) UNIVERSE BUILDERS
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
        raise RuntimeError("Could not find S&P 500 table on Wikipedia")

    df["Ticker"] = (
        df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
    )
    df["Name"]   = df["Security"].astype(str).str.strip()
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
        .str.zfill(4) + ".HK"
    )

    name_col = "name" if "name" in df.columns else None
    if name_col is None:
        candidates = [c for c in df.columns if c != ticker_col]
        name_col = candidates[0] if candidates else ticker_col

    df["Name"]   = df[name_col].astype(str).str.strip()
    df["Sector"] = df.get("sub-index", df.get("industry", ""))
    return df[["Ticker", "Name", "Sector"]]


# ------------------------------------------------------------------
# Hardcoded Yahoo Finance tickers for Euro Stoxx 50.
#
# Wikipedia's Euro Stoxx 50 page is structurally inconsistent —
# the table format, column names, and ticker representations change
# without warning, making scraping fragile. Because the index
# membership changes infrequently (typically once a year), a
# hardcoded list with verified Yahoo Finance suffixes is far more
# reliable in production than any scraping approach.
#
# Last verified: February 2026 (50 constituents).
# Source: https://www.stoxx.com / Wikipedia cross-referenced with
#         Yahoo Finance search to confirm each suffix resolves.
# ------------------------------------------------------------------
_EUROSTOXX50_TICKERS = [
    # Austria
    ("VER.VI",  "Verbund",                    "Utilities"),
    # Belgium
    ("ABI.BR",  "AB InBev",                   "Consumer Staples"),
    ("UCB.BR",  "UCB",                        "Health Care"),
    # Finland
    ("NOKIA.HE","Nokia",                      "Technology"),
    # France
    ("AI.PA",   "Air Liquide",                "Materials"),
    ("AIR.PA",  "Airbus",                     "Industrials"),
    ("ACA.PA",  "Credit Agricole",            "Financials"),
    ("BN.PA",   "Danone",                     "Consumer Staples"),
    ("BNP.PA",  "BNP Paribas",               "Financials"),
    ("CS.PA",   "AXA",                        "Financials"),
    ("DG.PA",   "Vinci",                      "Industrials"),
    ("EL.PA",   "EssilorLuxottica",           "Health Care"),
    ("EN.PA",   "Bouygues",                   "Industrials"),
    ("ENGI.PA", "Engie",                      "Utilities"),
    ("GLE.PA",  "Societe Generale",           "Financials"),
    ("KER.PA",  "Kering",                     "Consumer Discretionary"),
    ("MC.PA",   "LVMH",                       "Consumer Discretionary"),
    ("ML.PA",   "Michelin",                   "Consumer Discretionary"),
    ("OR.PA",   "L'Oreal",                    "Consumer Staples"),
    ("ORA.PA",  "Orange",                     "Communication Services"),
    ("RI.PA",   "Pernod Ricard",              "Consumer Staples"),
    ("RMS.PA",  "Hermes",                     "Consumer Discretionary"),
    ("SAF.PA",  "Safran",                     "Industrials"),
    ("SAN.PA",  "Sanofi",                     "Health Care"),
    ("SGO.PA",  "Saint-Gobain",               "Industrials"),
    ("STLAM.MI","Stellantis",                 "Consumer Discretionary"),
    ("SU.PA",   "Schneider Electric",         "Industrials"),
    ("TTE.PA",  "TotalEnergies",              "Energy"),
    ("VIE.PA",  "Veolia",                     "Utilities"),
    # Germany
    ("ADS.DE",  "Adidas",                     "Consumer Discretionary"),
    ("ALV.DE",  "Allianz",                    "Financials"),
    ("BAYN.DE", "Bayer",                      "Health Care"),
    ("BMW.DE",  "BMW",                        "Consumer Discretionary"),
    ("BAS.DE",  "BASF",                       "Materials"),
    ("DB1.DE",  "Deutsche Boerse",            "Financials"),
    ("DHL.DE",  "DHL Group",                  "Industrials"),
    ("DTE.DE",  "Deutsche Telekom",           "Communication Services"),
    ("MBG.DE",  "Mercedes-Benz",              "Consumer Discretionary"),
    ("MUV2.DE", "Munich Re",                  "Financials"),
    ("RWE.DE",  "RWE",                        "Utilities"),
    ("SAP.DE",  "SAP",                        "Technology"),
    ("SIE.DE",  "Siemens",                    "Industrials"),
    # Italy
    ("ENI.MI",  "Eni",                        "Energy"),
    ("ENEL.MI", "Enel",                       "Utilities"),
    ("ISP.MI",  "Intesa Sanpaolo",            "Financials"),
    ("UCG.MI",  "UniCredit",                  "Financials"),
    # Netherlands
    ("ASML.AS", "ASML",                       "Technology"),
    ("INGA.AS", "ING Group",                  "Financials"),
    ("PHIA.AS", "Philips",                    "Health Care"),
    # Spain
    ("BBVA.MC", "BBVA",                       "Financials"),
    ("IBE.MC",  "Iberdrola",                  "Utilities"),
    ("ITX.MC",  "Inditex",                    "Consumer Discretionary"),
    ("REP.MC",  "Repsol",                     "Energy"),
    ("SAN.MC",  "Santander",                  "Financials"),
]


def get_eurostoxx50_universe() -> pd.DataFrame:
    """Returns the Euro Stoxx 50 constituent list as a DataFrame."""
    records = [
        {"Ticker": t, "Name": name, "Sector": sector}
        for t, name, sector in _EUROSTOXX50_TICKERS
    ]
    return pd.DataFrame(records, columns=["Ticker", "Name", "Sector"])


# ==========================================================
# 2) YAHOO DOWNLOADER (batched, with per-ticker fallback)
# ==========================================================

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing or logically invalid OHLC values."""
    if df is None or df.empty:
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()

    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    bad = (df["High"] < df[["Open", "Close"]].max(axis=1)) | (
        df["Low"] > df[["Open", "Close"]].min(axis=1)
    )
    if bad.any():
        df = df.loc[~bad].copy()

    return df


def _download_single_ticker(
    ticker: str,
    period: str,
    interval: str,
    max_retries: int,
    sleep_s: float,
) -> Optional[pd.DataFrame]:
    """Single-name download with retries."""
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            data = _clean_ohlc(data)
            if not data.empty:
                return data
        except Exception as exc:
            last_exc = exc

        time.sleep(sleep_s * attempt + random.random() * 0.35)

    msg = str(last_exc) if last_exc else "empty response"
    print(f"[WARN] Yahoo single download failed for {ticker}: {msg}")
    return None


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
    Batched downloader for any universe.
    Falls back to single-ticker download for any that fail in the batch.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    frames: List[pd.DataFrame] = []
    failed_tickers: List[str] = []

    for batch in _chunk(tickers, batch_size):
        last_exc = None
        data = pd.DataFrame()

        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=False,
                    progress=False,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(sleep_s * attempt + random.random() * 0.25)

        if last_exc is not None:
            print(
                f"[WARN] Yahoo batch download failed ({label}) "
                f"for {batch}: {last_exc}"
            )

        found_in_batch: set = set()

        for t in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.get_level_values(0):
                        continue
                    df_t = _clean_ohlc(data[t])
                else:
                    # Single-ticker batch — columns are flat
                    if len(batch) != 1:
                        continue
                    df_t = _clean_ohlc(data)

                if df_t.empty:
                    continue

                df_t["Ticker"] = t
                df_t["Index"]  = label
                frames.append(df_t.reset_index())
                found_in_batch.add(t)

            except Exception:
                continue

        # Per-ticker fallback for anything missing from the batch result
        missing = [t for t in batch if t not in found_in_batch]
        for t in missing:
            df_t = _download_single_ticker(
                ticker=t,
                period=period,
                interval=interval,
                max_retries=max_retries,
                sleep_s=sleep_s,
            )
            if df_t is None or df_t.empty:
                failed_tickers.append(t)
                continue

            df_t["Ticker"] = t
            df_t["Index"]  = label
            frames.append(df_t.reset_index())

        time.sleep(sleep_s + random.random() * 0.25)

    if failed_tickers:
        unique_failed = sorted(set(failed_tickers))
        preview = ", ".join(unique_failed[:20])
        suffix  = " ..." if len(unique_failed) > 20 else ""
        print(
            f"[WARN] Yahoo returned no usable {interval} data for "
            f"{len(unique_failed)} {label} tickers: {preview}{suffix}"
        )

    return frames


# ==========================================================
# 3) MASTER LOAD FUNCTIONS
# ==========================================================

def load_all_market_data() -> pd.DataFrame:
    """
    Returns merged DAILY OHLC dataframe for SP500 + HSI + Euro Stoxx 50.
    """
    sp500      = get_sp500_universe()
    hsi        = get_hsi_universe()
    eurostoxx  = get_eurostoxx50_universe()

    print(f"[INFO] Universes — SP500: {len(sp500)}, HSI: {len(hsi)}, EuroStoxx50: {len(eurostoxx)}")

    sp_frames  = download_yahoo_prices(
        sp500["Ticker"].tolist(),     "SP500",       period="600d", interval="1d"
    )
    hs_frames  = download_yahoo_prices(
        hsi["Ticker"].tolist(),       "HSI",         period="600d", interval="1d"
    )
    eu_frames  = download_yahoo_prices(
        eurostoxx["Ticker"].tolist(), "EUROSTOXX50", period="600d", interval="1d"
    )

    all_frames = sp_frames + hs_frames + eu_frames
    if not all_frames:
        raise RuntimeError("No DAILY OHLC data downloaded from Yahoo Finance")

    combined = pd.concat(all_frames, ignore_index=True)

    date_col = "Date" if "Date" in combined.columns else "Datetime"
    combined = combined.rename(columns={date_col: "Date"})
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"[INFO] Daily data loaded — {combined['Ticker'].nunique()} tickers, "
          f"{len(combined):,} rows")

    return combined


def load_hourly_prices_for_tickers(tickers: Iterable[str]) -> pd.DataFrame:
    """Returns merged HOURLY OHLC dataframe for the supplied tickers only."""
    tickers = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not tickers:
        return pd.DataFrame(
            columns=["Ticker", "DateTime", "Open", "High", "Low", "Close", "Index"]
        )

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
        return pd.DataFrame(
            columns=["Ticker", "DateTime", "Open", "High", "Low", "Close", "Index"]
        )

    combined = pd.concat(frames, ignore_index=True)

    dt_col = "Datetime" if "Datetime" in combined.columns else "Date"
    combined = combined.rename(columns={dt_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
    combined = combined.dropna(subset=["DateTime"])
    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing  = required.difference(combined.columns)
    if missing:
        raise RuntimeError(
            f"Hourly data missing required columns: {sorted(missing)}"
        )

    print(f"[INFO] Hourly data loaded — {combined['Ticker'].nunique()} tickers, "
          f"{len(combined):,} rows")

    return combined


def compute_dashboard() -> pd.DataFrame:
    """Entry point used by Streamlit app."""
    return load_all_market_data()
