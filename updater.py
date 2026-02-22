"""
updater.py
Yahoo-only downloader
SP500 + HSI unchanged
EU uses fast batched Yahoo approach (365d)
No stooq
"""

from __future__ import annotations

from io import StringIO
from typing import List

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
        raise RuntimeError("Could not find S&P500 table")

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
        if any(x in cols for x in ["ticker", "code"]):
            df = t.copy()
            break
    if df is None:
        raise RuntimeError("Could not find HSI table")

    df.columns = [str(c).lower() for c in df.columns]

    ticker_col = None
    for c in df.columns:
        if "code" in c or "ticker" in c:
            ticker_col = c
            break
    if ticker_col is None:
        raise RuntimeError("HSI ticker column not found")

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

    name_col = df.columns[0]
    df["Name"] = df[name_col].astype(str).str.strip()
    df["Sector"] = ""

    return df[["Ticker", "Name", "Sector"]]


def get_eurostoxx50_universe() -> pd.DataFrame:
    """
    Static list for stability.
    """
    tickers = [
        "ASML.AS","ADYEN.AS","AIR.PA","ALV.DE","ABI.BR","BBVA.MC","BAS.DE",
        "BAYN.DE","BMW.DE","BNP.PA","CRG.IR","CS.PA","DAI.DE","DG.PA",
        "DPW.DE","ENEL.MI","ENGI.PA","ENI.MI","IBE.MC","IFX.DE",
        "ITX.MC","KER.PA","LIN.DE","MC.PA","MUV2.DE","NOKIA.HE",
        "OR.PA","PHIA.AS","RI.PA","RMS.PA","SAN.PA","SAN.MC",
        "SAP.DE","SIE.DE","SU.PA","TTE.PA","VOW3.DE","VNA.DE",
        "BN.PA","ISP.MI","GLE.PA","AI.PA","PRX.AS","UCG.MI",
        "DTE.DE","ADS.DE","HEIA.AS","AMS.MC","STLAM.MI"
    ]

    return pd.DataFrame({
        "Ticker": tickers,
        "Name": tickers,
        "Sector": ""
    })


# ==========================================================
# 2) DOWNLOADERS
# ==========================================================

def download_batch(
    tickers: List[str],
    label: str,
    period: str,
    interval: str,
    batch_size: int = 25,
) -> List[pd.DataFrame]:

    frames: List[pd.DataFrame] = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

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

            if data.empty:
                continue

            for t in batch:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if t not in data.columns.get_level_values(0):
                            continue
                        df_t = data[t]
                    else:
                        df_t = data

                    df_t = df_t.dropna(subset=["Open", "High", "Low", "Close"])
                    if df_t.empty:
                        continue

                    df_t = df_t.copy()
                    df_t["Ticker"] = t
                    df_t["Index"] = label
                    frames.append(df_t.reset_index())

                except Exception:
                    continue

        except Exception:
            continue

    return frames


# ==========================================================
# 3) MASTER LOADER
# ==========================================================

def load_all_market_data() -> pd.DataFrame:

    # ---- SP500 (UNCHANGED) ----
    sp500 = get_sp500_universe()
    sp_frames = download_batch(
        sp500["Ticker"].tolist(),
        label="SP500",
        period="600d",
        interval="1d",
        batch_size=25,
    )

    # ---- HSI (UNCHANGED) ----
    hsi = get_hsi_universe()
    hsi_frames = download_batch(
        hsi["Ticker"].tolist(),
        label="HSI",
        period="600d",
        interval="1d",
        batch_size=20,
    )

    # ---- EU (MODIFIED AS REQUESTED) ----
    euro = get_eurostoxx50_universe()
    eu_frames = download_batch(
        euro["Ticker"].tolist(),
        label="EUROSTOXX50",
        period="365d",        # <-- changed to 365
        interval="1d",
        batch_size=10,        # smaller batch for EU stability
    )

    all_frames = sp_frames + hsi_frames + eu_frames

    if not all_frames:
        raise RuntimeError("No data downloaded from Yahoo")

    final_df = pd.concat(all_frames, ignore_index=True)

    final_df = final_df.sort_values(["Ticker", "Date"])
    final_df.reset_index(drop=True, inplace=True)

    return final_df


# ==========================================================
# 4) STREAMLIT ENTRYPOINT
# ==========================================================

def compute_dashboard():
    """
    Entry point used by Streamlit app.
    """
    df = load_all_market_data()
    return df
