""" updater.py (Streamlit-friendly, Yahoo-only)
Rules:
1) DON'T change anything for SP500 and HSI (same universe builders + same batched downloader usage)
2) Remove Euro Stoxx 50 support
3) Yahoo-only
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
# 1) UNIVERSE BUILDERS (UNCHANGED)
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
        df[ticker_col].astype(str).str.extract(r"(\d+)", expand=False).astype(str).str.zfill(4) + ".HK"
    )

    name_col = "name" if "name" in df.columns else None
    if name_col is None:
        candidates = [c for c in df.columns if c != ticker_col]
        name_col = candidates[0] if candidates else ticker_col

    df["Name"] = df[name_col].astype(str).str.strip()
    df["Sector"] = df.get("sub-index", df.get("industry", ""))
    return df[["Ticker", "Name", "Sector"]]


# ==========================================================
# 2) YAHOO DOWNLOADER (SP500/HSI batched)
# ==========================================================
def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that have valid OHLC.
    Avoid dropping because Volume/Adj Close is missing.
    Also enforce basic OHLC sanity.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()

    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    bad = (df["High"] < df[["Open", "Close"]].max(axis=1)) | (df["Low"] > df[["Open", "Close"]].min(axis=1))
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
    """Single-name download with robust retries."""
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
            if data.empty:
                time.sleep(sleep_s * attempt + random.random() * 0.35)
                continue
            return data
        except Exception as exc:
            last_exc = exc
            time.sleep(sleep_s * attempt + random.random() * 0.35)

    if last_exc is not None:
        print(f"[WARN] Yahoo single download failed for {ticker}: {last_exc}")
    else:
        print(f"[WARN] Yahoo single download returned no data for {ticker}")
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
    Batched downloader (used for SP500 and HSI).
    Same shape as original design; batch + per-ticker fallback.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    frames: List[pd.DataFrame] = []
    failed_tickers: List[str] = []

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
                    threads=False,
                    progress=False,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(sleep_s * attempt + random.random() * 0.25)

        if last_exc is not None:
            print(f"[WARN] Yahoo batch download failed ({label}) for {batch}: {last_exc}")
            data = pd.DataFrame()

        found_in_batch = set()
        for t in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.get_level_values(0):
                        continue
                    df_t = _clean_ohlc(data[t])
                else:
                    if len(batch) != 1:
                        continue
                    df_t = _clean_ohlc(data)

                if df_t.empty:
                    continue

                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t.reset_index())
                found_in_batch.add(t)
            except Exception:
                continue

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
            df_t["Index"] = label
            frames.append(df_t.reset_index())

        time.sleep(sleep_s + random.random() * 0.25)

    if failed_tickers:
        print(
            f"[WARN] Yahoo returned no usable {interval} data for {len(failed_tickers)} "
            f"{label} tickers: {', '.join(sorted(set(failed_tickers))[:20])}"
            + (" ..." if len(set(failed_tickers)) > 20 else "")
        )

    return frames


# ==========================================================
# 3) MASTER FUNCTIONS
# ==========================================================
def load_all_market_data() -> pd.DataFrame:
    """
    Returns merged DAILY OHLC dataframe for SP500 + HSI.
    - SP500/HSI: batched downloader (same call pattern)
    - No Euro Stoxx 50
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()

    sp_frames = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d", interval="1d")
    hs_frames = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d", interval="1d")

    if not (sp_frames or hs_frames):
        raise RuntimeError("No DAILY OHLC data downloaded from Yahoo")

    combined = pd.concat(sp_frames + hs_frames, ignore_index=True)

    date_col = "Date" if "Date" in combined.columns else "Datetime"
    combined = combined.rename(columns={date_col: "Date"})
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return combined


def load_hourly_prices_for_tickers(tickers: Iterable[str]) -> pd.DataFrame:
    """Returns merged HOURLY OHLC dataframe ONLY for the tickers you pass in."""
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


def compute_dashboard() -> pd.DataFrame:
    """Entry point used by Streamlit app."""
    return load_all_market_data()
