"""
updater.py  (simple, Streamlit-friendly version)

✓ Builds SP500, HSI, EURO STOXX 50 universes
✓ Downloads last 600 days of OHLC from Yahoo
✓ Batches requests with yf.download (no manual threads)
✓ Robust parsing of yfinance outputs (MultiIndex vs single-index)
✓ Returns ONE CLEAN MERGED DATAFRAME
✓ No parquet saving, no disk IO
"""

from __future__ import annotations

from io import StringIO
from typing import List

import pandas as pd
import requests
import yfinance as yf


# ==========================================================
# 1. UNIVERSE BUILDERS
# ==========================================================

def _fetch_wiki_tables(url: str, timeout: int = 15) -> List[pd.DataFrame]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))


def get_sp500_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _fetch_wiki_tables(url)

    df = None
    for t in tables:
        if "Symbol" in t.columns and "Security" in t.columns:
            df = t.copy()
            break

    if df is None:
        raise RuntimeError("Could not find S&P500 table on Wikipedia")

    df["Ticker"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.strip()
    df["Name"] = df["Security"].astype(str).str.strip()
    df["Sector"] = df["GICS Sector"].astype(str).str.strip() if "GICS Sector" in df.columns else None

    out = df[["Ticker", "Name", "Sector"]].dropna(subset=["Ticker"]).copy()
    out = out[out["Ticker"] != ""].reset_index(drop=True)
    return out


def get_hsi_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    tables = _fetch_wiki_tables(url)

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        # The constituents table layout can vary; look for likely columns.
        if any(x in cols for x in ["ticker", "code", "sehk", "constituent"]):
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
        raise RuntimeError("Could not find ticker/code column for HSI")

    # Build Yahoo HK tickers like 0005.HK
    codes = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .dropna()
        .astype(str)
        .str.zfill(4)
    )

    df = df.loc[codes.index].copy()
    df["Ticker"] = codes.values + ".HK"

    # Name column
    name_col = None
    for c in df.columns:
        if c in ("name", "constituent", "company", "stock", "security"):
            name_col = c
            break
    if name_col is None:
        # fallback to first non-ticker column
        possible = [c for c in df.columns if c != ticker_col]
        name_col = possible[0] if possible else ticker_col

    df["Name"] = df[name_col].astype(str).str.strip()
    df["Sector"] = df.get("sub-index", df.get("industry", None))

    out = df[["Ticker", "Name", "Sector"]].dropna(subset=["Ticker"]).copy()
    out = out[out["Ticker"] != ""].reset_index(drop=True)
    return out


def get_eurostoxx50_universe() -> pd.DataFrame:
    """
    Wikipedia generally provides Yahoo-style exchange tickers already (e.g., AIR.PA, SAP.DE).
    We keep it tolerant to column label tweaks.
    """
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    tables = _fetch_wiki_tables(url)

    table = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "ticker" in cols and "name" in cols and len(t) >= 40:
            table = t.copy()
            break

    if table is None:
        raise RuntimeError("EURO STOXX 50 scrape failed: constituents table with Ticker/Name not found")

    table.columns = [str(c).strip() for c in table.columns]

    out = pd.DataFrame(
        {
            "Ticker": table["Ticker"].astype(str).str.strip(),
            "Name": table["Name"].astype(str).str.strip(),
            "Sector": table["Sector"].astype(str).str.strip() if "Sector" in table.columns else None,
        }
    )

    out = out.dropna(subset=["Ticker"])
    out = out[out["Ticker"] != ""].reset_index(drop=True)

    # Optional: some people accidentally scrape "Ticker" without suffixes; this is a guardrail.
    # We do NOT force suffixes here because Wikipedia typically already includes them.
    return out[["Ticker", "Name", "Sector"]]


# ==========================================================
# 2. YAHOO DOWNLOADER (robust batching)
# ==========================================================

def download_yahoo_prices(
    tickers: List[str],
    label: str,
    period: str = "600d",
    interval: str = "1d",
) -> List[pd.DataFrame]:
    """
    Robust Yahoo downloader:
    - Handles yfinance returning MultiIndex (expected for multi-ticker) OR single-index
      (common when only 1 ticker succeeds / Yahoo flaky).
    - Falls back to per-ticker download only when batch collapses to single-index.
    - Returns list of per-ticker frames with added columns: Ticker, Index.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    batch_size = 40
    frames: List[pd.DataFrame] = []
    failed: List[str] = []

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
            failed.extend(batch)
            continue

        if data is None or data.empty:
            failed.extend(batch)
            continue

        # ----------------------------
        # Case A: MultiIndex columns (ideal)
        # ----------------------------
        if isinstance(data.columns, pd.MultiIndex):
            fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

            lvl0 = [str(x) for x in data.columns.get_level_values(0)]
            lvl1 = [str(x) for x in data.columns.get_level_values(1)]
            lvl0_has_fields = any(x in fields for x in lvl0)
            lvl1_has_fields = any(x in fields for x in lvl1)

            # Sometimes it comes back as (Field, Ticker) instead of (Ticker, Field)
            if lvl0_has_fields and not lvl1_has_fields:
                # (Field, Ticker)
                for t in batch:
                    try:
                        sub = data.xs(t, axis=1, level=1, drop_level=True).dropna()
                        if sub.empty:
                            failed.append(t)
                            continue
                        sub = sub.copy()
                        sub["Ticker"] = t
                        sub["Index"] = label
                        frames.append(sub.reset_index())
                    except Exception:
                        failed.append(t)
            else:
                # (Ticker, Field)
                available = set(map(str, data.columns.get_level_values(0)))
                for t in batch:
                    try:
                        if t not in available:
                            failed.append(t)
                            continue
                        sub = data[t].dropna()
                        if sub.empty:
                            failed.append(t)
                            continue
                        sub = sub.copy()
                        sub["Ticker"] = t
                        sub["Index"] = label
                        frames.append(sub.reset_index())
                    except Exception:
                        failed.append(t)

        # ----------------------------
        # Case B: Single-index columns (batch collapsed)
        # ----------------------------
        else:
            cols = set(map(str, data.columns))
            ohlc_ok = {"Open", "High", "Low", "Close"}.issubset(cols)

            if not ohlc_ok:
                failed.extend(batch)
                continue

            # Per-ticker fallback only in this collapse case
            for t in batch:
                try:
                    single = yf.download(
                        t,
                        period=period,
                        interval=interval,
                        auto_adjust=False,
                        threads=False,
                        progress=False,
                    )
                    if single is None or single.empty:
                        failed.append(t)
                        continue
                    single = single.dropna()
                    if single.empty:
                        failed.append(t)
                        continue
                    single = single.copy()
                    single["Ticker"] = t
                    single["Index"] = label
                    frames.append(single.reset_index())
                except Exception:
                    failed.append(t)

    return frames


# ==========================================================
# 3. MASTER FUNCTIONS (engine/app calls these)
# ==========================================================

def load_all_market_data() -> pd.DataFrame:
    """
    Returns merged DAILY OHLC dataframe for SP500 + HSI + EURO STOXX 50.
    Output includes: Date, Open, High, Low, Close, (Adj Close, Volume if provided), Ticker, Index
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    eurostoxx50 = get_eurostoxx50_universe()

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d", interval="1d")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d", interval="1d")
    euro = download_yahoo_prices(eurostoxx50["Ticker"].tolist(), "EUROSTOXX50", period="600d", interval="1d")

    if not sp and not hs and not euro:
        raise RuntimeError("No OHLC data downloaded from Yahoo (all universes empty)")

    # Fail loud if Europe missing so you don't get silent SP500+HSI only
    if not euro:
        raise RuntimeError("EUROSTOXX50 download returned 0 tickers (Yahoo fetch/parsing failed)")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    # Normalize date column
    if "Date" not in combined.columns:
        # yfinance sometimes uses 'index' naming depending on reset_index; but we standardize above
        raise RuntimeError("Combined daily dataframe missing 'Date' column")

    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Guardrails (optional)
    if any(str(t).endswith(".SI") for t in combined["Ticker"].unique()):
        raise RuntimeError("Found .SI tickers; STI still present")

    return combined


def load_all_market_data_hourly() -> pd.DataFrame:
    """
    Returns merged HOURLY OHLC dataframe for SP500 + HSI + EURO STOXX 50.
    Output includes: DateTime, Open, High, Low, Close, (Adj Close, Volume if provided), Ticker, Index
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    eurostoxx50 = get_eurostoxx50_universe()

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="60d", interval="60m")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="60d", interval="60m")
    euro = download_yahoo_prices(eurostoxx50["Ticker"].tolist(), "EUROSTOXX50", period="60d", interval="60m")

    if not sp and not hs and not euro:
        raise RuntimeError("No hourly OHLC data downloaded from Yahoo (all universes empty)")

    if not euro:
        raise RuntimeError("EUROSTOXX50 hourly download returned 0 tickers (Yahoo fetch/parsing failed)")

    combined = pd.concat(sp + hs + euro, ignore_index=True)

    datetime_col = "Datetime" if "Datetime" in combined.columns else "Date"
    if datetime_col not in combined.columns:
        raise RuntimeError("Hourly data missing Datetime/Date column")

    combined = combined.rename(columns={datetime_col: "DateTime"})
    combined["DateTime"] = pd.to_datetime(combined["DateTime"], errors="coerce")
    combined = combined.dropna(subset=["DateTime"])
    combined = combined.sort_values(["Ticker", "DateTime"]).reset_index(drop=True)

    required = {"Ticker", "DateTime", "Open", "High", "Low", "Close"}
    missing = required.difference(combined.columns)
    if missing:
        raise RuntimeError(f"Hourly data missing required columns: {sorted(missing)}")

    if any(str(t).endswith(".SI") for t in combined["Ticker"].unique()):
        raise RuntimeError("Found .SI tickers; STI still present")

    return combined
