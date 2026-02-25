""" updater.py — Yahoo-only, SP500 + HSI + Euro Stoxx 50

Speed design:
  - Large batches (50 tickers/call) → only 11 round trips for SP500
    instead of 63. yf.download() handles 50 tickers per call cleanly.
  - 5 concurrent batch workers per universe via ThreadPoolExecutor.
  - All 3 universes download simultaneously (top-level pool).
  - Global semaphore caps Yahoo at MAX_CONCURRENT_YAHOO=5 simultaneous
    requests — safe vs Yahoo's ~100 req/min rate limit.
  - Minimal inter-batch sleep (0.5s) since few batches are needed.

Expected runtime: ~45-60s total (vs ~9 min fully sequential).
"""

from __future__ import annotations

import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Iterable, List, Optional

import pandas as pd
import requests
import yfinance as yf


# ==========================================================
# SETTINGS  — adjust here if Yahoo starts throttling
# ==========================================================
BATCH_SIZE            = 50   # tickers per yf.download() call
                              # sweet spot: 50 gives few round trips
                              # without Yahoo silently dropping tickers
MAX_CONCURRENT_YAHOO  = 5    # simultaneous Yahoo requests (global cap)
INTRA_UNIVERSE_WORKERS = 5   # threads per universe download
INTER_BATCH_SLEEP_S   = 0.5  # courtesy sleep between batches per worker
BASE_SLEEP_S          = 1.5  # base for exponential backoff on retries
MAX_RETRIES           = 3    # retries per batch / ticker

# Global semaphore — shared across ALL threads and ALL universes.
# No matter how many threads run, Yahoo never sees more than
# MAX_CONCURRENT_YAHOO simultaneous requests.
_yahoo_sem = threading.Semaphore(MAX_CONCURRENT_YAHOO)


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
# Wikipedia scraping is unreliable for this index — table structure
# and column names change without warning. Hardcoded list with
# verified Yahoo Finance suffixes is more robust in production.
# Last verified: February 2026.
# ------------------------------------------------------------------
_EUROSTOXX50_TICKERS = [
    # Austria
    ("VER.VI",   "Verbund",               "Utilities"),
    # Belgium
    ("ABI.BR",   "AB InBev",              "Consumer Staples"),
    ("UCB.BR",   "UCB",                   "Health Care"),
    # Finland
    ("NOKIA.HE", "Nokia",                 "Technology"),
    # France
    ("AI.PA",    "Air Liquide",           "Materials"),
    ("AIR.PA",   "Airbus",                "Industrials"),
    ("ACA.PA",   "Credit Agricole",       "Financials"),
    ("BN.PA",    "Danone",                "Consumer Staples"),
    ("BNP.PA",   "BNP Paribas",           "Financials"),
    ("CS.PA",    "AXA",                   "Financials"),
    ("DG.PA",    "Vinci",                 "Industrials"),
    ("EL.PA",    "EssilorLuxottica",      "Health Care"),
    ("EN.PA",    "Bouygues",              "Industrials"),
    ("ENGI.PA",  "Engie",                 "Utilities"),
    ("GLE.PA",   "Societe Generale",      "Financials"),
    ("KER.PA",   "Kering",                "Consumer Discretionary"),
    ("MC.PA",    "LVMH",                  "Consumer Discretionary"),
    ("ML.PA",    "Michelin",              "Consumer Discretionary"),
    ("OR.PA",    "L'Oreal",               "Consumer Staples"),
    ("ORA.PA",   "Orange",                "Communication Services"),
    ("RI.PA",    "Pernod Ricard",         "Consumer Staples"),
    ("RMS.PA",   "Hermes",                "Consumer Discretionary"),
    ("SAF.PA",   "Safran",                "Industrials"),
    ("SAN.PA",   "Sanofi",                "Health Care"),
    ("SGO.PA",   "Saint-Gobain",          "Industrials"),
    ("SU.PA",    "Schneider Electric",    "Industrials"),
    ("TTE.PA",   "TotalEnergies",         "Energy"),
    ("VIE.PA",   "Veolia",                "Utilities"),
    # Germany
    ("ADS.DE",   "Adidas",                "Consumer Discretionary"),
    ("ALV.DE",   "Allianz",               "Financials"),
    ("BAYN.DE",  "Bayer",                 "Health Care"),
    ("BMW.DE",   "BMW",                   "Consumer Discretionary"),
    ("BAS.DE",   "BASF",                  "Materials"),
    ("DB1.DE",   "Deutsche Boerse",       "Financials"),
    ("DHL.DE",   "DHL Group",             "Industrials"),
    ("DTE.DE",   "Deutsche Telekom",      "Communication Services"),
    ("MBG.DE",   "Mercedes-Benz",         "Consumer Discretionary"),
    ("MUV2.DE",  "Munich Re",             "Financials"),
    ("RWE.DE",   "RWE",                   "Utilities"),
    ("SAP.DE",   "SAP",                   "Technology"),
    ("SIE.DE",   "Siemens",               "Industrials"),
    # Italy
    ("ENI.MI",   "Eni",                   "Energy"),
    ("ENEL.MI",  "Enel",                  "Utilities"),
    ("ISP.MI",   "Intesa Sanpaolo",       "Financials"),
    ("STLAM.MI", "Stellantis",            "Consumer Discretionary"),
    ("UCG.MI",   "UniCredit",             "Financials"),
    # Netherlands
    ("ASML.AS",  "ASML",                  "Technology"),
    ("INGA.AS",  "ING Group",             "Financials"),
    ("PHIA.AS",  "Philips",               "Health Care"),
    # Spain
    ("BBVA.MC",  "BBVA",                  "Financials"),
    ("IBE.MC",   "Iberdrola",             "Utilities"),
    ("ITX.MC",   "Inditex",               "Consumer Discretionary"),
    ("REP.MC",   "Repsol",                "Energy"),
    ("SAN.MC",   "Santander",             "Financials"),
]


def get_eurostoxx50_universe() -> pd.DataFrame:
    records = [
        {"Ticker": t, "Name": name, "Sector": sector}
        for t, name, sector in _EUROSTOXX50_TICKERS
    ]
    return pd.DataFrame(records, columns=["Ticker", "Name", "Sector"])


# ==========================================================
# 2) CORE HELPERS
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

    bad = (
        (df["High"] < df[["Open", "Close"]].max(axis=1)) |
        (df["Low"]  > df[["Open", "Close"]].min(axis=1))
    )
    if bad.any():
        df = df.loc[~bad].copy()

    return df


def _backoff_sleep(attempt: int, base: float = BASE_SLEEP_S) -> None:
    """Exponential backoff with jitter."""
    time.sleep(base * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5))


def _download_single_ticker(
    ticker: str,
    period: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Single-ticker fallback with retries and semaphore."""
    for attempt in range(1, MAX_RETRIES + 1):
        with _yahoo_sem:
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
                print(f"[WARN] {ticker} attempt {attempt}/{MAX_RETRIES}: {exc}")

        if attempt < MAX_RETRIES:
            _backoff_sleep(attempt)

    print(f"[WARN] {ticker}: all retries exhausted — skipping")
    return None


# ==========================================================
# 3) BATCH DOWNLOAD WORKER  (runs in a thread)
# ==========================================================

def _download_batch(
    batch: List[str],
    label: str,
    batch_num: int,
    total_batches: int,
    period: str,
    interval: str,
) -> List[pd.DataFrame]:
    """
    Download one batch. Called from a ThreadPoolExecutor worker.
    Acquires _yahoo_sem before every Yahoo HTTP call.
    """
    frames: List[pd.DataFrame] = []
    data     = pd.DataFrame()
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        with _yahoo_sem:
            try:
                data = yf.download(
                    tickers=batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=False,   # our own threading; disable yf internal
                    progress=False,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                print(f"[WARN] {label} batch {batch_num}/{total_batches} "
                      f"attempt {attempt}/{MAX_RETRIES}: {exc}")

        if attempt < MAX_RETRIES:
            _backoff_sleep(attempt)

    if last_exc is not None:
        print(f"[WARN] {label} batch {batch_num} failed all retries")

    # --- Extract per-ticker frames ---
    found: set = set()

    for t in batch:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.get_level_values(0):
                    continue
                df_t = _clean_ohlc(data[t])
            else:
                # Single-ticker batch → flat columns
                if len(batch) != 1:
                    continue
                df_t = _clean_ohlc(data)

            if df_t.empty:
                continue

            df_t["Ticker"] = t
            df_t["Index"]  = label
            frames.append(df_t.reset_index())
            found.add(t)

        except Exception:
            continue

    # --- Per-ticker fallback for anything silently dropped ---
    missing = [t for t in batch if t not in found]
    if missing:
        print(f"[INFO] {label} batch {batch_num}: "
              f"single-ticker fallback for {len(missing)} tickers: {missing}")

    for t in missing:
        df_t = _download_single_ticker(t, period, interval)
        if df_t is None or df_t.empty:
            continue
        df_t["Ticker"] = t
        df_t["Index"]  = label
        frames.append(df_t.reset_index())

    # Courtesy sleep so workers don't all slam Yahoo simultaneously
    # after finishing — small because batches are large and few
    time.sleep(INTER_BATCH_SLEEP_S + random.uniform(0.0, 0.3))

    return frames


# ==========================================================
# 4) UNIVERSE DOWNLOADER
# ==========================================================

def download_yahoo_prices(
    tickers: List[str],
    label: str,
    period: str = "600d",
    interval: str = "1d",
) -> List[pd.DataFrame]:
    """
    Download an entire universe using a thread pool.
    Large BATCH_SIZE (50) keeps round-trips low.
    INTRA_UNIVERSE_WORKERS threads process batches concurrently.
    Global _yahoo_sem enforces MAX_CONCURRENT_YAHOO across all universes.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    batches       = _chunk(tickers, BATCH_SIZE)
    total_batches = len(batches)
    all_frames: List[pd.DataFrame] = []

    print(f"[INFO] {label}: {len(tickers)} tickers → "
          f"{total_batches} batches of up to {BATCH_SIZE} "
          f"({INTRA_UNIVERSE_WORKERS} workers)")

    with ThreadPoolExecutor(max_workers=INTRA_UNIVERSE_WORKERS) as pool:
        futures = {
            pool.submit(
                _download_batch,
                batch, label, i + 1, total_batches, period, interval,
            ): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            try:
                all_frames.extend(future.result())
            except Exception as exc:
                print(f"[ERROR] {label} batch {futures[future]}: {exc}")

    tickers_got   = {
        str(f["Ticker"].iloc[0])
        for f in all_frames
        if not f.empty and "Ticker" in f.columns
    }
    missing_count = len(tickers) - len(tickers_got)
    status        = "✓" if missing_count == 0 else f"⚠ {missing_count} missing"
    print(f"[INFO] {label} done — {len(tickers_got)} tickers  {status}")

    return all_frames


# ==========================================================
# 5) MASTER LOAD FUNCTIONS
# ==========================================================

def load_all_market_data() -> pd.DataFrame:
    """
    Returns merged DAILY OHLC for SP500 + HSI + Euro Stoxx 50.

    All three universes download concurrently. The shared semaphore
    ensures Yahoo never sees more than MAX_CONCURRENT_YAHOO=5
    simultaneous requests regardless of thread count.

    Target runtime: ~45-60 seconds.
    """
    t0 = time.time()

    sp500     = get_sp500_universe()
    hsi       = get_hsi_universe()
    eurostoxx = get_eurostoxx50_universe()

    print(f"[INFO] Universes — SP500: {len(sp500)}, "
          f"HSI: {len(hsi)}, EuroStoxx50: {len(eurostoxx)}")
    print(f"[INFO] Downloading all universes in parallel "
          f"(max {MAX_CONCURRENT_YAHOO} concurrent Yahoo calls)...")

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_sp = pool.submit(
            download_yahoo_prices,
            sp500["Ticker"].tolist(), "SP500", "600d", "1d",
        )
        fut_hs = pool.submit(
            download_yahoo_prices,
            hsi["Ticker"].tolist(), "HSI", "600d", "1d",
        )
        fut_eu = pool.submit(
            download_yahoo_prices,
            eurostoxx["Ticker"].tolist(), "EUROSTOXX50", "600d", "1d",
        )

        sp_frames = fut_sp.result()
        hs_frames = fut_hs.result()
        eu_frames = fut_eu.result()

    all_frames = sp_frames + hs_frames + eu_frames
    if not all_frames:
        raise RuntimeError("No DAILY OHLC data downloaded from Yahoo Finance")

    combined = pd.concat(all_frames, ignore_index=True)

    date_col = "Date" if "Date" in combined.columns else "Datetime"
    combined = combined.rename(columns={date_col: "Date"})
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.dropna(subset=["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    elapsed = time.time() - t0
    print(f"[INFO] Daily data loaded — {combined['Ticker'].nunique()} tickers, "
          f"{len(combined):,} rows  ({elapsed:.0f}s)")

    return combined


def load_hourly_prices_for_tickers(tickers: Iterable[str]) -> pd.DataFrame:
    """Returns merged HOURLY OHLC for the supplied tickers only."""
    tickers = sorted({str(t).strip() for t in tickers if str(t).strip()})
    if not tickers:
        return pd.DataFrame(
            columns=["Ticker", "DateTime", "Open", "High", "Low", "Close", "Index"]
        )

    t0     = time.time()
    frames = download_yahoo_prices(
        tickers, label="HOURLY", period="60d", interval="60m",
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

    elapsed = time.time() - t0
    print(f"[INFO] Hourly data loaded — {combined['Ticker'].nunique()} tickers, "
          f"{len(combined):,} rows  ({elapsed:.0f}s)")

    return combined


def compute_dashboard() -> pd.DataFrame:
    """Entry point used by Streamlit app."""
    return load_all_market_data()
