"""
updater.py  (Streamlit-friendly, Yahoo-only, robust EU handling)

Rules you set:
1) DON'T change anything for SP500 and HSI (same universe builders + same batched downloader usage)
2) EU = single-name downloads only, period=365d (no batch)
3) Remove Stooq dependency (Yahoo-only)

What I improved to make it as close to "foolproof" as Yahoo allows:
- EU download runs in WAVES with escalating cooldown + jitter
- Detect systemic blocking (too many empties/errors) and cool down automatically
- Safer row filtering: only require OHLC columns (Yahoo often gives NaN volume for non-US)
- Hard validation of OHLC integrity
- Optional: returns failures list via print warnings (Streamlit logs)
"""

from __future__ import annotations

import random
import time
from io import StringIO
from typing import Iterable, List, Optional, Tuple

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
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

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
    NOTE: Constituents change over time; update list on STOXX rebalances.
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
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df


# ==========================================================
# 2) YAHOO DOWNLOADER (SP500/HSI batched)
#     - Keep structure as your original
# ==========================================================

def _chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows that have valid OHLC. Avoid dropping because Volume/Adj Close is missing.
    Also enforce basic OHLC sanity.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()

    df = df.copy()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # basic sanity: High >= max(Open,Close) and Low <= min(Open,Close)
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
    """
    Single-name download with robust retries.
    """
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
    Same shape as your original design; still does batch + per-ticker fallback.
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
# 2B) EURO STOXX 50 DAILY (single-name, 365d) — robust waves
# ==========================================================

def download_euro_daily_single_robust(
    tickers: List[str],
    label: str = "EUROSTOXX50",
    period: str = "365d",
    interval: str = "1d",
    # per-ticker tries inside a wave
    per_ticker_retries: int = 3,
    base_sleep_s: float = 1.25,
    # wave mechanics
    waves: int = 3,
    wave_cooldown_s: float = 20.0,
    systemic_block_threshold: float = 0.55,
) -> List[pd.DataFrame]:
    """
    "As close to foolproof as Yahoo allows" EU downloader.

    - Single-name only
    - Runs multiple WAVES:
        Wave 1: normal pacing
        Wave 2+: retries the failures after cooldown
    - If we see systemic blocking (too many failures in a wave),
      we pause longer before continuing.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return []

    frames: List[pd.DataFrame] = []
    remaining = tickers[:]
    succeeded = set()

    for wave in range(1, waves + 1):
        if not remaining:
            break

        print(f"[INFO] EU wave {wave}/{waves}: attempting {len(remaining)} tickers")
        wave_failed: List[str] = []
        wave_ok = 0

        # escalate pacing per wave
        sleep_s = base_sleep_s * (1.0 + 0.6 * (wave - 1))

        for t in remaining:
            df_t = _download_single_ticker(
                ticker=t,
                period=period,
                interval=interval,
                max_retries=per_ticker_retries,
                sleep_s=sleep_s,
            )
            if df_t is None or df_t.empty:
                wave_failed.append(t)
            else:
                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t.reset_index())
                succeeded.add(t)
                wave_ok += 1

            time.sleep(sleep_s + random.random() * 0.35)

        attempted = len(remaining)
        fail_rate = (attempted - wave_ok) / max(attempted, 1)
        print(f"[INFO] EU wave {wave}: ok={wave_ok}/{attempted} fail_rate={fail_rate:.2%}")

        # systemic block handling: pause more aggressively
        if fail_rate >= systemic_block_threshold and wave < waves:
            extra = wave_cooldown_s * (2.0 + wave)  # longer pause if things look blocked
            print(f"[WARN] EU appears blocked/throttled (fail_rate {fail_rate:.0%}). Cooling down {extra:.0f}s...")
            time.sleep(extra)

        # normal cooldown between waves
        if wave < waves and wave_failed:
            cooldown = wave_cooldown_s * (1.0 + 0.5 * (wave - 1))
            print(f"[INFO] EU cooldown before next wave: {cooldown:.0f}s (retrying {len(wave_failed)} failed)")
            time.sleep(cooldown)

        remaining = wave_failed

    # Final report
    failed_final = [t for t in tickers if t not in succeeded]
    if failed_final:
        print(
            f"[WARN] EU final failures: {len(failed_final)}/{len(tickers)} tickers. "
            f"Examples: {', '.join(failed_final[:20])}"
            + (" ..." if len(failed_final) > 20 else "")
        )

    return frames


# ==========================================================
# 3) MASTER FUNCTIONS
# ==========================================================

def load_all_market_data() -> pd.DataFrame:
    """
    Returns merged DAILY OHLC dataframe for SP500 + HSI + EURO STOXX 50.

    - SP500/HSI: batched downloader (same call pattern)
    - EU: single-name robust waves, 365d
    - No Stooq
    """
    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    euro = get_eurostoxx50_universe()

    # SP500 / HSI (keep same call pattern)
    sp_frames = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d", interval="1d")
    hs_frames = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="600d", interval="1d")

    # EU single-name robust (365d)
    eu_frames = download_euro_daily_single_robust(
        euro["Ticker"].tolist(),
        label="EUROSTOXX50",
        period="365d",
        interval="1d",
        per_ticker_retries=3,
        base_sleep_s=1.25,
        waves=3,
        wave_cooldown_s=20.0,
        systemic_block_threshold=0.55,
    )

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
