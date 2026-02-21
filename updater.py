"""
updater.py  (simple, Streamlit-friendly version)

✓ Builds SP500, HSI, STI universes
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


def get_sti_universe():
    data = [
        ("D05.SI", "DBS Group Holdings", "Financials"),
        ("U11.SI", "United Overseas Bank", "Financials"),
        ("O39.SI", "Oversea-Chinese Banking Corporation", "Financials"),
        ("C07.SI", "Jardine Matheson", "Conglomerate"),
        ("C09.SI", "City Developments", "Real Estate"),
        ("C38U.SI", "CapitaLand Integrated Commercial Trust", "Real Estate"),
        ("C52.SI", "ComfortDelGro", "Transportation"),
        ("F34.SI", "Frasers Logistics & Commercial Trust", "Real Estate"),
        ("G13.SI", "Genting Singapore", "Entertainment"),
        ("H78.SI", "Hongkong Land", "Real Estate"),
        ("J36.SI", "Jardine Cycle & Carriage", "Industrial"),
        ("M44U.SI", "Mapletree Logistics Trust", "Real Estate"),
        ("ME8U.SI", "Mapletree Industrial Trust", "Real Estate"),
        ("N2IU.SI", "NetLink NBN Trust", "Utilities"),
        ("S63.SI", "Singapore Airlines", "Transportation"),
        ("S68.SI", "Singapore Exchange", "Financials"),
        ("S58.SI", "Sembcorp Industries", "Utilities"),
        ("U96.SI", "SATS Ltd", "Services"),
        ("S07.SI", "Singapore Technologies Engineering", "Industrial"),
        ("Z74.SI", "Singtel", "Telecom"),
        ("BN4.SI", "Keppel Corporation", "Industrial"),
        ("M01.SI", "Micro-Mechanics", "Industrial"),
        ("A17U.SI", "CapitaLand Ascendas REIT", "Real Estate"),
        ("BS6.SI", "Yangzijiang Shipbuilding", "Industrial"),
        ("C31.SI", "CapitaLand Investment", "Real Estate"),
        ("E5H.SI", "Emperador Inc", "Consumer Staples"),
        ("5DP.SI", "Delfi Limited", "Consumer Goods"),
        ("D01.SI", "Dairy Farm International", "Consumer Staples"),
        ("K71U.SI", "Keppel DC REIT", "Real Estate"),
        ("H78.SI", "Hongkong Land Holdings", "Real Estate"),
    ]

    return pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])


# ==========================================================
# 2. YAHOO DOWNLOADER (600 days, batched)
# ==========================================================

def download_yahoo_prices(tickers, label, period="600d", interval="1d"):
    """
    Downloads last `period` of OHLC for a list of tickers.
    Uses yf.download in batches of 40 tickers (no manual threads).
    Returns a list of clean DataFrames.
    """
    print(f"\nDownloading {label}: {len(tickers)} tickers")

    if not tickers:
        return []

    batch_size = 40
    frames = []
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} tickers")

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
        except Exception as e:
            print(f"  ERROR downloading batch: {e}")
            failed.extend(batch)
            continue

        # For a multi-ticker request, yfinance returns a column MultiIndex
        for t in batch:
            try:
                df_t = data[t].dropna().copy()
                df_t["Ticker"] = t
                df_t["Index"] = label
                frames.append(df_t.reset_index())
            except Exception as e:
                print(f"  Failed to parse {t}: {e}")
                failed.append(t)

    print(f"Completed {label}: {len(frames)} OK, {len(failed)} failed")
    return frames


# ==========================================================
# 3. MASTER FUNCTION (engine/app call this)
# ==========================================================

def load_all_market_data():
    """
    Returns a merged OHLC dataframe for SP500 + HSI + STI
    without saving anything to disk.
    """
    print("Building universes...")

    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    sti = get_sti_universe()

    print("SP500:", len(sp500))
    print("HSI:  ", len(hsi))
    print("STI:  ", len(sti))

    # Download last ~600 days of data
    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="600d")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI",   period="600d")
    st = download_yahoo_prices(sti["Ticker"].tolist(), "STI",   period="600d")

    if not (sp or hs or st):
        raise RuntimeError("No OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + st, ignore_index=True)

    # Ensure standard columns
    combined.rename(columns={"Date": "Date"}, inplace=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print("\nFinal merged dataframe shape:", combined.shape)
    return combined


def load_all_market_data_hourly():
    """
    Returns merged hourly OHLC dataframe for SP500 + HSI + STI.
    Output columns include: Ticker, DateTime, Open, High, Low, Close.
    """
    print("Building universes for hourly data...")

    sp500 = get_sp500_universe()
    hsi = get_hsi_universe()
    sti = get_sti_universe()

    sp = download_yahoo_prices(sp500["Ticker"].tolist(), "SP500", period="60d", interval="60m")
    hs = download_yahoo_prices(hsi["Ticker"].tolist(), "HSI", period="60d", interval="60m")
    st = download_yahoo_prices(sti["Ticker"].tolist(), "STI", period="60d", interval="60m")

    if not (sp or hs or st):
        raise RuntimeError("No hourly OHLC data downloaded from Yahoo")

    combined = pd.concat(sp + hs + st, ignore_index=True)

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

    return combined
