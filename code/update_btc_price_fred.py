from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRED_XLSX = DATA_DIR / "btc_price_fred.xlsx"

FRED_SERIES_ID = "CBBTCUSD"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"
YAHOO_SYMBOL = "BTC-USD"
COINBASE_PRODUCT = "BTC-USD"


def _normalize_price_df(df: pd.DataFrame, upstream: str) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").dt.normalize()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    df["upstream"] = upstream
    return df[["ts", "price", "upstream"]]


def _download_fred() -> pd.DataFrame:
    df = pd.read_csv(FRED_CSV_URL)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    if date_col not in df.columns or FRED_SERIES_ID not in df.columns:
        raise ValueError(f"Unexpected FRED format, got columns: {list(df.columns)}")
    df = df[[date_col, FRED_SERIES_ID]].copy()
    df.columns = ["ts", "price"]
    return _normalize_price_df(df, "fred")


def _download_yahoo() -> pd.DataFrame:
    start = datetime(2014, 9, 17, tzinfo=timezone.utc)
    # Add a buffer so Yahoo includes the current UTC daily candle if available.
    end = datetime.now(timezone.utc) + timedelta(days=2)
    period1 = int(start.timestamp())
    period2 = int(end.timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{YAHOO_SYMBOL}?period1={period1}&period2={period2}&interval=1d"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = payload["chart"]["result"][0]
    timestamps = result.get("timestamp") or []
    closes = result["indicators"]["quote"][0].get("close") or []
    df = pd.DataFrame({"ts": timestamps, "price": closes})
    if df.empty:
        raise ValueError("Yahoo fallback returned no BTC rows.")
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
    return _normalize_price_df(df, "yahoo_fallback")


def _download_coinbase_recent() -> pd.DataFrame:
    url = (
        "https://api.exchange.coinbase.com/products/"
        f"{COINBASE_PRODUCT}/candles?granularity=86400"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    # Coinbase candle format: [time, low, high, open, close, volume].
    df = pd.DataFrame(payload, columns=["ts", "low", "high", "open", "price", "volume"])
    if df.empty:
        raise ValueError("Coinbase fallback returned no BTC rows.")
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(None)
    return _normalize_price_df(df[["ts", "price"]], "coinbase_fallback")


def _read_existing() -> pd.DataFrame | None:
    if not FRED_XLSX.exists():
        return None
    df = pd.read_excel(FRED_XLSX, engine="openpyxl")
    if len(df.columns) < 2:
        return None
    df = df.iloc[:, :3].copy()
    if len(df.columns) == 2:
        df.columns = ["ts", "price"]
        return _normalize_price_df(df, "existing")
    df.columns = ["ts", "price", "upstream"]
    df = _normalize_price_df(df, "existing")
    return df


def _append_yahoo_new_rows(base_df: pd.DataFrame) -> pd.DataFrame:
    print("Checking Yahoo fallback for rows newer than FRED/local data ...")
    try:
        yahoo_df = _download_yahoo()
    except Exception as yahoo_error:
        print(f"  Yahoo fallback failed: {yahoo_error}")
        print("Checking Coinbase fallback for recent daily candles ...")
        yahoo_df = _download_coinbase_recent()
    base_latest = base_df["ts"].max()
    newer = yahoo_df[yahoo_df["ts"] > base_latest].copy()
    if newer.empty:
        print(f"  No newer fallback rows. Latest remains {base_latest.date()}.")
        return base_df
    print(
        "  Appending %d fallback rows: %s ~ %s"
        % (len(newer), newer["ts"].min().date(), newer["ts"].max().date())
    )
    out = pd.concat([base_df, newer], ignore_index=True)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return out[["ts", "price", "upstream"]]


def update_btc_price_fred() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BTC daily data from FRED series {FRED_SERIES_ID} ...")
    try:
        df = _download_fred()
        df = _append_yahoo_new_rows(df)
    except Exception as fred_error:
        print(f"  FRED failed: {fred_error}")
        existing = _read_existing()
        if existing is not None and not existing.empty:
            print(
                "  Using existing local file as base: %s ~ %s"
                % (existing["ts"].min().date(), existing["ts"].max().date())
            )
            df = _append_yahoo_new_rows(existing)
        else:
            print("  No local base file found; downloading full Yahoo fallback history ...")
            df = _download_yahoo()

    if df.empty:
        raise ValueError("No valid BTC rows downloaded.")

    print(
        "  Rows: %d, %s ~ %s, price range: %.2f ~ %.2f, upstreams: %s"
        % (
            len(df),
            df["ts"].min().date(),
            df["ts"].max().date(),
            df["price"].min(),
            df["price"].max(),
            ", ".join(sorted(set(df["upstream"]))),
        )
    )

    FRED_XLSX.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(FRED_XLSX, index=False, engine="openpyxl")
    print(f"Saved latest FRED BTC data to {FRED_XLSX}")
    return df


def main():
    update_btc_price_fred()


if __name__ == "__main__":
    main()

