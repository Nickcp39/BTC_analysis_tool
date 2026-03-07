from __future__ import annotations
from pathlib import Path
from datetime import date
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FRED_XLSX = DATA_DIR / "btc_price_fred.xlsx"

FRED_SERIES_ID = "CBBTCUSD"
FRED_CSV_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES_ID}"


def update_btc_price_fred() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BTC daily data from FRED series {FRED_SERIES_ID} ...")
    df = pd.read_csv(FRED_CSV_URL)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    if date_col not in df.columns or FRED_SERIES_ID not in df.columns:
        raise ValueError(f"Unexpected FRED format, got columns: {list(df.columns)}")
    df = df[[date_col, FRED_SERIES_ID]].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "price"])

    if df.empty:
        raise ValueError("No valid BTC rows downloaded from FRED.")

    print(
        "  Rows: %d, %s ~ %s, price range: %.2f ~ %.2f"
        % (
            len(df),
            df["ts"].min().date(),
            df["ts"].max().date(),
            df["price"].min(),
            df["price"].max(),
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

