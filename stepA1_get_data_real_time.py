import pandas as pd
from pathlib import Path


def stepA1_get_data_real_time(
    series_id: str = "CBBTCUSD",
    output_excel: str = "data/btc_price_fred.xlsx",
) -> None:
    """
    Download latest BTC price data from FRED (Coinbase Bitcoin series)
    and save it as an Excel file.

    Parameters
    ----------
    series_id : str
        FRED series ID. Default is "CBBTCUSD" (Coinbase Bitcoin, USD).
    output_excel : str
        Path for the output Excel file (default: data/btc_price_fred.xlsx).
    """
    # FRED CSV endpoint for a given series
    csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    # Read CSV directly from FRED
    df = pd.read_csv(csv_url)

    # Ensure data directory exists, then save to Excel
    out_path = Path(output_excel)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)

    # Print latest data row (use actual column names from FRED CSV)
    latest = df.iloc[-1]
    date_col = df.columns[0]
    value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    print(f"Latest date: {latest[date_col]}")
    print(f"Latest price ({value_col}): {latest[value_col]}")
    print(f"Saved to: {output_excel}")


if __name__ == "__main__":
    stepA1_get_data_real_time()

