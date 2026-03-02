import pandas as pd


def stepA1_get_data_real_time(
    series_id: str = "CBBTCUSD",
    output_excel: str = "btc_price_fred.xlsx",
) -> None:
    """
    Download latest BTC price data from FRED (Coinbase Bitcoin series)
    and save it as an Excel file.

    Parameters
    ----------
    series_id : str
        FRED series ID. Default is "CBBTCUSD" (Coinbase Bitcoin, USD).
    output_excel : str
        Path for the output Excel file.
    """
    # FRED CSV endpoint for a given series
    csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    # Read CSV directly from FRED
    df = pd.read_csv(csv_url)

    # Save to Excel
    df.to_excel(output_excel, index=False)

    # Print latest data row as a simple sanity check
    latest = df.iloc[-1]
    date_col = "DATE"
    value_col = series_id

    print(f"最新日期: {latest[date_col]}")
    print(f"最新价格 ({series_id}): {latest[value_col]}")
    print(f"数据已保存到: {output_excel}")


if __name__ == "__main__":
    stepA1_get_data_real_time()

