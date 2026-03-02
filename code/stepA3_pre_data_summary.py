"""
StepA3_pre_data_summary: 合并所有 BTC 价格数据源，输出完整的日线数据
- 读取 data/ 下所有 BTC 相关的 xlsx 文件
- 合并、去重、按日期排序
- 输出：btc_merged_daily.csv 和数据概览
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
OUTDIR.mkdir(parents=True, exist_ok=True)


def read_fred_excel(fp: Path) -> pd.DataFrame:
    """FRED 格式：observation_date, CBBTCUSD"""
    df = pd.read_excel(fp, engine="openpyxl")
    df = df.iloc[:, :2].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["source"] = fp.stem
    return df.dropna(subset=["ts", "price"])


def read_yahoo_excel(fp: Path) -> pd.DataFrame:
    """Yahoo Finance 格式：Date, Open, High, Low, Close, Adj Close, Volume"""
    df = pd.read_excel(fp, engine="openpyxl")
    # 取 Date 和 Close 列
    if "Date" in df.columns:
        date_col = "Date"
        # Close 列可能有特殊字符
        close_col = [c for c in df.columns if "close" in c.lower()][0]
        df = df[[date_col, close_col]].copy()
        df.columns = ["ts", "price"]
    else:
        # 可能是无 header 格式
        df = df.iloc[:, :2].copy()
        df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["source"] = fp.stem
    return df.dropna(subset=["ts", "price"])


def read_two_col_excel(fp: Path) -> pd.DataFrame:
    """两列格式：日期, 价格（可能无 header）"""
    df = pd.read_excel(fp, engine="openpyxl", header=None)
    # 如果是横向排列，转置
    if df.shape[0] <= 3 and df.shape[1] > df.shape[0]:
        df = df.T
    df = df.iloc[:, :2].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["source"] = fp.stem
    return df.dropna(subset=["ts", "price"])


def load_all_btc_data() -> pd.DataFrame:
    """加载所有 BTC 数据文件并合并"""
    all_dfs = []
    
    # 1. btc_price_fred.xlsx (FRED 格式，最完整)
    fred_file = DATA_DIR / "btc_price_fred.xlsx"
    if fred_file.exists():
        df = read_fred_excel(fred_file)
        print(f"  {fred_file.name}: {len(df)} rows, {df['ts'].min().date()} ~ {df['ts'].max().date()}")
        all_dfs.append(df)
    
    # 2. btc_2015.xlsx (Yahoo 格式)
    f2015 = DATA_DIR / "btc_2015.xlsx"
    if f2015.exists():
        df = read_yahoo_excel(f2015)
        print(f"  {f2015.name}: {len(df)} rows, {df['ts'].min().date()} ~ {df['ts'].max().date()}")
        all_dfs.append(df)
    
    # 3. btc_2021.xlsx (两列格式)
    f2021 = DATA_DIR / "btc_2021.xlsx"
    if f2021.exists():
        df = read_two_col_excel(f2021)
        # 过滤掉异常日期 (1970 等)
        df = df[df["ts"] >= "2000-01-01"]
        if len(df) > 0:
            print(f"  {f2021.name}: {len(df)} rows, {df['ts'].min().date()} ~ {df['ts'].max().date()}")
            all_dfs.append(df)
    
    # 4. btc_2025.xlsx (两列格式)
    f2025 = DATA_DIR / "btc_2025.xlsx"
    if f2025.exists():
        df = read_two_col_excel(f2025)
        # 过滤掉异常日期
        df = df[df["ts"] >= "2000-01-01"]
        if len(df) > 0:
            print(f"  {f2025.name}: {len(df)} rows, {df['ts'].min().date()} ~ {df['ts'].max().date()}")
            all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No BTC data files found!")
    
    return pd.concat(all_dfs, ignore_index=True)


def merge_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """合并、去重、按日期排序"""
    # 标准化日期为日期（去掉时间部分）
    df["date"] = df["ts"].dt.normalize()
    
    # 按日期分组，如果有多个来源，优先使用 fred 数据
    source_priority = {"btc_price_fred": 1, "btc_2025": 2, "btc_2021": 3, "btc_2015": 4}
    df["priority"] = df["source"].map(lambda x: source_priority.get(x, 99))
    
    # 按日期和优先级排序，然后去重保留第一个
    df = df.sort_values(["date", "priority"])
    df_deduped = df.drop_duplicates(subset=["date"], keep="first")
    
    # 按日期排序
    df_deduped = df_deduped.sort_values("date").reset_index(drop=True)
    
    return df_deduped[["date", "price", "source"]]


def fill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    """填充缺失的日期（用前一天的价格）"""
    df = df.set_index("date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df["price"] = df["price"].ffill()
    df["source"] = df["source"].ffill()
    df = df.reset_index()
    df.columns = ["date", "price", "source"]
    return df


def main():
    print("=== StepA3_pre_data_summary: 合并 BTC 数据 ===\n")
    
    print("1. 加载所有数据文件...")
    raw_df = load_all_btc_data()
    print(f"   Total raw rows: {len(raw_df)}\n")
    
    print("2. 合并、去重、排序...")
    merged_df = merge_and_dedupe(raw_df)
    print(f"   After dedup: {len(merged_df)} unique dates\n")
    
    print("3. 填充缺失日期...")
    filled_df = fill_missing_days(merged_df)
    print(f"   After fill: {len(filled_df)} days\n")
    
    # 数据概览
    print("4. 数据概览:")
    print(f"   Date range: {filled_df['date'].min().date()} ~ {filled_df['date'].max().date()}")
    print(f"   Total days: {len(filled_df)}")
    print(f"   Price range: ${filled_df['price'].min():.2f} ~ ${filled_df['price'].max():.2f}")
    
    # 按年份统计
    filled_df["year"] = filled_df["date"].dt.year
    year_stats = filled_df.groupby("year").agg(
        days=("price", "count"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        sources=("source", lambda x: ", ".join(sorted(set(x))))
    )
    print("\n   By year:")
    for year, row in year_stats.iterrows():
        print(f"     {year}: {row['days']} days, ${row['min_price']:.0f} ~ ${row['max_price']:.0f}, sources: {row['sources']}")
    
    # 输出到 data 目录（永久保存，供其他脚本使用）
    data_csv = DATA_DIR / "btc_merged_daily.csv"
    data_xlsx = DATA_DIR / "btc_merged_daily.xlsx"
    
    filled_df[["date", "price", "source"]].to_csv(data_csv, index=False)
    filled_df[["date", "price", "source"]].to_excel(data_xlsx, index=False, engine="openpyxl")
    
    print(f"\n5. Output files (in data/):")
    print(f"   CSV:  {data_csv}")
    print(f"   XLSX: {data_xlsx}")
    
    # 输出 notes
    notes = [
        "=== BTC Merged Daily Data Summary ===",
        f"Generated: {CURRENT_DATE}",
        f"Date range: {filled_df['date'].min().date()} ~ {filled_df['date'].max().date()}",
        f"Total days: {len(filled_df)}",
        f"Price range: ${filled_df['price'].min():.2f} ~ ${filled_df['price'].max():.2f}",
        "",
        "Source priority: btc_price_fred > btc_2025 > btc_2021 > btc_2015",
        "Missing days filled with previous day's price.",
        "",
        "By year:",
    ]
    for year, row in year_stats.iterrows():
        notes.append(f"  {year}: {row['days']} days, ${row['min_price']:.0f} ~ ${row['max_price']:.0f}")
    
    notes_path = OUTDIR / "stepA3_pre_data_summary.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
    print(f"   Notes: {notes_path}")
    
    print("\nDone.")
    return filled_df


if __name__ == "__main__":
    main()
