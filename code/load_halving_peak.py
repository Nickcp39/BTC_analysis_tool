"""
从 data/btc_halving_peak_dates.xlsx 读取减半日、峰值日，供时间模型与画图统一使用。
表结构：cycle, halving_date, peak_date（每行一轮）。
"""
from pathlib import Path
from datetime import date
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "btc_halving_peak_dates.xlsx"


def load_halving_peak_dates(
    path: Path = DATA_FILE,
) -> Tuple[List[date], List[date]]:
    """
    读取 Excel，返回 (HALVINGS, TOPS)。
    HALVINGS[i] / TOPS[i] 为第 i 轮的减半日、峰值日（按 cycle 升序）。
    """
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"Halving/peak dates file not found: {path}")
    df = pd.read_excel(path, sheet_name="halving_peak", engine="openpyxl")
    df = df.sort_values("cycle").reset_index(drop=True)
    halving_col = "halving_date"
    peak_col = "peak_date"
    if halving_col not in df.columns or peak_col not in df.columns:
        raise ValueError(f"Expected columns {halving_col}, {peak_col}; got {list(df.columns)}")
    df[halving_col] = pd.to_datetime(df[halving_col]).dt.date
    df[peak_col] = pd.to_datetime(df[peak_col]).dt.date
    return list(df[halving_col]), list(df[peak_col])


if __name__ == "__main__":
    h, p = load_halving_peak_dates()
    print("HALVINGS:", h)
    print("TOPS:   ", p)
