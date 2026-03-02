"""
StepB4: 仅对比峰后波动率与峰后天数
- 只取 rel_day >= 0 的区段；Y 轴仍为 dd_pct（相对峰顶涨跌%），计算方式与 step05 一致
- 天数总量以 2025 为准：2025 峰后天数 = 曲线锚点日到最新日期；2017/2021 峰后天数按各自窗口，时间轴按 2025 峰后天数做缩放对齐
- 输出：各周期峰后天数、峰后波动率（std(dd_pct)）；叠加图（x 为 0~N_ref，N_ref=2025 峰后天数）
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"

F15 = DATA_DIR / "btc_2015.xlsx"
F21 = DATA_DIR / "btc_2021.xlsx"
FRED_2025 = DATA_DIR / "btc_price_fred.xlsx"
HALVING_PEAK_DATES = DATA_DIR / "btc_halving_peak_dates.xlsx"

def _load_halving_peak():
    import sys
    _code_dir = ROOT / "code"
    if str(_code_dir) not in sys.path:
        sys.path.insert(0, str(_code_dir))
    try:
        from load_halving_peak import load_halving_peak_dates
        h, p = load_halving_peak_dates(HALVING_PEAK_DATES)
        return h, p
    except Exception:
        return (
            [date(2012, 11, 28), date(2016, 7, 9), date(2020, 5, 11), date(2024, 4, 20)],
            [date(2013, 11, 29), date(2017, 12, 17), date(2021, 11, 10), date(2025, 8, 15)],
        )

_HALVINGS, _TOPS = _load_halving_peak()
HALVINGS: List[date] = _HALVINGS
TOPS: List[date] = _TOPS

HALVING_2016 = pd.Timestamp("2016-07-09")
HALVING_2020 = pd.Timestamp("2020-05-11")
HALVING_2024 = pd.Timestamp("2024-04-20")
# 与 step04 一致，便于峰前效果对比
PEAK_2017 = pd.Timestamp("2017-12-17")
PEAK_2021 = None
POST_DAYS = 60
MAX_PRE_DAYS = 300
PEAK_2025_ANCHOR = "table"  # 峰=08-15，峰后波动率相对 8 月 15 日算


def read_two_col_excel(fp: Path) -> pd.DataFrame:
    assert fp.exists(), f"未找到文件：{fp}"
    df = pd.read_excel(fp, engine="openpyxl", header=None)
    if df.shape[0] <= 3 and df.shape[1] > df.shape[0]:
        df = df.T
    df = df.iloc[:, :2].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna().drop_duplicates("ts").sort_values("ts")


def read_fred_excel(fp: Path) -> pd.DataFrame:
    assert fp.exists(), f"未找到文件：{fp}"
    df = pd.read_excel(fp, engine="openpyxl")
    df = df.iloc[:, :2].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna().drop_duplicates("ts").sort_values("ts")


def to_daily(df: pd.DataFrame) -> pd.Series:
    s = df.set_index("ts")["price"].resample("D").last()
    s = s.reindex(
        pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D")
    ).ffill()
    s.index.name = "ts"
    return s


def window_halving_to_peak(series: pd.Series, halving: pd.Timestamp, peak: pd.Timestamp, post_days=0, max_pre=None, end_date=None):
    """end_date: 若给出则窗口右端延伸到该日（用于 2025 延到最新），否则用 peak+post_days"""
    halving = pd.Timestamp(halving).normalize()
    peak = pd.Timestamp(peak).normalize()
    full_idx = pd.date_range(series.index.min(), max(series.index.max(), peak), freq="D")
    s = series.reindex(full_idx).ffill()
    start = halving
    if max_pre is not None:
        start = max(peak - pd.Timedelta(days=max_pre), start)
    if end_date is not None:
        end = min(pd.Timestamp(end_date).normalize(), s.index.max())
    else:
        end = min(peak + pd.Timedelta(days=post_days), s.index.max())
    return s.loc[start:end]


def pct_curve(series: pd.Series, anchor_date: pd.Timestamp):
    anchor_date = pd.Timestamp(anchor_date).normalize()
    anchor_px = float(
        series.loc[anchor_date] if anchor_date in series.index else series.loc[:anchor_date].iloc[-1]
    )
    rel_day = (series.index - anchor_date).days
    dd_pct = (series / anchor_px - 1.0) * 100.0
    return (
        pd.DataFrame({"rel_day": rel_day, "dd_pct": dd_pct.values}, index=series.index),
        anchor_px,
    )


def post_std(curve_rel_day_series: pd.Series):
    """峰后波动率：rel_day >= 0 的 dd_pct 的标准差（ddof=0）。"""
    idx = curve_rel_day_series.index[curve_rel_day_series.index >= 0]
    if len(idx) < 2:
        return float("nan")
    return float(curve_rel_day_series.loc[idx].std(ddof=0))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from matplotlib import rcParams, font_manager
        for p in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
            if p.exists():
                font_manager.fontManager.addfont(str(p))
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    PEAK_2025_DATE = TOPS[-1] if len(TOPS) > 3 else date(2025, 8, 15)
    PEAK_2025 = pd.Timestamp(PEAK_2025_DATE)

    s15 = to_daily(read_two_col_excel(F15))
    s21 = to_daily(read_two_col_excel(F21))
    s25 = to_daily(read_fred_excel(FRED_2025))
    latest_25 = s25.index.max().normalize()  # 2025 数据最新日期

    # 2017 / 2021：与 step05 一致，窗口到 peak+POST_DAYS
    win17 = window_halving_to_peak(s15, HALVING_2016, PEAK_2017, POST_DAYS, MAX_PRE_DAYS)
    curve17, _ = pct_curve(win17, PEAK_2017)
    c17 = curve17.set_index("rel_day")["dd_pct"]

    s21_year = s21.loc["2021-01-01":"2021-12-31"]
    peak21 = s21_year.idxmax() if PEAK_2021 is None else PEAK_2021
    win21 = window_halving_to_peak(s21, HALVING_2020, peak21, POST_DAYS, MAX_PRE_DAYS)
    curve21, _ = pct_curve(win21, peak21)
    c21 = curve21.set_index("rel_day")["dd_pct"]

    # 2025：窗口到最新日期（天数总量以 2025 峰后到最新为准）
    win25 = window_halving_to_peak(s25, HALVING_2024, PEAK_2025, POST_DAYS, MAX_PRE_DAYS, end_date=latest_25)
    if PEAK_2025_ANCHOR == "actual" and len(win25) > 0:
        peak25_anchor = win25.idxmax()
        curve25, _ = pct_curve(win25, peak25_anchor)
    else:
        peak25_anchor = PEAK_2025
        curve25, _ = pct_curve(win25, PEAK_2025)
    c25 = curve25.set_index("rel_day")["dd_pct"]

    # 只保留峰后 rel_day >= 0
    p17 = c17.loc[c17.index >= 0].sort_index()
    p21 = c21.loc[c21.index >= 0].sort_index()
    p25 = c25.loc[c25.index >= 0].sort_index()

    days_17 = int(p17.index.max()) if len(p17) else 0
    days_21 = int(p21.index.max()) if len(p21) else 0
    days_25 = int(p25.index.max()) if len(p25) else 0  # 2025 峰后天数 = 锚点日到最新日期

    vol_17 = post_std(c17)
    vol_21 = post_std(c21)
    vol_25 = post_std(c25)

    # 时间缩放：以 2025 峰后天数为总量 N_ref，2017/2021 的 post rel_day 映射到 0~N_ref
    N_ref = days_25
    if N_ref <= 0:
        N_ref = max(days_17, days_21, 1)
    scale_17 = N_ref / days_17 if days_17 else 1.0
    scale_21 = N_ref / days_21 if days_21 else 1.0

    # 缩放后的 x（0 ~ N_ref）与 y
    x17 = p17.index.values * scale_17
    y17 = p17.values
    x21 = p21.index.values * scale_21
    y21 = p21.values
    x25 = p25.index.values
    y25 = p25.values

    # ---------- 输出 ----------
    lines = [
        "=== StepB4 仅对比峰后波动率与峰后天数 ===",
        f"2025 曲线锚点: {pd.Timestamp(peak25_anchor).strftime('%Y-%m-%d')}",
        f"2025 数据最新日期: {latest_25.strftime('%Y-%m-%d')}",
        "",
        "峰后天数（天数总量以 2025 为准）:",
        f"  2017: {days_17} 天",
        f"  2021: {days_21} 天",
        f"  2025: {days_25} 天  (锚点→最新)",
        "",
        "峰后波动率 std(dd_pct), ddof=0:",
        f"  2017: {vol_17:.4f}",
        f"  2021: {vol_21:.4f}",
        f"  2025: {vol_25:.4f}",
        "",
        f"叠加图 x 轴范围: 0 ~ {N_ref}（与 2025 峰后天数一致），2017/2021 已按比例缩放。",
    ]
    txt_path = OUTDIR / "stepB4_post_peak_notes.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV
    import csv
    csv_path = OUTDIR / "stepB4_post_peak_metrics.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "post_peak_days", "post_peak_vol_std"])
        w.writerow(["2017", days_17, f"{vol_17:.6f}"])
        w.writerow(["2021", days_21, f"{vol_21:.6f}"])
        w.writerow(["2025", days_25, f"{vol_25:.6f}"])

    # 图：峰后叠加，x 为 0~N_ref
    plt.figure(figsize=(12, 4.5))
    plt.plot(x25, y25, label=f"2025（{days_25}天 vol={vol_25:.3f}）")
    plt.plot(x21, y21, label=f"2021（{days_21}天→{N_ref}天 vol={vol_21:.3f}）")
    plt.plot(x17, y17, label=f"2017（{days_17}天→{N_ref}天 vol={vol_17:.3f}）")
    plt.xlabel(f"峰后天数（缩放后，总量={N_ref} 与 2025 一致）")
    plt.ylabel("相对峰顶涨跌（%）")
    plt.title(f"StepB4 仅峰后对比 | 锚点={pd.Timestamp(peak25_anchor).strftime('%Y-%m-%d')} 最新={latest_25.strftime('%Y-%m-%d')}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "stepB4_post_peak_only.png", dpi=170)
    plt.close()

    print("StepB4: post-peak only (volatility & days), 2025 days = anchor to latest")
    print("  Post-peak days: 2017=%d, 2021=%d, 2025=%d (ref)" % (days_17, days_21, days_25))
    print("  Post-peak vol:  2017=%.4f, 2021=%.4f, 2025=%.4f" % (vol_17, vol_21, vol_25))
    print("  Output:", OUTDIR.resolve())
    print("  PNG:   ", PNG_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
