"""
Step04: 四周期按 peak 对齐、融合出图
- 4 个周期：2012→2013, 2016→2017, 2020→2021, 2024→2025
- 按 peak 做 anchor（rel_day=0 为峰日）
- 时间轴：按各 cycle 的 halving→peak 天数缩放，使「减半到峰」在图上等长
- 幅度：与 step01/02 一致，先按 VOL_LEVEL 退火缩放再出图（2025 为参考不缩放）
- 输出：visualization/YYYY-MM-DD/ 下 step04_*.csv、step04_*.txt、step04_*.png
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"

# 输入：2013 周期可选（无 btc_2012 则只画 3 周期）
F12 = DATA_DIR / "btc_2012.xlsx"
F15 = DATA_DIR / "btc_2015.xlsx"
F21 = DATA_DIR / "btc_2021.xlsx"
FRED_2025 = DATA_DIR / "btc_price_fred.xlsx"

# 4 周期锚点（与 step02/docs 一致）
HALVINGS: List[date] = [
    date(2012, 11, 28),
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 20),
]
TOPS: List[date] = [
    date(2013, 11, 29),
    date(2017, 12, 17),
    date(2021, 11, 10),
]
# 2025 峰：固定为 8 月 15 日（不用融合日期）
PEAK_2025_DATE = date(2025, 8, 15)

POST_DAYS = 60
MAX_PRE_DAYS = 300
VOL_LEVEL_2013, VOL_LEVEL_2017, VOL_LEVEL_2021, VOL_LEVEL_2025 = 20.0, 9.0, 3.0, 1.0
VOL_ALPHA = 0.5
SCALE_METHOD = "manual"
PRE_STD_SPAN = 90

CYCLE_LABELS = ["2013", "2017", "2021", "2025"]

# 输出文件名：必须带 step04 前缀
STEP04_CSV = "step04_four_cycles_scaled.csv"
STEP04_TXT = "step04_four_cycles_params.txt"
STEP04_PNG = "step04_four_cycles_scaled.png"



def read_two_col_excel(fp: Path) -> pd.DataFrame:
    assert fp.exists(), f"Missing: {fp}"
    df = pd.read_excel(fp, engine="openpyxl", header=None)
    if df.shape[0] <= 3 and df.shape[1] > df.shape[0]:
        df = df.T
    df = df.iloc[:, :2].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna().drop_duplicates("ts").sort_values("ts")


def read_fred_excel(fp: Path) -> pd.DataFrame:
    assert fp.exists(), f"Missing: {fp}"
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


def window_halving_to_peak(
    series: pd.Series,
    halving: pd.Timestamp,
    peak: pd.Timestamp,
    post_days: int = 0,
    max_pre: Optional[int] = None,
) -> pd.Series:
    halving = pd.Timestamp(halving).normalize()
    peak = pd.Timestamp(peak).normalize()
    full_idx = pd.date_range(
        series.index.min(), max(series.index.max(), peak), freq="D"
    )
    s = series.reindex(full_idx).ffill()
    start = halving
    if max_pre is not None:
        start = max(peak - pd.Timedelta(days=max_pre), start)
    end = min(peak + pd.Timedelta(days=post_days), s.index.max())
    return s.loc[start:end]


def pct_curve(series: pd.Series, anchor_date: pd.Timestamp) -> Tuple[pd.DataFrame, float]:
    anchor_date = pd.Timestamp(anchor_date).normalize()
    anchor_px = float(
        series.loc[anchor_date]
        if anchor_date in series.index
        else series.loc[:anchor_date].iloc[-1]
    )
    rel_day = (series.index - anchor_date).days
    dd_pct = (series / anchor_px - 1.0) * 100.0
    return (
        pd.DataFrame({"rel_day": rel_day, "dd_pct": dd_pct.values}, index=series.index),
        anchor_px,
    )


def pre_std(curve_rel_day_series: pd.Series, span: int = 90) -> float:
    idx = [d for d in range(-span, 1) if d in curve_rel_day_series.index]
    if not idx:
        return 0.0
    return float(curve_rel_day_series.loc[idx].std(ddof=0))


def scale_factor(
    std_old: float, std_new: float, level_old: float, level_new: float,
    method: str = "manual", alpha: float = 0.5,
) -> float:
    if method == "manual":
        return (level_new / level_old) ** alpha
    if method == "std":
        return (std_new / std_old) if std_old and std_old > 0 else 1.0
    left = (std_new / std_old) if std_old and std_old > 0 else 1.0
    right = (level_new / level_old) ** alpha
    return left * right


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

    peaks_ts = [
        pd.Timestamp(TOPS[0]),
        pd.Timestamp(TOPS[1]),
        pd.Timestamp(TOPS[2]),
        pd.Timestamp(PEAK_2025_DATE),
    ]
    halvings_ts = [pd.Timestamp(HALVINGS[i]) for i in range(4)]

    # 加载数据
    series_list: List[Optional[pd.Series]] = [None, None, None, None]
    if F12.exists():
        series_list[0] = to_daily(read_two_col_excel(F12))
    series_list[1] = to_daily(read_two_col_excel(F15))
    series_list[2] = to_daily(read_two_col_excel(F21))
    if not FRED_2025.exists():
        raise FileNotFoundError(f"Need {FRED_2025} for 2025 cycle. Run stepA1_get_data_real_time.py first.")
    if not F15.exists() or not F21.exists():
        raise FileNotFoundError(f"Need {F15} and {F21} for 2017/2021 cycles.")
    series_list[3] = to_daily(read_fred_excel(FRED_2025))

    # 2021 峰日自动取
    s21 = series_list[2]
    s21_year = s21.loc["2021-01-01":"2021-12-31"]
    peak21_ts = s21_year.idxmax()
    peaks_ts[2] = pd.Timestamp(peak21_ts)

    curves: List[Optional[pd.Series]] = [None, None, None, None]
    h2p_days: List[int] = [0, 0, 0, 0]

    for i in range(4):
        if series_list[i] is None:
            continue
        win = window_halving_to_peak(
            series_list[i], halvings_ts[i], peaks_ts[i], POST_DAYS, MAX_PRE_DAYS
        )
        curve_df, _ = pct_curve(win, peaks_ts[i])
        curves[i] = curve_df.set_index("rel_day")["dd_pct"]
        h2p_days[i] = (peaks_ts[i].normalize() - halvings_ts[i].normalize()).days

    # 参考周期：2025（时间轴以 2025 的 halving→peak 长度为基准）
    ref_idx = 3
    ref_h2p = h2p_days[ref_idx]
    if ref_h2p <= 0:
        ref_h2p = 537

    # 时间缩放：rel_day_scaled = rel_day * (ref_h2p / h2p_days_i)
    time_scale: List[float] = [
        ref_h2p / h2p_days[i] if h2p_days[i] > 0 else 1.0 for i in range(4)
    ]

    # 幅度缩放：以 2025 为参考，缩放 2013/2017/2021
    std_ref = pre_std(curves[ref_idx], span=min(PRE_STD_SPAN, max(5, h2p_days[ref_idx])))
    levels = [VOL_LEVEL_2013, VOL_LEVEL_2017, VOL_LEVEL_2021, VOL_LEVEL_2025]
    amp_scale: List[float] = [1.0, 1.0, 1.0, 1.0]
    for i in range(4):
        if i == ref_idx or curves[i] is None:
            continue
        d = h2p_days[i]
        std_i = pre_std(curves[i], span=min(PRE_STD_SPAN, max(5, d)))
        amp_scale[i] = scale_factor(
            std_i, std_ref, levels[i], VOL_LEVEL_2025,
            method=SCALE_METHOD, alpha=VOL_ALPHA,
        )

    # 构建缩放后的曲线：rel_day_scaled, dd_pct_scaled
    scaled_curves: List[Optional[pd.Series]] = [None, None, None, None]
    for i in range(4):
        if curves[i] is None:
            continue
        rd = curves[i].index.astype(float)
        rd_scaled = rd * time_scale[i]
        dd = curves[i].values * amp_scale[i]
        scaled_curves[i] = pd.Series(dd, index=rd_scaled).sort_index()

    # 合并到统一 X（用 2025 的 scaled rel_day 为基准，其余插值到该网格）
    c25 = scaled_curves[3]
    x_common = c25.index.values
    x_min, x_max = x_common.min(), x_common.max()

    merged = pd.DataFrame(index=x_common)
    merged.index.name = "rel_day_scaled"
    merged["2025"] = c25.values

    for i in range(3):
        if scaled_curves[i] is None:
            continue
        s = scaled_curves[i]
        # 插值到 x_common
        valid = (s.index >= x_min) & (s.index <= x_max)
        if not valid.any():
            continue
        x_s = s.index[valid].values
        y_s = s.values[valid]
        y_interp = np.interp(x_common, x_s, y_s)
        merged[CYCLE_LABELS[i]] = y_interp

    merged = merged.sort_index()
    merged.to_csv(OUTDIR / STEP04_CSV, encoding="utf-8-sig")

    # 摘要 TXT
    with open(OUTDIR / STEP04_TXT, "w", encoding="utf-8") as f:
        f.write("Step04: 4 cycles peak-aligned, time scaled by halving->peak, amplitude scaled\n")
        f.write(f"2025 peak (fixed): {PEAK_2025_DATE}\n")
        f.write("halving_to_peak days: " + ", ".join(f"{CYCLE_LABELS[i]}={h2p_days[i]}" for i in range(4) if curves[i] is not None) + "\n")
        f.write("time_scale (ref=2025): " + ", ".join(f"{CYCLE_LABELS[i]}={time_scale[i]:.4f}" for i in range(4)) + "\n")
        f.write("amp_scale (ref=2025): " + ", ".join(f"{CYCLE_LABELS[i]}={amp_scale[i]:.4f}" for i in range(4)) + "\n")

    # 出图：4 条线
    plt.figure(figsize=(12.5, 5))
    colors = ["C0", "C1", "C2", "C3"]
    for i in range(4):
        if scaled_curves[i] is None:
            continue
        s = scaled_curves[i]
        lab = CYCLE_LABELS[i] if i == ref_idx else f"{CYCLE_LABELS[i]} (x{amp_scale[i]:.2f})"
        plt.plot(s.index, s.values, label=lab, color=colors[i])
    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("rel_day_scaled (halving->peak normalized, peak=0)")
    plt.ylabel("dd_pct vs peak (%)")
    plt.title("4 cycles: peak-aligned, time scaled by h2p, amplitude scaled | 2025 ref")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / STEP04_PNG, dpi=170)
    plt.close()

    print("Step04 done. 2025 peak (fixed):", PEAK_2025_DATE)
    print("h2p_days:", [h2p_days[i] for i in range(4)])
    print("time_scale:", [round(time_scale[i], 4) for i in range(4)])
    print("amp_scale:", [round(amp_scale[i], 4) for i in range(4)])
    print("Output:", OUTDIR.resolve())
    print("PNG:", (PNG_DIR / STEP04_PNG).resolve())


if __name__ == "__main__":
    main()
