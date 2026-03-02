"""
Step02: 减半→峰值对比图（按 4 fusion model 逻辑）
- 多模型时间预测（基线中位数 + 回归 + 顶→顶外推）→ 融合得到 2025 峰值锚点
- 减半间隔在变长、波动在递减，用多模型综合后再做对齐画图
- 输入：data/ 下价格数据；输出：visualization/YYYY-MM-DD/ 与 visualization/YYYY-MM-DD/png/
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta, datetime
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median

# ========= 路径配置 =========
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"

# 输入文件
F15 = DATA_DIR / "btc_2015.xlsx"
F21 = DATA_DIR / "btc_2021.xlsx"
FRED_2025 = DATA_DIR / "btc_price_fred.xlsx"

# ========= 4 fusion 锚点（与 btc price predict 4 fusion model 一致）=========
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
BOTTOMS: List[date] = [
    date(2015, 1, 14),
    date(2018, 12, 15),
    date(2022, 11, 21),
]

# 画图用：历史周期对应减半/峰（与 step01 对齐）
HALVING_2016 = pd.Timestamp("2016-07-09")
HALVING_2020 = pd.Timestamp("2020-05-11")
HALVING_2024 = pd.Timestamp("2024-04-20")
PEAK_2017 = pd.Timestamp("2017-12-19")
PEAK_2021 = None  # 自动取 2021 年最高日

POST_DAYS = 60
MAX_PRE_DAYS = 300
VOL_LEVEL_2017, VOL_LEVEL_2021, VOL_LEVEL_2025 = 9.0, 3.0, 1.0
VOL_ALPHA = 0.5
SCALE_METHOD = "manual"
PRE_STD_SPAN = 90


# ========= 时间多模型（4 fusion 逻辑）=========
@dataclass
class MultiModelTimeResult:
    base_days: int
    base_date: date
    regress_days: int
    regress_date: date
    peak2peak_days: int
    peak2peak_date: date
    window: Tuple[date, date]


def predict_peak_multi() -> MultiModelTimeResult:
    """基线 + 修正A(回归) + 修正B(顶→顶)。"""
    halving_to_top = [(TOPS[i] - HALVINGS[i]).days for i in range(len(TOPS))]
    base_days = int(round(median(halving_to_top)))
    base_date = HALVINGS[-1] + timedelta(days=base_days)

    halving_intervals = [(HALVINGS[i + 1] - HALVINGS[i]).days for i in range(len(HALVINGS) - 1)]
    x = np.array(halving_intervals, dtype=float)
    y = np.array(halving_to_top, dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    regress_days = int(round(a * halving_intervals[-1] + b))
    regress_date = HALVINGS[-1] + timedelta(days=regress_days)

    peak2peak = [(TOPS[i + 1] - TOPS[i]).days for i in range(len(TOPS) - 1)]
    avg_p2p = int(round(np.mean(peak2peak)))
    peak2peak_date = TOPS[-1] + timedelta(days=avg_p2p)
    peak2peak_days = (peak2peak_date - HALVINGS[-1]).days

    lo = min(base_date, regress_date, peak2peak_date)
    hi = max(base_date, regress_date, peak2peak_date)
    return MultiModelTimeResult(
        base_days, base_date,
        regress_days, regress_date,
        peak2peak_days, peak2peak_date,
        (lo, hi),
    )


@dataclass
class TimeModelResult:
    center: date
    window_lo: date
    window_hi: date
    center_days: int
    corr_halving_vs_delay: float
    alt_center_by_median: date


def predict_peak_window_by_time(u_factor: float = 1.2, method: str = "regress") -> TimeModelResult:
    """时间模型 A：减半→大顶 回归/中位数 + 不确定度窗口。"""
    halving_to_top = np.array([(TOPS[i] - HALVINGS[i]).days for i in range(len(TOPS))], dtype=float)
    halving_intervals = np.array(
        [(HALVINGS[i + 1] - HALVINGS[i]).days for i in range(len(HALVINGS) - 1)], dtype=float
    )
    corr = float(np.corrcoef(halving_intervals, halving_to_top)[0, 1]) if len(halving_intervals) >= 2 else 0.0

    if method == "regress" and len(halving_to_top) >= 2:
        x, y = halving_intervals, halving_to_top
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        center_days = int(round(a * halving_intervals[-1] + b))
        resid = y - (a * x + b)
        sigma = float(np.sqrt(np.mean(resid ** 2))) if len(y) > 1 else float(np.std(y))
    else:
        center_days = int(round(median(halving_to_top)))
        sigma = float(np.std(halving_to_top))

    sigma *= u_factor
    anchor = HALVINGS[-1]
    center_dt = anchor + timedelta(days=center_days)
    window_lo = anchor + timedelta(days=int(round(center_days - sigma)))
    window_hi = anchor + timedelta(days=int(round(center_days + sigma)))
    alt_center = anchor + timedelta(days=int(round(median(halving_to_top))))

    return TimeModelResult(
        center=center_dt,
        window_lo=window_lo,
        window_hi=window_hi,
        center_days=center_days,
        corr_halving_vs_delay=corr,
        alt_center_by_median=alt_center,
    )


def fusion_time_center(
    multi: MultiModelTimeResult,
    weights: Tuple[float, float, float] = (1.0, 1.3, 1.1),
) -> date:
    """三时间中心加权融合（Median / Regression / TopToTop）。"""
    dates = [multi.base_date, multi.regress_date, multi.peak2peak_date]
    ordinals = [d.toordinal() for d in dates]
    w = np.array(weights, dtype=float)
    avg_ord = int(round(np.average(ordinals, weights=w)))
    return date.fromordinal(avg_ord)


# ========= 数据读取与对齐（与 step01 一致）=========
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


def window_halving_to_peak(
    series: pd.Series,
    halving: pd.Timestamp,
    peak: pd.Timestamp,
    post_days=0,
    max_pre=None,
):
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


def pct_curve(series: pd.Series, anchor_date: pd.Timestamp):
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


def pre_std(curve_rel_day_series: pd.Series, span=90):
    idx = [d for d in range(-span, 1) if d in curve_rel_day_series.index]
    return curve_rel_day_series.loc[idx].std(ddof=0)


def scale_factor(std_old, std_new, level_old, level_new, method="manual", alpha=0.5):
    if method == "manual":
        return (level_new / level_old) ** alpha
    if method == "std":
        return (std_new / std_old) if std_old and std_old > 0 else 1.0
    left = (std_new / std_old) if std_old and std_old > 0 else 1.0
    right = (level_new / level_old) ** alpha
    return left * right


def fit_metrics(a: pd.Series, b: pd.Series):
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        return np.nan, np.nan
    A, B = a.loc[idx].values, b.loc[idx].values
    r = float(np.corrcoef(A, B)[0, 1])
    rmse = float(np.sqrt(np.mean((A - B) ** 2)))
    return r, rmse


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

    # ---------- 1) 多模型时间预测 → 2025 峰值锚点 ----------
    multi = predict_peak_multi()
    time_res = predict_peak_window_by_time(u_factor=1.2, method="regress")
    PEAK_2025_DATE = fusion_time_center(multi, weights=(1.0, 1.3, 1.1))
    PEAK_2025 = pd.Timestamp(PEAK_2025_DATE)

    # ---------- 2) 读价格、对齐（2025 用融合锚点）----------
    s15 = to_daily(read_two_col_excel(F15))
    s21 = to_daily(read_two_col_excel(F21))
    s25 = to_daily(read_fred_excel(FRED_2025))

    win17 = window_halving_to_peak(s15, HALVING_2016, PEAK_2017, POST_DAYS, MAX_PRE_DAYS)
    curve17, _ = pct_curve(win17, PEAK_2017)
    c17 = curve17.set_index("rel_day")["dd_pct"]

    s21_year = s21.loc["2021-01-01":"2021-12-31"]
    peak21 = s21_year.idxmax() if PEAK_2021 is None else PEAK_2021
    win21 = window_halving_to_peak(s21, HALVING_2020, peak21, POST_DAYS, MAX_PRE_DAYS)
    curve21, _ = pct_curve(win21, peak21)
    c21 = curve21.set_index("rel_day")["dd_pct"]

    win25 = window_halving_to_peak(s25, HALVING_2024, PEAK_2025, POST_DAYS, MAX_PRE_DAYS)
    curve25, _ = pct_curve(win25, PEAK_2025)
    c25 = curve25.set_index("rel_day")["dd_pct"]

    d_17 = (PEAK_2017 - HALVING_2016).days
    d_21 = (peak21 - HALVING_2020).days
    d_25 = (PEAK_2025 - HALVING_2024).days

    # ---------- 3) 退火缩放 ----------
    std25 = pre_std(c25, span=min(PRE_STD_SPAN, max(5, d_25)))
    std21 = pre_std(c21, span=min(PRE_STD_SPAN, max(5, d_21)))
    std17 = pre_std(c17, span=min(PRE_STD_SPAN, max(5, d_17)))
    scale21 = scale_factor(
        std21, std25, VOL_LEVEL_2021, VOL_LEVEL_2025,
        method=SCALE_METHOD, alpha=VOL_ALPHA,
    )
    scale17 = scale_factor(
        std17, std25, VOL_LEVEL_2017, VOL_LEVEL_2025,
        method=SCALE_METHOD, alpha=VOL_ALPHA,
    )
    c21_s, c17_s = c21 * scale21, c17 * scale17

    def split_metrics(c_old):
        pre_idx = [i for i in c25.index if i <= 0 and i in c_old.index]
        post_idx = [i for i in c25.index if i >= 0 and i in c_old.index]
        return (
            fit_metrics(c25.loc[pre_idx], c_old.loc[pre_idx])
            + fit_metrics(c25.loc[post_idx], c_old.loc[post_idx])
        )

    m21_raw = split_metrics(c21)
    m17_raw = split_metrics(c17)
    m21_scal = split_metrics(c21_s)
    m17_scal = split_metrics(c17_s)

    # ---------- 4) 保存：时间模型摘要 + 对齐数据 + 图 ----------
    with open(OUTDIR / "step02_time_fusion.txt", "w", encoding="utf-8") as f:
        f.write("=== Step02 时间多模型（4 fusion 逻辑）===\n")
        f.write(f"基线（减半→顶中位数）: {multi.base_days} 天 → {multi.base_date}\n")
        f.write(f"修正A（回归预测）  : {multi.regress_days} 天 → {multi.regress_date}\n")
        f.write(f"修正B（顶→顶外推）: {multi.peak2peak_days} 天 → {multi.peak2peak_date}\n")
        f.write(f"综合预测窗口       : {multi.window[0]} → {multi.window[1]}\n")
        f.write(f"时间模型A 相关系数 r: {time_res.corr_halving_vs_delay:.3f}\n")
        f.write(f"时间模型A 窗口     : {time_res.window_lo} → {time_res.window_hi}\n")
        f.write(f"融合时间中心（用作 2025 锚点）: {PEAK_2025_DATE}\n")

    merged = (
        pd.DataFrame({"dd_pct_2025": c25})
        .join(pd.DataFrame({"dd_pct_2021": c21}), how="outer")
        .join(pd.DataFrame({"dd_pct_2017": c17}), how="outer")
    )
    merged["dd_pct_2021_scaled"] = merged["dd_pct_2021"] * scale21
    merged["dd_pct_2017_scaled"] = merged["dd_pct_2017"] * scale17
    merged.index.name = "rel_day"
    merged.to_csv(OUTDIR / "halving_to_peak_aligned_fusion.csv", encoding="utf-8-sig")

    with open(OUTDIR / "halving_to_peak_metrics_fusion.txt", "w", encoding="utf-8") as f:
        f.write(f"2025 峰值锚点（多模型融合）: {PEAK_2025_DATE}\n")
        f.write(f"减半→峰值天数：2017={d_17}，2021={d_21}，2025(融合)={d_25}\n")
        f.write(f"std(峰前)：2017={std17:.3f}，2021={std21:.3f}，2025={std25:.3f}\n")
        f.write(f"scale(method={SCALE_METHOD}, alpha={VOL_ALPHA}): 2017→25={scale17:.3f}，2021→25={scale21:.3f}\n\n")
        f.write("未缩放 r/RMSE（峰前 | 峰后）：\n")
        f.write(f"2021_raw : pre r={m21_raw[0]:.4f}, rmse={m21_raw[1]:.3f} | post r={m21_raw[2]:.4f}, rmse={m21_raw[3]:.3f}\n")
        f.write(f"2017_raw : pre r={m17_raw[0]:.4f}, rmse={m17_raw[1]:.3f} | post r={m17_raw[2]:.4f}, rmse={m17_raw[3]:.3f}\n\n")
        f.write("退火后 r/RMSE（峰前 | 峰后）：\n")
        f.write(f"2021_scal: pre r={m21_scal[0]:.4f}, rmse={m21_scal[1]:.3f} | post r={m21_scal[2]:.4f}, rmse={m21_scal[3]:.3f}\n")
        f.write(f"2017_scal: pre r={m17_scal[0]:.4f}, rmse={m17_scal[1]:.3f} | post r={m17_scal[2]:.4f}, rmse={m17_scal[3]:.3f}\n")

    # 图：标题注明 2025 锚点来自多模型融合
    title_note = f"2025锚点={PEAK_2025_DATE}（多模型融合）"

    plt.figure(figsize=(12.5, 4.5))
    plt.plot(c25.index, c25.values, label="2025（参考，融合锚点）")
    plt.plot(c21_s.index, c21_s.values, label=f"2021（缩放 ×{scale21:.2f}）")
    plt.plot(c17_s.index, c17_s.values, label=f"2017（缩放 ×{scale17:.2f}）")
    plt.axvline(0, linestyle="--")
    plt.xlabel("相对天数（锚点=0）")
    plt.ylabel("相对锚点涨跌（%）")
    plt.title(f"减半→峰值对齐（含峰后）：波动退火后 | {title_note}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "halving_to_peak_scaled_fusion.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12.5, 4.5))
    plt.plot(c25.index, c25.values, label="2025（原幅度，融合锚点）")
    plt.plot(c21.index, c21.values, label="2021（原幅度）")
    plt.plot(c17.index, c17.values, label="2017（原幅度）")
    plt.axvline(0, linestyle="--")
    plt.xlabel("相对天数（锚点=0）")
    plt.ylabel("相对锚点涨跌（%）")
    plt.title(f"减半→峰值对齐（含峰后）：原始幅度 | {title_note}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "halving_to_peak_unscaled_fusion.png", dpi=170)
    plt.close()

    print("Step02: time fusion + halving-to-peak plot")
    print("Time fusion center (2025 peak anchor):", PEAK_2025_DATE)
    print("Output dir:", OUTDIR.resolve())
    print("PNG dir:   ", PNG_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
