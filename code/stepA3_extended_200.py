"""
StepA3: 峰后时间轴使用原始天数（不缩放），但只展示峰后前 WINDOW_DAYS 天
- 2017/2021/2025 峰后都用真实天数，截取 0~WINDOW_DAYS 区间
- 默认看峰后前 600 天走势（可调 WINDOW_DAYS）
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"

WINDOW_DAYS = 600  # 峰后只看前 600 天
EXTEND_DAYS = 0    # 不再额外延伸

# 峰后缩放：以 2025 为基准（=1），只缩放 2017 / 2021
POST_SCALE_ALPHA = 1.0  # 1.0 表示按标准差完全对齐到 2025，可按需要调小

# 使用合并后的数据文件
MERGED_CSV = DATA_DIR / "btc_merged_daily.csv"
HALVING_PEAK_DATES = DATA_DIR / "btc_halving_peak_dates.xlsx"

PEAK_2025 = pd.Timestamp("2025-08-15")  # 2025 峰顶，峰前峰后都用这个日期

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
HALVING_2016 = pd.Timestamp("2016-07-09")
HALVING_2020 = pd.Timestamp("2020-05-11")
HALVING_2024 = pd.Timestamp("2024-04-20")
PEAK_2017 = pd.Timestamp("2017-12-17")
PEAK_2021 = None
POST_DAYS = 60
MAX_PRE_DAYS = 300
PRE_STD_SPAN = 90
VOL_LEVEL_2017, VOL_LEVEL_2021, VOL_LEVEL_2025 = 9.0, 3.0, 1.0
VOL_ALPHA = 0.5
SCALE_METHOD = "manual"

COLORS = {"2025": "#1a5276", "2021": "#e74c3c", "2017": "#27ae60"}


def load_merged_data() -> pd.Series:
    """加载合并后的 BTC 日线数据"""
    if not MERGED_CSV.exists():
        raise FileNotFoundError(f"请先运行 stepA3_pre_data_summary.py 生成 {MERGED_CSV}")
    df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    s = df.set_index("date")["price"]
    s.index.name = "ts"
    return s


def window_halving_to_peak(series: pd.Series, halving: pd.Timestamp, peak: pd.Timestamp, post_days=0, max_pre=None, end_date=None):
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


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%H%M%S")

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

    # 加载合并后的完整数据
    s_all = load_merged_data()
    latest_date = s_all.index.max().normalize()
    print(f"Loaded merged data: {s_all.index.min().date()} ~ {s_all.index.max().date()}")

    # 2017：峰后延伸到 2020 减半前（完整的下跌和恢复周期）
    end_17 = HALVING_2020 - pd.Timedelta(days=1)  # 2020 减半前一天
    win17 = window_halving_to_peak(s_all, HALVING_2016, PEAK_2017, POST_DAYS, MAX_PRE_DAYS, end_date=end_17)
    curve17, _ = pct_curve(win17, PEAK_2017)
    c17 = curve17.set_index("rel_day")["dd_pct"]

    # 2021：峰后延伸到 2024 减半前（完整的下跌和恢复周期）
    peak21 = pd.Timestamp("2021-11-10") if PEAK_2021 is None else PEAK_2021
    end_21 = HALVING_2024 - pd.Timedelta(days=1)  # 2024 减半前一天
    win21 = window_halving_to_peak(s_all, HALVING_2020, peak21, POST_DAYS, MAX_PRE_DAYS, end_date=end_21)
    curve21, _ = pct_curve(win21, peak21)
    c21 = curve21.set_index("rel_day")["dd_pct"]

    # 2025：峰顶 08-15，峰前峰后都正常使用
    win25 = window_halving_to_peak(s_all, HALVING_2024, PEAK_2025, POST_DAYS, MAX_PRE_DAYS, end_date=latest_date)
    curve25, anchor_px = pct_curve(win25, PEAK_2025)
    c25 = curve25.set_index("rel_day")["dd_pct"]

    d_17 = (PEAK_2017 - HALVING_2016).days
    d_21 = (peak21 - HALVING_2020).days
    d_25 = (PEAK_2025 - HALVING_2024).days

    std25 = pre_std(c25, span=min(PRE_STD_SPAN, max(5, d_25)))
    std21 = pre_std(c21, span=min(PRE_STD_SPAN, max(5, d_21)))
    std17 = pre_std(c17, span=min(PRE_STD_SPAN, max(5, d_17)))
    scale21 = scale_factor(std21, std25, VOL_LEVEL_2021, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)
    scale17 = scale_factor(std17, std25, VOL_LEVEL_2017, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)

    c17_s = c17 * scale17
    c21_s = c21 * scale21

    pre17 = c17_s.loc[c17_s.index <= 0].sort_index()
    pre21 = c21_s.loc[c21_s.index <= 0].sort_index()
    pre25 = c25.loc[c25.index <= 0].sort_index()

    # 峰后：只保留 0 ~ WINDOW_DAYS 区间（先用原始幅度）
    p17_post_raw = c17.loc[(c17.index >= 0) & (c17.index <= WINDOW_DAYS)].sort_index()
    p21_post_raw = c21.loc[(c21.index >= 0) & (c21.index <= WINDOW_DAYS)].sort_index()
    p25_post = c25.loc[(c25.index >= 0) & (c25.index <= WINDOW_DAYS)].sort_index()

    # 以 2025 峰后波动为 1，缩放 2017 / 2021 峰后幅度到接近 2025
    def _safe_std(s: pd.Series) -> float:
        return float(s.std(ddof=0)) if len(s) > 1 else float("nan")

    std25_post = _safe_std(p25_post)
    std17_post = _safe_std(p17_post_raw)
    std21_post = _safe_std(p21_post_raw)

    def _post_scale(std_cycle: float, std_base: float) -> float:
        if not std_cycle or std_cycle <= 0 or not std_base or std_base <= 0:
            return 1.0
        ratio = std_base / std_cycle
        return ratio ** POST_SCALE_ALPHA

    post_scale17 = _post_scale(std17_post, std25_post)
    post_scale21 = _post_scale(std21_post, std25_post)

    # 实际用于绘图的峰后曲线（2025 保持 1，不缩放）
    p17_post = p17_post_raw * post_scale17
    p21_post = p21_post_raw * post_scale21
    days_17 = int(p17_post.index.max()) if len(p17_post) else 0
    days_21 = int(p21_post.index.max()) if len(p21_post) else 0
    days_25 = int(p25_post.index.max()) if len(p25_post) else 0

    # 不缩放：2017/2021/2025 峰后都使用真实天数（已截断到 WINDOW_DAYS）
    x17 = p17_post.index.values
    x21 = p21_post.index.values
    x25 = p25_post.index.values

    # x 轴范围：左侧显示完整峰前，右侧固定到 WINDOW_DAYS
    x_max = WINDOW_DAYS
    x_min = int(pre25.index.min()) - 20

    plt.figure(figsize=(16, 5.5))

    # 2025：峰前+峰后连续（峰后不缩放，作为基准=1）
    x25_full = np.concatenate([pre25.index.values, x25])
    y25_full = np.concatenate([pre25.values, p25_post.values])
    plt.plot(x25_full, y25_full, color=COLORS["2025"], label=f"2025（峰顶0815 峰后{days_25}天，基准=1）", zorder=3)

    x21_full = np.concatenate([pre21.index.values, x21])
    y21_full = np.concatenate([pre21.values, p21_post.values])
    plt.plot(
        x21_full,
        y21_full,
        color=COLORS["2021"],
        label=f"2021（pre×{scale21:.2f} post×{post_scale21:.2f} 天数={days_21}）",
        zorder=2,
    )

    x17_full = np.concatenate([pre17.index.values, x17])
    y17_full = np.concatenate([pre17.values, p17_post.values])
    plt.plot(
        x17_full,
        y17_full,
        color=COLORS["2017"],
        label=f"2017（pre×{scale17:.2f} post×{post_scale17:.2f} 天数={days_17}）",
        zorder=2,
    )

    plt.axvline(0, linestyle="--", color="gray", linewidth=1.2, zorder=1)
    plt.xlim(left=x_min, right=x_max)
    plt.ylabel("相对峰顶涨跌（%）")
    plt.xlabel("左：峰前相对天数 | 右：峰后真实天数（仅前 %d 天，2025 基准=1）" % WINDOW_DAYS)
    plt.title(
        "StepA3：峰后前%d天对比（2025=1，缩放 17/21） | 2017=%d天 2021=%d天 2025=%d天 | 最新=%s"
        % (WINDOW_DAYS, days_17, days_21, days_25, latest_date.strftime("%Y-%m-%d"))
    )
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = PNG_DIR / f"stepA3_extended_200_{run_ts}.png"
    plt.savefig(out_png, dpi=170)
    plt.close()

    notes_path = OUTDIR / f"stepA3_notes_{run_ts}.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("StepA3: 峰前+峰后联合视图，峰后使用原始天数（不缩放），仅展示前 %d 天\n" % WINDOW_DAYS)
        f.write("峰后有效天数: 2017=%d, 2021=%d, 2025=%d\n" % (days_17, days_21, days_25))
        f.write("x 轴范围: %d ~ %d（左侧为峰前负天数，右侧为峰后天数）\n" % (x_min, x_max))
        f.write("峰后缩放: POST_SCALE_ALPHA=%.2f, post_scale17=%.4f, post_scale21=%.4f\n" % (POST_SCALE_ALPHA, post_scale17, post_scale21))

    print("StepA3 done. Post-peak scaled with 2025 as baseline=1, first %d days" % WINDOW_DAYS)
    print("  Post days (clipped): 2017=%d, 2021=%d, 2025=%d" % (days_17, days_21, days_25))
    print("  x_max=%d" % x_max)
    print("  PNG:", out_png.resolve())
    print("  Notes:", notes_path.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
