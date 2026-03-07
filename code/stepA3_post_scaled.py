"""
StepA3_post_scaled: 峰前按 A3 的方法缩放到 2025 波动等级，峰后再做一次递减缩放
- 峰前：沿用 stepA3 的缩放（2017/2021 分别乘以 scale17/scale21）
- 峰后：在原始 dd_pct 基础上，再乘以一个 post_gamma（<1），体现“同一周期内峰后 < 峰前”
- 同时保持跨周期排序：2017 波动 > 2021 波动 > 2025 波动
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"

WINDOW_DAYS = 600      # 峰后只看前 600 天
POST_GAMMA = 0.65      # 峰后相对于峰前的衰减系数（同一周期内 <1）

MERGED_CSV = DATA_DIR / "btc_merged_daily.csv"
HALVING_PEAK_DATES = DATA_DIR / "btc_halving_peak_dates.xlsx"

PEAK_2025 = pd.Timestamp("2025-08-15")

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

    s_all = load_merged_data()
    latest_date = s_all.index.max().normalize()
    print(f"Loaded merged data: {s_all.index.min().date()} ~ {s_all.index.max().date()}")

    # 2017：窗口到 2020 减半前
    end_17 = HALVING_2020 - pd.Timedelta(days=1)
    win17 = window_halving_to_peak(s_all, HALVING_2016, PEAK_2017, POST_DAYS, MAX_PRE_DAYS, end_date=end_17)
    curve17, _ = pct_curve(win17, PEAK_2017)
    c17 = curve17.set_index("rel_day")["dd_pct"]

    # 2021：窗口到 2024 减半前
    peak21 = pd.Timestamp("2021-11-10") if PEAK_2021 is None else PEAK_2021
    end_21 = HALVING_2024 - pd.Timedelta(days=1)
    win21 = window_halving_to_peak(s_all, HALVING_2020, peak21, POST_DAYS, MAX_PRE_DAYS, end_date=end_21)
    curve21, _ = pct_curve(win21, peak21)
    c21 = curve21.set_index("rel_day")["dd_pct"]

    # 2025：窗口到最新
    win25 = window_halving_to_peak(s_all, HALVING_2024, PEAK_2025, POST_DAYS, MAX_PRE_DAYS, end_date=latest_date)
    curve25, _ = pct_curve(win25, PEAK_2025)
    c25 = curve25.set_index("rel_day")["dd_pct"]

    d_17 = (PEAK_2017 - HALVING_2016).days
    d_21 = (peak21 - HALVING_2020).days
    d_25 = (PEAK_2025 - HALVING_2024).days

    std25_pre = pre_std(c25, span=min(PRE_STD_SPAN, max(5, d_25)))
    std21_pre = pre_std(c21, span=min(PRE_STD_SPAN, max(5, d_21)))
    std17_pre = pre_std(c17, span=min(PRE_STD_SPAN, max(5, d_17)))

    scale21 = scale_factor(std21_pre, std25_pre, VOL_LEVEL_2021, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)
    scale17 = scale_factor(std17_pre, std25_pre, VOL_LEVEL_2017, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)

    # 峰前缩放（与 A3 一致）
    c17_pre_scaled = c17 * scale17
    c21_pre_scaled = c21 * scale21

    pre17 = c17_pre_scaled.loc[c17_pre_scaled.index <= 0].sort_index()
    pre21 = c21_pre_scaled.loc[c21_pre_scaled.index <= 0].sort_index()
    pre25 = c25.loc[c25.index <= 0].sort_index()

    # 峰后 0~WINDOW_DAYS，先取原始，再做二次缩放
    p17_post_raw = c17.loc[(c17.index >= 0) & (c17.index <= WINDOW_DAYS)].sort_index()
    p21_post_raw = c21.loc[(c21.index >= 0) & (c21.index <= WINDOW_DAYS)].sort_index()
    p25_post_raw = c25.loc[(c25.index >= 0) & (c25.index <= WINDOW_DAYS)].sort_index()

    # 峰后缩放：在原始基础上乘以 post_gamma * pre_scale
    post_scale17 = scale17 * POST_GAMMA
    post_scale21 = scale21 * POST_GAMMA
    post_scale25 = 1.0 * POST_GAMMA  # 2025 也可以略微衰减，保持同一逻辑

    p17_post = p17_post_raw * post_scale17
    p21_post = p21_post_raw * post_scale21
    p25_post = p25_post_raw * post_scale25

    days_17 = int(p17_post.index.max()) if len(p17_post) else 0
    days_21 = int(p21_post.index.max()) if len(p21_post) else 0
    days_25 = int(p25_post.index.max()) if len(p25_post) else 0

    x17 = p17_post.index.values
    x21 = p21_post.index.values
    x25 = p25_post.index.values

    x_max = WINDOW_DAYS

    plt.figure(figsize=(16, 5.5))

    # 2025：峰前+峰后
    x25_full = np.concatenate([pre25.index.values, x25])
    y25_full = np.concatenate([pre25.values * post_scale25, p25_post.values])  # 峰前乘同一 post_scale25 以保持连续感
    plt.plot(x25_full, y25_full, color=COLORS["2025"], label=f"2025（峰顶0815 峰后≤{days_25}天）", zorder=3)

    x21_full = np.concatenate([pre21.index.values, x21])
    y21_full = np.concatenate([pre21.values * POST_GAMMA, p21_post.values])
    plt.plot(x21_full, y21_full, color=COLORS["2021"], label=f"2021（pre×{scale21:.2f} post×{post_scale21:.2f}）", zorder=2)

    x17_full = np.concatenate([pre17.index.values, x17])
    y17_full = np.concatenate([pre17.values * POST_GAMMA, p17_post.values])
    plt.plot(x17_full, y17_full, color=COLORS["2017"], label=f"2017（pre×{scale17:.2f} post×{post_scale17:.2f}）", zorder=2)

    plt.axvline(0, linestyle="--", color="gray", linewidth=1.2, zorder=1)
    plt.xlim(left=pre25.index.min() - 20, right=x_max)
    plt.ylabel("相对峰顶涨跌（%）")
    plt.xlabel("左：峰前相对天数 | 右：峰后真实天数（缩放后，仅前 %d 天）" % WINDOW_DAYS)
    plt.title(
        "StepA3_post_scaled：峰前按 A3 缩放，峰后额外递减 | γ=%.2f | 最新=%s"
        % (POST_GAMMA, latest_date.strftime("%Y-%m-%d"))
    )
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = PNG_DIR / f"stepA3_post_scaled_{run_ts}.png"
    plt.savefig(out_png, dpi=170)
    plt.close()

    notes_path = OUTDIR / f"stepA3_post_scaled_{run_ts}.txt"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("StepA3_post_scaled: 峰前按 A3 缩放，峰后额外递减\\n")
        f.write("POST_GAMMA=%.2f\\n" % POST_GAMMA)
        f.write("峰后有效天数(0-%d): 2017=%d, 2021=%d, 2025=%d\\n" % (WINDOW_DAYS, days_17, days_21, days_25))

    print("StepA3_post_scaled done. POST_GAMMA=%.2f" % POST_GAMMA)
    print("  Post days (clipped): 2017=%d, 2021=%d, 2025=%d" % (days_17, days_21, days_25))
    print("  PNG:", out_png.resolve())
    print("  Notes:", notes_path.resolve())


if __name__ == "__main__":
    main()

