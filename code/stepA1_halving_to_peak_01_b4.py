"""
StepA1: step01 + stepB4 结合 — 峰前 step01 波动退火，峰后 B4 天数以 2025 到最新为准、分开画
- 数据与锚点：2017 用 2017-12-17（与 step04 一致），2021 用 idxmax，2025 窗口到最新、锚点用窗口内最高价日
- 幅度：step01 的 VOL_LEVEL 退火（2017/2021 相对 2025 缩放），整图统一
- 输出：一张完美图 = 上子图峰前（rel_day 负到 0）、下子图峰后（x 0~N_ref，N_ref=2025 峰后天数）
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import List, Optional
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
# 峰锚点：用 08-15 则峰后波动率相对 8 月 15 日算
PEAK_2025_ANCHOR = "table"


def read_two_col_excel(fp: Path) -> pd.DataFrame:
    assert fp.exists(), f"未找到：{fp}"
    df = pd.read_excel(fp, engine="openpyxl", header=None)
    if df.shape[0] <= 3 and df.shape[1] > df.shape[0]:
        df = df.T
    df = df.iloc[:, :2].copy()
    df.columns = ["ts", "price"]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna().drop_duplicates("ts").sort_values("ts")


def read_fred_excel(fp: Path) -> pd.DataFrame:
    assert fp.exists(), f"未找到：{fp}"
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


def post_std(curve_rel_day_series: pd.Series):
    idx = curve_rel_day_series.index[curve_rel_day_series.index >= 0]
    if len(idx) < 2:
        return float("nan")
    return float(curve_rel_day_series.loc[idx].std(ddof=0))


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

    PEAK_2025 = pd.Timestamp(_TOPS[-1] if len(_TOPS) > 3 else date(2025, 8, 15))
    s15 = to_daily(read_two_col_excel(F15))
    s21 = to_daily(read_two_col_excel(F21))
    s25 = to_daily(read_fred_excel(FRED_2025))
    latest_25 = s25.index.max().normalize()

    # 2017 / 2021：step01 风格窗口
    win17 = window_halving_to_peak(s15, HALVING_2016, PEAK_2017, POST_DAYS, MAX_PRE_DAYS)
    curve17, _ = pct_curve(win17, PEAK_2017)
    c17 = curve17.set_index("rel_day")["dd_pct"]

    s21_year = s21.loc["2021-01-01":"2021-12-31"]
    peak21 = s21_year.idxmax() if PEAK_2021 is None else PEAK_2021
    win21 = window_halving_to_peak(s21, HALVING_2020, peak21, POST_DAYS, MAX_PRE_DAYS)
    curve21, _ = pct_curve(win21, peak21)
    c21 = curve21.set_index("rel_day")["dd_pct"]

    # 2025：窗口到最新；锚点用 08-15（table）以便峰后波动率相对 8 月 15 日算
    win25 = window_halving_to_peak(s25, HALVING_2024, PEAK_2025, POST_DAYS, MAX_PRE_DAYS, end_date=latest_25)
    if PEAK_2025_ANCHOR == "actual" and len(win25) > 0:
        peak25_anchor = win25.idxmax()
        curve25, _ = pct_curve(win25, peak25_anchor)
    else:
        peak25_anchor = PEAK_2025  # 2025-08-15
        curve25, _ = pct_curve(win25, PEAK_2025)
    c25 = curve25.set_index("rel_day")["dd_pct"]

    d_17 = (PEAK_2017 - HALVING_2016).days
    d_21 = (peak21 - HALVING_2020).days
    d_25 = (pd.Timestamp(peak25_anchor) - HALVING_2024).days

    # step01 幅度退火（整图统一，峰前峰后都用同一缩放）
    std25 = pre_std(c25, span=min(PRE_STD_SPAN, max(5, d_25)))
    std21 = pre_std(c21, span=min(PRE_STD_SPAN, max(5, d_21)))
    std17 = pre_std(c17, span=min(PRE_STD_SPAN, max(5, d_17)))
    scale21 = scale_factor(std21, std25, VOL_LEVEL_2021, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)
    scale17 = scale_factor(std17, std25, VOL_LEVEL_2017, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)

    c17_s = c17 * scale17
    c21_s = c21 * scale21

    # 峰后波动率：用与图中一致的缩放后曲线算（2017/2021 已×scale，2025 未缩）
    vol_post_17 = post_std(c17_s)
    vol_post_21 = post_std(c21_s)
    vol_post_25 = post_std(c25)
    vol_post_17_raw = post_std(c17)
    vol_post_21_raw = post_std(c21)
    vol_post_25_raw = post_std(c25)

    # ----- 峰前：rel_day <= 0 -----
    pre17 = c17_s.loc[c17_s.index <= 0].sort_index()
    pre21 = c21_s.loc[c21_s.index <= 0].sort_index()
    pre25 = c25.loc[c25.index <= 0].sort_index()

    # ----- 峰后：与 B4 一致用原始幅度（不缩放），这样峰后形态和 B4 一样好看 -----
    p17_post = c17.loc[c17.index >= 0].sort_index()   # raw，与 B4 一致
    p21_post = c21.loc[c21.index >= 0].sort_index()
    p25 = c25.loc[c25.index >= 0].sort_index()

    days_17 = int(p17_post.index.max()) if len(p17_post) else 0
    days_21 = int(p21_post.index.max()) if len(p21_post) else 0
    days_25 = int(p25.index.max()) if len(p25) else 0
    N_ref = days_25 if days_25 > 0 else max(days_17, days_21, 1)

    scale_t17 = N_ref / days_17 if days_17 else 1.0
    scale_t21 = N_ref / days_21 if days_21 else 1.0
    x17 = p17_post.index.values * scale_t17
    x21 = p21_post.index.values * scale_t21
    x25 = p25.index.values

    # ----- 一张图：峰前 step01 缩放，峰后 B4 原幅（与 B4 图一致） -----
    plt.figure(figsize=(13, 5.5))
    # 2025：峰前 rel_day<=0，峰后 0~N_ref
    x25_full = np.concatenate([pre25.index.values, x25])
    y25_full = np.concatenate([pre25.values, p25.values])
    plt.plot(x25_full, y25_full, label=f"2025（参考）", zorder=3)
    # 2021：峰前缩放，峰后原幅
    x21_full = np.concatenate([pre21.index.values, x21])
    y21_full = np.concatenate([pre21.values, p21_post.values])
    plt.plot(x21_full, y21_full, label=f"2021（峰前×{scale21:.2f} 峰后原幅）", zorder=2)
    # 2017：峰前缩放，峰后原幅
    x17_full = np.concatenate([pre17.index.values, x17])
    y17_full = np.concatenate([pre17.values, p17_post.values])
    plt.plot(x17_full, y17_full, label=f"2017（峰前×{scale17:.2f} 峰后原幅）", zorder=2)

    plt.axvline(0, linestyle="--", color="gray", linewidth=1.2, zorder=1)
    plt.ylabel("相对峰顶涨跌（%）")
    plt.xlabel("左：峰前相对天数（锚点=0） | 右：峰后天数（0~N_ref，与 2025 一致）")
    plt.title(f"StepA1：01+B4 一张图 | 峰前 step01 退火，峰后 B4 天数对齐 | 锚点={pd.Timestamp(peak25_anchor).strftime('%Y-%m-%d')} 最新={latest_25.strftime('%Y-%m-%d')}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = PNG_DIR / "stepA1_01_b4_combined.png"
    plt.savefig(out_png, dpi=170)
    plt.close()

    # 说明文件
    lines = [
        "=== StepA1 step01 + stepB4 结合 ===",
        f"2025 曲线锚点: {pd.Timestamp(peak25_anchor).strftime('%Y-%m-%d')}",
        f"2025 数据最新: {latest_25.strftime('%Y-%m-%d')}",
        f"减半→峰天数: 2017={d_17}, 2021={d_21}, 2025={d_25}",
        f"幅度缩放(step01): 2017×{scale17:.3f}, 2021×{scale21:.3f}",
        f"峰后天数: 2017={days_17}, 2021={days_21}, 2025={days_25} (N_ref)",
        "峰后波动率(峰后与 B4 一致用原幅): 2017={vol_post_17_raw:.4f}, 2021={vol_post_21_raw:.4f}, 2025={vol_post_25_raw:.4f}",
        "",
        "图: 峰前 step01 缩放，峰后 B4 原幅（与 B4 图一致）。左=峰前，右=峰后，0=峰顶。",
    ]
    with open(OUTDIR / "stepA1_notes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("StepA1 (01+B4) done. 2025 peak = 08-15")
    print("  Pre-peak: step01 amplitude scale 2017=%.2f, 2021=%.2f" % (scale17, scale21))
    print("  Post-peak days: 2017=%d, 2021=%d, 2025=%d" % (days_17, days_21, days_25))
    print("  Post-peak: raw amplitude (same as B4). Vol: 2017=%.4f, 2021=%.4f, 2025=%.4f" % (vol_post_17_raw, vol_post_21_raw, vol_post_25_raw))
    print("  PNG:", out_png.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
