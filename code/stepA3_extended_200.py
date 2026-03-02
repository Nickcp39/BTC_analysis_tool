"""
StepA3: 同 A2 逻辑，但峰后 x 轴向右延伸 200 天（不以 2025 数据末端为终点）
- 颜色与 A2 一致
- 峰后天数轴：0 ~ N_ref + 200，留出未来空间

TODO/FIXME: 此处有问题待修复（依赖 A2 逻辑，需一并解决）
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

EXTEND_DAYS = 200

F15 = DATA_DIR / "btc_2015.xlsx"
F21 = DATA_DIR / "btc_2021.xlsx"
FRED_2025 = DATA_DIR / "btc_price_fred.xlsx"
HALVING_PEAK_DATES = DATA_DIR / "btc_halving_peak_dates.xlsx"

PEAK_2025_PRE = pd.Timestamp("2025-08-15")
PEAK_2025_POST_START = pd.Timestamp("2025-10-03")

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

    s15 = to_daily(read_two_col_excel(F15))
    s21 = to_daily(read_two_col_excel(F21))
    s25 = to_daily(read_fred_excel(FRED_2025))
    latest_25 = s25.index.max().normalize()

    # 2017 / 2021：用全部可用峰后数据（不截断）
    win17 = window_halving_to_peak(s15, HALVING_2016, PEAK_2017, POST_DAYS, MAX_PRE_DAYS, end_date=s15.index.max())
    curve17, _ = pct_curve(win17, PEAK_2017)
    c17 = curve17.set_index("rel_day")["dd_pct"]

    s21_year = s21.loc["2021-01-01":"2021-12-31"]
    peak21 = s21_year.idxmax() if PEAK_2021 is None else PEAK_2021
    win21 = window_halving_to_peak(s21, HALVING_2020, peak21, POST_DAYS, MAX_PRE_DAYS, end_date=s21.index.max())
    curve21, _ = pct_curve(win21, peak21)
    c21 = curve21.set_index("rel_day")["dd_pct"]

    win25_pre = window_halving_to_peak(s25, HALVING_2024, PEAK_2025_PRE, 0, MAX_PRE_DAYS, end_date=PEAK_2025_PRE)
    curve25_pre, anchor_px = pct_curve(win25_pre, PEAK_2025_PRE)
    c25_pre = curve25_pre.set_index("rel_day")["dd_pct"]

    win25_post = s25.loc[PEAK_2025_POST_START:latest_25]
    if len(win25_post) > 0:
        rel_day_post = (win25_post.index - PEAK_2025_PRE).days
        dd_pct_post = (win25_post.values / anchor_px - 1.0) * 100.0
        c25_post = pd.Series(dd_pct_post, index=rel_day_post).sort_index()
    else:
        c25_post = pd.Series(dtype=float)

    d_17 = (PEAK_2017 - HALVING_2016).days
    d_21 = (peak21 - HALVING_2020).days
    d_25 = (PEAK_2025_PRE - HALVING_2024).days

    std25 = pre_std(c25_pre, span=min(PRE_STD_SPAN, max(5, d_25)))
    std21 = pre_std(c21, span=min(PRE_STD_SPAN, max(5, d_21)))
    std17 = pre_std(c17, span=min(PRE_STD_SPAN, max(5, d_17)))
    scale21 = scale_factor(std21, std25, VOL_LEVEL_2021, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)
    scale17 = scale_factor(std17, std25, VOL_LEVEL_2017, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)

    c17_s = c17 * scale17
    c21_s = c21 * scale21

    pre17 = c17_s.loc[c17_s.index <= 0].sort_index()
    pre21 = c21_s.loc[c21_s.index <= 0].sort_index()
    pre25 = c25_pre.loc[c25_pre.index <= 0].sort_index()

    p17_post = c17.loc[c17.index >= 0].sort_index()
    p21_post = c21.loc[c21.index >= 0].sort_index()
    days_17 = int(p17_post.index.max()) if len(p17_post) else 0
    days_21 = int(p21_post.index.max()) if len(p21_post) else 0
    days_25 = int(c25_post.index.max()) if len(c25_post) else 0
    N_ref = days_25 if days_25 > 0 else max(days_17, days_21, 1)

    scale_t17 = N_ref / days_17 if days_17 else 1.0
    scale_t21 = N_ref / days_21 if days_21 else 1.0
    x17 = p17_post.index.values * scale_t17
    x21 = p21_post.index.values * scale_t21

    x_max = N_ref + EXTEND_DAYS
    x25_post = np.linspace(1, N_ref, len(c25_post)) if len(c25_post) > 1 else np.array([1.0])

    plt.figure(figsize=(16, 5.5))

    x25_pre = pre25.index.values
    y25_pre = pre25.values
    plt.plot(x25_pre, y25_pre, color=COLORS["2025"], zorder=3)
    plt.plot(x25_post, c25_post.values, color=COLORS["2025"], label="2025（峰前->0815 峰后->1003 中间跳过）", zorder=3)

    x21_full = np.concatenate([pre21.index.values, x21])
    y21_full = np.concatenate([pre21.values, p21_post.values])
    plt.plot(x21_full, y21_full, color=COLORS["2021"], label=f"2021（峰前x{scale21:.2f} 峰后原幅）", zorder=2)

    x17_full = np.concatenate([pre17.index.values, x17])
    y17_full = np.concatenate([pre17.values, p17_post.values])
    plt.plot(x17_full, y17_full, color=COLORS["2017"], label=f"2017（峰前x{scale17:.2f} 峰后原幅）", zorder=2)

    plt.axvline(0, linestyle="--", color="gray", linewidth=1.2, zorder=1)
    plt.xlim(left=pre25.index.min() - 20, right=x_max)
    plt.ylabel("相对峰顶涨跌（%）")
    plt.xlabel("左：峰前相对天数 | 右：峰后天数（延伸 +%d 天）" % EXTEND_DAYS)
    plt.title("StepA3：同 A2，峰后轴延伸 %d 天 | 最新=%s" % (EXTEND_DAYS, latest_25.strftime("%Y-%m-%d")))
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = PNG_DIR / "stepA3_extended_200.png"
    plt.savefig(out_png, dpi=170)
    plt.close()

    with open(OUTDIR / "stepA3_notes.txt", "w", encoding="utf-8") as f:
        f.write("StepA3: 同 A2，峰后 x 轴延伸 %d 天至 %d\n" % (EXTEND_DAYS, x_max))
        f.write("2025 数据末端: %d 天\n" % N_ref)

    print("StepA3 done. Post-peak axis extended to %d days" % x_max)
    print("  PNG:", out_png.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
