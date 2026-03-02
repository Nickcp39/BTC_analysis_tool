"""
Step05: 减半→峰值对比 + 周期时间缩放（step02 基础 + step03 时间缩放）
- 完成 step02 的基本：读数据、减半→峰窗口、pct 曲线、幅度缩放（可选 0.58/0.33 等）
- 在此基础上对 cycle 做时间缩放：x 轴为「缩放后天数」（参考 step03），天数不再绝对，峰前峰后同尺度对齐
- 时间缩放比：以 2025 周期为参考(1.0)，2017/2021 的 rel_day 按 本周期天数/参考周期天数 映射到同一时间轴
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 路径配置（与 step02 一致）=========
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
# 与 step04 一致：2017 峰用 12-17（step04 用 TOPS[1]）；若用 12-19 则与 step02 一致
PEAK_2017 = pd.Timestamp("2017-12-17")
PEAK_2021 = None

POST_DAYS = 60
MAX_PRE_DAYS = 300
PRE_STD_SPAN = 90

# 幅度缩放：若指定则用固定 ratio（如 0.58/0.33 让峰前完美对齐）；否则用 step02 的 level+alpha 计算
AMPLITUDE_SCALE_2017: Optional[float] = 0.58
AMPLITUDE_SCALE_2021: Optional[float] = 0.33
# 2025 自身幅度系数：<1 表示降低 2025 在图上的波动观感（把 2025 曲线整体缩小）
AMPLITUDE_SCALE_2025: float = 0.85
VOL_LEVEL_2017, VOL_LEVEL_2021, VOL_LEVEL_2025 = 9.0, 3.0, 1.0
VOL_ALPHA = 0.5
SCALE_METHOD = "manual"

# 2025 双峰/锚点与周期长度分离：
# - 曲线锚点："actual"=窗口内最高价日；"table"=data 表峰日 2025-08-15（与 step04 一致时用 "table"）
# - 周期长度 d_25："anchor"=与曲线锚点同一天；"table"=表格峰日 8 月 15
PEAK_2025_ANCHOR: str = "actual"   # 若要让 pre-peak 效果与 step04 一致，改为 "table"
PEAK_2025_DAYS_FROM: str = "table"


# ========= 数据与曲线（与 step02 一致）=========
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


def window_halving_to_peak(series: pd.Series, halving: pd.Timestamp, peak: pd.Timestamp, post_days=0, max_pre=None):
    halving = pd.Timestamp(halving).normalize()
    peak = pd.Timestamp(peak).normalize()
    full_idx = pd.date_range(series.index.min(), max(series.index.max(), peak), freq="D")
    s = series.reindex(full_idx).ffill()
    start = halving
    if max_pre is not None:
        start = max(peak - pd.Timedelta(days=max_pre), start)
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
    """
    波动率（仅峰前）：取 rel_day 在 [-span, 0] 区间内 dd_pct 的标准差（ddof=0）。
    即「峰前 span 天相对峰顶涨跌幅」的离散程度，只用于未设固定 AMPLITUDE_SCALE 时的 scale_factor 计算。
    """
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


# ========= Step03 风格：减半→顶峰 各周期天数与时间缩放比 =========
def halving_to_peak_days_ref_step03() -> Tuple[int, int, int, float, float, float]:
    """
    返回 (d_17, d_21, d_25, time_scale_17, time_scale_21, time_scale_25)。
    以 2025 周期为参考（time_scale_25=1.0），2017/2021 的 rel_day 映射为 scaled_day = rel_day * (d_25/本周期天数)。
    """
    d_17 = (PEAK_2017 - HALVING_2016).days
    # 2021 峰用数据或自动
    s21 = to_daily(read_two_col_excel(F21))
    s21_year = s21.loc["2021-01-01":"2021-12-31"]
    peak21_ts = s21_year.idxmax() if PEAK_2021 is None else pd.Timestamp(PEAK_2021)
    d_21 = (peak21_ts - HALVING_2020).days
    # 2025 峰
    peak25_date = TOPS[-1] if len(TOPS) > 3 else date(2025, 8, 15)
    PEAK_2025 = pd.Timestamp(peak25_date)
    d_25 = (PEAK_2025 - HALVING_2024).days

    # 以 2025 为参考：缩放后 1 单位 = 2025 的 1 天
    time_scale_25 = 1.0
    time_scale_17 = d_25 / d_17 if d_17 else 1.0
    time_scale_21 = d_25 / d_21 if d_21 else 1.0
    return d_17, d_21, d_25, time_scale_17, time_scale_21, time_scale_25


def apply_time_scale(rel_day_series: pd.Series, time_scale: float) -> pd.Series:
    """rel_day -> scaled_day，天数按周期长度比例缩放。"""
    return (rel_day_series * time_scale).astype(float)


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

    # ---------- 1) Step02 基本：读价格、窗口、pct 曲线 ----------
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
    # 2025 曲线锚点：actual=窗口内最高价日（真顶，如 10 月 4）；table=表格峰日（如 8 月 15「峰开始」）
    if PEAK_2025_ANCHOR == "actual" and len(win25) > 0:
        peak25_anchor = win25.idxmax()
        curve25, anchor_px25 = pct_curve(win25, peak25_anchor)
    else:
        peak25_anchor = PEAK_2025
        curve25, anchor_px25 = pct_curve(win25, PEAK_2025)
    c25 = curve25.set_index("rel_day")["dd_pct"]

    d_17 = (PEAK_2017 - HALVING_2016).days
    d_21 = (peak21 - HALVING_2020).days
    # 周期长度 d_25：anchor=与曲线锚点同一天；table=表格峰日（与「峰开始」8 月 15 一致）
    if PEAK_2025_DAYS_FROM == "anchor":
        d_25 = (pd.Timestamp(peak25_anchor) - HALVING_2024).days
    else:
        d_25 = (PEAK_2025 - HALVING_2024).days

    # ---------- 2) Step03 风格时间缩放比（以 2025 为参考，用当前 d_25 保证与锚点一致）----------
    ts25 = 1.0
    ts17 = d_25 / d_17 if d_17 else 1.0
    ts21 = d_25 / d_21 if d_21 else 1.0

    # 每条曲线的 scaled_day = rel_day * time_scale
    c17_scaled_t = apply_time_scale(c17.index.to_series(), ts17)
    c21_scaled_t = apply_time_scale(c21.index.to_series(), ts21)
    c25_scaled_t = apply_time_scale(c25.index.to_series(), ts25)

    # 用 scaled_day 做索引的序列（便于对齐画图）
    s17_t = pd.Series(c17.values, index=c17_scaled_t.values, name="dd_pct")
    s21_t = pd.Series(c21.values, index=c21_scaled_t.values, name="dd_pct")
    s25_t = pd.Series(c25.values, index=c25_scaled_t.values, name="dd_pct")

    # ---------- 3) 幅度缩放：固定 ratio 或 step02 的 scale_factor ----------
    if AMPLITUDE_SCALE_2017 is not None and AMPLITUDE_SCALE_2021 is not None:
        scale17 = AMPLITUDE_SCALE_2017
        scale21 = AMPLITUDE_SCALE_2021
    else:
        std25 = pre_std(c25, span=min(PRE_STD_SPAN, max(5, d_25)))
        std21 = pre_std(c21, span=min(PRE_STD_SPAN, max(5, d_21)))
        std17 = pre_std(c17, span=min(PRE_STD_SPAN, max(5, d_17)))
        scale21 = scale_factor(std21, std25, VOL_LEVEL_2021, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)
        scale17 = scale_factor(std17, std25, VOL_LEVEL_2017, VOL_LEVEL_2025, method=SCALE_METHOD, alpha=VOL_ALPHA)

    s17_amp = s17_t * scale17
    s21_amp = s21_t * scale21
    s25_amp = s25_t * AMPLITUDE_SCALE_2025  # 降低 2025 波动观感

    # ---------- 4) 输出与绘图 ----------
    lines = [
        "=== Step05 减半→峰值 + 周期时间缩放 ===",
        f"2025 峰值锚点(表格): {PEAK_2025_DATE}",
        f"2025 曲线锚点(画图用): {pd.Timestamp(peak25_anchor).strftime('%Y-%m-%d')}",
        f"减半→峰天数: 2017={d_17}, 2021={d_21}, 2025={d_25} (d_25来源={PEAK_2025_DAYS_FROM})",
        f"时间缩放比(以2025=1): 2017→{ts17:.4f}, 2021→{ts21:.4f}, 2025={ts25:.4f}",
        f"幅度缩放: 2017×{scale17:.3f}, 2021×{scale21:.3f}, 2025×{AMPLITUDE_SCALE_2025:.3f}",
        "",
        "【波动率计算】",
        "  pre_std: 峰前 span 天(默认90)内 dd_pct 的标准差，即「相对峰顶涨跌幅」的离散度。",
        "  仅在不设固定 AMPLITUDE_SCALE_* 时参与 scale_factor；manual 时用 VOL_LEVEL^alpha。",
        "x 轴: 缩放后天数（非绝对天数），便于峰前峰后同尺度对齐。",
    ]
    txt_path = OUTDIR / "step05_time_scaled_notes.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 图1：时间缩放 + 幅度缩放（x=scaled_day），2025 也按 AMPLITUDE_SCALE_2025 缩小
    plt.figure(figsize=(12.5, 4.5))
    plt.plot(s25_amp.index, s25_amp.values, label=f"2025（参考 幅缩×{AMPLITUDE_SCALE_2025:.2f}）")
    plt.plot(s21_amp.index, s21_amp.values, label=f"2021（时缩×{ts21:.3f} 幅缩×{scale21:.2f}）")
    plt.plot(s17_amp.index, s17_amp.values, label=f"2017（时缩×{ts17:.3f} 幅缩×{scale17:.2f}）")
    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("缩放后天数（以 2025 周期为参考，非绝对天数）")
    plt.ylabel("相对锚点涨跌（%）")
    plt.title(f"减半→峰值：周期时间缩放 + 幅度缩放 | 锚点={PEAK_2025_DATE}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "step05_scaled_time_and_amplitude.png", dpi=170)
    plt.close()

    # 图2：仅时间缩放、不缩幅度（看形态）
    plt.figure(figsize=(12.5, 4.5))
    plt.plot(s25_t.index, s25_t.values, label="2025")
    plt.plot(s21_t.index, s21_t.values, label="2021")
    plt.plot(s17_t.index, s17_t.values, label="2017")
    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("缩放后天数（以 2025 周期为参考，非绝对天数）")
    plt.ylabel("相对锚点涨跌（%）")
    plt.title(f"减半→峰值：仅周期时间缩放（原始幅度）| 锚点={PEAK_2025_DATE}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "step05_scaled_time_only.png", dpi=170)
    plt.close()

    # CSV：缩放后对齐数据（scaled_day 为公共 x，三列 dd_pct）
    all_days = np.unique(np.concatenate([
        np.round(s25_amp.index, 2),
        np.round(s21_amp.index, 2),
        np.round(s17_amp.index, 2),
    ]))
    all_days = np.sort(all_days)
    df_out = pd.DataFrame({"scaled_day": all_days})
    df_out["dd_pct_2025"] = np.interp(all_days, s25_amp.index.values, s25_amp.values)
    df_out["dd_pct_2021"] = np.interp(all_days, s21_amp.index.values, s21_amp.values)
    df_out["dd_pct_2017"] = np.interp(all_days, s17_amp.index.values, s17_amp.values)
    df_out.to_csv(OUTDIR / "step05_halving_to_peak_time_scaled.csv", index=False, encoding="utf-8-sig")

    print("Step05: halving-to-peak with cycle time scaling (step02 base + step03 time scale)")
    print("  Peak anchor:", PEAK_2025_DATE)
    print("  Days: 2017=%d, 2021=%d, 2025=%d" % (d_17, d_21, d_25))
    print("  Time scale (ref=2025): 2017=%.4f, 2021=%.4f" % (ts17, ts21))
    print("  Amplitude scale: 2017=%.2f, 2021=%.2f, 2025=%.2f" % (scale17, scale21, AMPLITUDE_SCALE_2025))
    print("  2025 curve anchor:", pd.Timestamp(peak25_anchor).strftime("%Y-%m-%d"), "| d_25 from:", PEAK_2025_DAYS_FROM)
    print("Output:", OUTDIR.resolve())
    print("PNG:   ", PNG_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
