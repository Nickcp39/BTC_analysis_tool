"""
Step01: 减半→峰值对比图（与 notebook 同逻辑，用最新数据 + 历史数据）
- 输入：data/btc_2015.xlsx, data/btc_2021.xlsx, data/btc_price_fred.xlsx（最新）
- 输出：visualization/YYYY-MM-DD/（CSV、TXT）、visualization/YYYY-MM-DD/png/（PNG）
"""
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 路径配置 =========
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"

# 输入文件（2017/2021 用历史表，2025 用最新 FRED）
F15 = DATA_DIR / "btc_2015.xlsx"
F21 = DATA_DIR / "btc_2021.xlsx"
FRED_2025 = DATA_DIR / "btc_price_fred.xlsx"

# 减半 & 峰值
HALVING_2016 = pd.Timestamp("2016-07-09")
HALVING_2020 = pd.Timestamp("2020-05-11")
HALVING_2024 = pd.Timestamp("2024-04-20")
PEAK_2017 = pd.Timestamp("2017-12-19")
PEAK_2021 = None  # None = 自动取 2021 年最高日
PEAK_2025 = pd.Timestamp("2025-08-15")  # 假设峰值
POST_DAYS = 60
MAX_PRE_DAYS = 300

# 波动退火
VOL_LEVEL_2017, VOL_LEVEL_2021, VOL_LEVEL_2025 = 9.0, 3.0, 1.0
VOL_ALPHA = 0.5
SCALE_METHOD = "manual"
PRE_STD_SPAN = 90


def read_two_col_excel(fp: Path) -> pd.DataFrame:
    """两列 Excel（时间、价格），兼容横排两行。"""
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
    """FRED 格式 Excel：第一列日期、第二列价格（如 observation_date, CBBTCUSD）。"""
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

    # 中文与负号（Windows）
    try:
        from matplotlib import rcParams, font_manager
        for p in [
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
        ]:
            if p.exists():
                font_manager.fontManager.addfont(str(p))
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # 读取：2017/2021 两列表，2025 用 FRED 最新
    s15 = to_daily(read_two_col_excel(F15))
    s21 = to_daily(read_two_col_excel(F21))
    s25 = to_daily(read_fred_excel(FRED_2025))

    # 三段：减半→峰值（含峰后）
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

    # 退火缩放
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

    # 保存 CSV、TXT 到 visualization/YYYY-MM-DD/
    merged = (
        pd.DataFrame({"dd_pct_2025": c25})
        .join(pd.DataFrame({"dd_pct_2021": c21}), how="outer")
        .join(pd.DataFrame({"dd_pct_2017": c17}), how="outer")
    )
    merged["dd_pct_2021_scaled"] = merged["dd_pct_2021"] * scale21
    merged["dd_pct_2017_scaled"] = merged["dd_pct_2017"] * scale17
    merged.index.name = "rel_day"
    merged.to_csv(OUTDIR / "halving_to_peak_aligned.csv", encoding="utf-8-sig")

    with open(OUTDIR / "halving_to_peak_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"减半→峰值天数：2017={d_17}，2021={d_21}，2025(假设)={d_25}\n")
        f.write(f"std(峰前)：2017={std17:.3f}，2021={std21:.3f}，2025={std25:.3f}\n")
        f.write(f"scale(method={SCALE_METHOD}, alpha={VOL_ALPHA}): 2017→25={scale17:.3f}，2021→25={scale21:.3f}\n\n")
        f.write("未缩放 r/RMSE（峰前 | 峰后）：\n")
        f.write(f"2021_raw : pre r={m21_raw[0]:.4f}, rmse={m21_raw[1]:.3f} | post r={m21_raw[2]:.4f}, rmse={m21_raw[3]:.3f}\n")
        f.write(f"2017_raw : pre r={m17_raw[0]:.4f}, rmse={m17_raw[1]:.3f} | post r={m17_raw[2]:.4f}, rmse={m17_raw[3]:.3f}\n\n")
        f.write("退火后 r/RMSE（峰前 | 峰后）：\n")
        f.write(f"2021_scal: pre r={m21_scal[0]:.4f}, rmse={m21_scal[1]:.3f} | post r={m21_scal[2]:.4f}, rmse={m21_scal[3]:.3f}\n")
        f.write(f"2017_scal: pre r={m17_scal[0]:.4f}, rmse={m17_scal[1]:.3f} | post r={m17_scal[2]:.4f}, rmse={m17_scal[3]:.3f}\n")

    # 图保存到 visualization/YYYY-MM-DD/png/
    plt.figure(figsize=(12.5, 4.5))
    plt.plot(c25.index, c25.values, label="2025（参考）")
    plt.plot(c21_s.index, c21_s.values, label=f"2021（缩放 ×{scale21:.2f}）")
    plt.plot(c17_s.index, c17_s.values, label=f"2017（缩放 ×{scale17:.2f}）")
    plt.axvline(0, linestyle="--")
    plt.xlabel("相对天数（锚点=0）")
    plt.ylabel("相对锚点涨跌（%）")
    plt.title("减半→峰值对齐（含峰后）：波动退火后")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "halving_to_peak_scaled.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12.5, 4.5))
    plt.plot(c25.index, c25.values, label="2025（原幅度）")
    plt.plot(c21.index, c21.values, label="2021（原幅度）")
    plt.plot(c17.index, c17.values, label="2017（原幅度）")
    plt.axvline(0, linestyle="--")
    plt.xlabel("相对天数（锚点=0）")
    plt.ylabel("相对锚点涨跌（%）")
    plt.title("减半→峰值对齐（含峰后）：原始幅度")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PNG_DIR / "halving_to_peak_unscaled.png", dpi=170)
    plt.close()

    print("Output dir:", OUTDIR.resolve())
    print("PNG dir:   ", PNG_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
