"""
StepD1: 核心点位检测器（候选点 → 人工定稿用）
- 目标：把"人眼挑核心点"算法化。两套方法、多档粒度，给人眼对照挑选。
    1) ZigZag（对数价格反转阈值 θ）：相对上一拐点反转 ≥ θ% 才算新核心点。一个旋钮。
    2) PIP（Perceptually Important Points）：取前 N 个视觉最显著的点。
- 输入：data/btc_merged_daily.csv（连续日频，2014→2026）
- 输出：output/core_points/ 下的 PNG 叠图 + 候选点 CSV
- 间隔口径：相邻核心点逐个量（gap_days、与上一段的比值）
说明：起点/末点是"试探点"（非反转确认），CSV 里 confirmed=False 标注。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:  # Windows 控制台默认 cp1252，强制 UTF-8 以便打印中文
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SERIES_CSV = ROOT / "data" / "btc_merged_daily.csv"
OUTDIR = ROOT / "output" / "core_points"

# 旋钮：ZigZag 反转阈值（小数，0.35 = 35%）；PIP 点数
ZIGZAG_THRESHOLDS = [0.25, 0.35, 0.50]
ZIGZAG_PRIMARY = 0.35
PIP_NS = [10, 14]


def setup_cjk_font():
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


def read_series() -> pd.Series:
    df = pd.read_csv(SERIES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    s = df.dropna(subset=["price"]).set_index("date")["price"].sort_index()
    # 已是日频；补齐缺口以防万一
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()
    s.index.name = "date"
    return s


def zigzag_idx(prices: np.ndarray, pct: float) -> list:
    """对数价格等价：用相对上一极值的百分比反转。返回拐点位置下标列表。"""
    n = len(prices)
    if n == 0:
        return []
    piv = [0]
    direction = 0  # 0 未定, 1 上, -1 下
    ext_i, ext_p = 0, prices[0]
    for i in range(1, n):
        p = prices[i]
        if direction == 0:
            if p >= prices[0] * (1 + pct):
                direction, ext_i, ext_p = 1, i, p
            elif p <= prices[0] * (1 - pct):
                direction, ext_i, ext_p = -1, i, p
        elif direction == 1:  # 上行：追最高
            if p > ext_p:
                ext_i, ext_p = i, p
            elif p <= ext_p * (1 - pct):  # 反转确认 → ext 是高点
                piv.append(ext_i)
                direction, ext_i, ext_p = -1, i, p
        else:  # 下行：追最低
            if p < ext_p:
                ext_i, ext_p = i, p
            elif p >= ext_p * (1 + pct):  # 反转确认 → ext 是低点
                piv.append(ext_i)
                direction, ext_i, ext_p = 1, i, p
    if piv[-1] != ext_i:
        piv.append(ext_i)
    return piv


def pip_idx(logp: np.ndarray, n_points: int) -> list:
    """PIP：迭代加入到当前折线垂直距离最大的点（对数价格上算距离）。"""
    n = len(logp)
    idxs = [0, n - 1]
    while len(idxs) < n_points:
        s = sorted(idxs)
        best_d, best_i = -1.0, None
        for a, b in zip(s[:-1], s[1:]):
            if b - a < 2:
                continue
            ya, yb = logp[a], logp[b]
            for j in range(a + 1, b):
                yline = ya + (yb - ya) * (j - a) / (b - a)
                d = abs(logp[j] - yline)
                if d > best_d:
                    best_d, best_i = d, j
        if best_i is None:
            break
        idxs.append(best_i)
    return sorted(idxs)


def classify(s: pd.Series, idxs: list) -> pd.DataFrame:
    """给每个拐点标 H/L（按邻点比较），并算相邻间隔与比值。"""
    idxs = sorted(set(idxs))
    rows = []
    for k, i in enumerate(idxs):
        price = float(s.iloc[i])
        if k == 0:
            typ = "H" if price > float(s.iloc[idxs[1]]) else "L"
        elif k == len(idxs) - 1:
            typ = "H" if price > float(s.iloc[idxs[-2]]) else "L"
        else:
            lo, hi = float(s.iloc[idxs[k - 1]]), float(s.iloc[idxs[k + 1]])
            typ = "H" if price >= lo and price >= hi else "L"
        rows.append({"i": k, "date": s.index[i], "price": round(price, 1), "type": typ})
    df = pd.DataFrame(rows)
    df["gap_days"] = df["date"].diff().dt.days
    df["ratio_vs_prev"] = (df["gap_days"] / df["gap_days"].shift(1)).round(3)
    df["confirmed"] = True
    if len(df):
        df.loc[df.index[0], "confirmed"] = False   # 起点：试探
        df.loc[df.index[-1], "confirmed"] = False   # 末点：当前未确认反转
    return df


def plot_overlay(s: pd.Series, df: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(15, 6))
    plt.plot(s.index, s.values, lw=0.8, color="#b0b0b0", zorder=1)
    plt.yscale("log")
    hi = df[df["type"] == "H"]
    lo = df[df["type"] == "L"]
    plt.scatter(hi["date"], hi["price"], s=70, c="#1db954", marker="o",
                edgecolors="k", linewidths=0.6, zorder=3, label="核心高点 H")
    plt.scatter(lo["date"], lo["price"], s=70, c="#2ecc71", marker="o",
                edgecolors="k", linewidths=0.6, zorder=3, label="核心低点 L")
    for _, r in df.iterrows():
        plt.annotate(f"{int(r['i'])}\n{r['date']:%y-%m}",
                     (r["date"], r["price"]),
                     textcoords="offset points", xytext=(0, 8 if r["type"] == "H" else -22),
                     ha="center", fontsize=7.5, color="#0a5")
    plt.title(title)
    plt.ylabel("价格（对数）")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()
    prices = s.values
    logp = np.log(prices)
    print(f"数据：{s.index.min():%Y-%m-%d} → {s.index.max():%Y-%m-%d}，{len(s)} 天\n")

    # ZigZag 多档
    for th in ZIGZAG_THRESHOLDS:
        idxs = zigzag_idx(prices, th)
        df = classify(s, idxs)
        tag = f"zigzag_{int(th*100)}pct"
        plot_overlay(s, df, f"ZigZag 候选核心点（反转阈值 {int(th*100)}%）：{len(df)} 个点",
                     OUTDIR / f"{tag}.png")
        df.to_csv(OUTDIR / f"{tag}.csv", index=False, encoding="utf-8-sig")
        print(f"[ZigZag {int(th*100)}%] {len(df)} 个候选点 → {tag}.png / .csv")

    # PIP 多档
    for npt in PIP_NS:
        idxs = pip_idx(logp, npt)
        df = classify(s, idxs)
        tag = f"pip_{npt}"
        plot_overlay(s, df, f"PIP 候选核心点（前 {npt} 个最显著）：{len(df)} 个点",
                     OUTDIR / f"{tag}.png")
        df.to_csv(OUTDIR / f"{tag}.csv", index=False, encoding="utf-8-sig")
        print(f"[PIP N={npt}] {len(df)} 个候选点 → {tag}.png / .csv")

    # 打印主档清单供直接审阅
    idxs = zigzag_idx(prices, ZIGZAG_PRIMARY)
    dfp = classify(s, idxs)
    print(f"\n===== 主档候选：ZigZag {int(ZIGZAG_PRIMARY*100)}% =====")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(dfp[["i", "date", "price", "type", "gap_days", "ratio_vs_prev", "confirmed"]]
              .to_string(index=False))
    print(f"\n输出目录：{OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
