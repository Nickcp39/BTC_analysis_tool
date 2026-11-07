# -*- coding: utf-8 -*-
"""大底之后的走势，和前三轮比像不像？按 10-01 版「时间加速、幅度退火」框架检验（2026-11-07）。

口径：本轮顶 2025-08-12，底 2026-06-30（不用 10-05）。前三轮的底：2015-01-14、2018-12-15、2022-11-21。
数据：2015-02 以前用 Bitstamp（同 10-01 fit_cycles.py），之后用 data/btc_merged_daily.csv（FRED 到 11-06 收盘）。

做法：把四轮都对齐到底（第 0 天），比较「底后第 d 天的对数涨幅」。
- 不缩放：直接比。
- 按框架预期缩放（全部来自 analysis_runs/2026-10-01_cycle_time_amplitude_fit/SUMMARY.md）：
    时间：下一轮牛市天数 ÷ 那一轮牛市天数。两种时钟——
          A「牛市每轮 ×0.94」= 935 天（顶 2029-01-20）；B「顶→顶每轮 -50 天」= 1,001 天（顶 2029-03-27）。
    幅度：下一轮牛市对数涨幅 G = 1.44（×0.687 拟合）÷ 那一轮牛市对数涨幅。
- 最佳拟合：网格搜索 (时间倍数, 幅度倍数)，看数据自己挑出来的倍数是不是「<1 的时间、≈0.687ⁿ 的幅度」。

输出（本目录）：post_bottom_similarity.txt、post_bottom_similarity.csv、png/post_bottom_similarity.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
EARLY_CSV = ROOT / "analysis_runs" / "2026-10-01_cycle_time_amplitude_fit" / "bitstamp_btcusd_daily_2011_2015.csv"
MERGED_CSV = ROOT / "data" / "btc_merged_daily.csv"
OUT_TXT = HERE / "post_bottom_similarity.txt"
OUT_CSV = HERE / "post_bottom_similarity.csv"
T = pd.Timestamp
SWITCH = T("2015-02-01")
AS_OF = T("2026-11-06")

BOTTOM_NOW, TOP_NOW = T("2026-06-30"), T("2025-08-12")
# (名字, 底, 之后的顶, 牛市对数涨幅 —— 来自 10-01 cycle_phases.csv)
HIST = [
    ("2013 轮底 2015-01", T("2015-01-14"), T("2017-12-16"), 4.741774778684165),
    ("2017 轮底 2018-12", T("2018-12-15"), T("2021-11-08"), 3.0544524755106495),
    ("2021 轮底 2022-11", T("2022-11-21"), T("2025-08-12"), 2.057947169542941),
]
G_NEXT = 1.44                       # P1：牛市对数涨幅 6.49 × 0.687⁴
BULL_A, BULL_B = 935, 1001          # A 牛市 ×0.94；B 顶→顶 -50 天
COLORS = {"2013": "#cbd5e1", "2017": "#94a3b8", "2021": "#0d9488"}


def load_price() -> pd.Series:
    early = pd.read_csv(EARLY_CSV, parse_dates=["date"]).set_index("date")["close"]
    late = pd.read_csv(MERGED_CSV, parse_dates=["date"]).set_index("date")["price"]
    return pd.concat([early[early.index < SWITCH], late[late.index >= SWITCH]]).sort_index()


def log_path(s: pd.Series, bottom: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    seg = s.loc[bottom:end]
    return pd.Series(np.log(seg.values / float(s[bottom])), index=(seg.index - bottom).days)


def scaled(hist: pd.Series, days: np.ndarray, kt: float, ka: float) -> np.ndarray:
    """本轮第 d 天 ↔ 历史第 d/kt 天；幅度 × ka。"""
    return ka * np.interp(days / kt, hist.index.to_numpy(), hist.to_numpy())


def metrics(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    r = float(np.corrcoef(y, yhat)[0, 1])
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return r, rmse


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    out = lines.append

    s = load_price().loc[:AS_OF]
    p0 = float(s[BOTTOM_NOW])
    now = log_path(s, BOTTOM_NOW, AS_OF)
    days = now.index.to_numpy().astype(float)
    y = now.to_numpy()
    n = int(days[-1])
    out(f"本轮：底 {BOTTOM_NOW.date()} ${p0:,.0f}；截至 {AS_OF.date()} 底后第 {n} 天 ${float(s.iloc[-1]):,.0f}（{np.exp(y[-1]) - 1:+.1%}）")
    hi_d = int(now.idxmax())
    out(f"底后最高：第 {hi_d} 天 {np.exp(now.max()) - 1:+.1%}；现在距它 {np.exp(y[-1] - now.max()) - 1:+.1%}")
    out("")

    rows = []
    hist_paths = {}
    for name, bot, top, g in HIST:
        h = log_path(s, bot, top)
        bull_days = (top - bot).days
        hist_paths[name] = (h, bot, g, bull_days)
        ka = G_NEXT / g
        for label, kt, kav in [
            ("不缩放", 1.0, 1.0),
            ("只缩幅度", 1.0, ka),
            ("预期 A（牛市×0.94，935 天）", BULL_A / bull_days, ka),
            ("预期 B（顶→顶-50，1,001 天）", BULL_B / bull_days, ka),
        ]:
            yh = scaled(h, days, kt, kav)
            r, rmse = metrics(y, yh)
            rows.append({"参照": name, "缩放": label, "时间倍数": round(kt, 3), "幅度倍数": round(kav, 3),
                         "相关系数r": round(r, 3), "对数RMSE": round(rmse, 3),
                         f"第{n}天_参照涨幅%": round((np.exp(yh[-1]) - 1) * 100, 1),
                         f"第{n}天_实际涨幅%": round((np.exp(y[-1]) - 1) * 100, 1)})
        # 最佳拟合网格
        best = None
        for kt in np.arange(0.40, 1.81, 0.01):
            if n / kt > h.index.max():
                continue
            base = np.interp(days / kt, h.index.to_numpy(), h.to_numpy())
            denom = float(base @ base)
            if denom <= 0:
                continue
            kav = float(base @ y) / denom          # 给定 kt，最小二乘的幅度倍数
            rmse = float(np.sqrt(np.mean((y - kav * base) ** 2)))
            if best is None or rmse < best[2]:
                best = (kt, kav, rmse)
        kt, kav, rmse = best
        yh = scaled(h, days, kt, kav)
        r, _ = metrics(y, yh)
        rows.append({"参照": name, "缩放": "数据最佳拟合", "时间倍数": round(kt, 2), "幅度倍数": round(kav, 3),
                     "相关系数r": round(r, 3), "对数RMSE": round(rmse, 3),
                     f"第{n}天_参照涨幅%": round((np.exp(yh[-1]) - 1) * 100, 1),
                     f"第{n}天_实际涨幅%": round((np.exp(y[-1]) - 1) * 100, 1)})
        out(f"{name}：牛市 {bull_days} 天、对数涨幅 {g:.2f}；预期幅度倍数 {ka:.3f}（= 0.687^{np.log(ka) / np.log(0.687):.1f}）；"
            f"预期时间倍数 A {BULL_A / bull_days:.3f} / B {BULL_B / bull_days:.3f}；数据最佳 时间 {kt:.2f} × 幅度 {kav:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    out("")
    pd.set_option("display.width", 250)
    out(df.to_string(index=False))

    # 里程碑：从底涨 25% / 40% 用了几天（原始 vs 按预期 A 缩放）
    out("")
    out("== 里程碑（底后首次达到）")
    for thr in [0.25, 0.40]:
        d_now = next((int(d) for d, v in now.items() if v >= np.log(1 + thr)), None)
        parts = [f"本轮 {d_now} 天"]
        for name, (h, bot, g, bull_days) in hist_paths.items():
            ka, kt = G_NEXT / g, BULL_A / bull_days
            d_raw = next((int(d) for d, v in h.items() if v >= np.log(1 + thr)), None)
            d_sc = next((int(d * kt) for d, v in h.items() if ka * v >= np.log(1 + thr)), None)
            parts.append(f"{name[:7]} 原始 {d_raw} 天 / 预期缩放后 {d_sc} 天")
        out(f"+{thr:.0%}：" + "；".join(parts))

    # 往后看：按预期 A 缩放的参照路径，在本轮第 180/270/365/450 天的位置
    out("")
    out("== 往后看（只是「如果规律延续」的参照，不是预测）：按预期 A 缩放")
    fwd = np.array([180, 270, 365, 450, 540])
    for name, (h, bot, g, bull_days) in hist_paths.items():
        kt, ka = BULL_A / bull_days, G_NEXT / g
        v = scaled(h, fwd.astype(float), kt, ka)
        seg_days = np.arange(n, n + 91, dtype=float)
        nxt = scaled(h, seg_days, kt, ka)
        dd = float(np.min(np.exp(nxt - np.maximum.accumulate(nxt))) - 1)
        out(f"{name}：" + "；".join(f"第 {d} 天（{(BOTTOM_NOW + pd.Timedelta(days=int(d))).date()}）${p0 * np.exp(x):,.0f}"
                                     for d, x in zip(fwd, v))
            + f"；未来 90 天参照路径内最大回撤 {dd:.1%}")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- 图 ----------------
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.3), dpi=160)
    span = np.arange(0, 541, dtype=float)
    for k, ax in enumerate(axs):
        for name, (h, bot, g, bull_days) in hist_paths.items():
            kt, ka = (1.0, 1.0) if k == 0 else (BULL_A / bull_days, G_NEXT / g)
            v = scaled(h, span, kt, ka)
            lab = name if k == 0 else f"{name}（时间×{kt:.2f}，幅度×{ka:.2f}）"
            ax.plot(span, (np.exp(v) - 1) * 100, color=COLORS[name[:4]], lw=1.8, label=lab)
        ax.plot(days, (np.exp(y) - 1) * 100, color="#0f172a", lw=3.4, label="本轮（底 2026-06-30）", zorder=6)
        ax.axvline(n, color="#7c3aed", ls=":", lw=1.2)
        ax.axhline(0, color="#475569", lw=0.9)
        ax.set_xlabel("距本轮底的天数（本轮的天）")
        ax.grid(True, color="#e2e8f0")
        ax.legend(loc="upper left", fontsize=8.6)
    axs[0].set_xlim(0, 540)
    axs[0].set_ylim(-20, 330)
    axs[0].set_ylabel("相对底的涨幅（%）")
    axs[0].set_title("不缩放：前三轮大底之后涨得快得多", fontsize=12.5, weight="bold")
    axs[1].set_title("按 10-01 框架缩放（牛市 935 天、对数涨幅 1.44）", fontsize=12.5, weight="bold")
    axs[1].set_xlim(0, 540)
    axs[1].set_ylim(-20, 200)
    axs[1].annotate(f"现在 第 {n} 天 {np.exp(y[-1]) - 1:+.0%}", (n, (np.exp(y[-1]) - 1) * 100), xytext=(14, -30),
                    textcoords="offset points", color="#7c3aed", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#7c3aed"))
    fig.suptitle(f"大底之后的走势相似度（截至 {AS_OF:%Y-%m-%d}）", fontsize=14, weight="bold")
    fig.text(0.99, 0.005, "右图：历史第 d 天的对数涨幅 × 幅度倍数，画在本轮第 d×时间倍数 天。倍数全部来自 10-01 四轮拟合，没有用本轮底后的数据去调。",
             ha="right", fontsize=8.3, color="#334155")
    fig.tight_layout()
    fig.savefig(PNG / "post_bottom_similarity.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
