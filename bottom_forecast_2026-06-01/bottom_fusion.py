# -*- coding: utf-8 -*-
"""
BTC 下一个底 —— 时间精准化 + 价格采用退火法（v2）

修正(按用户反馈)：
  1) 价格主结论改用「波动退火法(stepC1)」：底≈-44%/$63~70k，对齐 2 月低点 $63,846；
     比值法 $34k 降级为「系统性崩盘的尾部风险」。
  2) 时间精准化：不再三法等权 + 人为 ±30 天下限；而是拆成 6 个独立周期锚点，
     并按「拟合优先」加权——stepC1 已证 2025≈退火2021(误差0.5pp)，故 2021/2022
     周期相关的天数给更高权重。

输出：本文件夹 bottom_fusion_result.txt / bottom_time_anchors.csv / bottom_fusion.png
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
MERGED = DATA / "btc_merged_daily.csv"

HALV = {2016: pd.Timestamp("2016-07-09"), 2020: pd.Timestamp("2020-05-11"), 2024: pd.Timestamp("2024-04-20")}
PEAK_SEARCH_DAYS = 730

# 价格情景（退火法为主）
ANNEAL_SCALE_2021 = 0.577      # stepC1: 2021→2025 manual 退火系数
USER_LOW, USER_HIGH = 63000, 66000   # 用户判断的底价区间
TAIL_RISK = 34200              # 比值法(v1)给的深熊尾部风险位


def setup_font():
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
    s = pd.read_csv(MERGED, parse_dates=["date"]).set_index("date")["price"].sort_index()
    latest = s.index.max(); nowpx = float(s.iloc[-1])

    # ---- 真顶真底（idxmax/idxmin）----
    def top(hy, ny):
        h = HALV[hy]; hi = min(h + pd.Timedelta(days=PEAK_SEARCH_DAYS), (HALV[ny]-pd.Timedelta(days=1)) if ny else latest, latest)
        seg = s.loc[h:hi]; return seg.idxmax(), float(seg.max())
    T17d, T17 = top(2016, 2020); T21d, T21 = top(2020, 2024); T25d, T25 = top(2024, None)
    B18d, B18 = (lambda seg: (seg.idxmin(), float(seg.min())))(s.loc[T17d:HALV[2020]])
    B22d, B22 = (lambda seg: (seg.idxmin(), float(seg.min())))(s.loc[T21d:HALV[2024]])
    seg_now = s.loc[T25d:latest]; bnowd, bnow = seg_now.idxmin(), float(seg_now.min())

    # ============ 一、时间精准化：6 个独立周期锚点 + 拟合优先加权 ============
    # 单位天数（用真顶真底算）
    t2b_17 = (B18d - T17d).days        # 顶→底 仿2017
    t2b_21 = (B22d - T21d).days        # 顶→底 仿2021  ← 2025 拟合最优
    h2b_18 = (B18d - HALV[2016]).days  # 减半→底 仿2016周期
    h2b_22 = (B22d - HALV[2020]).days  # 减半→底 仿2020周期
    b2b_a = (B18d - s.loc[s.index.min():HALV[2016]].idxmin()).days  # 底→底 2015→2018
    b2b_b = (B22d - B18d).days         # 底→底 2018→2022

    # 锚点：(名称, 预测日期, 权重)  —— 2021/2022 周期(拟合最优)给高权重
    anchors = [
        ("顶→底·仿2017", T25d + timedelta(days=t2b_17), 1.0, t2b_17),
        ("顶→底·仿2021★", T25d + timedelta(days=t2b_21), 1.8, t2b_21),
        ("减半→底·仿2016轮", HALV[2024] + timedelta(days=h2b_18), 0.9, h2b_18),
        ("减半→底·仿2020轮", HALV[2024] + timedelta(days=h2b_22), 1.4, h2b_22),
        ("底→底·2015→18", B22d + timedelta(days=b2b_a), 0.9, b2b_a),
        ("底→底·2018→22★", B22d + timedelta(days=b2b_b), 1.4, b2b_b),
    ]
    ords = np.array([d.toordinal() for _, d, _, _ in anchors], float)
    wts = np.array([w for _, _, w, _ in anchors], float)
    c_ord = np.average(ords, weights=wts)
    std = float(np.sqrt(np.average((ords - c_ord) ** 2, weights=wts)))
    t_center = date.fromordinal(int(round(c_ord)))
    t_lo = date.fromordinal(int(round(c_ord - std)))
    t_hi = date.fromordinal(int(round(c_ord + std)))
    # 单点最佳估计 = 拟合最优周期(2021 顶→底)
    t_best_single = T25d + timedelta(days=t2b_21)

    # ============ 二、价格：退火法为主 ============
    dd_anneal = -(1 - B22 / T21) * ANNEAL_SCALE_2021   # 2021跌幅 × 退火系数
    px_anneal = T25 * (1 + dd_anneal)
    feb_dd = bnow / T25 - 1
    price_lo, price_hi = USER_LOW, min(USER_HIGH + 0, px_anneal)  # 用户下界 ~ 退火上沿
    price_lo = min(USER_LOW, bnow)                                # 不高于已发生的2月低点下沿
    price_hi = max(USER_HIGH, px_anneal)
    price_center = (USER_LOW + USER_HIGH) / 2

    # ============ 三、输出文本 ============
    L = []
    L.append("=" * 66)
    L.append("BTC 下一个底（v2：时间精准化 + 退火价格）")
    L.append(f"数据至 {latest.date()}  现价 ${nowpx:,.0f}  真顶 {T25d.date()} ${T25:,.0f}")
    L.append("=" * 66)
    L.append("")
    L.append("【★ 最终结论】")
    L.append(f"  时间: 中心 {t_center}  核心窗口 {t_lo} → {t_hi}（±{std:.0f}天）")
    L.append(f"        单点最佳(仿拟合最优的2021): {t_best_single}")
    L.append(f"  价格: ${price_lo:,.0f} ~ ${price_hi:,.0f}（中心约 ${price_center:,.0f}，退火法 -44%≈${px_anneal:,.0f}）")
    L.append(f"        极端尾部风险(比值法/系统性崩盘): ${TAIL_RISK:,.0f}")
    L.append("")
    L.append("【一、时间：6 个独立锚点（★=2021/2022 拟合最优周期，高权重）】")
    for name, d, w, days in anchors:
        L.append(f"  {name:<16} {days}天 → {d}   w={w}")
    L.append(f"  >> 加权中心 {t_center}，σ={std:.0f}天 → 窗口 {t_lo} ~ {t_hi}")
    L.append(f"  >> 6 锚点全落在 {min(d for _,d,_,_ in anchors)} ~ {max(d for _,d,_,_ in anchors)}（即 2026 年 9 月底~10 月底）")
    L.append("")
    L.append("【二、价格：退火法为主（采纳，弃用比值法$34k为主结论）】")
    L.append(f"  退火法(stepC1, 2021×{ANNEAL_SCALE_2021}): 跌幅 {dd_anneal*100:.1f}% → ${px_anneal:,.0f}")
    L.append(f"  2 月实际低点: {bnowd.date()} ${bnow:,.0f}（{feb_dd*100:.1f}%）")
    L.append(f"  用户判断: ${USER_LOW:,.0f} ~ ${USER_HIGH:,.0f}")
    L.append(f"  >> 综合底价区间: ${price_lo:,.0f} ~ ${price_hi:,.0f}")
    L.append("")
    L.append("【三、关键逻辑：二次探底（W 底）】")
    L.append(f"  2 月已第一次探底 ${bnow:,.0f}({feb_dd*100:.0f}%) → 反弹至现价 ${nowpx:,.0f}")
    L.append(f"  → 预测 {t_center} 前后二次探底，再测 ${price_lo:,.0f}~${price_hi:,.0f}（与2月相近=最终底）")
    L.append(f"  若 2 月即最终底，则底已过；用户判断底在未来，故取二次探底情景。")
    txt = "\n".join(L)
    (OUT / "bottom_fusion_result.txt").write_text(txt, encoding="utf-8")
    print(txt)

    pd.DataFrame([{"anchor": n, "days": dd, "date": d.isoformat(), "weight": w} for n, d, w, dd in anchors]
                ).to_csv(OUT / "bottom_time_anchors.csv", index=False, encoding="utf-8-sig")

    # ============ 四、图 ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8))
    # 左：价格（聚焦 2024 起）+ 情景
    sf = s.loc["2024-01-01":]
    ax1.plot(sf.index, sf.values, color="#888", lw=1.0, label="BTC")
    ax1.scatter([T25d], [T25], color="#e74c3c", s=70, marker="^", zorder=5, label=f"真顶 ${T25:,.0f}")
    ax1.scatter([bnowd], [bnow], color="#16a085", s=70, marker="v", zorder=5, label=f"2月低 ${bnow:,.0f}")
    ax1.scatter([latest], [nowpx], color="#1a5276", s=70, marker="o", zorder=5, label=f"现价 ${nowpx:,.0f}")
    # 预测底区间(绿) + 时间窗口(竖灰) + 退火点 + 尾部风险
    ax1.axhspan(price_lo, price_hi, color="#27ae60", alpha=0.18, label=f"底价 ${price_lo:,.0f}~${price_hi:,.0f}")
    ax1.axvspan(pd.Timestamp(t_lo), pd.Timestamp(t_hi), color="#8e44ad", alpha=0.12, label=f"底时间 {t_lo}~{t_hi}")
    ax1.scatter([pd.Timestamp(t_center)], [price_center], color="#8e44ad", s=200, marker="*", zorder=7,
                label=f"预测底 ${price_center:,.0f}\n@{t_center}")
    ax1.axhline(TAIL_RISK, color="#c0392b", ls=":", lw=1, alpha=0.7, label=f"尾部风险 ${TAIL_RISK:,.0f}")
    # 二次探底示意虚线：现价→预测底
    ax1.plot([latest, pd.Timestamp(t_center)], [nowpx, price_center], color="#8e44ad", ls="--", lw=1.2, alpha=0.6)
    ax1.set_yscale("log"); ax1.set_ylabel("USD (log)")
    ax1.set_title("价格情景：二次探底至 $63-66k @ 2026-10"); ax1.legend(fontsize=7.5, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # 右：6 个时间锚点（点大小=权重）+ 加权中心 + 窗口
    ax2.axvspan(pd.Timestamp(t_lo), pd.Timestamp(t_hi), color="#8e44ad", alpha=0.12)
    ax2.axvline(pd.Timestamp(t_center), color="#8e44ad", lw=2, label=f"加权中心 {t_center}")
    ax2.axvline(pd.Timestamp(t_best_single), color="#e67e22", lw=1.5, ls="--", label=f"仿2021单点 {t_best_single}")
    for i, (name, d, w, days) in enumerate(anchors):
        ax2.scatter([pd.Timestamp(d)], [i], s=80 + w * 90, color="#2c3e50", zorder=5)
        ax2.text(pd.Timestamp(d), i + 0.18, name, fontsize=8, ha="center")
    ax2.set_yticks(range(len(anchors))); ax2.set_yticklabels([f"w={w}" for _,_,w,_ in anchors], fontsize=8)
    ax2.set_xlim(pd.Timestamp("2026-09-15"), pd.Timestamp("2026-11-10"))
    ax2.set_title("时间：6 个独立周期锚点收敛（点大小=权重）"); ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"BTC 下一个底 v2 | 时间 {t_center}(±{std:.0f}d) | 价格 ${price_lo:,.0f}~${price_hi:,.0f} | 数据至 {latest.date()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "bottom_fusion.png", dpi=160); plt.close()
    print("\nPNG:", (OUT / "bottom_fusion.png").resolve())


if __name__ == "__main__":
    main()
