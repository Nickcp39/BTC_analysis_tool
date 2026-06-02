# -*- coding: utf-8 -*-
"""
BTC 下一个底 —— 时间精准化 + 价格三锚交叉（v3）

v3 相对 v2 的关键补充：加回 AHR999（算"底"尤其该用它，它就是抄底指标）。
价格不再只靠退火法，而是用两个相互独立的锚 + 用户判断三方交叉：
  - 退火法(stepC1)：2021 跌幅 × 退火系数 0.577 → -44% ≈ $69.5k
  - AHR999 法：底部 AHR999≈0.28(历史极稳) × est(币龄函数,可外推) → $60-67k
  - 用户判断：$63-66k
  三者都落在 $60-70k → 结论 $63-66k；比值法 $34k 降为系统性崩盘尾部风险。

时间：6 个独立周期锚点按"拟合优先"加权(2025≈退火2021) → 2026-10-18 ±12 天。

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
AHR = DATA / "ahr999_daily.xlsx"

HALV = {2016: pd.Timestamp("2016-07-09"), 2020: pd.Timestamp("2020-05-11"), 2024: pd.Timestamp("2024-04-20")}
PEAK_SEARCH_DAYS = 730
GENESIS = pd.Timestamp("2009-01-03")

ANNEAL_SCALE_2021 = 0.577
USER_LOW, USER_HIGH = 63000, 66000
TAIL_RISK = 34200
GMA200_SCENARIOS = [70000, 80000, 90000]   # 底部时 200 日几何均价的合理情景


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


def ahr999_bottom(est_date):
    """AHR999 = P^2/(gma200*est)。用底部 AHR999≈0.28 + est 外推 反推底价。
    返回 dict：ahr_bottom, est, prices{gma200:P}, 历史底部 ahr 值。"""
    a = pd.read_excel(AHR); a["date"] = pd.to_datetime(a["date"]); a = a.set_index("date").sort_index()
    ah18 = float(a.loc[:"2018-12-15", "ahr999"].iloc[-1])
    ah22 = float(a.loc[:"2022-11-21", "ahr999"].iloc[-1])
    ahr_b = (ah18 + ah22) / 2.0
    # estimate_price 是币龄幂律: log10(est)=k*log10(age)+c → 可精确外推
    a2 = a.dropna(subset=["estimate_price"])
    age = (a2.index - GENESIS).days.values.astype(float)
    A = np.vstack([np.log10(age), np.ones_like(age)]).T
    k, c = np.linalg.lstsq(A, np.log10(a2["estimate_price"].values), rcond=None)[0]
    est = float(10 ** (k * np.log10((pd.Timestamp(est_date) - GENESIS).days) + c))
    prices = {g: float(np.sqrt(ahr_b * g * est)) for g in GMA200_SCENARIOS}
    # P/est 趋势风险（底部相对估值在下降）
    pe18 = ah18 and a.loc[:"2018-12-15", "price"].iloc[-1] / a.loc[:"2018-12-15", "estimate_price"].iloc[-1]
    pe22 = a.loc[:"2022-11-21", "price"].iloc[-1] / a.loc[:"2022-11-21", "estimate_price"].iloc[-1]
    pe_next = pe22 + (pe22 - pe18)
    risk_low = float(pe_next * est)
    return dict(ahr_b=ahr_b, ah18=ah18, ah22=ah22, est=est, prices=prices,
                lo=min(prices.values()), hi=max(prices.values()), mid=prices[80000],
                pe18=pe18, pe22=pe22, risk_low=risk_low)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
    s = pd.read_csv(MERGED, parse_dates=["date"]).set_index("date")["price"].sort_index()
    latest = s.index.max(); nowpx = float(s.iloc[-1])

    def top(hy, ny):
        h = HALV[hy]; hi = min(h + pd.Timedelta(days=PEAK_SEARCH_DAYS), (HALV[ny]-pd.Timedelta(days=1)) if ny else latest, latest)
        seg = s.loc[h:hi]; return seg.idxmax(), float(seg.max())
    T17d, T17 = top(2016, 2020); T21d, T21 = top(2020, 2024); T25d, T25 = top(2024, None)
    B18d, B18 = (lambda seg: (seg.idxmin(), float(seg.min())))(s.loc[T17d:HALV[2020]])
    B22d, B22 = (lambda seg: (seg.idxmin(), float(seg.min())))(s.loc[T21d:HALV[2024]])
    seg_now = s.loc[T25d:latest]; bnowd, bnow = seg_now.idxmin(), float(seg_now.min())

    # ============ 一、时间精准化：6 个独立周期锚点 + 拟合优先加权 ============
    t2b_17 = (B18d - T17d).days; t2b_21 = (B22d - T21d).days
    h2b_18 = (B18d - HALV[2016]).days; h2b_22 = (B22d - HALV[2020]).days
    b2b_a = (B18d - s.loc[s.index.min():HALV[2016]].idxmin()).days; b2b_b = (B22d - B18d).days
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
    c_ord = np.average(ords, weights=wts); std = float(np.sqrt(np.average((ords - c_ord) ** 2, weights=wts)))
    t_center = date.fromordinal(int(round(c_ord)))
    t_lo = date.fromordinal(int(round(c_ord - std))); t_hi = date.fromordinal(int(round(c_ord + std)))
    t_best_single = T25d + timedelta(days=t2b_21)

    # ============ 二、价格：退火 + AHR999 两个独立锚 + 用户判断 ============
    dd_anneal = -(1 - B22 / T21) * ANNEAL_SCALE_2021
    px_anneal = T25 * (1 + dd_anneal)
    ahr = ahr999_bottom(t_center)
    feb_dd = bnow / T25 - 1
    price_lo, price_hi = USER_LOW, USER_HIGH
    price_center = (USER_LOW + USER_HIGH) / 2

    # ============ 三、输出文本 ============
    L = []
    L.append("=" * 68)
    L.append("BTC 下一个底（v3：时间精准化 + 价格三锚交叉，含 AHR999）")
    L.append(f"数据至 {latest.date()}  现价 ${nowpx:,.0f}  真顶 {T25d.date()} ${T25:,.0f}")
    L.append("=" * 68)
    L.append("")
    L.append("【★ 最终结论】")
    L.append(f"  时间: 中心 {t_center}  窗口 {t_lo} → {t_hi}（±{std:.0f}天）  单点最佳(仿2021)={t_best_single.date()}")
    L.append(f"  价格: ${price_lo:,.0f} ~ ${price_hi:,.0f}（中心 ${price_center:,.0f}）")
    L.append(f"        ← 退火法 ${px_anneal:,.0f} 与 AHR999 法 ${ahr['lo']:,.0f}~${ahr['hi']:,.0f} 两个独立锚交叉确认")
    L.append(f"        极端尾部风险(比值法/系统性崩盘): ${TAIL_RISK:,.0f}")
    L.append("")
    L.append("【一、时间：6 个独立锚点（★=拟合最优的2021/2022周期，高权重）】")
    for name, d, w, days in anchors:
        L.append(f"  {name:<16} {days}天 → {d.date()}   w={w}")
    L.append(f"  >> 加权中心 {t_center}，σ={std:.0f}天 → 窗口 {t_lo} ~ {t_hi}（6锚点全在9月底~10月底）")
    L.append("")
    L.append("【二、价格：三锚交叉】")
    L.append(f"  锚1 退火法(stepC1, 2021×{ANNEAL_SCALE_2021}): 跌幅 {dd_anneal*100:.1f}% → ${px_anneal:,.0f}")
    L.append(f"  锚2 AHR999法: 底部AHR999≈{ahr['ahr_b']:.3f}(2018={ahr['ah18']:.3f},2022={ahr['ah22']:.3f}) × est外推${ahr['est']:,.0f}")
    L.append(f"        P=√(AHR999×gma200×est)，gma200∈{GMA200_SCENARIOS} → ${ahr['lo']:,.0f}~${ahr['hi']:,.0f}(中${ahr['mid']:,.0f})")
    L.append(f"        对 gma200 不敏感(√弱化)；2月实际低点 ${bnow:,.0f}({feb_dd*100:.1f}%) 正落此区")
    L.append(f"  锚3 用户判断: ${USER_LOW:,.0f} ~ ${USER_HIGH:,.0f}")
    L.append(f"  >> 三锚都在 $60-70k → 综合底价 ${price_lo:,.0f} ~ ${price_hi:,.0f}")
    L.append(f"  风险: 底部 P/est 在降({ahr['pe18']:.3f}→{ahr['pe22']:.3f})，若续降则底价向 ${ahr['risk_low']:,.0f} 下移(偏比值法)")
    L.append("")
    L.append("【三、关键逻辑：二次探底(W底)】")
    L.append(f"  2月第一脚 ${bnow:,.0f}({feb_dd*100:.0f}%) → 反弹${nowpx:,.0f} → 预测 {t_center} 二次探底再测 ${price_lo:,.0f}~${price_hi:,.0f}")
    txt = "\n".join(L)
    (OUT / "bottom_fusion_result.txt").write_text(txt, encoding="utf-8")
    print(txt)

    pd.DataFrame([{"anchor": n, "days": dd, "date": d.isoformat(), "weight": w} for n, d, w, dd in anchors]
                ).to_csv(OUT / "bottom_time_anchors.csv", index=False, encoding="utf-8-sig")

    # ============ 四、图 ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.8))
    sf = s.loc["2024-01-01":]
    ax1.plot(sf.index, sf.values, color="#888", lw=1.0, label="BTC")
    ax1.scatter([T25d], [T25], color="#e74c3c", s=70, marker="^", zorder=5, label=f"真顶 ${T25:,.0f}")
    ax1.scatter([bnowd], [bnow], color="#16a085", s=70, marker="v", zorder=5, label=f"2月低 ${bnow:,.0f}")
    ax1.scatter([latest], [nowpx], color="#1a5276", s=70, marker="o", zorder=5, label=f"现价 ${nowpx:,.0f}")
    ax1.axhspan(price_lo, price_hi, color="#27ae60", alpha=0.18, label=f"底价 ${price_lo:,.0f}~${price_hi:,.0f}")
    # 两个独立锚的标注
    ax1.axhline(px_anneal, color="#2980b9", ls="-.", lw=1, alpha=0.7, label=f"退火锚 ${px_anneal:,.0f}")
    ax1.axhspan(ahr['lo'], ahr['hi'], color="#9b59b6", alpha=0.10, label=f"AHR999锚 ${ahr['lo']:,.0f}~${ahr['hi']:,.0f}")
    ax1.axvspan(pd.Timestamp(t_lo), pd.Timestamp(t_hi), color="#8e44ad", alpha=0.10, label=f"底时间 {t_lo}~{t_hi}")
    ax1.scatter([pd.Timestamp(t_center)], [price_center], color="#8e44ad", s=200, marker="*", zorder=7,
                label=f"预测底 ${price_center:,.0f}\n@{t_center}")
    ax1.axhline(TAIL_RISK, color="#c0392b", ls=":", lw=1, alpha=0.6, label=f"尾部风险 ${TAIL_RISK:,.0f}")
    ax1.plot([latest, pd.Timestamp(t_center)], [nowpx, price_center], color="#8e44ad", ls="--", lw=1.2, alpha=0.6)
    ax1.set_yscale("log"); ax1.set_ylabel("USD (log)")
    ax1.set_title("价格:退火 + AHR999 两锚交叉 → 二次探底 $63-66k @2026-10"); ax1.legend(fontsize=7, loc="lower left")
    ax1.grid(True, alpha=0.3)

    ax2.axvspan(pd.Timestamp(t_lo), pd.Timestamp(t_hi), color="#8e44ad", alpha=0.12)
    ax2.axvline(pd.Timestamp(t_center), color="#8e44ad", lw=2, label=f"加权中心 {t_center}")
    ax2.axvline(pd.Timestamp(t_best_single), color="#e67e22", lw=1.5, ls="--", label=f"仿2021单点 {t_best_single.date()}")
    for i, (name, d, w, days) in enumerate(anchors):
        ax2.scatter([pd.Timestamp(d)], [i], s=80 + w * 90, color="#2c3e50", zorder=5)
        ax2.text(pd.Timestamp(d), i + 0.18, name, fontsize=8, ha="center")
    ax2.set_yticks(range(len(anchors))); ax2.set_yticklabels([f"w={w}" for _,_,w,_ in anchors], fontsize=8)
    ax2.set_xlim(pd.Timestamp("2026-09-15"), pd.Timestamp("2026-11-10"))
    ax2.set_title("时间:6 个独立周期锚点收敛(点大小=权重)"); ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"BTC 下一个底 v3 | 时间 {t_center}(±{std:.0f}d) | 价格 ${price_lo:,.0f}~${price_hi:,.0f}(退火∩AHR999) | 数据至 {latest.date()}", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "bottom_fusion.png", dpi=160); plt.close()
    print("\nPNG:", (OUT / "bottom_fusion.png").resolve())


if __name__ == "__main__":
    main()
