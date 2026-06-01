# -*- coding: utf-8 -*-
"""
BTC 下一个底「模糊正确」预测 —— 对称于当年 peak 四模型融合，但为「底」独立设计。

设计原则（回应：底不能是 peak 的顺手副产品）：
  - 当年的底 = 预测顶 × 回撤retr（retr 还撞网格上界），是 peak 的派生 → 不成立。
  - 这里让「底价 B」成为联合拟合中被独立求解的主角，同时满足三条底自己的关系：
      (1) 顶→底回撤   retr = B / 实际顶        ← 用已确认真顶 2025-10-05，不是预测顶
      (2) 底→底倍数   mult_b = B / 上一个底     ← 完全不依赖顶
      (3) 底→底趋势   log-linear 外推（早期增长猛、外推易爆，低权重）
  - 时间同样独立三法融合：减半→底 / 顶→底 / 底→底周期
  - AHR999 底部值作旁证；末端通胀校正；最后 Multi-Model Fusion 出时间+价格区间。

输出：本文件夹下 bottom_fusion_result.txt / bottom_price_models.csv / bottom_fusion.png
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent
MERGED = DATA / "btc_merged_daily.csv"
AHR = DATA / "ahr999_daily.xlsx"

HALVINGS = {2012: pd.Timestamp("2012-11-28"), 2016: pd.Timestamp("2016-07-09"),
            2020: pd.Timestamp("2020-05-11"), 2024: pd.Timestamp("2024-04-20")}
PEAK_SEARCH_DAYS = 730

# 年化通胀（沿用当年 notebook 的表）
ANNUAL_INFL = {2013:0.015,2014:0.016,2015:0.001,2016:0.013,2017:0.021,2018:0.024,
               2019:0.018,2020:0.012,2021:0.047,2022:0.080,2023:0.041,2024:0.029,
               2025:0.027,2026:0.025,2027:0.024}


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


def load_price():
    s = pd.read_csv(MERGED, parse_dates=["date"]).set_index("date")["price"].sort_index()
    return s


def find_tops_bottoms(s: pd.Series):
    """用 idxmax/idxmin 取真顶真底（顶限定减半后730天内，底取顶→下一减半间最低）。"""
    latest = s.index.max()
    out = {}
    # 顶
    for hy, ny in [(2016, 2020), (2020, 2024), (2024, None)]:
        h = HALVINGS[hy]
        hi = min(h + pd.Timedelta(days=PEAK_SEARCH_DAYS), (HALVINGS[ny] - pd.Timedelta(days=1)) if ny else latest, latest)
        seg = s.loc[h:hi]
        out[f"top_{hy}"] = (seg.idxmax(), float(seg.max()))
    # 底：2015（数据起点~2016减半）、2018（2017顶~2020减半）、2022（2021顶~2024减半）
    b15 = s.loc[s.index.min():HALVINGS[2016]]
    out["bot_2015"] = (b15.idxmin(), float(b15.min()))
    seg18 = s.loc[out["top_2016"][0]:HALVINGS[2020]]
    out["bot_2018"] = (seg18.idxmin(), float(seg18.min()))
    seg22 = s.loc[out["top_2020"][0]:HALVINGS[2024]]
    out["bot_2022"] = (seg22.idxmin(), float(seg22.min()))
    # 当前周期至今最低（仅供对照）
    seg26 = s.loc[out["top_2024"][0]:latest]
    out["bot_now"] = (seg26.idxmin(), float(seg26.min()))
    out["latest"] = (latest, float(s.iloc[-1]))
    return out


def lin_extrap(cycles, vals, target_cycle, logspace=False):
    x = np.array(cycles, float); y = np.array(vals, float)
    if logspace:
        y = np.log(y)
    a, b = np.polyfit(x, y, 1)
    pred = a * target_cycle + b
    return float(np.exp(pred) if logspace else pred), float(a), float(b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup_font()
    s = load_price()
    tb = find_tops_bottoms(s)

    T17d, T17 = tb["top_2016"]; T21d, T21 = tb["top_2020"]; T25d, T25 = tb["top_2024"]
    B15d, B15 = tb["bot_2015"]; B18d, B18 = tb["bot_2018"]; B22d, B22 = tb["bot_2022"]
    nowd, nowpx = tb["latest"]; bnowd, bnow = tb["bot_now"]

    # ===================== 1) 底时间模型（三法融合，对称 peak 时间 A）=====================
    # 顶→底
    t2b = [(B18d - T17d).days, (B22d - T21d).days]
    t2b_mean = float(np.mean(t2b))
    pred_t2b = T25d + timedelta(days=int(round(t2b_mean)))
    # 减半→底
    h2b = [(B18d - HALVINGS[2016]).days, (B22d - HALVINGS[2020]).days]
    h2b_mean = float(np.mean(h2b))
    pred_h2b = HALVINGS[2024] + timedelta(days=int(round(h2b_mean)))
    # 底→底周期
    b2b_days = [(B18d - B15d).days, (B22d - B18d).days]
    b2b_mean = float(np.mean(b2b_days))
    pred_b2b = B22d + timedelta(days=int(round(b2b_mean)))

    time_points = [
        ("顶→底", pred_t2b, 1.2),
        ("减半→底", pred_h2b, 1.0),
        ("底→底周期", pred_b2b, 1.1),
    ]
    ords = np.array([d.toordinal() for _, d, _ in time_points], float)
    w = np.array([wt for _, _, wt in time_points], float)
    t_center_ord = np.average(ords, weights=w)
    t_std = np.sqrt(np.average((ords - t_center_ord) ** 2, weights=w))
    # 给一个最小不确定度（历史顶→底就有 ~14 天散度，叠加样本少的风险，下限 30 天）
    t_sigma = max(t_std, 30.0)
    t_center = date.fromordinal(int(round(t_center_ord)))
    t_lo = date.fromordinal(int(round(t_center_ord - t_sigma)))
    t_hi = date.fromordinal(int(round(t_center_ord + t_sigma)))

    # ===================== 2) 底价格：三关系，B 为联合拟合主角 =====================
    # 关系比值（用真顶真底）
    retr = [B18 / T17, B22 / T21]                      # 顶→底回撤
    mult_b = [B18 / B15, B22 / B18]                    # 底→底倍数
    # 顶 cycle: 2017=2,2021=3,2025=4 ; 底 cycle: 2015=1,2018=2,2022=3, 预测2026=4
    retr_trend, ra, rb = lin_extrap([2, 3], retr, 4, logspace=False)      # 收敛(线性)
    multb_trend, ma, mb = lin_extrap([2, 3], mult_b, 4, logspace=True)    # 衰减(log)
    b2b_price_trend, ba, bb = lin_extrap([1, 2, 3], [B15, B18, B22], 4, logspace=True)  # 易爆,低权重

    # 三个独立隐含底价
    B_from_retr = T25 * retr_trend          # 顶→底回撤 × 真顶
    B_from_multb = B22 * multb_trend        # 底→底倍数 × 上一个底
    B_from_trend = b2b_price_trend          # 底→底 log 趋势(参考)

    # 联合拟合：B 为主角，最小化三关系 log 偏差。
    # 权重设计严格对称 peak：peak 主力是「底→顶倍数」，底的主力就是其镜像「顶→底回撤 retr」(高权重)；
    # 底→底倍数/趋势对应 peak 里的「顶→顶趋势」——本就是低权重软约束(peak 仅给 ~0.08)，且两点外推不稳，故大幅降权/剔除。
    w_retr, w_multb, w_trend = 1.0, 0.12, 0.0
    grid = np.arange(8000.0, 90000.0, 200.0)
    def cost(B):
        c = w_retr * (np.log(B / T25) - np.log(retr_trend)) ** 2
        c += w_multb * (np.log(B / B22) - np.log(multb_trend)) ** 2
        c += w_trend * (np.log(B) - np.log(b2b_price_trend)) ** 2
        return c
    costs = np.array([cost(B) for B in grid])
    i_best = int(np.argmin(costs))
    B_best = float(grid[i_best])
    band = grid[costs <= costs[i_best] * 1.25]
    B_lo, B_hi = float(band.min()), float(band.max())

    # 通胀校正（real→2026名义，跨度 2022底→2026 约 4 年；这里给最优点一个名义对照）
    def infl_factor(y0, y1):
        f = 1.0
        for y in range(y0 + 1, y1 + 1):
            f *= (1.0 + ANNUAL_INFL.get(y, 0.025))
        return f
    # B_from_multb 基于 2022 底（名义2022），名义化到 2026
    B_multb_nominal = B_from_multb * infl_factor(2022, 2026)
    # retr/最优本身已是 2025 名义量级（顶为2025名义），近似 2026 名义

    # ===================== 3) AHR999 底部旁证 =====================
    ahr_note = []
    try:
        a = pd.read_excel(AHR)
        a["date"] = pd.to_datetime(a["date"])
        a = a.dropna(subset=["ahr999"]).set_index("date")["ahr999"].sort_index()
        ah18 = float(a.asof(B18d)); ah22 = float(a.asof(B22d))
        ah_latest_d = a.index.max(); ah_latest = float(a.iloc[-1])
        # 底部 AHR999 cycle 2,3 → 4（线性）
        ahr_bot_trend, _, _ = lin_extrap([2, 3], [ah18, ah22], 4, logspace=False)
        ahr_note = [
            f"AHR999 历史底部值: 2018底={ah18:.3f}, 2022底={ah22:.3f}（趋势外推2026底≈{ahr_bot_trend:.3f}）",
            f"AHR999 数据最新: {ah_latest_d.date()}={ah_latest:.3f}（注意 AHR999 数据未更新到 2026）",
            "AHR999<0.45 通常为历史抄底区；底部值在抬升说明熊底相对估值越来越高（跌得越来越浅）。",
        ]
    except Exception as e:
        ahr_note = [f"AHR999 读取失败：{e}"]

    # ===================== 4) 输出文本 =====================
    L = []
    L.append("=" * 70)
    L.append("BTC 下一个底 预测（对称 peak 的独立多模型融合）")
    L.append(f"数据最新: {nowd.date()}  现价 ${nowpx:,.0f}")
    L.append("=" * 70)
    L.append("")
    L.append("【真顶 / 真底（idxmax/idxmin）】")
    L.append(f"  顶: 2017 {T17d.date()} ${T17:,.0f} | 2021 {T21d.date()} ${T21:,.0f} | 2025 {T25d.date()} ${T25:,.0f}")
    L.append(f"  底: 2015 {B15d.date()} ${B15:,.0f} | 2018 {B18d.date()} ${B18:,.0f} | 2022 {B22d.date()} ${B22:,.0f}")
    L.append(f"  本轮至今最低: {bnowd.date()} ${bnow:,.0f}（{(bnow/T25-1)*100:.1f}% vs 真顶）")
    L.append("")
    L.append("【一、底的时间模型（三法融合）】")
    L.append(f"  顶→底  : 历史 {t2b} 天 → 均值 {t2b_mean:.0f} → {pred_t2b}")
    L.append(f"  减半→底: 历史 {h2b} 天 → 均值 {h2b_mean:.0f} → {pred_h2b}")
    L.append(f"  底→底  : 历史 {b2b_days} 天 → 均值 {b2b_mean:.0f} → {pred_b2b}")
    L.append(f"  >> 融合时间中心: {t_center}   窗口(±{t_sigma:.0f}天): {t_lo} → {t_hi}")
    L.append("")
    L.append("【二、底的价格模型（B 为联合拟合主角）】")
    L.append(f"  [主力] 顶→底回撤 retr: 历史 {retr[0]:.3f}, {retr[1]:.3f} → 趋势外推 {retr_trend:.3f}  => 底≈${B_from_retr:,.0f}  (对称 peak 的底→顶倍数)")
    mb_warn = " ⚠倍数<1不可信(2015底$120偏低+两点外推),已大幅降权" if multb_trend < 1 else ""
    L.append(f"  [软约束] 底→底倍数 mult: 历史 {mult_b[0]:.2f}, {mult_b[1]:.2f} → 趋势外推 {multb_trend:.2f}  => 底≈${B_from_multb:,.0f}{mb_warn}")
    L.append(f"  [剔除] 底→底log趋势 : 外推 ${b2b_price_trend:,.0f}（早期增长过猛,外推爆掉,权重=0,仅展示）")
    L.append(f"  >> 联合拟合最优底(retr主导): ${B_best:,.0f}   模糊区间(cost≤1.25×min): ${B_lo:,.0f} ~ ${B_hi:,.0f}")
    L.append(f"  >> retr 敏感性: retr=0.27→${T25*0.27:,.0f} | 0.30→${T25*0.30:,.0f} | 0.35→${T25*0.35:,.0f}")
    L.append(f"  >> 通胀提示: retr/最优已是 2025 名义量级，距 2026 仅 ~1 年(≈2.5%)，名义微调可忽略")
    L.append("")
    L.append("【三、AHR999 底部旁证】")
    for ln in ahr_note:
        L.append("  " + ln)
    L.append("")
    L.append("【四、综合（Multi-Model Fusion）下一个底】")
    L.append(f"  ★ 时间: 中心 {t_center}，窗口 {t_lo} → {t_hi}")
    L.append(f"  ★ 价格: 中心 ${B_best:,.0f}，区间 ${B_lo:,.0f} ~ ${B_hi:,.0f}")
    L.append(f"     （相对真顶 {(B_best/T25-1)*100:.0f}%；相对现价 {(B_best/nowpx-1)*100:.0f}%）")
    L.append("")
    L.append("【对照与提醒】")
    L.append(f"  - 本轮至今最低已 {(bnow/T25-1)*100:.0f}%（{bnowd.date()} ${bnow:,.0f}）；本模型预测底更深，意味着若成立，底尚未出现。")
    L.append("  - 该结论比『波动退火法(stepC1: 底≈-44%/$64-70k)』深得多——两套方法论分歧大，见 README。")
    L.append("  - 价格模型仅 2 个比值样本，外推脆弱；时间模型三法高度一致(都指向2026-10)，更稳。")

    txt = "\n".join(L)
    (OUT / "bottom_fusion_result.txt").write_text(txt, encoding="utf-8")
    print(txt)

    # CSV
    pd.DataFrame({
        "model": ["retr_顶→底回撤", "mult_底→底倍数", "trend_底→底log", "联合拟合最优", "联合下界", "联合上界"],
        "implied_bottom_usd": [B_from_retr, B_from_multb, b2b_price_trend, B_best, B_lo, B_hi],
        "vs_real_peak_pct": [(x/T25-1)*100 for x in [B_from_retr, B_from_multb, b2b_price_trend, B_best, B_lo, B_hi]],
    }).to_csv(OUT / "bottom_price_models.csv", index=False, encoding="utf-8-sig")

    # ===================== 5) 图 =====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
    # 左：价格——历史顶底 + 预测底
    yrs_t = [T17d, T21d, T25d]; vt = [T17, T21, T25]
    yrs_b = [B15d, B18d, B22d]; vb = [B15, B18, B22]
    ax1.plot(s.index, s.values, color="#999", lw=0.7, alpha=0.6, label="BTC 价格")
    ax1.scatter(yrs_t, vt, color="#e74c3c", s=70, zorder=5, marker="^", label="历史真顶")
    ax1.scatter(yrs_b, vb, color="#27ae60", s=70, zorder=5, marker="v", label="历史真底")
    ax1.scatter([nowd], [nowpx], color="#1a5276", s=80, zorder=6, marker="o", label=f"现价 ${nowpx:,.0f}")
    pred_bd = pd.Timestamp(t_center)
    ax1.scatter([pred_bd], [B_best], color="#8e44ad", s=160, zorder=7, marker="*",
                label=f"预测底 ${B_best:,.0f} @ {t_center}")
    ax1.axhspan(B_lo, B_hi, color="#8e44ad", alpha=0.12)
    ax1.axvspan(pd.Timestamp(t_lo), pd.Timestamp(t_hi), color="#8e44ad", alpha=0.08)
    ax1.set_yscale("log")
    ax1.set_title("价格：历史顶/底 + 预测下一个底")
    ax1.set_ylabel("USD (log)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 右：三个价格模型的隐含底 + 联合区间
    names = ["retr\n顶→底回撤", "mult\n底→底倍数", "trend\n底→底log", "联合\n最优"]
    vals = [B_from_retr, B_from_multb, b2b_price_trend, B_best]
    colors = ["#e67e22", "#16a085", "#bbb", "#8e44ad"]
    bars = ax2.bar(names, vals, color=colors)
    ax2.axhspan(B_lo, B_hi, color="#8e44ad", alpha=0.15, label=f"联合区间 ${B_lo:,.0f}~${B_hi:,.0f}")
    ax2.axhline(bnow, color="#1a5276", ls="--", lw=1, label=f"本轮至今最低 ${bnow:,.0f}")
    ax2.axhline(nowpx, color="#333", ls=":", lw=1, label=f"现价 ${nowpx:,.0f}")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x()+b.get_width()/2, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax2.set_title("各价格模型隐含底（对数y）")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"BTC 下一个底预测 | 真顶 {T25d.date()} ${T25:,.0f} | 数据至 {nowd.date()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "bottom_fusion.png", dpi=160)
    plt.close()
    print("\nPNG:", (OUT / "bottom_fusion.png").resolve())
    print("CSV:", (OUT / "bottom_price_models.csv").resolve())


if __name__ == "__main__":
    main()
