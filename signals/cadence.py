"""周期节奏分析 → HTML 报告。

- 每个指标(AHR999 / MVRV / 价格 / 恐慌)单独出图，标出它在四轮各自的顶▲与底▼
- 把所有「(指标 × 周期) 的 峰→底天数」投影成本轮底日期 → voting 直方图取共识
- 峰用价格锚校验（减半→峰），底用各指标投票

产出 outputs/：
  cad_<indicator>_*.png       每个指标的顶/底标注图
  cad_voting_*.png            本轮底的投票直方图
  cadence_peaks.csv / cadence_bottoms.csv / cadence_votes.csv
  report_cadence_*.html       汇总（图内嵌）

用法: python signals/cadence.py        （如需最新先 charts.py --refresh）
"""
from __future__ import annotations
import base64
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import indicators as I

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

TOP_WINDOWS = {
    2011: ("2011-04-01", "2011-08-31"), 2013: ("2013-10-01", "2014-01-31"),
    2017: ("2017-11-01", "2018-01-31"), 2021: ("2021-10-01", "2021-12-31"),
    2025: ("2025-09-15", "2025-12-15"),
}
HALVING_OF = {2013: "2012-11-28", 2017: "2016-07-09", 2021: "2020-05-11", 2025: "2024-04-20"}
HALVING_OF = {k: pd.Timestamp(v) for k, v in HALVING_OF.items()}
COMPLETED = [2011, 2013, 2017, 2021]
IMMATURE = {2011, 2013}
BEAR_WINDOW, PEAK_SEARCH = 540, 760
PEAK_NEAR = 100   # 峰只标"真顶(价格)附近±100天内"的最高，避开早期第一波/次峰
CONTRACTION = 1428 / 1473
ROBUST = ["AHR999", "MVRV"]
CYCLE_COLOR = {2011: "#9ca3af", 2013: "#0ea5e9", 2017: "#16a34a", 2021: "#f59e0b", 2025: "#dc2626"}
GREEN, RED = "#15803d", "#b91c1c"


def _load():
    price = pd.read_csv(C.PRICE_CSV, parse_dates=["date"])
    fng = pd.read_csv(C.FNG_CSV, parse_dates=["date"])
    onc = pd.read_csv(C.ONCHAIN_CSV, parse_dates=["date"])
    ahr = I.compute_ahr999(price)[["date", "ahr999"]]
    return {"AHR999": (ahr, "ahr999", True), "MVRV": (onc, "mvrv", False),
            "价格": (price, "price", True), "恐慌": (fng, "value", False)}, price


def _ext(df, col, a, b, kind):
    w = df[(df["date"] >= a) & (df["date"] <= b)].dropna(subset=[col])
    if w.empty:
        return None, None
    r = w.loc[w[col].idxmax()] if kind == "max" else w.loc[w[col].idxmin()]
    return r["date"], float(r[col])


def detect(series):
    price = series["价格"][0]
    tops = {c: _ext(price, "price", pd.Timestamp(s), pd.Timestamp(e), "max")[0]
            for c, (s, e) in TOP_WINDOWS.items()}
    pk, bt = [], []
    for ind, (df, col, _) in series.items():
        for cyc, hv in HALVING_OF.items():
            top = tops[cyc]   # 真顶=价格全局高；各指标只取真顶附近±PEAK_NEAR天内的最高
            d, v = _ext(df, col, top - pd.Timedelta(days=PEAK_NEAR), top + pd.Timedelta(days=PEAK_NEAR), "max")
            if d is not None:
                pk.append(dict(cycle=cyc, indicator=ind, date=d, value=v, days_from_halving=(d - hv).days))
        for cyc in COMPLETED:
            d, v = _ext(df, col, tops[cyc], tops[cyc] + pd.Timedelta(days=BEAR_WINDOW), "min")
            if d is not None:
                bt.append(dict(cycle=cyc, indicator=ind, date=d, value=v, days_from_peak=(d - tops[cyc]).days))
    return tops, pd.DataFrame(pk), pd.DataFrame(bt)


def _save(fig, name):
    p = C.CHART_DIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[chart] {name}")
    return p


def chart_indicator(name, df, col, logy, peaks_df, bottoms_df, tops, end):
    a = df[df["date"] >= pd.Timestamp("2011-01-01")].dropna(subset=[col])
    fig, ax = plt.subplots(figsize=(14, 4.6))
    (ax.semilogy if logy else ax.plot)(a["date"], a[col], color="#374151", lw=1.0, zorder=1)
    pks = peaks_df[peaks_df["indicator"] == name]
    bts = bottoms_df[bottoms_df["indicator"] == name]
    for _, r in pks.iterrows():
        ax.scatter([r["date"]], [r["value"]], color=RED, marker="^", s=70, zorder=5)
        ax.annotate(f"{r['cycle']}顶\n{r['date'].date()}", (r["date"], r["value"]), color=RED,
                    fontsize=8, ha="center", xytext=(0, 8), textcoords="offset points")
    for _, r in bts.iterrows():
        ax.scatter([r["date"]], [r["value"]], color=GREEN, marker="v", s=70, zorder=5)
        ax.annotate(f"{r['cycle']}底\n{r['date'].date()}", (r["date"], r["value"]), color=GREEN,
                    fontsize=8, ha="center", xytext=(0, -22), textcoords="offset points")
    ax.set_title(f"{name} — 四轮 顶▲/底▼（该指标各自检测）", fontsize=14, fontweight="bold")
    ax.set_xlim(pd.Timestamp("2011-01-01"), end)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="both", alpha=0.2)
    return _save(fig, f"cad_{name}_{end.date()}.png")


def chart_voting(votes, peak25, consensus, end):
    """votes: list of projected bottom Timestamps（每个=一个指标×周期投票）。"""
    months = [pd.Timestamp(v.year, v.month, 1) for v in votes]
    cnt = Counter(months)
    xs = sorted(cnt)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar([x for x in xs], [cnt[x] for x in xs], width=22, color="#6366f1", alpha=0.85)
    ax.axvline(consensus, color=RED, lw=2, ls="--", label=f"投票共识(中位) {consensus.date()}")
    ax.set_title("本轮底「投票」：所有 指标×周期 的『峰→底天数』投影到本轮底日期，按月计票",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("票数"); ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    return _save(fig, f"cad_voting_{end.date()}.png")


def _b64(p): return base64.b64encode(p.read_bytes()).decode()


def main():
    series, price = _load()
    end = pd.Timestamp(price["date"].max().date())
    tops, peaks_df, bottoms_df = detect(series)
    peaks_df.to_csv(C.OUT_DIR / "cadence_peaks.csv", index=False, encoding="utf-8")
    bottoms_df.to_csv(C.OUT_DIR / "cadence_bottoms.csv", index=False, encoding="utf-8")
    peak25 = tops[2025]

    # 峰校验（价格，成熟周期）
    pk_mat = peaks_df[(peaks_df.indicator == "价格") & (peaks_df.cycle.isin([2017, 2021]))]["days_from_halving"]
    pk_mat_mean = float(pk_mat.mean())
    proj_peak = HALVING_OF[2025] + timedelta(days=int(round(pk_mat_mean)))

    # 投票：每个 指标×周期 的 峰→底天数 → 投影本轮底日期
    votes = [peak25 + timedelta(days=int(r["days_from_peak"])) for _, r in bottoms_df.iterrows()]
    votes_sorted = sorted(votes)
    consensus = votes_sorted[len(votes_sorted) // 2]  # 中位
    # 成熟+稳健共识（AHR999/MVRV，2017/2021）
    rob = bottoms_df[(bottoms_df.indicator.isin(ROBUST)) & (~bottoms_df.cycle.isin(IMMATURE))]["days_from_peak"]
    rob_mean = float(rob.mean())
    proj_rob = peak25 + timedelta(days=int(round(rob_mean)))
    proj_rob_c = peak25 + timedelta(days=int(round(rob_mean * CONTRACTION)))

    votes_df = bottoms_df.copy()
    votes_df["projected_bottom"] = [v.date() for v in votes]
    votes_df.to_csv(C.OUT_DIR / "cadence_votes.csv", index=False, encoding="utf-8")

    # 图
    imgs = []
    for name, (df, col, logy) in series.items():
        p = chart_indicator(name, df, col, logy, peaks_df, bottoms_df, tops, end)
        imgs.append((p, f"{name}：红▲=该指标检测的各轮顶，绿▼=各轮底。"
                        + ("AHR999/MVRV 定底准、但顶偏早(第一波抛物线)。" if name in ROBUST else
                           "价格顶可靠、但2013底受Mt.Gox坏数据干扰。" if name == "价格" else
                           "恐慌仅2018+、噪声大、顶底均不稳。")))
    pv = chart_voting(votes, peak25, consensus, end)

    # 投票表
    vt_rows = "".join(
        f"<tr><td>{r['cycle']}</td><td>{r['indicator']}</td><td>{int(r['days_from_peak'])}d</td>"
        f"<td>{(peak25+timedelta(days=int(r['days_from_peak']))).date()}</td></tr>"
        for _, r in bottoms_df.sort_values(['cycle', 'indicator']).iterrows())

    cur_days = (end - peak25).days
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>BTC 四轮 顶/底 + 投票 {end.date()}</title><style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0;}}
 .page{{max-width:1100px;margin:0 auto;padding:38px 26px 80px;}}
 h1{{font-size:23px;border-bottom:2px solid #1a1a1a;padding-bottom:10px;}}
 h2{{font-size:18px;margin-top:34px;}}
 .box{{background:#dcfce7;border:1px solid #15803d;border-radius:8px;padding:14px 18px;font-size:15px;margin:16px 0;}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;}}
 th,td{{border:1px solid #e5e5e3;padding:6px 10px;text-align:center;}} th{{background:#f3f4f6;}}
 figure{{margin:22px 0;}} img{{width:100%;border:1px solid #e5e5e3;border-radius:8px;}}
 figcaption{{color:#444;font-size:13px;margin-top:6px;}}
 .foot{{color:#9ca3af;font-size:12px;margin-top:34px;border-top:1px solid #e5e5e3;padding-top:12px;}}
</style></head><body><div class="page">
<h1>BTC 四轮 顶/底 检测 + 投票共识</h1>
<p style="color:#6b6b6b;font-size:14px;">数据至 {end.date()}　·　四轮：2011 / 2015 / 2018 / 2022 底　·　每个指标分别检测顶/底，最后投票</p>

<div class="box">
🗳️ <b>投票共识 — 本轮底 ≈ {consensus.strftime('%Y-%m')}</b>（全部 {len(votes)} 票的中位 {consensus.date()}）<br>
🎯 稳健口径（AHR999+MVRV，成熟2017/2021）：均值 {rob_mean:.0f}天 → <b>{proj_rob.date()}</b>（×小幅收缩 → {proj_rob_c.date()}）<br>
✅ 方法校验：用价格「减半→峰」{pk_mat_mean:.0f}天预测本轮峰 = {proj_peak.date()}，实际真顶 {peak25.date()}，<b>误差 {abs((proj_peak-peak25).days)} 天</b>
</div>

<h2>一、各指标分别检测的 四轮 顶▲/底▼</h2>
{''.join(f'<figure><img src="data:image/png;base64,{_b64(p)}"/><figcaption>{cap}</figcaption></figure>' for p, cap in imgs)}

<h2>二、投票：所有「指标 × 周期」投影到本轮底</h2>
<figure><img src="data:image/png;base64,{_b64(pv)}"/>
<figcaption>每一个(指标×周期)的「峰→底天数」加到本轮真顶({peak25.date()})上 = 一票，按月计票。早期/噪声票散落各月，成熟票聚集在 ~2026-10。</figcaption></figure>

<table><tr><th>周期</th><th>指标</th><th>峰→底天数</th><th>投影本轮底</th></tr>{vt_rows}</table>

<h2>三、结论</h2>
<div class="box">
本轮真顶 {peak25.date()}，至今 {cur_days} 天。<br>
<b>投票共识与稳健口径都指向 2026 年 10 月前后</b>（成熟节奏：减半→峰 ~{pk_mat_mean:.0f}天、峰→底 ~{rob_mean:.0f}天）。<br>
早期周期(2011/2013)与恐慌指数把少数票拉向更早(2026 上半年)，但属不成熟/噪声，参考价值低。
</div>
<div class="foot">峰用价格锚，底用 AHR999/MVRV（定底高度一致）。辅助分析，非买卖建议。signals/cadence.py 生成。</div>
</div></body></html>"""
    out = C.OUT_DIR / f"report_cadence_{end.date()}.html"
    out.write_text(html, encoding="utf-8")
    print(f"\n[html] {out}")
    print(f"投票共识(中位) {consensus.date()} | 稳健 {proj_rob.date()} | 峰校验误差 {abs((proj_peak-peak25).days)}天")


if __name__ == "__main__":
    main()
