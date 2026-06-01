"""
StepC1: 峰后见底预测（真顶锚点 + 波动退火）

目的：回答两个问题
  1) 距离本轮（2025）熊市底部还有多久 / 多少跌幅？
  2) 历史周期经退火后，对当前 2025 走势的拟合到底好不好？

与现有套件的关键差异：
  - 峰锚点统一用「窗口内最高价日」(真顶)：2017-12-17 / 2021-11-10 / 2025-10-05
    （现有 step/A/B 系列多用 08-15「峰起点」，会把 08-15→10-05 的上涨误算进峰后）
  - 峰后延伸到「下一轮减半前」完整熊市；2025 延伸到最新数据日
  - 底部 = 峰后到下一减半前的最低价日

输出：visualization/YYYY-MM-DD/ 下 stepC1_*.txt / *.csv / png/stepC1_*.png
"""
from __future__ import annotations
from pathlib import Path
from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE
PNG_DIR = OUTDIR / "png"
MERGED_CSV = DATA_DIR / "btc_merged_daily.csv"

# 减半日（与项目记忆一致）
HALVING = {2017: pd.Timestamp("2016-07-09"),
           2021: pd.Timestamp("2020-05-11"),
           2025: pd.Timestamp("2024-04-20")}
NEXT_HALVING = {2017: pd.Timestamp("2020-05-11"),
                2021: pd.Timestamp("2024-04-20"),
                2025: None}  # 2025 下一减半未到，用最新数据日
# 真顶只在减半后这个天数内搜索：牛市顶都在减半后约 1.5 年；
# 若开到「下一减半前」，2024-03 的新高会污染 2021 周期的顶。
PEAK_SEARCH_DAYS = 730

# 波动退火等级（人工设定，与现有脚本一致）：越往后波动越小
VOL_LEVEL = {2017: 9.0, 2021: 3.0, 2025: 1.0}
VOL_ALPHA = 0.5
PRE_STD_SPAN = 90
COLORS = {2025: "#1a5276", 2021: "#e74c3c", 2017: "#27ae60"}


def load_series() -> pd.Series:
    df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    s = df.set_index("date")["price"].sort_index()
    # merged 已按日 ffill，保证每天有值
    return s


def pre_std(dd_by_relday: pd.Series, span: int) -> float:
    idx = [d for d in range(-span, 1) if d in dd_by_relday.index]
    return float(dd_by_relday.loc[idx].std(ddof=0)) if idx else float("nan")


def scale_manual(level_old: float, level_new: float, alpha: float = 0.5) -> float:
    return (level_new / level_old) ** alpha


def fit_metrics(a: pd.Series, b: pd.Series):
    """两条以 rel_day 为索引的曲线，在公共整数天上的 r 与 RMSE。"""
    idx = a.index.intersection(b.index)
    if len(idx) < 3:
        return float("nan"), float("nan")
    A, B = a.loc[idx].values, b.loc[idx].values
    r = float(np.corrcoef(A, B)[0, 1])
    rmse = float(np.sqrt(np.mean((A - B) ** 2)))
    return r, rmse


def main():
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

    s = load_series()
    latest = s.index.max().normalize()

    # ---------- 1) 各周期真顶、底（绝对最低价日）、关键天数与跌幅 ----------
    info = {}
    for cyc in (2017, 2021, 2025):
        h = HALVING[cyc]
        nh = NEXT_HALVING[cyc] if NEXT_HALVING[cyc] is not None else latest + pd.Timedelta(days=1)
        # 真顶：减半后 PEAK_SEARCH_DAYS 天内的最高价（避免误抓下一轮新高）
        peak_hi = min(h + pd.Timedelta(days=PEAK_SEARCH_DAYS), nh - pd.Timedelta(days=1), latest)
        seg_full = s.loc[h:peak_hi]
        peak_day = seg_full.idxmax()
        peak_px = float(seg_full.max())
        # 峰后到下一减半前（2025 到最新）：最低价 = 底
        seg_post = s.loc[peak_day:(nh - pd.Timedelta(days=1) if NEXT_HALVING[cyc] is not None else latest)]
        bottom_day = seg_post.idxmin()
        bottom_px = float(seg_post.min())
        info[cyc] = dict(
            halving=h, peak_day=peak_day, peak_px=peak_px,
            bottom_day=bottom_day, bottom_px=bottom_px,
            h2p_days=(peak_day - h).days,
            p2b_days=(bottom_day - peak_day).days,
            dd_bottom=(bottom_px / peak_px - 1.0) * 100.0,
            post_latest_days=(seg_post.index.max() - peak_day).days,
        )

    # 2025 当前状态
    p25 = info[2025]
    cur_days = (latest - p25["peak_day"]).days
    cur_px = float(s.loc[latest])
    cur_dd = (cur_px / p25["peak_px"] - 1.0) * 100.0

    # ---------- 2) 退火系数（manual：人工等级；std：实际峰前波动比，作对照）----------
    # 峰前 dd 曲线（相对真顶），用于算实际峰前 std
    dd_pre = {}
    for cyc in (2017, 2021, 2025):
        h, peak_day, peak_px = info[cyc]["halving"], info[cyc]["peak_day"], info[cyc]["peak_px"]
        seg = s.loc[h:peak_day]
        rel = (seg.index - peak_day).days
        dd = (seg.values / peak_px - 1.0) * 100.0
        dd_pre[cyc] = pd.Series(dd, index=rel)
    std_pre = {cyc: pre_std(dd_pre[cyc], min(PRE_STD_SPAN, max(5, info[cyc]["h2p_days"]))) for cyc in (2017, 2021, 2025)}

    scale_manual_ = {cyc: scale_manual(VOL_LEVEL[cyc], VOL_LEVEL[2025], VOL_ALPHA) for cyc in (2017, 2021)}
    scale_std_ = {cyc: (std_pre[2025] / std_pre[cyc]) if std_pre[cyc] else 1.0 for cyc in (2017, 2021)}

    # ---------- 3) 峰后 dd 曲线（相对真顶），整数天索引 ----------
    dd_post = {}
    for cyc in (2017, 2021, 2025):
        peak_day, peak_px = info[cyc]["peak_day"], info[cyc]["peak_px"]
        end = (NEXT_HALVING[cyc] - pd.Timedelta(days=1)) if NEXT_HALVING[cyc] is not None else latest
        seg = s.loc[peak_day:end]
        rel = (seg.index - peak_day).days
        dd = (seg.values / peak_px - 1.0) * 100.0
        dd_post[cyc] = pd.Series(dd, index=rel).groupby(level=0).last()

    # ---------- 4) 拟合：退火后历史 vs 2025（峰前全段；峰后到当前 cur_days）----------
    def scaled(series_dd: pd.Series, k: float) -> pd.Series:
        return series_dd * k

    fit = {}
    for cyc in (2017, 2021):
        k = scale_manual_[cyc]
        # 峰前：公共负天数
        r_pre, rmse_pre = fit_metrics(dd_pre[2025], scaled(dd_pre[cyc], k))
        # 峰后：仅到当前 2025 已走的天数 [0, cur_days]
        post25 = dd_post[2025].loc[dd_post[2025].index <= cur_days]
        post_h = scaled(dd_post[cyc].loc[dd_post[cyc].index <= cur_days], k)
        r_post, rmse_post = fit_metrics(post25, post_h)
        fit[cyc] = dict(k=k, r_pre=r_pre, rmse_pre=rmse_pre, r_post=r_post, rmse_post=rmse_post)

    # ---------- 5) 历史在「峰后第 cur_days 天」的退火跌幅 vs 2025 实际 ----------
    same_day_pred = {}
    for cyc in (2017, 2021):
        hd = dd_post[cyc]
        # 取最接近 cur_days 的历史天
        nearest = hd.index[np.argmin(np.abs(hd.index.values - cur_days))]
        hist_dd = float(hd.loc[nearest])
        same_day_pred[cyc] = dict(
            hist_dd_raw=hist_dd,
            pred_dd_manual=hist_dd * scale_manual_[cyc],
            pred_dd_std=hist_dd * scale_std_[cyc],
        )

    # ---------- 6) 底部预测 ----------
    # (a) 时间法：历史「峰→底」天数 → 2025 真顶 + 天数
    p2b_list = [info[2017]["p2b_days"], info[2021]["p2b_days"]]
    p2b_med = float(np.median(p2b_list))
    p2b_mean = float(np.mean(p2b_list))
    peak25 = p25["peak_day"]
    bottom_pred_time = {
        "by_2017": peak25 + timedelta(days=info[2017]["p2b_days"]),
        "by_2021": peak25 + timedelta(days=info[2021]["p2b_days"]),
        "median": peak25 + timedelta(days=int(round(p2b_med))),
    }
    # (b) 退火价格法：历史底跌幅 × 退火系数 → 2025 预测底跌幅 → 底价
    bottom_pred_price = {}
    for cyc in (2017, 2021):
        dd_m = info[cyc]["dd_bottom"] * scale_manual_[cyc]
        dd_s = info[cyc]["dd_bottom"] * scale_std_[cyc]
        bottom_pred_price[cyc] = dict(
            dd_manual=dd_m, px_manual=p25["peak_px"] * (1 + dd_m / 100.0),
            dd_std=dd_s, px_std=p25["peak_px"] * (1 + dd_s / 100.0),
        )

    # ---------- 7) 输出 TXT ----------
    L = []
    L.append("=" * 64)
    L.append("StepC1 峰后见底预测（真顶锚点 + 波动退火）")
    L.append(f"数据最新日: {latest.date()}   生成日: {CURRENT_DATE}")
    L.append("=" * 64)
    L.append("")
    L.append("【一、各周期真顶 / 熊市底 / 关键天数】")
    for cyc in (2017, 2021, 2025):
        d = info[cyc]
        tag = "（至今最低，未必是最终底）" if cyc == 2025 else ""
        L.append(f"  {cyc}: 真顶 {d['peak_day'].date()} ${d['peak_px']:,.0f}  "
                 f"减半→峰 {d['h2p_days']}d")
        L.append(f"        底 {d['bottom_day'].date()} ${d['bottom_px']:,.0f}  "
                 f"峰→底 {d['p2b_days']}d  跌幅 {d['dd_bottom']:.1f}% {tag}")
    L.append("")
    L.append("【二、2025 当前状态】")
    L.append(f"  最新 {latest.date()}  ${cur_px:,.0f}  相对真顶(10-05) {cur_dd:.1f}%")
    L.append(f"  峰后已走 {cur_days} 天（历史见底需 {p2b_list[0]}~{p2b_list[1]} 天）")
    L.append(f"  峰后至今最低: {p25['bottom_day'].date()} ${p25['bottom_px']:,.0f} "
             f"({p25['dd_bottom']:.1f}%, 峰后 {p25['p2b_days']} 天)")
    L.append("")
    L.append("【三、波动退火系数】（把历史曲线压扁到 2025 量级）")
    L.append(f"  实际峰前波动 std(dd%): 2017={std_pre[2017]:.2f}  2021={std_pre[2021]:.2f}  2025={std_pre[2025]:.2f}")
    for cyc in (2017, 2021):
        L.append(f"  {cyc}→2025: manual(level^0.5)={scale_manual_[cyc]:.3f}   std比实测={scale_std_[cyc]:.3f}")
    L.append("")
    L.append("【四、拟合度（退火 manual 后，历史 vs 2025）】")
    for cyc in (2017, 2021):
        f = fit[cyc]
        L.append(f"  {cyc}: 峰前 r={f['r_pre']:.3f} rmse={f['rmse_pre']:.2f} | "
                 f"峰后(0~{cur_days}d) r={f['r_post']:.3f} rmse={f['rmse_post']:.2f}")
    L.append("")
    L.append(f"【五、当前点位拟合检验】2025 实际峰后{cur_days}天 = {cur_dd:.1f}%")
    for cyc in (2017, 2021):
        sp = same_day_pred[cyc]
        L.append(f"  {cyc} 同期原始 {sp['hist_dd_raw']:.1f}% × 退火 → "
                 f"manual预测 {sp['pred_dd_manual']:.1f}%  (误差 {cur_dd - sp['pred_dd_manual']:+.1f}pp)")
    L.append("")
    L.append("【六、底部预测】")
    L.append("  (a) 时间法（真顶 10-05 + 历史峰→底天数）：")
    L.append(f"      仿 2017({info[2017]['p2b_days']}d) → {bottom_pred_time['by_2017'].date()}")
    L.append(f"      仿 2021({info[2021]['p2b_days']}d) → {bottom_pred_time['by_2021'].date()}")
    L.append(f"      中位数({int(round(p2b_med))}d)     → {bottom_pred_time['median'].date()}")
    L.append("  (b) 退火价格法（历史底跌幅 × 退火系数）：")
    for cyc in (2017, 2021):
        bp = bottom_pred_price[cyc]
        L.append(f"      基于{cyc}: manual 底跌幅 {bp['dd_manual']:.1f}% → ${bp['px_manual']:,.0f}   "
                 f"(std法 {bp['dd_std']:.1f}% → ${bp['px_std']:,.0f})")
    # 综合判断
    pred_dds = [bottom_pred_price[c]["dd_manual"] for c in (2017, 2021)]
    worst, best = min(pred_dds), max(pred_dds)
    L.append("")
    L.append("【七、综合判断】")
    L.append(f"  退火模型预测底部跌幅区间约 {worst:.0f}% ~ {best:.0f}%（manual）。")
    L.append(f"  2025 至今最低已达 {p25['dd_bottom']:.1f}%（{p25['bottom_day'].date()}）。")
    if p25["dd_bottom"] <= best:
        L.append("  => 至今最低已进入/穿过退火预测底部区间：价格维度上底部可能已基本到位。")
    else:
        L.append(f"  => 距退火预测底部区间还差约 {best - p25['dd_bottom']:.0f}pp 跌幅。")
    L.append(f"  时间维度：若仿历史节奏，底约在 {bottom_pred_time['by_2017'].date()} ~ {bottom_pred_time['by_2021'].date()}。")
    L.append("  注意：结论高度依赖人工波动等级 9/3/1；若 2025 波动被低估，跌幅会更深。")

    txt = "\n".join(L)
    (OUTDIR / "stepC1_bottom_forecast.txt").write_text(txt, encoding="utf-8")
    print(txt)

    # ---------- 8) CSV：峰后对齐（退火 manual 后） ----------
    alld = np.arange(0, max(dd_post[2017].index.max(), dd_post[2021].index.max()) + 1)
    out = pd.DataFrame({"post_day": alld})
    out["dd_2025"] = np.interp(alld, dd_post[2025].index, dd_post[2025].values,
                               left=np.nan, right=np.nan)
    for cyc in (2017, 2021):
        out[f"dd_{cyc}_scaled"] = np.interp(alld, dd_post[cyc].index,
                                            dd_post[cyc].values * scale_manual_[cyc])
    out.to_csv(OUTDIR / "stepC1_postpeak_aligned.csv", index=False, encoding="utf-8-sig")

    # ---------- 9) 图：峰后对齐 + 底部标注 ----------
    plt.figure(figsize=(14, 6))
    for cyc in (2017, 2021):
        k = scale_manual_[cyc]
        x = dd_post[cyc].index.values
        y = dd_post[cyc].values * k
        plt.plot(x, y, color=COLORS[cyc], alpha=0.85,
                 label=f"{cyc}（退火×{k:.2f}） 底@{info[cyc]['p2b_days']}d {info[cyc]['dd_bottom']*k:.0f}%")
        # 历史底标注
        bd = info[cyc]["p2b_days"]
        plt.scatter([bd], [info[cyc]["dd_bottom"] * k], color=COLORS[cyc], s=60, zorder=5, marker="v")
    # 2025 实际（到最新）
    x25 = dd_post[2025].index.values
    y25 = dd_post[2025].values
    plt.plot(x25, y25, color=COLORS[2025], linewidth=2.4, zorder=6,
             label=f"2025（实际，至{latest.date()}）")
    plt.scatter([cur_days], [cur_dd], color=COLORS[2025], s=80, zorder=7, marker="o")
    plt.annotate(f"今天 {cur_days}d\n{cur_dd:.0f}%", (cur_days, cur_dd),
                 textcoords="offset points", xytext=(8, 10), color=COLORS[2025])
    # 2025 至今最低
    plt.scatter([p25["p2b_days"]], [p25["dd_bottom"]], color=COLORS[2025], s=70, zorder=7, marker="v")
    # 预测底部区间（时间 × 退火价格）
    for cyc in (2017, 2021):
        bd = info[cyc]["p2b_days"]
        plt.scatter([bd], [bottom_pred_price[cyc]["dd_manual"]], facecolors="none",
                    edgecolors=COLORS[2025], s=110, zorder=5, marker="*",
                    label=f"2025 预测底(仿{cyc}) {bottom_pred_price[cyc]['dd_manual']:.0f}% @{bd}d")

    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(cur_days, color=COLORS[2025], linestyle=":", alpha=0.5)
    plt.xlabel("峰后天数（真顶=0）")
    plt.ylabel("相对真顶涨跌（%）")
    plt.title(f"StepC1 峰后见底预测：真顶锚点 + 波动退火 | 2025真顶 {peak25.date()} | 最新 {latest.date()}")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = PNG_DIR / "stepC1_bottom_forecast.png"
    plt.savefig(out_png, dpi=170)
    plt.close()
    print("\nPNG:", out_png.resolve())
    print("CSV:", (OUTDIR / 'stepC1_postpeak_aligned.csv').resolve())


if __name__ == "__main__":
    main()
