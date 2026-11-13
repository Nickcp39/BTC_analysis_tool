# -*- coding: utf-8 -*-
"""用户手给的 2021 ↔ 2025 配对点位（2026-11-13）：逐段记录时间比、幅度比，并按配对分段对齐画图。

配对见 user_pairs.csv：
- user：用户给的三对（对1 2022-04-02↔2026-01-17、对2 2022-08-13↔2026-05-09、对3 2022-11-12↔2026-06-04）
- anchor：报告锚点（顶 2021-11-08↔2025-08-12、底 2022-11-21↔2026-06-30）
- marked_points：底后的跳涨、第一个小高峰（本轮是用户标的，2021 轮是同定义候选）
只记录、只按配对画，不另外拟合（项目约定：用户对齐，Claude 记录）。

输出（本目录）：
- user_pairs_table.csv、user_pairs_numbers.txt   每对的离顶天数、相对顶的对数跌幅、AHR；相邻两对之间的时间比、幅度比
- png/user_pairs_2021_2025.png                    ① 2021 轮按配对分段对齐到本轮时间轴 ② 时间对照 ③ AHR（压缩读法）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
sys.path.insert(0, str(ROOT))
from signals.indicators import compute_ahr999  # noqa: E402

PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
AS_OF = pd.Timestamp("2026-11-12")
AHR_COMP = 0.57
SRC_STYLE = {"user": ("#dc2626", "-", "用户配对"), "anchor": ("#475569", "--", "报告锚点"),
             "marked_points": ("#2563eb", ":", "底后标点")}


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index().loc[:AS_OF]
    a = compute_ahr999(s.rename("price").reset_index()).set_index("date")["ahr999"]
    pr = pd.read_csv(HERE / "user_pairs.csv", parse_dates=["date_2021", "date_2025"])
    pr = pr.sort_values("date_2021").reset_index(drop=True)
    top21, top25 = pr.loc[pr.pair == "顶", "date_2021"].iloc[0], pr.loc[pr.pair == "顶", "date_2025"].iloc[0]
    bot21, bot25 = pr.loc[pr.pair == "底", "date_2021"].iloc[0], pr.loc[pr.pair == "底", "date_2025"].iloc[0]

    rows = []
    for _, r in pr.iterrows():
        d1, d2 = r.date_2021, r.date_2025
        l1, l2 = np.log(s[d1] / s[top21]), np.log(s[d2] / s[top25])
        rows.append({"配对": r.pair, "来源": r.source, "2021 日期": d1.date().isoformat(), "2025 日期": d2.date().isoformat(),
                     "2021 离顶天数": (d1 - top21).days, "2025 离顶天数": (d2 - top25).days,
                     "2021 价格": round(float(s[d1])), "2025 价格": round(float(s[d2])),
                     "2021 相对顶": round(float(np.exp(l1) - 1), 4), "2025 相对顶": round(float(np.exp(l2) - 1), 4),
                     "对数幅度比": round(float(l2 / l1), 3) if l1 != 0 else None,
                     "2021 AHR": round(float(a[d1]), 3), "2025 AHR": round(float(a[d2]), 3)})
    tb = pd.DataFrame(rows)
    tb.to_csv(HERE / "user_pairs_table.csv", index=False, encoding="utf-8-sig")
    lines = [f"数据截至 {AS_OF.date()} 收盘 ${s.iloc[-1]:,.0f}", "", "== 每一对", tb.to_string(index=False), "", "== 相邻两对之间（2025 ÷ 2021）"]
    segs = []
    for i in range(len(pr) - 1):
        r0, r1 = pr.iloc[i], pr.iloc[i + 1]
        t1, t2 = (r1.date_2021 - r0.date_2021).days, (r1.date_2025 - r0.date_2025).days
        m1, m2 = np.log(s[r1.date_2021] / s[r0.date_2021]), np.log(s[r1.date_2025] / s[r0.date_2025])
        segs.append((r0.pair, r1.pair, t1, t2, m1, m2))
        lines.append(f"{r0.pair} → {r1.pair}：2021 {t1} 天 {np.exp(m1) - 1:+.1%}；2025 {t2} 天 {np.exp(m2) - 1:+.1%}；"
                     f"时间比 {t2 / t1:.2f}；对数幅度比 {m2 / m1:.2f}")
    user = tb[tb["来源"] == "user"]
    k_amp = float(user["对数幅度比"].median())
    lines.append("")
    lines.append(f"三对用户配对的「相对顶对数跌幅比」：{', '.join(f'{v:.2f}' for v in user['对数幅度比'])}；中位 {k_amp:.2f}（下图 ① 用它缩放 2021 轮）")
    t_user = [(r["2025 离顶天数"], r["2021 离顶天数"]) for _, r in user.iterrows()]
    lines.append("用户配对的「离顶天数比」(2025 ÷ 2021)：" + "、".join(f"{x / y:.2f}" for x, y in t_user))
    lines.append("")
    lines.append("== AHR999 在每一对上：ln(AHR / 该轮底 AHR)，2025 ÷ 2021（看压缩是不是各个高度都一样）")
    for _, r in pr.iterrows():
        l1 = np.log(a[r.date_2021] / a[bot21])
        l2 = np.log(a[r.date_2025] / a[bot25])
        ratio = f"{l2 / l1:.2f}" if abs(l1) > 0.05 else "—（离底太近，比值没意义）"
        lines.append(f"{r.pair}：AHR {a[r.date_2021]:.3f} / {a[r.date_2025]:.3f}；离底 {l1:+.2f} / {l2:+.2f}；比 {ratio}")
    # 今天落在 2021 轮哪一天（最后一对之后按日历天 1:1）
    last = pr.iloc[-1]
    eq21 = last.date_2021 + (AS_OF - last.date_2025)
    lines.append(f"按配对分段（最后一对之后 1:1），本轮今天 {AS_OF.date()} 对应 2021 轮 {eq21.date()}（${s[eq21]:,.0f}，相对 2021 小高峰 {s[eq21] / s[last.date_2021] - 1:+.1%}）")
    (HERE / "user_pairs_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------- 图 ----------
    x21 = np.array([(d - top21).days for d in pr.date_2021], float)
    x25 = np.array([(d - top25).days for d in pr.date_2025], float)

    def warp(days):
        v = np.interp(days, x21, x25)
        v = np.where(days > x21[-1], x25[-1] + (days - x21[-1]), v)
        return np.where(days < x21[0], x25[0] + (days - x21[0]), v)

    fig, axs = plt.subplots(3, 1, figsize=(14, 14.5), dpi=160, gridspec_kw={"height_ratios": [1.35, 1, 1]})
    ax = axs[0]
    p25 = s.loc[top25 - pd.Timedelta(days=60):AS_OF]
    p21 = s.loc[top21 - pd.Timedelta(days=60): last.date_2021 + (AS_OF - last.date_2025) + pd.Timedelta(days=120)]
    d21 = (p21.index - top21).days.values.astype(float)
    ax.plot((p25.index - top25).days, p25.values / s[top25], color="#0f172a", lw=2.8, label="2025 轮（本轮）")
    ax.plot(warp(d21), np.exp(np.log(p21.values / s[top21]) * k_amp), color="#0d9488", lw=1.7,
            label=f"2021 轮：时间按配对分段对齐，对数幅度 ×{k_amp:.2f}（三对用户配对的中位）")
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for _, r in pr.iterrows():
        c, ls, _lab = SRC_STYLE[r.source]
        xx = (r.date_2025 - top25).days
        for axx in axs[[0, 2]]:
            axx.axvline(xx, color=c, ls=ls, lw=1.3, alpha=0.85)
        ax.text(xx, 0.97, f" {r.pair}\n {r.date_2021:%y-%m-%d}\n ↓\n {r.date_2025:%y-%m-%d}".replace("↓", "对"), transform=tr,
                fontsize=7.6, color=c, va="top")
    ax.axvline((AS_OF - top25).days, color="#7c3aed", ls=":", lw=1.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 1) * 100:+.0f}%"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("相对各自的顶（对数刻度）")
    ax.set_xlabel("本轮时间轴：距 2025-08-12 顶的天数（紫色点线 = 今天）")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="lower left", fontsize=8.8)
    ax.set_title("① 用你给的配对把 2021 轮分段对齐到本轮（红实线 = 你给的三对；灰虚线 = 报告的顶、底；蓝点线 = 底后标点）",
                 fontsize=11, weight="bold", loc="left")

    ax = axs[1]
    ax.plot([0, x21.max() + 60], [0, x21.max() + 60], color="#cbd5e1", lw=1, ls="--", label="时间不压缩（1:1）")
    ax.plot(x21, x25, color="#0f172a", lw=1.2, marker=None)
    for i, (_, r) in enumerate(pr.iterrows()):
        c, _ls, _lab = SRC_STYLE[r.source]
        ax.scatter([x21[i]], [x25[i]], color=c, s=50, zorder=5)
        ax.annotate(r.pair, (x21[i], x25[i]), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=c)
    for (p0, p1, t1, t2, m1, m2), i in zip(segs, range(len(segs))):
        mx, my = (x21[i] + x21[i + 1]) / 2, (x25[i] + x25[i + 1]) / 2
        ax.annotate(f"×{t2 / t1:.2f}", (mx, my), xytext=(-34, 8), textcoords="offset points", fontsize=8.8, color="#b45309", weight="bold")
    ax.set_xlabel("2021 轮：距 2021-11-08 顶的天数")
    ax.set_ylabel("本轮：距 2025-08-12 顶的天数")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="upper left", fontsize=8.8)
    ax.set_title("② 时间怎么对应：每一段的斜率 = 本轮这一段 ÷ 2021 轮这一段（橙色数字）", fontsize=11, weight="bold", loc="left")

    ax = axs[2]
    a25 = a.loc[top25 - pd.Timedelta(days=60):AS_OF]
    a21 = a.loc[p21.index]
    ax.plot((a25.index - top25).days, np.log(a25.values / a[bot25]), color="#0f172a", lw=2.6, label="2025 轮")
    ax.plot(warp(d21), np.log(a21.values / a[bot21]) * AHR_COMP, color="#0d9488", lw=1.6, label=f"2021 轮（×{AHR_COMP} 压缩，时间同 ①）")
    ax.plot(warp(d21), np.log(a21.values / a[bot21]), color="#0d9488", lw=1.0, ls="--", alpha=0.7, label="2021 轮（不压缩）")
    ax.set_ylim(-0.3, 2.0)
    ax.axhline(0, color="#475569", lw=0.8)
    ax.axvline((AS_OF - top25).days, color="#7c3aed", ls=":", lw=1.3)
    ax.set_ylabel("ln(AHR / 该轮底的 AHR)")
    ax.set_xlabel("本轮时间轴：距 2025-08-12 顶的天数")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.set_title("③ AHR999 离底多远（压缩读法）", fontsize=11, weight="bold", loc="left")
    axs[2].set_xlim(axs[0].get_xlim())
    fig.suptitle("2021 轮与 2025 轮的人工配对点位（用户 2026-11-13 给的三对 + 报告锚点 + 底后标点）", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG / "user_pairs_2021_2025.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
