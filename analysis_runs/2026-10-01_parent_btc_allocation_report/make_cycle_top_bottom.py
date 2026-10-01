# -*- coding: utf-8 -*-
"""2026-10-01 版新增：顶和底都有了，直接算三轮周期的完整尺寸。

口径（AGENTS.md）：2025 轮的顶 = 2025-08-12，不用 2025-10-05（那只是比 8 月高 1.1% 的回测）；
2025 轮的底 = 2026-06-30（顶后最低收盘）。历史两轮用各自的价格顶、价格底。

输出（写在本目录）：
- cycle_top_bottom_summary.csv            三轮的减半/顶/底日期、价格、各段天数、跌幅，以及本轮 ÷ 前两轮的比例
- png/cycle_top_bottom_alignment.png      左：按顶对齐的实际天数；右：时间按本轮「减半→顶」比例压缩后
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
PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
T = pd.Timestamp

# (周期, 减半, 顶, 底, 上一轮底, 上一轮顶, 颜色, 线宽)
CYCLES = [
    ("2017 轮", T("2016-07-09"), T("2017-12-16"), T("2018-12-15"), T("2015-01-14"), None, "#94a3b8", 1.8),
    ("2021 轮", T("2020-05-11"), T("2021-11-08"), T("2022-11-21"), T("2018-12-15"), T("2017-12-16"), "#475569", 1.9),
    ("2025 轮", T("2024-04-20"), T("2025-08-12"), T("2026-06-30"), T("2022-11-21"), T("2021-11-08"), "#0f766e", 3.4),
]
WINDOW_AFTER = 600


def setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def summary(s: pd.Series) -> pd.DataFrame:
    rows = []
    for name, halving, top, bottom, prev_bottom, prev_top, _, _ in CYCLES:
        rows.append({
            "周期": name,
            "减半": halving.date().isoformat(),
            "顶": top.date().isoformat(),
            "顶价": round(float(s[top]), 2),
            "底": bottom.date().isoformat(),
            "底价": round(float(s[bottom]), 2),
            "减半→顶_天": (top - halving).days,
            "顶→底_天": (bottom - top).days,
            "顶→底_跌幅": round(float(s[bottom] / s[top] - 1), 4),
            "减半→底_天": (bottom - halving).days,
            "上轮底→底_天": (bottom - prev_bottom).days,
            "上轮底→顶_天": (top - prev_bottom).days,
            "上轮顶→顶_天": (top - prev_top).days if prev_top is not None else None,
        })
    df = pd.DataFrame(rows)
    cur = df.iloc[-1]
    for i in range(2):
        ref = df.iloc[i]
        ratio = {"周期": f"2025 ÷ {ref['周期']}"}
        for col in ["减半→顶_天", "顶→底_天", "减半→底_天", "上轮底→底_天", "上轮底→顶_天", "上轮顶→顶_天"]:
            if pd.notna(ref[col]) and pd.notna(cur[col]):
                ratio[col] = round(cur[col] / ref[col], 3)
        ratio["顶→底_跌幅"] = round(cur["顶→底_跌幅"] / ref["顶→底_跌幅"], 3)
        df = pd.concat([df, pd.DataFrame([ratio])], ignore_index=True)
    return df


def compress_ratio(halving: pd.Timestamp, top: pd.Timestamp) -> float:
    """本轮减半→顶天数 ÷ 该轮减半→顶天数（< 1 表示本轮更快）。"""
    _, h25, t25, *_ = CYCLES[-1]
    return (t25 - h25).days / (top - halving).days


def chart(s: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4), dpi=170, sharey=True)
    latest = s.index.max()
    for k, ax in enumerate(axes):
        for name, halving, top, bottom, *_, color, lw in CYCLES:
            scale = 1.0 if k == 0 or name.startswith("2025") else compress_ratio(halving, top)
            seg = s.loc[top - pd.Timedelta(days=30): min(top + pd.Timedelta(days=WINDOW_AFTER), latest)]
            x = (seg.index - top).days * scale
            ax.plot(x, seg.values / s[top], color=color, lw=lw, label=name, zorder=5 if name.startswith("2025") else 3)
            bx, by = (bottom - top).days * scale, float(s[bottom] / s[top])
            ax.scatter([bx], [by], s=70 if name.startswith("2025") else 45, color=color, zorder=7,
                       edgecolor="white", linewidth=1.3)
            label = f"{name[:4]} 底：第 {bx:.0f} 天，{by - 1:+.0%}"
            # 历史两轮的标签放到左下空白处，避免压住曲线和坐标轴标题
            text_xy = {"2017": (150, 0.150), "2021": (150, 0.195), "2025": (bx + 28, by * 1.55)}[name[:4]]
            ax.annotate(label, xy=(bx, by), xytext=text_xy,
                        fontsize=9, color=color, weight="bold" if name.startswith("2025") else None,
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9})
        ax.axhline(1.0, color="#334155", lw=0.9, alpha=0.6)
        ax.axvline(0, color="#334155", lw=0.9, ls=":", alpha=0.7)
        ax.set_yscale("log")
        ticks = [0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t - 1:+.0%}" for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(0.13, 1.35)
        ax.set_xlim(-30, WINDOW_AFTER)
        ax.grid(True, color="#e2e8f0")
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_title("按顶对齐（实际天数）", fontsize=12.5, weight="bold")
    axes[0].set_xlabel("距顶的天数（2025 轮顶 = 2025-08-12）")
    axes[0].set_ylabel("相对各自的顶（对数刻度）")
    r17 = compress_ratio(CYCLES[0][1], CYCLES[0][2])
    r21 = compress_ratio(CYCLES[1][1], CYCLES[1][2])
    axes[1].set_title(f"时间按本轮「减半→顶」比例压缩后（2017×{r17:.3f}，2021×{r21:.3f}）", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("距顶的天数（历史两轮已压缩）")
    axes[1].axvspan(322, 332, color="#0f766e", alpha=0.12)
    fig.suptitle("顶和底都有了：三轮周期的顶→底对照", fontsize=15, weight="bold")
    fig.text(0.5, -0.01, "本轮从减半到顶比前两轮快约一成；按同一比例压缩后，前两轮的底都落在顶后第 332 天，本轮的底在第 322 天。",
             ha="center", fontsize=9.5, color="#334155")
    fig.tight_layout()
    fig.savefig(PNG / "cycle_top_bottom_alignment.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    PNG.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index()
    df = summary(s)
    df.to_csv(HERE / "cycle_top_bottom_summary.csv", index=False, encoding="utf-8-sig")
    chart(s)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    _, h25, t25, b25, *_ = CYCLES[-1]
    for name, halving, top, bottom, *_ in CYCLES[:2]:
        r = compress_ratio(halving, top)
        d = (bottom - top).days * r
        print(f"{name} 顶→底 {(bottom - top).days} 天 × {r:.3f} = {d:.1f} 天 → {(t25 + pd.Timedelta(days=round(d))).date()}")
    print(f"2025 轮实际 顶→底 {(b25 - t25).days} 天 → {b25.date()}")
    for name, _, top, bottom, *_ in CYCLES[:2]:
        amp = np.log(s[b25] / s[t25]) / np.log(s[bottom] / s[top])
        print(f"对数幅度比 2025/{name}: {amp:.3f}")


if __name__ == "__main__":
    main()
