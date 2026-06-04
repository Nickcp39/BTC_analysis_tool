"""
StepD4: 顶后路径对齐（按"走势形状"匹配，而非点对点）
- 把左段(2021顶后)与右段(2025顶后)都画成「相对各自顶的回撤% vs 距顶天数」。
- Panel1：原始时间轴直接叠（看右侧目前走到左侧的哪个阶段）。
- Panel2：对右侧时间轴乘最优 α（最小二乘拟合左曲线），看路径形状是否真的吻合，
          以及 α 是否≈1(同速)/<1(压缩)/>1(拉长)。
- 目的：用"前导走势是否一致"客观定位 ⑤ 等对应点，并判定右侧当前所处阶段。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepD1_core_points import read_series, setup_cjk_font

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "core_points"
LEFT_TOP = pd.Timestamp("2021-11-08")
RIGHT_TOP = pd.Timestamp("2025-10-05")

# 左侧已知地标（用于看右侧对应到哪个阶段）
LEFT_MARKS = [
    ("2022-01-22", "首低 -48%"),
    ("2022-03-29", "②次高 -30%"),
    ("2022-06-18", "急跌 ~-70%"),
    ("2022-11-21", "③底 -77%"),
]


def path(s, top, ndays):
    seg = s.loc[top: top + pd.Timedelta(days=ndays)]
    days = (seg.index - top).days.values.astype(float)
    pct = (seg.values / seg.iloc[0] - 1.0) * 100.0
    return days, pct, seg.index


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()

    ld, lp, lidx = path(s, LEFT_TOP, 430)
    rd, rp, ridx = path(s, RIGHT_TOP, 430)
    right_span = int(rd.max())
    print(f"右侧(2025顶后)目前有 {right_span} 天数据，到 {ridx.max():%Y-%m-%d}")

    def left_at(d):
        return np.interp(d, ld, lp)

    # 最优时间缩放 α：右侧 day d -> 左侧 α*d
    alphas = np.linspace(0.3, 4.0, 740)
    errs = [np.mean((left_at(a * rd) - rp) ** 2) for a in alphas]
    alpha = float(alphas[int(np.argmin(errs))])
    print(f"最优时间缩放 α = {alpha:.2f}  "
          f"({'右侧更快/压缩' if alpha>1 else '右侧更慢/拉长' if alpha<1 else '同速'})")

    # 右侧当前末端对应到左侧第几天 / 什么阶段
    end_left_day = alpha * right_span
    print(f"右侧末端({ridx.max():%Y-%m-%d}, {rp[-1]:.1f}%) ≈ 左侧第 {end_left_day:.0f} 天"
          f"（左侧那天回撤 {left_at(end_left_day):.1f}%, 日期约 {(LEFT_TOP+pd.Timedelta(days=end_left_day)):%Y-%m-%d}）")

    # 左侧地标对应到右侧（除以 α）现实天数
    print("\n左侧地标 → 右侧对应（按 α 折算）：")
    for d, name in LEFT_MARKS:
        ldays = (pd.Timestamp(d) - LEFT_TOP).days
        rdays = ldays / alpha
        rdate = RIGHT_TOP + pd.Timedelta(days=rdays)
        status = "已过" if rdays <= right_span else "未到"
        print(f"  {name:14} 左第{ldays:>4}天 → 右第{rdays:>5.0f}天 ≈ {rdate:%Y-%m-%d}  [{status}]")

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    ax = axes[0]
    ax.plot(ld, lp, color="#c0392b", lw=1.4, label="左段 2021顶后")
    ax.plot(rd, rp, color="#1f77b4", lw=1.8, label="右段 2025顶后")
    for d, name in LEFT_MARKS:
        ldays = (pd.Timestamp(d) - LEFT_TOP).days
        ax.scatter([ldays], [left_at(ldays)], c="#c0392b", s=45, zorder=3)
        ax.annotate(name, (ldays, left_at(ldays)), fontsize=8, color="#c0392b",
                    xytext=(3, -12), textcoords="offset points")
    ax.scatter([right_span], [rp[-1]], c="#1f77b4", s=55, zorder=3)
    ax.annotate(f"右侧当前\n{ridx.max():%y-%m-%d} {rp[-1]:.0f}%",
                (right_span, rp[-1]), fontsize=8, color="#1f77b4",
                xytext=(3, 8), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title("Panel1 原始时间轴：右侧目前走到左侧哪个阶段？")
    ax.set_xlabel("距顶天数"); ax.set_ylabel("相对顶回撤 %"); ax.legend()

    ax = axes[1]
    ax.plot(ld, lp, color="#c0392b", lw=1.4, label="左段 2021顶后")
    ax.plot(rd * alpha, rp, color="#1f77b4", lw=1.8,
            label=f"右段×α={alpha:.2f}（拉到左侧时间尺度）")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title(f"Panel2 时间缩放 α={alpha:.2f} 后：路径形状吻合吗？")
    ax.set_xlabel("距顶天数（左侧尺度）"); ax.set_ylabel("相对顶回撤 %"); ax.legend()

    plt.tight_layout()
    out = OUTDIR / "post_top_align.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n图：{out}")


if __name__ == "__main__":
    main()
