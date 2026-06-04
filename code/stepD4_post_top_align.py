"""
StepD4 (v2): 顶后路径对齐 —— 同时缩放【时间 α】+【幅度 A】，结构才可比
- 用户纠正：BTC 波动率持续下降→涨跌幅度逐周期变小，不能直接硬比幅度(−49% vs −77%)；
  时间也在缩放(不完美)。两轴都缩放后，结构是一致的。
- 做法：对数回撤 y=ln(P/顶)。模型 右(d) ≈ A · 左(α·d)。
  网格搜 α(时间，左日=α·右日；α<1=右更慢)，每个 α 闭式解 A(幅度比，右/左；A<1=右更浅)。
- 验证：用真实波动率(顶前365日 daily logret std)算各周期 vol，看拟合出的 A 是否≈ vol_右/vol_左
  （若接近，说明幅度差确实由"波动率下降"解释，而非过拟合）。
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
TOP_2017 = pd.Timestamp("2017-12-16")
LEFT_TOP = pd.Timestamp("2021-11-08")   # 左模板
RIGHT_TOP = pd.Timestamp("2025-10-05")  # 右(待对齐)

LEFT_MARKS = [
    ("2022-01-22", "首低"),
    ("2022-03-29", "②次高"),
    ("2022-06-18", "急跌"),
    ("2022-11-21", "③底"),
]


def logdd(s, top, ndays):
    seg = s.loc[top: top + pd.Timedelta(days=ndays)]
    days = (seg.index - top).days.values.astype(float)
    y = np.log(seg.values / seg.values[0])  # 对数回撤(<=0)
    return days, y, seg.index


def realized_vol(s, end, lookback=365):
    seg = s.loc[end - pd.Timedelta(days=lookback): end]
    r = np.diff(np.log(seg.values))
    return float(np.std(r))


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()

    ld, ly, _ = logdd(s, LEFT_TOP, 700)
    rd, ry, ridx = logdd(s, RIGHT_TOP, 700)
    right_span = int(rd.max())

    def lat(d):
        return np.interp(d, ld, ly)

    # 联合拟合 α(时间) + A(幅度)
    alphas = np.linspace(0.3, 2.5, 1101)
    best = None
    for a in alphas:
        t = lat(a * rd)
        denom = float(np.sum(t * t))
        A = float(np.sum(ry * t) / denom) if denom > 0 else 1.0
        err = float(np.mean((ry - A * t) ** 2))
        if best is None or err < best[2]:
            best = (a, A, err)
    alpha, A, _ = best

    # 真实波动率(顶前365日)
    v17 = realized_vol(s, TOP_2017)
    v21 = realized_vol(s, LEFT_TOP)
    v25 = realized_vol(s, RIGHT_TOP)
    A_vol = v25 / v21

    print(f"右段(2025顶后)现有 {right_span} 天，到 {ridx.max():%Y-%m-%d}")
    print(f"\n联合拟合：α(时间)={alpha:.2f}  ({'右更慢/拉长' if alpha<1 else '右更快/压缩'})"
          f"   A(幅度,右/左)={A:.2f}  (右只有左的 {A*100:.0f}% 幅度)")
    print(f"\n真实波动率(顶前365日 daily logret std)：2017={v17:.4f}  2021={v21:.4f}  2025={v25:.4f}")
    print(f"  逐周期下降比：2021/2017={v21/v17:.2f}   2025/2021={v25/v21:.2f}")
    print(f"  拟合 A={A:.2f}  vs  波动率预测 A=vol25/vol21={A_vol:.2f}  "
          f"→ {'吻合! 幅度差确由波动率下降解释' if abs(A-A_vol)<0.15 else '有偏差，幅度差不只波动率'}")

    print("\n左侧地标 → 右侧对应(按 α 折算实际日期)：")
    for d, name in LEFT_MARKS:
        ldays = (pd.Timestamp(d) - LEFT_TOP).days
        rdays = ldays / alpha
        rdate = RIGHT_TOP + pd.Timedelta(days=rdays)
        print(f"  {name:6} 左第{ldays:>4}天 → 右第{rdays:>5.0f}天 ≈ {rdate:%Y-%m-%d}  "
              f"[{'已过' if rdays<=right_span else '未到'}]")

    # ---- 画图：左=原始硬比；右=两轴缩放后 ----
    def pct(y):
        return (np.exp(y) - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))

    ax = axes[0]
    ax.plot(ld, pct(ly), color="#c0392b", lw=1.4, label="左 2021顶后")
    ax.plot(rd, pct(ry), color="#1f77b4", lw=1.8, label="右 2025顶后")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title("① 原始硬比（难看、比例不对）")
    ax.set_xlabel("距顶天数"); ax.set_ylabel("回撤 %"); ax.legend()

    ax = axes[1]
    ax.plot(ld, pct(ly), color="#c0392b", lw=1.6, label="左 2021顶后(模板)")
    # 右映射到左的框架：x=α·d，y=右幅度/A（除以A放大到左尺度）
    ax.plot(alpha * rd, pct(ry / A), color="#1f77b4", lw=2.0,
            label=f"右×(α={alpha:.2f}, A={A:.2f}) 缩放后")
    for d, name in LEFT_MARKS:
        ldays = (pd.Timestamp(d) - LEFT_TOP).days
        ax.scatter([ldays], [pct(lat(ldays))], c="#c0392b", s=45, zorder=3)
        ax.annotate(name, (ldays, pct(lat(ldays))), fontsize=8, color="#c0392b",
                    xytext=(3, -12), textcoords="offset points")
    ax.axvline(alpha * right_span, color="#1f77b4", ls="--", lw=0.9)
    ax.annotate(f"右侧当前到这\n≈左第{alpha*right_span:.0f}天",
                (alpha * right_span, pct(ly).min() * 0.5), fontsize=8, color="#1f77b4")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_title(f"② 两轴缩放后(α时间+A幅度)：结构是否一致？")
    ax.set_xlabel("距顶天数(左尺度)"); ax.set_ylabel("回撤 %(左尺度)"); ax.legend()

    plt.tight_layout()
    out = OUTDIR / "post_top_align.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n图：{out}")


if __name__ == "__main__":
    main()
