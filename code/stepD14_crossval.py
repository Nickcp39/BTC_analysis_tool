"""
StepD14: 2026 见底——多模型交叉验证（旧5模型 + AHR999估值底 + 本次节奏/叠加法）
- 读 2026-06-01 的 model_average_inputs.csv 与 ahr999_implied_bottom_prices.csv
- 加入本次"节奏/叠加"法估计（amp 0.53 → -54% → ~$57.7k；时间 ts≈1 → 2026-10）
- 输出统一对照表 + 共识区间 + 图
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepD1_core_points import setup_cjk_font

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "analysis_runs" / "2026-06-01_parent_report" / "tables" / "model_average_inputs.csv"
AHR = ROOT / "analysis_runs" / "2026-06-01_ahr999_bottom_check" / "tables" / "ahr999_implied_bottom_prices.csv"
OUT = ROOT / "output" / "core_points"


def main():
    setup_cjk_font()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    m = pd.read_csv(RUN)
    for _, r in m.iterrows():
        rows.append((r["model"].split("_")[0], r["center_price"], r["price_lo"], r["price_hi"], r["center_date"]))

    a = pd.read_csv(AHR)
    a = a[a["target_date"] == "2026-10-22"]
    deep = float(a[a["target"] == "hist_bottom_mean"]["implied_price"].iloc[0])
    mild = float(a[a["target"] == "mild_bottom_0_45"]["implied_price"].iloc[0])
    rows.append(("AHR999深底(0.27)", deep, deep * 0.97, deep * 1.03, "2026-10-22"))

    # 本次节奏/叠加法：amp 0.50~0.58，中枢 0.53
    top, dd21 = 124720.0, np.log(15756.0 / 67510.0)
    r_lo = top * np.exp(dd21 * 0.58)
    r_hi = top * np.exp(dd21 * 0.50)
    r_mid = top * np.exp(dd21 * 0.53)
    rows.append(("节奏/叠加(本次)", r_mid, r_lo, r_hi, "2026-10-18"))

    print(f"{'方法':22}{'中枢价':>11}{'区间':>20}{'中枢日期':>13}")
    for name, c, lo, hi, d in rows:
        print(f"{name:22}{c:>11,.0f}   [{lo:>7,.0f}, {hi:>7,.0f}]   {d:>11}")

    deep_prices = [c for name, c, *_ in rows if "0.45" not in name]
    print(f"\n共识(剔除浅底情景)：均值 ${np.mean(deep_prices):,.0f}  中位 ${np.median(deep_prices):,.0f}  "
          f"范围 ${min(deep_prices):,.0f}~${max(deep_prices):,.0f}")
    print(f"对照·若只到浅底 AHR999=0.45：${mild:,.0f}（更高）")
    print(f"时间：各模型中枢全在 2026-10（10-10~10-30）→ 支持 time_scale≈1（10月），非 v16 的 0.87（6月）")

    # 图
    plt.figure(figsize=(11, 5.5))
    names = [r[0] for r in rows]
    cen = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    yy = np.arange(len(rows))
    colors = ["#1db954" if "本次" in n else ("#e67e22" if "AHR" in n else "#5b8def") for n in names]
    plt.errorbar(cen, yy, xerr=[los, his], fmt="o", ecolor="#bbb", capsize=4,
                 markersize=8, markerfacecolor="none", linestyle="none")
    for i, (c, col) in enumerate(zip(cen, colors)):
        plt.scatter([c], [i], s=90, c=col, zorder=3)
    cons_lo, cons_hi = min(deep_prices), max(deep_prices)
    plt.axvspan(cons_lo, cons_hi, color="#1db954", alpha=0.08)
    plt.axvline(np.median(deep_prices), color="#1db954", ls="--", lw=1, label=f"共识中位 ${np.median(deep_prices):,.0f}")
    plt.yticks(yy, names)
    plt.xlabel("2025周期 见底价 (USD)")
    plt.title("2026 见底·多模型交叉验证（绿=本次节奏法 橙=AHR999 蓝=旧模型）")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = OUT / "bottom_crossval.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n图：{out}")


if __name__ == "__main__":
    main()
