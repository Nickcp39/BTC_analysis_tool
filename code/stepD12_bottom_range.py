"""
StepD12: 2025周期真底的【时间区间】+【价格区间】（多条独立路线取范围）
- 时间：用已验证的稳定节奏——顶+熊市、减半+(减半→底)、底+周期长、工作台(顶+rel)。
- 价格：用顶价 × 回撤退火——三档(同2021深 / 回撤趋势收缩 / 波动率退火浅)。
- 锚点价/日取自用户手标真顶真底。诚实：n小、退火到底的程度不确定→价格区间宽。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

T = pd.Timestamp
TOP25 = T("2025-10-05"); TOP25_PX = 124720.0
# 历史真顶/真底（手标）
TOP17, BOT18 = 19650.0, 3183.0
TOP21, BOT22 = 67510.0, 15850.0
HALV24 = T("2024-04-20")
BOT22_D = T("2022-11-09")
# 稳定节奏（前面跑出来的）
BEAR = [364, 366, 410]            # 顶→底
HALV_TO_BOT = [889, 912]          # 减半→底
BOT_TO_BOT = [1425, 1431]         # 底→底
VOL_RATIO = 0.54                  # vol25/vol21
WB_SHIFT, WB_TS = -58, 1.0        # 工作台拟合


def main():
    print("================ 时间区间 ================")
    cands = []
    for b in BEAR:
        d = TOP25 + pd.Timedelta(days=b); cands.append(("顶+熊市%d" % b, d))
    for h in HALV_TO_BOT:
        d = HALV24 + pd.Timedelta(days=h); cands.append(("减半+%d" % h, d))
    for bb in BOT_TO_BOT:
        d = BOT22_D + pd.Timedelta(days=bb); cands.append(("上轮底+%d" % bb, d))
    # 工作台模型：2021底 rel_day=366 → 2025 rel = 366*ts + shift
    wb_rel = 366 * WB_TS + WB_SHIFT
    cands.append(("工作台(顶+%d)" % wb_rel, TOP25 + pd.Timedelta(days=int(wb_rel))))
    for name, d in cands:
        print(f"  {name:16} → {d:%Y-%m-%d}")
    dates = [d for _, d in cands]
    # 主区间：剔除工作台(相位外推不稳)的三条稳健法收敛
    robust = [d for n, d in cands if not n.startswith("工作台")]
    print(f"\n  全部范围：{min(dates):%Y-%m-%d} ~ {max(dates):%Y-%m-%d}")
    print(f"  稳健三法收敛：{min(robust):%Y-%m-%d} ~ {max(robust):%Y-%m-%d}（中位 {sorted(robust)[len(robust)//2]:%Y-%m-%d}）")
    print(f"  → 中枢 ≈ 2026年10月初；区间 ~2026-09 至 2026-11")

    print("\n================ 价格区间 ================")
    dd17 = np.log(BOT18 / TOP17)   # 2017轮 顶→底 对数回撤
    dd21 = np.log(BOT22 / TOP21)   # 2021轮
    print(f"  历史 顶→底 回撤：2017={np.exp(dd17)-1:+.1%}  2021={np.exp(dd21)-1:+.1%}（在变浅）")
    shrink = dd21 / dd17           # 逐轮对数回撤收缩比
    scenarios = {
        "① 同2021深(不再退火)": dd21,
        "② 回撤趋势延续(×%.2f)" % shrink: dd21 * shrink,
        "③ 波动率退火(×%.2f)" % VOL_RATIO: dd21 * VOL_RATIO,
    }
    prices = {}
    for name, dd in scenarios.items():
        px = TOP25_PX * np.exp(dd)
        prices[name] = px
        print(f"  {name:24} 回撤{np.exp(dd)-1:+.1%} → ${px:,.0f}")
    lo, hi = min(prices.values()), max(prices.values())
    print(f"\n  价格区间：${lo:,.0f} ~ ${hi:,.0f}")
    print(f"  → 回撤已逐轮变浅(-84%→-76%)且波动率腰斩，倾向比2021浅；")
    print(f"    中枢 ≈ ${prices['② 回撤趋势延续(×%.2f)' % shrink]:,.0f}（回撤趋势法）；模型上限 ${prices['③ 波动率退火(×%.2f)' % VOL_RATIO]:,.0f}")


if __name__ == "__main__":
    main()
