"""
StepD10: 叠加拟合（红2021→绿2025）+ 追问 -70d 平移
- 模型(各自以"顶"为 rel_day=0)：green_y(g) ≈ amp · red_y((g - shift)/time_scale)
- 教训：自由三参数退化(amp/time/shift 互补偿，盲拟合滑向假时间压缩)。
  → 用独立证据钉住：amp=实测波动率比、time=1(宏观节奏稳)，只留 shift。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepD1_core_points import read_series

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOP21 = pd.Timestamp("2021-11-08")
TOP25 = pd.Timestamp("2025-10-05")
HALV21, HALV25 = pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-20")
BOT21, BOT25 = pd.Timestamp("2018-12-15"), pd.Timestamp("2022-11-09")


def curve(s, top, lo, hi):
    seg = s.loc[top - pd.Timedelta(days=lo): top + pd.Timedelta(days=hi)]
    rel = (seg.index - top).days.values.astype(float)
    y = np.log(seg.values / float(s.loc[:top].iloc[-1]))
    return rel, y


def rvol(s, end, lookback=365):
    seg = s.loc[end - pd.Timedelta(days=lookback): end]
    return float(np.std(np.diff(np.log(seg.values))))


def main():
    s = read_series()
    rr, ry = curve(s, TOP21, 560, 450)
    gr, gy = curve(s, TOP25, 540, 240)
    red_at = lambda d: np.interp(d, rr, ry, left=np.nan, right=np.nan)

    def fit(amp_fix=None, ts_fix=None):
        ts_grid = [ts_fix] if ts_fix else np.linspace(0.70, 1.30, 301)
        best = None
        for ts in ts_grid:
            for shift in np.arange(-150, 80, 1.0):
                pr = red_at((gr - shift) / ts)
                m = ~np.isnan(pr)
                if m.sum() < 50:
                    continue
                P, G = pr[m], gy[m]
                if amp_fix is not None:
                    amp = amp_fix
                else:
                    d = float(np.sum(P * P))
                    if d <= 0:
                        continue
                    amp = float(np.sum(G * P) / d)
                err = float(np.mean((G - amp * P) ** 2))
                if best is None or err < best[-1]:
                    best = (amp, ts, shift, err)
        return best

    vr = rvol(s, TOP25) / rvol(s, TOP21)
    print(f"实测波动率比 vol25/vol21 = {vr:.2f}（amp 的独立锚）\n")

    for label, kw in [("① 自由三参数(退化)", {}),
                      ("② 钉 amp=波动率比，time自由", {"amp_fix": round(vr, 2)}),
                      ("③ 钉 amp=波动率比 & time=1.0，只拟合 shift", {"amp_fix": round(vr, 2), "ts_fix": 1.0})]:
        amp, ts, shift, err = fit(**kw)
        print(f"{label}: amp={amp:.2f}  time={ts:.3f}  shift={shift:.0f}天  rmse={err**0.5:.3f}")

    print(f"\n对照另一工具手调：amp≈0.50  time≈0.9  shift≈-70")
    D = lambda a, b: (b - a).days
    print(f"\n70天 vs 宏观锚点差：减半→顶 {D(HALV21,TOP21)}vs{D(HALV25,TOP25)}(差{D(HALV21,TOP21)-D(HALV25,TOP25)})；"
          f"底→顶 {D(BOT21,TOP21)}vs{D(BOT25,TOP25)}(差{D(BOT21,TOP21)-D(BOT25,TOP25)}) → 都不到70")


if __name__ == "__main__":
    main()
