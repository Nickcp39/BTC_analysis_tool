"""
StepD5: 交互式标点工具（逐年/逐半年导航，你点大概位置 → 自动吸附最近真实高/低点）
用法（你本机终端运行，弹窗）：
    python code/stepD5_pick_points.py
一次只显示一段（默认一年），点完按 n 跳下一年；按 h 可把当前年切成上/下半年放大看。
操作：
    左键单击 = 标点（吸附到点击附近 ±吸附窗 天内最近的高/低点；红=高 绿=低）
    右键单击 = 撤销上一个点（全局最后一个）
    n / →    = 下一段（自动保存）       b / ←  = 上一段
    h        = 当前段 全年→上半年→下半年 循环放大
    + / -    = 调大/调小吸附窗
    s        = 保存                      关闭窗口 = 保存并退出
输出：output/core_points/manual_points.csv（n,date,price,type）+ manual_points.png
点完保存后告诉我，我读 CSV 继续做对齐/对应。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
for _bk in ("TkAgg", "Qt5Agg", "QtAgg", "MacOSX"):
    try:
        matplotlib.use(_bk)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepD1_core_points import read_series, zigzag_idx, classify, setup_cjk_font

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "core_points"
CSV = OUTDIR / "manual_points.csv"
PNG = OUTDIR / "manual_points.png"

CHUNK_MONTHS = 12     # 12=一年一段；改 6=半年一段
SNAP_DAYS = 15
REF_PIVOT_TH = 0.15
START_AT_END = True   # True=从最新一年(倒着)开始，用 ←/b 往更早走


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()
    ref = classify(s, zigzag_idx(s.values, REF_PIVOT_TH))

    # 生成导航窗口
    windows = []
    y0, y1 = s.index.min().year, s.index.max().year
    if CHUNK_MONTHS >= 12:
        for y in range(y0, y1 + 1):
            a = max(pd.Timestamp(y, 1, 1), s.index.min())
            b = min(pd.Timestamp(y, 12, 31), s.index.max())
            if a <= b:
                windows.append((a, b, str(y)))
    else:
        for y in range(y0, y1 + 1):
            for m0 in range(1, 13, CHUNK_MONTHS):
                a = max(pd.Timestamp(y, m0, 1), s.index.min())
                m1 = m0 + CHUNK_MONTHS - 1
                b = min(pd.Timestamp(y, m1, 1) + pd.offsets.MonthEnd(0), s.index.max())
                if a <= b:
                    windows.append((a, b, f"{y}-{m0:02d}~{min(m1,12):02d}"))

    # 续标：载入已有点（去重），避免重开覆盖；从最早已标点所在年继续，便于往更早补
    existing, seen = [], set()
    if CSV.exists():
        old = pd.read_csv(CSV)
        old["date"] = pd.to_datetime(old["date"])
        for _, r in old.sort_values("date").iterrows():
            d = r["date"].normalize()
            if d in seen:
                continue
            seen.add(d)
            existing.append({"date": d, "price": float(r["price"]), "type": str(r["type"])})

    def win_idx_for(date):
        for i, (a, b, _) in enumerate(windows):
            if a <= date <= b:
                return i
        return len(windows) - 1

    start_idx = (win_idx_for(min(p["date"] for p in existing)) if existing
                 else (len(windows) - 1 if START_AT_END else 0))
    st = {"idx": start_idx, "sub": 0, "snap": SNAP_DAYS, "pts": existing}

    def cur_range():
        a, b, label = windows[st["idx"]]
        if st["sub"] == 1:
            return a, a + (b - a) / 2, label + " 上半"
        if st["sub"] == 2:
            return a + (b - a) / 2, b, label + " 下半"
        return a, b, label

    def snap(d, logpx):
        w = pd.Timedelta(days=st["snap"])
        seg = s.loc[d - w: d + w]
        if len(seg) == 0:
            i = s.index.get_indexer([d], method="nearest")[0]
            return s.index[i], float(s.iloc[i]), "H"
        hi_d, hi_p = seg.idxmax(), float(seg.max())
        lo_d, lo_p = seg.idxmin(), float(seg.min())
        if abs(np.log(hi_p) - logpx) <= abs(np.log(lo_p) - logpx):
            return hi_d, hi_p, "H"
        return lo_d, lo_p, "L"

    def save():
        if not st["pts"]:
            return
        df = pd.DataFrame(sorted(st["pts"], key=lambda x: x["date"]))
        df.insert(0, "n", range(1, len(df) + 1))
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(CSV, index=False, encoding="utf-8-sig")
        print(f"[自动保存] {len(df)} 点 → {CSV}", flush=True)

    fig, ax = plt.subplots(figsize=(15, 7))

    def replot():
        a, b, label = cur_range()
        ax.clear()
        seg = s.loc[a:b]
        ax.plot(seg.index, seg.values, lw=1.0, color="#888", zorder=1)
        ax.set_yscale("log")
        r = ref[(ref["date"] >= a) & (ref["date"] <= b)]
        ax.scatter(r["date"], r["price"], s=16, c="#e6e6e6", edgecolors="#bbb",
                   linewidths=0.4, zorder=2)
        for i, p in enumerate(sorted(st["pts"], key=lambda x: x["date"])):
            if a <= p["date"] <= b:
                c = "#e74c3c" if p["type"] == "H" else "#1db954"
                ax.scatter([p["date"]], [p["price"]], s=100, c=c,
                           edgecolors="k", linewidths=0.8, zorder=5)
                ax.annotate(f"{i+1}", (p["date"], p["price"]),
                            textcoords="offset points", xytext=(0, 9), ha="center",
                            fontsize=9, fontweight="bold", zorder=6)
        ax.set_ylabel("价格(对数)")
        ax.set_title(f"[{label}]  第{st['idx']+1}/{len(windows)}段  吸附±{st['snap']}天  共{len(st['pts'])}点\n"
                     f"左键标点 右键撤销 | ←/b 更早一年  →/n 更晚一年  h半年放大 +/-吸附窗 s保存")
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            d = pd.Timestamp(mdates.num2date(event.xdata)).tz_localize(None).normalize()
            dd, px, typ = snap(d, np.log(event.ydata))
            st["pts"].append({"date": dd, "price": round(px, 2), "type": typ})
            print(f"  +点 {dd:%Y-%m-%d} {px:.0f} ({typ})", flush=True)
            save()          # 每点一下立刻落盘，崩了也不丢
            replot()
        elif event.button == 3:
            if st["pts"]:
                rm = st["pts"].pop()
                print(f"  -撤销 {rm['date']:%Y-%m-%d}", flush=True)
                save()
                replot()

    def on_key(event):
        if event.key in ("n", "right"):
            save()
            st["idx"] = min(st["idx"] + 1, len(windows) - 1)
            st["sub"] = 0
            replot()
        elif event.key in ("b", "left"):
            st["idx"] = max(st["idx"] - 1, 0)
            st["sub"] = 0
            replot()
        elif event.key == "h":
            st["sub"] = (st["sub"] + 1) % 3
            replot()
        elif event.key in ("+", "="):
            st["snap"] += 5
            replot()
        elif event.key == "-":
            st["snap"] = max(3, st["snap"] - 5)
            replot()
        elif event.key == "s":
            save()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    def on_close(e):
        try:
            save()
            fig.savefig(PNG, dpi=150)
        except Exception as ex:
            print("（关闭时存图出错，但点已在 CSV）", ex, flush=True)

    fig.canvas.mpl_connect("close_event", on_close)

    replot()
    print(f"窗口已打开：已载入 {len(st['pts'])} 个已标点，从 {windows[st['idx']][2]} 段继续。"
          f"←或b回更早年份，h半年，每点一下自动追加到 {CSV}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
