"""
StepD5: 交互式标点工具（你点大概位置 → 自动吸附最近的真实高/低点）
用法（在你本机终端运行，会弹窗）：
    python code/stepD5_pick_points.py
操作：
    左键单击 = 标一个点（自动吸附到点击位置附近 ±SNAP_DAYS 天内最近的高点或低点）
    右键单击 = 撤销上一个点
    s 键      = 保存
    + / -     = 调大/调小吸附窗口（点不准时可调大）
    工具栏放大镜/小手 = 缩放/平移（激活时不会误标点）
    关闭窗口  = 自动保存
输出：
    output/core_points/manual_points.csv   （n, date, price, type）
    output/core_points/manual_points.png   （快照，便于我这边查看你点了哪些）
点完保存后告诉我，我读 CSV 继续做对齐/对应分析。
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

# 可改：标点范围（默认全历史）、吸附窗口、参考 pivots 阈值
WIN_START, WIN_END = None, None          # 例：pd.Timestamp("2021-06-01"), pd.Timestamp("2026-05-31")
SNAP_DAYS = 15
REF_PIVOT_TH = 0.15                        # 灰色参考点（仅作视觉引导）


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()
    if WIN_START is not None:
        s = s.loc[WIN_START:WIN_END]

    state = {"snap": SNAP_DAYS, "pts": [], "artists": []}

    def snap(click_date, click_logpx):
        w = pd.Timedelta(days=state["snap"])
        seg = s.loc[click_date - w: click_date + w]
        if len(seg) == 0:
            # 退而求其次：全局最近日期
            i = s.index.get_indexer([click_date], method="nearest")[0]
            return s.index[i], float(s.iloc[i]), "H"
        hi_d, hi_p = seg.idxmax(), float(seg.max())
        lo_d, lo_p = seg.idxmin(), float(seg.min())
        if abs(np.log(hi_p) - click_logpx) <= abs(np.log(lo_p) - click_logpx):
            return hi_d, hi_p, "H"
        return lo_d, lo_p, "L"

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(s.index, s.values, lw=0.9, color="#888", zorder=1)
    ax.set_yscale("log")
    # 参考 pivots（灰，仅引导）
    ref = classify(s, zigzag_idx(s.values, REF_PIVOT_TH))
    ax.scatter(ref["date"], ref["price"], s=14, c="#e0e0e0", edgecolors="#bbb",
               linewidths=0.4, zorder=2)
    ax.set_ylabel("价格(对数)")

    def title():
        ax.set_title(f"左键=标点(吸附±{state['snap']}天) | 右键=撤销 | s=保存 | +/-改窗口 | "
                     f"已标 {len(state['pts'])} 点")

    def redraw():
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        for i, p in enumerate(sorted(state["pts"], key=lambda x: x["date"])):
            c = "#e74c3c" if p["type"] == "H" else "#1db954"
            sc = ax.scatter([p["date"]], [p["price"]], s=95, c=c,
                            edgecolors="k", linewidths=0.8, zorder=5)
            an = ax.annotate(f"{i+1}", (p["date"], p["price"]),
                             textcoords="offset points", xytext=(0, 9),
                             ha="center", fontsize=9, fontweight="bold", zorder=6)
            state["artists"] += [sc, an]
        title()
        fig.canvas.draw_idle()

    def save():
        if not state["pts"]:
            print("还没有标点，未保存。")
            return
        df = pd.DataFrame(sorted(state["pts"], key=lambda x: x["date"]))
        df.insert(0, "n", range(1, len(df) + 1))
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.to_csv(CSV, index=False, encoding="utf-8-sig")
        fig.savefig(PNG, dpi=150)
        print(f"已保存 {len(df)} 点 → {CSV}")
        print(df.to_string(index=False))

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        tb = fig.canvas.toolbar
        if tb is not None and getattr(tb, "mode", "") != "":
            return  # 缩放/平移激活时不标点
        if event.button == 1:
            d = pd.Timestamp(mdates.num2date(event.xdata)).tz_localize(None).normalize()
            dd, px, typ = snap(d, np.log(event.ydata))
            state["pts"].append({"date": dd, "price": round(px, 2), "type": typ})
            redraw()
        elif event.button == 3:
            if state["pts"]:
                state["pts"].pop()
                redraw()

    def on_key(event):
        if event.key == "s":
            save()
        elif event.key in ("+", "="):
            state["snap"] += 5
            title(); fig.canvas.draw_idle()
        elif event.key == "-":
            state["snap"] = max(3, state["snap"] - 5)
            title(); fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: save())

    title()
    plt.tight_layout()
    print("窗口已打开：左键标点，右键撤销，s 保存，关闭窗口自动保存。")
    plt.show()


if __name__ == "__main__":
    main()
