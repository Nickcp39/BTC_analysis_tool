"""逐段对比账本生成器（边比边改：改下面 RED/GREEN 锚点再重跑即可）。

- 只重算"脚手架"列(锚点/日期/gap/绿数据可用性note)
- 已填的测量/手填列(ratio, manual_*, recorded_at)按 seg_id 原样保留,不会被冲掉
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
LEDGER = OUT / "segment_ratio_ledger_v1.csv"
DATA = Path(__file__).resolve().parents[2] / "data" / "btc_merged_daily.csv"

# ---- 锚点（改这里）----
RED = {"peak": "2021-11-08", "bottom": "2022-11-21"}    # 2021 周期
GREEN = {"peak": "2025-08-12", "bottom": "2026-02-05"}   # 2025 周期（顶=主顶 8/12；底暂定）

# ---- 段定义: (id, 标签, 锚点, pre 偏移, post 偏移)；偏移用 pandas DateOffset 关键字 ----
SEGMENTS = [
    (1, "顶前半年", "peak", {"months": 6}, None),
    (2, "顶后半年", "peak", None, {"months": 6}),
    (3, "顶前一年", "peak", {"years": 1}, None),
    (4, "顶后一年", "peak", None, {"years": 1}),
    (5, "顶前两年", "peak", {"years": 2}, None),
    (6, "顶前三年", "peak", {"years": 3}, None),
    (7, "底前半年", "bottom", {"months": 6}, None),
    (8, "底后半年", "bottom", None, {"months": 6}),
    (9, "底前一年", "bottom", {"years": 1}, None),
    (10, "底后一年", "bottom", None, {"years": 1}),
]

COLS = ["seg_id", "seg_label", "anchor", "red_anchor", "green_anchor", "anchor_gap_days",
        "red_start", "red_end", "green_start", "green_end",
        "red_days", "green_days", "time_ratio", "red_log_move", "green_log_move", "amp_ratio",
        "manual_amp", "manual_time", "manual_shift", "note", "recorded_at"]
PRESERVE = ["red_days", "green_days", "time_ratio", "red_log_move", "green_log_move", "amp_ratio",
            "manual_amp", "manual_time", "manual_shift", "recorded_at"]


def span(anchor: str, pre, post):
    a = pd.Timestamp(anchor)
    start = a - pd.DateOffset(**pre) if pre else a
    end = a + pd.DateOffset(**post) if post else a
    return start, end


def main() -> None:
    data_end = pd.read_csv(DATA, parse_dates=["date"])["date"].max()
    rows = []
    for sid, label, anchor, pre, post in SEGMENTS:
        ra, ga = pd.Timestamp(RED[anchor]), pd.Timestamp(GREEN[anchor])
        rs, re = span(RED[anchor], pre, post)
        gs, ge = span(GREEN[anchor], pre, post)
        if ga > data_end:
            note = "绿无数据"
        elif ge > data_end:
            note = f"绿只到+{(data_end - ga).days}d"
        else:
            note = ""
        rows.append({
            "seg_id": sid, "seg_label": label, "anchor": anchor,
            "red_anchor": RED[anchor], "green_anchor": GREEN[anchor], "anchor_gap_days": (ga - ra).days,
            "red_start": rs.date(), "red_end": re.date(), "green_start": gs.date(), "green_end": ge.date(),
            "note": note,
        })
    df = pd.DataFrame(rows).reindex(columns=COLS)

    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        for c in PRESERVE:
            if c in old.columns:
                df[c] = df["seg_id"].map(dict(zip(old["seg_id"], old[c])))

    df.to_csv(LEDGER, index=False, encoding="utf-8-sig")
    pgap = (pd.Timestamp(GREEN["peak"]) - pd.Timestamp(RED["peak"])).days
    bgap = (pd.Timestamp(GREEN["bottom"]) - pd.Timestamp(RED["bottom"])).days
    print(f"red={RED}  green={GREEN}")
    print(f"peak gap={pgap}d  bottom gap={bgap}d  diff={pgap - bgap}d  | data_end={data_end.date()}")
    print(df[["seg_id", "seg_label", "green_anchor", "red_start", "red_end", "green_start", "green_end", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
