"""BTC multi-cycle manual workbench v20 — page builder.

Made by Claude (Claude Code, Opus 5.5), 2026-11-13, rebuilt from the v19 segment
workbench (analysis_runs/2026-06-04_segment_experiments/). The logic is unchanged:
each cycle is aligned to an anchor (peak / bottom / halving), its days are
transformed as rel_day x time + shift and its log move vs the anchor as log x amp.
The user drags; nothing here fits parameters automatically.

What is new in v20: five cycles at once (2011 / 2013 / 2017 / 2021 / 2025) with one
reference cycle, drag on the chart, undo, price or AHR999, project-file samples with
a PNG snapshot per sample, and the June v19 samples readable in the same page.

Run:  python cycle_workbench_v20.py        (rebuild with the latest data)
      python workbench_server_v20.py       (then open http://127.0.0.1:8766/)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from signals.indicators import compute_ahr999  # noqa: E402

MERGED = ROOT / "data" / "btc_merged_daily.csv"          # FRED CBBTCUSD from 2014-12-01 (+ Yahoo last row)
HISTORY = ROOT / "data" / "btc_price_daily.csv"          # CryptoCompare full history from 2010-07-17
MANUAL_POINTS = ROOT / "output" / "core_points" / "manual_points.csv"
POINT_DIR = ROOT / "analysis_runs" / "2026-11-13_manual_point_compare"
KEY_POINTS = POINT_DIR / "marked_points.csv"
USER_PAIRS = POINT_DIR / "user_pairs.csv"
KEY_SOURCES = {"user", "user_pair", "report_anchor", "manual_points"}
TEMPLATE = OUT / "cycle_workbench_v20_template.html"
HTML = OUT / "cycle_workbench_v20.html"
BUILD_LOG = OUT / "cycle_workbench_v20_build.txt"

# Anchors follow project memory: 2025 top = 2025-08-12 (never 10-05), bottom 2026-06-30.
CYCLES = [
    {"id": "2011", "label": "2011 轮", "color": "#9085e9", "halving": None,
     "peak": "2011-06-08", "bottom": "2011-11-18",
     "note": "减半前的一轮；早期数据来自 CryptoCompare，成交稀薄，只作参照"},
    {"id": "2013", "label": "2013 轮", "color": "#c98500", "halving": "2012-11-28",
     "peak": "2013-11-30", "bottom": "2015-01-14",
     "note": "2014-12 之前是 CryptoCompare 数据，2013 年顶部含 Mt.Gox 溢价；2014-02-06～02-25 Mt.Gox 崩盘段用前后插值"},
    {"id": "2017", "label": "2017 轮", "color": "#3987e5", "halving": "2016-07-09",
     "peak": "2017-12-16", "bottom": "2018-12-15", "note": ""},
    {"id": "2021", "label": "2021 轮", "color": "#d95926", "halving": "2020-05-11",
     "peak": "2021-11-08", "bottom": "2022-11-21", "note": ""},
    {"id": "2025", "label": "2025 轮（本轮）", "color": "#199e70", "halving": "2024-04-20",
     "peak": "2025-08-12", "bottom": "2026-06-30", "note": "顶 = 2025-08-12（真顶，不用 10-05）"},
]

# CryptoCompare follows Mt.Gox while it collapsed (e.g. $112 on 2014-02-20 while other
# exchanges traded near $600). Those days are replaced by log-linear interpolation
# between 2014-02-05 and 2014-02-26. Other big single-day moves before 2014 are real
# (thin early markets) and are left alone.
GOX_WINDOW = ("2014-02-06", "2014-02-25")


def load_prices() -> tuple[pd.Series, list[str], str]:
    new = pd.read_csv(MERGED, parse_dates=["date"]).set_index("date")["price"].sort_index()
    old = pd.read_csv(HISTORY, parse_dates=["date"]).set_index("date")["price"].sort_index()
    fred_start = new.index.min()
    old = old[old.index < fred_start].asfreq("D").interpolate()
    logp = np.log(old)
    gox = (old.index >= GOX_WINDOW[0]) & (old.index <= GOX_WINDOW[1])
    logp[gox] = np.nan
    old_clean = np.exp(logp.interpolate())
    s = pd.concat([old_clean, new]).sort_index()
    s = s[~s.index.duplicated(keep="last")].asfreq("D").ffill()
    return s, [d.date().isoformat() for d in old.index[gox]], fred_start.date().isoformat()


def load_marks(series: pd.Series) -> list[dict]:
    marks: list[dict] = []
    if MANUAL_POINTS.exists():
        df = pd.read_csv(MANUAL_POINTS, parse_dates=["date"]).drop_duplicates("date")
        for _, r in df.iterrows():
            marks.append({"date": r["date"].date().isoformat(), "type": str(r["type"]),
                          "label": f"手标 {r['type']} #{int(r['n'])}"})
    found: dict[str, list[str]] = {}
    if KEY_POINTS.exists():
        kp = pd.read_csv(KEY_POINTS, dtype=str).fillna("")
        for _, r in kp.iterrows():
            if r["date"] and r["source"] in KEY_SOURCES:
                found.setdefault(r["date"], []).append(f"{r['cycle']} {r['label']}")
    if USER_PAIRS.exists():
        up = pd.read_csv(USER_PAIRS, dtype=str).fillna("")
        for _, r in up.iterrows():
            for col in ("date_2021", "date_2025"):
                if r[col]:
                    found.setdefault(r[col], []).append(f"配对 {r['pair']}")
    for date, labels in sorted(found.items()):
        if pd.Timestamp(date) in series.index:
            marks.append({"date": date, "type": "K", "label": " / ".join(dict.fromkeys(labels))})
    return marks


def sig(x: float, n: int) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.{n}g}")


def main() -> None:
    series, spikes, fred_start = load_prices()
    ahr = compute_ahr999(pd.DataFrame({"date": series.index, "price": series.values}))
    marks = load_marks(series)
    data = {
        "start": series.index[0].date().isoformat(),
        "end": series.index[-1].date().isoformat(),
        "fred_start": fred_start,
        "price": [sig(v, 6) for v in series.values],
        "ahr": [sig(v, 5) for v in ahr["ahr999"].values],
        "cycles": CYCLES,
        "marks": marks,
        "despiked": spikes,
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA_JSON__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    HTML.write_text(html, encoding="utf-8")
    lines = [
        "cycle_workbench_v20 build (made by Claude)",
        f"data: {data['start']} -> {data['end']} ({len(series)} days); FRED from {fred_start}",
        f"last row: {data['end']} {series.iloc[-1]:,.2f} (may be an intraday Yahoo point)",
        f"Mt.Gox days interpolated: {len(spikes)} ({spikes[0]} -> {spikes[-1]})",
        f"marks: {sum(m['type'] != 'K' for m in marks)} hand marks + {sum(m['type'] == 'K' for m in marks)} confirmed points/pairs",
    ]
    for c in CYCLES:
        parts = []
        for key in ("halving", "peak", "bottom"):
            if c[key]:
                ts = pd.Timestamp(c[key])
                parts.append(f"{key} {c[key]} ${series.loc[ts]:,.2f} AHR {ahr.set_index('date').loc[ts, 'ahr999']:.3f}"
                             if ts in series.index else f"{key} {c[key]} (no data)")
        lines.append(f"{c['id']}: " + " | ".join(parts))
    lines.append(f"html: {HTML}")
    BUILD_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
