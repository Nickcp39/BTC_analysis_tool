from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parent
SAMPLES_JSON = OUT / "segment_cycle_samples_v19_saved.json"
SUMMARY_CSV = OUT / "segment_cycle_samples_v19_summary.csv"
SUMMARY_MD = OUT / "segment_cycle_samples_v19_summary.md"


def _median(vals: list[float]) -> float | None:
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return None
    n = len(vals)
    if n % 2:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def load_samples() -> list[dict]:
    data = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    return data.get("samples", [])


def summarize(samples: list[dict]) -> pd.DataFrame:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        key = (
            s.get("pair", ""),
            s.get("anchor_type", ""),
            s.get("left_anchor", ""),
            s.get("right_anchor", ""),
            int(s.get("pre_days") or 0),
            int(s.get("post_days") or 0),
        )
        groups[key].append(s)

    rows = []
    for (pair, anchor_type, left_anchor, right_anchor, pre_days, post_days), items in sorted(groups.items()):
        amps = [float(x.get("amp_scale")) for x in items if x.get("amp_scale") is not None]
        times = [float(x.get("time_scale")) for x in items if x.get("time_scale") is not None]
        shifts = [float(x.get("shift_days")) for x in items if x.get("shift_days") is not None]
        rmses = [float(x.get("rmse")) for x in items if x.get("rmse") is not None]
        rows.append(
            {
                "pair": pair,
                "anchor_type": anchor_type,
                "left_anchor": left_anchor,
                "right_anchor": right_anchor,
                "pre_days": pre_days,
                "post_days": post_days,
                "n": len(items),
                "amp_median": _median(amps),
                "amp_min": min(amps) if amps else None,
                "amp_max": max(amps) if amps else None,
                "time_median": _median(times),
                "time_min": min(times) if times else None,
                "time_max": max(times) if times else None,
                "shift_median": _median(shifts),
                "shift_min": min(shifts) if shifts else None,
                "shift_max": max(shifts) if shifts else None,
                "rmse_median": _median(rmses),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(samples: list[dict], summary: pd.DataFrame) -> None:
    lines = [
        "# Segment Cycle Samples v19 Summary",
        "",
        f"Samples: {len(samples)}",
        "",
        "## Grouped Windows",
        "",
        "| pair | anchor | left -> right | window | n | amp median | time median | shift median | rmse median |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        window = f"-{int(r.pre_days)} / +{int(r.post_days)}"
        anchors = f"{r.left_anchor} -> {r.right_anchor}"
        lines.append(
            "| {pair} | {anchor} | {anchors} | {window} | {n} | {amp} | {time} | {shift} | {rmse} |".format(
                pair=r.pair,
                anchor=r.anchor_type,
                anchors=anchors,
                window=window,
                n=int(r.n),
                amp=_fmt(r.amp_median, 3),
                time=_fmt(r.time_median, 3),
                shift=_fmt(r.shift_median, 1),
                rmse=_fmt(r.rmse_median, 4),
            )
        )

    lines.extend(["", "## Raw Samples", ""])
    for i, s in enumerate(samples, start=1):
        lines.append(
            f"{i}. {s.get('pair')} {s.get('anchor_type')} "
            f"{s.get('left_anchor')} -> {s.get('right_anchor')} "
            f"window -{s.get('pre_days')}/+{s.get('post_days')} "
            f"amp={s.get('amp_scale')} time={s.get('time_scale')} shift={s.get('shift_days')} rmse={s.get('rmse')}"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    samples = load_samples()
    summary = summarize(samples)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    write_markdown(samples, summary)
    print(f"samples: {len(samples)}")
    print(f"summary rows: {len(summary)}")
    print(summary.to_string(index=False))
    print("csv:", SUMMARY_CSV)
    print("md:", SUMMARY_MD)


if __name__ == "__main__":
    main()
