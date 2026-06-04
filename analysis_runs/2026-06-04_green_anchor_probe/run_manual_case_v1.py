from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
CASES = Path(__file__).resolve().parent / "manual_cycle_cases_v1.json"
OUT = Path(__file__).resolve().parent
NODES = OUT / "segment_nodes_v5.csv"


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def load_cases() -> list[dict]:
    data = json.loads(CASES.read_text(encoding="utf-8"))
    return data["cases"]


def nearest(series: pd.Series, date_text: str) -> pd.Timestamp:
    target = pd.Timestamp(date_text)
    if target in series.index:
        return target
    return series.index[int(abs((series.index - target).days).argmin())]


def project_price(anchor_right_price: float, anchor_left_price: float, left_price: float, amp: float) -> float:
    return float(anchor_right_price * np.exp(np.log(left_price / anchor_left_price) * amp))


def build_case_outputs(case: dict, series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    left_anchor = nearest(series, case["left_anchor"])
    right_anchor = nearest(series, case["right_anchor"])
    left_anchor_price = float(series.loc[left_anchor])
    right_anchor_price = float(series.loc[right_anchor])
    amp = float(case["amp_scale"])
    time_scale = float(case["time_scale"])
    shift = float(case["shift_days"])
    pre = int(case["pre_days"])
    post = int(case["post_days"])

    left = series.loc[left_anchor - pd.Timedelta(days=pre) : left_anchor + pd.Timedelta(days=post)]
    right = series.loc[right_anchor - pd.Timedelta(days=pre) : right_anchor + pd.Timedelta(days=post)]

    left_rows = []
    for d, px in left.items():
        rel = (d - left_anchor).days
        model_rel = rel * time_scale + shift
        model_date = right_anchor + pd.Timedelta(days=int(round(model_rel)))
        proj = project_price(right_anchor_price, left_anchor_price, float(px), amp)
        left_rows.append(
            {
                "cycle": "left_projected",
                "source_date": d.date().isoformat(),
                "model_date": model_date.date().isoformat(),
                "rel_day": rel,
                "model_rel_day": model_rel,
                "source_price": float(px),
                "projected_price": proj,
                "projected_return_from_right_anchor_pct": (proj / right_anchor_price - 1.0) * 100.0,
            }
        )

    right_rows = []
    for d, px in right.items():
        rel = (d - right_anchor).days
        right_rows.append(
            {
                "cycle": "right_actual",
                "source_date": d.date().isoformat(),
                "model_date": d.date().isoformat(),
                "rel_day": rel,
                "model_rel_day": rel,
                "source_price": float(px),
                "projected_price": float(px),
                "projected_return_from_right_anchor_pct": (float(px) / right_anchor_price - 1.0) * 100.0,
            }
        )

    overlay = pd.DataFrame(left_rows + right_rows)

    nodes = pd.read_csv(NODES, parse_dates=["date"])
    nodes["cycle"] = nodes["cycle"].astype(str)
    left_cycle = str(pd.Timestamp(case["left_anchor"]).year if pd.Timestamp(case["left_anchor"]).year < 2024 else 2025)
    if "2021" in case["left_anchor"] or "2022" in case["left_anchor"] or "2023" in case["left_anchor"]:
        left_cycle = "2021"
    elif "2017" in case["left_anchor"] or "2018" in case["left_anchor"] or "2019" in case["left_anchor"]:
        left_cycle = "2017"
    else:
        left_cycle = "2025"

    node_window = nodes[
        (nodes["cycle"] == left_cycle)
        & (nodes["date"] >= left_anchor - pd.Timedelta(days=pre))
        & (nodes["date"] <= left_anchor + pd.Timedelta(days=post))
    ].copy()
    if left_anchor not in set(node_window["date"]):
        node_window = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "anchor_index": 0,
                            "date": left_anchor,
                            "kind": "manual_anchor",
                            "price": left_anchor_price,
                        }
                    ]
                ),
                node_window,
            ],
            ignore_index=True,
        )
    node_window = node_window.sort_values("date")

    node_rows = []
    for _, row in node_window.iterrows():
        d = pd.Timestamp(row["date"])
        if d not in series.index:
            continue
        rel = (d - left_anchor).days
        model_rel = rel * time_scale + shift
        model_date = right_anchor + pd.Timedelta(days=int(round(model_rel)))
        source_px = float(series.loc[d])
        proj = project_price(right_anchor_price, left_anchor_price, source_px, amp)
        node_rows.append(
            {
                "left_node_index": row.get("anchor_index", ""),
                "left_date": d.date().isoformat(),
                "left_rel_day": rel,
                "left_kind": row.get("kind", ""),
                "left_price": source_px,
                "model_date": model_date.date().isoformat(),
                "model_rel_day": int(round(model_rel)),
                "projected_price": proj,
                "projected_return_from_right_anchor_pct": (proj / right_anchor_price - 1.0) * 100.0,
            }
        )
    node_projection = pd.DataFrame(node_rows)

    actual_latest = series.index.max()
    current_date = pd.Timestamp("2026-06-04")
    if current_date not in series.index:
        current_price = np.nan
    else:
        current_price = float(series.loc[current_date])

    future = pd.DataFrame(left_rows)
    future["model_date_ts"] = pd.to_datetime(future["model_date"])
    future_after_today = future[future["model_date_ts"] >= current_date].copy()
    low_row = future_after_today.loc[future_after_today["projected_price"].idxmin()].to_dict()

    summary = {
        "case_id": case["id"],
        "case_name": case["name"],
        "left_anchor": left_anchor.date().isoformat(),
        "right_anchor": right_anchor.date().isoformat(),
        "left_anchor_price": round(left_anchor_price, 2),
        "right_anchor_price": round(right_anchor_price, 2),
        "amp_scale": amp,
        "time_scale": time_scale,
        "shift_days": shift,
        "current_date": current_date.date().isoformat(),
        "current_price_from_data": None if np.isnan(current_price) else round(current_price, 2),
        "data_latest": actual_latest.date().isoformat(),
        "projected_future_low_date": low_row["model_date"],
        "projected_future_low_price": round(float(low_row["projected_price"]), 2),
        "projected_future_low_source_date": low_row["source_date"],
        "days_until_projected_low_from_2026_06_04": int((pd.Timestamp(low_row["model_date"]) - current_date).days),
    }
    return overlay, node_projection, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    case = load_cases()[0]
    overlay, nodes, summary = build_case_outputs(case, series)

    prefix = OUT / "case01_peak_2021_to_2025"
    overlay.to_csv(prefix.with_name(prefix.name + "_overlay.csv"), index=False, encoding="utf-8-sig")
    nodes.to_csv(prefix.with_name(prefix.name + "_nodes.csv"), index=False, encoding="utf-8-sig")
    prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Case 01: Peak 2021 -> Peak 2025",
        "",
        "## Parameters",
        f"- left anchor: `{summary['left_anchor']}` (${summary['left_anchor_price']:,.0f})",
        f"- right anchor: `{summary['right_anchor']}` (${summary['right_anchor_price']:,.0f})",
        f"- amp scale: `{summary['amp_scale']}`",
        f"- time scale: `{summary['time_scale']}`",
        f"- shift days: `{summary['shift_days']}`",
        "",
        "## Projection",
        f"- projected future low: `{summary['projected_future_low_date']}`",
        f"- projected low price: `${summary['projected_future_low_price']:,.0f}`",
        f"- source analog date: `{summary['projected_future_low_source_date']}`",
        f"- days from 2026-06-04: `{summary['days_until_projected_low_from_2026_06_04']}`",
        "",
        "## Key Projected Nodes",
        nodes[["left_date", "left_kind", "model_date", "projected_price", "projected_return_from_right_anchor_pct"]]
        .to_string(index=False, formatters={
            "projected_price": lambda x: f"${x:,.0f}",
            "projected_return_from_right_anchor_pct": lambda x: f"{x:.1f}%",
        }),
    ]
    prefix.with_name(prefix.name + "_report.md").write_text("\n".join(md), encoding="utf-8")

    log_path = OUT / "manual_review_log_v1.csv"
    if log_path.exists():
        log = pd.read_csv(log_path)
        mask = log["case_id"].eq(case["id"])
        if mask.any():
            log.loc[mask, "status"] = "codex_initial_run_complete"
            log.loc[mask, "manual_amp_scale"] = case["amp_scale"]
            log.loc[mask, "manual_time_scale"] = case["time_scale"]
            log.loc[mask, "manual_shift_days"] = case["shift_days"]
            log.loc[mask, "notes"] = (
                f"Initial run: projected low {summary['projected_future_low_date']} "
                f"at ${summary['projected_future_low_price']:,.0f}"
            )
            log.to_csv(log_path, index=False, encoding="utf-8-sig")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(prefix.with_name(prefix.name + "_report.md"))


if __name__ == "__main__":
    main()
