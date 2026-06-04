from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
NODES = ROOT / "analysis_runs" / "2026-06-04_green_anchor_probe" / "segment_nodes_v5.csv"
OUT = Path(__file__).resolve().parent

LEFT_TOP = pd.Timestamp("2021-11-08")
RIGHT_TOP = pd.Timestamp("2025-10-05")
MIN_RATIO = 0.70
MAX_RATIO = 1.30
MAX_REL_DAY = 520


def kind_group(kind: str) -> str:
    if kind in {"drop_end", "rally_start"}:
        return "low_or_start"
    if kind in {"rally_end", "drop_start"}:
        return "high_or_end"
    if "up" in kind:
        return "up_continuation"
    if "down" in kind:
        return "down_continuation"
    return "other"


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def load_nodes() -> pd.DataFrame:
    df = pd.read_csv(NODES, parse_dates=["date"])
    df["cycle"] = df["cycle"].astype(str)
    df["kind_group"] = df["kind"].map(kind_group)
    return df


def post_peak_nodes(nodes: pd.DataFrame, cycle: str, top: pd.Timestamp) -> pd.DataFrame:
    g = nodes[
        (nodes["cycle"] == cycle)
        & (nodes["date"] >= top)
        & (nodes["date"] <= top + pd.Timedelta(days=MAX_REL_DAY))
    ].copy()
    g["rel_from_peak"] = (g["date"] - top).dt.days
    return g.reset_index(drop=True)


def build_matches(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, a in left.iterrows():
        if int(a["rel_from_peak"]) <= 0:
            continue
        candidates = right[right["rel_from_peak"] > 0].copy()
        candidates["time_ratio"] = candidates["rel_from_peak"] / float(a["rel_from_peak"])
        candidates = candidates[
            (candidates["time_ratio"] >= MIN_RATIO)
            & (candidates["time_ratio"] <= MAX_RATIO)
        ]
        if candidates.empty:
            continue
        candidates["kind_penalty"] = np.where(
            candidates["kind"].eq(a["kind"]),
            0.0,
            np.where(candidates["kind_group"].eq(a["kind_group"]), 0.35, 1.25),
        )
        candidates["ratio_penalty"] = np.abs(np.log(candidates["time_ratio"]))
        candidates["score_match"] = candidates["ratio_penalty"] + candidates["kind_penalty"]
        b = candidates.sort_values(["score_match", "ratio_penalty"]).iloc[0]
        rows.append(
            {
                "left_index": int(a["anchor_index"]),
                "right_index": int(b["anchor_index"]),
                "left_date": a["date"].date().isoformat(),
                "right_date": b["date"].date().isoformat(),
                "left_rel_day": int(a["rel_from_peak"]),
                "right_rel_day": int(b["rel_from_peak"]),
                "time_ratio": round(float(b["rel_from_peak"] / a["rel_from_peak"]), 4),
                "left_kind": a["kind"],
                "right_kind": b["kind"],
                "left_group": a["kind_group"],
                "right_group": b["kind_group"],
                "left_price": round(float(a["price"]), 2),
                "right_price": round(float(b["price"]), 2),
                "score_match": round(float(b["score_match"]), 4),
            }
        )

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand

    # One-to-one, preserving the best ratio/kind match first.
    cand = cand.sort_values(["score_match", "left_rel_day"]).reset_index(drop=True)
    used_left = set()
    used_right = set()
    kept = []
    for _, row in cand.iterrows():
        li, ri = int(row["left_index"]), int(row["right_index"])
        if li in used_left or ri in used_right:
            continue
        used_left.add(li)
        used_right.add(ri)
        kept.append(row.to_dict())
    return pd.DataFrame(kept).sort_values("left_rel_day").reset_index(drop=True)


def amplitude_scale(series: pd.Series, matches: pd.DataFrame) -> float:
    left_top_px = float(series.loc[LEFT_TOP])
    right_top_px = float(series.loc[RIGHT_TOP])
    ratios = []
    for _, m in matches.iterrows():
        l_norm = np.log(float(m["left_price"]) / left_top_px)
        r_norm = np.log(float(m["right_price"]) / right_top_px)
        if abs(l_norm) > 1e-9:
            ratios.append(r_norm / l_norm)
    return float(np.median(ratios)) if ratios else 1.0


def path_df(series: pd.Series, amp: float) -> pd.DataFrame:
    latest = series.index.max()
    left = series.loc[LEFT_TOP : LEFT_TOP + pd.Timedelta(days=MAX_REL_DAY)]
    right = series.loc[RIGHT_TOP : min(latest, RIGHT_TOP + pd.Timedelta(days=MAX_REL_DAY))]
    left_top_px = float(series.loc[LEFT_TOP])
    right_top_px = float(series.loc[RIGHT_TOP])
    left_df = pd.DataFrame(
        {
            "cycle": "2021_annealed",
            "date": left.index,
            "rel_day": (left.index - LEFT_TOP).days,
            "price": left.values,
            "log_norm": np.log(left.values / left_top_px) * amp,
        }
    )
    right_df = pd.DataFrame(
        {
            "cycle": "2025_actual",
            "date": right.index,
            "rel_day": (right.index - RIGHT_TOP).days,
            "price": right.values,
            "log_norm": np.log(right.values / right_top_px),
        }
    )
    return pd.concat([left_df, right_df], ignore_index=True)


def write_html(plot: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, matches: pd.DataFrame, amp: float) -> None:
    p = plot.copy()
    p["date"] = p["date"].dt.date.astype(str)
    left_payload = left.copy()
    right_payload = right.copy()
    for df in (left_payload, right_payload):
        df["date"] = df["date"].dt.date.astype(str)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Ratio Guard v14</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ padding:12px 18px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 430px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 82px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 82px); overflow:auto; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.green {{ color:#00e676; }}
.red {{ color:#ff6a5f; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
</style>
</head>
<body>
<header><h1>v14 时间比例守门：对应点 rel-day 比例必须在 0.7 ~ 1.3</h1></header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const DATA = {json.dumps(p.to_dict(orient="records"), ensure_ascii=False)};
const LEFT = {json.dumps(left_payload.to_dict(orient="records"), ensure_ascii=False)};
const RIGHT = {json.dumps(right_payload.to_dict(orient="records"), ensure_ascii=False)};
const MATCHES = {json.dumps(matches.to_dict(orient="records"), ensure_ascii=False)};
const AMP = {round(amp, 4)};
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function x(d) {{ return M.l + d.rel_day / {MAX_REL_DAY} * (W - M.l - M.r); }}
function y(d) {{
  const min=-0.9, max=0.16;
  return H - M.b - (d.log_norm - min) / (max - min) * (H - M.t - M.b);
}}
function color(cycle) {{ return cycle === "2025_actual" ? "#00e676" : "#ff6a5f"; }}
for (let i=0;i<9;i++) {{
  const gx = M.l + i/8*(W-M.l-M.r);
  el("line", {{x1:gx,y1:M.t,x2:gx,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
}}
for (let i=0;i<7;i++) {{
  const gy = M.t + i/6*(H-M.t-M.b);
  el("line", {{x1:M.l,y1:gy,x2:W-M.r,y2:gy,stroke:"rgba(255,255,255,.08)"}});
}}
["2021_annealed","2025_actual"].forEach(cycle => {{
  const rows = DATA.filter(d => d.cycle === cycle).sort((a,b)=>a.rel_day-b.rel_day);
  let path = "";
  rows.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
  el("path", {{d:path, fill:"none", stroke:color(cycle), "stroke-width":cycle==="2025_actual"?2.4:2.1, opacity:cycle==="2025_actual"?.95:.78}});
}});
function pointFromMatch(m, side) {{
  if (side === "left") return {{rel_day:m.left_rel_day, log_norm:Math.log(m.left_price / DATA.find(d=>d.cycle==="2021_annealed" && d.rel_day===0).price)*AMP, label:m.left_index, date:m.left_date, kind:m.left_kind}};
  return {{rel_day:m.right_rel_day, log_norm:Math.log(m.right_price / DATA.find(d=>d.cycle==="2025_actual" && d.rel_day===0).price), label:m.right_index, date:m.right_date, kind:m.right_kind}};
}}
MATCHES.forEach(m => {{
  const a = pointFromMatch(m, "left");
  const b = pointFromMatch(m, "right");
  el("line", {{x1:x(a),y1:y(a),x2:x(b),y2:y(b),stroke:"rgba(255,255,255,.32)","stroke-width":1}});
  [a,b].forEach((p,i) => {{
    const c = el("circle", {{cx:x(p),cy:y(p),r:5.8,fill:i===0?"#ff6a5f":"#00e676",stroke:"#120807","stroke-width":1.3}});
    c.addEventListener("mousemove", ev => {{
      tip.style.display = "block";
      tip.style.left = ev.clientX + 12 + "px";
      tip.style.top = ev.clientY + 12 + "px";
      tip.innerHTML = `<b>${{i===0?"2021":"2025"}} #${{p.label}}</b><br>${{p.date}}<br>${{p.kind}}`;
    }});
    c.addEventListener("mouseleave", () => tip.style.display = "none");
  }});
}});
document.getElementById("summary").innerHTML = `
  <div class="strong">约束</div>
  <div>时间比例 = 2025 rel-day / 2021 rel-day</div>
  <div>允许区间：{MIN_RATIO} ~ {MAX_RATIO}</div>
  <div>通过匹配：${{MATCHES.length}} 个</div>
  <div>幅度退火系数：${{AMP}}</div>
  <br>
`;
document.getElementById("list").innerHTML = MATCHES.map(m => `
  <div class="item">
    <div class="strong">#${{m.left_index}} → #${{m.right_index}} · ratio ${{m.time_ratio}}</div>
    <div>${{m.left_date}} rel ${{m.left_rel_day}}d → ${{m.right_date}} rel ${{m.right_rel_day}}d</div>
    <div>${{m.left_kind}} → ${{m.right_kind}}</div>
  </div>
`).join("");
</script>
</body>
</html>"""
    (OUT / "green_anchor_ratio_guard_v14.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    nodes = load_nodes()
    left = post_peak_nodes(nodes, "2021", LEFT_TOP)
    right = post_peak_nodes(nodes, "2025", RIGHT_TOP)
    matches = build_matches(left, right)
    amp = amplitude_scale(series, matches)
    plot = path_df(series, amp)
    matches.to_csv(OUT / "ratio_guard_matches_v14.csv", index=False, encoding="utf-8-sig")
    plot.to_csv(OUT / "ratio_guard_overlay_v14.csv", index=False, encoding="utf-8-sig")
    write_html(plot, left, right, matches, amp)
    print(f"matches={len(matches)} amp={amp:.4f}")
    print(matches[["left_date", "right_date", "left_rel_day", "right_rel_day", "time_ratio", "left_kind", "right_kind"]].to_string(index=False))


if __name__ == "__main__":
    main()
