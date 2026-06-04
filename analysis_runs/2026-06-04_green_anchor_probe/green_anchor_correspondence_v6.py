from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NODES = OUT / "segment_nodes_v5.csv"

HALVINGS = {
    "2017": pd.Timestamp("2016-07-09"),
    "2021": pd.Timestamp("2020-05-11"),
    "2025": pd.Timestamp("2024-04-20"),
}
PAIR_SPANS = {
    ("2017", "2021"): (HALVINGS["2021"] - HALVINGS["2017"]).days,
    ("2021", "2025"): (HALVINGS["2025"] - HALVINGS["2021"]).days,
}


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


def load_nodes() -> pd.DataFrame:
    if not NODES.exists():
        raise FileNotFoundError(f"Missing {NODES}; run green_anchor_segments_v5.py first.")
    df = pd.read_csv(NODES, parse_dates=["date"])
    df["cycle"] = df["cycle"].astype(str)
    df["kind_group"] = df["kind"].map(kind_group)
    return df


def match_pair(df: pd.DataFrame, cycle_a: str, cycle_b: str, tolerance_days: int = 70) -> pd.DataFrame:
    span = PAIR_SPANS[(cycle_a, cycle_b)]
    a = df[df["cycle"] == cycle_a].copy()
    b = df[df["cycle"] == cycle_b].copy()
    candidates = []

    for _, left in a.iterrows():
        target = left["date"] + pd.Timedelta(days=span)
        for _, right in b.iterrows():
            delta = int((right["date"] - target).days)
            abs_delta = abs(delta)
            if abs_delta > tolerance_days:
                continue
            same_kind = left["kind"] == right["kind"]
            same_group = left["kind_group"] == right["kind_group"]
            if same_kind:
                kind_penalty = 0
            elif same_group:
                kind_penalty = 12
            else:
                kind_penalty = 32

            # Prefer similar placement within each halving cycle.
            phase_delta = abs(float(left["days_from_halving"]) - float(right["days_from_halving"]))
            move_delta = abs(float(left["prev_move_pct"]) - float(right["prev_move_pct"]))
            score = abs_delta + kind_penalty + phase_delta * 0.08 + min(move_delta, 80.0) * 0.05
            candidates.append(
                {
                    "from_cycle": cycle_a,
                    "to_cycle": cycle_b,
                    "from_index": int(left["anchor_index"]),
                    "to_index": int(right["anchor_index"]),
                    "from_date": left["date"].date().isoformat(),
                    "to_date": right["date"].date().isoformat(),
                    "target_date": target.date().isoformat(),
                    "span_days": span,
                    "date_delta_days": delta,
                    "abs_date_delta_days": abs_delta,
                    "from_kind": left["kind"],
                    "to_kind": right["kind"],
                    "from_group": left["kind_group"],
                    "to_group": right["kind_group"],
                    "from_days_from_halving": int(left["days_from_halving"]),
                    "to_days_from_halving": int(right["days_from_halving"]),
                    "phase_delta_days": int(right["days_from_halving"] - left["days_from_halving"]),
                    "score": round(score, 4),
                }
            )

    cand = pd.DataFrame(candidates)
    if cand.empty:
        return cand

    # Greedy one-to-one matching by score.
    cand = cand.sort_values(["score", "abs_date_delta_days"]).reset_index(drop=True)
    used_left = set()
    used_right = set()
    rows = []
    for _, row in cand.iterrows():
        left_key = int(row["from_index"])
        right_key = int(row["to_index"])
        if left_key in used_left or right_key in used_right:
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        rows.append(row.to_dict())
    return pd.DataFrame(rows).sort_values(["from_index"]).reset_index(drop=True)


def build_matches(df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        match_pair(df, "2017", "2021"),
        match_pair(df, "2021", "2025"),
    ]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def build_triplets(matches: pd.DataFrame) -> pd.DataFrame:
    left = matches[matches["from_cycle"] == "2017"].copy()
    right = matches[matches["from_cycle"] == "2021"].copy()
    rows = []
    for _, a in left.iterrows():
        b = right[right["from_index"] == int(a["to_index"])]
        if b.empty:
            continue
        b = b.iloc[0]
        rows.append(
            {
                "index_2017": int(a["from_index"]),
                "index_2021": int(a["to_index"]),
                "index_2025": int(b["to_index"]),
                "date_2017": a["from_date"],
                "date_2021": a["to_date"],
                "date_2025": b["to_date"],
                "kind_2017": a["from_kind"],
                "kind_2021": a["to_kind"],
                "kind_2025": b["to_kind"],
                "delta_2017_to_2021": int(a["date_delta_days"]),
                "delta_2021_to_2025": int(b["date_delta_days"]),
                "score_sum": round(float(a["score"]) + float(b["score"]), 4),
            }
        )
    return pd.DataFrame(rows)


def write_html(nodes: pd.DataFrame, matches: pd.DataFrame, triplets: pd.DataFrame) -> None:
    node_payload = []
    for cycle in ["2017", "2021", "2025"]:
        g = nodes[nodes["cycle"] == cycle].copy()
        node_payload.extend(g.to_dict(orient="records"))
    for item in node_payload:
        item["date"] = pd.Timestamp(item["date"]).date().isoformat()
    data_json = json.dumps(node_payload, ensure_ascii=False)
    match_json = json.dumps(matches.to_dict(orient="records"), ensure_ascii=False)
    triplet_json = json.dumps(triplets.to_dict(orient="records"), ensure_ascii=False)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Node Correspondence v6</title>
<style>
body {{ margin:0; background:#171b22; color:#e8edf5; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; gap:12px; align-items:center; padding:12px 16px; background:#1f2530; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ font-size:16px; margin:0; }}
main {{ display:grid; grid-template-columns:1fr 390px; gap:12px; padding:12px; }}
.panel {{ background:#202631; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 86px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 86px); overflow:auto; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aab6c7; }}
.strong {{ color:#fff; font-weight:700; }}
.green {{ color:#00e676; }}
.blue {{ color:#73d0ff; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button.active {{ background:#315489; border-color:#9bbfff; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>节点对应关系 v6：先按 4 年跨度连线</h1>
  <button id="togglePairs" class="active">显示相邻周期</button>
  <button id="toggleTriplets" class="active">显示三周期链</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel">
    <div id="summary"></div>
    <div id="list"></div>
  </aside>
</main>
<div id="tip" class="tip"></div>
<script>
const NODES = {data_json};
const MATCHES = {match_json};
const TRIPLETS = {triplet_json};
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const W = 1200, H = 720, M = {{l:72,r:34,t:28,b:46}};
let showPairs = true, showTriplets = true;
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function xScale(day) {{
  const maxDay = Math.max(...NODES.map(d => +d.days_from_halving));
  return M.l + day / maxDay * (W - M.l - M.r);
}}
function yCycle(cycle) {{
  const rows = {{ "2017": 150, "2021": 360, "2025": 570 }};
  return rows[cycle];
}}
function color(kind) {{
  if (kind === "drop_end" || kind === "rally_start") return "#00e676";
  if (kind === "rally_end" || kind === "drop_start") return "#73d0ff";
  return "#d8ff6a";
}}
function nodeKey(cycle, index) {{ return cycle + ":" + index; }}
function render() {{
  clear();
  const byKey = new Map(NODES.map(n => [nodeKey(n.cycle, n.anchor_index), n]));
  for (let i=0;i<10;i++) {{
    const x = M.l + i/9*(W-M.l-M.r);
    el("line", {{x1:x,y1:M.t,x2:x,y2:H-M.b,stroke:"rgba(255,255,255,.07)"}});
  }}
  ["2017","2021","2025"].forEach(c => {{
    const y = yCycle(c);
    el("line", {{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:"rgba(255,255,255,.16)","stroke-width":1}});
    const label = el("text", {{x:18,y:y+5,fill:"#fff","font-size":15}});
    label.textContent = c;
  }});
  if (showPairs) {{
    MATCHES.forEach(m => {{
      const a = byKey.get(nodeKey(m.from_cycle, m.from_index));
      const b = byKey.get(nodeKey(m.to_cycle, m.to_index));
      if (!a || !b) return;
      const x1 = xScale(+a.days_from_halving), y1 = yCycle(a.cycle);
      const x2 = xScale(+b.days_from_halving), y2 = yCycle(b.cycle);
      el("line", {{x1,y1,x2,y2,stroke:"rgba(0,230,118,.38)","stroke-width":1.1}});
    }});
  }}
  if (showTriplets) {{
    TRIPLETS.forEach(t => {{
      const a = byKey.get(nodeKey("2017", t.index_2017));
      const b = byKey.get(nodeKey("2021", t.index_2021));
      const c = byKey.get(nodeKey("2025", t.index_2025));
      if (!a || !b || !c) return;
      const pts = [a,b,c].map(n => xScale(+n.days_from_halving) + "," + yCycle(n.cycle)).join(" ");
      el("polyline", {{points:pts,fill:"none",stroke:"rgba(255,255,255,.72)","stroke-width":2.2}});
    }});
  }}
  NODES.forEach(n => {{
    const x = xScale(+n.days_from_halving), y = yCycle(n.cycle);
    const c = el("circle", {{cx:x,cy:y,r:6,fill:color(n.kind),stroke:"#06140c","stroke-width":1.3}});
    c.addEventListener("mousemove", ev => {{
      tip.style.display = "block";
      tip.style.left = ev.clientX + 12 + "px";
      tip.style.top = ev.clientY + 12 + "px";
      tip.innerHTML = `<b>${{n.cycle}} #${{n.anchor_index}}</b><br>${{n.date}} · ${{n.kind}}<br>halving+${{n.days_from_halving}}d<br>price ${{Math.round(n.price).toLocaleString()}}`;
    }});
    c.addEventListener("mouseleave", () => tip.style.display = "none");
    const txt = el("text", {{x:x+7,y:y-8,fill:"#fff","font-size":10}});
    txt.textContent = n.anchor_index;
  }});
  document.getElementById("summary").innerHTML = `
    <div class="strong">4 年跨度匹配</div>
    <div>2017→2021: ${{MATCHES.filter(m => m.from_cycle === "2017").length}} 条</div>
    <div>2021→2025: ${{MATCHES.filter(m => m.from_cycle === "2021").length}} 条</div>
    <div>三周期链: ${{TRIPLETS.length}} 条</div>
    <br>
  `;
  document.getElementById("list").innerHTML = TRIPLETS.map(t => `
    <div class="item">
      <div class="strong">#${{t.index_2017}} → #${{t.index_2021}} → #${{t.index_2025}}</div>
      <div>${{t.date_2017}} · ${{t.date_2021}} · ${{t.date_2025}}</div>
      <div>${{t.kind_2017}} / ${{t.kind_2021}} / ${{t.kind_2025}}</div>
      <div>偏差: <span class="green">${{t.delta_2017_to_2021}}d</span>, <span class="blue">${{t.delta_2021_to_2025}}d</span></div>
    </div>
  `).join("");
}}
document.getElementById("togglePairs").onclick = e => {{
  showPairs = !showPairs; e.target.classList.toggle("active", showPairs); render();
}};
document.getElementById("toggleTriplets").onclick = e => {{
  showTriplets = !showTriplets; e.target.classList.toggle("active", showTriplets); render();
}};
render();
</script>
</body>
</html>"""
    (OUT / "green_anchor_correspondence_v6.html").write_text(html, encoding="utf-8")


def main() -> None:
    nodes = load_nodes()
    matches = build_matches(nodes)
    triplets = build_triplets(matches)
    matches.to_csv(OUT / "node_matches_4y_v6.csv", index=False, encoding="utf-8-sig")
    triplets.to_csv(OUT / "node_triplets_4y_v6.csv", index=False, encoding="utf-8-sig")
    write_html(nodes, matches, triplets)


if __name__ == "__main__":
    main()
