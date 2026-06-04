from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

FOUR_YEAR_DAYS = 1461
MATCH_TOLERANCE_DAYS = 80


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def point_line_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    line = end - start
    denom = np.linalg.norm(line)
    if denom == 0:
        return np.linalg.norm(points - start, axis=1)
    x0, y0 = points[:, 0], points[:, 1]
    x1, y1 = start
    x2, y2 = end
    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denom


def rdp_indices(points: np.ndarray, epsilon: float, start: int = 0, end: int | None = None) -> list[int]:
    if end is None:
        end = len(points) - 1
    if end <= start + 1:
        return [start, end]
    segment = points[start : end + 1]
    dist = point_line_distance(segment, points[start], points[end])
    rel_idx = int(np.argmax(dist))
    max_dist = float(dist[rel_idx])
    idx = start + rel_idx
    if max_dist <= epsilon:
        return [start, end]
    left = rdp_indices(points, epsilon, start, idx)
    right = rdp_indices(points, epsilon, idx, end)
    return left[:-1] + right


def merge_close_vertices(vertices: list[int], y: np.ndarray, min_gap: int = 10) -> list[int]:
    vertices = sorted(set(vertices))
    if not vertices:
        return []
    groups: list[list[int]] = [[vertices[0]]]
    for idx in vertices[1:]:
        if idx - groups[-1][-1] < min_gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    out = []
    for group in groups:
        if len(group) == 1:
            out.append(group[0])
        else:
            vals = y[group]
            center = float(np.mean(vals))
            out.append(group[int(np.argmax(np.abs(vals - center)))])
    return sorted(out)


def classify(prev_slope: float, next_slope: float) -> str:
    flat = 0.0012

    def state(slope: float) -> str:
        if slope > flat:
            return "up"
        if slope < -flat:
            return "down"
        return "flat"

    a, b = state(prev_slope), state(next_slope)
    if a == "down" and b in {"flat", "up"}:
        return "drop_end"
    if a == "flat" and b == "up":
        return "rally_start"
    if a == "up" and b in {"flat", "down"}:
        return "rally_end"
    if a == "flat" and b == "down":
        return "drop_start"
    return f"{a}_to_{b}"


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


def build_nodes(series: pd.Series) -> pd.DataFrame:
    y = np.log(series).rolling(5, center=True, min_periods=2).mean().bfill().ffill()
    x_norm = np.linspace(0.0, 1.0, len(y))
    y_norm = (y.values - y.values.min()) / max(y.values.max() - y.values.min(), 1e-9)
    points = np.column_stack([x_norm, y_norm])
    vertices = merge_close_vertices(rdp_indices(points, epsilon=0.012), y.values, min_gap=9)

    rows = []
    for pos, idx in enumerate(vertices):
        if idx == 0 or idx == len(series) - 1:
            continue
        prev_idx = vertices[pos - 1]
        next_idx = vertices[pos + 1]
        prev_days = max(idx - prev_idx, 1)
        next_days = max(next_idx - idx, 1)
        prev_move = float(y.iloc[idx] - y.iloc[prev_idx])
        next_move = float(y.iloc[next_idx] - y.iloc[idx])
        prev_slope = prev_move / prev_days
        next_slope = next_move / next_days
        angle_change = abs(next_slope - prev_slope)
        total_power = abs(prev_move) + abs(next_move)
        if max(abs(prev_move), abs(next_move)) < 0.035 and angle_change < 0.0013:
            continue
        d = series.index[idx]
        rows.append(
            {
                "node_index": len(rows) + 1,
                "date": d.date().isoformat(),
                "days_from_start": int((d - series.index.min()).days),
                "price": round(float(series.iloc[idx]), 2),
                "kind": classify(prev_slope, next_slope),
                "kind_group": kind_group(classify(prev_slope, next_slope)),
                "prev_days": prev_days,
                "next_days": next_days,
                "prev_move_pct": round((np.exp(prev_move) - 1.0) * 100.0, 2),
                "next_move_pct": round((np.exp(next_move) - 1.0) * 100.0, 2),
                "score": round(float(angle_change * total_power * 10000.0), 4),
            }
        )
    return pd.DataFrame(rows)


def build_matches(nodes: pd.DataFrame) -> pd.DataFrame:
    nodes_dt = nodes.copy()
    nodes_dt["date_ts"] = pd.to_datetime(nodes_dt["date"])
    rows = []
    for _, a in nodes_dt.iterrows():
        target = a["date_ts"] + pd.Timedelta(days=FOUR_YEAR_DAYS)
        future = nodes_dt[nodes_dt["date_ts"] > a["date_ts"]].copy()
        future["delta"] = (future["date_ts"] - target).dt.days
        future["abs_delta"] = future["delta"].abs()
        future = future[future["abs_delta"] <= MATCH_TOLERANCE_DAYS]
        if future.empty:
            continue
        future["kind_penalty"] = np.where(
            future["kind"].eq(a["kind"]),
            0,
            np.where(future["kind_group"].eq(a["kind_group"]), 12, 34),
        )
        future["score_match"] = future["abs_delta"] + future["kind_penalty"] + np.minimum(
            np.abs(future["score"].astype(float) - float(a["score"])) * 0.03,
            20,
        )
        b = future.sort_values(["score_match", "abs_delta"]).iloc[0]
        rows.append(
            {
                "from_index": int(a["node_index"]),
                "to_index": int(b["node_index"]),
                "from_date": a["date"],
                "to_date": b["date"],
                "target_date": target.date().isoformat(),
                "span_days": FOUR_YEAR_DAYS,
                "delta_days": int(b["delta"]),
                "abs_delta_days": int(b["abs_delta"]),
                "from_kind": a["kind"],
                "to_kind": b["kind"],
                "from_group": a["kind_group"],
                "to_group": b["kind_group"],
                "match_score": round(float(b["score_match"]), 4),
            }
        )
    return pd.DataFrame(rows)


def build_html(series: pd.Series, nodes: pd.DataFrame, matches: pd.DataFrame) -> None:
    prices = [[d.date().isoformat(), round(float(v), 2)] for d, v in series.items()]
    node_payload = nodes.to_dict(orient="records")
    match_payload = matches.to_dict(orient="records")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Full History Nodes v7</title>
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
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button.active {{ background:#315489; border-color:#9bbfff; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>BTC 全历史走势节点 v7：2014-12 至 2026-05，按 4 年跨度连线</h1>
  <button id="toggleNodes" class="active">节点</button>
  <button id="toggleLinks" class="active">4年连线</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const PRICES = {json.dumps(prices, ensure_ascii=False)};
const NODES = {json.dumps(node_payload, ensure_ascii=False)};
const MATCHES = {json.dumps(match_payload, ensure_ascii=False)};
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const W = 1200, H = 720, M = {{l:76,r:28,t:28,b:46}};
let showNodes = true, showLinks = true;
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function xScale(t, min, max) {{ return M.l + (t - min) / (max - min) * (W - M.l - M.r); }}
function yScale(v, min, max) {{
  const a = Math.log(v), lo = Math.log(min), hi = Math.log(max);
  return H - M.b - (a - lo) / (hi - lo) * (H - M.t - M.b);
}}
function money(v) {{ return "$" + Math.round(v).toLocaleString(); }}
function color(kind) {{
  if (kind === "drop_end" || kind === "rally_start") return "#00e676";
  if (kind === "rally_end" || kind === "drop_start") return "#73d0ff";
  return "#d8ff6a";
}}
function render() {{
  clear();
  const prices = PRICES.map(p => [new Date(p[0]).getTime(), +p[1], p[0]]);
  const xs = prices.map(p => p[0]), ys = prices.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  for (let i=0;i<10;i++) {{
    const x = M.l + i/9*(W-M.l-M.r);
    el("line", {{x1:x,y1:M.t,x2:x,y2:H-M.b,stroke:"rgba(255,255,255,.07)"}});
  }}
  for (let i=0;i<7;i++) {{
    const y = M.t + i/6*(H-M.t-M.b);
    el("line", {{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:"rgba(255,255,255,.07)"}});
  }}
  let path = "";
  prices.forEach((p,i) => {{
    const x = xScale(p[0], xmin, xmax), y = yScale(p[1], ymin, ymax);
    path += (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
  }});
  el("path", {{d:path, fill:"none", stroke:"#ff8177", "stroke-width":1.8}});
  const byIndex = new Map(NODES.map(n => [n.node_index, n]));
  if (showLinks) {{
    MATCHES.forEach(m => {{
      const a = byIndex.get(m.from_index), b = byIndex.get(m.to_index);
      if (!a || !b) return;
      const x1 = xScale(new Date(a.date).getTime(), xmin, xmax);
      const y1 = yScale(a.price, ymin, ymax);
      const x2 = xScale(new Date(b.date).getTime(), xmin, xmax);
      const y2 = yScale(b.price, ymin, ymax);
      el("line", {{x1,y1,x2,y2,stroke:"rgba(0,230,118,.36)","stroke-width":1.2}});
    }});
  }}
  if (showNodes) {{
    NODES.forEach(n => {{
      const x = xScale(new Date(n.date).getTime(), xmin, xmax);
      const y = yScale(+n.price, ymin, ymax);
      const c = el("circle", {{cx:x,cy:y,r:5.2,fill:color(n.kind),stroke:"#06140c","stroke-width":1.2}});
      c.addEventListener("mousemove", ev => {{
        tip.style.display = "block";
        tip.style.left = ev.clientX + 12 + "px";
        tip.style.top = ev.clientY + 12 + "px";
        tip.innerHTML = `<b>#${{n.node_index}} ${{n.date}}</b><br>${{n.kind}} · ${{money(n.price)}}<br>前段 ${{n.prev_move_pct}}% / ${{n.prev_days}}d<br>后段 ${{n.next_move_pct}}% / ${{n.next_days}}d`;
      }});
      c.addEventListener("mouseleave", () => tip.style.display = "none");
    }});
  }}
  document.getElementById("summary").innerHTML = `<div class="strong">全历史节点：${{NODES.length}}</div><div>4年连线：${{MATCHES.length}}</div><br>`;
  document.getElementById("list").innerHTML = MATCHES.slice(0, 120).map(m => `
    <div class="item">
      <div class="strong">#${{m.from_index}} → #${{m.to_index}} · 偏差 ${{m.delta_days}}d</div>
      <div>${{m.from_date}} → ${{m.to_date}} · 目标 ${{m.target_date}}</div>
      <div>${{m.from_kind}} → ${{m.to_kind}}</div>
    </div>`).join("");
}}
document.getElementById("toggleNodes").onclick = e => {{ showNodes = !showNodes; e.target.classList.toggle("active", showNodes); render(); }};
document.getElementById("toggleLinks").onclick = e => {{ showLinks = !showLinks; e.target.classList.toggle("active", showLinks); render(); }};
render();
</script>
</body>
</html>"""
    (OUT / "green_anchor_full_history_v7.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    nodes = build_nodes(series)
    matches = build_matches(nodes)
    nodes.to_csv(OUT / "full_history_nodes_v7.csv", index=False, encoding="utf-8-sig")
    matches.to_csv(OUT / "full_history_matches_4y_v7.csv", index=False, encoding="utf-8-sig")
    build_html(series, nodes, matches)
    print(f"Data: {series.index.min().date()} ~ {series.index.max().date()} ({len(series)} days)")
    print(f"Nodes: {len(nodes)}")
    print(f"4y matches: {len(matches)}")


if __name__ == "__main__":
    main()
