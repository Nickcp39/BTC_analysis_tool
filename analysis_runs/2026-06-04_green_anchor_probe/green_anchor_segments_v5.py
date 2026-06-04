from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

HALVINGS = {
    "2017": pd.Timestamp("2016-07-09"),
    "2021": pd.Timestamp("2020-05-11"),
    "2025": pd.Timestamp("2024-04-20"),
}
NEXT_HALVINGS = {
    "2017": pd.Timestamp("2020-05-11"),
    "2021": pd.Timestamp("2024-04-20"),
    "2025": None,
}


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def point_line_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    line = end - start
    denom = np.linalg.norm(line)
    if denom == 0:
        return np.linalg.norm(points - start, axis=1)
    return np.abs(np.cross(line, start - points)) / denom


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
    if not vertices:
        return []
    vertices = sorted(set(vertices))
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
            continue
        # Keep the most extreme point in a tight group.
        vals = y[group]
        center = float(np.mean(vals))
        out.append(group[int(np.argmax(np.abs(vals - center)))])
    return sorted(out)


def segment_vertices(series: pd.Series, cycle: str) -> tuple[pd.Series, list[int]]:
    start = HALVINGS[cycle]
    end = NEXT_HALVINGS[cycle] or series.index.max()
    s = series.loc[start:end].copy()
    y = np.log(s).rolling(5, center=True, min_periods=2).mean().bfill().ffill()
    x_norm = np.linspace(0.0, 1.0, len(y))
    y_norm = (y.values - y.values.min()) / max(y.values.max() - y.values.min(), 1e-9)
    points = np.column_stack([x_norm, y_norm])

    # Later cycles are smoother, so use a slightly lower tolerance to keep local regime nodes.
    epsilon = {"2017": 0.018, "2021": 0.017, "2025": 0.014}[cycle]
    vertices = rdp_indices(points, epsilon)
    vertices = merge_close_vertices(vertices, y.values, min_gap=9)
    return s, vertices


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


def build_table(series: pd.Series) -> tuple[pd.DataFrame, dict]:
    rows = []
    payload = {}
    for cycle in ["2017", "2021", "2025"]:
        s, vertices = segment_vertices(series, cycle)
        y = np.log(s).rolling(5, center=True, min_periods=2).mean().bfill().ffill()
        payload[cycle] = {"series": s, "vertices": vertices}
        for pos, idx in enumerate(vertices):
            if idx == 0 or idx == len(s) - 1:
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
            move_power = max(abs(prev_move), abs(next_move))
            total_power = abs(prev_move) + abs(next_move)
            if move_power < 0.045 and angle_change < 0.0015:
                continue
            d = s.index[idx]
            rows.append(
                {
                    "cycle": cycle,
                    "date": d.date().isoformat(),
                    "days_from_halving": (d - HALVINGS[cycle]).days,
                    "price": round(float(s.iloc[idx]), 2),
                    "kind": classify(prev_slope, next_slope),
                    "prev_days": prev_days,
                    "next_days": next_days,
                    "prev_move_pct": round((np.exp(prev_move) - 1.0) * 100.0, 2),
                    "next_move_pct": round((np.exp(next_move) - 1.0) * 100.0, 2),
                    "prev_slope": round(prev_slope, 6),
                    "next_slope": round(next_slope, 6),
                    "score": round(float(angle_change * total_power * 10000.0), 4),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, payload
    table = table.sort_values(["cycle", "days_from_halving"]).reset_index(drop=True)
    table["anchor_index"] = table.groupby("cycle").cumcount() + 1
    table["interval_from_prev_days"] = table.groupby("cycle")["days_from_halving"].diff()
    return table, payload


def write_html(series: pd.Series, table: pd.DataFrame, payload: dict) -> None:
    data = []
    for cycle in ["2017", "2021", "2025"]:
        s = payload[cycle]["series"]
        anchors = table[table["cycle"] == cycle].to_dict(orient="records")
        data.append(
            {
                "cycle": cycle,
                "prices": [[d.date().isoformat(), round(float(v), 2)] for d, v in s.items()],
                "anchors": anchors,
            }
        )
    data_json = json.dumps(data, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Segment Nodes v5</title>
<style>
body {{ margin:0; background:#171b22; color:#e8edf5; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; gap:10px; align-items:center; padding:12px 16px; background:#1f2530; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ font-size:16px; margin:0 14px 0 0; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button.active {{ background:#315489; border-color:#9bbfff; }}
main {{ display:grid; grid-template-columns:1fr 360px; gap:12px; padding:12px; }}
.panel {{ background:#202631; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 86px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 86px); overflow:auto; }}
.item {{ display:grid; grid-template-columns:42px 1fr; gap:8px; padding:7px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aab6c7; }}
.idx {{ color:#00e676; font-weight:700; }}
.date {{ color:#fff; font-weight:700; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>走势段节点 v5：同一看法，多种形态</h1>
  <button data-cycle="2017">2017</button>
  <button data-cycle="2021">2021</button>
  <button class="active" data-cycle="2025">2025</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const DATA = {data_json};
let active = "2025";
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const W = 1200, H = 720, M = {{l:76,r:28,t:28,b:46}};
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
  const d = DATA.find(x => x.cycle === active);
  const prices = d.prices.map(p => [new Date(p[0]).getTime(), +p[1], p[0]]);
  const xs = prices.map(p => p[0]), ys = prices.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  for (let i=0;i<8;i++) {{
    const x = M.l + i/7*(W-M.l-M.r);
    el("line", {{x1:x,y1:M.t,x2:x,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  }}
  for (let i=0;i<7;i++) {{
    const y = M.t + i/6*(H-M.t-M.b);
    el("line", {{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:"rgba(255,255,255,.08)"}});
  }}
  let path = "";
  prices.forEach((p,i) => {{
    const x = xScale(p[0], xmin, xmax), y = yScale(p[1], ymin, ymax);
    path += (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
  }});
  el("path", {{d:path, fill:"none", stroke:"#ff8177", "stroke-width":2}});
  const anchors = d.anchors;
  let seg = "";
  anchors.forEach((a,i) => {{
    const x = xScale(new Date(a.date).getTime(), xmin, xmax);
    const y = yScale(+a.price, ymin, ymax);
    seg += (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
  }});
  el("path", {{d:seg, fill:"none", stroke:"rgba(0,230,118,.55)", "stroke-width":1.4, "stroke-dasharray":"5 5"}});
  for (let i=1;i<anchors.length;i++) {{
    const a = anchors[i-1], b = anchors[i];
    const x1 = xScale(new Date(a.date).getTime(), xmin, xmax);
    const x2 = xScale(new Date(b.date).getTime(), xmin, xmax);
    const y = H - 18;
    el("line", {{x1,y1:y,x2,y2:y,stroke:"#00e676","stroke-width":1.1}});
    const tx = el("text", {{x:(x1+x2)/2,y:y-5,fill:"#dfffea","font-size":11,"text-anchor":"middle"}});
    tx.textContent = Math.round(b.interval_from_prev_days) + "d";
  }}
  anchors.forEach(a => {{
    const x = xScale(new Date(a.date).getTime(), xmin, xmax);
    const y = yScale(+a.price, ymin, ymax);
    const c = el("circle", {{cx:x,cy:y,r:6,fill:color(a.kind),stroke:"#06140c","stroke-width":1.5}});
    c.addEventListener("mousemove", ev => {{
      tip.style.display = "block";
      tip.style.left = ev.clientX + 12 + "px";
      tip.style.top = ev.clientY + 12 + "px";
      tip.innerHTML = `<b>#${{a.anchor_index}} ${{a.date}}</b><br>${{a.kind}} · ${{money(a.price)}}<br>前段 ${{a.prev_move_pct}}% / ${{a.prev_days}}d<br>后段 ${{a.next_move_pct}}% / ${{a.next_days}}d<br>间隔 ${{a.interval_from_prev_days || "-"}} 天`;
    }});
    c.addEventListener("mouseleave", () => tip.style.display = "none");
    const t = el("text", {{x:x+8,y:y-8,fill:"#fff","font-size":12}});
    t.textContent = a.anchor_index;
  }});
  document.getElementById("summary").innerHTML = `<b>${{active}}</b>：${{anchors.length}} 个走势段节点`;
  document.getElementById("list").innerHTML = anchors.map(a => `
    <div class="item"><div class="idx">#${{a.anchor_index}}</div><div>
      <div class="date">${{a.date}} · ${{money(a.price)}}</div>
      <div>${{a.kind}} · 前段 ${{a.prev_move_pct}}%/${{a.prev_days}}d · 后段 ${{a.next_move_pct}}%/${{a.next_days}}d · 间隔 ${{a.interval_from_prev_days || "-"}}d</div>
    </div></div>`).join("");
}}
document.querySelectorAll("button").forEach(b => b.onclick = () => {{
  active = b.dataset.cycle;
  document.querySelectorAll("button").forEach(x => x.classList.toggle("active", x === b));
  render();
}});
render();
</script>
</body>
</html>"""
    (OUT / "green_anchor_segments_v5.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    table, payload = build_table(series)
    table.to_csv(OUT / "segment_nodes_v5.csv", index=False, encoding="utf-8-sig")
    write_html(series, table, payload)


if __name__ == "__main__":
    main()
