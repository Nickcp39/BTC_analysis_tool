from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def write_html(series: pd.Series) -> None:
    latest = series.index.max()
    start = latest - pd.Timedelta(days=365 * 5)
    five_year = series.loc[start:latest].copy()
    payload = [[d.date().isoformat(), round(float(v), 2)] for d, v in five_year.items()]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Visible ZigZag v8</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{
  display:flex; align-items:center; gap:16px; padding:12px 18px;
  background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1);
}}
h1 {{ margin:0; font-size:16px; }}
.control {{ display:flex; align-items:center; gap:8px; color:#aeb8c8; font-size:13px; }}
input[type=range] {{ width:180px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 88px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 88px); overflow:auto; }}
.item {{ display:grid; grid-template-columns:42px 1fr; gap:8px; padding:7px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.idx {{ color:#00e676; font-weight:700; }}
.date {{ color:#fff; font-weight:700; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button.active {{ background:#315489; border-color:#9bbfff; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>BTC 5年可视尺度 ZigZag 节点 v8</h1>
  <div class="control">
    <span>反转阈值</span>
    <input id="threshold" type="range" min="5" max="18" value="10" step="1" />
    <b id="thresholdText">10%</b>
  </div>
  <button id="toggleLine" class="active">折线</button>
  <button id="toggleIntervals" class="active">间隔</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const RAW = {json.dumps(payload, ensure_ascii=False)};
const DATA = RAW.map(d => ({{ date:new Date(d[0]), dateText:d[0], price:+d[1] }}));
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const W = 1200, H = 720, M = {{l:76,r:28,t:30,b:48}};
let showLine = true, showIntervals = true;

function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function money(v) {{ return "$" + Math.round(v).toLocaleString(); }}
function pct(a, b) {{ return (b / a - 1) * 100; }}

function zigzag(data, thresholdPct) {{
  const threshold = thresholdPct / 100;
  if (data.length < 3) return [];
  let pivots = [];
  let trend = 0; // 1 up, -1 down, 0 unknown
  let extremeIdx = 0;
  let baseIdx = 0;

  for (let i = 1; i < data.length; i++) {{
    const change = data[i].price / data[baseIdx].price - 1;
    if (trend === 0) {{
      if (change >= threshold) {{
        trend = 1;
        pivots.push({{...data[baseIdx], kind:"low"}});
        extremeIdx = i;
      }} else if (change <= -threshold) {{
        trend = -1;
        pivots.push({{...data[baseIdx], kind:"high"}});
        extremeIdx = i;
      }} else {{
        if (data[i].price < data[baseIdx].price) baseIdx = i;
        if (data[i].price > data[baseIdx].price) baseIdx = i;
      }}
      continue;
    }}

    if (trend === 1) {{
      if (data[i].price >= data[extremeIdx].price) extremeIdx = i;
      const reverse = data[i].price / data[extremeIdx].price - 1;
      if (reverse <= -threshold) {{
        pivots.push({{...data[extremeIdx], kind:"high"}});
        trend = -1;
        extremeIdx = i;
      }}
    }} else {{
      if (data[i].price <= data[extremeIdx].price) extremeIdx = i;
      const reverse = data[i].price / data[extremeIdx].price - 1;
      if (reverse >= threshold) {{
        pivots.push({{...data[extremeIdx], kind:"low"}});
        trend = 1;
        extremeIdx = i;
      }}
    }}
  }}
  if (pivots.length === 0 || pivots[pivots.length - 1].dateText !== data[extremeIdx].dateText) {{
    pivots.push({{...data[extremeIdx], kind: trend === 1 ? "high" : "low"}});
  }}
  return pivots.map((p, i) => {{
    const prev = i ? pivots[i - 1] : null;
    const next = i < pivots.length - 1 ? pivots[i + 1] : null;
    return {{
      ...p,
      index: i + 1,
      interval: prev ? Math.round((p.date - prev.date) / 86400000) : null,
      prevMove: prev ? pct(prev.price, p.price) : null,
      nextMove: next ? pct(p.price, next.price) : null,
    }};
  }});
}}

function render() {{
  clear();
  const threshold = +document.getElementById("threshold").value;
  document.getElementById("thresholdText").textContent = threshold + "%";
  const pivots = zigzag(DATA, threshold);
  const xs = DATA.map(d => d.date.getTime()), ys = DATA.map(d => d.price);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const x = d => M.l + (d.date.getTime() - xmin) / (xmax - xmin) * (W - M.l - M.r);
  const y = d => {{
    const lo = Math.log(ymin), hi = Math.log(ymax), v = Math.log(d.price);
    return H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b);
  }};

  for (let i=0;i<8;i++) {{
    const gx = M.l + i/7*(W-M.l-M.r);
    el("line", {{x1:gx,y1:M.t,x2:gx,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  }}
  for (let i=0;i<6;i++) {{
    const gy = M.t + i/5*(H-M.t-M.b);
    el("line", {{x1:M.l,y1:gy,x2:W-M.r,y2:gy,stroke:"rgba(255,255,255,.08)"}});
  }}

  let path = "";
  DATA.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
  el("path", {{d:path, fill:"none", stroke:"#ff8177", "stroke-width":2}});

  if (showLine && pivots.length > 1) {{
    let zz = "";
    pivots.forEach((d,i) => zz += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
    el("path", {{d:zz, fill:"none", stroke:"rgba(0,230,118,.7)", "stroke-width":1.8, "stroke-dasharray":"6 5"}});
  }}

  if (showIntervals) {{
    for (let i=1;i<pivots.length;i++) {{
      const a = pivots[i - 1], b = pivots[i];
      const x1 = x(a), x2 = x(b), yy = H - 18;
      el("line", {{x1,y1:yy,x2,y2:yy,stroke:"#00e676","stroke-width":1.2}});
      const t = el("text", {{x:(x1+x2)/2,y:yy-5,fill:"#dfffea","font-size":11,"text-anchor":"middle"}});
      t.textContent = b.interval + "d";
    }}
  }}

  pivots.forEach(p => {{
    const fill = p.kind === "low" ? "#00e676" : "#73d0ff";
    const c = el("circle", {{cx:x(p),cy:y(p),r:6,fill,stroke:"#06140c","stroke-width":1.5}});
    c.addEventListener("mousemove", ev => {{
      tip.style.display = "block";
      tip.style.left = ev.clientX + 12 + "px";
      tip.style.top = ev.clientY + 12 + "px";
      tip.innerHTML = `<b>#${{p.index}} ${{p.dateText}}</b><br>${{p.kind}} · ${{money(p.price)}}<br>间隔 ${{p.interval || "-"}}d<br>前段 ${{p.prevMove === null ? "-" : p.prevMove.toFixed(1) + "%"}}<br>后段 ${{p.nextMove === null ? "-" : p.nextMove.toFixed(1) + "%"}}`;
    }});
    c.addEventListener("mouseleave", () => tip.style.display = "none");
    const t = el("text", {{x:x(p)+8,y:y(p)-8,fill:"#fff","font-size":12}});
    t.textContent = p.index;
  }});

  document.getElementById("summary").innerHTML = `<b>最近 5 年</b><br>数据：${{DATA[0].dateText}} ~ ${{DATA[DATA.length-1].dateText}}<br>节点：${{pivots.length}} 个<br>阈值：${{threshold}}%<br><br>`;
  document.getElementById("list").innerHTML = pivots.map(p => `
    <div class="item">
      <div class="idx">#${{p.index}}</div>
      <div>
        <div class="date">${{p.dateText}} · ${{money(p.price)}}</div>
        <div>${{p.kind}} · 间隔 ${{p.interval || "-"}}d · 前段 ${{p.prevMove === null ? "-" : p.prevMove.toFixed(1)+"%"}} · 后段 ${{p.nextMove === null ? "-" : p.nextMove.toFixed(1)+"%"}}</div>
      </div>
    </div>
  `).join("");
}}

document.getElementById("threshold").addEventListener("input", render);
document.getElementById("toggleLine").onclick = e => {{ showLine = !showLine; e.target.classList.toggle("active", showLine); render(); }};
document.getElementById("toggleIntervals").onclick = e => {{ showIntervals = !showIntervals; e.target.classList.toggle("active", showIntervals); render(); }};
render();
</script>
</body>
</html>"""
    (OUT / "green_anchor_visible_zigzag_v8.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    write_html(series)
    latest = series.index.max()
    start = latest - pd.Timedelta(days=365 * 5)
    print(f"Wrote {OUT / 'green_anchor_visible_zigzag_v8.html'}")
    print(f"Window: {start.date()} ~ {latest.date()}")


if __name__ == "__main__":
    main()
