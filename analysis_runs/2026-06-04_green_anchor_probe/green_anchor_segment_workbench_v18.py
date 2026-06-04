from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent


PRESETS = [
    {
        "name": "Peak +/- 6m: 2021 top vs 2025 top",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-10-05",
        "pre_days": 183,
        "post_days": 183,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": -59,
        "tag": "peak_half_year",
    },
    {
        "name": "Peak after 1y: 2021 top vs 2025 top",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-10-05",
        "pre_days": 60,
        "post_days": 365,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": -59,
        "tag": "peak_post_year",
    },
    {
        "name": "Bottom +/- 6m: 2022 low vs 2026 low",
        "left_anchor": "2022-11-21",
        "right_anchor": "2026-02-05",
        "pre_days": 183,
        "post_days": 183,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
        "tag": "bottom_half_year",
    },
    {
        "name": "Lower high +/- 4m",
        "left_anchor": "2022-03-29",
        "right_anchor": "2025-11-10",
        "pre_days": 120,
        "post_days": 120,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
        "tag": "lower_high",
    },
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def write_html(series: pd.Series) -> None:
    data = [[d.date().isoformat(), round(float(v), 2)] for d, v in series.items()]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Segment Alignment Workbench v18</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; align-items:center; gap:10px; padding:12px 16px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; white-space:nowrap; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 10px; cursor:pointer; }}
button:hover {{ background:#354154; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 500px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 86px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 86px); overflow:auto; }}
.grid {{ display:grid; grid-template-columns:118px 1fr; gap:8px; align-items:center; font-size:12px; color:#b5c0d0; }}
input, select, textarea {{ width:100%; color:#eaf1fb; background:#1b2029; border:1px solid rgba(255,255,255,.14); border-radius:6px; padding:7px; }}
input[type=range] {{ padding:0; }}
textarea {{ min-height:105px; resize:vertical; font-family:Consolas,monospace; font-size:12px; }}
.section {{ padding:10px 0; border-bottom:1px solid rgba(255,255,255,.08); }}
.row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.red {{ color:#ff6a5f; }}
.green {{ color:#00e676; }}
.muted {{ color:#aeb8c8; font-size:12px; line-height:1.45; }}
.pill {{ display:inline-block; padding:2px 6px; border-radius:999px; background:#354154; color:#dce7f7; font-size:11px; margin-right:4px; }}
</style>
</head>
<body>
<header>
  <h1>BTC 片段对比采样器 v18</h1>
  <button id="saveSnapshot">保存当前片段样本</button>
  <button id="downloadLibrary">下载样本库JSON</button>
  <button id="clearSnapshots">清空样本库</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel">
    <div class="section">
      <div class="grid">
        <label>预设</label><select id="preset"></select>
        <label>样本名称</label><input id="name" />
        <label>标签</label><input id="tag" />
        <label>左中心日期</label><input id="leftAnchor" type="date" />
        <label>右中心日期</label><input id="rightAnchor" type="date" />
        <label>前窗口天数</label><input id="preDays" type="number" min="1" max="1500" />
        <label>后窗口天数</label><input id="postDays" type="number" min="1" max="1500" />
      </div>
    </div>
    <div class="section">
      <div class="grid">
        <label>高度系数</label><div><input id="amp" type="range" min="0.10" max="1.50" step="0.01" /><span id="ampText"></span></div>
        <label>时间 scale</label><div><input id="timeScale" type="range" min="0.50" max="1.50" step="0.01" /><span id="timeText"></span></div>
        <label>水平平移</label><div><input id="shift" type="range" min="-500" max="500" step="1" /><span id="shiftText"></span></div>
      </div>
    </div>
    <div class="section">
      <div class="muted" id="summary"></div>
    </div>
    <div class="section">
      <div class="strong">当前样本 JSON</div>
      <textarea id="jsonBox"></textarea>
      <div class="row" style="margin-top:8px">
        <button id="applyJson">应用JSON</button>
        <button id="copyJson">复制JSON</button>
      </div>
    </div>
    <div class="section">
      <div class="strong">已保存样本</div>
      <div id="savedList"></div>
    </div>
  </aside>
</main>
<script>
const RAW = {json.dumps(data, ensure_ascii=False)}.map(d => ({{dateText:d[0], date:new Date(d[0]), price:+d[1]}}));
const PRESETS = {json.dumps(PRESETS, ensure_ascii=False)};
const STORE = "btc_segment_alignment_samples_v18";
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
let saved = loadSaved();

function loadSaved() {{
  try {{ return JSON.parse(localStorage.getItem(STORE) || "[]"); }} catch(e) {{ return []; }}
}}
function persist() {{ localStorage.setItem(STORE, JSON.stringify(saved, null, 2)); }}
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clearSvg() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function nearest(dateText) {{
  const t = new Date(dateText).getTime();
  let best = RAW[0], bd = Infinity;
  for (const r of RAW) {{
    const d = Math.abs(r.date.getTime() - t);
    if (d < bd) {{ best = r; bd = d; }}
  }}
  return best;
}}
function currentSample() {{
  return {{
    id: "sample_" + new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0,14),
    name: document.getElementById("name").value,
    tag: document.getElementById("tag").value,
    left_anchor: document.getElementById("leftAnchor").value,
    right_anchor: document.getElementById("rightAnchor").value,
    pre_days: +document.getElementById("preDays").value,
    post_days: +document.getElementById("postDays").value,
    amp_scale: +document.getElementById("amp").value,
    time_scale: +document.getElementById("timeScale").value,
    shift_days: +document.getElementById("shift").value,
    transform: "left_rel_plot = left_rel_day * time_scale + shift_days; left_log_norm_plot = left_log_norm * amp_scale",
    saved_at: new Date().toISOString()
  }};
}}
function setControls(s) {{
  document.getElementById("name").value = s.name;
  document.getElementById("tag").value = s.tag || "";
  document.getElementById("leftAnchor").value = s.left_anchor;
  document.getElementById("rightAnchor").value = s.right_anchor;
  document.getElementById("preDays").value = s.pre_days;
  document.getElementById("postDays").value = s.post_days;
  document.getElementById("amp").value = s.amp_scale;
  document.getElementById("timeScale").value = s.time_scale;
  document.getElementById("shift").value = s.shift_days;
  render();
}}
function rowsFor(s) {{
  const la = nearest(s.left_anchor), ra = nearest(s.right_anchor);
  const leftStart = new Date(la.date.getTime() - s.pre_days * 86400000);
  const leftEnd = new Date(la.date.getTime() + s.post_days * 86400000);
  const rightStart = new Date(ra.date.getTime() - s.pre_days * 86400000);
  const rightEnd = new Date(ra.date.getTime() + s.post_days * 86400000);
  const left = RAW.filter(r => r.date >= leftStart && r.date <= leftEnd).map(r => {{
    const rel = (r.date - la.date) / 86400000;
    return {{cycle:"left", dateText:r.dateText, rel_day:rel, rel_plot:rel*s.time_scale+s.shift_days, price:r.price, log_plot:Math.log(r.price/la.price)*s.amp_scale}};
  }});
  const right = RAW.filter(r => r.date >= rightStart && r.date <= rightEnd).map(r => {{
    const rel = (r.date - ra.date) / 86400000;
    return {{cycle:"right", dateText:r.dateText, rel_day:rel, rel_plot:rel, price:r.price, log_plot:Math.log(r.price/ra.price)}};
  }});
  return {{left,right,la,ra}};
}}
function render() {{
  clearSvg();
  const s = currentSample();
  document.getElementById("ampText").textContent = s.amp_scale.toFixed(2);
  document.getElementById("timeText").textContent = s.time_scale.toFixed(2);
  document.getElementById("shiftText").textContent = s.shift_days + "d";
  const {{left,right,la,ra}} = rowsFor(s);
  const all = [...left, ...right];
  const xmin = Math.min(...all.map(d=>d.rel_plot)), xmax = Math.max(...all.map(d=>d.rel_plot));
  const ymin = Math.min(...all.map(d=>d.log_plot)), ymax = Math.max(...all.map(d=>d.log_plot));
  function x(d) {{ return M.l + (d.rel_plot-xmin)/(xmax-xmin)*(W-M.l-M.r); }}
  function y(d) {{ return H-M.b - (d.log_plot-ymin)/(ymax-ymin)*(H-M.t-M.b); }}
  for (let i=0;i<9;i++) el("line", {{x1:M.l+i/8*(W-M.l-M.r),y1:M.t,x2:M.l+i/8*(W-M.l-M.r),y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  for (let i=0;i<7;i++) el("line", {{x1:M.l,y1:M.t+i/6*(H-M.t-M.b),x2:W-M.r,y2:M.t+i/6*(H-M.t-M.b),stroke:"rgba(255,255,255,.08)"}});
  [{{rows:left,color:"#ff6a5f",w:2.1}},{{rows:right,color:"#00e676",w:2.4}}].forEach(line => {{
    let path = "";
    line.rows.sort((a,b)=>a.rel_plot-b.rel_plot).forEach((d,i)=> path += (i?"L":"M")+x(d).toFixed(2)+","+y(d).toFixed(2));
    el("path", {{d:path,fill:"none",stroke:line.color,"stroke-width":line.w,opacity:.9}});
  }});
  document.getElementById("summary").innerHTML = `
    <div><span class="red">红</span>：左片段 ${{s.left_anchor}}，经过手动高度/时间/平移</div>
    <div><span class="green">绿</span>：右片段 ${{s.right_anchor}}，原样显示</div>
    <div>左锚价：${{Math.round(la.price).toLocaleString()}} · 右锚价：${{Math.round(ra.price).toLocaleString()}}</div>
    <div>窗口：前 ${{s.pre_days}} 天 / 后 ${{s.post_days}} 天</div>
  `;
  document.getElementById("jsonBox").value = JSON.stringify(s, null, 2);
  renderSaved();
}}
function renderSaved() {{
  document.getElementById("savedList").innerHTML = saved.map((s,i)=>`
    <div class="item">
      <div class="strong">${{i+1}}. ${{s.name}}</div>
      <div><span class="pill">${{s.tag || "untagged"}}</span>${{s.left_anchor}} → ${{s.right_anchor}}</div>
      <div>amp ${{s.amp_scale}} · time ${{s.time_scale}} · shift ${{s.shift_days}}d · window ${{s.pre_days}}/${{s.post_days}}</div>
      <div class="row" style="margin-top:6px"><button onclick="loadSavedIndex(${{i}})">载入</button><button onclick="deleteSavedIndex(${{i}})">删除</button></div>
    </div>
  `).join("");
}}
function download(name, text) {{
  const blob = new Blob([text], {{type:"application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}}
function loadSavedIndex(i) {{ setControls(saved[i]); }}
function deleteSavedIndex(i) {{ saved.splice(i,1); persist(); renderSaved(); }}

document.getElementById("preset").innerHTML = PRESETS.map((p,i)=>`<option value="${{i}}">${{i+1}}. ${{p.name}}</option>`).join("");
document.getElementById("preset").addEventListener("change", e => setControls(PRESETS[+e.target.value]));
["name","tag","leftAnchor","rightAnchor","preDays","postDays","amp","timeScale","shift"].forEach(id => document.getElementById(id).addEventListener("input", render));
document.getElementById("saveSnapshot").onclick = () => {{ saved.push(currentSample()); persist(); renderSaved(); alert("已保存当前片段样本。"); }};
document.getElementById("downloadLibrary").onclick = () => download("btc_segment_alignment_samples_v18.json", JSON.stringify({{version:"v18", samples:saved}}, null, 2));
document.getElementById("clearSnapshots").onclick = () => {{ if(confirm("清空所有已保存样本？")) {{ saved=[]; persist(); renderSaved(); }} }};
document.getElementById("applyJson").onclick = () => {{ try {{ setControls(JSON.parse(document.getElementById("jsonBox").value)); }} catch(e) {{ alert("JSON解析失败"); }} }};
document.getElementById("copyJson").onclick = async () => {{ await navigator.clipboard.writeText(document.getElementById("jsonBox").value); alert("已复制"); }};
setControls(PRESETS[0]);
</script>
</body>
</html>"""
    (OUT / "green_anchor_segment_workbench_v18.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_html(load_price())
    print(OUT / "green_anchor_segment_workbench_v18.html")


if __name__ == "__main__":
    main()
