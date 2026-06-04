from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent


DEFAULT_CASES = [
    {
        "name": "2021 peak -> 2025 peak",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-10-05",
        "pre_days": 220,
        "post_days": 520,
        "amp_scale": 0.50,
        "height_scale": 0.50,
        "drawdown_ratio": 0.50,
        "time_scale": 1.00,
        "shift_days": -59,
    },
    {
        "name": "2021 bottom -> 2026 local bottom",
        "left_anchor": "2022-11-21",
        "right_anchor": "2026-02-05",
        "pre_days": 300,
        "post_days": 220,
        "amp_scale": 0.60,
        "height_scale": 0.60,
        "drawdown_ratio": 0.60,
        "time_scale": 1.00,
        "shift_days": 0,
    },
    {
        "name": "2022 lower high -> 2025 lower high",
        "left_anchor": "2022-03-29",
        "right_anchor": "2025-11-10",
        "pre_days": 180,
        "post_days": 260,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
    },
    {
        "name": "2022 capitulation low -> 2026 local low",
        "left_anchor": "2022-06-18",
        "right_anchor": "2026-02-22",
        "pre_days": 220,
        "post_days": 220,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
    },
    {
        "name": "2022 rebound high -> 2026 rebound high",
        "left_anchor": "2022-08-14",
        "right_anchor": "2026-05-08",
        "pre_days": 180,
        "post_days": 180,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
    },
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def payload(series: pd.Series) -> list[list[object]]:
    return [[d.date().isoformat(), round(float(v), 2)] for d, v in series.items()]


def write_html(series: pd.Series) -> None:
    data_json = json.dumps(payload(series), ensure_ascii=False)
    cases_json = json.dumps(DEFAULT_CASES, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Manual Cycle Workbench v17</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; align-items:center; gap:14px; padding:12px 16px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; white-space:nowrap; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 10px; cursor:pointer; }}
button:hover {{ background:#354154; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 470px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 86px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 86px); overflow:auto; }}
.grid {{ display:grid; grid-template-columns:118px 1fr; gap:8px; align-items:center; font-size:12px; color:#b5c0d0; }}
input, select, textarea {{ width:100%; color:#eaf1fb; background:#1b2029; border:1px solid rgba(255,255,255,.14); border-radius:6px; padding:7px; }}
input[type=range] {{ padding:0; }}
textarea {{ min-height:110px; resize:vertical; font-family:Consolas,monospace; font-size:12px; }}
.section {{ padding:10px 0; border-bottom:1px solid rgba(255,255,255,.08); }}
.row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.red {{ color:#ff6a5f; }}
.green {{ color:#00e676; }}
.muted {{ color:#aeb8c8; font-size:12px; line-height:1.45; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
</style>
</head>
<body>
<header>
  <h1>BTC 人工周期结构对比工作台 v17</h1>
  <button id="newCase">新建对比</button>
  <button id="saveCase">保存/更新当前</button>
  <button id="deleteCase">删除当前</button>
  <button id="downloadAll">下载规则库JSON</button>
  <button id="exportCsv">导出当前曲线CSV</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel">
    <div class="section">
      <div class="grid">
        <label>当前case</label><select id="caseSelect"></select>
        <label>名称</label><input id="caseName" />
        <label>左锚点</label><input id="leftAnchor" type="date" />
        <label>右锚点</label><input id="rightAnchor" type="date" />
        <label>左窗口前</label><input id="preDays" type="number" min="10" max="1200" step="1" />
        <label>左窗口后</label><input id="postDays" type="number" min="10" max="1400" step="1" />
      </div>
    </div>
    <div class="section">
      <div class="grid">
        <label>高度系数</label><div><input id="amp" type="range" min="0.10" max="1.50" step="0.01" /><span id="ampText"></span></div>
        <label>时间scale</label><div><input id="timeScale" type="range" min="0.50" max="1.50" step="0.01" /><span id="timeScaleText"></span></div>
        <label>水平平移</label><div><input id="shift" type="range" min="-360" max="360" step="1" /><span id="shiftText"></span></div>
      </div>
    </div>
    <div class="section">
      <div id="summary" class="muted"></div>
    </div>
    <div class="section">
      <div class="strong">当前设置 JSON</div>
      <textarea id="settingsJson"></textarea>
      <div class="row" style="margin-top:8px">
        <button id="applyJson">从JSON应用</button>
        <button id="copyJson">复制JSON</button>
      </div>
    </div>
    <div id="caseList"></div>
  </aside>
</main>
<div id="tip" class="tip"></div>
<script>
const RAW = {data_json}.map(d => ({{dateText:d[0], date:new Date(d[0]), price:+d[1]}}));
const DEFAULT_CASES = {cases_json};
const STORAGE_KEY = "btc_manual_cycle_workbench_v17";
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
let cases = loadCases();
let activeIndex = 0;
let currentRows = [];

function loadCases() {{
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) {{
    try {{ return JSON.parse(saved); }} catch (e) {{}}
  }}
  return structuredClone(DEFAULT_CASES);
}}
function persist() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(cases, null, 2)); }}
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function byDate(dateText) {{
  const target = new Date(dateText).getTime();
  let best = RAW[0], bestD = Infinity;
  for (const row of RAW) {{
    const d = Math.abs(row.date.getTime() - target);
    if (d < bestD) {{ best = row; bestD = d; }}
  }}
  return best;
}}
function heightScale(c) {{
  return +(c.height_scale ?? c.drawdown_ratio ?? c.amp_scale ?? 1);
}}
function rowsFor(c) {{
  const leftAnchor = byDate(c.left_anchor);
  const rightAnchor = byDate(c.right_anchor);
  const leftStart = new Date(leftAnchor.date.getTime() - c.pre_days * 86400000);
  const leftEnd = new Date(leftAnchor.date.getTime() + c.post_days * 86400000);
  const rightStart = new Date(rightAnchor.date.getTime() - c.pre_days * 86400000);
  const rightEnd = new Date(rightAnchor.date.getTime() + c.post_days * 86400000);
  const left = RAW.filter(r => r.date >= leftStart && r.date <= leftEnd).map(r => {{
    const rel = (r.date - leftAnchor.date) / 86400000;
    return {{
      cycle:"left",
      dateText:r.dateText,
      rel_day:rel,
      rel_plot:rel * c.time_scale + c.shift_days,
      price:r.price,
      log_norm:Math.log(r.price / leftAnchor.price),
      log_plot:Math.log(r.price / leftAnchor.price) * heightScale(c)
    }};
  }});
  const right = RAW.filter(r => r.date >= rightStart && r.date <= rightEnd).map(r => {{
    const rel = (r.date - rightAnchor.date) / 86400000;
    return {{
      cycle:"right",
      dateText:r.dateText,
      rel_day:rel,
      rel_plot:rel,
      price:r.price,
      log_norm:Math.log(r.price / rightAnchor.price),
      log_plot:Math.log(r.price / rightAnchor.price)
    }};
  }});
  return {{left, right, leftAnchor, rightAnchor}};
}}
function draw() {{
  clear();
  const c = cases[activeIndex];
  const {{left, right, leftAnchor, rightAnchor}} = rowsFor(c);
  currentRows = [...left, ...right];
  const all = currentRows;
  const xmin = Math.min(...all.map(d=>d.rel_plot)), xmax = Math.max(...all.map(d=>d.rel_plot));
  const ymin = Math.min(...all.map(d=>d.log_plot)), ymax = Math.max(...all.map(d=>d.log_plot));
  function x(d) {{ return M.l + (d.rel_plot - xmin) / (xmax - xmin) * (W - M.l - M.r); }}
  function y(d) {{ return H - M.b - (d.log_plot - ymin) / (ymax - ymin) * (H - M.t - M.b); }}
  for (let i=0;i<9;i++) {{
    const gx = M.l + i/8*(W-M.l-M.r);
    el("line", {{x1:gx,y1:M.t,x2:gx,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  }}
  for (let i=0;i<7;i++) {{
    const gy = M.t + i/6*(H-M.t-M.b);
    el("line", {{x1:M.l,y1:gy,x2:W-M.r,y2:gy,stroke:"rgba(255,255,255,.08)"}});
  }}
  [{{rows:left, color:"#ff6a5f", width:2.1}}, {{rows:right, color:"#00e676", width:2.4}}].forEach(s => {{
    let path = "";
    s.rows.sort((a,b)=>a.rel_plot-b.rel_plot).forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
    el("path", {{d:path, fill:"none", stroke:s.color, "stroke-width":s.width, opacity:.9}});
  }});
  const zero = {{rel_plot:0, log_plot:0}};
  el("line", {{x1:x(zero), y1:M.t, x2:x(zero), y2:H-M.b, stroke:"rgba(255,255,255,.35)"}});
  updateSummary(c, leftAnchor, rightAnchor);
}}
function updateSummary(c, leftAnchor, rightAnchor) {{
  const json = currentCase();
  document.getElementById("summary").innerHTML = `
    <div><span class="red">红</span>：左路径，以 ${{c.left_anchor}} 为锚点，高度/时间/平移后显示</div>
    <div><span class="green">绿</span>：右路径，以 ${{c.right_anchor}} 为锚点，原样显示</div>
    <div>左锚价格：${{Math.round(leftAnchor.price).toLocaleString()}} · 右锚价格：${{Math.round(rightAnchor.price).toLocaleString()}}</div>
    <div>公式：left_rel_plot = left_rel_day × time_scale + shift_days</div>
  `;
  document.getElementById("settingsJson").value = JSON.stringify(json, null, 2);
  renderCaseList();
}}
function currentCase() {{
  return {{
    name: document.getElementById("caseName").value,
    left_anchor: document.getElementById("leftAnchor").value,
    right_anchor: document.getElementById("rightAnchor").value,
    pre_days: +document.getElementById("preDays").value,
    post_days: +document.getElementById("postDays").value,
    amp_scale: +document.getElementById("amp").value,
    height_scale: +document.getElementById("amp").value,
    drawdown_ratio: +document.getElementById("amp").value,
    time_scale: +document.getElementById("timeScale").value,
    shift_days: +document.getElementById("shift").value
  }};
}}
function setControls(c) {{
  document.getElementById("caseName").value = c.name || "untitled";
  document.getElementById("leftAnchor").value = c.left_anchor;
  document.getElementById("rightAnchor").value = c.right_anchor;
  document.getElementById("preDays").value = c.pre_days;
  document.getElementById("postDays").value = c.post_days;
  document.getElementById("amp").value = heightScale(c);
  document.getElementById("timeScale").value = c.time_scale;
  document.getElementById("shift").value = c.shift_days;
  updateTexts();
}}
function updateTexts() {{
  document.getElementById("ampText").textContent = (+document.getElementById("amp").value).toFixed(2);
  document.getElementById("timeScaleText").textContent = (+document.getElementById("timeScale").value).toFixed(2);
  document.getElementById("shiftText").textContent = document.getElementById("shift").value + "d";
}}
function syncCaseFromControls() {{
  cases[activeIndex] = currentCase();
  updateTexts();
  draw();
}}
function renderCaseSelect() {{
  const sel = document.getElementById("caseSelect");
  sel.innerHTML = cases.map((c,i)=>`<option value="${{i}}">${{i+1}}. ${{c.name || "untitled"}}</option>`).join("");
  sel.value = activeIndex;
}}
function renderCaseList() {{
  document.getElementById("caseList").innerHTML = cases.map((c,i)=>`
    <div class="item">
      <div class="strong">${{i+1}}. ${{c.name}}</div>
      <div>${{c.left_anchor}} → ${{c.right_anchor}}</div>
      <div>height/drawdown ${{heightScale(c).toFixed(2)}} · time ${{c.time_scale}} · shift ${{c.shift_days}}d</div>
    </div>
  `).join("");
}}
function download(name, text, type="application/json") {{
  const blob = new Blob([text], {{type}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}}
["caseName","leftAnchor","rightAnchor","preDays","postDays","amp","timeScale","shift"].forEach(id => {{
  document.getElementById(id).addEventListener("input", syncCaseFromControls);
}});
document.getElementById("caseSelect").addEventListener("change", e => {{
  activeIndex = +e.target.value;
  setControls(cases[activeIndex]);
  draw();
}});
document.getElementById("newCase").onclick = () => {{
  cases.push({{...currentCase(), name:"new comparison"}});
  activeIndex = cases.length - 1;
  renderCaseSelect();
  setControls(cases[activeIndex]);
  persist();
  draw();
}};
document.getElementById("saveCase").onclick = () => {{
  cases[activeIndex] = currentCase();
  persist();
  renderCaseSelect();
  draw();
  alert("已保存到浏览器本地。");
}};
document.getElementById("deleteCase").onclick = () => {{
  if (cases.length <= 1) return alert("至少保留一个 case。");
  cases.splice(activeIndex, 1);
  activeIndex = Math.max(0, activeIndex - 1);
  renderCaseSelect();
  setControls(cases[activeIndex]);
  persist();
  draw();
}};
document.getElementById("downloadAll").onclick = () => {{
  download("btc_cycle_manual_rules_v17.json", JSON.stringify({{version:"v17", cases}}, null, 2));
}};
document.getElementById("exportCsv").onclick = () => {{
  const header = "cycle,date,rel_day,rel_plot,price,log_norm,log_plot\\n";
  const rows = currentRows.map(r => [r.cycle,r.dateText,r.rel_day,r.rel_plot,r.price,r.log_norm,r.log_plot].join(",")).join("\\n");
  download("btc_current_comparison_v17.csv", header + rows, "text/csv");
}};
document.getElementById("applyJson").onclick = () => {{
  try {{
    const parsed = JSON.parse(document.getElementById("settingsJson").value);
    cases[activeIndex] = parsed;
    setControls(parsed);
    persist();
    renderCaseSelect();
    draw();
  }} catch (e) {{
    alert("JSON 解析失败。");
  }}
}};
document.getElementById("copyJson").onclick = async () => {{
  await navigator.clipboard.writeText(document.getElementById("settingsJson").value);
  alert("已复制。");
}};
renderCaseSelect();
setControls(cases[activeIndex]);
draw();
</script>
</body>
</html>"""
    (OUT / "green_anchor_manual_workbench_v17.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    write_html(series)
    print(OUT / "green_anchor_manual_workbench_v17.html")


if __name__ == "__main__":
    main()
