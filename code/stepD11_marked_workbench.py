"""
StepD11: 人工周期对比工作台 v18（在 v17 基础上，叠加用户手标核心点）
- 复制自 analysis_runs/2026-06-04_green_anchor_probe/green_anchor_manual_workbench_v17.py
- 新增：读 output/core_points/manual_points.csv，把手标的高/低核心点叠到红(左)/绿(右)曲线上
  红圈=左周期(2021)的手标点(套 高度×/时间×/平移 变换)；绿圈=右周期(2025)手标点(原样)
  → 调 amp/time/shift 时可直接看"你标的核心点"对不对得齐。
- 默认参数用 stepD10 拟合值：amp(高度)=0.54、time=1.0、shift=-58。
- 输出：output/core_points/marked_workbench_v18.html（自包含，双击即开）
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "btc_merged_daily.csv"
MARKS_CSV = ROOT / "output" / "core_points" / "manual_points.csv"
OUT = ROOT / "output" / "core_points"

DEFAULT_CASES = [
    {
        "name": "2021顶 -> 2025顶（含手标点）",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-10-05",
        "pre_days": 400,
        "post_days": 240,
        "amp_scale": 0.54,
        "height_scale": 0.54,
        "drawdown_ratio": 0.54,
        "time_scale": 1.00,
        "shift_days": -58,
    },
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def load_marks() -> list:
    if not MARKS_CSV.exists():
        return []
    df = pd.read_csv(MARKS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return [[d.date().isoformat(), round(float(p), 2), str(t)]
            for d, p, t in zip(df["date"], df["price"], df["type"])]


def payload(series: pd.Series) -> list:
    return [[d.date().isoformat(), round(float(v), 2)] for d, v in series.items()]


def write_html(series: pd.Series, marks: list) -> None:
    data_json = json.dumps(payload(series), ensure_ascii=False)
    marks_json = json.dumps(marks, ensure_ascii=False)
    cases_json = json.dumps(DEFAULT_CASES, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC 周期工作台 v18（含手标核心点）</title>
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
.muted {{ color:#aeb8c8; font-size:12px; line-height:1.5; }}
.strong {{ color:#fff; font-weight:700; }}
.red {{ color:#ff6a5f; }}
.green {{ color:#00e676; }}
label.ck {{ display:flex; gap:6px; align-items:center; font-size:12px; color:#b5c0d0; }}
</style>
</head>
<body>
<header>
  <h1>BTC 周期工作台 v18 · 含手标核心点</h1>
  <button id="saveCase">保存/更新当前</button>
  <button id="downloadAll">下载规则库JSON</button>
  <label class="ck"><input type="checkbox" id="showMarks" checked style="width:auto" />显示手标点</label>
  <label class="ck"><input type="checkbox" id="connectMarks" style="width:auto" />连线核心点</label>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel">
    <div class="section">
      <div class="grid">
        <label>左锚点(顶)</label><input id="leftAnchor" type="date" />
        <label>右锚点(顶)</label><input id="rightAnchor" type="date" />
        <label>窗口·前</label><input id="preDays" type="number" min="10" max="1200" step="1" />
        <label>窗口·后</label><input id="postDays" type="number" min="10" max="1400" step="1" />
      </div>
    </div>
    <div class="section">
      <div class="grid">
        <label>高度系数</label><div><input id="amp" type="range" min="0.10" max="1.50" step="0.01" /><span id="ampText"></span></div>
        <label>时间scale</label><div><input id="timeScale" type="range" min="0.50" max="1.50" step="0.01" /><span id="timeScaleText"></span></div>
        <label>水平平移</label><div><input id="shift" type="range" min="-360" max="360" step="1" /><span id="shiftText"></span></div>
      </div>
    </div>
    <div class="section"><div id="summary" class="muted"></div></div>
    <div class="section">
      <div class="strong">当前设置 JSON</div>
      <textarea id="settingsJson"></textarea>
      <div class="row" style="margin-top:8px">
        <button id="applyJson">从JSON应用</button>
        <button id="copyJson">复制JSON</button>
      </div>
    </div>
  </aside>
</main>
<script>
const RAW = {data_json}.map(d => ({{dateText:d[0], date:new Date(d[0]), t:new Date(d[0]).getTime(), price:+d[1]}}));
const MARKS = {marks_json}.map(d => ({{dateText:d[0], t:new Date(d[0]).getTime(), price:+d[1], type:d[2]}}));
const DEFAULT_CASES = {cases_json};
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
let cases = structuredClone(DEFAULT_CASES);
let activeIndex = 0;

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
  for (const row of RAW) {{ const d = Math.abs(row.t - target); if (d < bestD) {{ best = row; bestD = d; }} }}
  return best;
}}
function heightScale(c) {{ return +(c.height_scale ?? c.amp_scale ?? 1); }}
function curveRows(c, anchor, side) {{
  const start = anchor.t - c.pre_days*86400000, end = anchor.t + c.post_days*86400000;
  return RAW.filter(r => r.t >= start && r.t <= end).map(r => {{
    const rel = (r.t - anchor.t)/86400000;
    const ln = Math.log(r.price/anchor.price);
    return side==="left"
      ? {{rel_plot: rel*c.time_scale + c.shift_days, log_plot: ln*heightScale(c)}}
      : {{rel_plot: rel, log_plot: ln}};
  }});
}}
function markRows(c, anchor, side) {{
  const start = anchor.t - c.pre_days*86400000, end = anchor.t + c.post_days*86400000;
  return MARKS.filter(m => m.t >= start && m.t <= end).map(m => {{
    const rel = (m.t - anchor.t)/86400000;
    const ln = Math.log(m.price/anchor.price);
    return side==="left"
      ? {{rel_plot: rel*c.time_scale + c.shift_days, log_plot: ln*heightScale(c), type:m.type, dateText:m.dateText}}
      : {{rel_plot: rel, log_plot: ln, type:m.type, dateText:m.dateText}};
  }});
}}
function draw() {{
  clear();
  const c = cases[activeIndex];
  const la = byDate(c.left_anchor), ra = byDate(c.right_anchor);
  const left = curveRows(c, la, "left"), right = curveRows(c, ra, "right");
  const lm = markRows(c, la, "left"), rm = markRows(c, ra, "right");
  const all = [...left, ...right];
  const xmin = Math.min(...all.map(d=>d.rel_plot)), xmax = Math.max(...all.map(d=>d.rel_plot));
  const ymin = Math.min(...all.map(d=>d.log_plot)), ymax = Math.max(...all.map(d=>d.log_plot));
  const x = d => M.l + (d.rel_plot - xmin)/(xmax - xmin)*(W-M.l-M.r);
  const y = d => H - M.b - (d.log_plot - ymin)/(ymax - ymin)*(H-M.t-M.b);
  for (let i=0;i<9;i++) {{ const gx=M.l+i/8*(W-M.l-M.r); el("line",{{x1:gx,y1:M.t,x2:gx,y2:H-M.b,stroke:"rgba(255,255,255,.07)"}}); }}
  for (let i=0;i<7;i++) {{ const gy=M.t+i/6*(H-M.t-M.b); el("line",{{x1:M.l,y1:gy,x2:W-M.r,y2:gy,stroke:"rgba(255,255,255,.07)"}}); }}
  [{{rows:left,color:"#ff6a5f",w:2.0}},{{rows:right,color:"#00e676",w:2.3}}].forEach(s => {{
    let path=""; s.rows.sort((a,b)=>a.rel_plot-b.rel_plot).forEach((d,i)=>path+=(i?"L":"M")+x(d).toFixed(2)+","+y(d).toFixed(2));
    el("path",{{d:path,fill:"none",stroke:s.color,"stroke-width":s.w,opacity:.85}});
  }});
  el("line",{{x1:x({{rel_plot:0}}),y1:M.t,x2:x({{rel_plot:0}}),y2:H-M.b,stroke:"rgba(255,255,255,.35)","stroke-dasharray":"4 4"}});
  if (document.getElementById("showMarks").checked) {{
    const connect = document.getElementById("connectMarks").checked;
    [{{rows:lm,stroke:"#ff6a5f",fill:"none"}},{{rows:rm,stroke:"#063",fill:"#00e676"}}].forEach(s => {{
      if (connect) {{ let p=""; s.rows.sort((a,b)=>a.rel_plot-b.rel_plot).forEach((d,i)=>p+=(i?"L":"M")+x(d).toFixed(2)+","+y(d).toFixed(2));
        el("path",{{d:p,fill:"none",stroke:s.stroke,"stroke-width":1,opacity:.5,"stroke-dasharray":"3 3"}}); }}
      s.rows.forEach(m => {{
        const r = m.type==="H" ? 5.2 : 4.2;
        el("circle",{{cx:x(m),cy:y(m),r:r,fill:s.fill,stroke:s.stroke,"stroke-width":1.6,opacity:.95}});
      }});
    }});
  }}
  updateSummary(c, la, ra, lm.length, rm.length);
}}
function updateSummary(c, la, ra, nl, nr) {{
  document.getElementById("summary").innerHTML = `
    <div><span class="red">红</span>=左周期(${{c.left_anchor}})，套 高度×${{heightScale(c).toFixed(2)}} / 时间×${{c.time_scale}} / 平移${{c.shift_days}}d；手标点 ${{nl}} 个</div>
    <div><span class="green">绿</span>=右周期(${{c.right_anchor}})，原样；手标点 ${{nr}} 个</div>
    <div>圈=手标核心点（大圈=高 H，小圈=低 L）。看红圈能否落到绿圈上。</div>
    <div>公式：left_rel_plot = rel_day × time_scale + shift；left_log = ln(P/锚) × 高度</div>`;
  document.getElementById("settingsJson").value = JSON.stringify(c, null, 2);
}}
function currentCase() {{
  return {{
    name: cases[activeIndex].name,
    left_anchor: document.getElementById("leftAnchor").value,
    right_anchor: document.getElementById("rightAnchor").value,
    pre_days: +document.getElementById("preDays").value,
    post_days: +document.getElementById("postDays").value,
    amp_scale: +document.getElementById("amp").value,
    height_scale: +document.getElementById("amp").value,
    time_scale: +document.getElementById("timeScale").value,
    shift_days: +document.getElementById("shift").value
  }};
}}
function setControls(c) {{
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
function sync() {{ cases[activeIndex] = currentCase(); updateTexts(); draw(); }}
["leftAnchor","rightAnchor","preDays","postDays","amp","timeScale","shift"].forEach(id =>
  document.getElementById(id).addEventListener("input", sync));
["showMarks","connectMarks"].forEach(id => document.getElementById(id).addEventListener("change", draw));
function download(name, text) {{
  const blob = new Blob([text], {{type:"application/json"}}); const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href=url; a.download=name; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}}
document.getElementById("saveCase").onclick = () => {{ cases[activeIndex]=currentCase(); draw(); alert("已更新当前 case（在内存里）。用'下载规则库JSON'保存到文件。"); }};
document.getElementById("downloadAll").onclick = () => download("btc_workbench_v18_rules.json", JSON.stringify({{version:"v18", cases}}, null, 2));
document.getElementById("applyJson").onclick = () => {{ try {{ cases[activeIndex]=JSON.parse(document.getElementById("settingsJson").value); setControls(cases[activeIndex]); draw(); }} catch(e) {{ alert("JSON解析失败"); }} }};
document.getElementById("copyJson").onclick = async () => {{ await navigator.clipboard.writeText(document.getElementById("settingsJson").value); alert("已复制"); }};
setControls(cases[activeIndex]);
draw();
</script>
</body>
</html>"""
    (OUT / "marked_workbench_v18.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    marks = load_marks()
    write_html(series, marks)
    print(f"手标点 {len(marks)} 个 已叠加")
    print("HTML:", OUT / "marked_workbench_v18.html")


if __name__ == "__main__":
    main()
