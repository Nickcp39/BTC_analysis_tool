from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

LEFT_TOP = pd.Timestamp("2021-11-08")
RIGHT_TOP = pd.Timestamp("2025-10-05")
PRE_DAYS = 220
POST_DAYS = 520


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def make_window(series: pd.Series, top: pd.Timestamp, label: str) -> pd.DataFrame:
    seg = series.loc[top - pd.Timedelta(days=PRE_DAYS) : top + pd.Timedelta(days=POST_DAYS)]
    top_price = float(series.loc[top])
    return pd.DataFrame(
        {
            "cycle": label,
            "date": seg.index,
            "rel_day": (seg.index - top).days.astype(float),
            "price": seg.values,
            "log_norm": np.log(seg.values / top_price),
        }
    )


def volatility_amp_scale(left: pd.DataFrame, right: pd.DataFrame) -> float:
    # Compare realized log-return volatility on the visible pre/post shared range.
    # This adjusts height only; time is not compressed.
    common_min = max(left["rel_day"].min(), right["rel_day"].min())
    common_max = min(left["rel_day"].max(), right["rel_day"].max(), 220)
    l = left[(left["rel_day"] >= common_min) & (left["rel_day"] <= common_max)].copy()
    r = right[(right["rel_day"] >= common_min) & (right["rel_day"] <= common_max)].copy()
    l_ret = l["log_norm"].diff().dropna()
    r_ret = r["log_norm"].diff().dropna()
    if len(l_ret) < 20 or len(r_ret) < 20 or float(l_ret.std()) == 0:
        return 1.0
    return float(r_ret.std() / l_ret.std())


def write_html(plot: pd.DataFrame, default_amp: float) -> None:
    payload = plot.copy()
    payload["date"] = payload["date"].dt.date.astype(str)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Amplitude Only v16</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; align-items:center; gap:18px; padding:12px 18px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; }}
.ctrl {{ display:flex; align-items:center; gap:8px; font-size:13px; color:#aeb8c8; }}
input[type=range] {{ width:170px; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button:hover {{ background:#354154; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 410px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 92px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 92px); overflow:auto; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.green {{ color:#00e676; }}
.red {{ color:#ff6a5f; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
textarea {{ width:100%; min-height:96px; resize:vertical; color:#dce7f7; background:#1c2129; border:1px solid rgba(255,255,255,.14); border-radius:6px; padding:8px; font-family:Consolas,monospace; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>v16 只改高度：时间不缩放</h1>
  <div class="ctrl">高度系数 <input id="amp" type="range" min="0.20" max="1.20" value="{default_amp:.3f}" step="0.01"><b id="ampText"></b></div>
  <div class="ctrl">时间 scale <input id="timeScale" type="range" min="0.70" max="1.30" value="1.00" step="0.01"><b id="timeScaleText"></b></div>
  <div class="ctrl">水平平移 <input id="shift" type="range" min="-180" max="180" value="0" step="1"><b id="shiftText"></b></div>
  <button id="saveSettings">保存当前设置</button>
  <button id="downloadSettings">下载JSON</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const RAW = {json.dumps(payload.to_dict(orient="records"), ensure_ascii=False)};
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
function clear() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function color(cycle) {{ return cycle === "2025_actual" ? "#00e676" : "#ff6a5f"; }}
function label(cycle) {{ return cycle === "2025_actual" ? "2025 actual" : "2021 height-adjusted"; }}
let lastSettings = null;
function render() {{
  clear();
  const amp = +document.getElementById("amp").value;
  const timeScale = +document.getElementById("timeScale").value;
  const shift = +document.getElementById("shift").value;
  document.getElementById("ampText").textContent = amp.toFixed(2);
  document.getElementById("timeScaleText").textContent = timeScale.toFixed(2);
  document.getElementById("shiftText").textContent = shift + "d";
  const data = RAW.map(d => {{
    const out = {{...d}};
    if (out.cycle === "2021_height_only") {{
      out.rel_plot = +out.rel_day * timeScale + shift;
      out.log_plot = +out.log_norm * amp;
    }} else {{
      out.rel_plot = +out.rel_day;
      out.log_plot = +out.log_norm;
    }}
    return out;
  }});
  const xmin = -{PRE_DAYS}, xmax = {POST_DAYS};
  const vals = data.map(d => d.log_plot);
  const ymin = Math.min(...vals), ymax = Math.max(...vals);
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
  el("line", {{x1:M.l + (0 - xmin)/(xmax-xmin)*(W-M.l-M.r), y1:M.t, x2:M.l + (0 - xmin)/(xmax-xmin)*(W-M.l-M.r), y2:H-M.b, stroke:"rgba(255,255,255,.45)"}});
  ["2021_height_only","2025_actual"].forEach(cycle => {{
    const rows = data.filter(d => d.cycle === cycle).sort((a,b)=>a.rel_plot-b.rel_plot);
    let path = "";
    rows.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
    el("path", {{d:path, fill:"none", stroke:color(cycle), "stroke-width":cycle==="2025_actual"?2.4:2.1, opacity:cycle==="2025_actual"?.95:.78}});
  }});
  document.getElementById("summary").innerHTML = `
    <div class="strong">当前设置</div>
    <div><span class="red">红</span>：2021，只调整高度</div>
    <div><span class="green">绿</span>：2025 实际</div>
    <div>红线时间scale：${{timeScale.toFixed(2)}}</div>
    <div>红线水平平移：${{shift}} 天</div>
    <div>红线高度系数：${{amp.toFixed(2)}}</div>
    <div>公式：red_rel_plot = red_rel_day × time_scale + shift</div>
    <br>
  `;
  lastSettings = {{
    version: "v16_manual_scale",
    amp_scale: Number(amp.toFixed(4)),
    time_scale: Number(timeScale.toFixed(4)),
    shift_days: shift,
    transform: "red_rel_plot = red_rel_day * time_scale + shift_days; red_log_norm = red_log_norm * amp_scale",
    left_anchor: "2021-11-08",
    right_anchor: "2025-10-05",
    pre_days: {PRE_DAYS},
    post_days: {POST_DAYS},
    saved_at: new Date().toISOString()
  }};
  document.getElementById("list").innerHTML = `
    <div class="item">现在可以同时手动调高度、整体时间scale、水平平移。时间scale是整体伸缩，不是分段扭曲。</div>
    <div class="item">默认高度系数来自可见窗口日波动率比：2025 / 2021。</div>
    <div class="item"><div class="strong">当前设置 JSON</div><textarea id="settingsJson" readonly>${{JSON.stringify(lastSettings, null, 2)}}</textarea></div>
  `;
}}
document.getElementById("amp").addEventListener("input", render);
document.getElementById("timeScale").addEventListener("input", render);
document.getElementById("shift").addEventListener("input", render);
document.getElementById("saveSettings").addEventListener("click", () => {{
  if (!lastSettings) render();
  localStorage.setItem("btc_v16_manual_settings", JSON.stringify(lastSettings, null, 2));
  const box = document.getElementById("settingsJson");
  if (box) box.value = JSON.stringify(lastSettings, null, 2);
  alert("已保存到浏览器本地，JSON 也在右侧。");
}});
document.getElementById("downloadSettings").addEventListener("click", () => {{
  if (!lastSettings) render();
  const blob = new Blob([JSON.stringify(lastSettings, null, 2)], {{type:"application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "btc_manual_scale_settings_v16.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}});
const saved = localStorage.getItem("btc_v16_manual_settings");
if (saved) {{
  try {{
    const s = JSON.parse(saved);
    if (typeof s.amp_scale === "number") document.getElementById("amp").value = s.amp_scale;
    if (typeof s.time_scale === "number") document.getElementById("timeScale").value = s.time_scale;
    if (typeof s.shift_days === "number") document.getElementById("shift").value = s.shift_days;
  }} catch (e) {{}}
}}
render();
</script>
</body>
</html>"""
    (OUT / "green_anchor_amplitude_only_v16.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    left = make_window(series, LEFT_TOP, "2021_height_only")
    right = make_window(series, RIGHT_TOP, "2025_actual")
    amp = volatility_amp_scale(left, right)
    plot = pd.concat([left, right], ignore_index=True)
    plot.to_csv(OUT / "amplitude_only_v16.csv", index=False, encoding="utf-8-sig")
    write_html(plot, amp)
    print(f"amp={amp:.4f}")
    print(OUT / "green_anchor_amplitude_only_v16.html")


if __name__ == "__main__":
    main()
