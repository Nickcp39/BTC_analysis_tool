from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent


CASES = [
    {
        "id": "case01_peak_pre1y_2021_vs_2025",
        "name": "Peak 前一年: 2021 peak pre-1y vs 2025 peak pre-1y",
        "category": "peak_pre_1y",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-10-05",
        "pre_days": 365,
        "post_days": 0,
        "notes": "只比较 peak 前一年到 peak 当天的路径；先整体调高度、时间 scale、水平平移。",
    },
]


BACKLOG = [
    ("case01", "peak_pre_1y", "Peak 前一年"),
    ("case02", "peak_post_1y", "Peak 后一年"),
    ("case03", "peak_pm_6m", "Peak 前后半年"),
    ("case04", "bottom_pm_6m", "Bottom 前后半年"),
    ("case05", "bottom_pre_1y", "Bottom 前一年"),
    ("case06", "bottom_post_1y", "Bottom 后一年"),
    ("case07", "peak_pre_2y", "Peak 前两年"),
    ("case08", "peak_pre_3y", "Peak 前三年"),
    ("case09+", "local_peak_trough", "人工补充波峰/波谷片段"),
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def nearest(series: pd.Series, date_text: str) -> pd.Timestamp:
    target = pd.Timestamp(date_text)
    if target in series.index:
        return target
    return series.index[int(abs((series.index - target).days).argmin())]


def make_window(series: pd.Series, anchor: pd.Timestamp, pre: int, post: int, label: str) -> pd.DataFrame:
    seg = series.loc[anchor - pd.Timedelta(days=pre) : anchor + pd.Timedelta(days=post)]
    anchor_price = float(series.loc[anchor])
    return pd.DataFrame(
        {
            "series": label,
            "date": seg.index,
            "rel_day": (seg.index - anchor).days.astype(float),
            "price": seg.values.astype(float),
            "log_norm": np.log(seg.values.astype(float) / anchor_price),
        }
    )


def score_params(left: pd.DataFrame, right: pd.DataFrame, amp: float, time_scale: float, shift: float) -> tuple[float, int]:
    x_left = left["rel_day"].values * time_scale + shift
    y_left = left["log_norm"].values * amp
    order = np.argsort(x_left)
    x_left = x_left[order]
    y_left = y_left[order]

    x_right = right["rel_day"].values
    y_right = right["log_norm"].values
    mask = (x_right >= x_left.min()) & (x_right <= x_left.max())
    if mask.sum() < 90:
        return float("inf"), int(mask.sum())
    interp = np.interp(x_right[mask], x_left, y_left)
    rmse = float(np.sqrt(np.mean((interp - y_right[mask]) ** 2)))
    # Tiny regularization so the auto-start does not overfit with wild movement.
    rmse += abs(time_scale - 1.0) * 0.012 + abs(shift) / 10000.0
    return rmse, int(mask.sum())


def coarse_fit(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    best: dict | None = None
    for amp in np.arange(0.25, 0.91, 0.025):
        for time_scale in np.arange(0.70, 1.301, 0.025):
            for shift in range(-180, 181, 5):
                rmse, common = score_params(left, right, float(amp), float(time_scale), float(shift))
                if best is None or rmse < best["rmse"]:
                    best = {
                        "amp_scale": round(float(amp), 4),
                        "time_scale": round(float(time_scale), 4),
                        "shift_days": int(shift),
                        "rmse": round(float(rmse), 6),
                        "common_days": common,
                    }
    assert best is not None
    return best


def build_overlay(left: pd.DataFrame, right: pd.DataFrame, params: dict) -> pd.DataFrame:
    l = left.copy()
    l["rel_plot"] = l["rel_day"] * params["time_scale"] + params["shift_days"]
    l["log_plot"] = l["log_norm"] * params["amp_scale"]
    r = right.copy()
    r["rel_plot"] = r["rel_day"]
    r["log_plot"] = r["log_norm"]
    out = pd.concat([l, r], ignore_index=True)
    out["date"] = out["date"].dt.date.astype(str)
    return out


def write_case_html(case: dict, overlay: pd.DataFrame, summary: dict) -> Path:
    html_path = OUT / f"{case['id']}.html"
    data_json = json.dumps(overlay.to_dict(orient="records"), ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{case['name']}</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; align-items:center; gap:14px; padding:12px 16px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; min-width:310px; }}
.ctrl {{ display:flex; align-items:center; gap:7px; color:#aeb8c8; font-size:13px; }}
input[type=range] {{ width:170px; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 10px; cursor:pointer; }}
button:hover {{ background:#354154; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 430px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 88px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 88px); overflow:auto; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.red {{ color:#ff6a5f; }}
.green {{ color:#00e676; }}
textarea {{ width:100%; min-height:130px; resize:vertical; color:#dce7f7; background:#1c2129; border:1px solid rgba(255,255,255,.14); border-radius:6px; padding:8px; font-family:Consolas,monospace; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>{case['name']}</h1>
  <div class="ctrl">高度 <input id="amp" type="range" min="0.10" max="1.50" step="0.01" value="{summary['auto_amp_scale']:.4f}"><b id="ampText"></b></div>
  <div class="ctrl">时间scale <input id="timeScale" type="range" min="0.50" max="1.50" step="0.01" value="{summary['auto_time_scale']:.4f}"><b id="timeText"></b></div>
  <div class="ctrl">平移 <input id="shift" type="range" min="-360" max="360" step="1" value="{summary['auto_shift_days']}"><b id="shiftText"></b></div>
  <button id="save">保存当前设置</button>
  <button id="download">下载JSON</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><textarea id="jsonBox"></textarea></aside>
</main>
<script>
const RAW = {data_json};
const SUMMARY = {summary_json};
const STORE = "btc_segment_experiment_" + SUMMARY.case_id;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
let latestSettings = null;
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while(svg.firstChild) svg.removeChild(svg.firstChild); }}
function render() {{
  clear();
  const amp = +document.getElementById("amp").value;
  const timeScale = +document.getElementById("timeScale").value;
  const shift = +document.getElementById("shift").value;
  document.getElementById("ampText").textContent = amp.toFixed(2);
  document.getElementById("timeText").textContent = timeScale.toFixed(2);
  document.getElementById("shiftText").textContent = shift + "d";
  const data = RAW.map(d => {{
    const row = {{...d}};
    if (row.series === "left") {{
      row.rel_plot = row.rel_day * timeScale + shift;
      row.log_plot = row.log_norm * amp;
    }}
    return row;
  }});
  const xmin = Math.min(...data.map(d=>+d.rel_plot)), xmax = Math.max(...data.map(d=>+d.rel_plot));
  const ymin = Math.min(...data.map(d=>+d.log_plot)), ymax = Math.max(...data.map(d=>+d.log_plot));
  function x(d) {{ return M.l + (+d.rel_plot-xmin)/(xmax-xmin)*(W-M.l-M.r); }}
  function y(d) {{ return H-M.b - (+d.log_plot-ymin)/(ymax-ymin)*(H-M.t-M.b); }}
  for (let i=0;i<9;i++) el("line", {{x1:M.l+i/8*(W-M.l-M.r),y1:M.t,x2:M.l+i/8*(W-M.l-M.r),y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  for (let i=0;i<7;i++) el("line", {{x1:M.l,y1:M.t+i/6*(H-M.t-M.b),x2:W-M.r,y2:M.t+i/6*(H-M.t-M.b),stroke:"rgba(255,255,255,.08)"}});
  ["left","right"].forEach(series => {{
    const rows = data.filter(d=>d.series===series).sort((a,b)=>a.rel_plot-b.rel_plot);
    let path = "";
    rows.forEach((d,i)=> path += (i?"L":"M")+x(d).toFixed(2)+","+y(d).toFixed(2));
    el("path", {{d:path,fill:"none",stroke:series==="left"?"#ff6a5f":"#00e676","stroke-width":series==="left"?2.1:2.4,opacity:.9}});
  }});
  latestSettings = {{
    case_id: SUMMARY.case_id,
    name: SUMMARY.name,
    category: SUMMARY.category,
    left_anchor: SUMMARY.left_anchor,
    right_anchor: SUMMARY.right_anchor,
    pre_days: SUMMARY.pre_days,
    post_days: SUMMARY.post_days,
    amp_scale: Number(amp.toFixed(4)),
    time_scale: Number(timeScale.toFixed(4)),
    shift_days: shift,
    saved_at: new Date().toISOString()
  }};
  document.getElementById("summary").innerHTML = `
    <div class="item"><div class="strong">自动初始值</div>
    amp ${{SUMMARY.auto_amp_scale}} · time ${{SUMMARY.auto_time_scale}} · shift ${{SUMMARY.auto_shift_days}}d · rmse ${{SUMMARY.auto_rmse}}</div>
    <div class="item"><span class="red">红</span>：${{SUMMARY.left_anchor}} 前一年路径，经手动变换<br><span class="green">绿</span>：${{SUMMARY.right_anchor}} 前一年实际路径</div>
    <div class="item">目标：你手动调到视觉上最像，然后保存 JSON。</div>
  `;
  document.getElementById("jsonBox").value = JSON.stringify(latestSettings, null, 2);
}}
document.getElementById("amp").addEventListener("input", render);
document.getElementById("timeScale").addEventListener("input", render);
document.getElementById("shift").addEventListener("input", render);
document.getElementById("save").onclick = () => {{
  localStorage.setItem(STORE, JSON.stringify(latestSettings, null, 2));
  alert("已保存到浏览器本地。");
}};
document.getElementById("download").onclick = () => {{
  const blob = new Blob([JSON.stringify(latestSettings, null, 2)], {{type:"application/json"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = SUMMARY.case_id + "_manual_settings.json"; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}};
const saved = localStorage.getItem(STORE);
if (saved) {{
  try {{
    const s = JSON.parse(saved);
    document.getElementById("amp").value = s.amp_scale;
    document.getElementById("timeScale").value = s.time_scale;
    document.getElementById("shift").value = s.shift_days;
  }} catch(e) {{}}
}}
render();
</script>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


def run_case(case: dict, series: pd.Series) -> dict:
    left_anchor = nearest(series, case["left_anchor"])
    right_anchor = nearest(series, case["right_anchor"])
    left = make_window(series, left_anchor, case["pre_days"], case["post_days"], "left")
    right = make_window(series, right_anchor, case["pre_days"], case["post_days"], "right")
    fit = coarse_fit(left, right)
    overlay_df = build_overlay(left, right, fit)

    overlay_path = OUT / f"{case['id']}_overlay_auto.csv"
    overlay_df.to_csv(overlay_path, index=False, encoding="utf-8-sig")

    summary = {
        "case_id": case["id"],
        "name": case["name"],
        "category": case["category"],
        "left_anchor": left_anchor.date().isoformat(),
        "right_anchor": right_anchor.date().isoformat(),
        "left_anchor_price": round(float(series.loc[left_anchor]), 2),
        "right_anchor_price": round(float(series.loc[right_anchor]), 2),
        "pre_days": case["pre_days"],
        "post_days": case["post_days"],
        "auto_amp_scale": fit["amp_scale"],
        "auto_time_scale": fit["time_scale"],
        "auto_shift_days": fit["shift_days"],
        "auto_rmse": fit["rmse"],
        "common_days": fit["common_days"],
        "overlay_csv": overlay_path.name,
    }
    summary_path = OUT / f"{case['id']}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_case_html(case, overlay_df, summary)
    summary["html"] = html_path.name
    return summary


def write_plan_and_log(summaries: list[dict]) -> None:
    plan = [
        "# Segment Experiment Plan",
        "",
        "## 当前运行",
        "Case01: 对比 2021 peak 前一年 与 2025 peak 前一年。",
        "",
        "## 原则",
        "- 一次只跑一个明确片段类型。",
        "- Codex 先给自动初始值和可调 HTML。",
        "- 用户人工调高度、时间 scale、平移，并保存 JSON。",
        "- Codex 记录人工设置，再跑下一段。",
        "",
        "## 后续队列",
    ]
    for key, title in [
        ("case02", "peak 后一年"),
        ("case03", "peak 前后半年"),
        ("case04", "bottom 前后半年"),
        ("case05", "bottom 前一年"),
        ("case06", "bottom 后一年"),
        ("case07", "peak 前两年"),
        ("case08", "peak 前三年"),
    ]:
        plan.append(f"- {key}: {title}")
    (OUT / "SEGMENT_EXPERIMENT_PLAN.md").write_text("\n".join(plan), encoding="utf-8")

    rows = []
    for s in summaries:
        rows.append(
            {
                "case_id": s["case_id"],
                "name": s["name"],
                "status": "auto_initial_run_complete",
                "left_anchor": s["left_anchor"],
                "right_anchor": s["right_anchor"],
                "pre_days": s["pre_days"],
                "post_days": s["post_days"],
                "auto_amp_scale": s["auto_amp_scale"],
                "auto_time_scale": s["auto_time_scale"],
                "auto_shift_days": s["auto_shift_days"],
                "auto_rmse": s["auto_rmse"],
                "manual_amp_scale": "",
                "manual_time_scale": "",
                "manual_shift_days": "",
                "visual_grade_1_5": "",
                "manual_notes": "",
                "html": s["html"],
                "overlay_csv": s["overlay_csv"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "segment_experiment_log.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    summaries = [run_case(CASES[0], series)]
    write_plan_and_log(summaries)
    print(json.dumps(summaries[0], indent=2, ensure_ascii=False))
    print(OUT / f"{CASES[0]['id']}.html")


if __name__ == "__main__":
    main()
