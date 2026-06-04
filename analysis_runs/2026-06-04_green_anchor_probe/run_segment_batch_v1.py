from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

PEAKS = [
    {"cycle": "2017", "date": "2017-12-17"},
    {"cycle": "2021", "date": "2021-11-08"},
    {"cycle": "2025", "date": "2025-10-05"},
]

PRE_DAYS = 183
POST_DAYS = 183


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def nearest(series: pd.Series, date_text: str) -> pd.Timestamp:
    target = pd.Timestamp(date_text)
    if target in series.index:
        return target
    return series.index[int(abs((series.index - target).days).argmin())]


def window(series: pd.Series, anchor: pd.Timestamp, pre: int, post: int) -> pd.DataFrame:
    seg = series.loc[anchor - pd.Timedelta(days=pre) : anchor + pd.Timedelta(days=post)]
    anchor_px = float(series.loc[anchor])
    return pd.DataFrame(
        {
            "date": seg.index,
            "rel_day": (seg.index - anchor).days.astype(float),
            "price": seg.values,
            "log_norm": np.log(seg.values / anchor_px),
        }
    )


def amp_by_vol(left: pd.DataFrame, right: pd.DataFrame) -> float:
    idx_min = max(left["rel_day"].min(), right["rel_day"].min())
    idx_max = min(left["rel_day"].max(), right["rel_day"].max())
    l = left[(left["rel_day"] >= idx_min) & (left["rel_day"] <= idx_max)]["log_norm"].diff().dropna()
    r = right[(right["rel_day"] >= idx_min) & (right["rel_day"] <= idx_max)]["log_norm"].diff().dropna()
    if len(l) < 20 or len(r) < 20 or float(l.std()) == 0:
        return 1.0
    return float(r.std() / l.std())


def rmse_for(left: pd.DataFrame, right: pd.DataFrame, amp: float, time_scale: float, shift: float) -> float:
    l = left.copy()
    l["rel_plot"] = l["rel_day"] * time_scale + shift
    l["log_plot"] = l["log_norm"] * amp
    r = right.copy()
    r["rel_plot"] = r["rel_day"]
    r["log_plot"] = r["log_norm"]

    # Compare by nearest integer plotted day.
    l["day"] = l["rel_plot"].round().astype(int)
    r["day"] = r["rel_plot"].round().astype(int)
    m = pd.merge(l[["day", "log_plot"]], r[["day", "log_plot"]], on="day", suffixes=("_left", "_right"))
    if len(m) < 20:
        return float("inf")
    return float(np.sqrt(np.mean((m["log_plot_left"] - m["log_plot_right"]) ** 2)))


def coarse_fit(left: pd.DataFrame, right: pd.DataFrame, amp0: float) -> dict:
    best = None
    for amp in np.arange(max(0.15, amp0 - 0.20), min(1.30, amp0 + 0.21), 0.02):
        for time_scale in np.arange(0.70, 1.31, 0.02):
            for shift in range(-120, 121, 5):
                score = rmse_for(left, right, float(amp), float(time_scale), float(shift))
                if best is None or score < best["rmse"]:
                    best = {
                        "amp_scale": round(float(amp), 4),
                        "time_scale": round(float(time_scale), 4),
                        "shift_days": int(shift),
                        "rmse": round(score, 5),
                    }
    return best or {"amp_scale": amp0, "time_scale": 1.0, "shift_days": 0, "rmse": None}


def overlay(left: pd.DataFrame, right: pd.DataFrame, params: dict) -> pd.DataFrame:
    l = left.copy()
    l["series"] = "left_projected"
    l["rel_plot"] = l["rel_day"] * params["time_scale"] + params["shift_days"]
    l["log_plot"] = l["log_norm"] * params["amp_scale"]
    r = right.copy()
    r["series"] = "right_actual"
    r["rel_plot"] = r["rel_day"]
    r["log_plot"] = r["log_norm"]
    out = pd.concat([l, r], ignore_index=True)
    out["date"] = out["date"].dt.date.astype(str)
    return out


def run_pair(series: pd.Series, left_peak: dict, right_peak: dict) -> dict:
    left_anchor = nearest(series, left_peak["date"])
    right_anchor = nearest(series, right_peak["date"])
    left = window(series, left_anchor, PRE_DAYS, POST_DAYS)
    right = window(series, right_anchor, PRE_DAYS, POST_DAYS)
    amp0 = amp_by_vol(left, right)
    fit = coarse_fit(left, right, amp0)
    over = overlay(left, right, fit)

    pair_id = f"peak_half_year_{left_peak['cycle']}_to_{right_peak['cycle']}"
    over.to_csv(OUT / f"{pair_id}_overlay.csv", index=False, encoding="utf-8-sig")
    summary = {
        "case_id": pair_id,
        "case_type": "peak_half_year",
        "left_cycle": left_peak["cycle"],
        "right_cycle": right_peak["cycle"],
        "left_anchor": left_anchor.date().isoformat(),
        "right_anchor": right_anchor.date().isoformat(),
        "left_anchor_price": round(float(series.loc[left_anchor]), 2),
        "right_anchor_price": round(float(series.loc[right_anchor]), 2),
        "pre_days": PRE_DAYS,
        "post_days": POST_DAYS,
        "amp_by_vol": round(amp0, 4),
        **fit,
    }
    (OUT / f"{pair_id}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_html(summaries: list[dict]) -> None:
    payload = {}
    for s in summaries:
        df = pd.read_csv(OUT / f"{s['case_id']}_overlay.csv")
        payload[s["case_id"]] = df.to_dict(orient="records")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Peak Half-Year Batch v1</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; gap:12px; align-items:center; padding:12px 16px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; }}
select {{ color:#eaf1fb; background:#1b2029; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 390px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 86px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 86px); overflow:auto; }}
.strong {{ color:#fff; font-weight:700; }}
.red {{ color:#ff6a5f; }}
.green {{ color:#00e676; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
</style>
</head>
<body>
<header>
  <h1>Case 01 batch: Peak 前后半年</h1>
  <select id="caseSelect"></select>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div></aside>
</main>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const SUMMARIES = {json.dumps(summaries, ensure_ascii=False)};
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while(svg.firstChild) svg.removeChild(svg.firstChild); }}
function render(id) {{
  clear();
  const rows = DATA[id];
  const vals = rows.map(r=>+r.log_plot);
  const xmin = Math.min(...rows.map(r=>+r.rel_plot)), xmax = Math.max(...rows.map(r=>+r.rel_plot));
  const ymin = Math.min(...vals), ymax = Math.max(...vals);
  function x(d) {{ return M.l + (+d.rel_plot-xmin)/(xmax-xmin)*(W-M.l-M.r); }}
  function y(d) {{ return H-M.b - (+d.log_plot-ymin)/(ymax-ymin)*(H-M.t-M.b); }}
  for (let i=0;i<9;i++) el("line", {{x1:M.l+i/8*(W-M.l-M.r),y1:M.t,x2:M.l+i/8*(W-M.l-M.r),y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  for (let i=0;i<7;i++) el("line", {{x1:M.l,y1:M.t+i/6*(H-M.t-M.b),x2:W-M.r,y2:M.t+i/6*(H-M.t-M.b),stroke:"rgba(255,255,255,.08)"}});
  ["left_projected","right_actual"].forEach(series => {{
    const r = rows.filter(d=>d.series===series).sort((a,b)=>a.rel_plot-b.rel_plot);
    let p = "";
    r.forEach((d,i)=> p += (i?"L":"M")+x(d).toFixed(2)+","+y(d).toFixed(2));
    el("path", {{d:p,fill:"none",stroke:series==="left_projected"?"#ff6a5f":"#00e676","stroke-width":series==="left_projected"?2.1:2.4,opacity:.9}});
  }});
  const s = SUMMARIES.find(x=>x.case_id===id);
  document.getElementById("summary").innerHTML = `
    <div class="strong">${{s.left_cycle}} peak → ${{s.right_cycle}} peak</div>
    <div><span class="red">红</span>：左周期 projected</div>
    <div><span class="green">绿</span>：右周期 actual</div>
    <br>
    <div>left anchor: ${{s.left_anchor}} (${{Math.round(s.left_anchor_price).toLocaleString()}})</div>
    <div>right anchor: ${{s.right_anchor}} (${{Math.round(s.right_anchor_price).toLocaleString()}})</div>
    <div>amp by vol: ${{s.amp_by_vol}}</div>
    <div>auto amp: ${{s.amp_scale}}</div>
    <div>auto time scale: ${{s.time_scale}}</div>
    <div>auto shift: ${{s.shift_days}}d</div>
    <div>rmse: ${{s.rmse}}</div>
    <br>
    <div class="item">这是 Codex 自动粗扫出来的初始值，不是最终答案。你后面可以在 v18 里人工修正并保存。</div>
  `;
}}
const sel = document.getElementById("caseSelect");
sel.innerHTML = SUMMARIES.map(s=>`<option value="${{s.case_id}}">${{s.left_cycle}} → ${{s.right_cycle}}</option>`).join("");
sel.onchange = e => render(e.target.value);
render(SUMMARIES[0].case_id);
</script>
</body>
</html>"""
    (OUT / "case01_peak_half_year_batch_v1.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    summaries = []
    for left_peak, right_peak in zip(PEAKS[:-1], PEAKS[1:]):
        summaries.append(run_pair(series, left_peak, right_peak))
    pd.DataFrame(summaries).to_csv(OUT / "case01_peak_half_year_batch_summary.csv", index=False, encoding="utf-8-sig")
    write_html(summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(OUT / "case01_peak_half_year_batch_v1.html")


if __name__ == "__main__":
    main()
