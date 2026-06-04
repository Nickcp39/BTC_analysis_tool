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
PRE_DAYS = 170
POST_DAYS_LEFT = 380
POST_DAYS_RIGHT = 238

LEFT_EMOTION = [
    ("top", "①顶", pd.Timestamp("2021-11-08")),
    ("lower_high", "②次高", pd.Timestamp("2022-03-29")),
    ("bottom", "③底", pd.Timestamp("2022-11-20")),
]
RIGHT_EMOTION = [
    ("top", "④顶", pd.Timestamp("2025-10-05")),
    ("lower_high", "⑤次高", pd.Timestamp("2025-11-10")),
    ("bottom", "⑥底", pd.Timestamp("2026-02-05")),
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def window(series: pd.Series, top: pd.Timestamp, pre: int, post: int, label: str) -> pd.DataFrame:
    seg = series.loc[top - pd.Timedelta(days=pre) : top + pd.Timedelta(days=post)].copy()
    top_price = float(series.loc[top])
    df = pd.DataFrame(
        {
            "date": seg.index,
            "rel_day": (seg.index - top).days,
            "price": seg.values,
            "norm_price": seg.values / top_price,
            "log_norm": np.log(seg.values / top_price),
            "cycle": label,
        }
    )
    return df


def emotion_points(series: pd.Series, points: list[tuple[str, str, pd.Timestamp]], top: pd.Timestamp, cycle: str) -> list[dict]:
    top_price = float(series.loc[top])
    out = []
    for role, label, date in points:
        d = date if date in series.index else series.index[int(abs((series.index - date).days).argmin())]
        price = float(series.loc[d])
        out.append(
            {
                "cycle": cycle,
                "role": role,
                "label": label,
                "date": d.date().isoformat(),
                "rel_day": int((d - top).days),
                "price": round(price, 2),
                "norm_price": round(price / top_price, 4),
            }
        )
    return out


def fit_pretrend(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    # Compare only the common pre-top window at daily relative days.
    common_start = max(left["rel_day"].min(), right["rel_day"].min())
    common_end = min(0, left["rel_day"].max(), right["rel_day"].max())
    l = left[(left["rel_day"] >= common_start) & (left["rel_day"] <= common_end)].set_index("rel_day")["log_norm"]
    r = right[(right["rel_day"] >= common_start) & (right["rel_day"] <= common_end)].set_index("rel_day")["log_norm"]
    idx = l.index.intersection(r.index)
    if len(idx) < 10:
        return {"pre_corr": None, "pre_rmse": None, "common_days": len(idx)}
    lv, rv = l.loc[idx].values, r.loc[idx].values
    return {
        "pre_corr": round(float(np.corrcoef(lv, rv)[0, 1]), 4),
        "pre_rmse": round(float(np.sqrt(np.mean((lv - rv) ** 2))), 4),
        "common_days": int(len(idx)),
    }


def write_html(left: pd.DataFrame, right: pd.DataFrame, points: list[dict], stats: dict) -> None:
    payload = pd.concat([left, right], ignore_index=True)
    payload["date"] = payload["date"].dt.date.astype(str)
    data_json = json.dumps(payload.to_dict(orient="records"), ensure_ascii=False)
    points_json = json.dumps(points, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Path Overlay v11</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ padding:12px 18px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 380px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 82px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 82px); overflow:auto; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.red {{ color:#ff5a4f; }}
.green {{ color:#00e676; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
</style>
</head>
<body>
<header><h1>v11 路径叠加：高点之前 + 顶后结构，不只是点位</h1></header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const DATA = {data_json};
const POINTS = {points_json};
const STATS = {stats_json};
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
function x(d) {{
  const min=-170, max=380;
  return M.l + (d.rel_day - min) / (max - min) * (W - M.l - M.r);
}}
function y(d) {{
  const min=0.22, max=1.18;
  const lo=Math.log(min), hi=Math.log(max), v=Math.log(d.norm_price);
  return H - M.b - (v - lo) / (hi - lo) * (H - M.t - M.b);
}}
function color(cycle) {{ return cycle === "2021_top" ? "#ff5a4f" : "#00e676"; }}
for (let i=0;i<9;i++) {{
  const gx = M.l + i/8*(W-M.l-M.r);
  el("line", {{x1:gx,y1:M.t,x2:gx,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
}}
for (let i=0;i<7;i++) {{
  const gy = M.t + i/6*(H-M.t-M.b);
  el("line", {{x1:M.l,y1:gy,x2:W-M.r,y2:gy,stroke:"rgba(255,255,255,.08)"}});
}}
el("line", {{x1:x({{rel_day:0}}),y1:M.t,x2:x({{rel_day:0}}),y2:H-M.b,stroke:"rgba(255,255,255,.55)","stroke-width":1}});
const topText = el("text", {{x:x({{rel_day:0}})+6,y:M.t+16,fill:"#fff","font-size":12}});
topText.textContent = "peak day = 0";
["2021_top","2025_top"].forEach(cycle => {{
  const rows = DATA.filter(d => d.cycle === cycle);
  let path = "";
  rows.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
  el("path", {{d:path, fill:"none", stroke:color(cycle), "stroke-width":2, opacity:cycle==="2021_top" ? .72 : .92}});
}});
POINTS.forEach(p => {{
  const c = el("circle", {{cx:x(p),cy:y(p),r:7,fill:p.role==="bottom" ? "#00c853" : "#ef4f43",stroke:"#130705","stroke-width":1.5}});
  c.addEventListener("mousemove", ev => {{
    tip.style.display = "block";
    tip.style.left = ev.clientX + 12 + "px";
    tip.style.top = ev.clientY + 12 + "px";
    tip.innerHTML = `<b>${{p.cycle}} ${{p.label}}</b><br>${{p.date}} · rel ${{p.rel_day}}d<br>norm ${{p.norm_price}}`;
  }});
  c.addEventListener("mouseleave", () => tip.style.display = "none");
  const t = el("text", {{x:x(p)+9,y:y(p)-9,fill:p.cycle==="2021_top" ? "#ff5a4f" : "#00e676","font-size":12,"font-weight":"700"}});
  t.textContent = p.label;
}});
document.getElementById("summary").innerHTML = `
  <div class="strong">路径比较</div>
  <div><span class="red">红</span>：2021 顶前/顶后，peak=2021-11-08</div>
  <div><span class="green">绿</span>：2025 顶前/顶后，peak=2025-10-05</div>
  <br>
  <div>顶前共同窗口：${{STATS.common_days}} 天</div>
  <div>顶前形态相关：${{STATS.pre_corr}}</div>
  <div>顶前 log RMSE：${{STATS.pre_rmse}}</div>
  <br>
`;
document.getElementById("list").innerHTML = POINTS.map(p => `
  <div class="item">
    <div class="strong">${{p.cycle}} · ${{p.label}}</div>
    <div>${{p.date}} · rel ${{p.rel_day}}d · norm ${{p.norm_price}}</div>
  </div>
`).join("");
</script>
</body>
</html>"""
    (OUT / "green_anchor_path_overlay_v11.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    left = window(series, LEFT_TOP, PRE_DAYS, POST_DAYS_LEFT, "2021_top")
    right = window(series, RIGHT_TOP, PRE_DAYS, POST_DAYS_RIGHT, "2025_top")
    points = emotion_points(series, LEFT_EMOTION, LEFT_TOP, "2021_top")
    points += emotion_points(series, RIGHT_EMOTION, RIGHT_TOP, "2025_top")
    stats = fit_pretrend(left, right)
    pd.concat([left, right], ignore_index=True).to_csv(OUT / "path_overlay_v11.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(points).to_csv(OUT / "path_overlay_points_v11.csv", index=False, encoding="utf-8-sig")
    write_html(left, right, points, stats)
    print(stats)


if __name__ == "__main__":
    main()
