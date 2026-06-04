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
CURRENT_LATEST = pd.Timestamp("2026-05-31")

# User-marked structure: top -> lower high -> bottom.
LEFT_ANCHORS = [
    ("top", "①顶", pd.Timestamp("2021-11-08")),
    ("lower_high", "②次高", pd.Timestamp("2022-03-29")),
    ("bottom", "③底", pd.Timestamp("2022-11-20")),
]
RIGHT_ANCHORS = [
    ("top", "④顶", pd.Timestamp("2025-10-05")),
    ("lower_high", "⑤次高", pd.Timestamp("2025-11-10")),
    ("bottom", "⑥底", pd.Timestamp("2026-02-05")),
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def nearest(series: pd.Series, target: pd.Timestamp) -> pd.Timestamp:
    if target in series.index:
        return target
    return series.index[int(abs((series.index - target).days).argmin())]


def piecewise_map(x: np.ndarray, src: list[float], dst: list[float]) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for i, value in enumerate(x):
        if value <= src[0]:
            scale = (dst[1] - dst[0]) / (src[1] - src[0])
            out[i] = dst[0] + (value - src[0]) * scale
        elif value >= src[-1]:
            scale = (dst[-1] - dst[-2]) / (src[-1] - src[-2])
            out[i] = dst[-1] + (value - src[-1]) * scale
        else:
            j = max(k for k in range(len(src) - 1) if src[k] <= value)
            scale = (dst[j + 1] - dst[j]) / (src[j + 1] - src[j])
            out[i] = dst[j] + (value - src[j]) * scale
    return out


def build(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    latest = min(series.index.max(), CURRENT_LATEST)
    post_right = int((latest - RIGHT_TOP).days)

    left_rel = [int((d - LEFT_TOP).days) for _, _, d in LEFT_ANCHORS]
    right_rel = [int((d - RIGHT_TOP).days) for _, _, d in RIGHT_ANCHORS]
    post_scales = [
        (right_rel[i + 1] - right_rel[i]) / (left_rel[i + 1] - left_rel[i])
        for i in range(len(left_rel) - 1)
    ]
    base_compression = float(np.median(post_scales))

    # If current visible pre-top window is 170d, the comparable historical pre-top
    # window is longer by 1 / compression. This keeps compression <= 1.
    right_pre = 170
    left_pre = int(round(right_pre / base_compression))
    left_extend = int(round(post_right / base_compression))

    src = [-left_pre] + left_rel + [left_rel[-1] + left_extend]
    dst = [-right_pre] + right_rel + [post_right]
    segment_scales = [
        (dst[i + 1] - dst[i]) / (src[i + 1] - src[i])
        for i in range(len(src) - 1)
    ]

    if any(s > 1.000001 for s in segment_scales):
        raise ValueError(f"Time compression must be <= 1, got {segment_scales}")

    left_start = LEFT_TOP - pd.Timedelta(days=left_pre)
    left_end = LEFT_TOP + pd.Timedelta(days=left_rel[-1] + left_extend)
    right_start = RIGHT_TOP - pd.Timedelta(days=right_pre)
    right_end = latest

    left = series.loc[left_start:left_end]
    right = series.loc[right_start:right_end]
    left_top_price = float(series.loc[nearest(series, LEFT_TOP)])
    right_top_price = float(series.loc[nearest(series, RIGHT_TOP)])

    left_df = pd.DataFrame(
        {
            "cycle": "2021_raw",
            "date": left.index,
            "rel_day_original": (left.index - LEFT_TOP).days,
            "price": left.values,
            "log_norm": np.log(left.values / left_top_price),
        }
    )
    right_df = pd.DataFrame(
        {
            "cycle": "2025_actual",
            "date": right.index,
            "rel_day": (right.index - RIGHT_TOP).days,
            "price": right.values,
            "log_norm": np.log(right.values / right_top_price),
        }
    )

    ratios = []
    anchor_rows = []
    for (role_l, label_l, date_l), (_, label_r, date_r) in zip(LEFT_ANCHORS, RIGHT_ANCHORS):
        dl, dr = nearest(series, date_l), nearest(series, date_r)
        ln_l = np.log(float(series.loc[dl]) / left_top_price)
        ln_r = np.log(float(series.loc[dr]) / right_top_price)
        if role_l != "top" and abs(ln_l) > 1e-9:
            ratios.append(ln_r / ln_l)
        anchor_rows.append(
            {
                "role": role_l,
                "left_label": label_l,
                "right_label": label_r,
                "left_date": dl.date().isoformat(),
                "right_date": dr.date().isoformat(),
                "left_rel_day": int((dl - LEFT_TOP).days),
                "right_rel_day": int((dr - RIGHT_TOP).days),
                "left_norm": round(float(series.loc[dl]) / left_top_price, 4),
                "right_norm": round(float(series.loc[dr]) / right_top_price, 4),
            }
        )
    amp_scale = float(np.median(ratios)) if ratios else 1.0

    warped = left_df.copy()
    warped["cycle"] = "2021_warped_annealed"
    warped["rel_day"] = piecewise_map(warped["rel_day_original"].values.astype(float), src, dst)
    warped["log_norm"] = warped["log_norm"] * amp_scale
    warped = warped[(warped["rel_day"] >= -right_pre) & (warped["rel_day"] <= post_right)]

    plot_df = pd.concat([right_df, warped], ignore_index=True)
    anchors = pd.DataFrame(anchor_rows)
    stats = {
        "amp_scale": round(amp_scale, 4),
        "base_compression": round(base_compression, 4),
        "left_pre_days": left_pre,
        "right_pre_days": right_pre,
        "time_map_src": src,
        "time_map_dst": dst,
        "post_right_days": post_right,
        "segment_time_compression_current_div_history": [round(s, 4) for s in segment_scales],
    }
    return plot_df, anchors, stats


def write_html(plot_df: pd.DataFrame, anchors: pd.DataFrame, stats: dict) -> None:
    payload = plot_df.copy()
    payload["date"] = payload["date"].dt.date.astype(str)
    data_json = json.dumps(payload.to_dict(orient="records"), ensure_ascii=False)
    anchor_json = json.dumps(anchors.to_dict(orient="records"), ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Annealed Time Warp v13</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ padding:12px 18px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 430px; gap:12px; padding:12px; }}
.panel {{ background:#242933; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 82px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 82px); overflow:auto; }}
.item {{ padding:8px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aeb8c8; }}
.strong {{ color:#fff; font-weight:700; }}
.green {{ color:#00e676; }}
.red {{ color:#ff6a5f; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
</style>
</head>
<body>
<header><h1>v13 严格时间压缩：所有比例 = 当前间隔 / 历史间隔，必须 ≤ 1</h1></header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const DATA = {data_json};
const ANCHORS = {anchor_json};
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
  const min=-STATS.right_pre_days, max=STATS.post_right_days;
  return M.l + (d.rel_day - min) / (max - min) * (W - M.l - M.r);
}}
function y(d) {{
  const min=-0.9, max=0.16;
  return H - M.b - (d.log_norm - min) / (max - min) * (H - M.t - M.b);
}}
function color(cycle) {{ return cycle === "2025_actual" ? "#00e676" : "#ff6a5f"; }}
for (let i=0;i<9;i++) {{
  const gx = M.l + i/8*(W-M.l-M.r);
  el("line", {{x1:gx,y1:M.t,x2:gx,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
}}
for (let i=0;i<7;i++) {{
  const gy = M.t + i/6*(H-M.t-M.b);
  el("line", {{x1:M.l,y1:gy,x2:W-M.r,y2:gy,stroke:"rgba(255,255,255,.08)"}});
}}
el("line", {{x1:x({{rel_day:0}}),y1:M.t,x2:x({{rel_day:0}}),y2:H-M.b,stroke:"rgba(255,255,255,.55)","stroke-width":1}});
["2021_warped_annealed","2025_actual"].forEach(cycle => {{
  const rows = DATA.filter(d => d.cycle === cycle).sort((a,b)=>a.rel_day-b.rel_day);
  let path = "";
  rows.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
  el("path", {{d:path, fill:"none", stroke:color(cycle), "stroke-width":cycle==="2025_actual"?2.4:2.1, opacity:cycle==="2025_actual"?.95:.78}});
}});
ANCHORS.forEach(a => {{
  const oldPoint = {{rel_day:a.right_rel_day, log_norm:Math.log(a.left_norm)*STATS.amp_scale}};
  const newPoint = {{rel_day:a.right_rel_day, log_norm:Math.log(a.right_norm)}};
  [oldPoint,newPoint].forEach((p,i) => {{
    const c = el("circle", {{cx:x(p),cy:y(p),r:6,fill:i===0?"#ff6a5f":"#00e676",stroke:"#120807","stroke-width":1.4}});
    c.addEventListener("mousemove", ev => {{
      tip.style.display = "block";
      tip.style.left = ev.clientX + 12 + "px";
      tip.style.top = ev.clientY + 12 + "px";
      tip.innerHTML = `<b>${{i===0?a.left_label:a.right_label}}</b><br>mapped rel ${{a.right_rel_day}}d`;
    }});
    c.addEventListener("mouseleave", () => tip.style.display = "none");
  }});
}});
document.getElementById("summary").innerHTML = `
  <div class="strong">图例</div>
  <div><span class="red">红</span>：2021 路径，已退火 + 压缩</div>
  <div><span class="green">绿</span>：2025 实际路径</div>
  <br>
  <div>幅度退火系数：${{STATS.amp_scale}}</div>
  <div>基础时间压缩：${{STATS.base_compression}}</div>
  <div>历史顶前窗口：${{STATS.left_pre_days}} 天 → 当前顶前窗口：${{STATS.right_pre_days}} 天</div>
  <div>各段时间压缩：${{STATS.segment_time_compression_current_div_history.join(" / ")}}</div>
  <br>
`;
document.getElementById("list").innerHTML = ANCHORS.map(a => `
  <div class="item">
    <div class="strong">${{a.role}}</div>
    <div>${{a.left_label}} ${{a.left_date}} → ${{a.right_label}} ${{a.right_date}}</div>
    <div>rel: ${{a.left_rel_day}}d → ${{a.right_rel_day}}d · norm: ${{a.left_norm}} → ${{a.right_norm}}</div>
  </div>
`).join("");
</script>
</body>
</html>"""
    (OUT / "green_anchor_annealed_warp_v13.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_df, anchors, stats = build(load_price())
    plot_df.to_csv(OUT / "annealed_warp_v13.csv", index=False, encoding="utf-8-sig")
    anchors.to_csv(OUT / "annealed_warp_anchors_v13.csv", index=False, encoding="utf-8-sig")
    write_html(plot_df, anchors, stats)
    print(stats)


if __name__ == "__main__":
    main()
