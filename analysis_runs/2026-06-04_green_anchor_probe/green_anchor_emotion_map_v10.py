from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

POINTS = [
    {"id": 1, "side": "left", "role": "top", "label": "①顶 2021", "date": "2021-11-08"},
    {"id": 2, "side": "left", "role": "lower_high", "label": "②次高", "date": "2022-03-29"},
    {"id": 3, "side": "left", "role": "bottom", "label": "③底", "date": "2022-11-20"},
    {"id": 4, "side": "right", "role": "top", "label": "④顶 2025", "date": "2025-10-05"},
    {"id": 5, "side": "right", "role": "lower_high", "label": "⑤次高", "date": "2025-11-10"},
    {"id": 6, "side": "right", "role": "bottom", "label": "⑥底", "date": "2026-02-05"},
]

PAIRS = [(1, 4), (2, 5), (3, 6)]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def nearest(series: pd.Series, date_text: str) -> tuple[str, float]:
    target = pd.Timestamp(date_text)
    if target in series.index:
        d = target
    else:
        distances = abs((series.index - target).days)
        d = series.index[int(distances.argmin())]
    return d.date().isoformat(), float(series.loc[d])


def build(series: pd.Series) -> tuple[list[dict], list[dict], dict]:
    pts = []
    for point in POINTS:
        d, price = nearest(series, point["date"])
        pts.append({**point, "date": d, "price": round(price, 2)})

    by_id = {p["id"]: p for p in pts}
    pairs = []
    for left_id, right_id in PAIRS:
        a, b = by_id[left_id], by_id[right_id]
        pairs.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "role": a["role"],
                "left_date": a["date"],
                "right_date": b["date"],
                "span_days": (pd.Timestamp(b["date"]) - pd.Timestamp(a["date"])).days,
                "left_price": a["price"],
                "right_price": b["price"],
                "price_ratio": round(b["price"] / a["price"], 4),
            }
        )

    left = [p for p in pts if p["side"] == "left"]
    right = [p for p in pts if p["side"] == "right"]
    left_intervals = [
        (pd.Timestamp(left[i]["date"]) - pd.Timestamp(left[i - 1]["date"])).days
        for i in range(1, len(left))
    ]
    right_intervals = [
        (pd.Timestamp(right[i]["date"]) - pd.Timestamp(right[i - 1]["date"])).days
        for i in range(1, len(right))
    ]
    compression = [
        round(right_intervals[i] / left_intervals[i], 4)
        for i in range(min(len(left_intervals), len(right_intervals)))
    ]
    stats = {
        "left_intervals": left_intervals,
        "right_intervals": right_intervals,
        "compression": compression,
        "avg_span_days": round(sum(p["span_days"] for p in pairs) / len(pairs), 1),
        "avg_compression": round(sum(compression) / len(compression), 4),
    }
    return pts, pairs, stats


def write_html(series: pd.Series, points: list[dict], pairs: list[dict], stats: dict) -> None:
    start = pd.Timestamp("2021-06-01")
    end = series.index.max()
    seg = series.loc[start:end]
    prices = [[d.date().isoformat(), round(float(v), 2)] for d, v in seg.items()]
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Emotion Map v10</title>
<style>
body {{ margin:0; background:#20242c; color:#eef3fb; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ padding:12px 18px; background:#1c2129; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ margin:0; font-size:16px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 390px; gap:12px; padding:12px; }}
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
<header><h1>BTC 顶后情绪结构对应 v10：2021 顶后 → 2025 顶后</h1></header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const PRICES = {json.dumps(prices, ensure_ascii=False)}.map(d => ({{date:new Date(d[0]), dateText:d[0], price:+d[1]}}));
const POINTS = {json.dumps(points, ensure_ascii=False)}.map(d => ({{...d, jsDate:new Date(d.date)}}));
const PAIRS = {json.dumps(pairs, ensure_ascii=False)};
const STATS = {json.dumps(stats, ensure_ascii=False)};
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const W = 1200, H = 720, M = {{l:76,r:28,t:30,b:48}};
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function money(v) {{ return "$" + Math.round(v).toLocaleString(); }}
const xs = PRICES.map(d => d.date.getTime()), ys = PRICES.map(d => d.price);
const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
const x = d => M.l + (d.jsDate.getTime() - xmin) / (xmax - xmin) * (W - M.l - M.r);
const xp = d => M.l + (d.date.getTime() - xmin) / (xmax - xmin) * (W - M.l - M.r);
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
PRICES.forEach((d,i) => path += (i ? "L" : "M") + xp(d).toFixed(2) + "," + y(d).toFixed(2));
el("path", {{d:path, fill:"none", stroke:"rgba(210,215,220,.58)", "stroke-width":2}});

function byId(id) {{ return POINTS.find(p => p.id === id); }}
PAIRS.forEach(p => {{
  const a = byId(p.left_id), b = byId(p.right_id);
  el("line", {{x1:x(a),y1:y(a),x2:x(b),y2:y(b),stroke:"rgba(0,230,118,.5)","stroke-width":1.6}});
}});
["left","right"].forEach(side => {{
  const pts = POINTS.filter(p => p.side === side);
  let pth = "";
  pts.forEach((p,i) => pth += (i ? "L" : "M") + x(p).toFixed(2) + "," + y(p).toFixed(2));
  el("path", {{d:pth, fill:"none", stroke:side==="left" ? "#ff5a4f" : "#00e676", "stroke-width":2.2, "stroke-dasharray":"6 5"}});
}});
POINTS.forEach(p => {{
  const fill = p.role === "bottom" ? "#00c853" : "#ef4f43";
  const c = el("circle", {{cx:x(p),cy:y(p),r:7,fill,stroke:"#140806","stroke-width":1.6}});
  c.addEventListener("mousemove", ev => {{
    tip.style.display = "block";
    tip.style.left = ev.clientX + 12 + "px";
    tip.style.top = ev.clientY + 12 + "px";
    tip.innerHTML = `<b>${{p.label}}</b><br>${{p.date}}<br>${{money(p.price)}}`;
  }});
  c.addEventListener("mouseleave", () => tip.style.display = "none");
  const t = el("text", {{x:x(p)+9,y:y(p)-10,fill:p.role==="bottom" ? "#00e676" : "#ff5a4f","font-size":13,"font-weight":"700"}});
  t.textContent = p.label;
}});
document.getElementById("summary").innerHTML = `
  <div class="strong">3 组对应</div>
  <div>平均跨周期跨度：${{STATS.avg_span_days}} 天</div>
  <div>左侧间隔：${{STATS.left_intervals.join(" / ")}} 天</div>
  <div>右侧间隔：${{STATS.right_intervals.join(" / ")}} 天</div>
  <div>顶后结构压缩：${{STATS.compression.join(" / ")}}</div>
  <div>平均压缩：${{STATS.avg_compression}}</div>
  <br>
`;
document.getElementById("list").innerHTML = PAIRS.map(p => `
  <div class="item">
    <div class="strong">${{p.role}} · 跨度 ${{p.span_days}} 天 · 价格倍率 ${{p.price_ratio}}</div>
    <div><span class="red">${{p.left_date}}</span> → <span class="green">${{p.right_date}}</span></div>
  </div>
`).join("");
</script>
</body>
</html>"""
    (OUT / "green_anchor_emotion_map_v10.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    points, pairs, stats = build(series)
    pd.DataFrame(points).to_csv(OUT / "emotion_points_v10.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(OUT / "emotion_pairs_v10.csv", index=False, encoding="utf-8-sig")
    write_html(series, points, pairs, stats)
    print(stats)


if __name__ == "__main__":
    main()
