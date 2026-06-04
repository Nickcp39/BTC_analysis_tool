from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

# Seeded from the user's annotated 5Y screenshot.
# These are the nearest available v5 segment nodes for the six visible correspondences.
LEFT_SIX = [
    "2021-11-10",  # prior cycle peak
    "2022-01-23",  # first major drop end
    "2022-04-03",  # rebound end
    "2022-06-18",  # capitulation low
    "2022-08-14",  # rebound end
    "2022-11-22",  # final bear low
]
RIGHT_SIX = [
    "2025-10-05",  # current cycle peak
    "2025-11-22",  # first major drop end
    "2026-01-15",  # rebound end
    "2026-02-22",  # local low
    "2026-05-08",  # rebound end
    "2026-05-31",  # latest visible low/end point
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def nearest_point(series: pd.Series, date_text: str) -> dict:
    target = pd.Timestamp(date_text)
    if target in series.index:
        d = target
    else:
        d = series.index[(series.index - target).days.astype("int64").argmin()]
    return {"date": d.date().isoformat(), "price": round(float(series.loc[d]), 2)}


def write_html(series: pd.Series) -> None:
    latest = series.index.max()
    start = latest - pd.Timedelta(days=365 * 5)
    seg = series.loc[start:latest].copy()
    prices = [[d.date().isoformat(), round(float(v), 2)] for d, v in seg.items()]
    left = [nearest_point(series, d) for d in LEFT_SIX]
    right = [nearest_point(series, d) for d in RIGHT_SIX]
    pairs = []
    for i, (a, b) in enumerate(zip(left, right), start=1):
        da, db = pd.Timestamp(a["date"]), pd.Timestamp(b["date"])
        pairs.append(
            {
                "index": i,
                "left_date": a["date"],
                "left_price": a["price"],
                "right_date": b["date"],
                "right_price": b["price"],
                "span_days": int((db - da).days),
                "left_interval": None if i == 1 else int((da - pd.Timestamp(left[i - 2]["date"])).days),
                "right_interval": None if i == 1 else int((db - pd.Timestamp(right[i - 2]["date"])).days),
            }
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Six Point Correspondence v9</title>
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
.green {{ color:#00e676; }}
.blue {{ color:#73d0ff; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
</style>
</head>
<body>
<header><h1>你画的 6 个点对应 v9：左侧结构 → 右侧结构</h1></header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const PRICES = {json.dumps(prices, ensure_ascii=False)}.map(d => ({{date:new Date(d[0]), dateText:d[0], price:+d[1]}}));
const PAIRS = {json.dumps(pairs, ensure_ascii=False)};
const POINTS = PAIRS.flatMap(p => [
  {{side:"left", index:p.index, date:new Date(p.left_date), dateText:p.left_date, price:p.left_price}},
  {{side:"right", index:p.index, date:new Date(p.right_date), dateText:p.right_date, price:p.right_price}}
]);
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
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
PRICES.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
el("path", {{d:path, fill:"none", stroke:"#ff8177", "stroke-width":2}});

function pointOf(side, index) {{ return POINTS.find(p => p.side === side && p.index === index); }}
PAIRS.forEach(p => {{
  const a = pointOf("left", p.index), b = pointOf("right", p.index);
  el("line", {{x1:x(a),y1:y(a),x2:x(b),y2:y(b),stroke:"rgba(0,230,118,.48)","stroke-width":1.6}});
}});
["left","right"].forEach(side => {{
  let zz = "";
  POINTS.filter(p => p.side === side).forEach((p,i) => zz += (i ? "L" : "M") + x(p).toFixed(2) + "," + y(p).toFixed(2));
  el("path", {{d:zz, fill:"none", stroke:side==="left" ? "rgba(115,208,255,.8)" : "rgba(0,230,118,.8)", "stroke-width":2, "stroke-dasharray":"6 5"}});
}});
POINTS.forEach(p => {{
  const c = el("circle", {{cx:x(p),cy:y(p),r:7,fill:p.side==="left" ? "#73d0ff" : "#00e676",stroke:"#06140c","stroke-width":1.6}});
  c.addEventListener("mousemove", ev => {{
    tip.style.display = "block";
    tip.style.left = ev.clientX + 12 + "px";
    tip.style.top = ev.clientY + 12 + "px";
    tip.innerHTML = `<b>${{p.side}} #${{p.index}}</b><br>${{p.dateText}}<br>${{money(p.price)}}`;
  }});
  c.addEventListener("mouseleave", () => tip.style.display = "none");
  const t = el("text", {{x:x(p)+9,y:y(p)-9,fill:"#fff","font-size":13,"font-weight":"700"}});
  t.textContent = p.index;
}});
const avgSpan = Math.round(PAIRS.reduce((s,p)=>s+p.span_days,0)/PAIRS.length);
document.getElementById("summary").innerHTML = `<div class="strong">6 点对应</div><div>平均跨度：${{avgSpan}} 天</div><br>`;
document.getElementById("list").innerHTML = PAIRS.map(p => `
  <div class="item">
    <div class="strong">#${{p.index}} · 跨度 ${{p.span_days}} 天</div>
    <div><span class="blue">${{p.left_date}}</span> → <span class="green">${{p.right_date}}</span></div>
    <div>左间隔 ${{p.left_interval || "-"}} 天 · 右间隔 ${{p.right_interval || "-"}} 天</div>
  </div>
`).join("");
</script>
</body>
</html>"""
    (OUT / "green_anchor_manual_six_v9.html").write_text(html, encoding="utf-8")
    pd.DataFrame(pairs).to_csv(OUT / "manual_six_pairs_v9.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_html(load_price())
    print(f"Wrote {OUT / 'green_anchor_manual_six_v9.html'}")


if __name__ == "__main__":
    main()
