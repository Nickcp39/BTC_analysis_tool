from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

HALVINGS = {
    "2017": pd.Timestamp("2016-07-09"),
    "2021": pd.Timestamp("2020-05-11"),
    "2025": pd.Timestamp("2024-04-20"),
}
NEXT_HALVINGS = {
    "2017": pd.Timestamp("2020-05-11"),
    "2021": pd.Timestamp("2024-04-20"),
    "2025": None,
}


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def terminal_low_candidates(series: pd.Series, cycle: str) -> pd.DataFrame:
    start = HALVINGS[cycle]
    end = NEXT_HALVINGS[cycle] or series.index.max()
    s = series.loc[start:end].copy()
    logp = np.log(s)
    smooth = logp.rolling(5, center=True, min_periods=2).mean().dropna()

    trough_idx, _ = find_peaks(-smooth.values, distance=6, prominence=0.012)
    rows = []

    # Different window sizes catch the same visual pattern at different scales.
    window_sets = [
        (10, 12, 0.055, 0.045),
        (18, 22, 0.075, 0.065),
        (32, 38, 0.105, 0.085),
        (55, 65, 0.145, 0.115),
        (90, 105, 0.19, 0.145),
    ]

    for idx in trough_idx:
        d = smooth.index[int(idx)].normalize()
        if d not in s.index:
            continue
        low_px = float(s.loc[d])
        low_log = float(logp.loc[d])

        for left_days, right_days, min_drop, min_rally in window_sets:
            left_start = max(s.index.min(), d - pd.Timedelta(days=left_days * 3))
            right_end = min(s.index.max(), d + pd.Timedelta(days=right_days * 3))
            left = s.loc[left_start:d]
            right = s.loc[d:right_end]
            if len(left) < left_days or len(right) < max(5, right_days // 2):
                continue

            left_peak_date = left.idxmax().normalize()
            left_peak_px = float(left.loc[left_peak_date])
            right_peak_date = right.idxmax().normalize()
            right_peak_px = float(right.loc[right_peak_date])
            if left_peak_date >= d or right_peak_date <= d:
                continue

            drop = low_px / left_peak_px - 1.0
            rally = right_peak_px / low_px - 1.0
            if abs(min(drop, 0.0)) < min_drop or max(rally, 0.0) < min_rally:
                continue

            left_len = max((d - left_peak_date).days, 1)
            right_len = max((right_peak_date - d).days, 1)
            left_slope = (low_log - float(logp.loc[left_peak_date])) / left_len
            right_slope = (float(logp.loc[right_peak_date]) - low_log) / right_len
            slope_flip = right_slope - left_slope

            # The drawn pattern: left side should be mostly falling into the low,
            # and the first right side should quickly get away from the low.
            left_tail = smooth.loc[max(smooth.index.min(), d - pd.Timedelta(days=left_days)):d]
            right_head = smooth.loc[d:min(smooth.index.max(), d + pd.Timedelta(days=right_days))]
            left_down_ratio = float((left_tail.diff().dropna() < 0).mean()) if len(left_tail) > 3 else 0.0
            right_up_ratio = float((right_head.diff().dropna() > 0).mean()) if len(right_head) > 3 else 0.0

            local_band = s.loc[
                max(s.index.min(), d - pd.Timedelta(days=max(8, left_days // 2))) :
                min(s.index.max(), d + pd.Timedelta(days=max(8, right_days // 2)))
            ]
            is_local_floor = low_px <= float(local_band.quantile(0.18))
            if not is_local_floor:
                continue

            score = (
                abs(drop)
                * max(rally, 0.0)
                * max(slope_flip, 0.0001)
                * (0.6 + left_down_ratio)
                * (0.6 + right_up_ratio)
                * 10000.0
            )
            rows.append(
                {
                    "cycle": cycle,
                    "date": d,
                    "price": low_px,
                    "left_peak_date": left_peak_date,
                    "left_peak_price": left_peak_px,
                    "right_peak_date": right_peak_date,
                    "right_peak_price": right_peak_px,
                    "drop_pct": drop * 100.0,
                    "rally_pct": rally * 100.0,
                    "left_days": left_len,
                    "right_days": right_len,
                    "left_down_ratio": left_down_ratio,
                    "right_up_ratio": right_up_ratio,
                    "slope_flip": slope_flip,
                    "scale_left": left_days,
                    "score": score,
                }
            )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return merge_candidates(df)


def merge_candidates(df: pd.DataFrame, min_gap: int = 14) -> pd.DataFrame:
    kept = []
    for _, row in df.iterrows():
        d = row["date"]
        if any(abs((d - old["date"]).days) < min_gap for old in kept):
            continue
        kept.append(row.to_dict())
    out = pd.DataFrame(kept).sort_values("date").reset_index(drop=True)
    return out


def build_table(series: pd.Series) -> pd.DataFrame:
    parts = []
    for cycle in ["2017", "2021", "2025"]:
        g = terminal_low_candidates(series, cycle)
        if g.empty:
            continue
        g = g.copy()
        g["anchor_index"] = np.arange(1, len(g) + 1)
        g["days_from_halving"] = (g["date"] - HALVINGS[cycle]).dt.days
        g["interval_from_prev_days"] = g["date"].diff().dt.days
        parts.append(g)
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    cols = [
        "cycle",
        "anchor_index",
        "date",
        "days_from_halving",
        "price",
        "drop_pct",
        "rally_pct",
        "left_peak_date",
        "right_peak_date",
        "left_days",
        "right_days",
        "left_down_ratio",
        "right_up_ratio",
        "slope_flip",
        "score",
        "interval_from_prev_days",
    ]
    table = table[cols].copy()
    for col in ["price", "drop_pct", "rally_pct", "left_down_ratio", "right_up_ratio", "slope_flip", "score"]:
        table[col] = table[col].astype(float).round(4)
    table["date"] = table["date"].dt.date.astype(str)
    table["left_peak_date"] = table["left_peak_date"].dt.date.astype(str)
    table["right_peak_date"] = table["right_peak_date"].dt.date.astype(str)
    return table


def build_html(series: pd.Series, table: pd.DataFrame) -> None:
    payload = []
    for cycle in ["2017", "2021", "2025"]:
        start = HALVINGS[cycle]
        end = NEXT_HALVINGS[cycle] or series.index.max()
        seg = series.loc[start:end]
        anchors = table[table["cycle"] == cycle].to_dict(orient="records")
        payload.append(
            {
                "cycle": cycle,
                "prices": [[d.date().isoformat(), round(float(v), 2)] for d, v in seg.items()],
                "anchors": anchors,
            }
        )
    data_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Waterfall Terminal Anchors</title>
<style>
body {{ margin:0; background:#171b22; color:#e8edf5; font-family:Arial,"Microsoft YaHei",sans-serif; }}
header {{ display:flex; gap:10px; align-items:center; padding:12px 16px; background:#1f2530; border-bottom:1px solid rgba(255,255,255,.1); }}
h1 {{ font-size:16px; margin:0 14px 0 0; }}
button {{ color:#dce7f7; background:#2b3442; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:7px 11px; cursor:pointer; }}
button.active {{ background:#315489; border-color:#9bbfff; }}
main {{ display:grid; grid-template-columns:1fr 340px; gap:12px; padding:12px; }}
.panel {{ background:#202631; border:1px solid rgba(255,255,255,.08); border-radius:8px; }}
#chart {{ width:100%; height:calc(100vh - 86px); display:block; }}
aside {{ padding:12px; max-height:calc(100vh - 86px); overflow:auto; }}
.item {{ display:grid; grid-template-columns:38px 1fr; gap:8px; padding:7px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:12px; color:#aab6c7; }}
.idx {{ color:#00e676; font-weight:700; }}
.date {{ color:#fff; font-weight:700; }}
.tip {{ position:fixed; display:none; pointer-events:none; background:rgba(3,6,10,.94); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.45; z-index:10; }}
@media (max-width:900px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>瀑布下跌终点 / 大涨启动点 v4</h1>
  <button data-cycle="2017">2017</button>
  <button data-cycle="2021">2021</button>
  <button class="active" data-cycle="2025">2025</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const DATA = {data_json};
let active = "2025";
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const NS = "http://www.w3.org/2000/svg";
const W = 1200, H = 720, M = {{l:76,r:28,t:28,b:46}};
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function clear() {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}
function xScale(t, min, max) {{ return M.l + (t - min) / (max - min) * (W - M.l - M.r); }}
function yScale(v, min, max) {{
  const a = Math.log(v), lo = Math.log(min), hi = Math.log(max);
  return H - M.b - (a - lo) / (hi - lo) * (H - M.t - M.b);
}}
function money(v) {{ return "$" + Math.round(v).toLocaleString(); }}
function render() {{
  clear();
  const d = DATA.find(x => x.cycle === active);
  const prices = d.prices.map(p => [new Date(p[0]).getTime(), +p[1], p[0]]);
  const xs = prices.map(p => p[0]), ys = prices.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  for (let i=0;i<8;i++) {{
    const x = M.l + i/7*(W-M.l-M.r);
    el("line", {{x1:x,y1:M.t,x2:x,y2:H-M.b,stroke:"rgba(255,255,255,.08)"}});
  }}
  for (let i=0;i<7;i++) {{
    const y = M.t + i/6*(H-M.t-M.b);
    el("line", {{x1:M.l,y1:y,x2:W-M.r,y2:y,stroke:"rgba(255,255,255,.08)"}});
  }}
  let path = "";
  prices.forEach((p,i) => {{
    const x = xScale(p[0], xmin, xmax), y = yScale(p[1], ymin, ymax);
    path += (i ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
  }});
  el("path", {{d:path, fill:"none", stroke:"#ff8177", "stroke-width":2}});
  const anchors = d.anchors;
  for (let i=1;i<anchors.length;i++) {{
    const a = anchors[i-1], b = anchors[i];
    const x1 = xScale(new Date(a.date).getTime(), xmin, xmax);
    const x2 = xScale(new Date(b.date).getTime(), xmin, xmax);
    const y = H - 18;
    el("line", {{x1,y1:y,x2,y2:y,stroke:"#00e676","stroke-width":1.3}});
    const tx = el("text", {{x:(x1+x2)/2,y:y-5,fill:"#dfffea","font-size":11,"text-anchor":"middle"}});
    tx.textContent = Math.round(b.interval_from_prev_days) + "d";
  }}
  anchors.forEach(a => {{
    const x = xScale(new Date(a.date).getTime(), xmin, xmax);
    const y = yScale(+a.price, ymin, ymax);
    const c = el("circle", {{cx:x,cy:y,r:6,fill:"#00e676",stroke:"#06140c","stroke-width":1.5}});
    c.addEventListener("mousemove", ev => {{
      tip.style.display = "block";
      tip.style.left = ev.clientX + 12 + "px";
      tip.style.top = ev.clientY + 12 + "px";
      tip.innerHTML = `<b>#${{a.anchor_index}} ${{a.date}}</b><br>${{money(a.price)}}<br>左跌 ${{a.drop_pct}}% / ${{a.left_days}}d<br>右涨 +${{a.rally_pct}}% / ${{a.right_days}}d<br>间隔 ${{a.interval_from_prev_days || "-"}} 天`;
    }});
    c.addEventListener("mouseleave", () => tip.style.display = "none");
    const t = el("text", {{x:x+8,y:y-8,fill:"#fff","font-size":12}});
    t.textContent = a.anchor_index;
  }});
  document.getElementById("summary").innerHTML = `<b>${{active}}</b>：${{anchors.length}} 个瀑布终点/启动点`;
  document.getElementById("list").innerHTML = anchors.map(a => `
    <div class="item"><div class="idx">#${{a.anchor_index}}</div><div>
      <div class="date">${{a.date}} · ${{money(a.price)}}</div>
      <div>左跌 ${{a.drop_pct}}% / ${{a.left_days}}d · 右涨 +${{a.rally_pct}}% / ${{a.right_days}}d · 间隔 ${{a.interval_from_prev_days || "-"}}d</div>
    </div></div>`).join("");
}}
document.querySelectorAll("button").forEach(b => b.onclick = () => {{
  active = b.dataset.cycle;
  document.querySelectorAll("button").forEach(x => x.classList.toggle("active", x === b));
  render();
}});
render();
</script>
</body>
</html>"""
    (OUT / "green_anchor_waterfall_v4.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    table = build_table(series)
    table.to_csv(OUT / "waterfall_anchors_v4.csv", index=False, encoding="utf-8-sig")
    build_html(series, table)


if __name__ == "__main__":
    main()
