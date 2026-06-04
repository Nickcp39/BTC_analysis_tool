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


def cluster_dates(events: list[dict], tolerance: int = 7) -> list[dict]:
    events = sorted(events, key=lambda x: x["date"])
    clusters: list[list[dict]] = []
    for event in events:
        if not clusters:
            clusters.append([event])
            continue
        center = pd.Timestamp(np.median([e["date"].value for e in clusters[-1]]))
        if abs((event["date"] - center).days) <= tolerance:
            clusters[-1].append(event)
        else:
            clusters.append([event])

    out = []
    for cluster in clusters:
        center = pd.Timestamp(np.median([e["date"].value for e in cluster])).normalize()
        best = max(cluster, key=lambda e: e["score"])
        out.append({**best, "date": center, "votes": len(cluster)})
    return out


def local_impulse_anchors(series: pd.Series, cycle: str) -> list[dict]:
    start = HALVINGS[cycle]
    end = NEXT_HALVINGS[cycle] or series.index.max()
    seg = series.loc[start:end].copy()
    logp = np.log(seg)

    events: list[dict] = []
    # Shorter windows capture the local "solutions" inside each big move.
    for smooth, lookback, lookahead, min_move in [
        (3, 12, 18, 0.075),
        (5, 18, 24, 0.095),
        (9, 28, 36, 0.125),
        (14, 42, 55, 0.16),
    ]:
        y = logp.rolling(smooth, center=True, min_periods=2).mean().dropna()
        troughs, _ = find_peaks(-y.values, distance=max(5, smooth * 2), prominence=0.025)
        peaks, _ = find_peaks(y.values, distance=max(5, smooth * 2), prominence=0.025)
        peak_dates = list(y.index[peaks])

        for idx in troughs:
            d = y.index[int(idx)].normalize()
            px = float(seg.loc[d])
            prev_window = seg.loc[max(seg.index.min(), d - pd.Timedelta(days=lookback * 3)) : d]
            next_window = seg.loc[d : min(seg.index.max(), d + pd.Timedelta(days=lookahead * 3))]
            if len(prev_window) < 3 or len(next_window) < 3:
                continue

            prev_peak_px = float(prev_window.max())
            prev_peak_date = prev_window.idxmax().normalize()
            next_peak_px = float(next_window.max())
            next_peak_date = next_window.idxmax().normalize()
            drop = px / prev_peak_px - 1.0
            rally = next_peak_px / px - 1.0

            # Big-drop local solution OR big-rally local start.
            if abs(min(drop, 0.0)) < min_move and max(rally, 0.0) < min_move:
                continue
            if prev_peak_date == d and next_peak_date == d:
                continue

            # Prefer troughs that sit between a preceding high and a later high.
            bracket_bonus = 0.08 if prev_peak_date < d < next_peak_date else 0.0
            score = abs(min(drop, 0.0)) + max(rally, 0.0) + bracket_bonus
            events.append(
                {
                    "cycle": cycle,
                    "date": d,
                    "price": px,
                    "drop_pct": drop * 100.0,
                    "rally_pct": rally * 100.0,
                    "prev_peak_date": prev_peak_date,
                    "next_peak_date": next_peak_date,
                    "score": score,
                    "scale": smooth,
                }
            )

    clustered = cluster_dates(events)
    # Keep many local solutions, but remove tiny duplicates.
    clustered = [e for e in clustered if e["votes"] >= 1]
    clustered = sorted(clustered, key=lambda e: e["date"])

    kept: list[dict] = []
    for event in clustered:
        if kept and (event["date"] - kept[-1]["date"]).days < 18:
            if event["score"] > kept[-1]["score"]:
                kept[-1] = event
            continue
        kept.append(event)
    return kept


def build_tables(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cycle in ["2017", "2021", "2025"]:
        anchors = local_impulse_anchors(series, cycle)
        for i, a in enumerate(anchors):
            prev_anchor = anchors[i - 1]["date"] if i else None
            rows.append(
                {
                    "cycle": cycle,
                    "anchor_index": i + 1,
                    "date": a["date"].date().isoformat(),
                    "days_from_halving": (a["date"] - HALVINGS[cycle]).days,
                    "price": round(a["price"], 2),
                    "drop_pct": round(a["drop_pct"], 2),
                    "rally_pct": round(a["rally_pct"], 2),
                    "prev_peak_date": a["prev_peak_date"].date().isoformat(),
                    "next_peak_date": a["next_peak_date"].date().isoformat(),
                    "interval_from_prev_days": None if prev_anchor is None else (a["date"] - prev_anchor).days,
                    "score": round(a["score"], 4),
                    "votes": a["votes"],
                    "scale": a["scale"],
                }
            )
    table = pd.DataFrame(rows)

    intervals = []
    for cycle, g in table.groupby("cycle"):
        span = max(g["days_from_halving"].max() - g["days_from_halving"].min(), 1)
        for _, row in g.dropna(subset=["interval_from_prev_days"]).iterrows():
            intervals.append(
                {
                    "cycle": cycle,
                    "to_anchor_index": int(row["anchor_index"]),
                    "interval_days": int(row["interval_from_prev_days"]),
                    "interval_div_span": round(float(row["interval_from_prev_days"] / span), 4),
                }
            )
    return table, pd.DataFrame(intervals)


def html_payload(series: pd.Series, table: pd.DataFrame) -> str:
    cycles = []
    for cycle in ["2017", "2021", "2025"]:
        start = HALVINGS[cycle]
        end = NEXT_HALVINGS[cycle] or series.index.max()
        seg = series.loc[start:end]
        g = table[table["cycle"] == cycle].copy()
        cycles.append(
            {
                "cycle": cycle,
                "prices": [
                    {"date": d.date().isoformat(), "price": round(float(v), 2)}
                    for d, v in seg.items()
                ],
                "anchors": g.to_dict(orient="records"),
            }
        )
    return json.dumps(cycles, ensure_ascii=False)


def write_html(series: pd.Series, table: pd.DataFrame) -> None:
    payload = html_payload(series, table)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BTC Local Impulse Anchors</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #171b22;
      --panel: #20252e;
      --grid: rgba(255,255,255,.12);
      --text: #e8edf5;
      --muted: #9aa7b8;
      --line: #ff8177;
      --green: #00e676;
      --cyan: #21b7ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, "Microsoft YaHei", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 12px 18px;
      background: rgba(23,27,34,.94);
      border-bottom: 1px solid rgba(255,255,255,.09);
    }}
    h1 {{
      margin: 0;
      font-size: 17px;
      font-weight: 650;
    }}
    button {{
      border: 1px solid rgba(255,255,255,.16);
      background: #29313d;
      color: var(--text);
      padding: 7px 10px;
      border-radius: 6px;
      cursor: pointer;
    }}
    button.active {{
      border-color: #8bb7ff;
      background: #315489;
    }}
    main {{
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      gap: 14px;
    }}
    .chart-wrap, aside {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,.08);
      border-radius: 8px;
    }}
    .chart-wrap {{
      min-height: calc(100vh - 88px);
      padding: 10px;
    }}
    svg {{
      width: 100%;
      height: calc(100vh - 112px);
      display: block;
    }}
    aside {{
      max-height: calc(100vh - 88px);
      overflow: auto;
      padding: 12px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 34px 1fr;
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid rgba(255,255,255,.07);
      font-size: 12px;
      color: var(--muted);
    }}
    .idx {{
      color: var(--green);
      font-weight: 700;
    }}
    .date {{
      color: var(--text);
      font-weight: 650;
    }}
    .axis text {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .axis path, .axis line {{
      stroke: rgba(255,255,255,.18);
    }}
    .grid line {{
      stroke: var(--grid);
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      padding: 8px 10px;
      border-radius: 6px;
      background: rgba(5,8,12,.92);
      border: 1px solid rgba(255,255,255,.14);
      color: var(--text);
      font-size: 12px;
      display: none;
      z-index: 10;
      line-height: 1.45;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>BTC 大涨/大跌局部解锚点</h1>
    <button data-cycle="2017">2017</button>
    <button data-cycle="2021">2021</button>
    <button data-cycle="2025" class="active">2025</button>
  </header>
  <main>
    <section class="chart-wrap">
      <svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg>
    </section>
    <aside>
      <div id="summary"></div>
      <div id="list"></div>
    </aside>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    const cycles = {payload};
    let active = "2025";
    const svg = d3.select("#chart");
    const tooltip = d3.select("#tooltip");
    const W = 1200, H = 720;
    const margin = {{ top: 28, right: 34, bottom: 44, left: 76 }};

    function fmtMoney(v) {{
      return "$" + Math.round(v).toLocaleString();
    }}

    function render() {{
      const data = cycles.find(d => d.cycle === active);
      const prices = data.prices.map(d => ({{ date: new Date(d.date), price: +d.price }}));
      const anchors = data.anchors.map(d => ({{ ...d, jsDate: new Date(d.date) }}));
      svg.selectAll("*").remove();

      const x = d3.scaleTime()
        .domain(d3.extent(prices, d => d.date))
        .range([margin.left, W - margin.right]);
      const y = d3.scaleLog()
        .domain(d3.extent(prices, d => d.price)).nice()
        .range([H - margin.bottom, margin.top]);

      svg.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0,${{H - margin.bottom}})`)
        .call(d3.axisBottom(x).ticks(8).tickSize(-(H - margin.top - margin.bottom)).tickFormat(""));
      svg.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(${{margin.left}},0)`)
        .call(d3.axisLeft(y).ticks(7, "~s").tickSize(-(W - margin.left - margin.right)).tickFormat(""));

      svg.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${{H - margin.bottom}})`)
        .call(d3.axisBottom(x).ticks(8));
      svg.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(${{margin.left}},0)`)
        .call(d3.axisLeft(y).ticks(7, "~s"));

      const line = d3.line()
        .x(d => x(d.date))
        .y(d => y(d.price));

      svg.append("path")
        .datum(prices)
        .attr("fill", "none")
        .attr("stroke", "var(--line)")
        .attr("stroke-width", 2)
        .attr("d", line);

      for (let i = 1; i < anchors.length; i++) {{
        const a = anchors[i - 1], b = anchors[i];
        const x1 = x(a.jsDate), x2 = x(b.jsDate);
        const yy = H - margin.bottom + 27;
        svg.append("line")
          .attr("x1", x1).attr("x2", x2).attr("y1", yy).attr("y2", yy)
          .attr("stroke", "var(--green)").attr("stroke-width", 1.4);
        svg.append("text")
          .attr("x", (x1 + x2) / 2).attr("y", yy - 5)
          .attr("text-anchor", "middle")
          .attr("fill", "white").attr("font-size", 11)
          .text(`${{b.interval_from_prev_days}}d`);
      }}

      const g = svg.append("g");
      g.selectAll("circle")
        .data(anchors)
        .enter()
        .append("circle")
        .attr("cx", d => x(d.jsDate))
        .attr("cy", d => y(d.price))
        .attr("r", 6)
        .attr("fill", "var(--green)")
        .attr("stroke", "#07120b")
        .attr("stroke-width", 1.5)
        .on("mousemove", (event, d) => {{
          tooltip.style("display", "block")
            .style("left", (event.clientX + 12) + "px")
            .style("top", (event.clientY + 12) + "px")
            .html(`<b>#${{d.anchor_index}} ${{d.date}}</b><br>${{fmtMoney(d.price)}}<br>前跌: ${{d.drop_pct}}%<br>后涨: +${{d.rally_pct}}%<br>间隔: ${{d.interval_from_prev_days || "-"}} 天`);
        }})
        .on("mouseleave", () => tooltip.style("display", "none"));

      g.selectAll("text")
        .data(anchors)
        .enter()
        .append("text")
        .attr("x", d => x(d.jsDate) + 8)
        .attr("y", d => y(d.price) - 8)
        .attr("fill", "white")
        .attr("font-size", 12)
        .text(d => d.anchor_index);

      d3.select("#summary").html(`<div style="font-weight:700;margin-bottom:8px">${{active}} 周期：${{anchors.length}} 个局部解锚点</div>`);
      d3.select("#list").html(anchors.map(d => `
        <div class="row">
          <div class="idx">#${{d.anchor_index}}</div>
          <div>
            <div class="date">${{d.date}} · ${{fmtMoney(d.price)}}</div>
            <div>前跌 ${{d.drop_pct}}% · 后涨 +${{d.rally_pct}}% · 间隔 ${{d.interval_from_prev_days || "-"}} 天</div>
          </div>
        </div>
      `).join(""));
    }}

    document.querySelectorAll("button[data-cycle]").forEach(btn => {{
      btn.addEventListener("click", () => {{
        active = btn.dataset.cycle;
        document.querySelectorAll("button[data-cycle]").forEach(b => b.classList.toggle("active", b === btn));
        render();
      }});
    }});
    render();
  </script>
</body>
</html>
"""
    (OUT / "green_anchor_local_v3.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    table, intervals = build_tables(series)
    table.to_csv(OUT / "local_impulse_anchors_v3.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUT / "local_impulse_intervals_v3.csv", index=False, encoding="utf-8-sig")
    write_html(series, table)


if __name__ == "__main__":
    main()
