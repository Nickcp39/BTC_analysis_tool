from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
NODES = ROOT / "analysis_runs" / "2026-06-04_green_anchor_probe" / "segment_nodes_v5.csv"
OUT = Path(__file__).resolve().parent

LEFT_TOP = pd.Timestamp("2021-11-08")
RIGHT_TOP = pd.Timestamp("2025-10-05")
MAX_REL_DAY = 520
MIN_INTERVAL_RATIO = 0.70
MAX_INTERVAL_RATIO = 1.30


def kind_group(kind: str) -> str:
    if kind in {"drop_end", "rally_start"}:
        return "low"
    if kind in {"rally_end", "drop_start"}:
        return "high"
    if "down" in kind:
        return "down"
    if "up" in kind:
        return "up"
    return "other"


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def load_post_nodes() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(NODES, parse_dates=["date"])
    df["cycle"] = df["cycle"].astype(str)
    df["kind_group"] = df["kind"].map(kind_group)

    def select(cycle: str, top: pd.Timestamp) -> pd.DataFrame:
        g = df[
            (df["cycle"] == cycle)
            & (df["date"] >= top)
            & (df["date"] <= top + pd.Timedelta(days=MAX_REL_DAY))
        ].copy()
        g["rel"] = (g["date"] - top).dt.days
        peak = {
            "cycle": cycle,
            "date": top,
            "days_from_halving": np.nan,
            "price": np.nan,
            "kind": "peak",
            "kind_group": "high",
            "anchor_index": 0,
            "rel": 0,
        }
        g = pd.concat([pd.DataFrame([peak]), g], ignore_index=True)
        g = g.drop_duplicates(subset=["rel", "kind_group"], keep="first").sort_values("rel").reset_index(drop=True)
        return g

    return select("2021", LEFT_TOP), select("2025", RIGHT_TOP)


def pair_cost(left: pd.Series, right: pd.Series) -> float:
    if left["kind"] == "peak" and right["kind"] == "peak":
        return 0.0
    if left["kind"] == right["kind"]:
        kind_penalty = 0.0
    elif left["kind_group"] == right["kind_group"]:
        kind_penalty = 0.25
    else:
        return 99.0
    return kind_penalty


def sequence_match(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    # Dynamic programming over ordered node pairs. A transition is legal only if
    # consecutive interval ratio = current-cycle interval / historical interval
    # falls inside 0.7~1.3.
    pairs = []
    for i, l in left.iterrows():
        for j, r in right.iterrows():
            cost = pair_cost(l, r)
            if cost >= 90:
                continue
            pairs.append((i, j, cost))

    dp: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}
    for i, j, cost in pairs:
        lrel, rrel = float(left.loc[i, "rel"]), float(right.loc[j, "rel"])
        if lrel == 0 and rrel == 0:
            dp[(i, j)] = (0.0, [(i, j)])
            continue
        best: tuple[float, list[tuple[int, int]]] | None = None
        for (pi, pj), (prev_score, prev_path) in dp.items():
            if pi >= i or pj >= j:
                continue
            l_gap = float(left.loc[i, "rel"] - left.loc[pi, "rel"])
            r_gap = float(right.loc[j, "rel"] - right.loc[pj, "rel"])
            if l_gap <= 0 or r_gap <= 0:
                continue
            ratio = r_gap / l_gap
            if not (MIN_INTERVAL_RATIO <= ratio <= MAX_INTERVAL_RATIO):
                continue
            # Reward longer valid chains, lightly penalize ratio drift and kind mismatch.
            score = prev_score - 10.0 + abs(np.log(ratio)) * 2.0 + cost
            candidate = (score, prev_path + [(i, j)])
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is not None:
            dp[(i, j)] = best

    best = min(dp.values(), key=lambda x: (x[0], -len(x[1])))
    rows = []
    for n, (i, j) in enumerate(best[1], start=1):
        l, r = left.loc[i], right.loc[j]
        if n == 1:
            ratio = None
            l_gap = None
            r_gap = None
        else:
            prev_i, prev_j = best[1][n - 2]
            l_gap = int(left.loc[i, "rel"] - left.loc[prev_i, "rel"])
            r_gap = int(right.loc[j, "rel"] - right.loc[prev_j, "rel"])
            ratio = r_gap / l_gap
        rows.append(
            {
                "seq": n,
                "left_index": int(l["anchor_index"]),
                "right_index": int(r["anchor_index"]),
                "left_date": pd.Timestamp(l["date"]).date().isoformat(),
                "right_date": pd.Timestamp(r["date"]).date().isoformat(),
                "left_rel": int(l["rel"]),
                "right_rel": int(r["rel"]),
                "left_kind": l["kind"],
                "right_kind": r["kind"],
                "left_group": l["kind_group"],
                "right_group": r["kind_group"],
                "left_gap": l_gap,
                "right_gap": r_gap,
                "interval_ratio": None if ratio is None else round(float(ratio), 4),
            }
        )
    return pd.DataFrame(rows)


def piecewise_map(x: np.ndarray, src: list[float], dst: list[float]) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for idx, val in enumerate(x):
        if val <= src[0]:
            scale = (dst[1] - dst[0]) / (src[1] - src[0])
            out[idx] = dst[0] + (val - src[0]) * scale
        elif val >= src[-1]:
            scale = (dst[-1] - dst[-2]) / (src[-1] - src[-2])
            out[idx] = dst[-1] + (val - src[-1]) * scale
        else:
            k = max(i for i in range(len(src) - 1) if src[i] <= val)
            scale = (dst[k + 1] - dst[k]) / (src[k + 1] - src[k])
            out[idx] = dst[k] + (val - src[k]) * scale
    return out


def build_overlay(series: pd.Series, matches: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    latest = series.index.max()
    right_end_rel = int((latest - RIGHT_TOP).days)
    src = matches["left_rel"].astype(float).tolist()
    dst = matches["right_rel"].astype(float).tolist()
    if dst[-1] < right_end_rel:
        last_scale = (dst[-1] - dst[-2]) / (src[-1] - src[-2])
        src.append(src[-1] + (right_end_rel - dst[-1]) / max(last_scale, 1e-6))
        dst.append(float(right_end_rel))

    left_end = LEFT_TOP + pd.Timedelta(days=int(np.ceil(src[-1])))
    left = series.loc[LEFT_TOP:left_end]
    right = series.loc[RIGHT_TOP:latest]
    left_top = float(series.loc[LEFT_TOP])
    right_top = float(series.loc[RIGHT_TOP])

    left_tmp = pd.DataFrame(
        {
            "cycle": "2021_warped",
            "date": left.index,
            "rel_original": (left.index - LEFT_TOP).days,
            "price": left.values,
            "log_norm": np.log(left.values / left_top),
        }
    )
    right_df = pd.DataFrame(
        {
            "cycle": "2025_actual",
            "date": right.index,
            "rel": (right.index - RIGHT_TOP).days.astype(float),
            "price": right.values,
            "log_norm": np.log(right.values / right_top),
        }
    )

    ratios = []
    for _, m in matches.iterrows():
        if int(m["left_rel"]) == 0:
            continue
        l_date, r_date = pd.Timestamp(m["left_date"]), pd.Timestamp(m["right_date"])
        l_norm = np.log(float(series.loc[l_date]) / left_top)
        r_norm = np.log(float(series.loc[r_date]) / right_top)
        if abs(l_norm) > 1e-9:
            ratios.append(r_norm / l_norm)
    amp = float(np.median(ratios)) if ratios else 1.0
    left_tmp["rel"] = piecewise_map(left_tmp["rel_original"].values.astype(float), src, dst)
    left_tmp["log_norm"] = left_tmp["log_norm"] * amp
    left_tmp = left_tmp[(left_tmp["rel"] >= 0) & (left_tmp["rel"] <= right_end_rel)]
    return pd.concat([left_tmp, right_df], ignore_index=True), amp


def write_html(plot: pd.DataFrame, matches: pd.DataFrame, amp: float) -> None:
    p = plot.copy()
    p["date"] = p["date"].dt.date.astype(str)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Sequence Warp v15</title>
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
<header><h1>v15 序列时间扭曲：连续段 ratio 必须在 0.7 ~ 1.3，然后再画重叠图</h1></header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel"><div id="summary"></div><div id="list"></div></aside>
</main>
<div id="tip" class="tip"></div>
<script>
const DATA = {json.dumps(p.to_dict(orient="records"), ensure_ascii=False)};
const MATCHES = {json.dumps(matches.to_dict(orient="records"), ensure_ascii=False)};
const AMP = {round(amp, 4)};
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const W = 1200, H = 720, M = {{l:72,r:28,t:30,b:48}};
function el(name, attrs={{}}) {{
  const n = document.createElementNS(NS, name);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k, v);
  svg.appendChild(n);
  return n;
}}
function x(d) {{
  const max = Math.max(...DATA.map(r=>+r.rel));
  return M.l + d.rel / max * (W - M.l - M.r);
}}
function y(d) {{
  const vals = DATA.map(r=>+r.log_norm);
  const min=Math.min(...vals), max=Math.max(...vals);
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
["2021_warped","2025_actual"].forEach(cycle => {{
  const rows = DATA.filter(d => d.cycle === cycle).sort((a,b)=>a.rel-b.rel);
  let path = "";
  rows.forEach((d,i) => path += (i ? "L" : "M") + x(d).toFixed(2) + "," + y(d).toFixed(2));
  el("path", {{d:path, fill:"none", stroke:color(cycle), "stroke-width":cycle==="2025_actual"?2.4:2.1, opacity:cycle==="2025_actual"?.95:.78}});
}});
MATCHES.forEach(m => {{
  const rows = DATA;
  const l = rows.find(d => d.cycle==="2021_warped" && Math.round(d.rel)===m.right_rel);
  const r = rows.find(d => d.cycle==="2025_actual" && Math.round(d.rel)===m.right_rel);
  if (!l || !r) return;
  [l,r].forEach((p,i) => el("circle", {{cx:x(p),cy:y(p),r:5.8,fill:i===0?"#ff6a5f":"#00e676",stroke:"#120807","stroke-width":1.3}}));
}});
document.getElementById("summary").innerHTML = `
  <div class="strong">结果</div>
  <div><span class="red">红</span>：2021 顶后路径，已按匹配序列 time-warp + 幅度退火</div>
  <div><span class="green">绿</span>：2025 顶后实际路径</div>
  <div>匹配节点：${{MATCHES.length}}</div>
  <div>幅度退火系数：${{AMP}}</div>
  <br>
`;
document.getElementById("list").innerHTML = MATCHES.map(m => `
  <div class="item">
    <div class="strong">#${{m.seq}} · ${{m.left_kind}} → ${{m.right_kind}}</div>
    <div>${{m.left_date}} rel ${{m.left_rel}}d → ${{m.right_date}} rel ${{m.right_rel}}d</div>
    <div>段间隔：${{m.left_gap || "-"}}d → ${{m.right_gap || "-"}}d · ratio ${{m.interval_ratio || "-"}}</div>
  </div>
`).join("");
</script>
</body>
</html>"""
    (OUT / "green_anchor_sequence_warp_v15.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    left, right = load_post_nodes()
    matches = sequence_match(left, right)
    plot, amp = build_overlay(series, matches)
    matches.to_csv(OUT / "sequence_warp_matches_v15.csv", index=False, encoding="utf-8-sig")
    plot.to_csv(OUT / "sequence_warp_overlay_v15.csv", index=False, encoding="utf-8-sig")
    write_html(plot, matches, amp)
    print(f"matches={len(matches)} amp={amp:.4f}")
    print(matches.to_string(index=False))


if __name__ == "__main__":
    main()
