from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent
HTML = OUT / "peak_structure_workbench_v18.html"


# 默认对比：以"顶"为 0 点对齐，重点看顶前的爬升结构。
# 红 = 历史顶（做 幅度/时间/平移 变换去套绿），绿 = 当前顶（原样）。
DEFAULT_CASES = [
    {
        "name": "2021顶 → 2025主顶(8/12)（顶前一年结构）",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-08-12",
        "pre_days": 365,
        "post_days": 120,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
    },
    {
        "name": "2017顶 → 2021顶（顶前一年结构·桥接用）",
        "left_anchor": "2017-12-17",
        "right_anchor": "2021-11-08",
        "pre_days": 365,
        "post_days": 120,
        "amp_scale": 0.33,
        "time_scale": 1.00,
        "shift_days": 0,
    },
]


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def payload(series: pd.Series) -> list[list[object]]:
    return [[d.date().isoformat(), round(float(v), 2)] for d, v in series.items()]


def write_html(series: pd.Series) -> None:
    html = (
        TEMPLATE
        .replace("__DATA_JSON__", json.dumps(payload(series), ensure_ascii=False))
        .replace("__CASES_JSON__", json.dumps(DEFAULT_CASES, ensure_ascii=False))
    )
    HTML.write_text(html, encoding="utf-8")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC 顶前结构对比工作台 v18</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#181c23; color:#eef3fb; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; }
header { display:flex; align-items:center; gap:8px; flex-wrap:wrap; padding:10px 14px; background:#141821; border-bottom:1px solid rgba(255,255,255,.1); }
h1 { margin:0 10px 0 0; font-size:15px; white-space:nowrap; }
button { color:#dce7f7; background:#28303d; border:1px solid rgba(255,255,255,.16); border-radius:6px; padding:6px 10px; cursor:pointer; font-size:12px; }
button:hover { background:#34404f; }
button.preset { background:#1f2937; }
button.accent { background:#1f6feb33; border-color:#1f6feb66; }
main { display:grid; grid-template-columns:minmax(0,1fr) 430px; gap:12px; padding:12px; }
.panel { background:#20242d; border:1px solid rgba(255,255,255,.08); border-radius:8px; }
#chart { width:100%; height:calc(100vh - 96px); display:block; cursor:crosshair; }
aside { padding:12px; max-height:calc(100vh - 96px); overflow:auto; }
.grid { display:grid; grid-template-columns:92px 1fr; gap:8px; align-items:center; font-size:12px; color:#b5c0d0; }
label { color:#aeb8c8; }
input, select, textarea { width:100%; color:#eaf1fb; background:#161b22; border:1px solid rgba(255,255,255,.14); border-radius:6px; padding:6px; font-size:12px; }
input[type=range] { padding:0; }
textarea { min-height:120px; resize:vertical; font-family:Consolas,monospace; font-size:12px; }
.section { padding:10px 0; border-bottom:1px solid rgba(255,255,255,.08); }
.row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.val { color:#9fe7ff; font-variant-numeric:tabular-nums; min-width:42px; text-align:right; }
.item { padding:7px 0; border-bottom:1px solid rgba(255,255,255,.07); font-size:12px; color:#aeb8c8; cursor:pointer; }
.item.active { color:#fff; }
.item:hover { color:#fff; }
.strong { color:#fff; font-weight:700; }
.red { color:#ff6a5f; } .green { color:#19e08a; }
.muted { color:#aeb8c8; font-size:12px; line-height:1.5; }
code { background:#11161d; padding:1px 4px; border-radius:4px; color:#cfe3ff; }
.tip { position:fixed; display:none; pointer-events:none; background:rgba(8,12,18,.96); border:1px solid rgba(255,255,255,.18); padding:8px 10px; border-radius:6px; font-size:12px; line-height:1.55; z-index:20; white-space:nowrap; }
.axis { fill:#8595a8; font-size:11px; }
.legend { font-size:12px; font-weight:700; }
</style>
</head>
<body>
<header>
  <h1>BTC 顶前结构对比 v18</h1>
  <span class="muted">窗口预设：</span>
  <button class="preset" data-pre="730" data-post="180">顶前两年</button>
  <button class="preset" data-pre="365" data-post="120">顶前一年</button>
  <button class="preset" data-pre="180" data-post="60">顶前半年</button>
  <button class="preset" data-pre="90" data-post="30">顶前三月</button>
  <span style="flex:1"></span>
  <button id="downloadAll">下载规则JSON</button>
  <button id="exportCsv">导出曲线CSV</button>
  <button id="resetAll" class="accent">恢复默认</button>
</header>
<main>
  <section class="panel"><svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg></section>
  <aside class="panel">
    <div class="section">
      <div class="grid">
        <label>对比 case</label><select id="caseSelect"></select>
        <label>名称</label><input id="caseName" />
        <label>红·历史顶</label><input id="leftAnchor" type="date" />
        <label>绿·当前顶</label><input id="rightAnchor" type="date" />
        <label>顶前(天)</label><input id="preDays" type="number" min="30" max="1200" step="5" />
        <label>顶后(天)</label><input id="postDays" type="number" min="0" max="600" step="5" />
      </div>
      <div class="row" style="margin-top:8px">
        <button id="newCase">新建</button>
        <button id="saveCase" class="accent">保存当前</button>
        <button id="deleteCase">删除</button>
      </div>
    </div>
    <div class="section">
      <div class="grid">
        <label>幅度系数</label><div class="row"><input id="amp" type="range" min="0.10" max="1.50" step="0.01" style="flex:1"/><span class="val" id="ampText"></span></div>
        <label>时间比例</label><div class="row"><input id="timeScale" type="range" min="0.50" max="1.50" step="0.01" style="flex:1"/><span class="val" id="timeScaleText"></span></div>
        <label>水平平移</label><div class="row"><input id="shift" type="range" min="-365" max="365" step="1" style="flex:1"/><span class="val" id="shiftText"></span></div>
      </div>
      <div class="muted" style="margin-top:8px">红线横向 <code>距顶天数 × 时间比例 + 平移</code>，纵向 <code>log收益 × 幅度系数</code>；绿线原样。两条都以各自的顶为 <code>0</code> 点对齐。鼠标移到图上可读两条曲线在同一对齐日的数值。</div>
    </div>
    <div class="section"><div id="summary" class="muted"></div></div>
    <div class="section">
      <div class="strong">当前设置 JSON</div>
      <textarea id="settingsJson"></textarea>
      <div class="row" style="margin-top:8px">
        <button id="applyJson">从JSON应用</button>
        <button id="copyJson">复制</button>
      </div>
    </div>
    <div id="caseList"></div>
  </aside>
</main>
<div id="tip" class="tip"></div>
<script>
const RAW = __DATA_JSON__.map(d => { const t = new Date(d[0] + "T00:00:00").getTime(); return { dateText:d[0], t:t, price:+d[1] }; });
const DEFAULT_CASES = __CASES_JSON__;
const STORAGE_KEY = "btc_peak_structure_workbench_v18b";
const DAY = 86400000;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
const W = 1200, H = 720, M = { l:64, r:120, t:24, b:40 };
let cases = loadCases();
let activeIndex = 0;
let view = null;
let hoverG = null;

function loadCases() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(saved) && saved.length) return saved;
  } catch (e) {}
  return structuredClone(DEFAULT_CASES);
}
function persist() { localStorage.setItem(STORAGE_KEY, JSON.stringify(cases)); }
function val(id) { return document.getElementById(id).value; }
function fmt(v) { return Math.round(v).toLocaleString(); }
function el(parent, name, attrs) {
  const n = document.createElementNS(NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  parent.appendChild(n);
  return n;
}
function clearNode(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function nearestRow(t) {
  let lo = 0, hi = RAW.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (RAW[mid].t < t) lo = mid + 1; else hi = mid; }
  if (lo > 0 && Math.abs(RAW[lo - 1].t - t) <= Math.abs(RAW[lo].t - t)) lo--;
  return RAW[lo];
}
function buildRows(c) {
  const la = nearestRow(new Date(c.left_anchor + "T00:00:00").getTime());
  const ra = nearestRow(new Date(c.right_anchor + "T00:00:00").getTime());
  const make = (anchor, transform) => RAW
    .map(r => ({ r, rel: Math.round((r.t - anchor.t) / DAY) }))
    .filter(o => o.rel >= -c.pre_days && o.rel <= c.post_days)
    .map(o => {
      const logn = Math.log(o.r.price / anchor.price);
      return {
        cycle: transform ? "left" : "right",
        dateText: o.r.dateText,
        rel_day: o.rel,
        rel_plot: transform ? o.rel * c.time_scale + c.shift_days : o.rel,
        price: o.r.price,
        log_norm: logn,
        log_plot: transform ? logn * c.amp_scale : logn,
        pct: (o.r.price / anchor.price - 1) * 100,
      };
    });
  return { left: make(la, true), right: make(ra, false), la, ra };
}
function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (span <= 0) return [lo];
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function drawPath(g, rows, px, py, color, w) {
  let d = "";
  rows.slice().sort((a, b) => a.rel_plot - b.rel_plot)
    .forEach((p, i) => d += (i ? "L" : "M") + px(p.rel_plot).toFixed(1) + "," + py(p.log_plot).toFixed(1));
  el(g, "path", { d:d, fill:"none", stroke:color, "stroke-width":w, opacity:.92, "stroke-linejoin":"round" });
}
function draw() {
  clearNode(svg);
  const c = cases[activeIndex];
  const { left, right, la, ra } = buildRows(c);
  if (!left.length || !right.length) {
    el(svg, "text", { x:W / 2, y:H / 2, "text-anchor":"middle", "class":"axis" }).textContent = "锚点窗口内无数据，检查日期";
    updateSidebar(c, la, ra);
    return;
  }
  const rows = left.concat(right);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const d of rows) {
    if (d.rel_plot < xmin) xmin = d.rel_plot;
    if (d.rel_plot > xmax) xmax = d.rel_plot;
    if (d.log_plot < ymin) ymin = d.log_plot;
    if (d.log_plot > ymax) ymax = d.log_plot;
  }
  const xpad = (xmax - xmin) * 0.02 || 1, ypad = (ymax - ymin) * 0.05 || 0.05;
  const X0 = xmin - xpad, X1 = xmax + xpad, Y0 = ymin - ypad, Y1 = ymax + ypad;
  const px = v => M.l + (v - X0) / (X1 - X0) * (W - M.l - M.r);
  const py = v => H - M.b - (v - Y0) / (Y1 - Y0) * (H - M.t - M.b);
  view = { left, right, X0, X1, px, py };

  const g = el(svg, "g", {});
  for (const v of niceTicks(Y0, Y1, 6)) {
    const y = py(v);
    el(g, "line", { x1:M.l, y1:y, x2:W - M.r, y2:y, stroke:"rgba(255,255,255,.06)" });
    el(g, "text", { x:M.l - 8, y:y + 4, "text-anchor":"end", "class":"axis" }).textContent = ((Math.exp(v) - 1) * 100).toFixed(0) + "%";
  }
  for (const v of niceTicks(X0, X1, 8)) {
    const x = px(v);
    el(g, "line", { x1:x, y1:M.t, x2:x, y2:H - M.b, stroke:"rgba(255,255,255,.06)" });
    el(g, "text", { x:x, y:H - M.b + 16, "text-anchor":"middle", "class":"axis" }).textContent = Math.round(v) + "d";
  }
  el(g, "line", { x1:px(0), y1:M.t, x2:px(0), y2:H - M.b, stroke:"rgba(255,255,255,.35)", "stroke-dasharray":"4 4" });
  el(g, "text", { x:px(0) + 4, y:M.t + 12, "class":"axis", fill:"#cdd7e6" }).textContent = "顶 (0)";
  el(g, "line", { x1:M.l, y1:py(0), x2:W - M.r, y2:py(0), stroke:"rgba(255,255,255,.18)" });

  drawPath(g, left, px, py, "#ff6a5f", 2.0);
  drawPath(g, right, px, py, "#19e08a", 2.6);

  const gEnd = right.reduce((a, b) => a.rel_day >= b.rel_day ? a : b, right[0]);
  el(g, "circle", { cx:px(gEnd.rel_plot), cy:py(gEnd.log_plot), r:4, fill:"#19e08a" });
  el(g, "text", { x:px(gEnd.rel_plot) - 6, y:py(gEnd.log_plot) - 8, "text-anchor":"end", "class":"axis", fill:"#19e08a" }).textContent = "绿末 " + gEnd.dateText;

  el(g, "text", { x:W - M.r + 8, y:M.t + 14, "class":"legend", fill:"#ff6a5f" }).textContent = "红 " + c.left_anchor;
  el(g, "text", { x:W - M.r + 8, y:M.t + 32, "class":"legend", fill:"#19e08a" }).textContent = "绿 " + c.right_anchor;

  hoverG = el(svg, "g", {});
  updateSidebar(c, la, ra);
}
function updateSidebar(c, la, ra) {
  document.getElementById("summary").innerHTML =
    `<div><span class="red">红</span> 历史顶 ${c.left_anchor}（$${fmt(la.price)}）— 做 幅度/时间/平移 变换</div>` +
    `<div><span class="green">绿</span> 当前顶 ${c.right_anchor}（$${fmt(ra.price)}）— 原样</div>` +
    `<div>窗口 顶前 ${c.pre_days}d → 顶后 ${c.post_days}d ｜ 幅度 ${(+c.amp_scale).toFixed(2)} · 时间 ${(+c.time_scale).toFixed(2)} · 平移 ${c.shift_days}d</div>` +
    `<div>纵轴=相对各自顶的 % ｜ 横轴=距顶天数</div>`;
  document.getElementById("settingsJson").value = JSON.stringify(currentCase(), null, 2);
  renderCaseList();
}
function currentCase() {
  return {
    name: val("caseName"),
    left_anchor: val("leftAnchor"),
    right_anchor: val("rightAnchor"),
    pre_days: +val("preDays"),
    post_days: +val("postDays"),
    amp_scale: +val("amp"),
    time_scale: +val("timeScale"),
    shift_days: +val("shift"),
  };
}
function setControls(c) {
  document.getElementById("caseName").value = c.name || "untitled";
  document.getElementById("leftAnchor").value = c.left_anchor;
  document.getElementById("rightAnchor").value = c.right_anchor;
  document.getElementById("preDays").value = c.pre_days;
  document.getElementById("postDays").value = c.post_days;
  document.getElementById("amp").value = c.amp_scale != null ? c.amp_scale : 0.5;
  document.getElementById("timeScale").value = c.time_scale != null ? c.time_scale : 1;
  document.getElementById("shift").value = c.shift_days != null ? c.shift_days : 0;
  updateTexts();
}
function updateTexts() {
  document.getElementById("ampText").textContent = (+val("amp")).toFixed(2);
  document.getElementById("timeScaleText").textContent = (+val("timeScale")).toFixed(2);
  document.getElementById("shiftText").textContent = val("shift") + "d";
}
function syncFromControls() {
  cases[activeIndex] = currentCase();
  updateTexts();
  draw();
}
function renderCaseSelect() {
  const sel = document.getElementById("caseSelect");
  sel.innerHTML = cases.map((c, i) => `<option value="${i}">${i + 1}. ${c.name || "untitled"}</option>`).join("");
  sel.value = activeIndex;
}
function renderCaseList() {
  const box = document.getElementById("caseList");
  box.innerHTML = cases.map((c, i) =>
    `<div class="item ${i === activeIndex ? "active" : ""}" data-i="${i}">
      <div class="strong">${i + 1}. ${c.name || "untitled"}</div>
      <div>${c.left_anchor} → ${c.right_anchor} · 顶前${c.pre_days}/顶后${c.post_days}</div>
      <div>幅度 ${(+c.amp_scale).toFixed(2)} · 时间 ${(+c.time_scale).toFixed(2)} · 平移 ${c.shift_days}d</div>
    </div>`).join("");
  box.querySelectorAll(".item").forEach(node => node.onclick = () => {
    activeIndex = +node.dataset.i;
    renderCaseSelect();
    setControls(cases[activeIndex]);
    draw();
  });
}
function download(name, text, type) {
  const blob = new Blob([text], { type: type || "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function svgPt(ev) {
  const rect = svg.getBoundingClientRect();
  return { x:(ev.clientX - rect.left) / rect.width * W, y:(ev.clientY - rect.top) / rect.height * H, cx:ev.clientX, cy:ev.clientY };
}
function nearestByPlot(rows, relPlot) {
  let best = rows[0], bd = Infinity;
  for (const p of rows) { const d = Math.abs(p.rel_plot - relPlot); if (d < bd) { bd = d; best = p; } }
  return best;
}
function side(rel) { return (rel >= 0 ? "后" : "前") + Math.abs(rel) + "d"; }
function hideHover() { tip.style.display = "none"; if (hoverG) clearNode(hoverG); }
svg.addEventListener("mousemove", ev => {
  if (!view) return;
  const p = svgPt(ev);
  if (p.x < M.l || p.x > W - M.r || p.y < M.t || p.y > H - M.b) { hideHover(); return; }
  const relPlot = view.X0 + (p.x - M.l) / (W - M.l - M.r) * (view.X1 - view.X0);
  const r = nearestByPlot(view.left, relPlot);
  const g = nearestByPlot(view.right, relPlot);
  clearNode(hoverG);
  el(hoverG, "line", { x1:p.x, y1:M.t, x2:p.x, y2:H - M.b, stroke:"rgba(255,255,255,.25)" });
  el(hoverG, "circle", { cx:view.px(r.rel_plot), cy:view.py(r.log_plot), r:4, fill:"#ff6a5f" });
  el(hoverG, "circle", { cx:view.px(g.rel_plot), cy:view.py(g.log_plot), r:4, fill:"#19e08a" });
  tip.style.display = "block";
  tip.style.left = (p.cx + 14) + "px";
  tip.style.top = (p.cy + 14) + "px";
  tip.innerHTML =
    `<div>对齐位置 ≈ 顶${side(Math.round(relPlot))}</div>` +
    `<div class="red">红 ${r.dateText} · 顶${side(r.rel_day)} · $${fmt(r.price)} · ${r.pct.toFixed(1)}%</div>` +
    `<div class="green">绿 ${g.dateText} · 顶${side(g.rel_day)} · $${fmt(g.price)} · ${g.pct.toFixed(1)}%</div>`;
});
svg.addEventListener("mouseleave", hideHover);

["caseName", "leftAnchor", "rightAnchor", "preDays", "postDays", "amp", "timeScale", "shift"].forEach(id =>
  document.getElementById(id).addEventListener("input", syncFromControls));
document.getElementById("caseSelect").addEventListener("change", e => {
  activeIndex = +e.target.value; setControls(cases[activeIndex]); draw();
});
document.querySelectorAll("button.preset").forEach(b => b.onclick = () => {
  document.getElementById("preDays").value = b.dataset.pre;
  document.getElementById("postDays").value = b.dataset.post;
  syncFromControls();
});
document.getElementById("newCase").onclick = () => {
  cases.push({ ...currentCase(), name:"新对比" });
  activeIndex = cases.length - 1; persist();
  renderCaseSelect(); setControls(cases[activeIndex]); draw();
};
document.getElementById("saveCase").onclick = () => {
  cases[activeIndex] = currentCase(); persist();
  renderCaseSelect(); draw();
};
document.getElementById("deleteCase").onclick = () => {
  if (cases.length <= 1) { alert("至少保留一个 case。"); return; }
  cases.splice(activeIndex, 1); activeIndex = Math.max(0, activeIndex - 1); persist();
  renderCaseSelect(); setControls(cases[activeIndex]); draw();
};
document.getElementById("resetAll").onclick = () => {
  if (!confirm("恢复到内置默认对比？当前浏览器里保存的 case 会被覆盖。")) return;
  cases = structuredClone(DEFAULT_CASES); activeIndex = 0; persist();
  renderCaseSelect(); setControls(cases[0]); draw();
};
document.getElementById("downloadAll").onclick = () =>
  download("peak_structure_rules_v18.json", JSON.stringify({ version:"v18", cases }, null, 2));
document.getElementById("exportCsv").onclick = () => {
  const { left, right } = buildRows(cases[activeIndex]);
  const head = "cycle,date,rel_day,rel_plot,price,pct_vs_peak,log_norm,log_plot\n";
  const body = left.concat(right).map(r =>
    [r.cycle, r.dateText, r.rel_day, r.rel_plot, r.price, r.pct.toFixed(3), r.log_norm.toFixed(5), r.log_plot.toFixed(5)].join(",")).join("\n");
  download("peak_structure_overlay_v18.csv", head + body, "text/csv");
};
document.getElementById("applyJson").onclick = () => {
  try {
    const parsed = JSON.parse(val("settingsJson"));
    cases[activeIndex] = parsed; setControls(parsed); persist();
    renderCaseSelect(); draw();
  } catch (e) { alert("JSON 解析失败。"); }
};
document.getElementById("copyJson").onclick = async () => {
  try { await navigator.clipboard.writeText(val("settingsJson")); } catch (e) {}
};

renderCaseSelect();
setControls(cases[activeIndex]);
draw();
</script>
</body>
</html>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    write_html(series)
    print("data:", series.index.min().date(), "->", series.index.max().date(),
          "(", len(series), "days,  latest $", round(float(series.iloc[-1]), 2), ")")
    print("wrote:", HTML)


if __name__ == "__main__":
    main()
