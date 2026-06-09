from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
MARKS_CSV = ROOT / "output" / "core_points" / "manual_points.csv"
OUT = Path(__file__).resolve().parent
HTML = OUT / "segment_cycle_workbench_v19.html"
CASE_LEDGER = OUT / "segment_cycle_case_ledger_v19.csv"
RUNBOOK = OUT / "SEGMENT_CYCLE_WORKBENCH_V19_RUNBOOK.md"


CYCLES = [
    {
        "id": "2013",
        "label": "2013 cycle",
        "halving": "2012-11-28",
        "peak": "2013-11-30",
        "bottom": "2015-01-14",
        "note": "Historical anchors; local data starts 2014-12, so peak windows are missing.",
    },
    {
        "id": "2017",
        "label": "2017 cycle",
        "halving": "2016-07-09",
        "peak": "2017-12-16",
        "bottom": "2018-12-15",
        "note": "Peak/bottom from marked historical cycle.",
    },
    {
        "id": "2021",
        "label": "2021 cycle",
        "halving": "2020-05-11",
        "peak": "2021-11-08",
        "bottom": "2022-11-21",
        "note": "User-marked reference cycle.",
    },
    {
        "id": "2025",
        "label": "2025 cycle",
        "halving": "2024-04-20",
        "peak": "2025-08-12",
        "bottom": "2026-02-05",
        "note": "Current cycle anchors are editable assumptions.",
    },
]


WINDOW_PRESETS = [
    {"id": "peak_pre_6m", "label": "Peak pre 6m", "anchor": "peak", "pre_days": 183, "post_days": 0},
    {"id": "peak_post_6m", "label": "Peak post 6m", "anchor": "peak", "pre_days": 0, "post_days": 183},
    {"id": "peak_pm_6m", "label": "Peak +/- 6m", "anchor": "peak", "pre_days": 183, "post_days": 183},
    {"id": "peak_pre_1y", "label": "Peak pre 1y", "anchor": "peak", "pre_days": 365, "post_days": 0},
    {"id": "peak_post_1y", "label": "Peak post 1y", "anchor": "peak", "pre_days": 0, "post_days": 365},
    {"id": "peak_pm_1y", "label": "Peak +/- 1y", "anchor": "peak", "pre_days": 365, "post_days": 365},
    {"id": "peak_pre_2y", "label": "Peak pre 2y", "anchor": "peak", "pre_days": 730, "post_days": 0},
    {"id": "peak_pre_3y", "label": "Peak pre 3y", "anchor": "peak", "pre_days": 1095, "post_days": 0},
    {"id": "bottom_pre_6m", "label": "Bottom pre 6m", "anchor": "bottom", "pre_days": 183, "post_days": 0},
    {"id": "bottom_post_6m", "label": "Bottom post 6m", "anchor": "bottom", "pre_days": 0, "post_days": 183},
    {"id": "bottom_pm_6m", "label": "Bottom +/- 6m", "anchor": "bottom", "pre_days": 183, "post_days": 183},
    {"id": "bottom_pre_1y", "label": "Bottom pre 1y", "anchor": "bottom", "pre_days": 365, "post_days": 0},
    {"id": "bottom_post_1y", "label": "Bottom post 1y", "anchor": "bottom", "pre_days": 0, "post_days": 365},
    {"id": "bottom_pm_1y", "label": "Bottom +/- 1y", "anchor": "bottom", "pre_days": 365, "post_days": 365},
    {"id": "bottom_pre_2y", "label": "Bottom pre 2y", "anchor": "bottom", "pre_days": 730, "post_days": 0},
    {"id": "bottom_pre_3y", "label": "Bottom pre 3y", "anchor": "bottom", "pre_days": 1095, "post_days": 0},
]


PAIR_ORDER = [
    ("2021", "2025"),
    ("2017", "2021"),
    ("2013", "2017"),
]


def load_price_rows() -> tuple[pd.Series, list[dict]]:
    df = pd.read_csv(DATA, parse_dates=["date"])
    series = df.set_index("date")["price"].sort_index().asfreq("D").ffill()
    rows = [
        {"date": d.date().isoformat(), "price": round(float(v), 2)}
        for d, v in series.items()
    ]
    return series, rows


def load_marks() -> list[dict]:
    if not MARKS_CSV.exists():
        return []
    df = pd.read_csv(MARKS_CSV, parse_dates=["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return [
        {
            "date": d.date().isoformat(),
            "price": round(float(p), 2),
            "type": str(t),
            "n": int(n) if pd.notna(n) else None,
        }
        for n, d, p, t in zip(df["n"], df["date"], df["price"], df["type"])
    ]


def cycle_by_id(cycle_id: str) -> dict:
    return next(c for c in CYCLES if c["id"] == cycle_id)


def build_cases(series: pd.Series) -> list[dict]:
    cases: list[dict] = []

    def coverage(anchor_date: str, pre_days: int, post_days: int) -> dict:
        anchor = pd.Timestamp(anchor_date)
        start = anchor - pd.Timedelta(days=pre_days)
        end = anchor + pd.Timedelta(days=post_days)
        data_start, data_end = series.index.min(), series.index.max()
        available_start = max(start, data_start)
        available_end = min(end, data_end)
        available_days = max(0, (available_end - available_start).days + 1)
        expected_days = pre_days + post_days + 1
        return {
            "expected_days": expected_days,
            "available_days": available_days,
            "coverage": round(available_days / expected_days, 4) if expected_days else 0,
            "missing_left": max(0, (data_start - start).days),
            "missing_right": max(0, (end - data_end).days),
        }

    case_no = 1
    preferred_first = ("2021", "2025", "peak_pre_1y")
    ordered: list[tuple[str, str, dict]] = []
    for left_id, right_id in PAIR_ORDER:
        for preset in WINDOW_PRESETS:
            item = (left_id, right_id, preset)
            if (left_id, right_id, preset["id"]) == preferred_first:
                ordered.insert(0, item)
            else:
                ordered.append(item)

    for left_id, right_id, preset in ordered:
        left_cycle = cycle_by_id(left_id)
        right_cycle = cycle_by_id(right_id)
        anchor_type = preset["anchor"]
        left_anchor = left_cycle[anchor_type]
        right_anchor = right_cycle[anchor_type]
        left_cov = coverage(left_anchor, preset["pre_days"], preset["post_days"])
        right_cov = coverage(right_anchor, preset["pre_days"], preset["post_days"])
        case_id = f"case{case_no:02d}_{preset['id']}_{left_id}_to_{right_id}"
        cases.append(
            {
                "id": case_id,
                "name": f"{preset['label']}: {left_id} -> {right_id}",
                "pair": f"{left_id}->{right_id}",
                "left_cycle": left_id,
                "right_cycle": right_id,
                "anchor_type": anchor_type,
                "left_anchor": left_anchor,
                "right_anchor": right_anchor,
                "pre_days": preset["pre_days"],
                "post_days": preset["post_days"],
                "amp_scale": 0.50 if right_id == "2025" else 0.60,
                "time_scale": 1.00,
                "shift_days": 0,
                "window_id": preset["id"],
                "window_label": preset["label"],
                "left_coverage": left_cov,
                "right_coverage": right_cov,
                "note": f"{left_cycle['label']} {anchor_type} vs {right_cycle['label']} {anchor_type}.",
            }
        )
        case_no += 1
    return cases


def write_case_ledger(cases: list[dict]) -> None:
    rows = []
    for c in cases:
        rows.append(
            {
                "case_id": c["id"],
                "name": c["name"],
                "pair": c["pair"],
                "anchor_type": c["anchor_type"],
                "left_anchor": c["left_anchor"],
                "right_anchor": c["right_anchor"],
                "pre_days": c["pre_days"],
                "post_days": c["post_days"],
                "default_amp": c["amp_scale"],
                "default_time": c["time_scale"],
                "default_shift": c["shift_days"],
                "left_coverage": c["left_coverage"]["coverage"],
                "right_coverage": c["right_coverage"]["coverage"],
            }
        )
    pd.DataFrame(rows).to_csv(CASE_LEDGER, index=False, encoding="utf-8-sig")


def write_runbook() -> None:
    RUNBOOK.write_text(
        """# Segment Cycle Workbench v19

Goal: manually compare BTC cycle structure around peak/bottom anchors across fixed windows, then save the ratio settings and export the visual evidence.

## Workflow
1. Open `segment_cycle_workbench_v19.html`.
2. Start with `Peak pre 1y: 2021 -> 2025`.
3. Adjust height, time scale, and horizontal shift.
4. Save Sample after every visually meaningful fit.
5. Export Sample Library JSON regularly.
6. Export PNG/SVG for the fits you want to discuss.

## First backlog
- Peak pre 1y: 2021 -> 2025
- Peak +/- 6m: 2021 -> 2025
- Peak post 1y: 2021 -> 2025
- Bottom +/- 6m: 2021 -> 2025
- Repeat the same windows for 2017 -> 2021, then compare ratios.

Notes: local data starts on 2014-12-01, so 2013 peak windows are placeholders until older data is added.
""",
        encoding="utf-8",
    )


def write_html(rows: list[dict], marks: list[dict], cases: list[dict]) -> None:
    html = (
        TEMPLATE.replace("__PRICE_JSON__", json.dumps(rows, ensure_ascii=False))
        .replace("__MARKS_JSON__", json.dumps(marks, ensure_ascii=False))
        .replace("__CYCLES_JSON__", json.dumps(CYCLES, ensure_ascii=False))
        .replace("__WINDOWS_JSON__", json.dumps(WINDOW_PRESETS, ensure_ascii=False))
        .replace("__CASES_JSON__", json.dumps(cases, ensure_ascii=False))
    )
    HTML.write_text(html, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BTC Segment Cycle Workbench v19</title>
<style>
:root {
  color-scheme: dark;
  --bg: #161a22;
  --panel: #202631;
  --panel2: #252c38;
  --line: rgba(255,255,255,.10);
  --muted: #a9b5c6;
  --text: #eef4ff;
  --red: #ff6a5f;
  --green: #17e08a;
  --blue: #58a6ff;
  --gold: #f7c948;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  letter-spacing: 0;
}
header {
  min-height: 58px;
  display: grid;
  grid-template-columns: minmax(260px, 1fr) auto;
  gap: 10px;
  align-items: center;
  padding: 10px 14px;
  background: #111722;
  border-bottom: 1px solid var(--line);
}
h1 { margin: 0; font-size: 16px; line-height: 1.3; }
button, select, input, textarea {
  color: var(--text);
  background: #151b25;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 6px;
  font: inherit;
}
button {
  min-height: 34px;
  padding: 6px 10px;
  cursor: pointer;
}
button:hover { background: #233044; }
button.primary { background: #1f6feb33; border-color: #58a6ff66; }
button.danger { background: #6d1f1f40; border-color: #ff6a5f55; }
select, input { min-height: 32px; padding: 5px 7px; }
input[type=range] { padding: 0; min-height: 0; width: 100%; }
textarea {
  width: 100%;
  min-height: 88px;
  resize: vertical;
  padding: 8px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
}
main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 460px;
  gap: 12px;
  padding: 12px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
}
.chart-panel { position: relative; min-height: calc(100vh - 84px); overflow: hidden; }
#chart { width: 100%; height: calc(100vh - 84px); display: block; cursor: crosshair; }
aside { padding: 12px; max-height: calc(100vh - 84px); overflow: auto; }
.toolbar { display: flex; gap: 8px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
.grid { display: grid; grid-template-columns: 96px 1fr; gap: 8px; align-items: center; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.section { padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,.08); }
.row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.muted { color: var(--muted); font-size: 12px; line-height: 1.55; }
.strong { color: #fff; font-weight: 700; }
.red { color: var(--red); }
.green { color: var(--green); }
.blue { color: var(--blue); }
.gold { color: var(--gold); }
.value { color: #d7e8ff; font-variant-numeric: tabular-nums; min-width: 54px; text-align: right; }
.sample {
  padding: 8px;
  border: 1px solid rgba(255,255,255,.08);
  background: #171d28;
  border-radius: 6px;
  margin: 8px 0;
  cursor: pointer;
}
.sample:hover { border-color: rgba(88,166,255,.45); }
.sample.active { border-color: rgba(23,224,138,.7); }
.sample-head {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: flex-start;
}
.sample button {
  min-height: 26px;
  padding: 3px 8px;
  font-size: 12px;
  flex: 0 0 auto;
}
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 2px 7px;
  border-radius: 999px;
  background: #111923;
  border: 1px solid rgba(255,255,255,.12);
  color: var(--muted);
  font-size: 12px;
}
.warn { color: #ffd37a; }
.ok { color: #9effc7; }
.axis { fill: #8694a7; font-size: 11px; }
.legend { font-size: 12px; font-weight: 700; }
.tip {
  position: fixed;
  display: none;
  pointer-events: none;
  z-index: 20;
  background: rgba(9,13,20,.96);
  border: 1px solid rgba(255,255,255,.18);
  padding: 8px 10px;
  border-radius: 6px;
  white-space: nowrap;
  font-size: 12px;
  line-height: 1.55;
}
@media (max-width: 1100px) {
  main { grid-template-columns: 1fr; }
  aside { max-height: none; }
  #chart, .chart-panel { height: 72vh; min-height: 520px; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>BTC Segment Cycle Workbench v19</h1>
    <div class="muted">比较 peak / bottom 前后不同时间段；手动保存高度比例、时间比例、平移和对比图。</div>
  </div>
  <div class="toolbar">
    <button id="saveSample" class="primary">保存样本</button>
    <button id="saveProject" class="primary">保存到项目文件</button>
    <button id="exportPng">导出 PNG</button>
    <button id="exportSvg">导出 SVG</button>
    <button id="exportCsv">导出 CSV</button>
    <button id="exportCurrent">下载当前 JSON</button>
    <button id="exportLibrary">下载样本库 JSON</button>
  </div>
</header>
<main>
  <section class="panel chart-panel">
    <svg id="chart" viewBox="0 0 1200 720" preserveAspectRatio="none"></svg>
  </section>
  <aside class="panel">
    <div class="section">
      <div class="grid">
        <label>实验 case</label><select id="caseSelect"></select>
        <label>窗口预设</label><select id="windowPreset"></select>
        <label>左周期</label><select id="leftCycle"></select>
        <label>右周期</label><select id="rightCycle"></select>
      </div>
      <div class="two" style="margin-top:8px">
        <div class="grid" style="grid-template-columns:80px 1fr">
          <label>左锚点</label><input id="leftAnchor" type="date" />
          <label>右锚点</label><input id="rightAnchor" type="date" />
        </div>
        <div class="grid" style="grid-template-columns:80px 1fr">
          <label>前天数</label><input id="preDays" type="number" min="0" max="1600" step="1" />
          <label>后天数</label><input id="postDays" type="number" min="0" max="1600" step="1" />
        </div>
      </div>
      <div class="row" style="margin-top:8px">
        <button id="newCase">新建当前组合</button>
        <button id="updateCase">更新 case</button>
        <button id="resetCase">恢复默认</button>
      </div>
    </div>

    <div class="section">
      <div class="grid">
        <label>高度比例</label><div class="row" style="flex-wrap:nowrap"><input id="amp" type="range" min="0.10" max="1.50" step="0.01" /><span id="ampText" class="value"></span></div>
        <label>时间比例</label><div class="row" style="flex-wrap:nowrap"><input id="timeScale" type="range" min="0.50" max="1.50" step="0.01" /><span id="timeText" class="value"></span></div>
        <label>水平平移</label><div class="row" style="flex-wrap:nowrap"><input id="shiftDays" type="range" min="-500" max="500" step="1" /><span id="shiftText" class="value"></span></div>
      </div>
      <div class="row" style="margin-top:8px">
        <button class="nudge" data-field="amp" data-d="-0.01">高度 -</button>
        <button class="nudge" data-field="amp" data-d="0.01">高度 +</button>
        <button class="nudge" data-field="timeScale" data-d="-0.01">时间 -</button>
        <button class="nudge" data-field="timeScale" data-d="0.01">时间 +</button>
        <button class="nudge" data-field="shiftDays" data-d="-5">左移</button>
        <button class="nudge" data-field="shiftDays" data-d="5">右移</button>
      </div>
      <div class="muted" style="margin-top:8px">
        红线 = 左周期，经 <span class="blue">rel_day × 时间比例 + 平移</span> 和 <span class="blue">log涨跌 × 高度比例</span> 变换；绿线 = 右周期原样。
      </div>
    </div>

    <div class="section">
      <div class="grid">
        <label>视觉评分</label>
        <select id="visualScore">
          <option value="">未评分</option>
          <option value="5">5 很像</option>
          <option value="4">4 可用</option>
          <option value="3">3 一般</option>
          <option value="2">2 勉强</option>
          <option value="1">1 不像</option>
        </select>
        <label>点标记</label>
        <div class="row">
          <label class="muted"><input id="showMarks" type="checkbox" checked /> 显示手标点</label>
          <label class="muted"><input id="connectMarks" type="checkbox" /> 连线</label>
        </div>
      </div>
      <textarea id="note" placeholder="记录为什么这个对齐像，或者哪里不像。"></textarea>
    </div>

    <div class="section">
      <div id="summary" class="muted"></div>
    </div>

    <div class="section">
      <div class="row" style="justify-content:space-between">
        <div class="strong">当前设置 JSON</div>
        <button id="copyCurrent">复制</button>
      </div>
      <textarea id="currentJson"></textarea>
    </div>

    <div class="section">
      <div class="row" style="justify-content:space-between">
        <div class="strong">已保存样本</div>
        <div class="row">
          <input id="importFile" type="file" accept="application/json" style="display:none" />
          <button id="importLibrary">导入 JSON</button>
          <button id="clearLibrary" class="danger">清空</button>
        </div>
      </div>
      <div id="sampleList"></div>
    </div>
  </aside>
</main>
<div id="tip" class="tip"></div>
<script>
const RAW = __PRICE_JSON__.map(d => ({
  dateText: d.date,
  t: new Date(d.date + "T00:00:00").getTime(),
  price: +d.price
}));
const MARKS = __MARKS_JSON__.map(d => ({
  dateText: d.date,
  t: new Date(d.date + "T00:00:00").getTime(),
  price: +d.price,
  type: d.type,
  n: d.n
}));
const CYCLES = __CYCLES_JSON__;
const WINDOWS = __WINDOWS_JSON__;
const DEFAULT_CASES = __CASES_JSON__;
const DAY = 86400000;
const NS = "http://www.w3.org/2000/svg";
const W = 1200, H = 720, M = { l: 70, r: 150, t: 28, b: 48 };
const STORE_CASES = "btc_segment_cycle_workbench_v19_cases";
const STORE_SAMPLES = "btc_segment_cycle_workbench_v19_samples";
const STORE_ACTIVE = "btc_segment_cycle_workbench_v19_active";
const SAVE_API = "http://127.0.0.1:8765/save_samples";
const DUPLICATE_TOLERANCE = { amp: 0.03, time: 0.03, shift: 10 };
const svg = document.getElementById("chart");
const tip = document.getElementById("tip");
let cases = loadCases();
let samples = dedupeSamples(loadSamples());
let activeIndex = loadActiveIndex();
let view = null;
let hoverG = null;

function loadCases() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_CASES));
    if (Array.isArray(saved) && saved.length) return saved;
  } catch (e) {}
  return structuredClone(DEFAULT_CASES);
}
function loadSamples() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_SAMPLES));
    if (Array.isArray(saved)) return saved;
  } catch (e) {}
  return [];
}
function loadActiveIndex() {
  const raw = +localStorage.getItem(STORE_ACTIVE);
  return Number.isFinite(raw) && raw >= 0 ? raw : 0;
}
function persistCases() {
  localStorage.setItem(STORE_CASES, JSON.stringify(cases));
  localStorage.setItem(STORE_ACTIVE, String(activeIndex));
}
function persistSamples() {
  localStorage.setItem(STORE_SAMPLES, JSON.stringify(samples));
}
function byId(id) { return document.getElementById(id); }
function val(id) { return byId(id).value; }
function num(id) { return +byId(id).value; }
function round4(v) { return Math.round(v * 10000) / 10000; }
function fmtUsd(v) { return "$" + Math.round(v).toLocaleString(); }
function fmtPct(logv) { return ((Math.exp(logv) - 1) * 100).toFixed(1) + "%"; }
function side(day) {
  const d = Math.round(day);
  if (d === 0) return "0d";
  return (d < 0 ? "前" : "后") + Math.abs(d) + "d";
}
function slug(s) { return String(s).replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, ""); }
function svgEl(parent, name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  parent.appendChild(node);
  return node;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function downloadText(filename, text, type = "application/json") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function projectPayload() {
  return {
    version: "v19",
    exported_at: new Date().toISOString(),
    samples,
    cases
  };
}
async function saveProjectLibrary(showAlert = true) {
  try {
    const resp = await fetch(SAVE_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(projectPayload())
    });
    const result = await resp.json();
    if (!resp.ok || !result.ok) throw new Error(result.error || "save failed");
    if (showAlert) {
      alert(`已保存到项目文件。\n样本 ${result.samples} 条\n${result.json}`);
    }
    return true;
  } catch (e) {
    if (showAlert) {
      alert("项目文件保存失败。请确认 segment_sample_save_server.py 正在运行；当前样本仍保存在浏览器 localStorage，可先点“下载样本库 JSON”备份。");
    }
    return false;
  }
}
function nearestRow(dateText) {
  const t = new Date(dateText + "T00:00:00").getTime();
  let lo = 0, hi = RAW.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (RAW[mid].t < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(RAW[lo - 1].t - t) <= Math.abs(RAW[lo].t - t)) lo--;
  return RAW[lo];
}
function cycle(id) { return CYCLES.find(c => c.id === id); }
function windowPreset(id) { return WINDOWS.find(w => w.id === id); }

function rowsFor(anchorDate, preDays, postDays, sideName, transform, params) {
  const anchor = nearestRow(anchorDate);
  const start = anchor.t - preDays * DAY;
  const end = anchor.t + postDays * DAY;
  return RAW
    .filter(r => r.t >= start && r.t <= end)
    .map(r => {
      const rel = Math.round((r.t - anchor.t) / DAY);
      const logNorm = Math.log(r.price / anchor.price);
      return {
        cycle: sideName,
        dateText: r.dateText,
        rel_day: rel,
        rel_plot: transform ? rel * params.time_scale + params.shift_days : rel,
        price: r.price,
        anchor_price: anchor.price,
        log_norm: logNorm,
        log_plot: transform ? logNorm * params.amp_scale : logNorm,
        pct_vs_anchor: (r.price / anchor.price - 1) * 100
      };
    });
}
function markRows(anchorDate, preDays, postDays, sideName, transform, params) {
  const anchor = nearestRow(anchorDate);
  const start = anchor.t - preDays * DAY;
  const end = anchor.t + postDays * DAY;
  return MARKS
    .filter(m => m.t >= start && m.t <= end)
    .map(m => {
      const rel = Math.round((m.t - anchor.t) / DAY);
      const logNorm = Math.log(m.price / anchor.price);
      return {
        cycle: sideName,
        dateText: m.dateText,
        rel_day: rel,
        rel_plot: transform ? rel * params.time_scale + params.shift_days : rel,
        price: m.price,
        log_norm: logNorm,
        log_plot: transform ? logNorm * params.amp_scale : logNorm,
        type: m.type,
        n: m.n
      };
    });
}
function currentCase() {
  const preset = windowPreset(val("windowPreset"));
  const windowLabel = preset?.label || "";
  const pairLabel = `${val("leftCycle")} -> ${val("rightCycle")}`;
  return {
    ...cases[activeIndex],
    name: `${windowLabel || "Manual window"}: ${pairLabel}`,
    left_cycle: val("leftCycle"),
    right_cycle: val("rightCycle"),
    window_id: val("windowPreset"),
    window_label: windowLabel,
    anchor_type: preset?.anchor || cases[activeIndex].anchor_type,
    left_anchor: val("leftAnchor"),
    right_anchor: val("rightAnchor"),
    pre_days: num("preDays"),
    post_days: num("postDays"),
    amp_scale: round4(num("amp")),
    time_scale: round4(num("timeScale")),
    shift_days: Math.round(num("shiftDays")),
    visual_score: val("visualScore"),
    user_note: val("note")
  };
}
function currentParams() {
  return {
    amp_scale: round4(num("amp")),
    time_scale: round4(num("timeScale")),
    shift_days: Math.round(num("shiftDays"))
  };
}
function buildCurrentRows() {
  const c = currentCase();
  const params = currentParams();
  return {
    c,
    leftAnchor: nearestRow(c.left_anchor),
    rightAnchor: nearestRow(c.right_anchor),
    left: rowsFor(c.left_anchor, c.pre_days, c.post_days, "left", true, params),
    right: rowsFor(c.right_anchor, c.pre_days, c.post_days, "right", false, params),
    leftMarks: markRows(c.left_anchor, c.pre_days, c.post_days, "left", true, params),
    rightMarks: markRows(c.right_anchor, c.pre_days, c.post_days, "right", false, params)
  };
}
function interpolateAt(rows, x) {
  if (!rows.length) return null;
  const sorted = rows.slice().sort((a, b) => a.rel_plot - b.rel_plot);
  if (x < sorted[0].rel_plot || x > sorted[sorted.length - 1].rel_plot) return null;
  let lo = 0;
  while (lo < sorted.length - 1 && sorted[lo + 1].rel_plot < x) lo++;
  const a = sorted[lo], b = sorted[Math.min(lo + 1, sorted.length - 1)];
  if (a.rel_plot === b.rel_plot) return a.log_plot;
  const t = (x - a.rel_plot) / (b.rel_plot - a.rel_plot);
  return a.log_plot + (b.log_plot - a.log_plot) * t;
}
function calcRmse(left, right) {
  if (left.length < 10 || right.length < 10) return { rmse: null, common: 0 };
  const lx0 = Math.min(...left.map(d => d.rel_plot));
  const lx1 = Math.max(...left.map(d => d.rel_plot));
  const common = right.filter(d => d.rel_plot >= lx0 && d.rel_plot <= lx1);
  if (common.length < 20) return { rmse: null, common: common.length };
  let s = 0;
  for (const r of common) {
    const y = interpolateAt(left, r.rel_plot);
    if (y == null) continue;
    const diff = y - r.log_plot;
    s += diff * diff;
  }
  return { rmse: Math.sqrt(s / common.length), common: common.length };
}
function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!Number.isFinite(span) || span <= 0) return [lo];
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 0.5; v += step) out.push(+v.toFixed(6));
  return out;
}
function drawPath(g, rows, px, py, color, width) {
  if (!rows.length) return;
  let d = "";
  rows.slice().sort((a, b) => a.rel_plot - b.rel_plot).forEach((p, i) => {
    d += (i ? "L" : "M") + px(p.rel_plot).toFixed(1) + "," + py(p.log_plot).toFixed(1);
  });
  svgEl(g, "path", {
    d,
    fill: "none",
    stroke: color,
    "stroke-width": width,
    opacity: .92,
    "stroke-linejoin": "round",
    "stroke-linecap": "round"
  });
}
function drawMarkers(g, rows, px, py, color, fill, connect) {
  if (!rows.length) return;
  const sorted = rows.slice().sort((a, b) => a.rel_plot - b.rel_plot);
  if (connect) {
    let d = "";
    sorted.forEach((p, i) => {
      d += (i ? "L" : "M") + px(p.rel_plot).toFixed(1) + "," + py(p.log_plot).toFixed(1);
    });
    svgEl(g, "path", { d, fill: "none", stroke: color, "stroke-width": 1.1, opacity: .45, "stroke-dasharray": "4 4" });
  }
  sorted.forEach(m => {
    svgEl(g, "circle", {
      cx: px(m.rel_plot),
      cy: py(m.log_plot),
      r: m.type === "H" ? 5.5 : 4.2,
      fill,
      stroke: color,
      "stroke-width": 1.6,
      opacity: .95
    });
  });
}
function draw() {
  clear(svg);
  const { c, leftAnchor, rightAnchor, left, right, leftMarks, rightMarks } = buildCurrentRows();
  if (!left.length || !right.length) {
    svgEl(svg, "rect", { x: 0, y: 0, width: W, height: H, fill: "#202631" });
    svgEl(svg, "text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "#a9b5c6", "font-size": 16 })
      .textContent = "这个窗口缺数据，换一个周期/窗口，或补更早历史数据。";
    updateSidebar(c, leftAnchor, rightAnchor, { rmse: null, common: 0 }, left.length, right.length);
    return;
  }
  const all = left.concat(right);
  let xmin = Math.min(...all.map(d => d.rel_plot));
  let xmax = Math.max(...all.map(d => d.rel_plot));
  let ymin = Math.min(...all.map(d => d.log_plot));
  let ymax = Math.max(...all.map(d => d.log_plot));
  const xpad = (xmax - xmin) * 0.025 || 1;
  const ypad = (ymax - ymin) * 0.08 || 0.03;
  xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;
  const px = x => M.l + (x - xmin) / (xmax - xmin) * (W - M.l - M.r);
  const py = y => H - M.b - (y - ymin) / (ymax - ymin) * (H - M.t - M.b);
  view = { left, right, leftMarks, rightMarks, px, py, xmin, xmax, ymin, ymax };

  const bg = svgEl(svg, "g", {});
  svgEl(bg, "rect", { x: 0, y: 0, width: W, height: H, fill: "#202631" });
  for (const v of niceTicks(ymin, ymax, 7)) {
    const y = py(v);
    svgEl(bg, "line", { x1: M.l, y1: y, x2: W - M.r, y2: y, stroke: "rgba(255,255,255,.065)" });
    svgEl(bg, "text", { x: M.l - 8, y: y + 4, "text-anchor": "end", "class": "axis" }).textContent = fmtPct(v);
  }
  for (const v of niceTicks(xmin, xmax, 9)) {
    const x = px(v);
    svgEl(bg, "line", { x1: x, y1: M.t, x2: x, y2: H - M.b, stroke: "rgba(255,255,255,.065)" });
    svgEl(bg, "text", { x, y: H - M.b + 17, "text-anchor": "middle", "class": "axis" }).textContent = Math.round(v) + "d";
  }
  svgEl(bg, "line", { x1: M.l, y1: py(0), x2: W - M.r, y2: py(0), stroke: "rgba(255,255,255,.22)" });
  svgEl(bg, "line", { x1: px(0), y1: M.t, x2: px(0), y2: H - M.b, stroke: "rgba(255,255,255,.35)", "stroke-dasharray": "4 4" });
  svgEl(bg, "text", { x: px(0) + 5, y: M.t + 15, "class": "axis" }).textContent = "anchor 0";

  const g = svgEl(svg, "g", {});
  drawPath(g, left, px, py, "#ff6a5f", 2.1);
  drawPath(g, right, px, py, "#17e08a", 2.6);

  if (byId("showMarks").checked) {
    const connect = byId("connectMarks").checked;
    drawMarkers(g, leftMarks, px, py, "#ff6a5f", "#202631", connect);
    drawMarkers(g, rightMarks, px, py, "#06452b", "#17e08a", connect);
  }

  svgEl(g, "text", { x: W - M.r + 10, y: M.t + 16, "class": "legend", fill: "#ff6a5f" })
    .textContent = "红 " + c.left_cycle + " " + c.anchor_type;
  svgEl(g, "text", { x: W - M.r + 10, y: M.t + 36, "class": "legend", fill: "#17e08a" })
    .textContent = "绿 " + c.right_cycle + " " + c.anchor_type;
  svgEl(g, "text", { x: W - M.r + 10, y: M.t + 58, "class": "axis" })
    .textContent = "height " + c.amp_scale.toFixed(2) + " / time " + c.time_scale.toFixed(2);

  hoverG = svgEl(svg, "g", {});
  updateSidebar(c, leftAnchor, rightAnchor, calcRmse(left, right), left.length, right.length, leftMarks.length, rightMarks.length);
}
function updateSidebar(c, leftAnchor, rightAnchor, score, leftN, rightN, leftMarkN = 0, rightMarkN = 0) {
  const cov = [];
  const leftStart = new Date(new Date(c.left_anchor + "T00:00:00").getTime() - c.pre_days * DAY).toISOString().slice(0, 10);
  const leftEnd = new Date(new Date(c.left_anchor + "T00:00:00").getTime() + c.post_days * DAY).toISOString().slice(0, 10);
  const rightStart = new Date(new Date(c.right_anchor + "T00:00:00").getTime() - c.pre_days * DAY).toISOString().slice(0, 10);
  const rightEnd = new Date(new Date(c.right_anchor + "T00:00:00").getTime() + c.post_days * DAY).toISOString().slice(0, 10);
  if (leftN < c.pre_days + c.post_days + 1) cov.push(`<span class="warn">左窗口数据 ${leftN}/${c.pre_days + c.post_days + 1}</span>`);
  if (rightN < c.pre_days + c.post_days + 1) cov.push(`<span class="warn">右窗口数据 ${rightN}/${c.pre_days + c.post_days + 1}</span>`);
  const rmseText = score.rmse == null ? "overlap too small" : score.rmse.toFixed(5);
  byId("summary").innerHTML = `
    <div><span class="red">红</span> ${c.left_cycle}: ${c.left_anchor} (${fmtUsd(leftAnchor.price)})，窗口 ${leftStart} → ${leftEnd}</div>
    <div><span class="green">绿</span> ${c.right_cycle}: ${c.right_anchor} (${fmtUsd(rightAnchor.price)})，窗口 ${rightStart} → ${rightEnd}</div>
    <div>参数：高度 ${c.amp_scale.toFixed(2)} · 时间 ${c.time_scale.toFixed(2)} · 平移 ${c.shift_days}d · overlap ${score.common}d · RMSE ${rmseText}</div>
    <div>手标点：红 ${leftMarkN} 个，绿 ${rightMarkN} 个。${cov.join(" · ")}</div>
    <div class="row" style="margin-top:6px">
      <span class="pill">${c.window_label}</span>
      <span class="pill">${c.anchor_type}</span>
      <span class="pill">${c.pair || c.left_cycle + "->" + c.right_cycle}</span>
    </div>`;
  byId("currentJson").value = JSON.stringify(currentRecord(score), null, 2);
  renderSamples();
}
function currentRecord(score) {
  const c = currentCase();
  return {
    version: "v19",
    saved_at: new Date().toISOString(),
    case_id: c.id,
    case_name: c.name,
    pair: c.left_cycle + "->" + c.right_cycle,
    left_cycle: c.left_cycle,
    right_cycle: c.right_cycle,
    anchor_type: c.anchor_type,
    window_id: c.window_id,
    window_label: c.window_label,
    left_anchor: c.left_anchor,
    right_anchor: c.right_anchor,
    pre_days: c.pre_days,
    post_days: c.post_days,
    amp_scale: c.amp_scale,
    time_scale: c.time_scale,
    shift_days: c.shift_days,
    rmse: score && score.rmse != null ? round4(score.rmse) : null,
    overlap_days: score ? score.common : null,
    visual_score: val("visualScore"),
    note: val("note")
  };
}
function comparisonKey(s) {
  return [
    s.left_cycle || "",
    s.right_cycle || "",
    s.anchor_type || "",
    s.window_id || "",
    s.left_anchor || "",
    s.right_anchor || "",
    s.pre_days ?? "",
    s.post_days ?? ""
  ].join("|");
}
function paramsAreNear(a, b) {
  return Math.abs((+a.amp_scale || 0) - (+b.amp_scale || 0)) <= DUPLICATE_TOLERANCE.amp
    && Math.abs((+a.time_scale || 0) - (+b.time_scale || 0)) <= DUPLICATE_TOLERANCE.time
    && Math.abs((+a.shift_days || 0) - (+b.shift_days || 0)) <= DUPLICATE_TOLERANCE.shift;
}
function findDuplicateSample(record) {
  const key = comparisonKey(record);
  return samples.findIndex(s => comparisonKey(s) === key && paramsAreNear(s, record));
}
function upsertSample(record) {
  const duplicateIndex = findDuplicateSample(record);
  if (duplicateIndex >= 0) {
    samples[duplicateIndex] = {
      ...samples[duplicateIndex],
      ...record,
      replaced_sample: samples[duplicateIndex].saved_at || null,
      duplicate_policy: `same comparison and params within amp ${DUPLICATE_TOLERANCE.amp}, time ${DUPLICATE_TOLERANCE.time}, shift ${DUPLICATE_TOLERANCE.shift}d`
    };
    return { action: "replaced", index: duplicateIndex };
  }
  samples.push(record);
  return { action: "added", index: samples.length - 1 };
}
function dedupeSamples(list) {
  const out = [];
  for (const sample of list) {
    const key = comparisonKey(sample);
    const duplicateIndex = out.findIndex(s => comparisonKey(s) === key && paramsAreNear(s, sample));
    if (duplicateIndex >= 0) {
      out[duplicateIndex] = {
        ...out[duplicateIndex],
        ...sample,
        replaced_sample: out[duplicateIndex].saved_at || null,
        duplicate_policy: `same comparison and params within amp ${DUPLICATE_TOLERANCE.amp}, time ${DUPLICATE_TOLERANCE.time}, shift ${DUPLICATE_TOLERANCE.shift}d`
      };
    } else {
      out.push(sample);
    }
  }
  return out;
}
function populateSelects() {
  byId("leftCycle").innerHTML = CYCLES.map(c => `<option value="${c.id}">${c.label}</option>`).join("");
  byId("rightCycle").innerHTML = CYCLES.map(c => `<option value="${c.id}">${c.label}</option>`).join("");
  byId("windowPreset").innerHTML = WINDOWS.map(w => `<option value="${w.id}">${w.label}</option>`).join("");
  renderCaseSelect();
}
function renderCaseSelect() {
  byId("caseSelect").innerHTML = cases.map((c, i) => `<option value="${i}">${String(i + 1).padStart(2, "0")} · ${c.name}</option>`).join("");
  if (activeIndex >= cases.length) activeIndex = 0;
  byId("caseSelect").value = String(activeIndex);
}
function setControls(c) {
  byId("leftCycle").value = c.left_cycle;
  byId("rightCycle").value = c.right_cycle;
  byId("windowPreset").value = c.window_id;
  byId("leftAnchor").value = c.left_anchor;
  byId("rightAnchor").value = c.right_anchor;
  byId("preDays").value = c.pre_days;
  byId("postDays").value = c.post_days;
  byId("amp").value = c.amp_scale;
  byId("timeScale").value = c.time_scale;
  byId("shiftDays").value = c.shift_days;
  byId("visualScore").value = c.visual_score || "";
  byId("note").value = c.user_note || "";
  updateTexts();
}
function updateTexts() {
  byId("ampText").textContent = (+val("amp")).toFixed(2);
  byId("timeText").textContent = (+val("timeScale")).toFixed(2);
  byId("shiftText").textContent = val("shiftDays") + "d";
}
function syncControls(saveIntoCase = true) {
  const preset = windowPreset(val("windowPreset"));
  if (preset && document.activeElement === byId("windowPreset")) {
    byId("preDays").value = preset.pre_days;
    byId("postDays").value = preset.post_days;
    byId("leftAnchor").value = cycle(val("leftCycle"))[preset.anchor];
    byId("rightAnchor").value = cycle(val("rightCycle"))[preset.anchor];
  }
  if ((document.activeElement === byId("leftCycle") || document.activeElement === byId("rightCycle")) && preset) {
    byId("leftAnchor").value = cycle(val("leftCycle"))[preset.anchor];
    byId("rightAnchor").value = cycle(val("rightCycle"))[preset.anchor];
  }
  if (saveIntoCase) {
    cases[activeIndex] = currentCase();
    persistCases();
    renderCaseSelect();
  }
  updateTexts();
  draw();
}
function renderSamples() {
  const box = byId("sampleList");
  if (!samples.length) {
    box.innerHTML = `<div class="muted" style="padding:8px 0">还没有保存样本。先把图调到你觉得像，再点“保存样本”。</div>`;
    return;
  }
  box.innerHTML = samples.slice().reverse().map((s, revIndex) => {
    const i = samples.length - 1 - revIndex;
    const score = s.visual_score ? ` · score ${s.visual_score}` : "";
    return `<div class="sample" data-i="${i}">
      <div class="sample-head">
        <div class="strong">${s.case_name || s.case_id}</div>
        <button class="delete-sample danger" data-i="${i}" title="删除这个样本">删除</button>
      </div>
      <div class="muted">${s.pair} · ${s.window_label} · ${s.left_anchor} → ${s.right_anchor}</div>
      <div class="muted">height ${(+s.amp_scale).toFixed(2)} · time ${(+s.time_scale).toFixed(2)} · shift ${s.shift_days}d · rmse ${s.rmse ?? "-"}${score}</div>
    </div>`;
  }).join("");
  box.querySelectorAll(".sample").forEach(node => {
    node.onclick = () => loadSample(+node.dataset.i);
  });
  box.querySelectorAll(".delete-sample").forEach(btn => {
    btn.onclick = ev => {
      ev.stopPropagation();
      const i = +btn.dataset.i;
      const s = samples[i];
      if (!s) return;
      if (!confirm(`删除这个样本？\n${s.case_name || s.case_id}`)) return;
      samples.splice(i, 1);
      persistSamples();
      saveProjectLibrary(false);
      renderSamples();
    };
  });
}
function loadSample(i) {
  const s = samples[i];
  if (!s) return;
  const idx = cases.findIndex(c => c.id === s.case_id);
  if (idx >= 0) {
    activeIndex = idx;
  } else {
    cases.push({
      id: s.case_id || "imported_" + Date.now(),
      name: s.case_name || "imported sample",
      left_cycle: s.left_cycle,
      right_cycle: s.right_cycle,
      window_id: s.window_id,
      anchor_type: s.anchor_type,
      left_anchor: s.left_anchor,
      right_anchor: s.right_anchor,
      pre_days: s.pre_days,
      post_days: s.post_days,
      amp_scale: s.amp_scale,
      time_scale: s.time_scale,
      shift_days: s.shift_days
    });
    activeIndex = cases.length - 1;
  }
  persistCases();
  renderCaseSelect();
  setControls({ ...cases[activeIndex], ...s, user_note: s.note });
  draw();
}
function activeScore() {
  const built = buildCurrentRows();
  return calcRmse(built.left, built.right);
}
function exportOverlayCsv() {
  const built = buildCurrentRows();
  const rows = built.left.concat(built.right);
  const head = "cycle,date,rel_day,rel_plot,price,anchor_price,pct_vs_anchor,log_norm,log_plot\n";
  const body = rows.map(r => [
    r.cycle,
    r.dateText,
    r.rel_day,
    r.rel_plot.toFixed(4),
    r.price,
    r.anchor_price,
    r.pct_vs_anchor.toFixed(4),
    r.log_norm.toFixed(6),
    r.log_plot.toFixed(6)
  ].join(",")).join("\n");
  const c = currentCase();
  downloadText(slug(c.id || "overlay") + "_overlay.csv", head + body, "text/csv");
}
function exportSvgText() {
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", NS);
  clone.setAttribute("width", "1200");
  clone.setAttribute("height", "720");
  const style = document.createElementNS(NS, "style");
  style.textContent = `.axis{fill:#8694a7;font-size:11px;font-family:Segoe UI,Arial,sans-serif}.legend{font-size:12px;font-weight:700;font-family:Segoe UI,Arial,sans-serif}`;
  clone.insertBefore(style, clone.firstChild);
  return `<?xml version="1.0" encoding="UTF-8"?>\n` + new XMLSerializer().serializeToString(clone);
}
function exportSvg() {
  downloadText(slug(currentCase().id || "btc_overlay") + ".svg", exportSvgText(), "image/svg+xml");
}
function exportPng() {
  const svgText = exportSvgText();
  const img = new Image();
  const url = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgText);
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = 1800;
    canvas.height = 1080;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#202631";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      const out = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = out;
      a.download = slug(currentCase().id || "btc_overlay") + ".png";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(out);
    }, "image/png");
  };
  img.src = url;
}
function svgPt(ev) {
  const rect = svg.getBoundingClientRect();
  return {
    x: (ev.clientX - rect.left) / rect.width * W,
    y: (ev.clientY - rect.top) / rect.height * H,
    cx: ev.clientX,
    cy: ev.clientY
  };
}
function nearestByPlot(rows, relPlot) {
  if (!rows.length) return null;
  let best = rows[0], bd = Infinity;
  for (const p of rows) {
    const d = Math.abs(p.rel_plot - relPlot);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}
function hideHover() {
  tip.style.display = "none";
  if (hoverG) clear(hoverG);
}
svg.addEventListener("mousemove", ev => {
  if (!view) return;
  const p = svgPt(ev);
  if (p.x < M.l || p.x > W - M.r || p.y < M.t || p.y > H - M.b) {
    hideHover();
    return;
  }
  const relPlot = view.xmin + (p.x - M.l) / (W - M.l - M.r) * (view.xmax - view.xmin);
  const l = nearestByPlot(view.left, relPlot);
  const r = nearestByPlot(view.right, relPlot);
  if (!l || !r) return;
  clear(hoverG);
  svgEl(hoverG, "line", { x1: p.x, y1: M.t, x2: p.x, y2: H - M.b, stroke: "rgba(255,255,255,.22)" });
  svgEl(hoverG, "circle", { cx: view.px(l.rel_plot), cy: view.py(l.log_plot), r: 4, fill: "#ff6a5f" });
  svgEl(hoverG, "circle", { cx: view.px(r.rel_plot), cy: view.py(r.log_plot), r: 4, fill: "#17e08a" });
  tip.style.display = "block";
  tip.style.left = (p.cx + 14) + "px";
  tip.style.top = (p.cy + 14) + "px";
  tip.innerHTML = `
    <div>对齐横坐标 ≈ ${side(relPlot)}</div>
    <div class="red">红 ${l.dateText} · ${side(l.rel_day)} · ${fmtUsd(l.price)} · ${l.pct_vs_anchor.toFixed(1)}%</div>
    <div class="green">绿 ${r.dateText} · ${side(r.rel_day)} · ${fmtUsd(r.price)} · ${r.pct_vs_anchor.toFixed(1)}%</div>`;
});
svg.addEventListener("mouseleave", hideHover);

byId("caseSelect").addEventListener("change", e => {
  activeIndex = +e.target.value;
  localStorage.setItem(STORE_ACTIVE, String(activeIndex));
  setControls(cases[activeIndex]);
  draw();
});
["windowPreset", "leftCycle", "rightCycle", "leftAnchor", "rightAnchor", "preDays", "postDays", "amp", "timeScale", "shiftDays", "visualScore", "note", "showMarks", "connectMarks"].forEach(id => {
  byId(id).addEventListener("input", () => syncControls(id !== "showMarks" && id !== "connectMarks"));
  byId(id).addEventListener("change", () => syncControls(id !== "showMarks" && id !== "connectMarks"));
});
document.querySelectorAll(".nudge").forEach(btn => {
  btn.onclick = () => {
    const field = byId(btn.dataset.field);
    field.value = +field.value + +btn.dataset.d;
    syncControls(true);
  };
});
byId("newCase").onclick = () => {
  const c = currentCase();
  const timestamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  c.id = `manual_${slug(c.window_id)}_${slug(c.left_cycle)}_to_${slug(c.right_cycle)}_${timestamp}`;
  c.name = `${c.window_label}: ${c.left_cycle} -> ${c.right_cycle} manual`;
  cases.push(c);
  activeIndex = cases.length - 1;
  persistCases();
  renderCaseSelect();
  setControls(c);
  draw();
};
byId("updateCase").onclick = () => {
  cases[activeIndex] = currentCase();
  persistCases();
  renderCaseSelect();
  draw();
};
byId("resetCase").onclick = () => {
  const original = DEFAULT_CASES.find(c => c.id === cases[activeIndex].id) || DEFAULT_CASES[0];
  cases[activeIndex] = structuredClone(original);
  persistCases();
  setControls(cases[activeIndex]);
  draw();
};
byId("saveSample").onclick = () => {
  const record = currentRecord(activeScore());
  const result = upsertSample(record);
  persistSamples();
  saveProjectLibrary(false);
  renderSamples();
  byId("currentJson").value = JSON.stringify(record, null, 2);
  if (result.action === "replaced") {
    alert("已顶掉一条参数很接近的旧样本，没有新增重复记录。");
  }
};
byId("exportLibrary").onclick = () => {
  downloadText("btc_segment_cycle_samples_v19.json", JSON.stringify(projectPayload(), null, 2));
};
byId("saveProject").onclick = () => saveProjectLibrary(true);
byId("importLibrary").onclick = () => byId("importFile").click();
byId("importFile").addEventListener("change", async ev => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  const text = await file.text();
  try {
    const parsed = JSON.parse(text);
    const incoming = Array.isArray(parsed) ? parsed : (parsed.samples || []);
    if (!Array.isArray(incoming)) throw new Error("No samples array");
    samples = dedupeSamples(samples.concat(incoming));
    persistSamples();
    saveProjectLibrary(false);
    renderSamples();
  } catch (e) {
    alert("JSON 导入失败");
  }
});
byId("clearLibrary").onclick = () => {
  if (!confirm("清空浏览器里保存的样本库？")) return;
  samples = [];
  persistSamples();
  saveProjectLibrary(false);
  renderSamples();
};
byId("exportCsv").onclick = exportOverlayCsv;
byId("exportSvg").onclick = exportSvg;
byId("exportPng").onclick = exportPng;
byId("exportCurrent").onclick = () => {
  const record = currentRecord(activeScore());
  downloadText(slug(record.case_id || "btc_segment") + "_settings.json", JSON.stringify(record, null, 2));
};
byId("copyCurrent").onclick = async () => {
  try { await navigator.clipboard.writeText(val("currentJson")); } catch (e) {}
};

populateSelects();
setControls(cases[activeIndex] || cases[0]);
draw();
</script>
</body>
</html>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series, rows = load_price_rows()
    marks = load_marks()
    cases = build_cases(series)
    write_case_ledger(cases)
    write_runbook()
    write_html(rows, marks, cases)
    print(f"data: {series.index.min().date()} -> {series.index.max().date()} ({len(series)} days)")
    print(f"cases: {len(cases)}")
    print(f"marks: {len(marks)}")
    print(f"html: {HTML}")
    print(f"ledger: {CASE_LEDGER}")


if __name__ == "__main__":
    main()
