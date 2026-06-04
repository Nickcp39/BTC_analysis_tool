from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


CASES = [
    {
        "id": "peak_2021_to_2025",
        "name": "Peak: 2021 top -> 2025 top",
        "left_anchor": "2021-11-08",
        "right_anchor": "2025-10-05",
        "pre_days": 260,
        "post_days": 520,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": -59,
        "status": "needs_manual_review",
        "question": "Does the full peak-before and peak-after path visually align?",
    },
    {
        "id": "bear_bottom_2022_to_2026",
        "name": "Bottom: 2022 bear low -> 2026 local low",
        "left_anchor": "2022-11-21",
        "right_anchor": "2026-02-05",
        "pre_days": 360,
        "post_days": 260,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
        "status": "needs_manual_review",
        "question": "Does bottoming structure before/after the low align?",
    },
    {
        "id": "lower_high_2022_to_2025",
        "name": "Lower high: 2022 rebound high -> 2025 lower high",
        "left_anchor": "2022-03-29",
        "right_anchor": "2025-11-10",
        "pre_days": 180,
        "post_days": 260,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
        "status": "needs_manual_review",
        "question": "Does the rebound-high / second-high structure align?",
    },
    {
        "id": "capitulation_2022_to_2026",
        "name": "Capitulation: 2022 Jun low -> 2026 Feb/Mar low",
        "left_anchor": "2022-06-18",
        "right_anchor": "2026-02-22",
        "pre_days": 220,
        "post_days": 220,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
        "status": "needs_manual_review",
        "question": "Does the sharp capitulation and rebound segment align?",
    },
    {
        "id": "rebound_2022_to_2026",
        "name": "Rebound: 2022 Aug high -> 2026 May high",
        "left_anchor": "2022-08-14",
        "right_anchor": "2026-05-08",
        "pre_days": 180,
        "post_days": 180,
        "amp_scale": 0.50,
        "time_scale": 1.00,
        "shift_days": 0,
        "status": "needs_manual_review",
        "question": "Does the post-bottom rebound structure align?",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rules = {
        "version": "manual_cycle_run_v1",
        "created_for": "BTC cycle structure comparison",
        "workflow": [
            "Codex generates candidate comparison cases.",
            "User opens v17 workbench and manually adjusts amp_scale, time_scale, shift_days, anchors, and windows.",
            "User saves/downloads JSON after each good alignment.",
            "Codex imports reviewed cases and derives default rules/search ranges.",
        ],
        "default_constraints": {
            "amp_scale_start": 0.50,
            "time_scale_manual_range": [0.70, 1.30],
            "shift_days_manual_range": [-360, 360],
            "rule": "time_scale is global per case; no segment warp unless user explicitly validates it",
        },
        "cases": CASES,
    }
    (OUT / "manual_cycle_cases_v1.json").write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for i, case in enumerate(CASES, start=1):
        rows.append(
            {
                "order": i,
                "case_id": case["id"],
                "name": case["name"],
                "left_anchor": case["left_anchor"],
                "right_anchor": case["right_anchor"],
                "pre_days": case["pre_days"],
                "post_days": case["post_days"],
                "initial_amp_scale": case["amp_scale"],
                "initial_time_scale": case["time_scale"],
                "initial_shift_days": case["shift_days"],
                "manual_amp_scale": "",
                "manual_time_scale": "",
                "manual_shift_days": "",
                "manual_left_anchor": "",
                "manual_right_anchor": "",
                "visual_grade_1_5": "",
                "notes": "",
                "status": case["status"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "manual_review_log_v1.csv", index=False, encoding="utf-8-sig")

    md = [
        "# BTC 人工周期结构对比运行计划",
        "",
        "## 分工",
        "- Codex：生成候选对比、维护脚本、记录参数、根据人工结果反推规则。",
        "- 用户：只负责看图、调整参数、保存认为对的设置。",
        "",
        "## 每轮流程",
        "1. 打开 `green_anchor_manual_workbench_v17.html`。",
        "2. 选择一个 case。",
        "3. 调整高度系数、时间 scale、水平平移、锚点和窗口。",
        "4. 觉得对了就点保存，并下载规则库 JSON。",
        "5. 把 JSON 给 Codex，Codex 记录到 `manual_review_log_v1.csv` 并生成下一轮候选。",
        "",
        "## 当前候选 case",
    ]
    for i, case in enumerate(CASES, start=1):
        md.extend(
            [
                f"{i}. {case['name']}",
                f"   - left: `{case['left_anchor']}`",
                f"   - right: `{case['right_anchor']}`",
                f"   - question: {case['question']}",
            ]
        )
    md.extend(
        [
            "",
            "## 产物",
            "- `manual_cycle_cases_v1.json`: 候选规则库",
            "- `manual_review_log_v1.csv`: 人工修正记录表",
            "- `green_anchor_manual_workbench_v17.html`: 可视化调参工作台",
        ]
    )
    (OUT / "MANUAL_CYCLE_RUN_PLAN.md").write_text("\n".join(md), encoding="utf-8")

    print(OUT / "manual_cycle_cases_v1.json")
    print(OUT / "manual_review_log_v1.csv")
    print(OUT / "MANUAL_CYCLE_RUN_PLAN.md")


if __name__ == "__main__":
    main()
