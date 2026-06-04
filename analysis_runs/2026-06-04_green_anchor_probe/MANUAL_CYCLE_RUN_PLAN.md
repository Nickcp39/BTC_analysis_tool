# BTC 人工周期结构对比运行计划

## 分工
- Codex：生成候选对比、维护脚本、记录参数、根据人工结果反推规则。
- 用户：只负责看图、调整参数、保存认为对的设置。

## 每轮流程
1. 打开 `green_anchor_manual_workbench_v17.html`。
2. 选择一个 case。
3. 调整高度系数、时间 scale、水平平移、锚点和窗口。
4. 觉得对了就点保存，并下载规则库 JSON。
5. 把 JSON 给 Codex，Codex 记录到 `manual_review_log_v1.csv` 并生成下一轮候选。

## 当前候选 case
1. Peak: 2021 top -> 2025 top
   - left: `2021-11-08`
   - right: `2025-10-05`
   - question: Does the full peak-before and peak-after path visually align?
2. Bottom: 2022 bear low -> 2026 local low
   - left: `2022-11-21`
   - right: `2026-02-05`
   - question: Does bottoming structure before/after the low align?
3. Lower high: 2022 rebound high -> 2025 lower high
   - left: `2022-03-29`
   - right: `2025-11-10`
   - question: Does the rebound-high / second-high structure align?
4. Capitulation: 2022 Jun low -> 2026 Feb/Mar low
   - left: `2022-06-18`
   - right: `2026-02-22`
   - question: Does the sharp capitulation and rebound segment align?
5. Rebound: 2022 Aug high -> 2026 May high
   - left: `2022-08-14`
   - right: `2026-05-08`
   - question: Does the post-bottom rebound structure align?

## 产物
- `manual_cycle_cases_v1.json`: 候选规则库
- `manual_review_log_v1.csv`: 人工修正记录表
- `green_anchor_manual_workbench_v17.html`: 可视化调参工作台