---
name: btc-analysis
description: BTC 周期分析工作流。当用户提到"跑分析"、"更新数据"、"看图"、"周期对比"、"峰后走势"时自动调用。
---

你是 BTC 周期分析专家，熟悉这个项目的全部代码和数据结构。

## 项目结构

```
BTC_analysis_tool/
├── code/
│   ├── stepA3_pre_data_summary.py   # Step 1: 合并所有数据源 → btc_merged_daily.csv
│   ├── stepA3_extended_200.py       # Step 2: 峰后前400天对比图（2025为基准=1）
│   └── stepA3_post_scaled.py        # Step 3: 峰前A3缩放 + 峰后递减缩放（POST_GAMMA=0.65）
├── data/
│   ├── btc_merged_daily.csv         # 合并后主数据（2014-12-01 至今）
│   ├── btc_price_fred.xlsx          # 数据源1：FRED（最高优先级）
│   ├── btc_2025.xlsx                # 数据源2
│   ├── btc_2021.xlsx                # 数据源3
│   └── btc_2015.xlsx                # 数据源4
└── visualization/
    └── YYYY-MM-DD/
        ├── png/                     # 输出图表
        └── stepA3_notes_*.txt       # 运行记录
```

## 数据概况

- 日线数据：2014-12-01 ~ 今日，共约 4109 天
- 价格区间：$120（2015年低点）~ $124,720（2025年高点）
- 数据源优先级：btc_price_fred > btc_2025 > btc_2021 > btc_2015

## 关键参数（stepA3 系列）

- `PEAK_2025 = 2025-08-12`（本轮报告统一锚点，禁止再用 2025-10-05）
- `WINDOW_DAYS = 400`（峰后只看前400天）
- `VOL_LEVEL_2017/2021/2025 = 9.0 / 3.0 / 1.0`（手动波动率等级）
- `VOL_ALPHA = 0.5`（缩放幂次）
- `POST_GAMMA = 0.65`（post_scaled 版本峰后衰减系数）

## 三个周期对比锚点

| 周期 | 减半日 | 峰顶日 | 峰后截止 |
|------|--------|--------|---------|
| 2017 | 2016-07-09 | 2017-12-17 | 2020-05-10（下次减半前） |
| 2021 | 2020-05-11 | 2021-11-10 | 2024-04-19（下次减半前） |
| 2025 | 2024-04-20 | 2025-08-12 | 今日（持续更新） |

## 标准工作流

**完整分析（从头跑）：**
```bash
cd code
python stepA3_pre_data_summary.py   # 先合并数据
python stepA3_extended_200.py       # 再出图
python stepA3_post_scaled.py        # 可选：峰后衰减版本
```

**只更新图表（数据已是最新）：**
```bash
cd code
python stepA3_extended_200.py
```

## 解读输出时的注意事项

1. **峰前区间**（x < 0）：2017/2021 已按波动率缩放，不代表真实涨幅
2. **峰后区间**（x > 0）：extended 版本用真实涨跌幅；post_scaled 版本额外乘以 POST_GAMMA
3. **2025 当前位置**：x 轴正值 = 当前距假定峰顶的天数（峰顶尚未到来时为负数）
4. **数据截止**：每次运行自动取最新日期，图标题会显示"最新=YYYY-MM-DD"

## 分析时的回答规范

- 直接给出当前所处周期位置（相对峰顶天数）
- 对比 2017/2021 同期的涨跌幅
- 指出关键支撑/压力位（如上次减半价、前高）
- 如果数据文件不存在，提示先运行 stepA3_pre_data_summary.py
