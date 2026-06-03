# Trend Rework Notes

这版重做回旧趋势图语言：曲线、锚点、log轴、历史路径映射，不再使用卡片式模型图。

## 核心结论
- 原始多模型均值：2026-10-22 / $54,283
- AHR999 地板修正后中心：2026-10-22 / $54,999
- AHR修正后核心价格区间：$49,482 ~ $65,190
- 时间窗口仍以 2026-10-01 → 2026-11-05 为主。

## 上次 peak 预测复盘
- 时间最准确：MedianCenter，2025-09-28，实际峰值 2025-10-05，误差 -7 天。
- 价格最准确：ModelA nominal 1%，预测 $122,062，实际峰值 $124,720，误差约 -2.1%。

## 输出图
- png/average_multi_model_bottom_trend.png
- png/model_1_post_peak_time_clock.png
- png/model_1b_post_peak_scaled_coefficient.png
- png/model_2_sqrt_log_vol_replay.png
- png/model_3_bottom_peak_ratio_log_trend.png
- png/model_4_ahr999_value_floor.png
- png/model_4b_ahr999_implied_price_trend.png
- png/peak_price_accuracy_trend.png
- png/peak_time_accuracy_trend.png
