# core_points 2026-11-06 快照

stepD1–D14 用 2026-11-06 数据（data/btc_merged_daily.csv：FRED 到 11-06 收盘，另有 Yahoo 11-07 盘中一点）重跑后的输出。
脚本本身仍写到 output/core_points/；跑完后整体复制到本目录，并把 output/core_points/ 恢复为 7 月版（git checkout，删掉新生成的 year_2026.png），保证旧版本不被改动。
stepD5（手动点选 GUI）没有重跑；manual_points.csv 沿用 7 月手标点。
stepD12/D13/D14 只打印文字结论，原文见 visualization/2026-11-06/stepD_logs/。日志里的图片路径写的是 output/core_points/，实际文件在本目录。

与 10-01 快照相比：
- stepD10/D12/D13/D14 的日志逐字相同（只依赖 2025 顶部之前的结构和固定参数）。
- zigzag 25% / 35%：底以来的高点从 2026-09-21 $86,397 换成 2026-10-04 $86,424（FRED 数据里 10-04 收盘略高）。zigzag 50% 不变。
- review_by_year.csv：2026 年多了一个低点 2026-11-07 $75,750。注意这是 Yahoo 11-07 盘中点，算法把序列最后一点当作暂定转折，不是确认的低点。
- 其余变化只在画到最新日期的图和 pip 表（pip_10/14、by_year/2026.png、post_top_align.png、year_2026.png、marked_workbench_v18.html）。
