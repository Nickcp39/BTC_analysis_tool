# core_points 2026-10-01 快照

stepD1–D14 用 2026-10-01 数据（data/btc_merged_daily.csv）重跑后的输出。
脚本本身仍写到 output/core_points/；跑完后整体复制到本目录，并把 output/core_points/ 恢复为 7 月版（git checkout，删掉新生成的 year_2026.png），保证旧版本不被改动。
stepD5（手动点选 GUI）没有重跑；manual_points.csv 沿用 7 月手标点。
stepD12/D13/D14 只打印文字结论，原文见 visualization/2026-10-01/stepD_logs/。日志里的图片路径写的是 output/core_points/，实际文件在本目录。
与 09-22 快照相比：stepD10/D12/D13/D14 的结论逐字相同（只依赖 2025 顶部之前的结构和固定参数）；变化只在画到最新日期的图和 pip 表（zigzag_*.png、pip_10/14 的 csv 和 png、by_year/2026.png、post_top_align.png、year_2026.png、marked_workbench_v18.html）；zigzag_*.csv、review_by_year.csv 与 09-22 版相同（最近 9 天没有形成新的转折点）。
