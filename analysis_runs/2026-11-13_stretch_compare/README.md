# 多周期拉伸对比图（2026-11-13）

和 10-01 / 11-06 报告「多周期价格走势横向对比」同一种图：历史一轮的曲线按 time（时间拉伸）、shift（平移）、amp（幅度拉伸）变换后叠到下一轮。
这一版的参数用点位近似拟合（配对点来自 ../2026-11-13_manual_point_compare/），并画出整条曲线拟合做对照；历史曲线越过最新一天的部分画成虚线，只是参照，不是预测。

运行：`python make_stretch_grid.py` → `png/stretch_grid.png`（六个窗口）、`png/stretch_A.png`…`stretch_F.png`、`stretch_params.csv`、`stretch_numbers.txt`。
不做 2017 → 2025 的直接对比（2021 ↔ 2025 为主，2017 ↔ 2021 作桥）。
