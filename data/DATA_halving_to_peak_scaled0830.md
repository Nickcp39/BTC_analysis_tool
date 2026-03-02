# 图 halving_to_peak_scaled0830 数据位置说明

画图代码：**btc price analysis V3.ipynb**（减半→峰值对齐、波动退火后的那一大段 cell）

---

## 输入数据（画图前读取）

| 用途 | 文件 | 位置（统一在 data 下） |
|------|------|------------------------|
| 2016 减半→2017 峰值 | btc_2015.xlsx | `data/btc_2015.xlsx` |
| 2020 减半→2021 峰值 | btc_2021.xlsx | `data/btc_2021.xlsx` |
| 2024 减半→2025 峰值 | btc_2025.xlsx | `data/btc_2025.xlsx` |

说明：每表两列（日期 + 价格 USD/BTC），由 `read_two_col_excel` 读入后转日频，再按减半/峰值窗口截取。

---

## 输出数据（该 cell 生成）

| 内容 | 文件 | 位置（统一在 data 下） |
|------|------|------------------------|
| 对齐后的百分比序列 CSV | halving_to_peak_aligned0830.csv | `data/halving_to_peak_aligned0830.csv` |
| 指标文本 | halving_to_peak_metrics0830.txt | `data/halving_to_peak_metrics0830.txt` |
| 图：退火（缩放）后 | halving_to_peak_scaled0830.png | `data/halving_to_peak_scaled0830.png` |
| 图：原始幅度 | halving_to_peak_unscaled0830.png | `data/halving_to_peak_unscaled0830.png` |

---

## 笔记本路径说明

- 该图的 cell 已改为从 `data/` 读入（F15/F21/F25），并写入 `data/`（OUTDIR = `data`）。
- 历史输出曾保存在 `btc_fit_outputs09142025/`，已复制到 `data/` 并存档。
