# BTC Analysis Tool - 项目记忆

## 项目目标
把现有 Python 分析脚本工程化为：
1. FastAPI 后端（核心算法 API 化）
2. 每日自动爬虫（FRED 数据拉取）
3. 微信小程序前端（图表展示）

## 当前状态
- 核心算法：已完成（12个Python脚本）
- 后端 API：未开始
- 爬虫调度：未开始
- 前端小程序：未开始

## 核心文件
| 文件 | 作用 |
|------|------|
| `code/stepA1_get_data_real_time.py` | 从 FRED 拉取最新 BTC 数据 → btc_price_fred.xlsx |
| `code/step01_halving_to_peak.py` | 核心分析：减半→峰值对齐图（波动退火） |
| `code/step02_halving_to_peak_fusion.py` | 多模型融合分析 |
| `data/btc_2015.xlsx` | 2017周期历史数据 |
| `data/btc_2021.xlsx` | 2021周期历史数据 |
| `data/btc_price_fred.xlsx` | 当前周期最新数据（每日更新） |
| `design/BTC价格预测小程序设计.md` | 完整产品设计文档 |

## 关键参数（不要随意改动）
- 减半日：2016-07-09 / 2020-05-11 / 2024-04-20
- 峰值日：2017-12-19 / 2021自动取最高 / 2025-08-15（假设）
- 波动退火：VOL_LEVEL_2017=9, VOL_LEVEL_2021=3, VOL_LEVEL_2025=1, alpha=0.5
- POST_DAYS=60, MAX_PRE_DAYS=300

## 技术栈决策
- 后端：Python + FastAPI（保持与现有脚本同语言，零移植风险）
- 数据库：SQLite（轻量，单机部署足够）
- 爬虫调度：APScheduler（内嵌在 FastAPI 进程）
- 前端：微信小程序 + ECharts
- 部署：Docker + 云服务器

## 开发规范
- 核心算法函数必须与现有 Python 脚本输出完全一致（数值验证）
- API 返回 JSON，字段名用英文小写下划线
- 数据库优先存原始数据，计算结果可缓存但需标注时间戳

## 构建/运行命令
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 API 服务（开发后添加）
uvicorn app.main:app --reload --port 8000

# 手动触发数据更新（开发后添加）
python -m app.scheduler run_now
```
