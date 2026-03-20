# Phase 1 实施计划：FastAPI 后端

## 目标
把现有 Python 脚本封装为 HTTP API，为小程序提供数据接口。

---

## 目录结构（完成后）

```
BTC_analysis_tool/
├── app/                        ← 新建
│   ├── __init__.py
│   ├── main.py                 ← FastAPI 入口
│   ├── routers/
│   │   └── btc.py              ← /api/btc/* 路由
│   ├── services/
│   │   ├── data_fetcher.py     ← 封装 stepA1 的拉数逻辑
│   │   └── analyzer.py         ← 封装 step01 的计算逻辑
│   ├── scheduler.py            ← APScheduler 每日任务
│   └── database.py             ← SQLite 连接
├── code/                       ← 原有脚本，只读不改
├── data/                       ← 原有数据
├── requirements.txt            ← 需追加新依赖
└── CLAUDE.md
```

---

## API 接口定义

### GET /api/btc/latest
返回最新一条 BTC 价格

```json
{
  "date": "2026-03-01",
  "price_usd": 85000.0,
  "updated_at": "2026-03-02T08:00:00"
}
```

### GET /api/btc/analysis
返回三周期对齐分析结果（核心接口）

```json
{
  "generated_at": "2026-03-02T08:00:00",
  "anchor_info": {
    "peak_2017": "2017-12-19",
    "peak_2021": "2021-11-10",
    "peak_2025": "2025-08-15"
  },
  "scale_factors": {
    "2017": 0.333,
    "2021": 0.577,
    "2025": 1.0
  },
  "curves": {
    "2017": [{"rel_day": -300, "dd_pct": -85.2}, ...],
    "2021": [{"rel_day": -300, "dd_pct": -72.1}, ...],
    "2025": [{"rel_day": -180, "dd_pct": -45.3}, ...]
  }
}
```

### GET /api/btc/predict?date=2026-04
输入目标年月，逆向推导价格区间

```json
{
  "target_date": "2026-04",
  "phase": "峰后→底",
  "price_low_usd": 40000,
  "price_high_usd": 55000,
  "explanation": "距峰值约 8 个月，处于顶→底回撤阶段（进度约65%）"
}
```

---

## 开发顺序

| 步骤 | 任务 | 对应现有脚本 |
|------|------|------------|
| 1 | 创建 FastAPI 骨架 + 健康检查 | 无 |
| 2 | 封装数据拉取服务 | stepA1_get_data_real_time.py |
| 3 | 封装核心分析服务 | step01_halving_to_peak.py |
| 4 | 实现 /api/btc/latest | - |
| 5 | 实现 /api/btc/analysis | - |
| 6 | 加入 SQLite 缓存 | - |
| 7 | 加入 APScheduler 每日定时 | - |
| 8 | 实现 /api/btc/predict | step03/step02 fusion |
| 9 | Docker 打包 | - |

---

## 新增依赖（追加到 requirements.txt）

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
apscheduler>=3.10.0
sqlalchemy>=2.0.0
httpx>=0.27.0
```

---

## 数值一致性验证方案

用现有脚本的某次输出做快照测试：
1. 取 `btc_fit_outputs/btc analysis/halving_to_peak_aligned.csv` 作为基准
2. API 用同一输入数据计算，结果与基准 CSV 的数值差 < 0.001%
3. 测试写在 `tests/test_analysis_consistency.py`
