# signals — BTC 底部信号框架

把"是不是熊底、能不能买"拆成四个可量化指标，抓取最新数据 → 各自打分 → 汇总成一个综合结论快照。

## 四个指标

| 指标 | 含义 | 数据源 | 自动化 |
|---|---|---|---|
| **AHR999** | 价格相对 200 日成本 + 长期估值中枢的偏离 | CryptoCompare 日线 | ✅ 全自动 |
| **恐慌贪婪指数** | 市场情绪冷热 | alternative.me | ✅ 全自动 |
| **长持者动作 (LTH)** | 长持者 30 日净增持/派发 | 链上(CoinDesk/Glassnode) | ⚠️ 半自动 |
| **长持者占比** | 长持供应 / 流通供应 | 链上 | ⚠️ 半自动 |

> FRED 在本机超时、Binance 地域封锁，故价格改用 CryptoCompare。长持者无免费 API，用 `data/lth_metrics.csv` 手工/定期维护。

## 用法

```bash
# 1. 只抓数据（刷新价格 + 恐慌指数）
python signals/sources.py

# 2. 跑分析（用本地数据）
python signals/snapshot.py

# 或一步到位：先抓再算
python signals/snapshot.py --refresh
```

Windows 控制台若中文乱码/报错，先设 `set PYTHONUTF8=1`（PowerShell: `$env:PYTHONUTF8=1`）。
运行需要 anaconda 环境（pandas/numpy）：`C:\Users\nickc\anaconda3\python.exe`。

## 产出

- `data/btc_price_daily.csv`、`data/fear_greed.csv` — 抓取的原始序列
- `outputs/snapshot_YYYY-MM-DD.json` / `.md` — 当天快照（含四指标读数 + 综合结论）

## 文件结构

```
signals/
  config.py      常量与阈值（改分区/打分规则在这里）
  sources.py     数据抓取与读取
  indicators.py  AHR999 / 恐慌 / LTH 的计算与打分
  snapshot.py    汇总 + 产出快照（入口）
```

## 打分规则（透明启发式，见 config.py）

- **AHR999**: <0.45 → +2；0.45~1.2 → +1；1.2~2 → -1；>2 → -2
- **恐慌贪婪**: ≤25 → +2；≤45 → +1；中性 0；贪婪 -1；极贪 -2
- **长持动作**: 净增持 +1 / 净派发 -1
- **长持占比**: ≥83% → +1；≤70% → -1；中间 0
- **综合**: 四项求和（满分 +6）→ 映射到结论区间（VERDICT_BANDS）

⚠️ 这是辅助判断的启发式，**不是买卖触发器**。底通常是一段区间，执行上建议分批/区间，而非择时一把梭。

## 维护 LTH 数据

`data/lth_metrics.csv` 列：`date, lth_supply_btc, circulating_supply_btc, lth_ratio, lth_net_change_30d_btc, phase, source`。
每隔 1–3 周从 CoinDesk / Glassnode / CryptoQuant 查最新长持供应与净变化，追加一行即可（`lth_ratio = lth_supply / circulating`）。超过 21 天会在抓取时提示数据偏旧。

## 待扩展（下一步可加的指标）

- STH/LTH 已实现价格交叉（牛熊拐点的更精确信号）
- MVRV / NUPL
- ETF 资金净流入流出
- 与 2018 / 2022 两轮熊底同期读数的横向对比
