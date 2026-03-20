# BTC 周期插件详细设计 V1

## 1. 文档目的

这是系统的**第一个机会层插件**的完整实现规格。

从"12 个松散脚本"到"一个结构化插件"的完整映射。

### 与其他文档的关系

- [01_机会层插件规范](./01_机会层插件规范.md) → 定义通用插件接口，本文档是 BTC 插件的具体实现
- [03_核心算法规格书](./03_核心算法规格书.md) → 定义所有公式，本文档定义"这些公式如何组织成插件"
- [04_数据流与状态机](./04_数据流与状态机.md) → 定义数据层级，本文档定义"BTC 插件读哪些层、写哪些层"

---

## 2. 插件身份

```yaml
plugin_id: btc_cycle
plugin_version: "1.0.0"
display_name: "BTC 周期退火分析"
target: "BTC"
target_type: "crypto"
status: active
```

---

## 3. 现有脚本 → 插件模块映射

**12 个脚本重组为 4 个内部模块：**

```
plugins/btc_cycle/
├── manifest.yaml
├── plugin.py                      ← 入口, 实现 run()
│
├── data_loader.py                 ← 模块 1: 数据加载
│   来源脚本:
│     stepA1_get_data_real_time.py  (FRED 抓取)
│     stepA3_pre_data_summary.py    (多源合并)
│     load_halving_peak.py          (锚点加载)
│
├── annealing.py                   ← 模块 2: 退火对齐 (核心)
│   来源脚本:
│     step01_halving_to_peak.py     (3 周期退火)
│     step04_four_cycles.py         (4 周期双重缩放)
│     step05_time_scaled.py         (时间缩放退火)
│   核心函数:
│     window_halving_to_peak()
│     pct_curve()
│     pre_std()
│     scale_factor()
│     compute_annealing()           ← 新: 整合以上为一步
│
├── time_fusion.py                 ← 模块 3: 时间融合
│   来源脚本:
│     step02_halving_to_peak_fusion.py  (三模型融合)
│     step03_time_scale_contracting.py  (时间收缩比)
│   核心函数:
│     predict_median()
│     predict_regression()
│     predict_peak_to_peak()
│     fuse_predictions()
│
├── post_peak.py                   ← 模块 4: 峰后分析
│   来源脚本:
│     stepB4_post_peak_only.py
│     stepA1_halving_to_peak_01_b4.py
│     stepA2_peak_skip_gap.py
│     stepA3_extended_200.py
│     stepA3_post_scaled.py
│   核心函数:
│     extract_post_peak()
│     apply_post_scaling()
│     apply_gamma_decay()
│
├── signal_judge.py                ← 模块 5: 信号判定 (新增)
│   来源: 无现有脚本, 新增逻辑
│   功能: 综合退火+融合+峰后结果, 输出统一信号
│
├── config/
│   ├── anchors.yaml
│   └── parameters.yaml
│
└── tests/
    ├── test_data_loader.py
    ├── test_annealing.py
    ├── test_time_fusion.py
    ├── test_post_peak.py
    ├── test_signal_judge.py
    └── fixtures/                  ← 基准数据 (现有脚本输出)
        ├── halving_to_peak_aligned.csv
        ├── halving_to_peak_metrics.txt
        └── step02_time_fusion.txt
```

---

## 4. 插件运行流程

```
plugin.run()
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: 数据加载 (data_loader.py)                          │
│                                                              │
│  ① load_clean_prices()                                      │
│     → 从 L1 读取 btc_merged_daily 或从数据库读取              │
│     → 返回 pd.Series (index=date, value=price)              │
│                                                              │
│  ② load_anchors()                                            │
│     → 读取 anchors.yaml                                      │
│     → 返回 [(halving, peak, vol_level), ...]                 │
│                                                              │
│  ③ load_parameters()                                         │
│     → 读取 parameters.yaml                                   │
│     → 返回 config dict                                       │
│                                                              │
│  ④ validate_data()                                           │
│     → 新鲜度、完整性、异常值检查                               │
│     → 失败时直接返回 signal=insufficient_data                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: 退火对齐 (annealing.py)                            │
│                                                              │
│  对每个历史周期:                                              │
│  ① window = window_halving_to_peak(series, halving, peak)    │
│  ② (rel_day, dd_pct) = pct_curve(window, peak)              │
│  ③ std_pre = pre_std(dd_pct, span=90)                       │
│  ④ scale = scale_factor(vol_src, vol_tgt, alpha=0.5)        │
│  ⑤ dd_pct_scaled = dd_pct × scale                           │
│                                                              │
│  对当前周期 (2025):                                          │
│  ① window = series[halving_2024 : latest]                    │
│  ② (rel_day, dd_pct) = pct_curve(window, peak_2025)         │
│  ③ std_pre_2025 = pre_std(dd_pct, span=90)                  │
│  ④ scale = 1.0 (不缩放)                                     │
│                                                              │
│  匹配度:                                                     │
│  ⑥ r_pre, rmse_pre = correlation(2017_scaled, 2025, pre)    │
│  ⑦ r_post, rmse_post = correlation(2017_scaled, 2025, post) │
│                                                              │
│  输出: AnnealingResult                                       │
│    curves: {2017: (rel_day, dd_pct_scaled), 2021: ..., 2025}│
│    metrics: {r_pre, rmse_pre, r_post, rmse_post per cycle}   │
│    scale_factors: {2017: 0.333, 2021: 0.577}                │
│    pre_stds: {2017: 45.2, 2021: 18.7, 2025: 15.1}          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: 时间融合 (time_fusion.py)                          │
│                                                              │
│  ① peak_median = halving_2024 + median(h2p_days)            │
│  ② peak_regression = halving_2024 + regression(intervals)    │
│  ③ peak_p2p = peak_2021 + mean(peak_intervals)              │
│  ④ peak_fused = weighted_average(①②③, w=[1.0, 1.3, 1.1])   │
│  ⑤ window = [min(①②③), max(①②③)]                           │
│                                                              │
│  输出: TimeFusionResult                                      │
│    predictions: {median: date, regression: date, p2p: date}  │
│    fused_date: date                                          │
│    window: (start, end)                                      │
│    window_width_days: int                                    │
│    model_agreement: high/medium/low                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 4: 峰后分析 (post_peak.py)                            │
│  (仅当 current_date > peak_date 时运行)                      │
│                                                              │
│  ① post_curves = extract_post_peak(curves)                   │
│  ② post_stds = {cycle: std(post_dd_pct) for each}           │
│  ③ decay_curves = apply_gamma_decay(curves, gamma=0.65)      │
│  ④ time_scaled_post = apply_time_scaling(post_curves)        │
│                                                              │
│  输出: PostPeakResult                                        │
│    post_curves: {2017: ..., 2021: ..., 2025: ...}           │
│    post_stds: {2017: x, 2021: y, 2025: z}                  │
│    decay_curves: {...}                                       │
│    post_match: {r_2017, rmse_2017, r_2021, rmse_2021}       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Phase 5: 信号判定 (signal_judge.py)                         │
│                                                              │
│  输入: AnnealingResult + TimeFusionResult + PostPeakResult    │
│                                                              │
│  ① 周期阶段判定 (见下方 §5)                                  │
│  ② 风险等级判定 (见下方 §6)                                  │
│  ③ 置信度计算 (见下方 §7)                                    │
│  ④ 组装 PluginOutput (见下方 §8)                             │
│                                                              │
│  输出: PluginOutput (统一接口格式)                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. 周期阶段判定规则

**这是把"数字"翻译成"判断"的关键步骤。**

```python
def determine_cycle_stage(days_from_peak: int, days_since_halving: int) -> str:
    """
    基于时间位置判定周期阶段。

    时间轴:
    减半 ──────────────── 峰值 ──── 峰后 ─────── 下一减半
    │<── accumulation ──>│<peak>│<── decline ──>│

    rel_day = days_from_peak (峰值为 0)
    """

    if days_since_halving < 180:
        return "early_accumulation"     # 减半后早期, 市场还没反应

    if days_from_peak < -180:
        return "mid_cycle"              # 周期中段, 趋势建立中

    if days_from_peak < -60:
        return "approaching_peak"       # 接近峰值, 加速上涨期

    if days_from_peak < 0:
        return "near_peak"              # 临近峰值, 最危险区间

    if days_from_peak == 0:
        return "peak"                   # 峰值当天

    if days_from_peak <= 60:
        return "early_decline"          # 峰后早期, 可能是假跌

    if days_from_peak <= 365:
        return "post_peak_decline"      # 确认下跌趋势

    return "deep_bear"                  # 深熊, 等待下一周期
```

### 阶段特征表

| 阶段 | rel_day 范围 | 典型特征 | 历史参考 |
|------|-------------|----------|----------|
| early_accumulation | halving后 < 180天 | 价格低位盘整 | 2024-04~10 |
| mid_cycle | < -180 | 趋势性上涨开始 | 2024-10~2025-02 |
| approaching_peak | -180 ~ -60 | 加速上涨, 波动加大 | 2025-02~06 |
| near_peak | -60 ~ 0 | FOMO 情绪, 历史最高 | 2025-06~08 |
| peak | 0 | 峰值 | 2025-08-15(假设) |
| early_decline | 0 ~ 60 | 下跌但不确定是否见顶 | 2025-08~10 |
| post_peak_decline | 60 ~ 365 | 确认下跌, 熊市开始 | 2025-10~2026-08 |
| deep_bear | > 365 | 深熊, 底部区间 | 2026-08~ |

---

## 6. 风险等级判定规则

```python
def determine_risk_level(
    cycle_stage: str,
    r_match_avg: float,       # 平均匹配度 (2017+2021)/2
    pct_from_anchor: float,   # 当前价格相对峰值的涨跌幅
    time_fusion_agreement: str # 时间模型一致性
) -> str:
    """
    综合多个维度判定风险等级。

    风险 = 时间位置风险 × 模式确认度
    """

    # 基础风险 (由周期阶段决定)
    stage_risk = {
        "early_accumulation": 1,
        "mid_cycle": 2,
        "approaching_peak": 3,
        "near_peak": 4,
        "peak": 5,
        "early_decline": 4,
        "post_peak_decline": 4,    # 不适合入场, 但已有仓位的风险在释放
        "deep_bear": 2,            # 风险释放较多
    }
    base_risk = stage_risk[cycle_stage]

    # 模式确认调整 (匹配度高 = 判断可靠 = 风险评估更确定)
    if r_match_avg > 0.85:
        confirmation = 1.0         # 强确认
    elif r_match_avg > 0.70:
        confirmation = 0.8         # 中等确认
    else:
        confirmation = 0.5         # 弱确认, 不确定性大

    # 最终风险分
    risk_score = base_risk * confirmation

    # 映射到等级
    if risk_score >= 4.0:
        return "extreme"
    elif risk_score >= 3.0:
        return "high"
    elif risk_score >= 2.0:
        return "medium"
    else:
        return "low"
```

---

## 7. 置信度计算

```python
def calculate_confidence(
    data_quality: float,       # 数据完整度 0-1
    r_match_2017: float,       # 2017 匹配度 (相关系数)
    r_match_2021: float,       # 2021 匹配度
    time_window_width: int,    # 时间融合窗口宽度(天)
    days_beyond_data: int      # 预测超出数据多少天 (越远越不确信)
) -> float:
    """
    置信度 = 数据质量 × 模式匹配 × 时间一致性 × 外推衰减

    范围: 0.0 ~ 1.0
    """

    # 因子 1: 数据质量 (0.8 ~ 1.0)
    f_data = 0.8 + 0.2 * data_quality

    # 因子 2: 模式匹配 (取两个周期的平均)
    f_match = (r_match_2017 + r_match_2021) / 2
    f_match = max(0.3, f_match)    # 下限 0.3

    # 因子 3: 时间模型一致性
    if time_window_width < 30:
        f_time = 1.0               # 三模型高度一致
    elif time_window_width < 60:
        f_time = 0.9
    elif time_window_width < 90:
        f_time = 0.8
    else:
        f_time = 0.6               # 不确定性大

    # 因子 4: 外推衰减 (预测越远, 越不确信)
    f_extrapolation = max(0.5, 1.0 - days_beyond_data / 500)

    confidence = f_data * f_match * f_time * f_extrapolation
    return round(min(1.0, max(0.0, confidence)), 2)
```

---

## 8. 最终输出组装

```python
def build_output(
    annealing: AnnealingResult,
    time_fusion: TimeFusionResult,
    post_peak: PostPeakResult,      # 可能为 None (峰前)
    cycle_stage: str,
    risk_level: str,
    confidence: float,
    signal: str
) -> PluginOutput:

    return PluginOutput(
        target="BTC",
        plugin_id="btc_cycle",
        plugin_version="1.0.0",
        generated_at=datetime.now().isoformat(),
        data_freshness=annealing.latest_data_date,

        assessment=Assessment(
            signal=signal,
            risk_level=risk_level,
            confidence=confidence,
            time_window=TimeWindow(
                start=time_fusion.window[0].isoformat(),
                end=time_fusion.window[1].isoformat(),
                anchor=time_fusion.fused_date.isoformat(),
                description=f"多模型融合峰值窗口 (宽度 {time_fusion.window_width_days} 天)"
            )
        ),

        context={
            # 周期定位
            "cycle_stage": cycle_stage,
            "days_since_halving": annealing.days_since_halving,
            "days_from_peak_anchor": annealing.days_from_peak,
            "pct_from_anchor": annealing.pct_from_anchor,

            # 退火参数
            "scale_factor_2017": annealing.scale_factors["2017"],
            "scale_factor_2021": annealing.scale_factors["2021"],
            "pre_std_2017": annealing.pre_stds["2017"],
            "pre_std_2021": annealing.pre_stds["2021"],
            "pre_std_2025": annealing.pre_stds["2025"],

            # 匹配度
            "r_pre_2017": annealing.metrics["2017"].r_pre,
            "r_pre_2021": annealing.metrics["2021"].r_pre,
            "rmse_pre_2017": annealing.metrics["2017"].rmse_pre,
            "rmse_pre_2021": annealing.metrics["2021"].rmse_pre,

            # 时间融合
            "time_model_median": time_fusion.predictions["median"].isoformat(),
            "time_model_regression": time_fusion.predictions["regression"].isoformat(),
            "time_model_p2p": time_fusion.predictions["p2p"].isoformat(),
            "time_model_fused": time_fusion.fused_date.isoformat(),
            "time_window_width_days": time_fusion.window_width_days,

            # 峰后 (如有)
            "post_peak_days": post_peak.days if post_peak else None,
            "post_std_2025": post_peak.post_stds.get("2025") if post_peak else None,
        },

        reasoning=build_reasoning(cycle_stage, risk_level, annealing, time_fusion, post_peak)
    )
```

---

## 9. reasoning 生成规则

**reasoning 是给人看的，不是给机器看的。**

```python
def build_reasoning(stage, risk, annealing, time_fusion, post_peak) -> List[str]:
    reasons = []

    # 第一条: 周期位置 (最重要)
    stage_descriptions = {
        "early_accumulation": f"当前距减半 {annealing.days_since_halving} 天，处于周期早期积累阶段",
        "mid_cycle": f"当前距减半 {annealing.days_since_halving} 天，周期趋势已建立",
        "approaching_peak": f"距预计峰值约 {abs(annealing.days_from_peak)} 天，进入加速上涨期",
        "near_peak": f"距预计峰值仅 {abs(annealing.days_from_peak)} 天，处于最危险区间",
        "peak": "当前处于预计峰值位置",
        "early_decline": f"已过峰值 {annealing.days_from_peak} 天，尚未确认下跌趋势",
        "post_peak_decline": f"已过峰值 {annealing.days_from_peak} 天，下跌趋势已确认",
        "deep_bear": f"已过峰值 {annealing.days_from_peak} 天，处于深熊阶段",
    }
    reasons.append(stage_descriptions[stage])

    # 第二条: 历史匹配 (可信度支撑)
    r_avg = (annealing.metrics["2017"].r_pre + annealing.metrics["2021"].r_pre) / 2
    if r_avg > 0.85:
        reasons.append(f"与 2017/2021 周期退火曲线高度吻合 (平均 r={r_avg:.2f})，历史模式确认")
    elif r_avg > 0.70:
        reasons.append(f"与历史周期退火曲线较为吻合 (平均 r={r_avg:.2f})，模式部分确认")
    else:
        reasons.append(f"与历史周期退火曲线匹配度较低 (平均 r={r_avg:.2f})，判断可靠性下降")

    # 第三条: 时间融合 (窗口)
    width = time_fusion.window_width_days
    if width < 30:
        reasons.append(f"三个时间模型高度一致，峰值窗口仅 {width} 天")
    elif width < 60:
        reasons.append(f"时间模型基本一致，峰值窗口 {width} 天，不确定性适中")
    else:
        reasons.append(f"时间模型分歧较大，峰值窗口 {width} 天，时间判断不确定性较高")

    # 第四条: 价格位置 (辅助)
    pct = annealing.pct_from_anchor
    if pct > 0:
        reasons.append(f"当前价格高于峰值锚点 {pct:.1f}%，可能已过峰")
    elif pct > -20:
        reasons.append(f"当前价格低于峰值锚点 {abs(pct):.1f}%，处于峰值附近")
    elif pct > -50:
        reasons.append(f"当前价格低于峰值锚点 {abs(pct):.1f}%，已有明显回撤")
    else:
        reasons.append(f"当前价格低于峰值锚点 {abs(pct):.1f}%，深度回撤")

    # 第五条: 峰后 (如有)
    if post_peak and post_peak.days > 0:
        if post_peak.post_match.get("r_2017", 0) > 0.8:
            reasons.append("峰后走势与历史衰减模式高度一致")
        else:
            reasons.append("峰后走势与历史衰减模式存在偏差")

    return reasons
```

---

## 10. 信号判定决策树

```
                        数据充足?
                        /       \
                      否          是
                      │           │
              insufficient_data   │
                                  │
                            周期阶段?
                    ┌───────┼────────┐──────────┐
                    │       │        │          │
               early/mid  approach  near_peak  post_peak
                    │       │        │          │
                    │       │        │       匹配度?
                    │       │        │      /       \
                    │       │        │    高          低
                    │       │        │    │           │
                    │       │    匹配度?  high_risk  observe
                    │       │   /     \       │
                    │       │  高      低     (extreme)
                    │       │  │       │
                    │    observe observe
                    │    (high)  (med)
                    │
                 匹配度?
                /       \
              高          低
              │           │
         opportunity   track
           (low)       (med)
```

完整判定逻辑（signal_judge.py 核心）：

```python
def determine_signal(
    cycle_stage: str,
    risk_level: str,
    r_match_avg: float,
    confidence: float
) -> str:
    """
    从周期阶段、风险等级、匹配度综合判定最终信号。

    信号含义:
      opportunity      → 历史上这是好的入场区间
      track            → 值得关注但时机未到
      small_position   → 可以少量参与
      observe          → 只看不动
      high_risk        → 危险区间，远离
      insufficient_data → 数据不足，不判断
    """

    # 匹配度低 → 降级处理
    if r_match_avg < 0.5:
        if cycle_stage in ("post_peak_decline", "near_peak", "peak"):
            return "observe"      # 不确定但位置危险, 保守
        return "track"            # 不确定, 跟踪观察

    # 峰后阶段
    if cycle_stage in ("post_peak_decline", "deep_bear"):
        if risk_level == "extreme":
            return "high_risk"
        return "observe"          # 下跌确认, 观望等底

    # 峰值/临近峰值
    if cycle_stage in ("peak", "near_peak", "early_decline"):
        return "high_risk" if r_match_avg > 0.7 else "observe"

    # 接近峰值
    if cycle_stage == "approaching_peak":
        return "observe"          # 上涨但风险累积

    # 周期中段
    if cycle_stage == "mid_cycle":
        if r_match_avg > 0.8:
            return "small_position"
        return "track"

    # 早期积累
    if cycle_stage == "early_accumulation":
        if r_match_avg > 0.7:
            return "opportunity"
        return "track"

    return "observe"              # 兜底
```

---

## 11. 图表输出规格

BTC 插件需要输出以下图表数据（供前端渲染）：

### 11.1 主图：退火对齐

```json
{
  "chart_id": "halving_to_peak_annealed",
  "title": "减半→峰值对齐（含峰后）：波动退火后",
  "x_label": "相对峰值天数",
  "y_label": "相对峰值涨跌幅 (%)",
  "annotations": {
    "peak_line": { "x": 0, "style": "dashed", "color": "gray", "label": "峰值日" }
  },
  "series": [
    {
      "name": "2017 (退火后, scale=0.333)",
      "color": "#1f77b4",
      "style": "solid",
      "data": [{"x": -300, "y": -28.4}, ...]
    },
    {
      "name": "2021 (退火后, scale=0.577)",
      "color": "#ff7f0e",
      "style": "solid",
      "data": [{"x": -300, "y": -41.6}, ...]
    },
    {
      "name": "2025 (当前)",
      "color": "#2ca02c",
      "style": "solid",
      "lineWidth": 2.5,
      "data": [{"x": -180, "y": -45.3}, ...]
    }
  ]
}
```

### 11.2 辅助图：时间融合

```json
{
  "chart_id": "time_fusion_models",
  "title": "峰值时间多模型预测",
  "type": "timeline",
  "markers": [
    { "date": "2025-09-29", "label": "中位数模型", "color": "#1f77b4" },
    { "date": "2025-10-12", "label": "回归模型", "color": "#ff7f0e" },
    { "date": "2025-10-24", "label": "峰峰外推", "color": "#2ca02c" },
    { "date": "2025-10-05", "label": "融合结果", "color": "#d62728", "style": "bold" }
  ],
  "window": {
    "start": "2025-09-29",
    "end": "2025-10-24",
    "fill": "rgba(255,0,0,0.1)"
  }
}
```

### 11.3 辅助图：4 周期对比（含 2013）

```json
{
  "chart_id": "four_cycles_comparison",
  "title": "四周期退火 + 时间缩放对比",
  "note": "X轴和Y轴均经过缩放, 所有周期映射到2025时间轴"
}
```

---

## 12. 测试策略

### 12.1 单元测试

```
每个内部模块的核心函数:

test_pct_curve():
  输入: 已知价格序列 + 锚点
  验证: rel_day 和 dd_pct 精确到小数点后 6 位

test_pre_std():
  输入: 已知 dd_pct 序列
  验证: 与 numpy std(ddof=0) 结果一致

test_scale_factor():
  输入: VOL_LEVEL=9, 1, alpha=0.5
  验证: == (1/9)^0.5 = 0.33333...

test_time_fusion():
  输入: 已知 halving/peak 日期
  验证: 三模型日期 + 融合日期与手算一致
```

### 12.2 快照对比测试（最重要）

```
用现有脚本的输出文件作为基准:

test_vs_step01_csv():
  基准: visualization/YYYY-MM-DD/halving_to_peak_aligned.csv
  方法: 用同一输入运行插件, 对比每个 rel_day 的 dd_pct
  容差: |diff| < 0.001%

test_vs_step01_metrics():
  基准: visualization/YYYY-MM-DD/halving_to_peak_metrics.txt
  方法: 对比 scale_factors, pre_stds, r, rmse
  容差: |diff| < 0.0001

test_vs_step02_fusion():
  基准: visualization/YYYY-MM-DD/step02_time_fusion.txt
  方法: 对比三模型预测日期
  容差: 日期完全一致
```

### 12.3 集成测试

```
test_full_pipeline():
  从 FRED 数据 → 插件 run() → PluginOutput
  验证:
    1. output.assessment.signal 是合法枚举值
    2. output.assessment.confidence ∈ [0, 1]
    3. output.context 包含所有必需字段
    4. output.reasoning 非空
    5. 曲线数据点数 > 100

test_insufficient_data():
  输入: 空数据或过期数据
  验证: signal = "insufficient_data"
```

---

## 13. 性能要求

| 操作 | 目标耗时 | 说明 |
|------|----------|------|
| 数据加载 (本地) | < 500ms | 读 Excel/CSV |
| 退火计算 | < 2s | 3 周期窗口提取 + 退火 |
| 时间融合 | < 100ms | 简单数学运算 |
| 峰后分析 | < 1s | 可选, 仅峰后运行 |
| 信号判定 | < 50ms | 规则判定 |
| **完整 run()** | **< 5s** | 从加载到输出 |

首次运行需要从 FRED 拉数据，额外 5-10s 网络延迟。
