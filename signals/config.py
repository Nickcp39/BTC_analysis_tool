"""信号框架的全局配置：路径、AHR999 参数、各指标的分区阈值与打分规则。

这一层只放"常量与规则"，不放逻辑。改阈值/换路径都在这里改。
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parent.parent          # BTC_analysis_tool/
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_CSV = DATA_DIR / "btc_price_daily.csv"           # 价格日线（CryptoCompare 抓取）
FNG_CSV = DATA_DIR / "fear_greed.csv"                  # 恐慌贪婪指数（alternative.me 抓取）
LTH_CSV = DATA_DIR / "lth_metrics.csv"                 # 长持者指标（半自动维护，见 README）
ONCHAIN_CSV = DATA_DIR / "onchain_mvrv.csv"            # MVRV/已实现市值/已实现价格（bitcoin-data.com）
CHART_DIR = OUT_DIR / "charts"                          # 图表输出
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据源
CRYPTOCOMPARE_HISTODAY = (
    "https://min-api.cryptocompare.com/data/v2/histoday"
    "?fsym=BTC&tsym=USD&allData=true"
)
ALTME_FNG = "https://api.alternative.me/fng/?limit=0&format=json"  # limit=0 = 全历史
# CoinMetrics 社区公开 CSV：全历史(2010 至今)，含 MVRV/市值/流通量，无限流。
# 已实现市值 = 市值 / MVRV；已实现价格 = 已实现市值 / 流通量（推导）。
CM_BTC_CSV = "https://raw.githubusercontent.com/coinmetrics-io/data/master/csv/btc.csv"
# 备用：bitcoin-data.com（仅最近~4年，且有限流）
BD_MVRV = "https://bitcoin-data.com/v1/mvrv"
BD_REALIZED_CAP = "https://bitcoin-data.com/v1/realized-cap"
BD_REALIZED_PRICE = "https://bitcoin-data.com/v1/realized-price"

# ---------------------------------------------------------------- AHR999 参数
# ahr999 = (price / gma200) * (price / estimate_price)
#   gma200        = 近 200 日收盘的几何均值
#   estimate_price= 10^(K*log10(币龄天数) + B)，币龄自 2009-01-03 起算
BITCOIN_BIRTH = "2009-01-03"
LEN_GMA = 200
AHR_K = 5.84
AHR_B = -17.01

# AHR999 分区（九神原版 + 常用扩展）
AHR_DEEP_VALUE = 0.45      # < 0.45：深度价值区（历史抄底）
AHR_ACC_TOP = 1.20         # 0.45~1.20：核心定投区
AHR_TOP = 4.0              # >= 4：牛市顶部信号

# ---------------------------------------------------------------- 恐慌贪婪分区
# alternative.me 自带 classification，这里只定义"打分用"的阈值
FNG_EXTREME_FEAR = 25      # <= 25：极度恐惧
FNG_FEAR = 45              # <= 45：恐惧
FNG_GREED = 55            # >= 55：贪婪
FNG_EXTREME_GREED = 75     # >= 75：极度贪婪

# ---------------------------------------------------------------- 长持者分区
LTH_BOTTOM_RATIO = 0.83    # 长持占比 >= 83%：逼近历史熊底水平（约 85%）
LTH_TOP_RATIO = 0.70       # 长持占比 <= 70%：派发充分（偏顶部）

# ---------------------------------------------------------------- 综合打分 → 结论
# 每个指标给一个分数，求和后映射到结论。分数越高越偏"底部/可买"。
VERDICT_BANDS = [
    (5, "强烈抄底区：四指标高度共振，历史上接近熊底"),
    (3, "价值买入区：多数指标指向底部，适合按计划分批吸筹"),
    (1, "偏价值：部分指标转好，可维持定投"),
    (-1, "中性：无明显方向，按原节奏"),
    (-3, "偏过热：注意风险，减少加仓"),
    (-99, "过热/顶部区：考虑止盈"),
]
