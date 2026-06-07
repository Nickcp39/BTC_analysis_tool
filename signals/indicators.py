"""指标层：把原始数据算成指标 + 给出"底部/可买"方向打分。

每个指标产出一个 dict：{value..., zone(文字), score(整数)}。
score 越高越偏"底部/可买"，供 snapshot 汇总成综合结论。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


# ============================================================ AHR999
def compute_ahr999(price_df: pd.DataFrame) -> pd.DataFrame:
    """输入 [date, price] → 输出带 gma200 / estimate_price / ahr999 的完整序列。"""
    df = price_df.sort_values("date").reset_index(drop=True).copy()
    logp = np.log(df["price"])
    df["gma200"] = np.exp(logp.rolling(C.LEN_GMA, min_periods=C.LEN_GMA).mean())
    df["avg_index"] = df["price"] / df["gma200"]

    birth = pd.Timestamp(C.BITCOIN_BIRTH)
    age_days = (df["date"] - birth).dt.days.astype(float)
    age_days = age_days.where(age_days > 0, np.nan)
    df["estimate_price"] = np.power(10.0, C.AHR_K * np.log10(age_days) + C.AHR_B)
    df["estimate_index"] = df["price"] / df["estimate_price"]
    df["ahr999"] = df["avg_index"] * df["estimate_index"]
    return df


def ahr999_zone(v: float) -> str:
    if v < C.AHR_DEEP_VALUE:
        return "深度价值区（抄底）"
    if v < C.AHR_ACC_TOP:
        return "核心定投区"
    if v < C.AHR_TOP:
        return "偏热区"
    return "顶部信号区"


def ahr999_signal(price_df: pd.DataFrame) -> dict:
    df = compute_ahr999(price_df)
    last = df.dropna(subset=["ahr999"]).iloc[-1]
    v = float(last["ahr999"])
    if v < C.AHR_DEEP_VALUE:
        score = 2
    elif v < C.AHR_ACC_TOP:
        score = 1
    elif v < 2.0:
        score = -1
    else:
        score = -2
    # 距上次跌破 0.45 的天数（信息性）
    below = df["ahr999"] < C.AHR_DEEP_VALUE
    return {
        "name": "AHR999",
        "date": last["date"].date().isoformat(),
        "value": round(v, 3),
        "price": round(float(last["price"]), 2),
        "gma200": round(float(last["gma200"]), 2),
        "estimate_price": round(float(last["estimate_price"]), 2),
        "zone": ahr999_zone(v),
        "score": score,
        "in_deep_value": bool(v < C.AHR_DEEP_VALUE),
    }


# ============================================================ 恐慌贪婪
def fng_zone(v: int) -> str:
    if v <= C.FNG_EXTREME_FEAR:
        return "极度恐惧"
    if v <= C.FNG_FEAR:
        return "恐惧"
    if v < C.FNG_GREED:
        return "中性"
    if v < C.FNG_EXTREME_GREED:
        return "贪婪"
    return "极度贪婪"


def fng_signal(fng_df: pd.DataFrame) -> dict:
    df = fng_df.sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    v = int(last["value"])
    if v <= C.FNG_EXTREME_FEAR:
        score = 2
    elif v <= C.FNG_FEAR:
        score = 1
    elif v < C.FNG_GREED:
        score = 0
    elif v < C.FNG_EXTREME_GREED:
        score = -1
    else:
        score = -2
    return {
        "name": "恐慌贪婪指数",
        "date": last["date"].date().isoformat(),
        "value": v,
        "zone": fng_zone(v),
        "score": score,
    }


# ============================================================ 长持者
def lth_signal(lth_df: pd.DataFrame) -> dict:
    df = lth_df.sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    ratio = float(last["lth_ratio"])
    net30 = float(last.get("lth_net_change_30d_btc", 0) or 0)
    phase = str(last.get("phase", "")).strip()

    # 动作分：净增持 +1 / 净派发 -1
    action_score = 1 if net30 > 0 else (-1 if net30 < 0 else 0)
    # 占比分：逼近熊底水平 +1 / 派发充分 -1
    ratio_score = 1 if ratio >= C.LTH_BOTTOM_RATIO else (-1 if ratio <= C.LTH_TOP_RATIO else 0)

    return {
        "name": "长持者(LTH)",
        "date": last["date"].date().isoformat(),
        "lth_supply_btc": float(last["lth_supply_btc"]),
        "lth_ratio": ratio,
        "lth_net_change_30d_btc": net30,
        "phase": phase or ("吸筹" if net30 > 0 else "派发" if net30 < 0 else "持平"),
        "ratio_zone": ("逼近熊底水平" if ratio >= C.LTH_BOTTOM_RATIO
                       else "派发充分" if ratio <= C.LTH_TOP_RATIO else "中性"),
        "action_score": action_score,
        "ratio_score": ratio_score,
        "score": action_score + ratio_score,   # 长持者贡献两个子分
        "source": str(last.get("source", "")),
    }
