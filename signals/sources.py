"""数据层：抓取并落地最新数据。

- 价格日线  → CryptoCompare（FRED 在本环境超时，改用 CryptoCompare）
- 恐慌指数  → alternative.me
- 长持者    → 半自动 CSV（无免费 API，见 README；这里只负责读取 + 校验）

每个 fetch_* 返回 DataFrame，并写入 config 里指定的 CSV。
直接运行本文件 = 刷新全部可自动抓取的数据。
"""
from __future__ import annotations
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

_HEADERS = {"User-Agent": "Mozilla/5.0 (btc-signals-framework)"}


def _get_json(url: str, timeout: int = 40, retries: int = 6) -> dict:
    """带 429 退避重试的 JSON GET（bitcoin-data.com 有速率限制）。"""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = 20 * (attempt + 1)
                print(f"   429 限流，{wait}s 后重试 ({attempt + 1}/{retries})…")
                time.sleep(wait)
                continue
            raise


# ------------------------------------------------------------------ 价格
def fetch_btc_daily() -> pd.DataFrame:
    """BTC 美元日线收盘（全历史）→ data/btc_price_daily.csv"""
    j = _get_json(C.CRYPTOCOMPARE_HISTODAY)
    if j.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare 返回异常: {j.get('Message')}")
    rows = j["Data"]["Data"]
    df = pd.DataFrame(rows)[["time", "close"]]
    df = df[df["close"] > 0].copy()
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df = df[["date", "close"]].rename(columns={"close": "price"}).reset_index(drop=True)
    df.to_csv(C.PRICE_CSV, index=False, encoding="utf-8")
    print(f"[price] {len(df)} 行  {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}  "
          f"最新收盘 ${df['price'].iloc[-1]:,.0f}  -> {C.PRICE_CSV.name}")
    return df


# ------------------------------------------------------------------ 恐慌贪婪
def fetch_fear_greed() -> pd.DataFrame:
    """恐慌贪婪指数全历史 → data/fear_greed.csv"""
    j = _get_json(C.ALTME_FNG)
    data = j.get("data", [])
    if not data:
        raise RuntimeError("alternative.me 未返回数据")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    df["value"] = df["value"].astype(int)
    df = df[["date", "value", "value_classification"]].sort_values("date").reset_index(drop=True)
    df.to_csv(C.FNG_CSV, index=False, encoding="utf-8")
    last = df.iloc[-1]
    print(f"[fng]   {len(df)} 行  最新 {last['date'].date()}  {last['value']} ({last['value_classification']})  "
          f"-> {C.FNG_CSV.name}")
    return df


# ------------------------------------------------------------------ 链上：MVRV / 已实现市值
def fetch_onchain() -> pd.DataFrame:
    """MVRV + 已实现市值 + 已实现价格 全历史(2010 至今) → data/onchain_mvrv.csv

    数据源 CoinMetrics 社区 CSV（无限流、回溯 2010）。
    MVRV 直接取 CapMVRVCur；已实现市值 = 市值/MVRV；已实现价格 = 已实现市值/流通量。
    """
    use = ["time", "CapMVRVCur", "CapMrktCurUSD", "SplyCur"]
    df = pd.read_csv(C.CM_BTC_CSV, usecols=lambda c: c in use)
    df = df.rename(columns={"time": "date", "CapMVRVCur": "mvrv"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in ["mvrv", "CapMrktCurUSD", "SplyCur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mvrv", "CapMrktCurUSD"])
    df["realized_cap"] = df["CapMrktCurUSD"] / df["mvrv"]
    df["realized_price"] = df["realized_cap"] / df["SplyCur"]
    df = df[["date", "mvrv", "realized_cap", "realized_price"]].sort_values("date").reset_index(drop=True)
    df.to_csv(C.ONCHAIN_CSV, index=False, encoding="utf-8")
    last = df.iloc[-1]
    print(f"[onchain] {len(df)} 行  {df['date'].iloc[0].date()} ~ {last['date'].date()}  "
          f"MVRV {last['mvrv']:.3f}  已实现市值 ${last['realized_cap']/1e12:.3f}T  -> {C.ONCHAIN_CSV.name}")
    return df


# ------------------------------------------------------------------ 长持者（读取 + 校验）
def load_lth_metrics() -> pd.DataFrame:
    """读取半自动维护的长持者指标 CSV。若不存在则报错提示。"""
    if not C.LTH_CSV.exists():
        raise FileNotFoundError(
            f"缺少 {C.LTH_CSV}。这是半自动文件，需手工/定期从链上数据源(CoinDesk/Glassnode/CryptoQuant)更新。"
        )
    df = pd.read_csv(C.LTH_CSV)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    last = df.iloc[-1]
    age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - last["date"]).days
    stale = "  ⚠️数据偏旧，建议更新" if age_days > 21 else ""
    print(f"[lth]   {len(df)} 行  最新 {last['date'].date()} (距今 {age_days} 天){stale}  "
          f"占比 {last['lth_ratio']:.1%}  -> {C.LTH_CSV.name}")
    return df


def refresh_all() -> None:
    print("=== 刷新自动数据源 ===")
    fetch_btc_daily()
    fetch_fear_greed()
    fetch_onchain()
    try:
        load_lth_metrics()
    except FileNotFoundError as e:
        print(f"[lth]   {e}")
    print("=== 完成 ===")


if __name__ == "__main__":
    refresh_all()
