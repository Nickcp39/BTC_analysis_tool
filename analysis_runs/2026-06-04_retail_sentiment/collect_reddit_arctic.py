import csv
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests


OUT_DIR = Path(__file__).resolve().parent
RAW_CSV = OUT_DIR / "reddit_retail_comments_raw.csv"
SUMMARY_JSON = OUT_DIR / "reddit_retail_summary.json"
REPORT_MD = OUT_DIR / "retail_sentiment_report.md"

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
UA = "btc-retail-sentiment-research/1.0"
FIELDS = "id,author,body,score,created_utc,subreddit,link_id,parent_id"

WINDOWS = [
    {
        "period": "2022_luna_celsius",
        "label": "2022 LUNA/Celsius crash",
        "after": "2022-05-08",
        "before": "2022-07-15",
        "target": 3000,
        "queries": ["bitcoin", "btc", "luna", "ust", "celsius", "3ac", "crash", "dump", "dip", "liquidation"],
    },
    {
        "period": "2022_ftx",
        "label": "2022 FTX crash",
        "after": "2022-11-02",
        "before": "2022-11-25",
        "target": 2600,
        "queries": ["bitcoin", "btc", "ftx", "sbf", "alameda", "binance", "withdraw", "crash", "dump", "contagion"],
    },
    {
        "period": "2026_current",
        "label": "2026-06 current drawdown",
        "after": "2026-06-01",
        "before": "2026-06-05",
        "target": 3000,
        "queries": ["bitcoin", "btc", "drop", "crash", "dump", "liquidation", "etf", "saylor", "strategy", "ai", "dip"],
    },
]

SUBREDDITS = [
    "Bitcoin",
    "CryptoCurrency",
    "CryptoMarkets",
    "Buttcoin",
    "StockMarket",
    "China_irl",
    "real_China_irl",
    "Go_Stock",
]

CATEGORIES = {
    "panic_capitulation": ["panic", "scared", "fear", "worried", "blood", "dead", "over", "zero", "capitulation", "despair", "崩", "慌", "完了", "归零", "血", "割肉"],
    "buy_dip_hodl": ["buy the dip", "btd", "dca", "hodl", "stack", "accumulate", "cheap", "discount", "buying more", "买入", "抄底", "定投", "拿住", "持有", "便宜"],
    "leverage_liquidation": ["leverage", "liquidation", "liquidated", "margin", "longs", "shorts", "overleveraged", "爆仓", "杠杆", "清算", "合约"],
    "institutional_flow": ["etf", "institution", "blackrock", "fidelity", "wall street", "saylor", "strategy", "mstr", "机构", "现货", "资金流出", "贝莱德"],
    "macro_liquidity": ["fed", "rates", "inflation", "liquidity", "recession", "dollar", "stocks", "nasdaq", "gold", "宏观", "加息", "降息", "流动性", "美股", "黄金"],
    "fraud_cefi_counterparty": ["scam", "fraud", "ponzi", "ftx", "sbf", "alameda", "celsius", "voyager", "blockfi", "3ac", "terra", "luna", "ust", "exchange", "withdrawals", "bankruptcy", "insolvent", "骗局", "旁氏", "交易所", "提现", "破产", "暴雷", "挤兑"],
    "ai_capital_rotation": ["ai", "nvidia", "nvda", "palantir", "pltr", "tech stocks", "人工智能", "英伟达", "科技股"],
    "cycle_thesis": ["cycle", "halving", "four year", "4 year", "bear market", "bull market", "ath", "周期", "减半", "熊市", "牛市", "新高"],
}


def clean_body(body: str) -> str:
    body = (body or "").replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", body)


def classify(body: str) -> list:
    lower = body.lower()
    hits = [cat for cat, terms in CATEGORIES.items() if any(term.lower() in lower for term in terms)]
    return hits or ["other"]


def fetch_page(task: dict, before: str) -> list:
    params = {
        "subreddit": task["subreddit"],
        "body": task["query"],
        "after": task["after"],
        "before": before,
        "limit": 100,
        "sort": "desc",
        "fields": FIELDS,
    }
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=45)
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            if attempt == 2:
                return []
            time.sleep(1 + attempt)
    return []


def normalize(item: dict, task: dict) -> Optional[dict]:
    body = clean_body(item.get("body", ""))
    if len(body) < 12 or body in {"[deleted]", "[removed]"}:
        return None
    if item.get("author") in {"AutoModerator", "CryptoDaily-", "coinfeeds-bot"}:
        return None
    created = int(item.get("created_utc") or 0)
    cats = classify(body)
    return {
        "period": task["period"],
        "period_label": task["label"],
        "query": task["query"],
        "subreddit": item.get("subreddit") or task["subreddit"],
        "id": item.get("id", ""),
        "link_id": item.get("link_id", ""),
        "parent_id": item.get("parent_id", ""),
        "author": item.get("author", ""),
        "created_utc": created,
        "created_at": datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else "",
        "score": item.get("score", 0),
        "categories": "|".join(cats),
        "body": body,
        "permalink": f"https://www.reddit.com/r/{item.get('subreddit')}/comments/{str(item.get('link_id', '')).replace('t3_', '')}/_/{item.get('id')}/",
    }


def collect_task(task: dict, max_pages: int = 4) -> list:
    rows = []
    before = task["before"]
    for _ in range(max_pages):
        data = fetch_page(task, before)
        if not data:
            break
        min_ts = None
        for item in data:
            row = normalize(item, task)
            if row:
                rows.append(row)
            created = int(item.get("created_utc") or 0)
            min_ts = created if min_ts is None else min(min_ts, created)
        if not min_ts or len(data) < 100:
            break
        before = str(min_ts - 1)
        time.sleep(0.1)
    return rows


def write_csv(rows: list) -> None:
    fields = ["period", "period_label", "query", "subreddit", "id", "link_id", "parent_id", "author", "created_utc", "created_at", "score", "categories", "body", "permalink"]
    with RAW_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list) -> dict:
    by_period = defaultdict(list)
    for row in rows:
        by_period[row["period"]].append(row)
    summary = {}
    for period, items in by_period.items():
        cats, subs, queries = Counter(), Counter(), Counter()
        for row in items:
            subs[row["subreddit"]] += 1
            queries[row["query"]] += 1
            for cat in row["categories"].split("|"):
                cats[cat] += 1
        summary[period] = {
            "count": len(items),
            "categories": dict(cats.most_common()),
            "subreddits": dict(subs.most_common()),
            "queries": dict(queries.most_common()),
            "top_comments": sorted(
                [{"score": r["score"], "subreddit": r["subreddit"], "created_at": r["created_at"], "categories": r["categories"], "body": r["body"][:520], "url": r["permalink"]} for r in items],
                key=lambda x: int(x["score"] or 0),
                reverse=True,
            )[:25],
        }
    return summary


def write_outputs(rows: list) -> None:
    rows = sorted(rows, key=lambda r: (r["period"], r["created_utc"], r["id"]))
    write_csv(rows)
    summary = summarize(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# BTC Retail Sentiment Dataset",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total Reddit comments: {len(rows)}",
        "",
        "## Period Counts",
        "",
        "| Period | Comments | Top Categories | Top Subreddits |",
        "|---|---:|---|---|",
    ]
    for window in WINDOWS:
        info = summary.get(window["period"], {"count": 0, "categories": {}, "subreddits": {}})
        cats = ", ".join(f"{k}:{v}" for k, v in list(info["categories"].items())[:6])
        subs = ", ".join(f"{k}:{v}" for k, v in list(info["subreddits"].items())[:5])
        lines.append(f"| {window['label']} | {info['count']} | {cats} | {subs} |")
    lines.extend([
        "",
        "## Method Notes",
        "",
        "- Source: Arctic Shift Reddit historical archive API.",
        "- Sampling: keyword comment search within crash windows, up to 4 descending pages per period/subreddit/query.",
        "- Classification: deterministic keyword tags; comments can have multiple tags.",
        "- Zhihu is not merged into this raw file; accessible Zhihu pages triggered anti-bot checks in manual tests.",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for window in WINDOWS:
        for subreddit in SUBREDDITS:
            for query in window["queries"]:
                tasks.append({
                    "period": window["period"],
                    "label": window["label"],
                    "after": window["after"],
                    "before": window["before"],
                    "query": query,
                    "subreddit": subreddit,
                })

    dedup = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(collect_task, task): task for task in tasks}
        for idx, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                if row["id"]:
                    dedup[row["id"]] = row
            if idx % 20 == 0:
                write_outputs(list(dedup.values()))
                print(f"done={idx}/{len(tasks)} unique={len(dedup)}", flush=True)
    write_outputs(list(dedup.values()))
    print(f"final unique={len(dedup)}", flush=True)
    print(RAW_CSV, flush=True)


if __name__ == "__main__":
    main()
