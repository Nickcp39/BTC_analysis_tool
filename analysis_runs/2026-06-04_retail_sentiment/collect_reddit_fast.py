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
API = "https://api.pullpush.io/reddit/search/comment/"
UA = "btc-retail-sentiment-research/1.0"

WINDOWS = [
    ("2022_luna_celsius", "2022 LUNA/Celsius crash", "2022-05-08", "2022-07-15",
     ["bitcoin", "btc", "luna", "ust", "celsius", "3ac", "crash", "dump", "dip", "liquidation"]),
    ("2022_ftx", "2022 FTX crash", "2022-11-02", "2022-11-25",
     ["bitcoin", "btc", "ftx", "sbf", "alameda", "binance", "withdraw", "crash", "dump", "contagion"]),
    ("2026_current", "2026-06 current drawdown", "2026-06-01", "2026-06-05",
     ["bitcoin", "btc", "drop", "crash", "dump", "liquidation", "etf", "saylor", "strategy", "ai", "dip"]),
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


def ts(date_s: str) -> int:
    return int(datetime.fromisoformat(date_s).replace(tzinfo=timezone.utc).timestamp())


def clean_body(body: str) -> str:
    body = (body or "").replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", body)


def classify(body: str) -> list[str]:
    lower = body.lower()
    hits = [cat for cat, terms in CATEGORIES.items() if any(term.lower() in lower for term in terms)]
    return hits or ["other"]


def fetch_task(task: dict) -> list[dict]:
    params = {
        "q": task["query"],
        "subreddit": task["subreddit"],
        "after": task["after"],
        "before": task["before"],
        "size": 100,
        "sort": task["sort"],
        "sort_type": "created_utc",
    }
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=25)
            if r.status_code == 429:
                time.sleep(2 + attempt)
                continue
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            if attempt == 2:
                return []
            time.sleep(0.8 + attempt)
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
        "permalink": f"https://www.reddit.com{item.get('permalink', '')}" if item.get("permalink") else "",
    }


def write_csv(rows: list[dict]) -> None:
    fields = ["period", "period_label", "query", "subreddit", "id", "link_id", "parent_id", "author", "created_utc", "created_at", "score", "categories", "body", "permalink"]
    with RAW_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    by_period = defaultdict(list)
    for row in rows:
        by_period[row["period"]].append(row)
    out = {}
    for period, items in by_period.items():
        cats, subs, queries = Counter(), Counter(), Counter()
        examples = defaultdict(list)
        for row in items:
            subs[row["subreddit"]] += 1
            queries[row["query"]] += 1
            for cat in row["categories"].split("|"):
                cats[cat] += 1
                if len(examples[cat]) < 8:
                    examples[cat].append({"score": row["score"], "body": row["body"][:360], "url": row["permalink"]})
        out[period] = {
            "count": len(items),
            "categories": dict(cats.most_common()),
            "subreddits": dict(subs.most_common()),
            "queries": dict(queries.most_common()),
            "examples": dict(examples),
            "top_comments": sorted(
                [{"score": r["score"], "subreddit": r["subreddit"], "created_at": r["created_at"], "categories": r["categories"], "body": r["body"][:500], "url": r["permalink"]} for r in items],
                key=lambda x: int(x["score"] or 0),
                reverse=True,
            )[:20],
        }
    return out


def write_report(rows: list[dict], summary: dict) -> None:
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
    for period, label, *_ in WINDOWS:
        info = summary.get(period, {"count": 0, "categories": {}, "subreddits": {}})
        cats = ", ".join(f"{k}:{v}" for k, v in list(info["categories"].items())[:6])
        subs = ", ".join(f"{k}:{v}" for k, v in list(info["subreddits"].items())[:5])
        lines.append(f"| {label} | {info['count']} | {cats} | {subs} |")
    lines += [
        "",
        "## Method Notes",
        "",
        "- Source: PullPush Reddit historical comment search.",
        "- Sampling: top 100 comments for each period/subreddit/query/sort pair, then deduplicated by comment id.",
        "- Classification: deterministic keyword tags; comments can have multiple tags.",
        "- Zhihu is not in this file; accessible Zhihu pages triggered anti-bot checks in manual tests.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for period, label, after, before, queries in WINDOWS:
        for subreddit in SUBREDDITS:
            for query in queries:
                for sort in ("desc", "asc"):
                    tasks.append({
                        "period": period,
                        "label": label,
                        "after": ts(after),
                        "before": ts(before),
                        "query": query,
                        "subreddit": subreddit,
                        "sort": sort,
                    })

    dedup = {}
    done = 0
    with ThreadPoolExecutor(max_workers=18) as ex:
        futures = {ex.submit(fetch_task, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            for item in fut.result():
                row = normalize(item, task)
                if row and row["id"]:
                    dedup[row["id"]] = row
            done += 1
            if done % 40 == 0:
                rows = list(dedup.values())
                write_csv(rows)
                print(f"done={done}/{len(tasks)} unique={len(rows)}", flush=True)

    rows = sorted(dedup.values(), key=lambda r: (r["period"], r["created_utc"], r["id"]))
    write_csv(rows)
    summary = summarize(rows)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(rows, summary)
    print(f"final unique={len(rows)}", flush=True)
    print(RAW_CSV, flush=True)
    print(SUMMARY_JSON, flush=True)
    print(REPORT_MD, flush=True)


if __name__ == "__main__":
    main()
