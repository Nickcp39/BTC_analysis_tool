import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests


OUT_DIR = Path(__file__).resolve().parent
RAW_CSV = OUT_DIR / "reddit_retail_comments_raw.csv"
SUMMARY_JSON = OUT_DIR / "reddit_retail_summary.json"
REPORT_MD = OUT_DIR / "retail_sentiment_report.md"

API = "https://api.pullpush.io/reddit/search/comment/"
UA = "btc-retail-sentiment-research/1.0"


WINDOWS = [
    {
        "period": "2022_luna_celsius",
        "label": "2022 LUNA/Celsius crash",
        "after": "2022-05-08",
        "before": "2022-07-15",
        "target": 2600,
        "queries": ["bitcoin", "btc", "luna", "ust", "celsius", "3ac", "crash", "dump", "dip", "liquidation"],
    },
    {
        "period": "2022_ftx",
        "label": "2022 FTX crash",
        "after": "2022-11-02",
        "before": "2022-11-25",
        "target": 2200,
        "queries": ["bitcoin", "btc", "ftx", "sbf", "alameda", "binance", "withdraw", "crash", "dump", "contagion"],
    },
    {
        "period": "2026_current",
        "label": "2026-06 current drawdown",
        "after": "2026-06-01",
        "before": "2026-06-05",
        "target": 2600,
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
    "panic_capitulation": [
        "panic", "scared", "fear", "worried", "blood", "dead", "over", "zero", "sell everything",
        "capitulation", "despair", "崩", "慌", "完了", "归零", "血", "割肉",
    ],
    "buy_dip_hodl": [
        "buy the dip", "btd", "dca", "hodl", "stack", "accumulate", "cheap", "discount",
        "buying more", "买入", "抄底", "定投", "拿住", "持有", "便宜",
    ],
    "blame_leverage_liquidation": [
        "leverage", "liquidation", "liquidated", "margin", "longs", "shorts", "overleveraged",
        "爆仓", "杠杆", "清算", "合约",
    ],
    "institutional_flow": [
        "etf", "institution", "blackrock", "fidelity", "wall street", "saylor", "strategy", "mstr",
        "机构", "现货", "资金流出", "贝莱德",
    ],
    "macro_rates_liquidity": [
        "fed", "rates", "inflation", "liquidity", "recession", "dollar", "stocks", "nasdaq", "gold",
        "宏观", "加息", "降息", "流动性", "美股", "黄金",
    ],
    "fraud_cefi_counterparty": [
        "scam", "fraud", "ponzi", "ftx", "sbf", "alameda", "celsius", "voyager", "blockfi",
        "3ac", "terra", "luna", "ust", "exchange", "withdrawals", "bankruptcy", "insolvent",
        "骗局", "旁氏", "交易所", "提现", "破产", "暴雷", "挤兑",
    ],
    "ai_capital_rotation": [
        "ai", "nvidia", "nvda", "palantir", "pltr", "tech stocks", "人工智能", "英伟达", "科技股",
    ],
    "cycle_thesis": [
        "cycle", "halving", "four year", "4 year", "bear market", "bull market", "ath",
        "周期", "减半", "熊市", "牛市", "新高",
    ],
}


def ts(date_s: str) -> int:
    return int(datetime.fromisoformat(date_s).replace(tzinfo=timezone.utc).timestamp())


def clean_body(body: str) -> str:
    body = (body or "").replace("\r", " ").replace("\n", " ").strip()
    body = re.sub(r"\s+", " ", body)
    return body


def is_retail_comment(row: dict) -> bool:
    body = clean_body(row.get("body", ""))
    if not body or body in {"[deleted]", "[removed]"}:
        return False
    if row.get("author") in {"AutoModerator", "CryptoDaily-", "coinfeeds-bot"}:
        return False
    if len(body) < 12:
        return False
    return True


def classify(body: str) -> list[str]:
    lower = body.lower()
    hits = []
    for category, needles in CATEGORIES.items():
        if any(needle.lower() in lower for needle in needles):
            hits.append(category)
    return hits or ["other"]


def fetch_page(params: dict) -> list[dict]:
    for attempt in range(4):
        try:
            resp = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 + attempt * 3)
                continue
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("data", [])
        except Exception:
            if attempt == 3:
                return []
            time.sleep(1 + attempt * 2)
    return []


def collect_window(window: dict) -> list[dict]:
    seen = set()
    rows = []
    start_ts = ts(window["after"])
    end_ts = ts(window["before"])

    for subreddit in SUBREDDITS:
        if len(rows) >= window["target"]:
            break
        for query in window["queries"]:
            if len(rows) >= window["target"]:
                break
            cursor = start_ts
            empty_pages = 0
            while len(rows) < window["target"] and cursor < end_ts and empty_pages < 2:
                params = {
                    "q": query,
                    "subreddit": subreddit,
                    "after": cursor,
                    "before": end_ts,
                    "size": 100,
                    "sort": "asc",
                    "sort_type": "created_utc",
                }
                data = fetch_page(params)
                if not data:
                    empty_pages += 1
                    break
                empty_pages = 0
                max_seen_ts = cursor
                added = 0
                for item in data:
                    cid = item.get("id")
                    max_seen_ts = max(max_seen_ts, int(item.get("created_utc") or cursor))
                    if not cid or cid in seen or not is_retail_comment(item):
                        continue
                    body = clean_body(item.get("body", ""))
                    cats = classify(body)
                    seen.add(cid)
                    rows.append(
                        {
                            "period": window["period"],
                            "period_label": window["label"],
                            "query": query,
                            "subreddit": item.get("subreddit") or subreddit,
                            "id": cid,
                            "link_id": item.get("link_id", ""),
                            "parent_id": item.get("parent_id", ""),
                            "author": item.get("author", ""),
                            "created_utc": int(item.get("created_utc") or 0),
                            "created_at": datetime.fromtimestamp(int(item.get("created_utc") or 0), tz=timezone.utc).isoformat(),
                            "score": item.get("score", 0),
                            "categories": "|".join(cats),
                            "body": body,
                            "permalink": f"https://www.reddit.com{item.get('permalink', '')}" if item.get("permalink") else "",
                        }
                    )
                    added += 1
                cursor = max_seen_ts + 1
                if added == 0 and len(data) < 100:
                    break
                time.sleep(0.15)
    return rows


def write_outputs(rows: list[dict]) -> None:
    fields = [
        "period", "period_label", "query", "subreddit", "id", "link_id", "parent_id", "author",
        "created_utc", "created_at", "score", "categories", "body", "permalink",
    ]
    with RAW_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_period = defaultdict(list)
    for row in rows:
        by_period[row["period"]].append(row)

    summary = {}
    for period, items in by_period.items():
        cat_counter = Counter()
        sub_counter = Counter()
        query_counter = Counter()
        for row in items:
            sub_counter[row["subreddit"]] += 1
            query_counter[row["query"]] += 1
            for cat in row["categories"].split("|"):
                cat_counter[cat] += 1
        summary[period] = {
            "count": len(items),
            "subreddits": dict(sub_counter.most_common()),
            "queries": dict(query_counter.most_common()),
            "categories": dict(cat_counter.most_common()),
            "top_comments": [
                {
                    "created_at": row["created_at"],
                    "subreddit": row["subreddit"],
                    "score": row["score"],
                    "categories": row["categories"],
                    "body": row["body"][:500],
                    "permalink": row["permalink"],
                }
                for row in sorted(items, key=lambda r: int(r.get("score") or 0), reverse=True)[:12]
            ],
        }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# BTC Retail Sentiment Pull",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total comments: {len(rows)}",
        "",
        "## Counts",
        "",
        "| Period | Comments | Top categories | Top subreddits |",
        "|---|---:|---|---|",
    ]
    for window in WINDOWS:
        period = window["period"]
        info = summary.get(period, {"count": 0, "categories": {}, "subreddits": {}})
        cats = ", ".join(f"{k}:{v}" for k, v in list(info["categories"].items())[:5])
        subs = ", ".join(f"{k}:{v}" for k, v in list(info["subreddits"].items())[:5])
        lines.append(f"| {window['label']} | {info['count']} | {cats} | {subs} |")
    lines.extend(["", "## Notes", "", "- Source: PullPush Reddit comment search API.", "- Zhihu is not included in this raw file because accessible pages triggered anti-bot verification during this run."])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for window in WINDOWS:
        rows = collect_window(window)
        print(window["period"], len(rows))
        all_rows.extend(rows)
    dedup = {}
    for row in all_rows:
        dedup[row["id"]] = row
    write_outputs(list(dedup.values()))
    print(RAW_CSV)
    print(SUMMARY_JSON)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
