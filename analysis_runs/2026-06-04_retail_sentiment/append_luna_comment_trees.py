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
API_POSTS = "https://arctic-shift.photon-reddit.com/api/posts/search"
API_TREE = "https://arctic-shift.photon-reddit.com/api/comments/tree"
UA = "btc-retail-sentiment-research/1.0"

FIELDS = ["period", "period_label", "query", "subreddit", "id", "link_id", "parent_id", "author", "created_utc", "created_at", "score", "categories", "body", "permalink"]

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


def clean_body(body):
    return re.sub(r"\s+", " ", (body or "").replace("\r", " ").replace("\n", " ").strip())


def classify(body):
    lower = body.lower()
    hits = [cat for cat, terms in CATEGORIES.items() if any(term.lower() in lower for term in terms)]
    return hits or ["other"]


def load_existing():
    rows = {}
    if not RAW_CSV.exists():
        return rows
    with RAW_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row
    return rows


def search_posts():
    queries = ["terra", "luna", "celsius", "3ac", "bitcoin crash", "bitcoin below", "crypto winter"]
    subreddits = ["CryptoCurrency", "Bitcoin", "CryptoMarkets", "Buttcoin"]
    posts = {}
    for subreddit in subreddits:
        for query in queries:
            params = {
                "subreddit": subreddit,
                "query": query,
                "after": "2022-05-08",
                "before": "2022-07-15",
                "limit": 25,
                "sort": "desc",
                "fields": "id,title,score,num_comments,created_utc,subreddit",
            }
            r = requests.get(API_POSTS, params=params, headers={"User-Agent": UA}, timeout=45)
            if not r.ok:
                continue
            for post in r.json().get("data", []):
                if int(post.get("num_comments") or 0) >= 20:
                    posts[post["id"]] = post
            time.sleep(0.1)
    return sorted(posts.values(), key=lambda p: int(p.get("num_comments") or 0), reverse=True)[:35]


def flatten_tree(nodes):
    flat = []
    for node in nodes:
        if node.get("kind") != "t1":
            continue
        data = node.get("data") or {}
        flat.append(data)
        replies = data.get("replies")
        if isinstance(replies, dict):
            children = ((replies.get("data") or {}).get("children") or [])
            flat.extend(flatten_tree(children))
    return flat


def fetch_tree(post):
    params = {"link_id": "t3_" + post["id"], "limit": 25000, "start_breadth": 999, "start_depth": 20}
    r = requests.get(API_TREE, params=params, headers={"User-Agent": UA}, timeout=60)
    if not r.ok:
        return []
    return flatten_tree(r.json().get("data", []))


def normalize_comment(item, post):
    body = clean_body(item.get("body", ""))
    if len(body) < 12 or body in {"[deleted]", "[removed]"}:
        return None
    if item.get("author") in {"AutoModerator", "CryptoDaily-", "coinfeeds-bot"}:
        return None
    created = int(item.get("created_utc") or 0)
    subreddit = item.get("subreddit") or post.get("subreddit", "")
    cid = item.get("id", "")
    return {
        "period": "2022_luna_celsius",
        "period_label": "2022 LUNA/Celsius crash",
        "query": "comment_tree:" + post.get("title", "")[:80],
        "subreddit": subreddit,
        "id": cid,
        "link_id": item.get("link_id") or "t3_" + post["id"],
        "parent_id": item.get("parent_id", ""),
        "author": item.get("author", ""),
        "created_utc": created,
        "created_at": datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else "",
        "score": item.get("score", 0),
        "categories": "|".join(classify(body)),
        "body": body,
        "permalink": f"https://www.reddit.com/r/{subreddit}/comments/{post['id']}/_/{cid}/",
    }


def write_csv(rows):
    ordered = sorted(rows.values(), key=lambda r: (r["period"], int(r.get("created_utc") or 0), r["id"]))
    with RAW_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    return ordered


def write_summary(rows):
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
            "queries": dict(queries.most_common(20)),
            "top_comments": sorted(
                [{"score": r["score"], "subreddit": r["subreddit"], "created_at": r["created_at"], "categories": r["categories"], "body": r["body"][:520], "url": r["permalink"]} for r in items],
                key=lambda x: int(x["score"] or 0),
                reverse=True,
            )[:25],
        }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# BTC Retail Sentiment Dataset",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total Reddit comments: {len(rows)}",
        "",
        "| Period | Comments | Top Categories | Top Subreddits |",
        "|---|---:|---|---|",
    ]
    labels = {
        "2022_luna_celsius": "2022 LUNA/Celsius crash",
        "2022_ftx": "2022 FTX crash",
        "2026_current": "2026-06 current drawdown",
    }
    for period, label in labels.items():
        info = summary.get(period, {"count": 0, "categories": {}, "subreddits": {}})
        cats = ", ".join(f"{k}:{v}" for k, v in list(info["categories"].items())[:6])
        subs = ", ".join(f"{k}:{v}" for k, v in list(info["subreddits"].items())[:5])
        lines.append(f"| {label} | {info['count']} | {cats} | {subs} |")
    lines.extend([
        "",
        "Sources: Arctic Shift comment search plus comment-tree pulls from high-comment 2022 LUNA/Celsius posts.",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_existing()
    posts = search_posts()
    print("posts", len(posts), flush=True)
    added = 0
    for i, post in enumerate(posts, 1):
        comments = fetch_tree(post)
        for item in comments:
            row = normalize_comment(item, post)
            if row and row["id"] not in rows:
                rows[row["id"]] = row
                added += 1
        if i % 5 == 0:
            ordered = write_csv(rows)
            write_summary(ordered)
            print(f"post={i}/{len(posts)} total={len(rows)} added={added}", flush=True)
        time.sleep(0.15)
    ordered = write_csv(rows)
    write_summary(ordered)
    print(f"final total={len(rows)} added={added}", flush=True)


if __name__ == "__main__":
    main()
