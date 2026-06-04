import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
RAW_CSV = OUT_DIR / "reddit_retail_comments_raw.csv"
ENRICHED_CSV = OUT_DIR / "reddit_retail_comments_classified_v2.csv"
SUMMARY_JSON = OUT_DIR / "retail_sentiment_summary_v2.json"
REPORT_MD = OUT_DIR / "retail_sentiment_comparison_v2.md"

FIELDS = ["period", "period_label", "query", "subreddit", "id", "link_id", "parent_id", "author", "created_utc", "created_at", "score", "categories_v1", "categories_v2", "stance", "body", "permalink"]


PATTERNS = {
    "fraud_cefi_counterparty": [
        r"\bftx\b", r"\bsbf\b", r"\balameda\b", r"\bcelsius\b", r"\bvoyager\b", r"\bblockfi\b",
        r"\b3ac\b", r"\bterra\b", r"\bluna\b", r"\bust\b", r"\bdo kwon\b", r"\bmashinsky\b",
        r"\bponzi\b", r"\bscam\w*\b", r"\bfraud\w*\b", r"\bbankrupt\w*\b", r"\binsolven\w*\b",
        r"\bwithdraw\w*\b", r"交易所", r"提现", r"破产", r"暴雷", r"挤兑", r"骗局", r"旁氏",
    ],
    "institutional_flow": [
        r"\betf\w*\b", r"\bblackrock\b", r"\bfidelity\b", r"\bwall street\b", r"\bsaylor\b",
        r"\bstrategy\b", r"\bmstr\b", r"\binstitution\w*\b", r"\boutflow\w*\b", r"\binflow\w*\b",
        r"机构", r"现货", r"资金流出", r"贝莱德",
    ],
    "leverage_liquidation": [
        r"\bleverage\w*\b", r"\bliquidat\w*\b", r"\bmargin\b", r"\blongs?\b", r"\bshorts?\b",
        r"\bfutures?\b", r"爆仓", r"杠杆", r"清算", r"合约",
    ],
    "macro_liquidity": [
        r"\bfed\b", r"\brates?\b", r"\binflation\b", r"\bliquidity\b", r"\brecession\b",
        r"\bdollar\b", r"\bstocks?\b", r"\bnasdaq\b", r"\bgold\b", r"\bbonds?\b", r"\btreasur\w*\b",
        r"宏观", r"加息", r"降息", r"流动性", r"美股", r"黄金",
    ],
    "ai_capital_rotation": [
        r"\bai\b", r"\bnvidia\b", r"\bnvda\b", r"\bpalantir\b", r"\bpltr\b", r"\btech stocks?\b",
        r"人工智能", r"英伟达", r"科技股",
    ],
    "buy_dip_hodl": [
        r"\bbuy the dip\b", r"\bbtd\b", r"\bdca\b", r"\bhodl\b", r"\bstack\w*\b", r"\baccumulat\w*\b",
        r"\bcheap\b", r"\bdiscount\b", r"\bbuying more\b", r"买入", r"抄底", r"定投", r"拿住", r"持有", r"便宜",
    ],
    "panic_capitulation": [
        r"\bpanic\b", r"\bscared\b", r"\bfear\b", r"\bworried\b", r"\bblood\b", r"\bdead\b",
        r"\bzero\b", r"\bcapitulat\w*\b", r"\bdespair\b", r"\bcrash\w*\b", r"\bdump\w*\b",
        r"崩", r"慌", r"完了", r"归零", r"血", r"割肉",
    ],
    "cycle_thesis": [
        r"\bcycle\b", r"\bhalving\b", r"\bfour year\b", r"\b4 year\b", r"\bbear market\b",
        r"\bbull market\b", r"\bath\b", r"周期", r"减半", r"熊市", r"牛市", r"新高",
    ],
}

BULLISH = [r"\bbuy\b", r"\bdca\b", r"\bhodl\b", r"\bstack\w*\b", r"\baccumulat\w*\b", r"\bbull\w*\b", r"\bcheap\b", r"抄底", r"买入", r"定投"]
BEARISH = [r"\bsell\b", r"\bcrash\w*\b", r"\bdump\w*\b", r"\bdead\b", r"\bpanic\b", r"\bscam\w*\b", r"\bfraud\w*\b", r"\bponzi\b", r"暴跌", r"崩", r"骗局", r"割肉"]


def clean(text):
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def has(patterns, text):
    return any(re.search(p, text, flags=re.I) for p in patterns)


def classify(text):
    cats = [name for name, pats in PATTERNS.items() if has(pats, text)]
    return cats or ["other"]


def stance(text):
    bull = has(BULLISH, text)
    bear = has(BEARISH, text)
    if bull and bear:
        return "mixed"
    if bull:
        return "buy_supportive"
    if bear:
        return "bearish_fear"
    return "neutral_or_analysis"


def read_rows():
    with RAW_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            body = clean(row.get("body", ""))
            cats = classify(body)
            yield {
                "period": row["period"],
                "period_label": row["period_label"],
                "query": row["query"],
                "subreddit": row["subreddit"],
                "id": row["id"],
                "link_id": row["link_id"],
                "parent_id": row["parent_id"],
                "author": row["author"],
                "created_utc": row["created_utc"],
                "created_at": row["created_at"],
                "score": row["score"],
                "categories_v1": row.get("categories", ""),
                "categories_v2": "|".join(cats),
                "stance": stance(body),
                "body": body,
                "permalink": row["permalink"],
            }


def pct(n, d):
    return round(n * 100.0 / d, 1) if d else 0


def main():
    rows = list(read_rows())
    with ENRICHED_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_period = defaultdict(list)
    for row in rows:
        by_period[row["period"]].append(row)

    summary = {}
    for period, items in by_period.items():
        cats = Counter()
        stance_counts = Counter()
        subs = Counter()
        days = Counter()
        examples = defaultdict(list)
        for row in items:
            stance_counts[row["stance"]] += 1
            subs[row["subreddit"]] += 1
            days[row["created_at"][:10]] += 1
            for cat in row["categories_v2"].split("|"):
                cats[cat] += 1
                if cat != "other" and len(examples[cat]) < 6:
                    examples[cat].append({
                        "score": int(row["score"] or 0),
                        "body": row["body"][:300],
                        "url": row["permalink"],
                    })
        top_comments = sorted(
            [{"score": int(r["score"] or 0), "body": r["body"][:420], "categories": r["categories_v2"], "url": r["permalink"]} for r in items],
            key=lambda x: x["score"],
            reverse=True,
        )[:15]
        summary[period] = {
            "count": len(items),
            "categories": dict(cats.most_common()),
            "category_percent": {k: pct(v, len(items)) for k, v in cats.most_common()},
            "stance": dict(stance_counts.most_common()),
            "stance_percent": {k: pct(v, len(items)) for k, v in stance_counts.most_common()},
            "subreddits": dict(subs.most_common()),
            "daily_counts": dict(sorted(days.items())),
            "examples": dict(examples),
            "top_comments": top_comments,
        }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = {
        "2022_luna_celsius": "2022 LUNA/Celsius",
        "2022_ftx": "2022 FTX",
        "2026_current": "2026-06 当前",
    }
    lines = [
        "# BTC 散户评论情绪对比 v2",
        "",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}",
        f"样本总数: {len(rows)} 条 Reddit 评论",
        "",
        "## 1. 样本量",
        "",
        "| 时段 | 评论数 | 主要来源 |",
        "|---|---:|---|",
    ]
    for period, label in labels.items():
        info = summary.get(period, {})
        subs = ", ".join(f"{k}:{v}" for k, v in list(info.get("subreddits", {}).items())[:4])
        lines.append(f"| {label} | {info.get('count', 0)} | {subs} |")
    lines.extend(["", "## 2. 分类占比", "", "| 时段 | 欺诈/CeFi | 机构/ETF | 杠杆清算 | 宏观流动性 | AI分流 | 抄底/HODL | 恐慌投降 | 周期叙事 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for period, label in labels.items():
        p = summary[period]["category_percent"]
        lines.append(
            f"| {label} | {p.get('fraud_cefi_counterparty',0)}% | {p.get('institutional_flow',0)}% | "
            f"{p.get('leverage_liquidation',0)}% | {p.get('macro_liquidity',0)}% | {p.get('ai_capital_rotation',0)}% | "
            f"{p.get('buy_dip_hodl',0)}% | {p.get('panic_capitulation',0)}% | {p.get('cycle_thesis',0)}% |"
        )
    lines.extend(["", "## 3. 立场粗分", "", "| 时段 | 抄底/支持 | 看空/恐慌 | 矛盾混合 | 中性/分析 |", "|---|---:|---:|---:|---:|"])
    for period, label in labels.items():
        p = summary[period]["stance_percent"]
        lines.append(f"| {label} | {p.get('buy_supportive',0)}% | {p.get('bearish_fear',0)}% | {p.get('mixed',0)}% | {p.get('neutral_or_analysis',0)}% |")
    lines.extend([
        "",
        "## 4. 结论",
        "",
        "- 2022 LUNA/Celsius: 评论核心是对中心化借贷、算法稳定币和高收益产品的信任崩塌；散户讨论集中在 Do Kwon、Mashinsky、破产、提款、平台风险。",
        "- 2022 FTX: 欺诈/CeFi 标签最高，讨论从价格下跌转成交易所信用、托管风险和行业系统性腐烂；恐慌占比也更高。",
        "- 2026 当前: 机构/ETF、Saylor/Strategy、AI/科技股分流、宏观流动性和杠杆清算显著上升；这轮更像资金结构和叙事溢价的挤压，还不是 2022 那种信用链条崩坏。",
        "- 抄底/HODL 三段都存在，但 2026 当前更像“老信仰 + 新怀疑”并存；2022 的抄底更多夹杂幸存者反思。",
        "",
        "## 5. 数据文件",
        "",
        f"- 原始评论: `{RAW_CSV.name}`",
        f"- v2 分类评论: `{ENRICHED_CSV.name}`",
        f"- v2 统计 JSON: `{SUMMARY_JSON.name}`",
    ])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_MD)


if __name__ == "__main__":
    main()
