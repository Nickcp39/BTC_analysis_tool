"""汇总层：跑全部四个指标 → 一张当前读数表 + 综合结论。

用法:
    python signals/snapshot.py            # 用本地已有数据算
    python signals/snapshot.py --refresh  # 先抓最新数据再算

产出:
    outputs/snapshot_YYYY-MM-DD.json
    outputs/snapshot_YYYY-MM-DD.md
并在终端打印摘要。
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sources as S
import indicators as I


def _verdict(total: int) -> str:
    for threshold, text in C.VERDICT_BANDS:
        if total >= threshold:
            return text
    return C.VERDICT_BANDS[-1][1]


def build_snapshot(refresh: bool = False) -> dict:
    if refresh:
        S.fetch_btc_daily()
        S.fetch_fear_greed()

    price_df = pd.read_csv(C.PRICE_CSV, parse_dates=["date"])
    fng_df = pd.read_csv(C.FNG_CSV, parse_dates=["date"])
    lth_df = S.load_lth_metrics()

    ahr = I.ahr999_signal(price_df)
    fng = I.fng_signal(fng_df)
    lth = I.lth_signal(lth_df)

    total = ahr["score"] + fng["score"] + lth["score"]
    max_score = 2 + 2 + 2   # AHR(2) + FNG(2) + LTH(action1+ratio1=2)

    return {
        "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        "btc_price": ahr["price"],
        "indicators": {"ahr999": ahr, "fear_greed": fng, "lth": lth},
        "composite_score": total,
        "max_score": max_score,
        "verdict": _verdict(total),
    }


def to_markdown(snap: dict) -> str:
    a, f, l = snap["indicators"]["ahr999"], snap["indicators"]["fear_greed"], snap["indicators"]["lth"]
    lines = [
        f"# BTC 底部信号快照 — {snap['generated_at'][:10]}",
        "",
        f"**当前 BTC ≈ ${snap['btc_price']:,.0f}**　|　"
        f"综合得分 **{snap['composite_score']:+d} / {snap['max_score']}**　|　"
        f"**结论：{snap['verdict']}**",
        "",
        "| 指标 | 当前值 | 区位 | 信号分 |",
        "|---|---|---|---|",
        f"| AHR999 ({a['date']}) | {a['value']} | {a['zone']} | {a['score']:+d} |",
        f"| 恐慌贪婪 ({f['date']}) | {f['value']} | {f['zone']} | {f['score']:+d} |",
        f"| 长持动作 ({l['date']}) | 30日净{'增' if l['lth_net_change_30d_btc']>0 else '减'} "
        f"{abs(l['lth_net_change_30d_btc']):,.0f} BTC | {l['phase']} | {l['action_score']:+d} |",
        f"| 长持占比 ({l['date']}) | {l['lth_ratio']:.1%} | {l['ratio_zone']} | {l['ratio_score']:+d} |",
        "",
        f"- AHR999 辅助：现价 ${a['price']:,.0f}，200日成本 ${a['gma200']:,.0f}，"
        f"估值中枢 ${a['estimate_price']:,.0f}；是否深度价值区：{'是' if a['in_deep_value'] else '否'}",
        f"- 长持供应：{l['lth_supply_btc']:,.0f} BTC　来源：{l['source']}",
        "",
        "> 打分为透明启发式（见 signals/config.py），非买卖触发器。底常是一段区间，建议分批/区间执行。",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="先抓取最新数据再分析")
    args = ap.parse_args()

    snap = build_snapshot(refresh=args.refresh)
    md = to_markdown(snap)

    day = snap["generated_at"][:10]
    (C.OUT_DIR / f"snapshot_{day}.json").write_text(
        json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.OUT_DIR / f"snapshot_{day}.md").write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"\n已保存: outputs/snapshot_{day}.json / .md")


if __name__ == "__main__":
    main()
