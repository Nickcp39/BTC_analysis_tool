"""
Step03: 时间收缩与周期缩放比
- BTC 时间模型在收缩（某段时间越来越短），对比时需把后一周期按时间缩放以便对齐。
- 用 docs 里「越来越短」的那段：顶峰→下一减半（或可配置为 减半→顶峰，取实际缩短的那条）。
- 本步只算各周期天数与缩放比，供你查看；后续再用于曲线对齐。
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import List, Tuple
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
CURRENT_DATE = date.today().strftime("%Y-%m-%d")
OUTDIR = ROOT / "visualization" / CURRENT_DATE

# ========= 锚点（与 step02 / docs 一致）=========
HALVINGS: List[date] = [
    date(2012, 11, 28),
    date(2016, 7, 9),
    date(2020, 5, 11),
    date(2024, 4, 20),
]
TOPS: List[date] = [
    date(2013, 11, 29),
    date(2017, 12, 17),
    date(2021, 11, 10),
]


@dataclass
class CycleDuration:
    """单周期的时间跨度（天）与相对参考的缩放比。"""
    cycle_label: str
    start_date: date
    end_date: date
    days: int
    scale_to_ref: float  # 参考周期天数 / 本周期天数；>1 表示本周期更短，需拉长对齐


def halving_to_peak_days() -> List[CycleDuration]:
    """减半→当轮大顶（docs: 366, 526, 548 → 逐步延长）。"""
    ref_days = (TOPS[0] - HALVINGS[0]).days
    out: List[CycleDuration] = []
    labels = ["2012→2013", "2016→2017", "2020→2021"]
    for i in range(len(TOPS)):
        d = (TOPS[i] - HALVINGS[i]).days
        scale = ref_days / d if d else 1.0
        out.append(CycleDuration(
            cycle_label=labels[i],
            start_date=HALVINGS[i],
            end_date=TOPS[i],
            days=d,
            scale_to_ref=scale,
        ))
    return out


def peak_to_next_halving_days() -> List[CycleDuration]:
    """顶峰→下一减半（这段在缩短：第一段很长，后面变短）。"""
    # 参考用第一段（最长），后面周期更短，缩放比 >1 表示要拉长后周期
    ref_days = (HALVINGS[1] - TOPS[0]).days
    out: List[CycleDuration] = []
    labels = ["2013→2016", "2017→2020", "2021→2024"]
    for i in range(len(TOPS)):
        d = (HALVINGS[i + 1] - TOPS[i]).days
        scale = ref_days / d if d else 1.0
        out.append(CycleDuration(
            cycle_label=labels[i],
            start_date=TOPS[i],
            end_date=HALVINGS[i + 1],
            days=d,
            scale_to_ref=scale,
        ))
    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) 减半→顶峰（docs 写的是「逐步延长」）
    h2p = halving_to_peak_days()
    # 2) 顶峰→下一减半（第一段最长，后面缩短 → 用这条做「收缩」）
    p2h = peak_to_next_halving_days()

    # 判断哪条是「越来越短」：看相邻是否递减
    def is_shortening(cycles: List[CycleDuration]) -> bool:
        if len(cycles) < 2:
            return False
        for i in range(1, len(cycles)):
            if cycles[i].days >= cycles[i - 1].days:
                return False
        return True

    def is_lengthening(cycles: List[CycleDuration]) -> bool:
        if len(cycles) < 2:
            return False
        for i in range(1, len(cycles)):
            if cycles[i].days <= cycles[i - 1].days:
                return False
        return True

    use_contracting = p2h  # 默认用「顶峰→下一减半」作为收缩时间
    if is_shortening(p2h):
        name_used = "顶峰→下一减半"
        used = p2h
    elif is_shortening(h2p):
        name_used = "减半→顶峰"
        used = h2p
    else:
        # 若都不严格单调短，仍用顶峰→下一减半（第一段明显长，后两段短）
        name_used = "顶峰→下一减半"
        used = p2h

    # 输出：缩放比表
    lines = [
        "=== Step03 时间收缩与周期缩放比 ===",
        "",
        "【减半→当轮大顶】docs: 366, 526, 548 天 → 逐步延长",
        *[f"  {c.cycle_label}: {c.days} 天  (相对第1周期缩放比={c.scale_to_ref:.4f})" for c in h2p],
        "",
        "【顶峰→下一减半】第一段长，后段缩短 → 用于对比时缩放后周期",
        *[f"  {c.cycle_label}: {c.days} 天  (相对第1周期缩放比={c.scale_to_ref:.4f})" for c in p2h],
        "",
        f"采用时间: {name_used}",
        "缩放含义: scale_to_ref = 第1周期天数 / 本周期天数；>1 表示本周期更短，对比时把本周期时间轴拉长该倍数。",
        "",
        "--- 各周期缩放比（用于时间轴对齐）---",
    ]
    for c in used:
        lines.append(f"  {c.cycle_label}: days={c.days}, scale_to_ref={c.scale_to_ref:.4f}")

    txt_path = OUTDIR / "step03_time_scale_ratios.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # CSV：方便后续程序读
    import csv
    csv_path = OUTDIR / "step03_time_scale_ratios.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "cycle_label", "start_date", "end_date", "days", "scale_to_ref"])
        for c in h2p:
            w.writerow(["halving_to_peak", c.cycle_label, c.start_date, c.end_date, c.days, f"{c.scale_to_ref:.6f}"])
        for c in p2h:
            w.writerow(["peak_to_next_halving", c.cycle_label, c.start_date, c.end_date, c.days, f"{c.scale_to_ref:.6f}"])

    # 控制台打印（仅关键数字，避免 Windows 控制台编码问题）
    print("Step03 time scale ratios (see TXT for full text):")
    for i, c in enumerate(used):
        print(f"  cycle_{i} {c.days}d scale_to_ref={c.scale_to_ref:.4f}")
    print("Output:", txt_path.resolve())
    print("CSV:  ", csv_path.resolve())


if __name__ == "__main__":
    main()
