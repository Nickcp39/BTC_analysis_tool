"""
Test step02 time model only: compare with 4 fusion notebook / user post.
Expected (from post):
  基线（减半→顶中位数）: 526 天 -> 2025-09-28
  修正A（回归预测）  : 563 天 -> 2025-11-04
  修正B（顶→顶外推）: 560 天 -> 2025-11-01
  综合预测窗口       : 2025-09-28 -> 2025-11-04
  Time model A: r ~ 0.98, center 2025-11-04, alt_center 2025-09-28, window 2025-10-16 -> 2025-11-23
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from step02_halving_to_peak_fusion import (
    HALVINGS,
    TOPS,
    predict_peak_multi,
    predict_peak_window_by_time,
    fusion_time_center,
)


def main():
    multi = predict_peak_multi()
    time_res = predict_peak_window_by_time(u_factor=1.2, method="regress")
    fusion = fusion_time_center(multi, weights=(1.0, 1.3, 1.1))

    print("=== Step02 time model (same logic as 4 fusion) ===\n")
    print("Baseline (median halving->top):", multi.base_days, "days ->", multi.base_date)
    print("Correction A (regression)     :", multi.regress_days, "days ->", multi.regress_date)
    print("Correction B (top->top)       :", multi.peak2peak_days, "days ->", multi.peak2peak_date)
    print("Combined window               :", multi.window[0], "->", multi.window[1])
    print()
    print("Time model A:")
    print("  corr(halving_interval vs halving_to_top):", f"{time_res.corr_halving_vs_delay:.3f}")
    print("  center (regression):", time_res.center)
    print("  alt_center (median):", time_res.alt_center_by_median)
    print("  window:", time_res.window_lo, "->", time_res.window_hi)
    print()
    print("Fusion time center (for 2025 peak anchor):", fusion)

    expected = {
        "base_days": 526,
        "base_date": "2025-09-28",
        "regress_days": 563,
        "regress_date": "2025-11-04",
        "peak2peak_days": 560,
        "peak2peak_date": "2025-11-01",
        "window_lo": "2025-09-28",
        "window_hi": "2025-11-04",
        "corr": 0.980,
    }
    ok = True
    if multi.base_days != expected["base_days"]:
        print("\nMISMATCH base_days: got", multi.base_days, "expected", expected["base_days"]); ok = False
    if str(multi.base_date) != expected["base_date"]:
        print("\nMISMATCH base_date: got", multi.base_date, "expected", expected["base_date"]); ok = False
    if multi.regress_days != expected["regress_days"]:
        print("\nMISMATCH regress_days: got", multi.regress_days, "expected", expected["regress_days"]); ok = False
    if str(multi.regress_date) != expected["regress_date"]:
        print("\nMISMATCH regress_date: got", multi.regress_date, "expected", expected["regress_date"]); ok = False
    if multi.peak2peak_days != expected["peak2peak_days"]:
        print("\nMISMATCH peak2peak_days: got", multi.peak2peak_days, "expected", expected["peak2peak_days"]); ok = False
    if str(multi.peak2peak_date) != expected["peak2peak_date"]:
        print("\nMISMATCH peak2peak_date: got", multi.peak2peak_date, "expected", expected["peak2peak_date"]); ok = False
    if str(multi.window[0]) != expected["window_lo"] or str(multi.window[1]) != expected["window_hi"]:
        print("\nMISMATCH window: got", multi.window, "expected", expected["window_lo"], "->", expected["window_hi"]); ok = False
    if abs(time_res.corr_halving_vs_delay - expected["corr"]) > 0.01:
        print("\nMISMATCH corr: got", time_res.corr_halving_vs_delay, "expected ~", expected["corr"]); ok = False
    if ok:
        print("\n[OK] All time outputs match your post.")


if __name__ == "__main__":
    main()
