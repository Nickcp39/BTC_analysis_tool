"""
Test 4-fusion price logic: four-relation (nominal) and four-relation (Real) -> compare with user post.

Expected from post:
  Four-relation (real->nominal): best real top~122062 bottom~42722; best nominal top~133381 bottom~48369;
    band top 131264~201977, band bottom 27201~73244
  Four-relation (Real->nominal): best Real top~142615 bottom~48489; best Nominal top~142615 bottom~50894;
    band top 128961~204061, band bottom 40109~66083
  Combined: peak avg 141142 range 122062~166511; bottom avg 48770 range 42722~53096
  Fusion: price center 133754, price band 128459~138959
"""
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

# ========== 1) Four-relation (real value) -> nominal with annual infl (same as notebook 1st price cell) ==========
BOTTOMS = np.array([150.0, 3200.0, 15500.0])
TOPS = np.array([19800.0, 69000.0])
mult_hist = np.array([TOPS[0] / BOTTOMS[0], TOPS[1] / BOTTOMS[1]])
retr_hist = np.array([BOTTOMS[1] / TOPS[0], BOTTOMS[2] / TOPS[1]])

cycles_m = np.array([2.0, 3.0])
a_m, b_m = np.polyfit(cycles_m, np.log(mult_hist), 1)
mult_pred_trend = float(np.exp(a_m * 4 + b_m))
a_r, b_r = np.polyfit(cycles_m, retr_hist, 1)
retr_pred_trend = float(a_r * 4 + b_r)
a_t, b_t = np.polyfit(cycles_m, np.log(TOPS), 1)
top_trend_4 = float(np.exp(a_t * 4 + b_t))
cycles_b = np.array([2.0, 3.0, 4.0])
a_b, b_b = np.polyfit(cycles_b, np.log(BOTTOMS), 1)
bottom_trend_5 = float(np.exp(a_b * 5 + b_b))

B3 = BOTTOMS[-1]
curr_price_anchor = 120000.0
hard_floor = True
w_mult, w_retr, w_top, w_bottom, w_floor = 0.7, 0.9, 0.6, 0.10, 10.0

mult_grid = np.linspace(4.0, 20.0, 641)
retr_grid = np.linspace(0.20, 0.35, 301)

best = None
records = []
for M4 in mult_grid:
    T4_real = B3 * M4
    if hard_floor and (T4_real < curr_price_anchor):
        continue
    floor_pen = 0.0
    if (not hard_floor) and (T4_real < curr_price_anchor):
        floor_pen = w_floor * (np.log(curr_price_anchor) - np.log(T4_real)) ** 2
    cost_mult = w_mult * (np.log(M4) - np.log(mult_pred_trend)) ** 2
    cost_top = w_top * (np.log(T4_real) - np.log(top_trend_4)) ** 2
    for R4 in retr_grid:
        B5_real = T4_real * R4
        cost_retr = w_retr * (R4 - retr_pred_trend) ** 2
        cost_bot = w_bottom * (np.log(B5_real) - np.log(bottom_trend_5)) ** 2
        cost = cost_mult + cost_retr + cost_top + cost_bot + floor_pen
        records.append((cost, M4, R4, T4_real, B5_real))
        if (best is None) or (cost < best[0]):
            best = (cost, M4, R4, T4_real, B5_real)

records = np.array(records, dtype=float)
min_cost = best[0]
band = records[records[:, 0] <= (min_cost * 1.25)]
T4_real_band = band[:, 3]
B5_real_band = band[:, 4]

infl_annual = 0.03
years_bottom_to_top = 3.0
years_top_to_next_bottom = 1.2

def adj_infl(x, annual_rate, years):
    x = np.asarray(x, dtype=float)
    factor = (1.0 + annual_rate) ** years
    return x * factor

T4_nominal_best = adj_infl(best[3], infl_annual, years_bottom_to_top)
B5_nominal_best = adj_infl(best[4], infl_annual, years_bottom_to_top + years_top_to_next_bottom)
T4_nominal_band = adj_infl(T4_real_band, infl_annual, years_bottom_to_top)
B5_nominal_band = adj_infl(B5_real_band, infl_annual, years_bottom_to_top + years_top_to_next_bottom)

print("=== 1) Four-relation (real) -> nominal (annual infl) ===")
print(f"Best real:    top ~ {best[3]:,.0f}, bottom ~ {best[4]:,.0f}")
print(f"Best nominal: top ~ {T4_nominal_best:,.0f}, bottom ~ {B5_nominal_best:,.0f}")
print(f"Band nominal top:    {np.min(T4_nominal_band):,.0f} ~ {np.max(T4_nominal_band):,.0f}")
print(f"Band nominal bottom: {np.min(B5_nominal_band):,.0f} ~ {np.max(B5_nominal_band):,.0f}")

exp_nominal_top = 133381
exp_nominal_bot = 48369
exp_band_top_lo, exp_band_top_hi = 131264, 201977
exp_band_bot_lo, exp_band_bot_hi = 27201, 73244

def check(name, got, exp, tol=0.02):
    if abs(got - exp) / max(exp, 1) <= tol:
        print(f"  [OK] {name}: got {got:,.0f} vs expected ~{exp:,.0f}")
        return True
    print(f"  [MISMATCH] {name}: got {got:,.0f} vs expected ~{exp:,.0f}")
    return False

ok1 = (
    check("nominal top best", T4_nominal_best, exp_nominal_top)
    and check("nominal bottom best", B5_nominal_best, exp_nominal_bot)
    and check("band top lo", np.min(T4_nominal_band), exp_band_top_lo, 0.05)
    and check("band top hi", np.max(T4_nominal_band), exp_band_top_hi, 0.05)
    and check("band bottom lo", np.min(B5_nominal_band), exp_band_bot_lo, 0.05)
    and check("band bottom hi", np.max(B5_nominal_band), exp_band_bot_hi, 0.05)
)

# ========== 2) Four-relation (Real, full inflation) -> nominal ==========
BOTTOMS_NOMINAL = np.array([150.0, 3200.0, 15500.0])
BOTTOM_YEARS = np.array([2015, 2018, 2022])
TOPS_NOMINAL = np.array([19800.0, 69000.0])
TOP_YEARS = np.array([2017, 2021])
CURR_PRICE_ANCHOR_NOMINAL = 120_000.0
CURR_ANCHOR_YEAR = 2025

ANNUAL_INFL: Dict[int, float] = {
    2013: 0.015, 2014: 0.016, 2015: 0.001, 2016: 0.013,
    2017: 0.021, 2018: 0.024, 2019: 0.018, 2020: 0.012,
    2021: 0.047, 2022: 0.080, 2023: 0.041, 2024: 0.029,
    2025: 0.027, 2026: 0.025, 2027: 0.024, 2028: 0.022, 2029: 0.020,
}

BASE_REAL_YEAR = 2025
PRED_TOP_YEAR = 2025
PRED_NEXT_BOTTOM_YEAR = 2027

def build_price_level_from_rates(annual_rates: Dict[int, float], base_year: int) -> Dict[int, float]:
    years = sorted(annual_rates.keys())
    if base_year not in years:
        years = sorted(set(years) | {base_year})
    start = min(years)
    idx = {start: 1.0}
    for y in range(start + 1, max(years) + 1):
        r = annual_rates.get(y, 0.0)
        idx[y] = idx[y - 1] * (1.0 + r)
    base = idx[base_year]
    return {y: idx[y] / base for y in idx}

def to_real(nominal_price: np.ndarray, years: np.ndarray, level: Dict[int, float], base_year: int) -> np.ndarray:
    return np.array([float(p) / level[y] for p, y in zip(nominal_price, years)], dtype=float)

def real_to_nominal(real_price: float, from_year: int, to_year: int, level: Dict[int, float]) -> float:
    return float(real_price) * level[to_year]

LEVEL = build_price_level_from_rates(ANNUAL_INFL, BASE_REAL_YEAR)
BOTTOMS_REAL = to_real(BOTTOMS_NOMINAL, BOTTOM_YEARS, LEVEL, BASE_REAL_YEAR)
TOPS_REAL = to_real(TOPS_NOMINAL, TOP_YEARS, LEVEL, BASE_REAL_YEAR)
CURR_ANCHOR_REAL = to_real(np.array([CURR_PRICE_ANCHOR_NOMINAL]), np.array([CURR_ANCHOR_YEAR]), LEVEL, BASE_REAL_YEAR)[0]

mult_hist_r = np.array([TOPS_REAL[0] / BOTTOMS_REAL[0], TOPS_REAL[1] / BOTTOMS_REAL[1]])
retr_hist_r = np.array([BOTTOMS_REAL[1] / TOPS_REAL[0], BOTTOMS_REAL[2] / TOPS_REAL[1]])
a_m, b_m = np.polyfit(cycles_m, np.log(mult_hist_r), 1)
mult_pred_trend_r = float(np.exp(a_m * 4 + b_m))
a_r, b_r = np.polyfit(cycles_m, retr_hist_r, 1)
retr_pred_trend_r = float(a_r * 4 + b_r)
a_t, b_t = np.polyfit(cycles_m, np.log(TOPS_REAL), 1)
top_trend_4_real = float(np.exp(a_t * 4 + b_t))
a_b, b_b = np.polyfit(cycles_b, np.log(BOTTOMS_REAL), 1)
bottom_trend_5_real = float(np.exp(a_b * 5 + b_b))

B3_real = BOTTOMS_REAL[-1]
prev_top_real = TOPS_REAL[-1]
w_mult, w_retr, w_top, w_bottom, w_floor = 0.6, 1.0, 0.8, 0.08, 12.0
M_min = max(4.0, CURR_ANCHOR_REAL / B3_real, prev_top_real / B3_real)
mult_grid_r = np.linspace(M_min, 20.0, int((20.0 - M_min) / 0.025) + 1)
retr_grid_r = np.linspace(0.26, 0.34, 161)

best2 = None
records2 = []
for M4 in mult_grid_r:
    T4_real = B3_real * M4
    if hard_floor and (T4_real < CURR_ANCHOR_REAL or T4_real < prev_top_real):
        continue
    floor_pen = 0.0
    if (not hard_floor) and (T4_real < CURR_ANCHOR_REAL):
        floor_pen += w_floor * (np.log(CURR_ANCHOR_REAL) - np.log(T4_real)) ** 2
    if (not hard_floor) and (T4_real < prev_top_real):
        floor_pen += w_floor * (np.log(prev_top_real) - np.log(T4_real)) ** 2
    cost_mult = w_mult * (np.log(M4) - np.log(mult_pred_trend_r)) ** 2
    cost_top = w_top * (np.log(T4_real) - np.log(top_trend_4_real)) ** 2
    for R4 in retr_grid_r:
        B5_real = T4_real * R4
        cost_retr = w_retr * (R4 - retr_pred_trend_r) ** 2
        cost_bot = w_bottom * (np.log(B5_real) - np.log(bottom_trend_5_real)) ** 2
        cost = cost_mult + cost_retr + cost_top + cost_bot + floor_pen
        records2.append((cost, M4, R4, T4_real, B5_real))
        if (best2 is None) or (cost < best2[0]):
            best2 = (cost, M4, R4, T4_real, B5_real)

records2 = np.array(records2, dtype=float)
min_cost2 = best2[0]
band2 = records2[records2[:, 0] <= (min_cost2 * 1.25)]
T4_real_band2 = band2[:, 3]
B5_real_band2 = band2[:, 4]

T4_nom_best2 = real_to_nominal(best2[3], BASE_REAL_YEAR, PRED_TOP_YEAR, LEVEL)
B5_nom_best2 = real_to_nominal(best2[4], BASE_REAL_YEAR, PRED_NEXT_BOTTOM_YEAR, LEVEL)
T4_nom_band2 = np.array([real_to_nominal(x, BASE_REAL_YEAR, PRED_TOP_YEAR, LEVEL) for x in T4_real_band2])
B5_nom_band2 = np.array([real_to_nominal(x, BASE_REAL_YEAR, PRED_NEXT_BOTTOM_YEAR, LEVEL) for x in B5_real_band2])

def band_str(vals: np.ndarray, qlo=0.10, qhi=0.90) -> Tuple[float, float]:
    return float(np.quantile(vals, qlo)), float(np.quantile(vals, qhi))

t4_lo, t4_hi = band_str(T4_nom_band2, 0.10, 0.90)
b5_lo, b5_hi = band_str(B5_nom_band2, 0.10, 0.90)

print("\n=== 2) Four-relation (Real, full infl) -> nominal ===")
print(f"Best Real:    top ~ {best2[3]:,.0f}, bottom ~ {best2[4]:,.0f}")
print(f"Best Nominal: top ~ {T4_nom_best2:,.0f}, bottom ~ {B5_nom_best2:,.0f}")
print(f"Band (10~90p) top:    {t4_lo:,.0f} ~ {t4_hi:,.0f}")
print(f"Band (10~90p) bottom: {b5_lo:,.0f} ~ {b5_hi:,.0f}")

exp_real_top, exp_real_bot = 142615, 48489
exp_nom_top2, exp_nom_bot2 = 142615, 50894
exp_t4_lo, exp_t4_hi = 128961, 204061
exp_b5_lo, exp_b5_hi = 40109, 66083

ok2 = (
    check("Real top best", best2[3], exp_real_top)
    and check("Real bottom best", best2[4], exp_real_bot)
    and check("Nominal top best", T4_nom_best2, exp_nom_top2)
    and check("Nominal bottom best", B5_nom_best2, exp_nom_bot2)
    and check("band top 10p", t4_lo, exp_t4_lo, 0.03)
    and check("band top 90p", t4_hi, exp_t4_hi, 0.03)
    and check("band bottom 10p", b5_lo, exp_b5_lo, 0.03)
    and check("band bottom 90p", b5_hi, exp_b5_hi, 0.03)
)

# ========== 3) Combined (fixed model outputs) ==========
time_points = [
    ("Baseline", "2025-09-28", 1.0),
    ("Regression", "2025-11-04", 1.5),
    ("Top2Top", "2025-11-01", 1.2),
    ("Combined", "2025-10-16", 1.0),
]
price_ranges = {
    "ModelA": {"peak": (122062, 122062), "bottom": (42722, 42722)},
    "ModelB": {"peak": (133381, 133381), "bottom": (48369, 48369)},
    "ModelC": {"peak": (142615, 142615), "bottom": (50894, 50894)},
    "ModelD": {"peak": (128961, 204061), "bottom": (40109, 66083)},
}
dates = [datetime.strptime(d[1], "%Y-%m-%d").toordinal() for d in time_points]
weights = [d[2] for d in time_points]
peaks = [(v["peak"][0] + v["peak"][1]) / 2 for v in price_ranges.values()]
bottoms = [(v["bottom"][0] + v["bottom"][1]) / 2 for v in price_ranges.values()]
peak_avg, peak_min, peak_max = np.mean(peaks), np.min(peaks), np.max(peaks)
bottom_avg, bottom_min, bottom_max = np.mean(bottoms), np.min(bottoms), np.max(bottoms)

print("\n=== 3) Combined (from fixed model outputs) ===")
print(f"Peak: avg={peak_avg:,.0f}, range={peak_min:,.0f}~{peak_max:,.0f}")
print(f"Bottom: avg={bottom_avg:,.0f}, range={bottom_min:,.0f}~{bottom_max:,.0f}")

ok3 = (
    check("peak avg", peak_avg, 141142)
    and check("peak min", peak_min, 122062)
    and check("peak max", peak_max, 166511)
    and check("bottom avg", bottom_avg, 48770)
    and check("bottom min", bottom_min, 42722)
    and check("bottom max", bottom_max, 53096)
)

print("\n" + ("[OK] All price outputs match post." if (ok1 and ok2 and ok3) else "[CHECK] Some price values differ (see above)."))
