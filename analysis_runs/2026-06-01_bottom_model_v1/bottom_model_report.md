# BTC Bottom Model V1

Generated: 2026-06-01
Data: 2014-12-01 to 2026-05-31
Latest BTC: $73,688.35 on 2026-05-31

## Peak anchor
- Actual 2025 cycle high since 2024 halving: 2025-10-05 at $124,720.09
- Current position: day 238 after peak, drawdown -40.92%

## New bottom forecast
- Center date: 2026-10-30
- Date band 10-90%: 2026-09-11 to 2027-01-03
- Date band 25-75%: 2026-09-30 to 2026-11-22
- Center price: $43,611
- Price band 10-90%: $35,224 to $52,829
- Price band 25-75%: $38,607 to $48,111

## Model components
                              model bottom_date  bottom_price  sigma_days  sigma_price_pct  weight                                                                                                       note
           peak_to_bottom_time_mean  2026-10-10      40074.15    20.50000             0.14    1.25                              2017/2021 peak-to-bottom days, price uses contracting bottom/peak ratio trend
   bottom_to_next_halving_regressed  2026-12-30      44270.64    43.02135             0.15    0.75 estimate next halving interval by regression; contributes late timing, price uses trend/core blended ratio
bottom_to_next_halving_conservative  2026-10-23      40074.15    33.02135             0.14    0.90                                                 assume next halving interval stays near 2020-2024 interval
       old_core_bottom_price_anchor  2026-10-16      49399.68    42.00000             0.12    1.00             reuse prior core product bottom ensemble as price anchor, updated with actual 2025 peak timing

## Interpretation
This model treats 2025-10-05 as the real peak. It combines peak-to-bottom timing, bottom-to-next-halving timing, historical bottom/peak ratio decay, and the old core bottom-price ensemble.
The result should be read as a probabilistic bottom zone, not a single exact day.