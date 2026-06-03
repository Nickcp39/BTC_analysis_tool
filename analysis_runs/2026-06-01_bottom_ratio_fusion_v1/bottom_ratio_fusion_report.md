# BTC Bottom Ratio Fusion V1

Generated: 2026-06-01
Data: 2014-12-01 to 2026-05-31
Latest BTC: $73,688.35 on 2026-05-31

## Why this version
This is the bottom-side analogue of the old peak product: use ratio/log-ratio models for price, and independent timing clocks for date.
The key correction is to apply volatility decay in log space with a square root: log drawdown * sqrt(vol_2025 / vol_history).

## Current anchor
- 2025 actual peak: 2025-10-05 at $124,720.09
- Current: day 238 after peak, drawdown -40.92%

## Bottom forecast
- Center date: 2026-10-30
- Date band 25-75%: 2026-10-03 to 2026-11-18
- Date band 10-90%: 2026-09-16 to 2026-12-21
- Center price: $57,868
- Price band 25-75%: $47,680 to $65,155
- Price band 10-90%: $42,202 to $77,418

## Historical observations
cycle  peak_date  peak_price bottom_date  bottom_price  days_peak_to_bottom  bottom_to_peak_ratio  log_bottom_to_peak_ratio
 2017 2017-12-17    19378.99  2018-12-15       3183.00                  363              0.164250                 -1.806365
 2021 2021-11-10    64896.86  2022-11-21      15755.75                  376              0.242781                 -1.415594

## Sqrt-log replay
cycle  raw_log_ratio  sqrt_vol_scale  annealed_log_ratio  annealed_ratio  annealed_price_from_2025_peak bottom_date_by_replay
 2017      -1.806365        0.333333           -0.602122        0.547648                   68302.759334            2026-10-03
 2021      -1.415594        0.577350           -0.817293        0.441625                   55079.546968            2026-10-16

## Price models
                       name  bottom_price  sigma_pct  weight                                                                           note
bottom_peak_log_ratio_trend  44757.076343       0.14    1.15            log(bottom/peak ratio) trend from 2017->2021 extrapolated one cycle
 bottom_to_bottom_log_trend  77990.467503       0.18    0.85            log(bottom price) trend from 2018->2022 extrapolated to next bottom
 sqrt_log_vol_annealed_mean  61691.153151       0.15    1.20 historical log drawdown compressed by sqrt volatility scale, mean of 2017/2021
 sqrt_log_vol_annealed_2021  55079.546968       0.13    1.10               2021-only sqrt-log replay; best shape match in prior diagnostics
old_core_bottom_soft_anchor  49399.678571       0.16    0.75                   prior core artifact bottom ensemble, retained as soft anchor

## Time models
                            name bottom_date  sigma_days  weight                                                                        note
             peak_to_bottom_mean  2026-10-10    20.50000    1.05                                       mean of 2017/2021 peak-to-bottom days
       peak_to_bottom_regression  2026-10-29    30.00000    0.85 linear extension of peak-to-bottom days, analogous to top timing regression
   bottom_to_next_halving_stable  2026-10-23    33.02135    1.00                          bottom->next halving has been stable near 524 days
bottom_to_next_halving_regressed  2026-12-30    45.02135    0.45                        late tail if next halving interval keeps lengthening