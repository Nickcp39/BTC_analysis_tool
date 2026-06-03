# BTC Bottom Model Sqrt-Log V1

Generated: 2026-06-01
Data: 2014-12-01 to 2026-05-31
Latest BTC: $73,688.35 on 2026-05-31

## Core formula
`annealed_log_drawdown = log(historical_bottom / historical_peak) * sqrt(vol_2025 / vol_history)`

This is the user's corrected core: volatility decays by cycle, and the historical amplitude is compressed in log space with a square-root factor.

## Peak anchor
- Actual 2025 cycle high since 2024 halving: 2025-10-05 at $124,720.09
- Current position: day 238 after peak, drawdown -40.92%, log drawdown -0.5262

## New bottom forecast
- Center date: 2026-10-22
- Date band 10-90%: 2026-09-10 to 2026-12-08
- Date band 25-75%: 2026-09-27 to 2026-11-07
- Center price: $60,385
- Price band 10-90%: $49,007 to $72,812
- Price band 25-75%: $53,670 to $66,110

## Historical sqrt-log components
cycle  peak_date  peak_price bottom_date  bottom_price  days_peak_to_bottom  raw_bottom_ratio  raw_log_drawdown  sqrt_vol_scale_to_2025  annealed_log_drawdown  annealed_2025_bottom_price
 2017 2017-12-17    19378.99  2018-12-15       3183.00                  363          0.164250         -1.806365                0.333333              -0.602122                68302.759334
 2021 2021-11-10    64896.86  2022-11-21      15755.75                  376          0.242781         -1.415594                0.577350              -0.817293                55079.546968

## Model components
                              model bottom_date  bottom_price  sigma_days  sigma_price_pct  weight                                                                             note
               sqrt_log_replay_2017  2026-10-03      68302.76    24.00000             0.16    0.70 log(bottom/peak) from 2017 multiplied by sqrt(1/9), then replayed from 2025 peak
               sqrt_log_replay_2021  2026-10-16      55079.55    24.00000             0.14    1.25 log(bottom/peak) from 2021 multiplied by sqrt(1/3), then replayed from 2025 peak
              sqrt_log_blended_time  2026-10-10      61691.15    24.50000             0.13    1.00           mean of sqrt-log replay prices with mean 2017/2021 peak-to-bottom time
bottom_to_next_halving_conservative  2026-10-23      59223.51    37.02135             0.15    0.80              stable bottom-to-next-halving interval, using sqrt-log price anchor
   bottom_to_next_halving_regressed  2026-12-30      56755.86    45.02135             0.17    0.45            slower next halving regression; keeps late-date tail but lower weight