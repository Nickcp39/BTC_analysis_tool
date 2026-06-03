# BTC Bottom Multi-Model Conclusion

Generated: 2026-06-01

## Weighted conclusion
- Time window: 2026-10-01 -> 2026-11-05 (average center: 2026-10-22)
- Wide time window: 2026-09-16 -> 2026-12-04
- Price bottom: average=$54,283, range=$46,874~$60,547
- Wide price range: $42,445~$70,343

## Equal-weight check
- Time window: 2026-10-01 -> 2026-11-03 (average center: 2026-10-20)
- Price bottom: average=$53,200, range=$46,072~$59,525

## Model inputs
                        model  weight                                                                                                           note center_date    date_lo    date_hi date_wide_lo date_wide_hi  center_price     price_lo     price_hi  price_wide_lo  price_wide_hi
       M1_time_price_blend_v1    0.75                                          first independent bottom model: timing + old core price + ratio trend  2026-10-30 2026-09-30 2026-11-22   2026-09-11   2027-01-03  43611.422191 38607.375635 48111.192298   35224.130019   52829.021949
           M2_sqrt_log_vol_v1    1.20                                                                  strict log drawdown sqrt volatility annealing  2026-10-22 2026-09-27 2026-11-07   2026-09-10   2026-12-08  60385.182422 53670.148292 66109.723669   49007.240533   72811.612416
    M3_bottom_ratio_fusion_v1    1.30                               closest to old peak conclusion style: ratio/log-ratio + sqrt-vol + timing fusion  2026-10-30 2026-10-03 2026-11-18   2026-09-16   2026-12-21  57867.528282 47680.154870 65154.682846   42202.417700   77418.050471
  M4_old_core_bottom_artifact    0.85 prior core artifact bottom result: avg 48770, range 42722~53096; time inferred from real peak + historical p2b  2026-10-10 2026-10-03 2026-10-16   2026-09-28   2026-10-29  48770.000000 42722.000000 53096.000000   40109.000000   66083.000000
M5_peak_to_bottom_timing_only    0.65                      actual peak + historical 363/376 day bottom timing; price anchored to ratio-fusion median  2026-10-10 2026-10-03 2026-10-16   2026-09-16   2026-11-07  55368.000000 47680.000000 65155.000000   42202.000000   77418.000000

## Readout
The weighted version gives more influence to the ratio/log-ratio fusion and sqrt-log volatility models, because those match the original peak-model philosophy. The old core bottom artifact remains as a soft anchor rather than the final answer.