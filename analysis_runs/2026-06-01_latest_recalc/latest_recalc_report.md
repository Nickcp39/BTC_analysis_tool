# BTC latest recalculation report

- Generated: 2026-06-01
- Data range: 2014-12-01 ~ 2026-05-31 (4200 daily rows)
- Latest BTC price: $73,688.35
- 2025 assumed anchor: 2025-08-15 @ $117,455.68
- 2025 actual high since halving: 2025-10-05 @ $124,720.09

## Bottom timing by historical peak-to-bottom days
            anchor_name anchor_date  anchor_price latest_date  latest_price  current_day_after_anchor  current_drawdown_pct history_cycle  history_days_to_bottom projected_bottom_date_by_time  days_from_latest_to_projected_bottom  history_bottom_drawdown_pct_raw
assumed_peak_2025_08_15  2025-08-15     117455.68  2026-05-31      73688.35                       289            -37.262847          2017                     363                    2026-08-13                                    74                       -83.574995
assumed_peak_2025_08_15  2025-08-15     117455.68  2026-05-31      73688.35                       289            -37.262847          2021                     376                    2026-08-26                                    87                       -75.721861
 actual_high_2025_10_05  2025-10-05     124720.09  2026-05-31      73688.35                       238            -40.917017          2017                     363                    2026-10-03                                   125                       -83.574995
 actual_high_2025_10_05  2025-10-05     124720.09  2026-05-31      73688.35                       238            -40.917017          2021                     376                    2026-10-16                                   138                       -75.721861

## Fit metrics vs observed 2025 path
           anchor history_cycle                       variant  days_compared  scale_used     corr  rmse_pct_points  mae_pct_points  pred_latest_dd_pct  obs_latest_dd_pct
2025_assumed_0815          2017                           raw            290    1.000000 0.748385        33.106189       31.507476          -66.466467         -37.262847
2025_assumed_0815          2017            std_scaled_to_2025            290    1.061301 0.748385        36.417312       34.890624          -70.540908         -37.262847
2025_assumed_0815          2017 manual_pre_scale_x_post_gamma            290    0.216667 0.748385        17.225022       14.468796          -14.401068         -37.262847
2025_assumed_0815          2021                           raw            290    1.000000 0.814348        21.968155       19.493680          -68.827182         -37.262847
2025_assumed_0815          2021            std_scaled_to_2025            290    0.827363 0.814348        14.942933       12.361783          -56.945090         -37.262847
2025_assumed_0815          2021 manual_pre_scale_x_post_gamma            290    0.375278 0.814348        12.550978       10.952882          -25.829305         -37.262847

Best RMSE fit: 2021 / manual_pre_scale_x_post_gamma, r=0.814, RMSE=12.55 percentage points.