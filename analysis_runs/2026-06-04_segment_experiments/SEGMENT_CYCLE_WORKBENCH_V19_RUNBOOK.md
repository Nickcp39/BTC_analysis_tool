# Segment Cycle Workbench v19

Goal: manually compare BTC cycle structure around peak/bottom anchors across fixed windows, then save the ratio settings and export the visual evidence.

## Workflow
0. Rebuild with fresh data: `python segment_cycle_workbench_v19.py` (reads data/btc_merged_daily.csv).
1. Start `python segment_sample_save_server.py` and open http://127.0.0.1:8765/ (the page loads saved samples from the project file; Save writes segment_cycle_samples_v19_saved.json/.csv plus a backup).
2. Start with `Bottom +/- 1y: 2021 -> 2025` (bottoms 2022-11-21 / 2026-06-30). Yellow rings = user-confirmed points and pairs from analysis_runs/2026-11-13_manual_point_compare/.
3. Adjust height, time scale, and horizontal shift.
4. Save Sample after every visually meaningful fit.
5. Export Sample Library JSON regularly.
6. Export PNG/SVG for the fits you want to discuss.

## First backlog
- Peak pre 1y: 2021 -> 2025
- Peak +/- 6m: 2021 -> 2025
- Peak post 1y: 2021 -> 2025
- Bottom +/- 6m: 2021 -> 2025
- Repeat the same windows for 2017 -> 2021, then compare ratios.

Notes: local data starts on 2014-12-01, so 2013 peak windows are placeholders until older data is added.
