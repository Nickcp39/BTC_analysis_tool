# Green Anchor Probe

This is a first-pass, falsifiable translation of the hand-marked green dots:
multi-scale persistent turning points on log BTC daily price.

Lower `dtw_phase_distance` means the anchor-time layout is more similar after normalizing each cycle's anchor span.

## Cycle Similarity
| cycle_a | cycle_b | anchors_a | anchors_b | dtw_phase_distance |
| --- | --- | --- | --- | --- |
| 2017 | 2021 | 18 | 16 | 0.0122 |
| 2017 | 2025 | 18 | 8 | 0.0278 |
| 2021 | 2025 | 16 | 8 | 0.0253 |

## Anchor Counts
| cycle | anchors |
| --- | --- |
| 2017 | 18 |
| 2021 | 16 |
| 2025 | 8 |