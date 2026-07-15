# LUNA16 Shepard Synthetic Surface QC

This report evaluates the synthetic 2D images generated with `surface_method=shepard`:

- Ground truth root: `data/luna16_saliency_synthetic_gt_shepard`
- Detector root: `data/luna16_saliency_synthetic_detector_top5_minprob0.5_shepard`

For Shepard surfaces, each nodule/detection region is represented as one disk `(x, y, z, radius)`. Pixels inside a disk are sampled from that disk's exact `z`; pixels outside all disks are inverse-distance blended from disk edges.

Lower `qc_badness` is better. The score is a triage heuristic combining foreground coverage, foreground contrast, z-surface range, edge variation, and surface gradient roughness.

- Ground-truth generated cases analyzed: **888**
- Detector top-5 generated cases analyzed: **888**
- Paired cases compared by `seriesuid`: **888**
- Detector lower/better `qc_badness`: **180 / 888 (20.3%)**
- Detector equal-or-higher/worse `qc_badness`: **708 / 888 (79.7%)**

## Source Summary

| source        |   n |   qc_badness_mean |   qc_badness_median |   qc_badness_p95 |   foreground_fraction_mean |   z_range_mean |   grad_p99_mean |
|:--------------|----:|------------------:|--------------------:|-----------------:|---------------------------:|---------------:|----------------:|
| ground_truth  | 888 |             0.080 |               0.019 |            0.314 |                      0.543 |         32.414 |           0.657 |
| detector_top5 | 888 |             0.154 |               0.131 |            0.433 |                      0.522 |         61.040 |           1.189 |

## QC Buckets

`good_or_low_review` is `qc_badness <= 0.10`, `moderate_review` is `0.10 < qc_badness <= 0.30`, and `high_review` is `qc_badness > 0.30`.

| source        | qc_bucket          |   n |
|:--------------|:-------------------|----:|
| ground_truth  | good_or_low_review | 613 |
| ground_truth  | moderate_review    | 220 |
| ground_truth  | high_review        |  55 |
| detector_top5 | good_or_low_review | 386 |
| detector_top5 | moderate_review    | 341 |
| detector_top5 | high_review        | 161 |

## Shepard-Specific Interpretation

The ground-truth Shepard set is cleaner by this heuristic: its mean `qc_badness` is `0.080` versus `0.154` for detector Shepard, and `613 / 888` GT cases fall in `good_or_low_review` versus `386 / 888` detector cases.

The main detector failure mode is geometric rather than image-intensity based. Detector Shepard surfaces have higher mean `z_range` (`61.040` vs `32.414`) and higher mean `grad_p99` (`1.189` vs `0.657`). In practical terms, detector candidates often span more cranio-caudal depth than the GT/pseudo regions, so the inverse-distance surface has to transition across a larger `z` interval.

The `control_labels` table below should be read as a saved-output proxy, not as the actual Shepard interpolation nodes. Shepard uses one disk per region; the saved control labels are retained for compatibility with the RBF workflow and QC tooling.

| source        |   control_labels |   count |   mean |   median |
|:--------------|-----------------:|--------:|-------:|---------:|
| ground_truth  |                5 |     456 |  0.004 |    0.000 |
| ground_truth  |               10 |     240 |  0.127 |    0.098 |
| ground_truth  |               15 |      96 |  0.169 |    0.161 |
| ground_truth  |               20 |      34 |  0.209 |    0.218 |
| ground_truth  |               25 |      30 |  0.252 |    0.270 |
| ground_truth  |               30 |      13 |  0.168 |    0.174 |
| ground_truth  |               35 |       7 |  0.346 |    0.316 |
| ground_truth  |               40 |       4 |  0.217 |    0.246 |
| ground_truth  |               45 |       7 |  0.304 |    0.271 |
| ground_truth  |               60 |       1 |  0.371 |    0.371 |
| detector_top5 |                5 |     263 |  0.006 |    0.000 |
| detector_top5 |               10 |     199 |  0.130 |    0.112 |
| detector_top5 |               15 |     137 |  0.226 |    0.205 |
| detector_top5 |               20 |      79 |  0.263 |    0.238 |
| detector_top5 |               25 |     202 |  0.272 |    0.251 |
| detector_top5 |               30 |       3 |  0.335 |    0.298 |
| detector_top5 |               35 |       1 |  0.316 |    0.316 |
| detector_top5 |               40 |       1 |  0.322 |    0.322 |
| detector_top5 |               45 |       3 |  0.263 |    0.255 |

## Paired Delta Summary

Delta is detector minus ground truth; negative `qc_badness` means detector is better by this heuristic.

| metric                                      |   mean |    std |      5% |    25% |   50% |    75% |     95% |
|:--------------------------------------------|-------:|-------:|--------:|-------:|------:|-------:|--------:|
| foreground_fraction_delta_detector_minus_gt | -0.021 |  0.126 |  -0.306 | -0.030 | 0.000 |  0.015 |   0.150 |
| foreground_std_delta_detector_minus_gt      |  0.002 |  0.010 |  -0.011 | -0.003 | 0.000 |  0.004 |   0.020 |
| z_range_delta_detector_minus_gt             | 28.626 | 64.433 | -69.950 |  0.000 | 0.744 | 56.405 | 149.720 |
| z_std_delta_detector_minus_gt               |  4.411 | 10.266 | -10.849 |  0.000 | 0.651 |  9.723 |  24.069 |
| grad_p99_delta_detector_minus_gt            |  0.531 |  1.403 |  -1.464 |  0.000 | 0.022 |  1.099 |   3.184 |
| grad_max_delta_detector_minus_gt            |  1.708 |  7.742 |  -4.400 |  0.000 | 0.079 |  2.208 |  10.628 |
| qc_badness_delta_detector_minus_gt          |  0.074 |  0.156 |  -0.158 |  0.000 | 0.013 |  0.162 |   0.364 |

## Detector Most Worse Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.159996104466052855396410079250 |           0.000 |                 0.580 |                                0.580 |        0.000 |            251.661 |         0.000 |               3.605 |                    0.697 |                          0.658 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.373433682859788429397781158572 |           0.000 |                 0.577 |                                0.577 |        0.000 |            203.958 |         0.000 |               6.024 |                    0.652 |                          0.405 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.428038562098395445838061018440 |           0.000 |                 0.560 |                                0.560 |        0.000 |            234.727 |         0.000 |               4.823 |                    0.622 |                          0.609 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.168605638657404145360275453085 |           0.000 |                 0.560 |                                0.560 |        0.000 |            223.850 |         0.000 |               5.926 |                    0.684 |                          0.725 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.246225645401227472829175288633 |           0.000 |                 0.552 |                                0.552 |        0.000 |            243.338 |         0.000 |               4.274 |                    0.629 |                          0.664 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.302403227435841351528721627052 |           0.000 |                 0.549 |                                0.549 |        0.000 |            212.834 |         0.000 |               4.067 |                    0.526 |                          0.446 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.168737928729363683423228050295 |           0.000 |                 0.545 |                                0.545 |        0.000 |            220.632 |         0.000 |               5.445 |                    0.597 |                          0.568 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.627998298349675613581885874395 |           0.000 |                 0.534 |                                0.534 |        0.000 |            215.230 |         0.000 |               5.483 |                    0.623 |                          0.608 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.253317247142837717905329340520 |           0.090 |                 0.620 |                                0.531 |       31.000 |            179.319 |         1.189 |               8.975 |                    0.597 |                          0.586 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.341557859428950960906150406596 |           0.000 |                 0.528 |                                0.528 |        0.000 |            235.995 |         0.000 |               3.311 |                    0.537 |                          0.522 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.138813197521718693188313387015 |           0.000 |                 0.524 |                                0.524 |        0.000 |            201.137 |         0.000 |               5.845 |                    0.530 |                          0.612 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.219254430927834326484477690403 |           0.059 |                 0.574 |                                0.515 |       26.000 |            226.250 |         0.364 |               4.441 |                    0.640 |                          0.360 |

## Detector Most Better Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.140253591510022414496468423138 |           0.482 |                 0.001 |                               -0.481 |      190.000 |              0.200 |         5.152 |               0.005 |                    0.574 |                          0.550 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.264090899378396711987322794314 |           0.468 |                 0.000 |                               -0.468 |      154.000 |              0.000 |         6.426 |               0.000 |                    0.570 |                          0.369 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.139444426690868429919252698606 |           0.331 |                 0.000 |                               -0.331 |      120.000 |              0.000 |         3.758 |               0.000 |                    0.618 |                          0.613 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.994459772950022352718462251777 |           0.316 |                 0.000 |                               -0.316 |      141.000 |              0.000 |         1.834 |               0.000 |                    0.638 |                          0.626 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.249404938669582150398726875826 |           0.299 |                 0.000 |                               -0.299 |      111.000 |              0.000 |         3.021 |               0.000 |                    0.544 |                          0.609 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.333319057944372470283038483725 |           0.293 |                 0.000 |                               -0.293 |      120.000 |              0.000 |         2.406 |               0.000 |                    0.515 |                          0.588 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.306948744223170422945185006551 |           0.290 |                 0.000 |                               -0.290 |      104.000 |              0.000 |         3.553 |               0.000 |                    0.577 |                          0.536 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.596908385953413160131451426904 |           0.278 |                 0.000 |                               -0.278 |      104.000 |              0.000 |         3.261 |               0.000 |                    0.559 |                          0.566 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.129982010889624423230394257528 |           0.269 |                 0.000 |                               -0.269 |      115.000 |              0.000 |         1.900 |               0.000 |                    0.583 |                          0.430 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.286061375572911414226912429210 |           0.269 |                 0.000 |                               -0.269 |       88.000 |              0.000 |         3.558 |               0.000 |                    0.637 |                          0.494 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.294120933998772507043263238704 |           0.289 |                 0.029 |                               -0.260 |      137.000 |             12.000 |         1.339 |               0.241 |                    0.471 |                          0.435 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.592821488053137951302246128864 |           0.258 |                 0.002 |                               -0.256 |       98.000 |              0.800 |         2.956 |               0.022 |                    0.530 |                          0.544 |

## Output Files

- `ground_truth_surface_qc_metrics.csv`
- `detector_top5_surface_qc_metrics.csv`
- `paired_surface_qc_comparison.csv`
