# LUNA16 Detector vs Ground-Truth Synthetic Surface QC

Lower `qc_badness` is better. The score is a triage heuristic combining foreground coverage, foreground contrast, z-surface range, edge variation, and surface gradient roughness.

- Ground-truth generated cases analyzed: **888**
- Detector top-5 generated cases analyzed: **888**
- Paired cases compared by `seriesuid`: **888**
- Detector lower/better `qc_badness`: **36 / 888 (4.1%)**
- Detector equal-or-higher/worse `qc_badness`: **852 / 888 (95.9%)**

## Source Summary

| source        |   n |   qc_badness_mean |   qc_badness_median |   qc_badness_p95 |   foreground_fraction_mean |   z_range_mean |   grad_p99_mean |
|:--------------|----:|------------------:|--------------------:|-----------------:|---------------------------:|---------------:|----------------:|
| ground_truth  | 888 |             0.080 |               0.019 |            0.314 |                      0.543 |         32.414 |           0.657 |
| detector_top5 | 888 |             0.344 |               0.346 |            0.545 |                      0.561 |        136.965 |           3.161 |

## QC Buckets

`good_or_low_review` is `qc_badness <= 0.10`, `moderate_review` is `0.10 < qc_badness <= 0.30`, and `high_review` is `qc_badness > 0.30`.

| source        | qc_bucket          |   n |
|:--------------|:-------------------|----:|
| ground_truth  | good_or_low_review | 613 |
| ground_truth  | moderate_review    | 220 |
| ground_truth  | high_review        |  55 |
| detector_top5 | good_or_low_review |  20 |
| detector_top5 | moderate_review    | 300 |
| detector_top5 | high_review        | 568 |

## Likely Explanation

The detector-top5 run uses five detector candidates for every scan. In these saved outputs, each nodule/detection region contributes five labelled control points, so detector-top5 has `25` control labels for every case. Ground-truth generation is much less constrained: most cases have one or two nodule regions, corresponding to `5` or `10` control labels.

This matches the QC failure mode: more control regions force the interpolated surface to satisfy points across a larger cranio-caudal spread, increasing `z_range` and `grad_p99`. The foreground image statistics stay similar, but the detector surfaces are geometrically rougher.

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
| detector_top5 |                5 |       6 |  0.000 |    0.000 |
| detector_top5 |               10 |      14 |  0.107 |    0.097 |
| detector_top5 |               15 |       7 |  0.202 |    0.235 |
| detector_top5 |               20 |      11 |  0.215 |    0.245 |
| detector_top5 |               25 |      14 |  0.244 |    0.242 |
| detector_top5 |               30 |      11 |  0.328 |    0.309 |
| detector_top5 |               35 |     825 |  0.356 |    0.357 |

## Paired Delta Summary

Delta is detector minus ground truth; negative `qc_badness` means detector is better by this heuristic.

| metric                                      |    mean |    std |     5% |    25% |     50% |     75% |     95% |
|:--------------------------------------------|--------:|-------:|-------:|-------:|--------:|--------:|--------:|
| foreground_fraction_delta_detector_minus_gt |   0.018 |  0.115 | -0.203 | -0.012 |   0.012 |   0.053 |   0.220 |
| foreground_std_delta_detector_minus_gt      |   0.000 |  0.009 | -0.014 | -0.005 |   0.000 |   0.005 |   0.013 |
| z_range_delta_detector_minus_gt             | 104.551 | 64.387 | -0.713 | 56.471 | 109.473 | 151.935 | 201.101 |
| z_std_delta_detector_minus_gt               |  14.142 | 10.398 | -2.514 |  6.990 |  14.285 |  21.106 |  31.216 |
| grad_p99_delta_detector_minus_gt            |   2.503 |  1.683 |  0.000 |  1.345 |   2.460 |   3.520 |   5.412 |
| grad_max_delta_detector_minus_gt            |  11.494 | 15.005 | -0.919 |  3.296 |   7.226 |  14.922 |  40.995 |
| qc_badness_delta_detector_minus_gt          |   0.264 |  0.157 |  0.002 |  0.147 |   0.271 |   0.379 |   0.508 |

## Detector Most Worse Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.292049618819567427252971059233 |           0.000 |                 0.676 |                                0.676 |        0.000 |            238.086 |         0.000 |               6.816 |                    0.590 |                          0.450 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.301582691063019848479942618641 |           0.000 |                 0.662 |                                0.662 |        0.000 |            266.999 |         0.000 |               6.468 |                    0.578 |                          0.708 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.302403227435841351528721627052 |           0.000 |                 0.654 |                                0.654 |        0.000 |            232.012 |         0.000 |               5.487 |                    0.526 |                          0.518 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.428038562098395445838061018440 |           0.000 |                 0.637 |                                0.637 |        0.000 |            243.218 |         0.000 |               6.847 |                    0.622 |                          0.633 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.168737928729363683423228050295 |           0.000 |                 0.632 |                                0.632 |        0.000 |            224.193 |         0.000 |               8.254 |                    0.597 |                          0.548 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.341557859428950960906150406596 |           0.000 |                 0.632 |                                0.632 |        0.000 |            248.057 |         0.000 |               6.246 |                    0.537 |                          0.562 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.254138388912084634057282064266 |           0.000 |                 0.620 |                                0.620 |        0.000 |            216.357 |         0.000 |               8.009 |                    0.679 |                          0.676 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.655242448149322898770987310561 |           0.000 |                 0.616 |                                0.616 |        0.000 |            243.818 |         0.000 |               6.659 |                    0.652 |                          0.628 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.246225645401227472829175288633 |           0.000 |                 0.616 |                                0.616 |        0.000 |            243.338 |         0.000 |               6.124 |                    0.629 |                          0.621 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.149041668385192796520281592139 |           0.000 |                 0.612 |                                0.612 |        0.000 |            241.713 |         0.000 |               3.850 |                    0.524 |                          0.394 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.199261544234308780356714831537 |           0.000 |                 0.604 |                                0.604 |        0.000 |            263.298 |         0.000 |               4.123 |                    0.447 |                          0.683 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.194632613233275988184244485809 |           0.000 |                 0.601 |                                0.601 |        0.000 |            194.987 |         0.000 |               8.070 |                    0.394 |                          0.574 |

## Detector Most Better Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.592821488053137951302246128864 |           0.258 |                 0.013 |                               -0.245 |       98.000 |              5.742 |         2.956 |               0.083 |                    0.530 |                          0.543 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.111496024928645603833332252962 |           0.160 |                 0.000 |                               -0.160 |       66.000 |              0.000 |         1.289 |               0.000 |                    0.593 |                          0.595 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.159665703190517688573100822213 |           0.186 |                 0.036 |                               -0.150 |       61.000 |             16.196 |         2.619 |               0.266 |                    0.606 |                          0.514 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.339484970190920330170416228517 |           0.241 |                 0.105 |                               -0.136 |      101.000 |             12.913 |         2.074 |               0.223 |                    0.598 |                          0.228 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.106379658920626694402549886949 |           0.124 |                 0.000 |                               -0.124 |       55.000 |              0.000 |         0.758 |               0.000 |                    0.484 |                          0.526 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.230416590143922549745658357505 |           0.366 |                 0.243 |                               -0.122 |      139.000 |             96.648 |         4.053 |               2.199 |                    0.649 |                          0.519 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.910435939545691201820711078950 |           0.360 |                 0.248 |                               -0.112 |      155.000 |            100.643 |         2.273 |               2.038 |                    0.632 |                          0.643 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518 |           0.371 |                 0.266 |                               -0.105 |      142.000 |             96.300 |         4.204 |               3.285 |                    0.586 |                          0.595 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.223650122819238796121876338881 |           0.287 |                 0.189 |                               -0.098 |      100.000 |             85.112 |         3.542 |               1.269 |                    0.533 |                          0.589 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.264090899378396711987322794314 |           0.468 |                 0.373 |                               -0.095 |      154.000 |            140.318 |         6.426 |               4.270 |                    0.570 |                          0.446 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.272190966764020277652079081128 |           0.242 |                 0.154 |                               -0.088 |       97.000 |             69.852 |         2.301 |               1.067 |                    0.521 |                          0.451 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.315918264676377418120578391325 |           0.308 |                 0.235 |                               -0.073 |      134.000 |             94.090 |         1.554 |               2.411 |                    0.554 |                          0.599 |

## Output Files

- `ground_truth_surface_qc_metrics.csv`
- `detector_top5_surface_qc_metrics.csv`
- `paired_surface_qc_comparison.csv`
