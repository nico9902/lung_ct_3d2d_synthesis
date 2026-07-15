# LUNA16 Detector vs Ground-Truth Synthetic Surface QC

Lower `qc_badness` is better. The score is a triage heuristic combining foreground coverage, foreground contrast, z-surface range, edge variation, and surface gradient roughness.

- Ground-truth generated cases analyzed: **888**
- Detector top-5 generated cases analyzed: **888**
- Paired cases compared by `seriesuid`: **888**
- Detector lower/better `qc_badness`: **82 / 888 (9.2%)**
- Detector equal-or-higher/worse `qc_badness`: **806 / 888 (90.8%)**

## Source Summary

| source        |   n |   qc_badness_mean |   qc_badness_median |   qc_badness_p95 |   foreground_fraction_mean |   z_range_mean |   grad_p99_mean |
|:--------------|----:|------------------:|--------------------:|-----------------:|---------------------------:|---------------:|----------------:|
| ground_truth  | 888 |             0.102 |               0.030 |            0.364 |                      0.538 |         39.338 |           0.802 |
| detector_top5 | 888 |             0.332 |               0.320 |            0.565 |                      0.538 |        128.623 |           2.747 |

## QC Buckets

`good_or_low_review` is `qc_badness <= 0.10`, `moderate_review` is `0.10 < qc_badness <= 0.30`, and `high_review` is `qc_badness > 0.30`.

| source        | qc_bucket          |   n |
|:--------------|:-------------------|----:|
| ground_truth  | good_or_low_review | 579 |
| ground_truth  | moderate_review    | 227 |
| ground_truth  | high_review        |  82 |
| detector_top5 | good_or_low_review |  16 |
| detector_top5 | moderate_review    | 373 |
| detector_top5 | high_review        | 499 |

## Likely Explanation

The detector-top5 run uses five detector candidates for every scan. In these saved outputs, each nodule/detection region contributes five labelled control points, so detector-top5 has `25` control labels for every case. Ground-truth generation is much less constrained: most cases have one or two nodule regions, corresponding to `5` or `10` control labels.

This matches the QC failure mode: more control regions force the interpolated surface to satisfy points across a larger cranio-caudal spread, increasing `z_range` and `grad_p99`. The foreground image statistics stay similar, but the detector surfaces are geometrically rougher.

| source        |   control_labels |   count |   mean |   median |
|:--------------|-----------------:|--------:|-------:|---------:|
| ground_truth  |                5 |     455 |  0.017 |    0.012 |
| ground_truth  |               10 |     241 |  0.151 |    0.127 |
| ground_truth  |               15 |      96 |  0.188 |    0.164 |
| ground_truth  |               20 |      34 |  0.244 |    0.261 |
| ground_truth  |               25 |      30 |  0.301 |    0.332 |
| ground_truth  |               30 |      13 |  0.212 |    0.190 |
| ground_truth  |               35 |       7 |  0.423 |    0.418 |
| ground_truth  |               40 |       4 |  0.319 |    0.360 |
| ground_truth  |               45 |       7 |  0.426 |    0.390 |
| ground_truth  |               60 |       1 |  0.401 |    0.401 |
| detector_top5 |               25 |     888 |  0.332 |    0.320 |

## Paired Delta Summary

Delta is detector minus ground truth; negative `qc_badness` means detector is better by this heuristic.

| metric                                      |   mean |    std |      5% |    25% |    50% |     75% |     95% |
|:--------------------------------------------|-------:|-------:|--------:|-------:|-------:|--------:|--------:|
| foreground_fraction_delta_detector_minus_gt | -0.000 |  0.137 |  -0.281 | -0.031 |  0.007 |   0.050 |   0.211 |
| foreground_std_delta_detector_minus_gt      |  0.000 |  0.013 |  -0.014 | -0.004 |  0.001 |   0.005 |   0.015 |
| z_range_delta_detector_minus_gt             | 89.285 | 69.812 | -18.154 | 38.816 | 91.116 | 139.322 | 195.373 |
| z_std_delta_detector_minus_gt               | 17.443 | 14.349 |  -5.461 |  7.580 | 17.487 |  26.889 |  41.101 |
| grad_p99_delta_detector_minus_gt            |  1.945 |  1.942 |  -1.089 |  0.879 |  1.844 |   2.895 |   5.194 |
| grad_max_delta_detector_minus_gt            |  6.591 | 11.977 |  -1.686 |  1.466 |  3.626 |   7.862 |  27.992 |
| qc_badness_delta_detector_minus_gt          |  0.231 |  0.177 |  -0.043 |  0.116 |  0.236 |   0.350 |   0.507 |

## Detector Most Worse Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.160124400349792614505500125883 |           0.017 |                 0.917 |                                0.901 |        5.758 |            306.000 |         0.158 |              12.086 |                    0.405 |                          0.608 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.975254950136384517744116790879 |           0.009 |                 0.898 |                                0.889 |        3.189 |            275.000 |         0.080 |              12.139 |                    0.601 |                          0.386 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.254138388912084634057282064266 |           0.014 |                 0.728 |                                0.714 |        4.899 |            238.305 |         0.130 |               8.779 |                    0.578 |                          0.651 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.199975006921901879512837687266 |           0.062 |                 0.771 |                                0.709 |       22.082 |            258.865 |         0.218 |               7.741 |                    0.327 |                          0.343 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.109882169963817627559804568094 |           0.022 |                 0.683 |                                0.661 |        8.418 |            275.000 |         0.191 |               6.751 |                    0.593 |                          0.561 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.199261544234308780356714831537 |           0.020 |                 0.676 |                                0.656 |        7.444 |            279.572 |         0.167 |               4.960 |                    0.421 |                          0.634 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.341557859428950960906150406596 |           0.018 |                 0.673 |                                0.655 |        6.906 |            250.242 |         0.143 |               6.480 |                    0.486 |                          0.448 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.153536305742006952753134773630 |           0.336 |                 0.981 |                                0.645 |      136.530 |            284.534 |         2.904 |              13.852 |                    0.555 |                          0.350 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.292049618819567427252971059233 |           0.012 |                 0.631 |                                0.619 |        4.482 |            240.576 |         0.106 |               5.701 |                    0.594 |                          0.400 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.194632613233275988184244485809 |           0.015 |                 0.625 |                                0.609 |        5.238 |            197.959 |         0.149 |               8.043 |                    0.382 |                          0.513 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.127965161564033605177803085629 |           0.010 |                 0.615 |                                0.605 |        3.612 |            222.623 |         0.095 |               7.260 |                    0.476 |                          0.421 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.272348349298439120568330857680 |           0.016 |                 0.613 |                                0.597 |        5.664 |            231.868 |         0.137 |               5.736 |                    0.502 |                          0.614 |

## Detector Most Better Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.811825890493256320617655474043 |           0.733 |                 0.253 |                               -0.479 |      243.024 |            106.017 |         8.673 |               1.665 |                    0.602 |                          0.629 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.229664630348267553620068691756 |           0.829 |                 0.404 |                               -0.426 |      278.076 |            156.521 |        10.559 |               3.383 |                    0.575 |                          0.558 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.167500254299688235071950909530 |           0.729 |                 0.338 |                               -0.391 |      275.967 |            143.613 |         7.448 |               2.517 |                    0.589 |                          0.665 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.306112617218006614029386065035 |           0.499 |                 0.167 |                               -0.333 |      162.512 |             63.812 |         6.720 |               1.727 |                    0.562 |                          0.535 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.217589936421986638139451480826 |           0.453 |                 0.196 |                               -0.256 |      208.260 |             84.568 |         2.261 |               0.875 |                    0.537 |                          0.596 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.324649110927013926557500550446 |           0.517 |                 0.263 |                               -0.254 |      182.885 |             47.005 |         6.369 |               1.575 |                    0.510 |                          0.128 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.184019785706727365023450012318 |           0.500 |                 0.251 |                               -0.249 |      174.335 |            109.615 |         5.567 |               1.108 |                    0.573 |                          0.603 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.237915456403882324748189195892 |           0.612 |                 0.369 |                               -0.242 |      214.906 |            138.096 |         6.574 |               3.263 |                    0.620 |                          0.684 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.264090899378396711987322794314 |           0.327 |                 0.086 |                               -0.241 |      127.260 |             37.911 |         3.187 |               0.609 |                    0.584 |                          0.353 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.103115201714075993579787468219 |           0.297 |                 0.074 |                               -0.223 |      105.459 |             29.222 |         3.337 |               0.652 |                    0.561 |                          0.545 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.175318131822744218104175746898 |           0.478 |                 0.267 |                               -0.211 |      167.124 |             95.564 |         5.348 |               2.691 |                    0.605 |                          0.617 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.910435939545691201820711078950 |           0.370 |                 0.159 |                               -0.211 |      156.731 |             62.337 |         2.151 |               1.315 |                    0.631 |                          0.637 |

## Output Files

- `ground_truth_surface_qc_metrics.csv`
- `detector_top5_surface_qc_metrics.csv`
- `paired_surface_qc_comparison.csv`
