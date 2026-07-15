# LUNA16 Detector vs Ground-Truth Synthetic Surface QC

Lower `qc_badness` is better. The score is a triage heuristic combining foreground coverage, foreground contrast, z-surface range, edge variation, and surface gradient roughness.

- Ground-truth generated cases analyzed: **888**
- Detector top-5 generated cases analyzed: **888**
- Paired cases compared by `seriesuid`: **888**
- Detector lower/better `qc_badness`: **62 / 888 (7.0%)**
- Detector equal-or-higher/worse `qc_badness`: **826 / 888 (93.0%)**

## Source Summary

| source        |   n |   qc_badness_mean |   qc_badness_median |   qc_badness_p95 |   foreground_fraction_mean |   z_range_mean |   grad_p99_mean |
|:--------------|----:|------------------:|--------------------:|-----------------:|---------------------------:|---------------:|----------------:|
| ground_truth  | 888 |             0.103 |               0.030 |            0.394 |                      0.536 |         40.601 |           0.768 |
| detector_top5 | 888 |             0.389 |               0.379 |            0.641 |                      0.548 |        148.126 |           3.503 |

## QC Buckets

`good_or_low_review` is `qc_badness <= 0.10`, `moderate_review` is `0.10 < qc_badness <= 0.30`, and `high_review` is `qc_badness > 0.30`.

| source        | qc_bucket          |   n |
|:--------------|:-------------------|----:|
| ground_truth  | good_or_low_review | 584 |
| ground_truth  | moderate_review    | 214 |
| ground_truth  | high_review        |  90 |
| detector_top5 | good_or_low_review |  18 |
| detector_top5 | moderate_review    | 209 |
| detector_top5 | high_review        | 661 |

## Likely Explanation

The detector-top5 run uses five detector candidates for every scan. In these saved outputs, each nodule/detection region contributes five labelled control points, so detector-top5 has `25` control labels for every case. Ground-truth generation is much less constrained: most cases have one or two nodule regions, corresponding to `5` or `10` control labels.

This matches the QC failure mode: more control regions force the interpolated surface to satisfy points across a larger cranio-caudal spread, increasing `z_range` and `grad_p99`. The foreground image statistics stay similar, but the detector surfaces are geometrically rougher.

| source        |   control_labels |   count |   mean |   median |
|:--------------|-----------------:|--------:|-------:|---------:|
| ground_truth  |                1 |     313 |  0.019 |    0.012 |
| ground_truth  |                2 |     156 |  0.158 |    0.133 |
| ground_truth  |                3 |      64 |  0.206 |    0.187 |
| ground_truth  |                4 |      24 |  0.246 |    0.259 |
| ground_truth  |                5 |     167 |  0.055 |    0.013 |
| ground_truth  |                6 |       9 |  0.202 |    0.200 |
| ground_truth  |                7 |       4 |  0.448 |    0.439 |
| ground_truth  |                8 |       2 |  0.159 |    0.159 |
| ground_truth  |                9 |       4 |  0.378 |    0.371 |
| ground_truth  |               10 |      84 |  0.135 |    0.104 |
| ground_truth  |               12 |       1 |  0.476 |    0.476 |
| ground_truth  |               15 |      32 |  0.184 |    0.171 |
| ground_truth  |               20 |      10 |  0.268 |    0.206 |
| ground_truth  |               25 |       6 |  0.356 |    0.373 |
| ground_truth  |               30 |       4 |  0.376 |    0.392 |
| ground_truth  |               35 |       3 |  0.428 |    0.451 |
| ground_truth  |               40 |       2 |  0.418 |    0.418 |
| ground_truth  |               45 |       3 |  0.400 |    0.398 |
| detector_top5 |                5 |       6 |  0.012 |    0.009 |
| detector_top5 |               10 |      14 |  0.112 |    0.103 |
| detector_top5 |               15 |       7 |  0.213 |    0.242 |
| detector_top5 |               20 |      11 |  0.233 |    0.247 |
| detector_top5 |               25 |      14 |  0.271 |    0.268 |
| detector_top5 |               30 |      11 |  0.397 |    0.366 |
| detector_top5 |               35 |     825 |  0.402 |    0.386 |

## Paired Delta Summary

Delta is detector minus ground truth; negative `qc_badness` means detector is better by this heuristic.

| metric                                      |    mean |    std |      5% |    25% |     50% |     75% |     95% |
|:--------------------------------------------|--------:|-------:|--------:|-------:|--------:|--------:|--------:|
| foreground_fraction_delta_detector_minus_gt |   0.012 |  0.126 |  -0.217 | -0.028 |   0.010 |   0.063 |   0.236 |
| foreground_std_delta_detector_minus_gt      |  -0.000 |  0.014 |  -0.015 | -0.005 |   0.001 |   0.006 |   0.014 |
| z_range_delta_detector_minus_gt             | 107.525 | 72.950 | -16.348 | 57.131 | 115.828 | 157.529 | 220.772 |
| z_std_delta_detector_minus_gt               |  20.473 | 15.336 |  -5.247 | 10.696 |  21.458 |  30.330 |  43.869 |
| grad_p99_delta_detector_minus_gt            |   2.734 |  2.327 |  -0.342 |  1.502 |   2.525 |   3.757 |   6.222 |
| grad_max_delta_detector_minus_gt            |  11.901 | 18.309 |   0.064 |  3.322 |   6.469 |  13.411 |  44.106 |
| qc_badness_delta_detector_minus_gt          |   0.286 |  0.194 |  -0.028 |  0.166 |   0.298 |   0.404 |   0.583 |

## Detector Most Worse Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492 |           0.008 |                 1.000 |                                0.992 |        2.871 |            289.000 |         0.058 |              14.792 |                    0.583 |                          0.385 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.310626494937915759224334597176 |           0.018 |                 1.000 |                                0.982 |        6.026 |            300.000 |         0.177 |              16.879 |                    0.442 |                          0.510 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.329326052298830421573852261436 |           0.023 |                 0.983 |                                0.960 |        9.374 |            267.000 |         0.178 |              14.603 |                    0.451 |                          0.434 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.160124400349792614505500125883 |           0.017 |                 0.903 |                                0.887 |        5.767 |            306.000 |         0.158 |              11.745 |                    0.404 |                          0.611 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.419601611032172899567156073142 |           0.008 |                 0.858 |                                0.850 |        3.120 |            265.000 |         0.070 |              11.634 |                    0.586 |                          0.499 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.724562063158320418413995627171 |           0.009 |                 0.857 |                                0.849 |        3.143 |            298.046 |         0.070 |               9.840 |                    0.597 |                          0.325 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.877026508860018521147620598474 |           0.015 |                 0.856 |                                0.841 |        5.011 |            278.000 |         0.141 |              10.287 |                    0.506 |                          0.429 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.149041668385192796520281592139 |           0.005 |                 0.842 |                                0.837 |        2.094 |            247.340 |         0.040 |              10.932 |                    0.536 |                          0.449 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.975254950136384517744116790879 |           0.007 |                 0.820 |                                0.814 |        2.333 |            275.000 |         0.058 |              10.647 |                    0.672 |                          0.454 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.179162671133894061547290922949 |           0.190 |                 1.000 |                                0.810 |       86.216 |            315.000 |         0.917 |              15.053 |                    0.579 |                          0.496 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.888615810685807330497715730842 |           0.013 |                 0.786 |                                0.773 |        4.504 |            227.493 |         0.111 |              11.342 |                    0.670 |                          0.455 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.254138388912084634057282064266 |           0.012 |                 0.751 |                                0.739 |        4.281 |            241.826 |         0.098 |               9.334 |                    0.681 |                          0.674 |

## Detector Most Better Than Ground Truth

| seriesuid                                                        |   qc_badness_gt |   qc_badness_detector |   qc_badness_delta_detector_minus_gt |   z_range_gt |   z_range_detector |   grad_p99_gt |   grad_p99_detector |   foreground_fraction_gt |   foreground_fraction_detector |
|:-----------------------------------------------------------------|----------------:|----------------------:|-------------------------------------:|-------------:|-------------------:|--------------:|--------------------:|-------------------------:|-------------------------------:|
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.331211682377519763144559212009 |           0.909 |                 0.288 |                               -0.621 |      257.416 |             99.580 |        13.431 |               3.609 |                    0.384 |                          0.600 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.116703382344406837243058680403 |           0.788 |                 0.182 |                               -0.607 |      290.000 |             81.118 |         8.585 |               0.830 |                    0.594 |                          0.679 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.111780708132595903430640048766 |           0.911 |                 0.335 |                               -0.576 |      322.425 |            142.296 |        10.119 |               1.958 |                    0.482 |                          0.609 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.307946352302138765071461362398 |           0.608 |                 0.345 |                               -0.264 |      203.179 |            145.151 |         7.274 |               2.506 |                    0.594 |                          0.573 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.592821488053137951302246128864 |           0.272 |                 0.014 |                               -0.258 |      116.025 |              5.789 |         2.404 |               0.104 |                    0.507 |                          0.545 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866 |           0.556 |                 0.311 |                               -0.245 |      175.974 |            119.939 |         8.038 |               2.956 |                    0.642 |                          0.655 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.306112617218006614029386065035 |           0.508 |                 0.287 |                               -0.221 |      163.887 |            121.315 |         6.352 |               2.127 |                    0.522 |                          0.566 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.230416590143922549745658357505 |           0.492 |                 0.287 |                               -0.204 |      192.556 |            107.672 |         4.287 |               2.747 |                    0.608 |                          0.398 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.232058316950007760548968840196 |           0.397 |                 0.216 |                               -0.181 |      128.202 |             79.986 |         4.478 |               1.939 |                    0.504 |                          0.530 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.111496024928645603833332252962 |           0.171 |                 0.007 |                               -0.164 |       70.425 |              2.434 |         1.178 |               0.071 |                    0.602 |                          0.594 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.176030616406569931557298712518 |           0.476 |                 0.316 |                               -0.160 |      181.204 |            106.270 |         3.938 |               3.594 |                    0.609 |                          0.604 |
| 1.3.6.1.4.1.14519.5.2.1.6279.6001.213140617640021803112060161074 |           0.436 |                 0.283 |                               -0.152 |      167.512 |            120.987 |         3.940 |               2.086 |                    0.602 |                          0.594 |

## Output Files

- `ground_truth_surface_qc_metrics.csv`
- `detector_top5_surface_qc_metrics.csv`
- `paired_surface_qc_comparison.csv`
