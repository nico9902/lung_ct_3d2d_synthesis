# Control Point Position Distribution Comparison

Comparison between generated pseudo-regions in scans without nodules and real nodule-derived regions in scans with nodules. Coordinates are normalized to relative `z,y,x` in `[0,1]`.

## Sample Counts

| group                  |   control_points |   region_centers |
|:-----------------------|-----------------:|-----------------:|
| with_nodules_real      |             5895 |             1179 |
| without_nodules_pseudo |             2830 |              566 |

## KS Distances

KS distance close to 0 means similar 1D distributions; larger values mean more mismatch.

| sample_level   | axis   |   ks_distance |
|:---------------|:-------|--------------:|
| control_points | rel_z  |         0.100 |
| control_points | rel_y  |         0.131 |
| control_points | rel_x  |         0.094 |
| region_centers | rel_z  |         0.102 |
| region_centers | rel_y  |         0.145 |
| region_centers | rel_x  |         0.100 |

## Summary Statistics

| sample_level   | group                  |    n |   rel_z_mean |   rel_z_std |   rel_z_p05 |   rel_z_p50 |   rel_z_p95 |   rel_y_mean |   rel_y_std |   rel_x_mean |   rel_x_std |
|:---------------|:-----------------------|-----:|-------------:|------------:|------------:|------------:|------------:|-------------:|------------:|-------------:|------------:|
| control_points | with_nodules_real      | 5895 |        0.832 |       0.213 |       0.372 |       0.970 |       1.000 |        0.564 |       0.219 |        0.481 |       0.280 |
| control_points | without_nodules_pseudo | 2830 |        0.832 |       0.191 |       0.420 |       0.938 |       0.998 |        0.617 |       0.215 |        0.478 |       0.244 |
| region_centers | with_nodules_real      | 1179 |        0.832 |       0.213 |       0.373 |       0.970 |       1.000 |        0.564 |       0.219 |        0.481 |       0.280 |
| region_centers | without_nodules_pseudo |  566 |        0.832 |       0.191 |       0.424 |       0.937 |       0.997 |        0.617 |       0.211 |        0.478 |       0.242 |

## Plots

![position_distribution_control_points.png](assets/position_distribution_control_points.png)

![position_distribution_region_centers.png](assets/position_distribution_region_centers.png)

![position_distribution_region_centers_scatter.png](assets/position_distribution_region_centers_scatter.png)


## Interpretation

- The largest region-center mismatch is on `rel_y` with KS=0.145.

- The comparison is better judged on region centers than on all control points, because each region contributes one center plus contour points.
