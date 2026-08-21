# FROC comparison vs previous detector

Baseline: `outputs/scpmnet_luna16_10fold/cv_aggregate/pooled_test_froc.csv`.
New detector: `outputs/scpmnet_luna16_10fold_normauto_amp_spv4_lr001_randomcrop_gradclip100_merged/cv_aggregate/pooled_test_froc.csv`.

- Baseline pooled mean FROC: `0.4633`
- New pooled mean FROC: `0.6914`
- Absolute improvement: `+0.2281`
- Relative improvement: `+49.2%`

| FP/scan | Baseline sensitivity | New sensitivity | Delta | Relative delta |
| ---: | ---: | ---: | ---: | ---: |
| 0.125 | 0.0000 | 0.3904 | +0.3904 | n/a |
| 0.25 | 0.0169 | 0.4958 | +0.4789 | +2840.0% |
| 0.5 | 0.3457 | 0.6239 | +0.2782 | +80.5% |
| 1 | 0.5759 | 0.7319 | +0.1560 | +27.1% |
| 2 | 0.7049 | 0.8187 | +0.1138 | +16.1% |
| 4 | 0.7749 | 0.8702 | +0.0953 | +12.3% |
| 8 | 0.8246 | 0.9089 | +0.0843 | +10.2% |

The largest gains are at low FP/scan, where the previous detector had near-zero sensitivity at 0.125-0.25 FP/scan.
