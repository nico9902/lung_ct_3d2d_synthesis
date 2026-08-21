# Adaptive Synthetic 2D Results

Questa sezione contiene i risultati delle rappresentazioni sintetiche 2D adaptive.

File principali:

- `luna16_synthetic_2d_cpmnetv2_bf16_top4_minprob0.50_pooled_results.md`: risultato migliore attuale, RBF + EfficientNetV2-S.
- `luna16_synthetic_2d.md`: risultati storici con GT synthetic e detector top5/minprob0.5.
- `luna16_synthetic_2d_top7_minprob0.3_results.md`: esperimenti top7/minprob0.3.
- `luna16_synthetic_2d_top3_minprob0.5_results.md` e `luna16_synthetic_2d_top4_minprob0.5_results.md`: sweep del working point.

Per il paper, il candidato principale e':

`CPMNetv2 BF16, top4/minprob0.50, RBF, EfficientNetV2-S`

con AUC pooled `0.8149` e MCC pooled `0.4780`.
