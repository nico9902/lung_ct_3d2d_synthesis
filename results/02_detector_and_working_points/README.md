# Detector And Working Points

Questa sezione contiene i risultati del detector e le analisi usate per scegliere il punto di lavoro delle sintetiche.

File principali:

- `README_results.md`: aggregazione FROC del detector CPMNetv2 BF16.
- `aggregate_froc_summary.csv`: FROC aggregata.
- `fold_froc_summary.csv`: FROC per fold.
- `froc_comparison_vs_baseline.md`: confronto con il detector baseline precedente.
- `performance_comparison_vs_scpmnet_luna16_10fold.md`: confronto del run normauto/spv4/lr0.01/gradclip.
- `test_froc.csv`: FROC del detector normauto baseline.

Il punto di lavoro usato per il risultato migliore delle sintetiche e':

- threshold: `0.50`
- top-k: `4`
- budget operativo: circa `2 FP/scan`
- copertura nodulare: circa `92%` al budget di 2 FP/scan nella valutazione detector aggregata.
