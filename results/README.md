# Results Directory

Questa cartella e' organizzata per preparare la scrittura del paper.

## Struttura

- `00_paper_overview/`: sintesi paper-ready, tabelle master, file Excel aggregati.
- `01_synthetic_2d_adaptive/`: risultati delle rappresentazioni sintetiche 2D adaptive, incluse RBF e Shepard.
- `02_detector_and_working_points/`: risultati del detector, FROC, confronti con baseline e punti di lavoro.
- `03_2d_nonadaptive_baselines/`: baseline 2D non-adaptive, per ora MIP assiale e MIP tri-view.
- `04_3d_volumetric_baseline/`: baseline volumetrica 3D ResNet18 e giustificazione della dimensione fissa.
- `05_qc_coverage_and_analysis/`: copertura noduli, mappe top-k/threshold e report QC.
- `99_legacy_flat_files/`: spazio per eventuali vecchi file mantenuti con nome originale.

## File chiave

- `00_paper_overview/paper_results_summary.md`: sintesi interpretativa per il paper.
- `00_paper_overview/master_pooled_results.csv`: tabella unica con i principali risultati pooled.
- `03_2d_nonadaptive_baselines/mip_efficientnetv2s_pooled_results.md`: risultati locali MIP assiale/tri-view.

## Convenzione

Le metriche principali sono riportate in forma pooled quando disponibile: le predizioni test delle 10 fold vengono concatenate e le metriche vengono ricalcolate sul totale. Questo e' diverso dalla media fold-wise.
