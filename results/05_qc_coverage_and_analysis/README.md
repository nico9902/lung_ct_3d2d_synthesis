# QC, Coverage And Analysis

Questa sezione raccoglie file di supporto per giustificare la qualita' delle sintetiche e la scelta del punto di lavoro detector.

Contenuti:

- coverage top-k/threshold del detector.
- report FPR/coverage.
- analisi QC sulle superfici sintetiche disponibili in `docs/`.

Per la scrittura del paper, questa sezione supporta tre punti:

1. Il working point non e' scelto solo sul downstream classifier, ma anche con un vincolo esplicito di budget FP/scan.
2. Il punto `threshold=0.50`, `top-k=4` mantiene alta copertura nodulare con budget contenuto.
3. Il confronto con MIP mostra che la proiezione adaptive concentra informazione rilevante meglio delle proiezioni fisse.
