# 2D Non-Adaptive Baselines

Questa sezione raccoglie le baseline 2D che non usano una proiezione adaptive guidata dai noduli.

Baseline disponibili:

- `mip_efficientnetv2s_pooled_results.md`: MIP assiale e MIP tri-view con EfficientNetV2-S.
- `central_slice_efficientnetv2s_results.md`: central axial slice con EfficientNetV2-S, completa su 10 fold.
- `detection_crop_mil_baseline.md`: crop 2D sui candidati detector con EfficientNetV2-S e aggregazione patient-level mean/max/attention.
- `slice_attention_2p5d_baseline.md`: tutte le slice assiali del volume preprocessato con EfficientNetV2-S condivisa e gated soft attention.

Motivo scientifico:

Le baseline non-adaptive servono a separare il contributo della compressione 2D dal contributo della proiezione adaptive. La MIP testa una compressione fissa dell'intero volume; la central slice testa una scelta ancora piu' semplice, cioe' una singola sezione anatomica fissa senza aggregazione. La baseline 2.5D slice-attention usa invece tutte le slice assiali con un encoder 2D condiviso e attention, quindi verifica se una forte aggregazione multi-slice non-adaptive e' sufficiente. La baseline detection-crop MIL risponde a un'altra domanda probabile da reviewer: se il detector ha gia' trovato i candidati, basta classificare direttamente i crop dei noduli e aggregarli a livello paziente? Se queste baseline restano sotto Adaptive RBF, il risultato supporta il contributo della superficie patient-level, non solo quello del detector o del backbone 2D pretrained.
