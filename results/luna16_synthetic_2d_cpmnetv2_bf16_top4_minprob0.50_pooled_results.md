# LUNA16 Synthetic 2D - CPMNetv2 BF16 Top4 MinProb0.50

Risultati pooled delle backbone addestrate sulle sintetiche generate dal detector CPMNetv2 10-fold BF16 con punto di lavoro:

- threshold: `0.50`
- top-k: `4`
- interpolazioni: `rbf`, `shepard`
- predizioni aggregate: tutti i sample dei 10 fold concatenati per backbone
- supporto pooled per ogni backbone: `796` sample, `320` maligni, `476` benigni

Nota: questi valori sono pooled, quindi non sono la media delle metriche fold-wise. L'AUC e calcolata sui `score` concatenati; MCC, accuracy, F1, precision e recall sono calcolati sulle predizioni binarie concatenate.

## Risultati Pooled

| Method | Backbone | AUC | MCC | Accuracy | F1 | Precision | Recall | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rbf | efficientnet_v2_s | **0.8149** | **0.4780** | 0.7513 | **0.6806** | 0.7033 | **0.6594** | 387 | 89 | 109 | 211 |
| rbf | densenet121 | 0.7989 | 0.4487 | 0.7374 | 0.6624 | 0.6856 | 0.6406 | 382 | 94 | 115 | 205 |
| shepard | efficientnet_v2_s | 0.7793 | 0.4308 | 0.7324 | 0.6283 | 0.7115 | 0.5625 | 403 | 73 | 140 | 180 |
| shepard | resnet50 | 0.7784 | 0.4256 | 0.7286 | 0.6376 | 0.6884 | 0.5938 | 390 | 86 | 130 | 190 |
| rbf | vgg16 | 0.7754 | 0.4568 | 0.7412 | 0.6677 | 0.6900 | 0.6469 | 383 | 93 | 113 | 207 |
| rbf | efficientnet_b0 | 0.7701 | 0.4155 | 0.7211 | 0.6442 | 0.6612 | 0.6281 | 373 | 103 | 119 | 201 |
| shepard | densenet121 | 0.7682 | 0.3918 | 0.7111 | 0.6254 | 0.6531 | 0.6000 | 374 | 102 | 128 | 192 |
| rbf | resnet50 | 0.7658 | 0.4122 | 0.7211 | 0.6361 | 0.6690 | 0.6062 | 380 | 96 | 126 | 194 |
| rbf | resnet18 | 0.7622 | 0.4050 | 0.7161 | 0.6378 | 0.6546 | 0.6219 | 371 | 105 | 121 | 199 |
| shepard | vgg16 | 0.7604 | 0.4747 | **0.7525** | 0.6501 | **0.7531** | 0.5719 | 416 | 60 | 137 | 183 |
| shepard | efficientnet_b0 | 0.7582 | 0.4069 | 0.7186 | 0.6328 | 0.6655 | 0.6031 | 379 | 97 | 127 | 193 |
| shepard | resnet18 | 0.7519 | 0.3599 | 0.6997 | 0.5858 | 0.6576 | 0.5281 | 388 | 88 | 151 | 169 |
| rbf | efficientnet_b1 | 0.7507 | 0.3576 | 0.6972 | 0.5936 | 0.6447 | 0.5500 | 379 | 97 | 144 | 176 |
| shepard | efficientnet_b1 | 0.7396 | 0.3071 | 0.6646 | 0.5911 | 0.5796 | 0.6031 | 336 | 140 | 127 | 193 |

## Sintesi

La configurazione migliore in termini di AUC pooled e:

`rbf + efficientnet_v2_s`

con `AUC = 0.8149`, `MCC = 0.4780`, `F1 = 0.6806`.

Rispetto ai risultati precedenti migliori, il nuovo punto di lavoro `threshold=0.50, topk=4` con interpolazione `rbf` e backbone `efficientnet_v2_s` e il candidato piu forte: supera la soglia di AUC 0.80 e mantiene anche il miglior MCC pooled.
