# Perche le sintetiche da detection rendono peggio delle sintetiche GT

Questo report usa i mosaici in `docs/luna16_synthetic_2d_detection_slice_gradcam_mosaics_classified_gray_fixed/` per spiegare visivamente perche il classificatore addestrato o valutato sulle sintetiche guidate dal detector ha performance piu basse rispetto a quello guidato dalle ground truth.

Il confronto quantitativo piu recente mostra il calo principale su Top5 minprob0.5: il miglior backbone GT (`efficientnet_v2_s`) raggiunge MCC `0.618` e AUC `0.873`, mentre il corrispondente detector Top5 arriva a MCC `0.439` e AUC `0.746`. Il delta e' quindi circa `-0.178` MCC e `-0.127` AUC.

Nei mosaici: cyan = detection che intercetta la maschera GT; arancione = detection miss; giallo/rosso = maschera GT. La colonna a destra mostra le Grad-CAM e l'esito della classificazione.

## Sintesi Dai Mosaici

I mosaici coprono `1075` righe visuali, corrispondenti a `796` campioni unici. Alcuni campioni compaiono in piu categorie diagnostiche, quindi le righe non sono mutuamente esclusive.

| Categoria | n | Noduli GT medi | Detection Top5 medie | Hit Top5 medi | Top5 zero-hit | Top5 no CSV | Top5 parziale/missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| both_correct | 515 | 1.59 | 2.25 | 1.06 | 155 | 123 | 219 |
| both_wrong | 75 | 1.80 | 2.57 | 1.19 | 14 | 8 | 28 |
| gt_correct_top5_wrong | 136 | 1.62 | 2.29 | 0.79 | 48 | 29 | 64 |
| gt_wrong_top5_correct | 70 | 1.84 | 2.70 | 1.15 | 12 | 9 | 25 |
| top5_false_positive | 85 | 1.40 | 2.69 | 0.69 | 39 | 7 | 50 |
| top5_true_positive | 194 | 2.40 | 3.46 | 1.82 | 4 | 3 | 61 |

Punti chiave:

- `272` righe hanno Top5 con zero hit sul nodulo GT.
- `447` righe con CSV disponibile hanno Top5 parziale o mancante rispetto ai noduli GT del campione.
- `375` righe sono multi-nodulo; tra queste, `198` hanno copertura Top5 parziale o mancante.
- `179` righe non hanno CSV Top5 disponibile nel set usato dai mosaici: sono casi da trattare come coverage incompleta dell'esperimento o del salvataggio delle candidate.

## 1. Il detector puo mancare completamente il segnale diagnostico

Nel caso sotto, la GT contiene tre noduli maligni e la sintetica GT viene classificata correttamente come maligna. Le righe Top3/Top4/Top5/Top7 mostrano invece solo miss: le candidate sono molto caudali o su regioni non corrispondenti ai noduli GT. La Grad-CAM della Top5 e' saliente, ma su una sintetica guidata da regioni sbagliate; il risultato e' `ERR pred benign`.

![Caso con tre noduli maligni GT e zero hit detector Top5](sample_jpegs/gt_correct_top5_wrong/gt_correct_top5_wrong_0040.jpg)

Lettura: quando nessuna candidate intercetta la maschera GT, la superficie sintetica detector-driven non contiene il vincolo anatomico corretto. Il classificatore vede una immagine strutturata, ma scollegata dal nodulo che determina l'etichetta.

## 2. Nei benigni, miss e falsi positivi anatomici possono spingere verso maligno

Questo campione e' benigno. La GT ha un solo nodulo e Synth GT e' corretta. Top5, invece, contiene cinque detection tutte miss: alcune sono su slice estreme, altre su strutture pleuriche o regioni lontane dalla maschera. La Grad-CAM Top5 si concentra su una regione non-GT e produce `ERR pred malignant`.

![Falso positivo Top5 con detection tutte fuori dalla maschera GT](sample_jpegs/top5_false_positive/top5_false_positive_0006.jpg)

Lettura: il detector introduce falsi punti di controllo. La sintetica resta visivamente plausibile, ma enfatizza aree che non sono il nodulo annotato. Questo aumenta i falsi positivi del classificatore, soprattutto quando la heatmap si appoggia su strutture periferiche o artefatti anatomici.

## 3. Nei campioni multi-nodulo, Top5 spesso copre solo una parte della GT

Qui il campione ha cinque noduli GT benigni. Top5 intercetta solo due noduli e aggiunge tre miss. La classificazione Top5 diventa `ERR pred malignant`, nonostante Synth GT sia `OK pred benign`. Il problema non e' solo "hit si/no": e' la perdita della distribuzione completa dei noduli del campione.

![Campione multi-nodulo con copertura Top5 parziale](sample_jpegs/gt_correct_top5_wrong/gt_correct_top5_wrong_0088.jpg)

Lettura: per LUNA16, l'etichetta del campione puo dipendere dall'insieme dei noduli e dalla loro severita. Se la sintetica detector-driven conserva solo una parte dei noduli o ne sostituisce altri con miss, il classificatore riceve un riassunto anatomico diverso da quello della GT.

## 4. Il ranking del detector e' un collo di bottiglia

In questo caso maligno, Top5 sbaglia perche i due noduli GT non entrano tra i primi cinque candidati: sono recuperati solo in posizione 6 e 7 nella riga Top7. Infatti Top7 torna `OK pred malignant`, mentre Top5 e' `ERR pred benign`.

![Caso in cui Top7 recupera noduli non inclusi da Top5](sample_jpegs/gt_correct_top5_wrong/gt_correct_top5_wrong_0003.jpg)

Lettura: aumentare `top_k` puo recuperare noduli reali che il ranking mette troppo in basso. Questo spiega perche Top7 RBF migliora rispetto a Top3/Top4 in alcuni report, ma non risolve tutto: aggiunge anche candidate rumorose, quindi il beneficio dipende dal rapporto tra noduli recuperati e falsi positivi introdotti.

## 5. Coverage incompleta o candidate mancanti rendono l'esperimento meno stabile

Questo campione maligno ha nove noduli GT. Le righe Top3/Top4/Top5 non hanno CSV candidate nel set visualizzato, mentre Top7 ha sette candidate e cinque hit. La GT e Top7 classificano correttamente maligno; Top3/Top4/Top5 risultano errate o non informative.

![Campione multi-nodulo con CSV Top5 mancante e Top7 disponibile](sample_jpegs/gt_correct_top5_wrong/gt_correct_top5_wrong_0036.jpg)

Lettura: una parte del gap puo derivare anche da copertura incompleta delle candidate salvate per alcune configurazioni. Nei mosaici ci sono `179` righe con Top5 senza CSV. Questi casi vanno separati dai veri miss del detector, perche indicano un problema di disponibilita o allineamento degli output oltre alla qualita delle detection.

## Conclusione

La differenza GT vs detection non sembra dovuta a una sola causa. Dai mosaici emergono quattro meccanismi ricorrenti:

1. il detector manca completamente il nodulo rilevante;
2. le candidate false introducono regioni salienti ma non diagnostiche;
3. nei multi-nodulo, Top5 non rappresenta tutti i noduli del campione;
4. il ranking mette noduli veri oltre il cutoff Top5, oppure mancano CSV candidate per alcune configurazioni.

Questo spiega perche il classificatore GT ha performance piu alte: la sintetica GT e' vincolata direttamente alle annotazioni, mentre la sintetica detector-driven eredita errori di localizzazione, ranking e coverage del detector.
