import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

def box_giou_3d(boxes1, boxes2):
    """
    Calcola il Generalized IoU (GIoU) 3D tra due set di bounding box.
    boxes1: [N, 6] -> [z, y, x, dz, dy, dx] (Predicted boxes)
    boxes2: [M, 6] -> [z, y, x, dz, dy, dx] (Ground Truth boxes)
    Ritorna:
        giou: tensore [N, M] con i GIoU tra tutti i pair di box
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
        
    z1, y1, x1, dz1, dy1, dx1 = boxes1.unbind(dim=-1)
    z1_max = z1 + dz1
    y1_max = y1 + dy1
    x1_max = x1 + dx1

    z2, y2, x2, dz2, dy2, dx2 = boxes2.unbind(dim=-1)
    z2_max = z2 + dz2
    y2_max = y2 + dy2
    x2_max = x2 + dx2

    z_inter_min = torch.max(z1.unsqueeze(1), z2.unsqueeze(0))
    y_inter_min = torch.max(y1.unsqueeze(1), y2.unsqueeze(0))
    x_inter_min = torch.max(x1.unsqueeze(1), x2.unsqueeze(0))

    z_inter_max = torch.min(z1_max.unsqueeze(1), z2_max.unsqueeze(0))
    y_inter_max = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))
    x_inter_max = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))

    inter_vol = ((z_inter_max - z_inter_min).clamp(min=0) *
                 (y_inter_max - y_inter_min).clamp(min=0) *
                 (x_inter_max - x_inter_min).clamp(min=0))

    vol1 = (dz1 * dy1 * dx1).unsqueeze(1)
    vol2 = (dz2 * dy2 * dx2).unsqueeze(0)

    union = vol1 + vol2 - inter_vol
    iou = inter_vol / union.clamp(min=1e-6)

    # Calcolo coordinate bounding box più piccolo che contiene entrambi
    z_enclose_min = torch.min(z1.unsqueeze(1), z2.unsqueeze(0))
    y_enclose_min = torch.min(y1.unsqueeze(1), y2.unsqueeze(0))
    x_enclose_min = torch.min(x1.unsqueeze(1), x2.unsqueeze(0))

    z_enclose_max = torch.max(z1_max.unsqueeze(1), z2_max.unsqueeze(0))
    y_enclose_max = torch.max(y1_max.unsqueeze(1), y2_max.unsqueeze(0))
    x_enclose_max = torch.max(x1_max.unsqueeze(1), x2_max.unsqueeze(0))

    enclose_vol = ((z_enclose_max - z_enclose_min).clamp(min=0) *
                   (y_enclose_max - y_enclose_min).clamp(min=0) *
                   (x_enclose_max - x_enclose_min).clamp(min=0))

    giou = iou - (enclose_vol - union) / enclose_vol.clamp(min=1e-6)

    return giou

def box_iou_3d(boxes1, boxes2):
    """
    Calcola l'IoU approssimativo (cubico) tra due set di bounding box 3D.
    boxes1: [N, 6] -> [z, y, x, dz, dy, dx]
    boxes2: [M, 6] -> [z, y, x, dz, dy, dx]
    Ritorna:
        iou: tensore [N, M] con gli IoU tra tutti i pair di box
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
        
    z1, y1, x1, dz1, dy1, dx1 = boxes1.unbind(dim=-1)
    z1_max = z1 + dz1
    y1_max = y1 + dy1
    x1_max = x1 + dx1

    z2, y2, x2, dz2, dy2, dx2 = boxes2.unbind(dim=-1)
    z2_max = z2 + dz2
    y2_max = y2 + dy2
    x2_max = x2 + dx2

    # Calcolo coordinate intersezione [N, M]
    z_inter_min = torch.max(z1.unsqueeze(1), z2.unsqueeze(0))
    y_inter_min = torch.max(y1.unsqueeze(1), y2.unsqueeze(0))
    x_inter_min = torch.max(x1.unsqueeze(1), x2.unsqueeze(0))

    z_inter_max = torch.min(z1_max.unsqueeze(1), z2_max.unsqueeze(0))
    y_inter_max = torch.min(y1_max.unsqueeze(1), y2_max.unsqueeze(0))
    x_inter_max = torch.min(x1_max.unsqueeze(1), x2_max.unsqueeze(0))

    # Volume di intersezione
    inter_vol = ((z_inter_max - z_inter_min).clamp(min=0) *
                 (y_inter_max - y_inter_min).clamp(min=0) *
                 (x_inter_max - x_inter_min).clamp(min=0))

    # Volumi dei singoli box
    vol1 = (dz1 * dy1 * dx1).unsqueeze(1) # [N, 1]
    vol2 = (dz2 * dy2 * dx2).unsqueeze(0) # [1, M]

    # IoU
    union = vol1 + vol2 - inter_vol
    iou = inter_vol / union.clamp(min=1e-6)

    return iou

def encode_boxes_3d(anchors, gt_boxes):
    """
    Calcola i target per la regressione dai ground truth box.
    anchors: [N, 6]
    gt_boxes: [N, 6]
    Ritorna: deltas: [N, 6]
    """
    z_a, y_a, x_a, dz_a, dy_a, dx_a = anchors.unbind(dim=-1)
    z_g, y_g, x_g, dz_g, dy_g, dx_g = gt_boxes.unbind(dim=-1)

    dz = (z_g - z_a) / dz_a
    dy = (y_g - y_a) / dy_a
    dx = (x_g - x_a) / dx_a
    
    ddz = torch.log(dz_g / dz_a)
    ddy = torch.log(dy_g / dy_a)
    ddx = torch.log(dx_g / dx_a)

    return torch.stack([dz, dy, dx, ddz, ddy, ddx], dim=-1)

def apply_deltas_to_anchors(anchors, deltas):
    """
    Decodifica le delte preziose per ottenere i bounding box assoluti.
    anchors: [N, 6] -> [z, y, x, dz, dy, dx]
    deltas:  [N, 6] -> [Δz, Δy, Δx, Δdz, Δdy, Δdx]
    """
    z_a, y_a, x_a, dz_a, dy_a, dx_a = anchors.unbind(dim=-1)
    dz, dy, dx, ddz, ddy, ddx = deltas.unbind(dim=-1)

    z_pred = z_a + dz * dz_a
    y_pred = y_a + dy * dy_a
    x_pred = x_a + dx * dx_a

    dz_pred = dz_a * torch.exp(ddz)
    dy_pred = dy_a * torch.exp(ddy)
    dx_pred = dx_a * torch.exp(ddx)

    return torch.stack([z_pred, y_pred, x_pred, dz_pred, dy_pred, dx_pred], dim=-1)

def compute_rpn_loss(cls_logits, reg_deltas, anchors, gt_boxes_list, pos_iou_thresh=0.5, neg_iou_thresh=0.1):
    """
    Calcola Focal Loss e Regression Loss.
    cls_logits: [B, num_total_anchors, 1]
    reg_deltas: [B, num_total_anchors, 6]
    anchors:    [B, num_total_anchors, 6]
    gt_boxes_list: Lista di lunghezza B, ogni elemento è un tensore [num_gts, 6] con i box reali
    
    Ritorna:
        cls_loss (tensor scalare)
        reg_loss (tensor scalare)
    """
    B = cls_logits.shape[0]
    num_total_anchors = cls_logits.shape[1]
    
    cls_loss_total = 0.0
    reg_loss_total = 0.0
    num_pos_total = 0

    for b in range(B):
        logits_b = cls_logits[b].squeeze(-1) # [N_anchors]
        deltas_b = reg_deltas[b]             # [N_anchors, 6]
        anchors_b = anchors[b]               # [N_anchors, 6]
        gt_b = gt_boxes_list[b]              # [N_gt, 6]

        if len(gt_b) == 0:
            # Nessun nodulo in questo volume
            targets_cls = torch.zeros_like(logits_b)
            # Solo Negative ancore, usiamo focal loss verso 0
            cls_loss_total += sigmoid_focal_loss(logits_b, targets_cls, reduction="sum")
            continue

        # 1. Match anchors con Ground Truth
        ious = box_iou_3d(anchors_b, gt_b) # [N_anchors, N_gt]
        max_iou, argmax_iou = ious.max(dim=1) # Per ogni ancora, il miglior GT

        # 2. Definisci label positive/negative
        pos_mask = max_iou >= pos_iou_thresh
        neg_mask = max_iou < neg_iou_thresh
        
        # Ignoriamo le ancore nel mezzo (neg_iou_thresh <= max_iou < pos_iou_thresh)
        valid_mask = pos_mask | neg_mask
        
        # Se non ci sono ancore positive ma c'è un GT, forziamo la miglior ancora globale per ogni GT a essere positiva
        # (opzionale, ma utile per piccoli oggetti in 3D)
        best_anchor_per_gt = ious.argmax(dim=0)
        pos_mask[best_anchor_per_gt] = True
        valid_mask[best_anchor_per_gt] = True
        
        # 3. Label target (classification)
        targets_cls = torch.zeros_like(logits_b)
        targets_cls[pos_mask] = 1.0

        # Calcolo Focal Loss per le ancore valide
        cls_loss = sigmoid_focal_loss(logits_b[valid_mask], targets_cls[valid_mask], reduction="sum")
        cls_loss_total += cls_loss

        # 4. Target deltas (regression)
        num_pos = pos_mask.sum().item()
        num_pos_total += num_pos
        
        if num_pos > 0:
            pos_anchors = anchors_b[pos_mask]
            # Associa ad ogni box l'indice del ground truth best-matching
            pos_gt_idx = argmax_iou[pos_mask]
            
            # Per le ancore forzate, dobbiamo fixare l'argmax
            # per semplicità, se una di esse era sotto soglia,
            # ci assicuriamo venga accoppiato al corretto GT.
            gt_labels_for_pos = gt_b[pos_gt_idx]

            pos_deltas = deltas_b[pos_mask]

            # Decodifica le delte per ottenere le ancore predette dai logits
            pred_boxes = apply_deltas_to_anchors(pos_anchors, pos_deltas)

            # Calcolo GIoU tra i box predetti e il GT assegnato
            # pred_boxes: [num_pos, 6], gt_labels_for_pos: [num_pos, 6]
            giou = box_giou_3d(pred_boxes, gt_labels_for_pos) # [num_pos, num_pos]
            
            # Dal momento che vogliamo il GIoU tra la singola ancor_positiva e il suo GT accoppiato 
            # consideriamo la diagonale della matrice
            giou_diag = giou.diag() # [num_pos]

            # La GIoU Loss è definita come 1 - GIoU
            reg_loss = (1.0 - giou_diag).sum()
            reg_loss_total += reg_loss

    # Normalize per il total numero di patch o ancore usate
    N_anchors_tot = B * num_total_anchors
    cls_loss_total = cls_loss_total / N_anchors_tot
    reg_loss_total = reg_loss_total / max(1, num_pos_total)

    return cls_loss_total, reg_loss_total

def compute_volume_loss(final_scores, gt_labels):
    """
    Calcola BCE Loss globale per ogni volume nel batch.
    final_scores: [B, 1] logits di output dal MILHead
    gt_labels:    [B, 1] etichette reali (es 1 se c'è almeno un nodulo nel volume, 0 altrimenti)
    """
    return F.binary_cross_entropy_with_logits(final_scores, gt_labels.float())
