from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sphere_diou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable 3D sphere IoU++/DIoU loss.

    `pred` and `target` are shaped [N, 4] as z, y, x, radius in feature-map units.
    """
    pred_center, pred_radius = pred[:, :3], pred[:, 3].clamp_min(eps)
    target_center, target_radius = target[:, :3], target[:, 3].clamp_min(eps)
    dist = torch.linalg.vector_norm(pred_center - target_center, dim=1).clamp_min(eps)
    no_overlap = dist >= (pred_radius + target_radius)
    pred_inside = (dist + pred_radius) <= target_radius
    target_inside = (dist + target_radius) <= pred_radius

    inter = torch.zeros_like(dist)
    small = torch.minimum(pred_radius, target_radius)
    large = torch.maximum(pred_radius, target_radius)
    inter = torch.where(pred_inside | target_inside, (4.0 / 3.0) * torch.pi * small.pow(3), inter)

    partial = ~(no_overlap | pred_inside | target_inside)
    if partial.any():
        pr = pred_radius[partial]
        tr = target_radius[partial]
        d = dist[partial]
        cos1 = ((tr.square() + d.square() - pr.square()) / (2 * tr * d)).clamp(-1 + eps, 1 - eps)
        cos2 = ((pr.square() + d.square() - tr.square()) / (2 * pr * d)).clamp(-1 + eps, 1 - eps)
        h1 = tr * (1 - cos1)
        h2 = pr * (1 - cos2)
        cap1 = torch.pi * h1.square() * (tr - h1 / 3)
        cap2 = torch.pi * h2.square() * (pr - h2 / 3)
        inter = inter.index_copy(0, partial.nonzero(as_tuple=False).flatten(), cap1 + cap2)

    vol_pred = (4.0 / 3.0) * torch.pi * pred_radius.pow(3)
    vol_target = (4.0 / 3.0) * torch.pi * target_radius.pow(3)
    union = (vol_pred + vol_target - inter).clamp_min(eps)
    siou = inter / union
    center_penalty = dist / (dist + pred_radius + target_radius).clamp_min(eps)
    return (1.0 - siou + center_penalty).mean()


class SCPMDetectionLoss(nn.Module):
    """Center-point matching loss for SCPM-Net outputs."""

    def __init__(
        self,
        stride: int = 2,
        positive_topk: int = 7,
        positive_radius_factor: float = 0.6,
        neg_pos_ratio: int = 20,
        focal_gamma: float = 2.0,
        refocal_threshold: float = 0.9,
        refocal_weight: float = 4.0,
        smooth_l1_beta: float = 1.0 / 9.0,
        lambda_cls: float = 1.0,
        lambda_radius: float = 1.0,
        lambda_offset: float = 1.0,
        lambda_siou: float = 1.0,
    ):
        super().__init__()
        self.stride = stride
        self.positive_topk = positive_topk
        self.positive_radius_factor = positive_radius_factor
        self.neg_pos_ratio = neg_pos_ratio
        self.focal_gamma = focal_gamma
        self.refocal_threshold = refocal_threshold
        self.refocal_weight = refocal_weight
        self.smooth_l1_beta = smooth_l1_beta
        self.lambda_cls = lambda_cls
        self.lambda_radius = lambda_radius
        self.lambda_offset = lambda_offset
        self.lambda_siou = lambda_siou

    def _targets(self, annotations: torch.Tensor, shape: torch.Size, device: torch.device) -> dict[str, torch.Tensor]:
        batch, _, depth, height, width = shape
        cls = torch.zeros((batch, 1, depth, height, width), device=device)
        valid = torch.zeros_like(cls, dtype=torch.bool)
        radius_t = torch.zeros_like(cls)
        offset_t = torch.zeros((batch, 3, depth, height, width), device=device)

        for b in range(batch):
            boxes = annotations[b]
            boxes = boxes[boxes[:, 3] > 0]
            for box in boxes:
                center = box[:3] / self.stride
                radius = (box[3] / self.stride).clamp_min(0.5)
                cz, cy, cx = center
                z0 = max(0, int(torch.floor(cz - radius * self.positive_radius_factor).item()))
                z1 = min(depth - 1, int(torch.ceil(cz + radius * self.positive_radius_factor).item()))
                y0 = max(0, int(torch.floor(cy - radius * self.positive_radius_factor).item()))
                y1 = min(height - 1, int(torch.ceil(cy + radius * self.positive_radius_factor).item()))
                x0 = max(0, int(torch.floor(cx - radius * self.positive_radius_factor).item()))
                x1 = min(width - 1, int(torch.ceil(cx + radius * self.positive_radius_factor).item()))
                zz, yy, xx = torch.meshgrid(
                    torch.arange(z0, z1 + 1, device=device),
                    torch.arange(y0, y1 + 1, device=device),
                    torch.arange(x0, x1 + 1, device=device),
                    indexing="ij",
                )
                points = torch.stack([zz, yy, xx], dim=-1).float().view(-1, 3)
                if points.numel() == 0:
                    points = torch.round(center).view(1, 3)
                dist = torch.linalg.vector_norm(points - center.view(1, 3), dim=1)
                inside = dist <= (radius * self.positive_radius_factor).clamp_min(1.0)
                points = points[inside] if inside.any() else points[dist.argmin()].view(1, 3)
                if points.size(0) > self.positive_topk:
                    points = points[torch.topk(-torch.linalg.vector_norm(points - center.view(1, 3), dim=1), self.positive_topk).indices]
                points = points.long()
                cls[b, 0, points[:, 0], points[:, 1], points[:, 2]] = 1.0
                valid[b, 0, points[:, 0], points[:, 1], points[:, 2]] = True
                radius_t[b, 0, points[:, 0], points[:, 1], points[:, 2]] = radius
                offset = center.view(3, 1) - points.float().T
                offset_t[b, :, points[:, 0], points[:, 1], points[:, 2]] = offset
        return {"cls": cls, "valid": valid, "radius": radius_t, "offset": offset_t}

    def _classification_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(target > 0, prob, 1 - prob)
        refocal = torch.ones_like(prob)
        hard_positive = (target > 0) & (prob < self.refocal_threshold)
        refocal = torch.where(hard_positive, torch.full_like(refocal, self.refocal_weight), refocal)
        focal = refocal * (1 - pt).pow(self.focal_gamma) * bce
        pos = focal[target > 0]
        neg = focal[target == 0]
        if pos.numel() == 0:
            return neg.topk(min(1024, neg.numel())).values.mean()
        neg_k = min(neg.numel(), max(pos.numel() * self.neg_pos_ratio, 1))
        return (pos.sum() + neg.topk(neg_k).values.sum()) / (pos.numel() + neg_k)

    def _single_head_loss(self, cls_logits: torch.Tensor, radius: torch.Tensor, offset: torch.Tensor, annotations: torch.Tensor):
        targets = self._targets(annotations, cls_logits.shape, cls_logits.device)
        cls_loss = self._classification_loss(cls_logits, targets["cls"])
        pos_mask = targets["valid"].expand_as(offset[:, :1])
        if not pos_mask.any():
            zero = cls_logits.sum() * 0
            return {"loss": cls_loss, "cls": cls_loss, "radius": zero, "offset": zero, "siou": zero}

        radius_loss = F.smooth_l1_loss(
            radius[targets["valid"]],
            targets["radius"][targets["valid"]],
            beta=self.smooth_l1_beta,
        )
        offset_loss = F.smooth_l1_loss(offset.permute(0, 2, 3, 4, 1)[targets["valid"].squeeze(1)], targets["offset"].permute(0, 2, 3, 4, 1)[targets["valid"].squeeze(1)])

        idx = targets["valid"].squeeze(1).nonzero(as_tuple=False)
        pred_offsets = offset.permute(0, 2, 3, 4, 1)[targets["valid"].squeeze(1)]
        pred_radius = radius[targets["valid"]].view(-1, 1)
        base = idx[:, 1:].float()
        pred_spheres = torch.cat([base + pred_offsets, pred_radius], dim=1)
        target_offsets = targets["offset"].permute(0, 2, 3, 4, 1)[targets["valid"].squeeze(1)]
        target_radius = targets["radius"][targets["valid"]].view(-1, 1)
        target_spheres = torch.cat([base + target_offsets, target_radius], dim=1)
        siou = sphere_diou_loss(pred_spheres, target_spheres)
        loss = self.lambda_cls * cls_loss + self.lambda_radius * radius_loss + self.lambda_offset * offset_loss + self.lambda_siou * siou
        return {"loss": loss, "cls": cls_loss, "radius": radius_loss, "offset": offset_loss, "siou": siou}

    def forward(self, outputs: dict[str, torch.Tensor], annotations: torch.Tensor) -> dict[str, torch.Tensor]:
        head1 = self._single_head_loss(outputs["Cls1"], outputs["Reg1"], outputs["Off1"], annotations)
        head2 = self._single_head_loss(outputs["Cls2"], outputs["Reg2"], outputs["Off2"], annotations)
        return {name: (head1[name] + head2[name]) * 0.5 for name in head1}
