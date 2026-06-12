import os
import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import defaultdict
import src.builder as builder
import src.utils as utils
import src.losses as custom_losses

class LungLitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = builder.build_model(cfg.model)
        self.criterion = builder.build_loss(cfg.model)
        
        # Segmentation loss and weight
        self.enable_segmentation = getattr(cfg.model, 'enable_segmentation', False)
        if self.enable_segmentation:
            self.pos_weight = torch.tensor([getattr(cfg.model, 'pos_weight', 1)])
            self.seg_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            self.seg_loss_weight = getattr(cfg.model, 'segmentation_loss_weight', 0.1)

        self.save_dir = cfg.exp.base_dir
        self.target_names = [""]

        self.step_outputs = defaultdict(lambda: defaultdict(list))
    
    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg.model, self.model)
        scheduler = builder.build_scheduler(self.cfg.model, optimizer)
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg.monitor.loss,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def shared_step(self, batch, split, extract_features=False):
        """Similar to traning step"""

        # Unpack batch - handle both with and without masks
        if len(batch) == 4:
            x, y, ids, masks = batch
            has_masks = True
        else:
            x, y, ids = batch
            has_masks = False
            masks = None
        
        # Forward pass
        if isinstance(self.model, builder.AFFNet):
            x = x.permute(0, 2, 1, 3, 4)
        
        # Check if we should supervise attention
        supervise_attention = getattr(self.cfg.model, 'supervise_attention', False)
        
        # Forward pass
        if supervise_attention:
            # Most models return (logit, attn_weights) or ((logit, seg), attn_weights)
            results = self.model(x, return_attn_weights=True)
            if isinstance(results, tuple):
                model_output, attn_weights = results
            else:
                model_output = results
                attn_weights = None
        else:
            model_output = self.model(x)
            attn_weights = None
                
        # Handle model output based on model type and segmentation
        if isinstance(self.model, builder.AFFNet):
            # AFFNet returns (logits, attention_related, multiview_related)
            logit = model_output[0]
            seg_output = None
        elif self.enable_segmentation and isinstance(model_output, tuple):
            logit, seg_output = model_output
        else:
            logit = model_output
            seg_output = None

        # Classification loss
        if isinstance(self.criterion, custom_losses.AFFNetLoss):
            # AFFNetLoss expects the full model output
            classification_loss = self.criterion(model_output, y)
        else:
            classification_loss = self.criterion(logit, y.view(-1, 1).float() if logit.shape[-1] == 1 else y.long())
        
        total_loss = classification_loss

        # Attention loss
        if supervise_attention and attn_weights is not None and has_masks:
            # Extract slice attention weights
            # Different models return different formats
            if isinstance(attn_weights, tuple):
                # SelfPatch_softSlice attention is (a_patches, a_slice)
                _, a_slice = attn_weights
            else:
                a_slice = attn_weights
            
            # a_slice shape should be (B, S, 1) or (B, S)
            if isinstance(a_slice, list):
                a_slice = a_slice[-1][:,0,1:]
            if len(a_slice.shape) == 3:
                a_slice = a_slice.squeeze(-1)
            
            b, s = a_slice.shape
            
            # Target: 1 if slice has nodule, 0 otherwise
            #has_nodule = (masks.sum(dim=(2, 3)) > 0).float() # (B, S)
            no_has_nodule = (masks.sum(dim=(2, 3)) == 0).float() # (B, S)
            
            # Compute loss per sample
            attn_loss_val = 0.0
            bs_count = 0
            for i in range(b):
                #if has_nodule[i].any():
                    # Target distribution is uniform over slices with nodules
                    # target = has_nodule[i] / (has_nodule[i].sum() + 1e-8)
                    
                    # Log-space for KLDiv
                    # pred_log = torch.log(a_slice[i].clamp(min=1e-8))
                    
                    # KL Divergence
                    # loss = F.kl_div(pred_log, target, reduction='sum')

                    # attn_loss_val += loss
                    # bs_count += 1

                neg_mask = no_has_nodule[i]
                # consider only samples that actually contain nodules
                if neg_mask.sum() < s:  # at least one positive slice exists

                    pred = a_slice[i]  # (S)

                    # normalize attention (optional but recommended)
                    # pred = pred / (pred.sum() + 1e-8)

                    # we want attention -> 0 on negative slices
                    loss = (pred * neg_mask).sum()

                    # alternative stronger penalty:
                    # loss = ((pred ** 2) * neg_mask).sum()

                    attn_loss_val += loss
                    bs_count += 1
            
            if bs_count > 0:
                attention_loss = attn_loss_val / bs_count
                attn_loss_weight = getattr(self.cfg.model, 'attention_loss_weight', 0.5)
                total_loss = total_loss + attn_loss_weight * attention_loss
                self.log(f"{split}/attention_loss", attention_loss, on_epoch=True, on_step=False, logger=True)

        # Segmentation loss (if enabled and masks available)
        if self.enable_segmentation and seg_output is not None and has_masks:
            # seg_output: (B, S, 1, H, W), masks: (B, S, H, W)
            b, s, _, h, w = seg_output.shape
            seg_output_flat = seg_output.view(b * s, 1, h, w)
            masks_flat = masks.view(b * s, h, w).float().unsqueeze(1)
            
            # Filter slices with nodules: only compute loss if mask.sum() > 0
            slice_mask_sums = masks_flat.sum(dim=(1, 2, 3))
            positive_slice_indices = slice_mask_sums > 0
            
            if positive_slice_indices.any():
                seg_output_nodules = seg_output_flat[positive_slice_indices]
                masks_nodules = masks_flat[positive_slice_indices]
                
                # BCE loss
                seg_bce_loss = self.seg_criterion(seg_output_nodules, masks_nodules)
                
                # Dice loss
                seg_probs_nodules = torch.sigmoid(seg_output_nodules)
                dice_loss = self._dice_loss(seg_probs_nodules, masks_nodules)
                
                # Combined segmentation loss
                seg_loss = seg_bce_loss + 1.5 * dice_loss
                
                # Total loss with weighting
                total_loss = classification_loss + self.seg_loss_weight * seg_loss
                
                # Log segmentation losses
                self.log(f"{split}/seg_bce_loss", seg_bce_loss, on_epoch=True, on_step=False, logger=True)
                self.log(f"{split}/seg_dice_loss", dice_loss, on_epoch=True, on_step=False, logger=True)
                self.log(f"{split}/seg_loss", seg_loss, on_epoch=True, on_step=False, logger=True)
                
                # Compute and log Dice score (metric)
                dice_score = self._dice_score(seg_probs_nodules, masks_nodules)
                self.log(f"{split}/dice_score", dice_score, on_epoch=True, on_step=False, logger=True, prog_bar=True)
            else:
                # No nodules in this batch's slices
                pass

        # Log classification loss and total loss
        self.log(f"{split}/classification_loss", classification_loss, on_epoch=True, on_step=False, logger=True)
        self.log(
            f"{split}/loss",
            total_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )

        self.step_outputs[split]["logit"].append(logit.detach().cpu())
        self.step_outputs[split]["y"].append(y.detach().cpu())
        self.step_outputs[split]["ids"].append(ids)

        return total_loss
    
    def _dice_loss(self, pred, target, smooth=1e-5):
        """Compute Dice loss for segmentation"""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def _dice_score(self, pred, target, threshold=0.5, smooth=1e-5):
        """Compute Dice score (metric) for segmentation"""
        pred_binary = (pred > threshold).float()
        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    def shared_epoch_end(self, split):
        y = torch.cat(self.step_outputs[split]["y"], dim=0)
        logit = torch.cat(self.step_outputs[split]["logit"], dim=0)
        prob = torch.sigmoid(logit)

        if split == "test":
            config_out_dir = os.path.join(self.save_dir, "config.pkl")
            pickle.dump(self.cfg, open(config_out_dir, "wb"))

            out_dir = os.path.join(self.save_dir, "test_preds.csv")
            all_p = prob.cpu().detach().tolist()
            all_label = y.cpu().detach().tolist()
            all_ids = [f for x in self.step_outputs[split]["ids"] for f in x]
            outfile = defaultdict(list)
            for pid, label, p in zip(all_ids, all_label, all_p):
                patient_id = pid
                outfile["patient_id"].append(patient_id)
                outfile["label"].append(label)
                outfile["prob"].append(p)

            df = pd.DataFrame.from_dict(outfile)
            df.to_csv(out_dir, index=False)
            print("=" * 80)
            print(f"Config saved at: {out_dir}")
            print(f"Predictions saved at: {out_dir}")
            print("=" * 80)

        # log auroc
        auroc_dict = utils.get_auroc(y, prob, self.target_names)
        for k, v in auroc_dict.items():
            self.log(f"{split}/{k}_auroc", v, on_epoch=True, on_step=False, logger=True, prog_bar=True)

        # log auprc
        auprc_dict = utils.get_auprc(y, prob, self.target_names)
        for k, v in auprc_dict.items():
            self.log(f"{split}/{k}_auprc", v, on_epoch=True, on_step=False, logger=True, prog_bar=True)

        # log mcc
        mcc_dict = utils.get_mcc(y, prob, self.target_names)
        for k, v in mcc_dict.items():
            self.log(f"{split}/{k}_mcc", v, on_epoch=True, on_step=False, logger=True, prog_bar=True)

        # log acc
        acc_dict = utils.get_acc(y, prob, self.target_names)
        for k, v in acc_dict.items():
            self.log(f"{split}/{k}_acc", v, on_epoch=True, on_step=False, logger=True, prog_bar=True)
        
        # self.step_outputs = defaultdict(lambda: defaultdict(list))
        del self.step_outputs[split]

class DetectionLitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = builder.build_model(cfg.model)
        # We will use compute_rpn_loss and compute_volume_loss instead of custom_losses.CIoULoss or FocalLoss
        self.save_dir = cfg.exp.base_dir
        self.step_outputs = defaultdict(lambda: defaultdict(list))

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg.model, self.model)
        scheduler = builder.build_scheduler(self.cfg.model, optimizer)
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg.monitor.loss,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        self.shared_epoch_end("val")

    def on_test_epoch_end(self):
        self.shared_epoch_end("test")

    def shared_step(self, batch, split):
        # Unpack batch
        if len(batch) == 4:
            x, y, ids, masks = batch
        else:
            raise ValueError("Detection requires masks to extract bounding boxes")

        # x is (B, D, 1, H, W) or (B, 1, H, W)
        b, d, c, h, w = x.shape

        masks = masks.view(b * d, h, w)
            
        gt_bboxes = []
        for m in masks:
            gt_bboxes.append(utils.mask_to_bbox(m))
        gt_bboxes = torch.from_numpy(np.array(gt_bboxes)).float().to(x.device)
        
        # We need to build gt_boxes_list which is a list of tensors [N_gt, 6] for each volume in the batch
        # Currently, masks are flattened over B*D.
        # gt_bboxes is [B*D, 4] where format is (x_min, y_min, w, h).
        # We need format (z, y, x, dz, dy, dx).
        
        gt_bboxes = gt_bboxes.view(b, d, 4) # [B, D, 4]
        gt_boxes_list = []
        
        for i in range(b):
            vol_gts = []
            for z_idx in range(d):
                # Check if slice has a valid box (e.g. w > 0 and h > 0)
                # In utils.mask_to_bbox, it usually returns (0,0,0,0) if mask is empty
                box_2d = gt_bboxes[i, z_idx]
                if box_2d[2] > 0 and box_2d[3] > 0:
                    x_min, y_min, w_box, h_box = box_2d
                    
                    # Convert 2D box to a 1-slice thick 3D box: [z, y, x, dz, dy, dx]
                    # This is an approximation since bounding box is 2D slice by slice.
                    # A better way would be 3D connected components on the mask,
                    # but for this script we will treat each 2D box as a 3D box of depth 1.
                    z_center = z_idx + 0.5
                    y_center = y_min + h_box / 2.0
                    x_center = x_min + w_box / 2.0
                    
                    vol_gts.append([z_center, y_center, x_center, 1.0, h_box, w_box])
                    
            if len(vol_gts) > 0:
                gt_boxes_list.append(torch.tensor(vol_gts, dtype=torch.float32, device=x.device))
            else:
                gt_boxes_list.append(torch.empty((0, 6), dtype=torch.float32, device=x.device))

        # Forward del modello
        # L'input x è (B, D, 1, H, W). Il modello 3D MNI SwinUNETR si aspetta (B, C, D, H, W)
        # Permutiamo x prima di passarlo
        x_3d = x.permute(0, 2, 1, 3, 4) if x.dim() == 5 else x
        proposals, cls_logits, reg_deltas, anchors, final_score = self.model(x_3d)
        
        # Map each prediction back to its patient ID
        for i in range(b):
            self.step_outputs[split]["ids"].extend([ids[i]] * d)

        # Import losses
        from src.loss import compute_rpn_loss, compute_volume_loss

        # Compute RPN Loss (Focal Loss and GIoU L1)
        rpn_cls_loss, rpn_reg_loss = compute_rpn_loss(
            cls_logits, 
            reg_deltas, 
            anchors, 
            gt_boxes_list,
            pos_iou_thresh=0.5,
            neg_iou_thresh=0.1
        )
        
        # Volume classification label is y. It needs to be shaped as (B, 1)
        y_label = y.view(-1, 1).float()
        
        # Compute volume loss (BCE)
        volume_loss = compute_volume_loss(final_score, y_label)
        
        # Compute total loss
        loss = rpn_cls_loss + rpn_reg_loss + volume_loss

        self.log(f"{split}/loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{split}/rpn_cls_loss", rpn_cls_loss, on_epoch=True, prog_bar=True)
        self.log(f"{split}/rpn_reg_loss", rpn_reg_loss, on_epoch=True, prog_bar=True)
        self.log(f"{split}/volume_loss", volume_loss, on_epoch=True, prog_bar=True)
        
        return loss

        # else:
        #     # 2D slice
        #     gt_bboxes = []
        #     for m in masks:
        #         gt_bboxes.append(utils.mask_to_bbox(m))
        #     gt_bboxes = torch.from_numpy(np.array(gt_bboxes)).float().to(x.device)
            
        #     bbox_preds, class_preds, volume_preds = self.model(x)
        #     # bbox_preds: (B*D, 4)
        #     # class_preds: (B*D, 1)
        #     # volume_preds: (B, 1)

        #     loss = self.criterion(bbox_preds, gt_bboxes)
            
        #     self.step_outputs[split]["class_preds"].extend(class_preds.detach().cpu().numpy())
        #     self.step_outputs[split]["volume_preds"].extend(class_preds.detach().cpu().numpy())
        #     self.step_outputs[split]["bbox_preds"].extend(bbox_preds.detach().cpu().numpy())
        #     self.step_outputs[split]["gt_bboxes"].extend(gt_bboxes.detach().cpu().numpy())
            
        #     self.log(f"{split}/loss", loss, on_epoch=True, prog_bar=True)
            
        #     return loss
    
    def shared_epoch_end(self, split):
        # log map (mean average precision)
        map_score = utils.calculate_map(self.step_outputs[split]["preds"], self.step_outputs[split]["gts"])
        self.log(f"{split}/map", map_score, on_epoch=True, prog_bar=True)

        # log sensitivity
        sensitivity = utils.calculate_sensitivity(self.step_outputs[split]["preds"], self.step_outputs[split]["gts"])
        self.log(f"{split}/sensitivity", sensitivity, on_epoch=True, prog_bar=True)

        # log specificity
        specificity = utils.calculate_specificity(self.step_outputs[split]["preds"], self.step_outputs[split]["gts"])
        self.log(f"{split}/specificity", specificity, on_epoch=True, prog_bar=True)

        # log accuracy
        accuracy = utils.calculate_accuracy(self.step_outputs[split]["preds"], self.step_outputs[split]["gts"])
        self.log(f"{split}/accuracy", accuracy, on_epoch=True, prog_bar=True)

        # log F1
        F1 = utils.calculate_F1(self.step_outputs[split]["preds"], self.step_outputs[split]["gts"])
        self.log(f"{split}/F1", F1, on_epoch=True, prog_bar=True)

        # log FROC
        FROC = utils.calculate_FROC(self.step_outputs[split]["preds"], self.step_outputs[split]["gts"])
        self.log(f"{split}/FROC", FROC, on_epoch=True, prog_bar=True)

        if split == "test":
            # Save results to CSV
            output_dir = os.path.join(self.save_dir, "detection_results.csv")
            results = []
            
            # Since we only track preds and gts for positive slices, we'll calculate individual IoUs here
            preds = self.step_outputs[split]["bbox_preds"]
            gts = self.step_outputs[split]["gt_bboxes"]
            ids = self.step_outputs[split]["ids"]
            
            for i in range(len(preds)):
                p_id = ids[i] if i < len(ids) else "unknown"
                iou = utils.calculate_iou(preds[i], gts[i])
                results.append({"patient_id": p_id, "iou": iou})
                
            df = pd.DataFrame(results)
            df.to_csv(output_dir, index=False)
            print("=" * 80)
            print(f"Detection results saved at: {output_dir}")
            print(f"Mean Test IoU: {df['iou'].mean():.4f}" if not df.empty else "Mean Test IoU: N/A")
            print("=" * 80)

            # Save a few visualizations
            vis_dir = os.path.join(self.save_dir, "test_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            for i in range(5):
                for j in range(b): # Save slices of the first sample
                    img_np = x[j, i, 0].detach().cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    utils.draw_bboxes_on_slice(
                        img_np,
                        bbox_preds[i].detach().cpu().numpy(),
                        gt_bboxes[i].detach().cpu().numpy(),
                        os.path.join(vis_dir, f"{ids[i]}_slice{j}_vis.png")
                    )
            
        self.step_outputs[split].clear()
        del self.step_outputs[split]

# import hydra
# @hydra.main(config_path="conf", config_name="config", version_base=None)
# def main(cfg):
#     from types import SimpleNamespace
    
#     # Initialize the model
#     model = LungLitModel(cfg)
#     model.target_names = ["class_0", "class_1"] # Required for metrics
#     model.step_outputs = {"train": {"logit": [], "y": [], "ids": []}, 
#                           "val": {"logit": [], "y": [], "ids": []}, 
#                           "test": {"logit": [], "y": [], "ids": []}}

#     # Initialize trainer and start training
#     trainer = pl.Trainer(max_epochs=10, accelerator="auto")
#     # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
#     # trainer.test(model)

# if __name__ == "__main__":
#     main()