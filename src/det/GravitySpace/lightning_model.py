import os
import sys
import time
import torch
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.ops import nms

# Ensure current directory is in path so we can import modules
sys.path.append(os.getcwd())

from src.det.GravitySpace.model.GravitySpaceAttentionNet import GravitySpaceAttentionNet
from src.det.GravitySpace.loss.GravityLoss import GravityLoss
from src.det.GravitySpace.anchors.gravity_points_config import gravity_points_config
from src.det.GravitySpace.evaluation.FROC import FROC
from src.det.GravitySpace.evaluation.FROC_plot import FROC_plot
from src.det.GravitySpace.evaluation.FROC_linear_plot import FROC_linear_plot

class GravitySpaceLitModel(pl.LightningModule):
    def __init__(self, 
                 backbone="ResNet-18",
                 pretrained=True,
                 attention="enhanced",
                 window_size=5,
                 sampling=5,
                 hidden_dim=512,
                 alpha=0.25,
                 gamma=2.0,
                 hook=10,
                 hook_gap=5,
                 anchor_config="grid-10",
                 image_shape=(352, 480),
                 lr=1e-4,
                 base_dir="outputs",
                 exp_name="gravity_space_detection",
                 distance=8,
                 nms_radius=8,
                 nms_z_radius=1,
                 nms_2d_iou_threshold=0.5,
                 nms_3d_iou_threshold=0.1,
                 score_threshold=0.0,
                 chunk_size=None,
                 input_channels=3,
                 save_qualitative=False,
                 qualitative_max_images=10,
                 qualitative_dir=None,
                 qualitative_only_with_findings=True,
                 qualitative_score_threshold=None,
                 qualitative_show_fp_text=False,
                 froc_normalization="slice"):
        super().__init__()
        self.save_hyperparameters()
        self.save_dir = os.path.join(base_dir, exp_name)
        self.test_step_outputs = []
        self.score_threshold = score_threshold
        self.save_qualitative = save_qualitative
        self.qualitative_max_images = qualitative_max_images
        self.qualitative_dir = qualitative_dir
        self.qualitative_only_with_findings = qualitative_only_with_findings
        self.qualitative_score_threshold = qualitative_score_threshold
        self.qualitative_show_fp_text = qualitative_show_fp_text
        self.froc_normalization = froc_normalization
        self._qualitative_saved = 0
        
        # --------------------- #
        # Gravity Points Config #
        # --------------------- #
        image_shape_np = np.array(image_shape)
        gp, gp_init, fm_shape = gravity_points_config(config=anchor_config,
                                                      image_shape=image_shape_np,
                                                      device=self.device)
        
        # Register gravity points as a buffer (no gradient)
        self.register_buffer("gravity_points", gp)
        
        num_gp_per_pixel = len(gp_init)
        self.fm_shape = fm_shape
        
        # -------------- #
        # Backbone Model #
        # -------------- #
        self.model = GravitySpaceAttentionNet(
            backbone=backbone,
            pretrained=pretrained,
            attention=attention,
            num_gravity_points_feature_map=num_gp_per_pixel,
            feature_map_shape=fm_shape,
            window_size=window_size,
            sampling=sampling,
            hidden_dim=hidden_dim,
            chunk_size=chunk_size,
            input_channels=input_channels
        )
        
        # ---------- #
        # Loss Model #
        # ---------- #
        self.criterion = GravityLoss(
            alpha=alpha,
            gamma=gamma,
            config=anchor_config,
            hook=hook,
            hook_gap=hook_gap,
            num_gravity_points_feature_map=num_gp_per_pixel,
            device=self.device,
            debug=False
        )

    def _reset_gravity_points_from_hparams(self):
        image_shape_np = np.array(self.hparams.image_shape)
        gp, _, _ = gravity_points_config(
            config=self.hparams.anchor_config,
            image_shape=image_shape_np,
            device=self.device,
        )
        self.gravity_points = gp.to(self.device)

    def _save_gravity_points_plot(
        self,
        image,
        centers_xy,
        scores,
        annotations,
        filename,
        distance,
        image_shape,
    ):
        if not self.save_qualitative:
            return
        if self._qualitative_saved >= self.qualitative_max_images:
            return

        print(f"Saving qualitative plot for {filename} with {len(centers_xy)} candidates and {torch.sum(annotations[:, 0] != -1).item()} GT nodules")
        gt = annotations[annotations[:, 0] != -1]
        if self.qualitative_only_with_findings and gt.numel() == 0:
            return

        configured_h, configured_w = image_shape
        threshold = self.qualitative_score_threshold
        if threshold is None:
            threshold = self.score_threshold

        if image.dim() == 3:
            image = image[image.shape[0] // 2]
        image_np = image.detach().cpu().float().numpy()
        actual_h, actual_w = image_np.shape
        shape_mismatch = (actual_h, actual_w) != (configured_h, configured_w)
        H, W = configured_h, configured_w
        if shape_mismatch:
            print(
                f"Warning: qualitative plot image shape {(actual_h, actual_w)} "
                f"does not match configured image_shape {(configured_h, configured_w)} for {filename}"
            )

        centers_xy = centers_xy.float().cpu()
        scores = scores.float().cpu()
        gt_xy = gt[:, :2].float().cpu()

        visible = (
            (centers_xy[:, 0] >= 0) & (centers_xy[:, 0] < W) &
            (centers_xy[:, 1] >= 0) & (centers_xy[:, 1] < H)
        )

        is_tp = torch.zeros(len(centers_xy), dtype=torch.bool)
        if gt_xy.numel() > 0 and centers_xy.numel() > 0:
            distances = torch.cdist(centers_xy, gt_xy)
            is_tp = torch.any(distances <= distance, dim=1)

        is_fp = (scores >= threshold) & ~is_tp & visible
        is_other = (scores < threshold) & ~is_tp & visible
        is_tp = is_tp & visible
        is_outside = ~visible

        out_dir = self.qualitative_dir or os.path.join(self.save_dir, "qualitative_gravity_points")
        os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(str(filename)))[0]
        out_path = os.path.join(out_dir, f"{self._qualitative_saved:04d}_{base_name}_gravity_points.png")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_np, cmap="gray")

        def scatter(mask, color, marker, label, size=12, alpha=0.75):
            if torch.any(mask):
                pts = centers_xy[mask]
                ax.scatter(
                    pts[:, 0].numpy(),
                    pts[:, 1].numpy(),
                    s=size,
                    c=color,
                    marker=marker,
                    linewidths=0.4,
                    edgecolors="black",
                    alpha=alpha,
                    label=f"{label} ({int(mask.sum().item())})",
                )

        scatter(is_other, "#9ca3af", ".", "Other", size=8, alpha=0.45)
        scatter(is_fp, "#ef4444", "x", "FP", size=20, alpha=0.9)
        scatter(is_tp, "#22c55e", "o", "TP", size=24, alpha=0.9)

        if torch.any(is_outside):
            clipped = centers_xy[is_outside].clone()
            clipped[:, 0] = clipped[:, 0].clamp(0, W - 1)
            clipped[:, 1] = clipped[:, 1].clamp(0, H - 1)
            ax.scatter(
                clipped[:, 0].numpy(),
                clipped[:, 1].numpy(),
                s=18,
                c="#f59e0b",
                marker="^",
                linewidths=0.4,
                edgecolors="black",
                alpha=0.8,
                label=f"Outside, clipped ({int(is_outside.sum().item())})",
            )

        if gt_xy.numel() > 0:
            ax.scatter(
                gt_xy[:, 0].numpy(),
                gt_xy[:, 1].numpy(),
                s=90,
                facecolors="none",
                edgecolors="#38bdf8",
                marker="s",
                linewidths=1.5,
                label=f"GT ({len(gt_xy)})",
            )

        if self.qualitative_show_fp_text and torch.any(is_fp):
            for idx in torch.nonzero(is_fp, as_tuple=False).flatten():
                x, y = centers_xy[idx]
                ax.text(float(x), float(y), f"{float(scores[idx]):.2f}", color="#ef4444", fontsize=6)

        title = f"{filename}\nAll candidates: {len(centers_xy)} | threshold: {threshold:.3f}"
        if shape_mismatch:
            title += f" | image shape mismatch: actual {(actual_h, actual_w)}, cfg {(configured_h, configured_w)}"
        ax.set_title(title)
        ax.set_xlim(0, W - 1)
        ax.set_ylim(H - 1, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        self._qualitative_saved += 1

    def _froc_units(self):
        normalization = self.froc_normalization
        if isinstance(normalization, str):
            normalization = normalization.lower()
            if normalization == "both":
                return ("slice", "scan")
            return (normalization,)

        units = tuple(str(unit).lower() for unit in normalization)
        if not units:
            return ("slice",)
        return units

    def _save_single_froc_output(self, unit, detections_tp_fp, total_annotations, denominator):
        os.makedirs(self.save_dir, exist_ok=True)

        if denominator <= 0:
            print(f"Skipping {unit}-normalized FROC: no evaluated {unit}s.")
            return
        if total_annotations <= 0:
            print(f"Skipping {unit}-normalized FROC: no annotations.")
            return

        FPS, sens = FROC(
            detections=detections_tp_fp,
            TotalNumOfImages=denominator,
            TotalNumOfAnnotations=total_annotations,
            debug=False
        )

        suffix = "test" if unit == "slice" else f"test_per_{unit}"
        label_x = "slices" if unit == "slice" else "scans"
        title = "FROC Test" if unit == "slice" else "FROC Test Per Scan"
        linear_title = "FROC Linear Test" if unit == "slice" else "FROC Linear Test Per Scan"
        metric_prefix = "test" if unit == "slice" else f"test/per_{unit}"

        coords_path = os.path.join(self.save_dir, f"FROC_Coords_{suffix}.csv")
        pd.DataFrame({"FPS": FPS, "sensitivity": sens}).to_csv(coords_path, index=False)

        froc_path = os.path.join(self.save_dir, f"FROC_{suffix}.png")
        FROC_plot(
            title=title,
            color="blue",
            label_x=label_x,
            experiment_ID=self.hparams.exp_name,
            FPS=FPS,
            sens=sens,
            FROC_path=froc_path,
            FROC_coords_path=coords_path
        )

        FROC_linear_plot(
            title=linear_title,
            color="blue",
            label_x=label_x,
            experiment_ID=self.hparams.exp_name,
            FPS=FPS,
            sens=sens,
            FROC_upper_limit=10,
            FROC_path=os.path.join(self.save_dir, f"FROC_Linear_{suffix}.png")
        )

        print(f"Saved {unit}-normalized FROC to {froc_path}")

        if self.logger:
            try:
                import wandb
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    wandb_key = "test/FROC" if unit == "slice" else f"test/FROC_per_{unit}"
                    self.logger.log_image(key=wandb_key, images=[froc_path])
            except ImportError:
                pass

        for target_fps in [0.1, 0.5, 1.0, 2.0, 4.0]:
            idx = np.searchsorted(FPS, target_fps)
            if idx < len(sens):
                self.log(f"{metric_prefix}/sens_at_{target_fps}_fps", sens[idx])

    def _save_froc_outputs(
        self,
        slice_detections,
        scan_detections,
        total_slice_annotations,
        total_scan_annotations,
        total_slices,
        total_scans,
    ):
        for unit in self._froc_units():
            if unit == "slice":
                if slice_detections is not None and len(slice_detections) > 0:
                    self._save_single_froc_output(
                        unit=unit,
                        detections_tp_fp=slice_detections,
                        total_annotations=total_slice_annotations,
                        denominator=total_slices,
                    )
            elif unit == "scan":
                if scan_detections is not None and len(scan_detections) > 0:
                    self._save_single_froc_output(
                        unit=unit,
                        detections_tp_fp=scan_detections,
                        total_annotations=total_scan_annotations,
                        denominator=total_scans,
                    )
            else:
                raise ValueError(
                    f"Unsupported FROC normalization '{unit}'. "
                    "Use 'slice', 'scan', 'both', or a list containing 'slice'/'scan'."
                )

    def _nms_3d_indices(self, candidates, xy_radius, z_radius, iou_threshold):
        if not candidates:
            return []

        try:
            from nms_3d import nms_3d
        except ImportError as exc:
            raise ImportError(
                "NMS-3D is required for GravitySpace FROC post-processing. "
                "Install it with `pip install nms-3d`."
            ) from exc

        boxes = torch.tensor([
            [
                candidate["score"],
                candidate["x"] - xy_radius,
                candidate["y"] - xy_radius,
                candidate["z"] - z_radius,
                candidate["x"] + xy_radius,
                candidate["y"] + xy_radius,
                candidate["z"] + z_radius,
            ]
            for candidate in candidates
        ], dtype=torch.float32)

        kept_boxes = nms_3d(
            prediction_boxes=boxes,
            iou_threshold=iou_threshold,
            debug=False,
        )
        if not isinstance(kept_boxes, torch.Tensor):
            kept_boxes = torch.as_tensor(kept_boxes, dtype=boxes.dtype)

        kept_indices = []
        used_indices = set()
        for kept_box in kept_boxes:
            matches = torch.all(torch.isclose(boxes, kept_box.to(boxes.dtype), atol=1e-5), dim=1)
            match_indices = torch.nonzero(matches, as_tuple=False).flatten().tolist()
            match_indices = [idx for idx in match_indices if idx not in used_indices]
            if match_indices:
                kept_indices.append(match_indices[0])
                used_indices.add(match_indices[0])

        return kept_indices

    def _match_scan_candidates(self, candidates, annotations_by_slice, distance):
        if not candidates:
            return None

        detections = np.zeros((len(candidates), 2))
        detections[:, 1] = [candidate["score"] for candidate in candidates]
        matched_detection_indices = set()

        for z in sorted(annotations_by_slice):
            ann_slice = annotations_by_slice[z]
            gt_coords = ann_slice[ann_slice[:, 0] != -1, :2]
            if gt_coords.numel() == 0:
                continue

            slice_indices = [
                idx for idx, candidate in enumerate(candidates)
                if candidate["z"] == z and idx not in matched_detection_indices
            ]
            if not slice_indices:
                continue

            pred_xy = torch.tensor(
                [[candidates[idx]["x"], candidates[idx]["y"]] for idx in slice_indices],
                dtype=torch.float32,
            )
            gt_xy = gt_coords.float()
            dist = torch.cdist(pred_xy, gt_xy)
            dist_positive = dist <= distance

            for gt_idx in range(gt_xy.shape[0]):
                local_pos_idx = torch.nonzero(dist_positive[:, gt_idx], as_tuple=False).flatten().tolist()
                pos_idx = [
                    slice_indices[local_idx]
                    for local_idx in local_pos_idx
                    if slice_indices[local_idx] not in matched_detection_indices
                ]
                detections[pos_idx, 0] = -1

                if pos_idx:
                    local_scores = [candidates[idx]["score"] for idx in pos_idx]
                    best_idx = pos_idx[int(np.argmax(local_scores))]
                    matched_detection_indices.add(best_idx)
                    detections[best_idx, 0] = 1.0

        return detections[detections[:, 0] != -1]

    def _build_3d_gt_nodules(self, annotations_by_slice, merge_distance):
        points = []
        for z in sorted(annotations_by_slice):
            ann_slice = annotations_by_slice[z]
            valid = ann_slice[ann_slice[:, 0] != -1]
            for ann in valid:
                rx = float(ann[2].item()) if ann.shape[0] > 2 and ann[2] >= 0 else merge_distance
                ry = float(ann[3].item()) if ann.shape[0] > 3 and ann[3] >= 0 else merge_distance
                rz = float(ann[4].item()) if ann.shape[0] > 4 and ann[4] >= 0 else 0.0
                points.append({
                    "x": float(ann[0].item()),
                    "y": float(ann[1].item()),
                    "z": int(z),
                    "rx": max(rx, 1.0),
                    "ry": max(ry, 1.0),
                    "rz": max(rz, 0.0),
                })

        if not points:
            return []

        parent = list(range(len(points)))

        def find(idx):
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for i, point_i in enumerate(points):
            for j in range(i + 1, len(points)):
                point_j = points[j]
                dz = abs(point_i["z"] - point_j["z"])
                if dz > 1:
                    continue
                xy_distance = np.hypot(point_i["x"] - point_j["x"], point_i["y"] - point_j["y"])
                radius_gate = max(point_i["rx"], point_i["ry"], point_j["rx"], point_j["ry"], merge_distance)
                if xy_distance <= radius_gate:
                    union(i, j)

        components = {}
        for idx in range(len(points)):
            components.setdefault(find(idx), []).append(points[idx])

        nodules = []
        for component in components.values():
            xs = np.asarray([point["x"] for point in component])
            ys = np.asarray([point["y"] for point in component])
            zs = np.asarray([point["z"] for point in component])
            rxs = np.asarray([point["rx"] for point in component])
            rys = np.asarray([point["ry"] for point in component])
            rzs = np.asarray([point["rz"] for point in component])
            z_extent_radius = (float(np.max(zs)) - float(np.min(zs))) / 2.0
            nodules.append({
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "z": float(np.mean(zs)),
                "rx": max(float(np.mean(rxs)), 1.0),
                "ry": max(float(np.mean(rys)), 1.0),
                "rz": max(float(np.mean(rzs)), z_extent_radius, 1.0),
            })

        return nodules

    def _match_scan_candidates_3d(self, candidates, annotations_by_slice, distance, z_distance, merge_distance):
        if not candidates:
            return None, 0

        nodules = self._build_3d_gt_nodules(
            annotations_by_slice=annotations_by_slice,
            merge_distance=merge_distance,
        )
        detections = np.zeros((len(candidates), 2))
        detections[:, 1] = [candidate["score"] for candidate in candidates]

        if not nodules:
            return detections, 0

        matched_detection_indices = set()
        for nodule in nodules:
            positive_indices = []
            for idx, candidate in enumerate(candidates):
                if idx in matched_detection_indices:
                    continue

                dx = candidate["x"] - nodule["x"]
                dy = candidate["y"] - nodule["y"]
                dz = candidate["z"] - nodule["z"]
                rx = max(nodule["rx"], distance)
                ry = max(nodule["ry"], distance)
                rz = max(nodule["rz"], z_distance, 1.0)
                ellipsoid_distance = (dx / rx) ** 2 + (dy / ry) ** 2 + (dz / rz) ** 2
                if ellipsoid_distance <= 1.0:
                    positive_indices.append(idx)

            detections[positive_indices, 0] = -1
            if positive_indices:
                local_scores = [candidates[idx]["score"] for idx in positive_indices]
                best_idx = positive_indices[int(np.argmax(local_scores))]
                matched_detection_indices.add(best_idx)
                detections[best_idx, 0] = 1.0

        return detections[detections[:, 0] != -1], len(nodules)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            eps=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,  # or max_steps
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step"
                "frequency": 1,
            },
        }

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

    def on_test_epoch_start(self):
        self.test_step_outputs = []
        self._qualitative_saved = 0
        self._reset_gravity_points_from_hparams()

    def on_test_epoch_end(self):
        self.shared_epoch_end("test")
    
    def shared_step(self, batch, split):
        torch.cuda.empty_cache()  # Pulisci cache GPU dopo ogni step

        t_batch = time.time()
        
        slices = batch['slices']           # B x S x C x H x W
        annotations = batch['annotations'] # B x S x 10 x 4 (10 max nodules, coords: cx, cy, rx, ry)
        slicenames = batch['slicenames']
        original_lengths = batch.get('original_lengths')
        eval_slice_start = batch.get('eval_slice_start')
        eval_slice_end = batch.get('eval_slice_end')
        
        #print(f"[{split}] Batch received, shapes: slices {slices.shape}, annotations {annotations.shape}")
        
        # Model forward
        t_forward = time.time()
        classifications, regressions = self.model(slices)
        t_after_forward = time.time()
        #print(f"[{split}] Model forward: {(t_after_forward - t_forward)*1000:.1f}ms")
        
        # Loss computation
        t_loss = time.time()
        cls_loss, reg_loss = self.criterion(
            images_batch=slices,
            classifications_batch=classifications,
            regressions_batch=regressions,
            gravity_points=self.gravity_points,
            annotations_batch=annotations,
            slice_lengths=original_lengths,
            eval_slice_start=eval_slice_start,
            eval_slice_end=eval_slice_end
        )
        t_after_loss = time.time()
        #print(f"[{split}] Loss computation: {(t_after_loss - t_loss)*1000:.1f}ms")
        
        total_loss = cls_loss + reg_loss

        # Logging
        self.log(f"{split}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=slices.size(0))
        self.log(f"{split}/cls_loss", cls_loss, on_epoch=True, batch_size=slices.size(0))
        self.log(f"{split}/reg_loss", reg_loss, on_epoch=True, batch_size=slices.size(0))

        # Collect for FROC
        if split == "test":
            self.test_step_outputs.append({
                "classifications": classifications.detach().cpu(),
                "regressions": regressions.detach().cpu(),
                "annotations": annotations.detach().cpu(),
                "slices": slices.detach().cpu(),
                "slicenames": slicenames,
                "case": batch["case"],
                "original_lengths": original_lengths.detach().cpu() if original_lengths is not None else None,
                "subvolume_start": batch["subvolume_start"].detach().cpu() if "subvolume_start" in batch else None,
                "eval_slice_start": batch["eval_slice_start"].detach().cpu() if "eval_slice_start" in batch else None,
                "eval_slice_end": batch["eval_slice_end"].detach().cpu() if "eval_slice_end" in batch else None
            })
        
        t_end = time.time()
        #print(f"[{split}] Total step time: {(t_end - t_batch)*1000:.1f}ms\n")
        
        return total_loss

    def shared_epoch_end(self, split):
        if split == "test":

            if not self.test_step_outputs:
                return

            print("\nComputing FROC after test...")
            all_slice_detections = []
            all_scan_detections = []
            total_slice_annotations = 0
            total_scan_annotations = 0
            total_slices = 0
            evaluated_cases = set()
            candidates_by_case = {}
            candidates_after_nms_2d_by_case = {}
            annotations_by_case = {}
            
            distance = self.hparams.distance
            nms_radius = self.hparams.nms_radius
            nms_z_radius = self.hparams.nms_z_radius
            nms_2d_iou_threshold = self.hparams.nms_2d_iou_threshold
            nms_3d_iou_threshold = self.hparams.nms_3d_iou_threshold
            score_threshold = self.score_threshold
            image_shape = self.hparams.image_shape
            H, W = image_shape
            
            gravity_points = self.gravity_points.cpu()
            
            total_eval_slices_for_nms_2d = 0
            for output in self.test_step_outputs:
                classifications = output["classifications"]
                lengths_batch = output.get("original_lengths")
                eval_start_batch = output.get("eval_slice_start")
                eval_end_batch = output.get("eval_slice_end")
                B = classifications.shape[0]
                for i in range(B):
                    num_slices = int(lengths_batch[i].item()) if lengths_batch is not None else len(output["slicenames"][i])
                    local_eval_start = int(eval_start_batch[i].item()) if eval_start_batch is not None else 0
                    local_eval_end = int(eval_end_batch[i].item()) if eval_end_batch is not None else num_slices
                    local_eval_end = min(local_eval_end, num_slices)
                    total_eval_slices_for_nms_2d += max(local_eval_end - local_eval_start, 0)

            nms_2d_pbar = tqdm(
                total=total_eval_slices_for_nms_2d,
                desc="NMS 2D slices",
                unit="slice",
                leave=True,
            )

            for output in self.test_step_outputs:
                classifications = output["classifications"]
                regressions = output["regressions"]
                annotations_batch = output["annotations"]
                slices_batch = output.get("slices")
                filenames_batch = output["slicenames"]
                cases_batch = output.get("case")
                lengths_batch = output.get("original_lengths")
                subvolume_start_batch = output.get("subvolume_start")
                eval_start_batch = output.get("eval_slice_start")
                eval_end_batch = output.get("eval_slice_end")
                B, S = classifications.shape[:2]
                
                for i in range(B):
                    case_name = None
                    if cases_batch is not None:
                        raw_case = cases_batch[i]
                        case_name = str(raw_case[0] if isinstance(raw_case, (list, tuple)) else raw_case).strip()
                    if not case_name:
                        case_name = f"batch_{len(evaluated_cases)}_{i}"
                    num_slices = int(lengths_batch[i].item()) if lengths_batch is not None else len(filenames_batch[i])
                    subvolume_start = int(subvolume_start_batch[i].item()) if subvolume_start_batch is not None else 0
                    local_eval_start = int(eval_start_batch[i].item()) if eval_start_batch is not None else 0
                    local_eval_end = int(eval_end_batch[i].item()) if eval_end_batch is not None else num_slices
                    local_eval_end = min(local_eval_end, num_slices)
                    evaluated_slice_count = max(local_eval_end - local_eval_start, 0)
                    total_slices += evaluated_slice_count
                    if evaluated_slice_count > 0 and case_name:
                        evaluated_cases.add(case_name)

                    annotation = annotations_batch[i, :num_slices]  # (num_slices, MAX_NODULES, 4)
                    total_slice_annotations += torch.sum(annotation[local_eval_start:local_eval_end, :, 0] != -1).item()
                    annotations_by_case.setdefault(case_name, {})

                    for j in range(local_eval_start, local_eval_end):
                        try:
                            global_z = subvolume_start + j
                            annotations_by_case[case_name][global_z] = annotation[j]

                            # Filename
                            raw = filenames_batch[i][j]
                            filename = str(raw[0] if isinstance(raw, (list, tuple)) else raw).strip()
                            
                            # Predictions
                            new_gravity = gravity_points + regressions[i, j]
                            scores = classifications[i, j, :, 1].clone()

                            self._save_gravity_points_plot(
                                image=slices_batch[i, j],
                                centers_xy=new_gravity,
                                scores=scores,
                                annotations=annotation[j],
                                filename=filename,
                                distance=distance,
                                image_shape=image_shape,
                            )
                            
                            # Invalid filter
                            invalid = (new_gravity[:, 0] < 0) | (new_gravity[:, 1] < 0) | \
                                    (new_gravity[:, 0] >= W) | (new_gravity[:, 1] >= H)
                            new_gravity[invalid] = -2
                            scores[invalid] = -2
                            
                            keep = new_gravity[:, 0] != -2
                            new_gravity = new_gravity[keep]
                            scores = scores[keep]
                            
                            if len(new_gravity) == 0:
                                continue
                                
                            # Score threshold
                            new_gravity = new_gravity.float()
                            scores = scores.float()
                            #print(f"Case {case_name}, slice {j}: candidates before score threshold={len(new_gravity)}")
                            keep = scores >= score_threshold
                            new_gravity = new_gravity[keep]
                            scores = scores[keep]
                            
                            if len(new_gravity) == 0:
                                continue
                            #print(f"Case {case_name}, slice {j}: candidates after score threshold={len(new_gravity)}")

                            # NMS 2D
                            boxes_2d = torch.stack([
                                new_gravity[:, 0] - nms_radius,
                                new_gravity[:, 1] - nms_radius,
                                new_gravity[:, 0] + nms_radius,
                                new_gravity[:, 1] + nms_radius,
                            ], dim=1).float()
                            nms_2d_idx = nms(
                                boxes=boxes_2d,
                                scores=scores,
                                iou_threshold=nms_2d_iou_threshold,
                            )
                            new_gravity = new_gravity[nms_2d_idx]
                            scores = scores[nms_2d_idx]

                            if len(new_gravity) == 0:
                                continue

                            candidates_after_nms_2d_by_case[case_name] = (
                                candidates_after_nms_2d_by_case.get(case_name, 0) + len(new_gravity)
                            )
                            case_candidates = candidates_by_case.setdefault(case_name, [])
                            for det_idx in range(len(new_gravity)):
                                case_candidates.append({
                                    "x": float(new_gravity[det_idx, 0].item()),
                                    "y": float(new_gravity[det_idx, 1].item()),
                                    "z": int(global_z),
                                    "score": float(scores[det_idx].item()),
                                })
                        finally:
                            nms_2d_pbar.update(1)
                        
            nms_2d_pbar.close()

            for case_name, candidates in tqdm(
                candidates_by_case.items(),
                desc="NMS 3D scans",
                unit="scan",
                leave=True,
            ):
                print(
                    f"2D NMS scan {case_name}: "
                    f"candidates_after_2d_nms={candidates_after_nms_2d_by_case.get(case_name, 0)}"
                )
                nms_start = time.perf_counter()
                kept_indices = self._nms_3d_indices(
                    candidates=candidates,
                    xy_radius=nms_radius,
                    z_radius=nms_z_radius,
                    iou_threshold=nms_3d_iou_threshold,
                )
                nms_elapsed = time.perf_counter() - nms_start
                print(
                    f"3D NMS scan {case_name}: "
                    f"candidates={len(candidates)} | kept={len(kept_indices)} | "
                    f"time={nms_elapsed:.2f}s"
                )
                kept_candidates = [candidates[idx] for idx in kept_indices]
                current_slice_detections = self._match_scan_candidates(
                    candidates=kept_candidates,
                    annotations_by_slice=annotations_by_case.get(case_name, {}),
                    distance=distance,
                )
                if current_slice_detections is not None and len(current_slice_detections) > 0:
                    all_slice_detections.append(current_slice_detections)

                current_scan_detections, num_scan_annotations = self._match_scan_candidates_3d(
                    candidates=kept_candidates,
                    annotations_by_slice=annotations_by_case.get(case_name, {}),
                    distance=distance,
                    z_distance=nms_z_radius,
                    merge_distance=nms_radius,
                )
                total_scan_annotations += num_scan_annotations
                if current_scan_detections is not None and len(current_scan_detections) > 0:
                    all_scan_detections.append(current_scan_detections)
                        
            slice_detections_tp_fp = np.concatenate(all_slice_detections, axis=0) if all_slice_detections else None
            scan_detections_tp_fp = np.concatenate(all_scan_detections, axis=0) if all_scan_detections else None
            total_scans = len(evaluated_cases)

            print(
                f"Total slice detections: {0 if slice_detections_tp_fp is None else len(slice_detections_tp_fp)} | "
                f"Total slice annotations: {total_slice_annotations} | "
                f"Total slices: {total_slices}"
            )
            print(
                f"Total scan detections: {0 if scan_detections_tp_fp is None else len(scan_detections_tp_fp)} | "
                f"Total scan annotations: {total_scan_annotations} | "
                f"Total scans: {total_scans}"
            )
            self._save_froc_outputs(
                slice_detections=slice_detections_tp_fp,
                scan_detections=scan_detections_tp_fp,
                total_slice_annotations=total_slice_annotations,
                total_scan_annotations=total_scan_annotations,
                total_slices=total_slices,
                total_scans=total_scans,
            )

            # Clean up
            self.test_step_outputs = []
