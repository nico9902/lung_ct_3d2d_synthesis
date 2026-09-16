from __future__ import annotations

import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import SliceAttentionClassifier


class SliceAttentionLightningModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        pretrained: bool = True,
        attention_dim: int = 256,
        dropout: float = 0.2,
        slice_chunk_size: int = 16,
        freeze_backbone: bool = False,
        freeze_half_backbone: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SliceAttentionClassifier(
            backbone=backbone,
            pretrained=pretrained,
            attention_dim=attention_dim,
            dropout=dropout,
            slice_chunk_size=slice_chunk_size,
            freeze_backbone=freeze_backbone,
            freeze_half_backbone=freeze_half_backbone,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.stage_outputs: dict[str, list[dict[str, object]]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        self.test_prediction_rows: list[dict[str, object]] = []
        self._epoch_start_time = 0.0

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        slices, slice_counts, labels, sample_ids = batch
        return (
            slices.to(device, non_blocking=True),
            slice_counts,
            labels.to(device, non_blocking=True),
            sample_ids,
        )

    def _prepare_slices(
        self,
        slices: torch.Tensor,
        slice_counts: torch.Tensor,
        augment: bool,
    ) -> torch.Tensor:
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        slices = slices.to(dtype).div_(255.0)
        if not augment:
            return slices

        counts = slice_counts.to(self.device, non_blocking=True)
        patient_index = torch.repeat_interleave(
            torch.arange(len(counts), device=self.device), counts
        )
        horizontal = torch.rand(len(counts), device=self.device) < 0.5
        vertical = torch.rand(len(counts), device=self.device) < 0.5
        add_noise = torch.rand(len(counts), device=self.device) < 0.25
        horizontal_slices = horizontal[patient_index]
        vertical_slices = vertical[patient_index]
        noisy_slices = add_noise[patient_index]
        slices[horizontal_slices] = torch.flip(slices[horizontal_slices], dims=[3])
        slices[vertical_slices] = torch.flip(slices[vertical_slices], dims=[2])
        slices[noisy_slices] = (
            slices[noisy_slices] + torch.randn_like(slices[noisy_slices]) * 0.015
        )
        return slices.clamp_(0.0, 1.0)

    def forward(self, slices: torch.Tensor, slice_counts: torch.Tensor):
        return self.model(slices, slice_counts)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        slices, slice_counts, labels, sample_ids = batch
        slices = self._prepare_slices(slices, slice_counts, augment=stage == "train")
        logits, _ = self(slices, slice_counts)
        loss = self.criterion(logits, labels)
        scores = torch.softmax(logits.float(), dim=1)[:, 1]
        predictions = torch.argmax(logits.float(), dim=1)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labels.shape[0],
        )
        self.stage_outputs[stage].append(
            {
                "labels": labels.detach(),
                "predictions": predictions.detach(),
                "scores": scores.detach(),
                "sample_ids": [str(sample_id) for sample_id in sample_ids],
            }
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    @staticmethod
    def _metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray):
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        return {
            "auc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else 0.0,
            "mcc": float(matthews_corrcoef(labels, predictions)),
            "acc": float(accuracy_score(labels, predictions)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    def _finish_stage(self, stage: str) -> None:
        outputs = self.stage_outputs[stage]
        if not outputs:
            return
        labels = torch.cat([item["labels"] for item in outputs]).cpu().numpy()
        predictions = torch.cat([item["predictions"] for item in outputs]).cpu().numpy()
        scores = torch.cat([item["scores"] for item in outputs]).cpu().numpy()
        metrics = self._metrics(labels, predictions, scores)
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", float(value), prog_bar=name in ("auc", "mcc", "acc"))

        if stage == "test":
            sample_ids = [
                sample_id for output in outputs for sample_id in output["sample_ids"]
            ]
            self.test_prediction_rows = [
                {
                    "sample_id": sample_id,
                    "split": "test",
                    "label": int(label),
                    "prediction": int(prediction),
                    "score": float(score),
                }
                for sample_id, label, prediction, score in zip(
                    sample_ids, labels, predictions, scores
                )
            ]
        outputs.clear()

    def on_train_epoch_start(self) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self) -> None:
        self._finish_stage("train")

    def on_validation_epoch_end(self) -> None:
        self._finish_stage("val")
        if not self.trainer.sanity_checking:
            self.log("epoch_seconds", time.perf_counter() - self._epoch_start_time)

    def on_test_epoch_end(self) -> None:
        self._finish_stage("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


__all__ = ["SliceAttentionLightningModule"]
