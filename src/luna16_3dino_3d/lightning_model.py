"""Partial-fine-tuning LightningModule for the 3DINO-ViT LUNA16 baseline.

Aggregation (per the task spec, no lesion-aware attention): window CLS
embeddings -> element-wise mean, element-wise max, concatenate(mean, max) ->
Linear(2*1024, 1) classification head, trained with BCEWithLogitsLoss.

Metrics/epoch-buffering pattern is reused from
``src/luna16_synthetic_2d/lightning_model.py`` (sklearn-based AUC/MCC/F1/
accuracy computed once per epoch from buffered predictions, `sync_dist=True`
logging), adapted to the one-patient-per-step batching used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.luna16_3dino_3d.model import (  # noqa: E402
    EMBED_DIM,
    apply_partial_freeze,
    build_backbone,
    load_checkpoint,
    print_freeze_report,
)


def _cosine_warmup_lambda(epoch: int, warmup_epochs: int, max_epochs: int, min_lr_ratio: float = 0.01) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


class Dino3DPartialFinetuneClassifier(pl.LightningModule):
    def __init__(
        self,
        checkpoint_path: str | None,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        micro_batch_windows: int = 32,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = build_backbone()
        if checkpoint_path:
            load_checkpoint(self.backbone, checkpoint_path)
        self.freeze_report = apply_partial_freeze(self.backbone)
        print_freeze_report(self.freeze_report)

        self.head = nn.Linear(2 * EMBED_DIM, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_outputs: list[dict] = []
        self.validation_outputs: list[dict] = []
        self.test_outputs: list[dict] = []

    def on_fit_start(self) -> None:
        metrics = {
            "trainable_params": float(self.freeze_report["trainable_params"]),
            "frozen_params": float(self.freeze_report["frozen_params"]),
            "total_params": float(self.freeze_report["total_params"]),
            "trainable_pct": self.freeze_report["trainable_pct"],
        }
        for logger in self.loggers:
            logger.log_metrics(metrics, step=0)

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (N, 1, 112, 112, 112) -> (N, 1024) CLS embeddings.

        Chunks the forward pass to bound peak memory (``micro_batch_windows``)
        while keeping the autograd graph intact across chunks (no detaching),
        so gradients still flow to every window's contribution to the
        backbone's trainable (second-half) blocks.
        """
        chunk_size = self.hparams.micro_batch_windows
        embeddings = []
        for start in range(0, windows.shape[0], chunk_size):
            embeddings.append(self.backbone(windows[start : start + chunk_size]))
        return torch.cat(embeddings, dim=0)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        embeddings = self.encode_windows(windows)
        mean_embedding = embeddings.mean(dim=0)
        max_embedding = embeddings.max(dim=0).values
        patient_feature = torch.cat([mean_embedding, max_embedding], dim=0)
        return self.head(patient_feature.unsqueeze(0)).squeeze()

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        windows, label, sample_id = batch
        windows = windows.to(self.device, non_blocking=True)
        label = torch.as_tensor(label, dtype=torch.float32, device=self.device)

        logit = self.forward(windows)
        loss = self.criterion(logit, label)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=1)

        score = torch.sigmoid(logit).detach()
        prediction = (score >= 0.5).long()
        output = {
            "label": int(label.item()),
            "prediction": int(prediction.item()),
            "score": float(score.item()),
            "sample_id": sample_id,
        }
        {"train": self.train_outputs, "val": self.validation_outputs, "test": self.test_outputs}[stage].append(output)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _compute_epoch_metrics(self, outputs: list[dict], stage: str) -> None:
        if not outputs:
            return
        labels = np.array([o["label"] for o in outputs])
        preds = np.array([o["prediction"] for o in outputs])
        scores = np.array([o["score"] for o in outputs])

        metrics = {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "mcc": matthews_corrcoef(labels, preds),
            "auc": roc_auc_score(labels, scores) if len(set(labels.tolist())) == 2 else 0.0,
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
        }
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, prog_bar=True, sync_dist=True)

        if stage == "test":
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            print(f"[3DINO][test] confusion_matrix tn={tn} fp={fp} fn={fn} tp={tp}", flush=True)

    def on_train_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.train_outputs, "train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.test_outputs, "test")
        self.test_prediction_rows = list(self.test_outputs)
        self.test_outputs.clear()

    def configure_optimizers(self):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr, "name": "backbone"},
                {"params": head_params, "lr": self.hparams.head_lr, "name": "head"},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: _cosine_warmup_lambda(
                epoch, self.hparams.warmup_epochs, self.hparams.max_epochs
            ),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
