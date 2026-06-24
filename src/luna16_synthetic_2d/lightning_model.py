from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .models import build_model


class SyntheticLuna16Classifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        class_names: list[str],
        lr: float,
        weight_decay: float,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_half_backbone: bool = False,
        freeze_first_layers: int = 0,
        unfreeze_last_layers: int = 0,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        output_dim = 1 if num_classes == 2 else num_classes
        self.model = build_model(
            backbone=backbone,
            num_classes=output_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            freeze_half_backbone=freeze_half_backbone,
            freeze_first_layers=freeze_first_layers,
            unfreeze_last_layers=unfreeze_last_layers,
        )
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        self.train_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_outputs: list[dict[str, torch.Tensor]] = []
        self.test_outputs: list[dict[str, torch.Tensor]] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels, _ = batch
        logits = self(images)
        if self.hparams.num_classes == 2:
            logits = logits.squeeze(1)
            loss = self.criterion(logits, labels.float())
            scores = torch.sigmoid(logits)
            predictions = (scores >= 0.5).long()
        else:
            loss = self.criterion(logits, labels)
            scores = torch.softmax(logits, dim=1)
            predictions = scores.argmax(dim=1)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        output = {
            "labels": labels.detach().cpu(),
            "predictions": predictions.detach().cpu(),
            "scores": scores.detach().cpu(),
        }
        if stage == "train":
            self.train_outputs.append(output)
        elif stage == "val":
            self.validation_outputs.append(output)
        elif stage == "test":
            self.test_outputs.append(output)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _compute_epoch_metrics(self, outputs: list[dict[str, torch.Tensor]], stage: str) -> None:
        if not outputs:
            return
        labels = torch.cat([item["labels"] for item in outputs]).numpy()
        predictions = torch.cat([item["predictions"] for item in outputs]).numpy()
        scores = torch.cat([item["scores"] for item in outputs]).numpy()
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(
            labels,
            predictions,
            average="binary" if self.hparams.num_classes == 2 else "macro",
            zero_division=0,
        )
        mcc = matthews_corrcoef(labels, predictions)
        self.log(f"{stage}_acc", accuracy, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_mcc", mcc, prog_bar=True, sync_dist=True)

        auc = 0.0
        observed_classes = set(labels.tolist())
        if self.hparams.num_classes == 2 and len(observed_classes) == 2:
            auc = roc_auc_score(labels, scores)
        elif self.hparams.num_classes > 2 and len(observed_classes) == self.hparams.num_classes:
            auc = roc_auc_score(labels, scores, multi_class="ovr", average="macro")
        self.log(f"{stage}_auc", auc, prog_bar=True, sync_dist=True)

        if stage == "test":
            class_indices = list(range(self.hparams.num_classes))
            print(
                classification_report(
                    labels,
                    predictions,
                    labels=class_indices,
                    target_names=self.hparams.class_names,
                    zero_division=0,
                )
            )
            print(confusion_matrix(labels, predictions, labels=class_indices))

    def on_train_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.train_outputs, "train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.test_outputs, "test")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
