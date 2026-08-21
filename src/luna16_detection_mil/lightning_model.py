from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

from .model import DetectionMILClassifier


class DetectionMILLightningModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str,
        pooling: str,
        num_classes: int,
        class_names: list[str],
        lr: float,
        weight_decay: float,
        pretrained: bool = True,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = DetectionMILClassifier(backbone=backbone, pooling=pooling, pretrained=pretrained)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, bags: torch.Tensor, valid_mask: torch.Tensor):
        return self.model(bags, valid_mask)

    def _shared_step(self, batch, stage: str):
        bags, valid_mask, candidate_probs, labels, sample_ids = batch
        logits, attention_weights = self(bags, valid_mask)
        loss = self.criterion(logits, labels.float())
        scores = torch.sigmoid(logits)
        predictions = (scores >= 0.5).long()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        output = {
            "labels": labels.detach().cpu(),
            "predictions": predictions.detach().cpu(),
            "scores": scores.detach().cpu(),
            "valid_counts": valid_mask.sum(dim=1).detach().cpu(),
            "max_detector_probability": candidate_probs.max(dim=1).values.detach().cpu(),
            "sample_ids": [str(sample_id) for sample_id in sample_ids],
        }
        if stage == "train":
            self.train_outputs.append(output)
        elif stage == "val":
            self.validation_outputs.append(output)
        elif stage == "test":
            output["attention_weights"] = attention_weights.detach().cpu()
            self.test_outputs.append(output)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _compute_epoch_metrics(self, outputs, stage: str):
        if not outputs:
            return
        labels = torch.cat([item["labels"] for item in outputs]).numpy()
        predictions = torch.cat([item["predictions"] for item in outputs]).numpy()
        scores = torch.cat([item["scores"] for item in outputs]).numpy()
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        mcc = matthews_corrcoef(labels, predictions)
        auc = roc_auc_score(labels, scores) if len(set(labels.tolist())) == 2 else 0.0
        self.log(f"{stage}_acc", accuracy, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_mcc", mcc, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_auc", auc, prog_bar=True, sync_dist=True)
        if stage == "test":
            print(classification_report(labels, predictions, target_names=self.hparams.class_names, zero_division=0))
            print(confusion_matrix(labels, predictions, labels=[0, 1]))

    def _prediction_rows(self, outputs, stage: str):
        if not outputs:
            return []
        labels = torch.cat([item["labels"] for item in outputs]).numpy()
        predictions = torch.cat([item["predictions"] for item in outputs]).numpy()
        scores = torch.cat([item["scores"] for item in outputs]).numpy()
        valid_counts = torch.cat([item["valid_counts"] for item in outputs]).numpy()
        max_probs = torch.cat([item["max_detector_probability"] for item in outputs]).numpy()
        sample_ids = [sample_id for item in outputs for sample_id in item["sample_ids"]]
        rows = []
        for sample_id, label, pred, score, valid_count, max_prob in zip(
            sample_ids, labels, predictions, scores, valid_counts, max_probs
        ):
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": stage,
                    "label": int(label),
                    "label_name": self.hparams.class_names[int(label)],
                    "prediction": int(pred),
                    "prediction_name": self.hparams.class_names[int(pred)],
                    "score": float(score),
                    "valid_detection_count": int(valid_count),
                    "max_detector_probability": float(max_prob),
                }
            )
        return rows

    def on_train_epoch_end(self):
        self._compute_epoch_metrics(self.train_outputs, "train")
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        self._compute_epoch_metrics(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        self._compute_epoch_metrics(self.test_outputs, "test")
        self.test_prediction_rows = self._prediction_rows(self.test_outputs, "test")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
