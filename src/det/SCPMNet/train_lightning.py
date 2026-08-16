from __future__ import annotations

from pathlib import Path
import sys

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.SCPMNet.datamodule import SCPMDataModule
from src.det.SCPMNet.lightning_model import SCPMLitModel


class StartEpochModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, start_epoch: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = int(start_epoch)

    def on_validation_end(self, trainer, pl_module):
        if int(trainer.current_epoch) + 1 < self.start_epoch:
            return
        return super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if int(trainer.current_epoch) + 1 < self.start_epoch:
            return
        return super().on_train_epoch_end(trainer, pl_module)


class ValidationScheduleCallback(pl.Callback):
    def __init__(self, start_epoch: int = 1, every_n_epochs: int = 1, before_start_every_n_epochs: int | None = None):
        super().__init__()
        self.start_epoch = max(1, int(start_epoch))
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.before_start_every_n_epochs = (
            None if before_start_every_n_epochs is None else max(1, int(before_start_every_n_epochs))
        )

    def setup(self, trainer, pl_module, stage: str):
        if stage == "fit":
            trainer.check_val_every_n_epoch = self.before_start_every_n_epochs or self.start_epoch

    def on_validation_end(self, trainer, pl_module):
        if int(trainer.current_epoch) + 1 >= self.start_epoch:
            trainer.check_val_every_n_epoch = self.every_n_epochs


class FiniteValueMonitor(pl.Callback):
    def __init__(
        self,
        check_logged_metrics: bool = True,
        check_gradients: bool = True,
        every_n_train_steps: int = 1,
    ):
        super().__init__()
        self.check_logged_metrics = bool(check_logged_metrics)
        self.check_gradients = bool(check_gradients)
        self.every_n_train_steps = max(1, int(every_n_train_steps))

    @staticmethod
    def _ensure_finite(name: str, value) -> None:
        if isinstance(value, torch.Tensor):
            if value.numel() and not torch.isfinite(value.detach()).all():
                raise FloatingPointError(f"Non-finite value detected in {name}.")
            return
        if isinstance(value, (float, int)):
            scalar = torch.tensor(float(value))
            if not torch.isfinite(scalar):
                raise FloatingPointError(f"Non-finite value detected in {name}: {value}.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, torch.Tensor):
            self._ensure_finite("train_step_output", outputs)
        elif isinstance(outputs, dict):
            for name, value in outputs.items():
                self._ensure_finite(f"train_step_output/{name}", value)

        if self.check_logged_metrics:
            for name, value in trainer.callback_metrics.items():
                self._ensure_finite(f"logged_metric/{name}", value)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.check_gradients or int(trainer.global_step) % self.every_n_train_steps:
            return
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all():
                raise FloatingPointError(f"Non-finite gradient detected in parameter {name}.")


def build_logger(cfg: DictConfig):
    if not cfg.get("use_wandb", False):
        return False

    from pytorch_lightning.loggers import WandbLogger
    wandb.login(key="5eb716fd87389d240533319f8751488e37103d23")
    logger = WandbLogger(
        project=cfg.wandb_project,
        name=cfg.wandb_name or cfg.experiment_name,
        entity=cfg.wandb_entity,
        save_dir=str(cfg.get("wandb_save_dir", "wandb2")),
        mode=cfg.wandb_mode,
        log_model=cfg.get("wandb_log_model", False),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger.experiment.define_metric("trainer/global_step")
    logger.experiment.define_metric("*", step_metric="trainer/global_step")
    return logger


def finalize_logger(logger, status: str) -> None:
    if not logger:
        return
    logger.finalize(status)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "finish"):
        experiment.finish(exit_code=0 if status == "success" else 1)


def run(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    pl.seed_everything(cfg.seed, workers=True)
    exp_dir = Path(cfg.output_dir) / cfg.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    val_mode_names = {
        "full_volume_loss": "full_volume",
        "fixed_crop_loss": "fixed_crop",
        "random_crop_loss": "random_crop",
        "full_volume_froc": "full_volume_froc",
    }
    val_loader_names = [val_mode_names.get(str(mode), str(mode)) for mode in (cfg.get("val_modes", []) or [])] or ["val"]

    datamodule = SCPMDataModule(
        csv_path=cfg.csv_path,
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        crop_size=tuple(cfg.crop_size),
        samples_per_volume=cfg.samples_per_volume,
        clip=tuple(cfg.clip),
        intensity_mode=cfg.get("intensity_mode", "hu"),
        normalized_volume_cache_dir=cfg.get("normalized_volume_cache_dir", None),
        positive_crop_prob=cfg.positive_crop_prob,
        mask_path_column=cfg.mask_path_column,
        skip_missing_images=cfg.skip_missing_images,
        val_full_volume=cfg.get("val_full_volume", False),
        val_modes=tuple(cfg.get("val_modes", []) or []),
        val_fixed_crop_seed=cfg.get("val_fixed_crop_seed", None),
        val_random_crop_samples_per_volume=cfg.get("val_random_crop_samples_per_volume", 1),
        test_full_volume=cfg.test_full_volume,
        sliding_window_stride=tuple(cfg.sliding_window_stride),
        pin_memory=cfg.accelerator in ("gpu", "cuda", "auto"),
    )
    model = SCPMLitModel(
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        stride=cfg.stride,
        positive_topk=cfg.positive_topk,
        positive_radius_factor=cfg.positive_radius_factor,
        neg_pos_ratio=cfg.neg_pos_ratio,
        focal_gamma=cfg.focal_gamma,
        refocal_threshold=cfg.refocal_threshold,
        refocal_weight=cfg.refocal_weight,
        smooth_l1_beta=cfg.smooth_l1_beta,
        lambda_cls=cfg.lambda_cls,
        lambda_radius=cfg.lambda_radius,
        lambda_offset=cfg.lambda_offset,
        lambda_siou=cfg.lambda_siou,
        using_sac=cfg.using_sac,
        optimizer_name=cfg.optimizer_name,
        momentum=cfg.momentum,
        warmup_epochs=cfg.warmup_epochs,
        warmup_lr=cfg.warmup_lr,
        lr_scheduler_name=cfg.get("lr_scheduler_name", "milestone"),
        lr_milestones=tuple(cfg.lr_milestones),
        lr_gamma=cfg.lr_gamma,
        cosine_restart_t_0=cfg.get("cosine_restart_t_0", 50),
        cosine_restart_t_mult=cfg.get("cosine_restart_t_mult", 1),
        cosine_eta_min=cfg.get("cosine_eta_min", 1e-6),
        decode_threshold=cfg.decode_threshold,
        decode_topk=cfg.decode_topk,
        nms_threshold=cfg.nms_threshold,
        final_topk=cfg.final_topk,
        evaluate_froc=cfg.evaluate_froc,
        evaluate_val_froc=cfg.get("evaluate_val_froc", None),
        evaluate_test_froc=cfg.get("evaluate_test_froc", None),
        val_loader_names=tuple(val_loader_names),
        froc_match_strategy=cfg.froc_match_strategy,
        froc_iou_threshold=cfg.froc_iou_threshold,
        froc_fp_rates=tuple(cfg.froc_fp_rates),
    )
    logger = build_logger(cfg)
    checkpoint_monitor = str(cfg.get("checkpoint_monitor", "val/loss"))
    checkpoint_mode = str(cfg.get("checkpoint_mode", "min"))
    checkpoint_every_n_epochs = cfg.get("checkpoint_every_n_epochs", None)
    checkpoint = StartEpochModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename=str(cfg.get("checkpoint_filename", "epoch={epoch:03d}-val_loss={val/loss:.4f}")),
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_top_k=cfg.get("checkpoint_save_top_k", 1),
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=checkpoint_every_n_epochs,
        start_epoch=cfg.get("checkpoint_start_epoch", 1),
    )
    callbacks = [checkpoint]
    if cfg.get("monitor_finite_values", False):
        callbacks.append(
            FiniteValueMonitor(
                check_logged_metrics=cfg.get("finite_monitor_check_logged_metrics", True),
                check_gradients=cfg.get("finite_monitor_check_gradients", True),
                every_n_train_steps=cfg.get("finite_monitor_every_n_train_steps", 1),
            )
        )
    if cfg.get("val_froc_start_epoch", None) is not None:
        callbacks.append(
            ValidationScheduleCallback(
                start_epoch=cfg.get("val_froc_start_epoch", 1),
                every_n_epochs=cfg.get("check_val_every_n_epoch", 1),
                before_start_every_n_epochs=cfg.get("val_froc_before_start_every_n_epoch", None),
            )
        )
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    val_froc_start_epoch = cfg.get("val_froc_start_epoch", None)
    before_start_every = cfg.get("val_froc_before_start_every_n_epoch", None)
    initial_check_val_every_n_epoch = (
        cfg.get("check_val_every_n_epoch", 1)
        if val_froc_start_epoch is None
        else max(1, int(before_start_every if before_start_every is not None else val_froc_start_epoch))
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        default_root_dir=str(exp_dir),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gradient_clip_val=cfg.get("gradient_clip_val", 0.0),
        gradient_clip_algorithm=cfg.get("gradient_clip_algorithm", "norm"),
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        check_val_every_n_epoch=initial_check_val_every_n_epoch,
        detect_anomaly=cfg.get("detect_anomaly", False),
    )
    status = "success"
    try:
        if cfg.test_only:
            datamodule.setup("test")
            if len(datamodule.test_ds) == 0:
                print("No test samples found after dataset filtering; skipping test.")
                return
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
            return
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.checkpoint)
        datamodule.setup("test")
        if len(datamodule.test_ds) == 0:
            print("No test samples found after dataset filtering; skipping final test.")
            return
        ckpt_path = checkpoint.best_model_path or "best"
        posthoc_top_k = int(cfg.get("posthoc_val_froc_top_k", 0) or 0)
        if posthoc_top_k > 0 and checkpoint.best_k_models:
            candidates = _top_checkpoint_paths(checkpoint, checkpoint_mode, posthoc_top_k)
            selected = _select_checkpoint_by_validation_froc(cfg, candidates, exp_dir)
            if selected is not None:
                ckpt_path = selected
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except Exception:
        status = "failed"
        raise
    finally:
        finalize_logger(logger, status)


def _top_checkpoint_paths(checkpoint: ModelCheckpoint, mode: str, top_k: int) -> list[str]:
    reverse = str(mode).lower() == "max"
    scored = [(str(path), float(score.detach().cpu() if hasattr(score, "detach") else score)) for path, score in checkpoint.best_k_models.items()]
    return [path for path, _ in sorted(scored, key=lambda item: item[1], reverse=reverse)[:top_k]]


def _select_checkpoint_by_validation_froc(cfg: DictConfig, checkpoint_paths: list[str], exp_dir: Path) -> str | None:
    if not checkpoint_paths:
        return None
    rows = []
    val_froc_dm = SCPMDataModule(
        csv_path=cfg.csv_path,
        data_root=cfg.data_root,
        batch_size=cfg.get("posthoc_val_froc_batch_size", None) or cfg.batch_size,
        num_workers=cfg.num_workers,
        crop_size=tuple(cfg.crop_size),
        samples_per_volume=cfg.samples_per_volume,
        clip=tuple(cfg.clip),
        intensity_mode=cfg.get("intensity_mode", "hu"),
        normalized_volume_cache_dir=cfg.get("normalized_volume_cache_dir", None),
        positive_crop_prob=cfg.positive_crop_prob,
        mask_path_column=cfg.mask_path_column,
        skip_missing_images=cfg.skip_missing_images,
        val_modes=("full_volume_froc",),
        val_fixed_crop_seed=cfg.get("val_fixed_crop_seed", None),
        val_random_crop_samples_per_volume=cfg.get("val_random_crop_samples_per_volume", 1),
        test_full_volume=cfg.test_full_volume,
        sliding_window_stride=tuple(cfg.sliding_window_stride),
        pin_memory=cfg.accelerator in ("gpu", "cuda", "auto"),
    )
    for ckpt_path in checkpoint_paths:
        candidate = SCPMLitModel.load_from_checkpoint(
            ckpt_path,
            evaluate_val_froc=True,
            evaluate_froc=True,
            evaluate_test_froc=cfg.get("evaluate_test_froc", None),
            val_loader_names=("full_volume_froc",),
            prediction_dir=f"posthoc_val_froc/{Path(ckpt_path).stem}",
        )
        candidate_dir = exp_dir / "posthoc_val_froc" / Path(ckpt_path).stem
        evaluator = pl.Trainer(
            accelerator=cfg.accelerator,
            devices=cfg.devices,
            precision=cfg.precision,
            default_root_dir=str(candidate_dir),
            logger=False,
            callbacks=[],
            num_sanity_val_steps=0,
            enable_progress_bar=False,
        )
        metrics = evaluator.validate(candidate, datamodule=val_froc_dm, verbose=False)
        mean_froc = float(metrics[0].get("val/mean_froc", float("-inf"))) if metrics else float("-inf")
        rows.append({"checkpoint": ckpt_path, "val_mean_froc": mean_froc})
    out_path = exp_dir / "posthoc_val_froc" / "top_loss_checkpoint_val_froc.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("val_mean_froc", ascending=False).to_csv(out_path, index=False)
    return max(rows, key=lambda row: row["val_mean_froc"])["checkpoint"] if rows else None


@hydra.main(version_base=None, config_path="conf", config_name="train_lightning")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
