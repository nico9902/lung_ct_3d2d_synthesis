from pathlib import Path
import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.det.CPMNetv2.lidc_datamodule import LIDCCPMNetDataModule
from src.det.CPMNetv2.lightning_model import CPMNetv2LitModel


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


def _maybe_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    pl.seed_everything(cfg.seed, workers=True)

    exp_dir = Path(cfg.output_dir) / cfg.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    pin_memory = cfg.accelerator in ("gpu", "cuda", "auto")
    datamodule = LIDCCPMNetDataModule.from_split_csv(
        csv_path=cfg.csv_path,
        images_dir=cfg.images_dir,
        annotations_dir=cfg.annotations_dir,
        batch_size=cfg.batch_size,
        view=cfg.view,
        num_workers=cfg.num_workers,
        crop_size=list(cfg.crop_size),
        overlap_size=list(cfg.overlap_size),
        spacing=list(cfg.spacing),
        num_samples=cfg.num_samples,
        pin_memory=pin_memory,
        labels_csv=cfg.labels_csv,
        val_full_volume=cfg.get("val_full_volume", False),
    )

    model = CPMNetv2LitModel(
        crop_size=list(cfg.crop_size),
        spacing=list(cfg.spacing),
        lr=cfg.lr,
        topk=cfg.topk,
        lambda_cls=cfg.lambda_cls,
        lambda_offset=cfg.lambda_offset,
        lambda_shape=cfg.lambda_shape,
        lambda_iou=cfg.lambda_iou,
        norm_type=cfg.norm_type,
        head_norm=cfg.head_norm,
        act_type=cfg.act_type,
        se=cfg.se,
        post_threshold=cfg.post_threshold,
        evaluate_froc=not cfg.no_froc,
        froc_iou_threshold=cfg.froc_iou_threshold,
        confidence_log_interval=cfg.confidence_log_interval,
        debug_target_stats=cfg.debug_target_stats,
        debug_target_stats_interval=cfg.debug_target_stats_interval,
    )

    checkpoint_cb = StartEpochModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename=str(cfg.get("checkpoint_filename", "epoch={epoch:03d}-val_loss={val/loss:.4f}")),
        monitor=str(cfg.get("checkpoint_monitor", "val/loss")),
        mode=str(cfg.get("checkpoint_mode", "min")),
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=cfg.get("checkpoint_every_n_epochs", None),
        start_epoch=cfg.get("checkpoint_start_epoch", 1),
    )
    logger = False
    if cfg.use_wandb:
        wandb.login(key="5eb716fd87389d240533319f8751488e37103d23")
        logger = WandbLogger(
            project=cfg.wandb_project,
            name=cfg.wandb_name or cfg.experiment_name,
            entity=cfg.wandb_entity,
            save_dir="./wandb2",
            mode=cfg.wandb_mode,
            log_model=cfg.wandb_log_model,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    callbacks = [checkpoint_cb, LearningRateMonitor(logging_interval="epoch")]
    if cfg.get("val_froc_start_epoch", None) is not None:
        callbacks.append(
            ValidationScheduleCallback(
                start_epoch=cfg.get("val_froc_start_epoch", 1),
                every_n_epochs=cfg.get("check_val_every_n_epoch", 1),
                before_start_every_n_epochs=cfg.get("val_froc_before_start_every_n_epoch", None),
            )
        )

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
        callbacks=callbacks,
        default_root_dir=str(exp_dir),
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
        val_check_interval=_maybe_number(cfg.val_check_interval),
        check_val_every_n_epoch=initial_check_val_every_n_epoch,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
    )

    if cfg.test_only:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.checkpoint, weights_only=False)
        return

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_cb.best_model_path or "best", weights_only=False)


@hydra.main(version_base=None, config_path="conf", config_name="train_lightning")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
