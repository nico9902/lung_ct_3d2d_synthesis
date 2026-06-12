import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Ensure project root is in path
sys.path.append(os.getcwd())

from src.det.GravitySpace.lightning_model import GravitySpaceLitModel
from src.det.GravitySpace.datamodule import LIDC_DataModule

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import torch
    import wandb
    import numpy as np
    import pandas as pd
    import re
    import pickle
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger

    # Create experiment directory
    exp_dir = os.path.join(cfg.exp.base_dir, cfg.exp.name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config for reproducibility
    config_out_dir = os.path.join(cfg.exp.base_dir, cfg.exp.name, "config.pkl")
    pickle.dump(cfg, open(config_out_dir, "wb"))

    # Set seed
    if cfg.seed:
        pl.seed_everything(cfg.seed)

    # 1. Setup Logging (Wandb)
    wandb_logger = None
    if cfg.exp.wandb.enabled:
        wandb.login(key="5eb716fd87389d240533319f8751488e37103d23")
        wandb_logger = pl.loggers.WandbLogger(
            project=cfg.project_name,
            name=cfg.exp.name,
            entity="domenico-paolo1999",
            save_dir="./wandb2",
            log_model="all",
        )
        # Log flat configuration
        flat_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_logger.log_hyperparams(flat_config)

    # 2. Scanning Patients for Splitting
    df = pd.read_csv(cfg.data.csv_path)
    all_cases = df["patient_id"].unique()
    print(f"Scanning {len(all_cases)} patients...")
    
    if len(all_cases) == 0:
        print(f"Error: No patients found in {cfg.data.images_dir}")
        return

    # Split cases
    train_cases = df[df["split"] == "train"]["patient_id"].unique()
    val_cases = df[df["split"] == "val"]["patient_id"].unique()
    test_cases = df[df["split"] == "test"]["patient_id"].unique()
    print(f"Training on {len(train_cases)} cases, validating on {len(val_cases)} cases, testing on {len(test_cases)} cases.")

    # 3. Setup DataModule
    dm = LIDC_DataModule(
        images_dir=cfg.data.images_dir,
        annotations_dir=cfg.data.annotations_dir,
        train_cases=train_cases,
        val_cases=val_cases,
        test_cases=test_cases,
        batch_size=cfg.data.batch_size,
        image_size=tuple(cfg.data.image_size),
        view=cfg.data.view,
        num_workers=cfg.data.num_workers,
        precomputed_centers_dir=cfg.data.get("precomputed_centers_dir", None),
        precomputed_slices_dir=cfg.data.get("precomputed_slices_dir", None),
        use_subvolumes=cfg.data.get("use_subvolumes", False),
        subvolume_depth=cfg.data.get("subvolume_depth", 32),
        subvolume_stride=cfg.data.get("subvolume_stride", 16),
        val_subvolume_stride=cfg.data.get("val_subvolume_stride", None),
        test_subvolume_stride=cfg.data.get("test_subvolume_stride", None),
        positive_fraction=cfg.data.get("positive_fraction", 0.7),
        samples_per_epoch=cfg.data.get("samples_per_epoch", None),
        input_mode=cfg.data.get("input_mode", "2d"),
        context_slices=cfg.data.get("context_slices", 3)
    )
    dm.setup()

    print("train_ds", len(dm.train_ds))
    print("val_ds", len(dm.val_ds))
    print("test_ds", len(dm.test_ds))

    # 4. Setup Lightning Model
    # data.image_size follows OpenCV dsize convention: (width, height).
    # Anchor generation and model post-processing expect image_shape as (height, width).
    model_image_shape = (cfg.data.image_size[1], cfg.data.image_size[0])

    # Extract params from model config
    model = GravitySpaceLitModel(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        attention=cfg.model.attention,
        window_size=cfg.model.window_size,
        sampling=cfg.model.sampling,
        hidden_dim=cfg.model.hidden_dim,
        lr=cfg.model.lr,
        anchor_config=cfg.model.anchor_config,
        distance=cfg.inference.get("distance", 8),
        nms_radius=cfg.inference.get("nms_box_radius", 8),
        nms_z_radius=cfg.inference.get("nms_z_radius", 1),
        nms_2d_iou_threshold=cfg.inference.get("nms_2d_iou_threshold", 0.5),
        nms_3d_iou_threshold=cfg.inference.get("nms_3d_iou_threshold", 0.1),
        score_threshold=cfg.inference.score_threshold,
        image_shape=model_image_shape,
        alpha=cfg.model.loss.alpha,
        gamma=cfg.model.loss.gamma,
        hook=cfg.model.loss.hook,
        hook_gap=cfg.model.loss.hook_gap,
        base_dir=cfg.exp.base_dir,
        exp_name=cfg.exp.name,
        chunk_size=cfg.model.get('chunk_size', 100),
        input_channels=3,
        save_qualitative=cfg.inference.get("save_qualitative", False),
        qualitative_max_images=cfg.inference.get("qualitative_max_images", 10),
        qualitative_dir=cfg.inference.get("qualitative_dir", None),
        qualitative_only_with_findings=cfg.inference.get("qualitative_only_with_findings", True),
        qualitative_score_threshold=cfg.inference.get("qualitative_score_threshold", None),
        qualitative_show_fp_text=cfg.inference.get("qualitative_show_fp_text", False),
        froc_normalization=cfg.inference.get("froc_normalization", "slice")
    )

    # 5. Callbacks
    exp_dir = os.path.join(cfg.exp.base_dir, cfg.exp.name)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=exp_dir,
        every_n_epochs=1,
        save_top_k=1,   # cambiato da -1 ad 1 per salvare solo il miglior modello
        monitor=cfg.monitor.metric,
        mode=cfg.monitor.mode,
    )
    # early_stop_callback = EarlyStopping(
    #     monitor="val/loss",    # metric to monitor
    #     min_delta=0.00,        # minimum change to qualify as improvement
    #     patience=10,           # stop after N epochs without improvement
    #     verbose=True,          # print messages
    #     mode="min"             # "min" for loss, "max" for accuracy
    # )
    callbacks = [
        lr_monitor,
        checkpoint_callback,
        # early_stop_callback,
    ]

    # 6. Trainer Execution
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        strategy=cfg.trainer.strategy,
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=exp_dir,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )

    # if .ckpt given then only testing without training
    if cfg.model.ckpt is not None and cfg.model.ckpt.endswith(".ckpt"):
        # Load the model checkpoint
        print(f"Loading model from {cfg.model.ckpt}")
        # best_epochs = (
        #     re.search(r"epoch=(\d+).*?-step", cfg.model.ckpt).group(1)
        #     if re.search(r"epoch=(\d+).*?-step", cfg.model.ckpt)
        #     else "Not found"
        # )
        # cfg.trainer.max_epochs = int(best_epochs)
        ckpt = cfg.model.ckpt
    else:
        ckpt = None

    if ckpt is not None:
        print("Starting testing...")
        trainer.test(model, dm, ckpt_path=ckpt)
    else:
        print("Starting training...")
        trainer.fit(model, dm)

        print("Testing...")
        trainer.test(model, dm, ckpt_path=checkpoint_callback.best_model_path)

if __name__ == "__main__":
    main()
