from omegaconf import DictConfig, OmegaConf
import hydra
import sys
import os

# add project root to path so we can import modules from src
sys.path.append(os.getcwd())

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import pytorch_lightning as pl
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import EarlyStopping
    import flatten_dict
    import src.datamodule as datamodule
    import src.builder as builder
    import wandb
    import torch
    import re
    
    # set seed
    if "seed" in cfg:
        seed_everything(cfg.seed)
    
    # saving checkpoints and logging with wandb.
    flat_config = flatten_dict.flatten(cfg, reducer="dot")
    save_dir = os.path.join(cfg.exp.base_dir, cfg.exp.name)

    # wandb logger
    wandb.login(key="5eb716fd87389d240533319f8751488e37103d23")
    wandb_logger = pl.loggers.WandbLogger(
        project=cfg.project_name,
        name=cfg.exp.name,
        entity="domenico-paolo1999",
        save_dir="./wandb2",
        log_model="all",
    )
    wandb_logger.log_hyperparams(flat_config)

    # if os.environ.get("LOCAL_RANK", None) is not None:
    # os.environ["WANDB_DIR"] = wandb.run.dir
    global_rank = os.environ.get("JSM_NAMESPACE_RANK")
    if global_rank == 0:
        wandb.define_metric(config.monitor.metric, summary="best", goal="maximize")

        # merge sweep configs
        run = wandb_logger.experiment
        run_config = [f"{k}={v}" for k, v in run.config.items()]
        run_config = OmegaConf.from_dotlist(run_config)
        config = OmegaConf.merge(config, run_config)  # update defaults to CLI

    # call backs
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir,
        every_n_epochs=1,
        save_top_k=1,   # cambiato da -1 ad 1 per salvare solo il miglior modello
        monitor=cfg.monitor.metric,
        mode=cfg.monitor.mode,
    )
    early_stop_callback = EarlyStopping(
        monitor="val/loss",    # metric to monitor
        min_delta=0.00,        # minimum change to qualify as improvement
        patience=10,           # stop after N epochs without improvement
        verbose=True,          # print messages
        mode="min"             # "min" for loss, "max" for accuracy
    )
    callbacks = [
        lr_monitor,
        checkpoint_callback,
        early_stop_callback,
    ]

    # Instantiate DataModule
    print("Instantiating DataModule...")
    dm = datamodule.DataModule(cfg.data)

    # # Compute pos_weight for nodule segmentation (if segmentation is enabled)
    # if cfg.model.enable_segmentation and cfg.data.return_mask:
    #     total_pos = 0
    #     total_neg = 0

    #     for batch in dm.train_dataloader():
    #         masks = batch[3]
    #         masks = masks.float()
    #         total_pos += masks.sum(dim=1).sum().item()
    #         total_neg += masks.numel() - masks.sum(dim=1).sum().item()

    #     pos_weight = total_neg / total_pos
    #     print("pos_weight =", pos_weight)
    #     pos_weight = torch.tensor([pos_weight])
    #     cfg.model.pos_weight = pos_weight

    # if .ckpt given then only testing without training
    if cfg.model.ckpt is not None and cfg.model.ckpt.endswith(".ckpt"):
        # Load the model checkpoint
        print(f"Loading model from {cfg.model.ckpt}")
        best_epochs = (
            re.search(r"epoch=(\d+).*?-step", cfg.model.ckpt).group(1)
            if re.search(r"epoch=(\d+).*?-step", cfg.model.ckpt)
            else "Not found"
        )
        # cfg.trainer.max_epochs = int(best_epochs)
        ckpt = cfg.model.ckpt
    else:
        ckpt = None

    # Instantiate Model
    print("Instantiating Model...")
    model = builder.build_lightning_model(cfg, ckpt=ckpt)

    model.save_dir = save_dir
    # PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        default_root_dir=save_dir,  
        devices=cfg.trainer.devices,
        logger=wandb_logger,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=cfg.trainer.limit_val_batches,
        callbacks=callbacks,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        num_sanity_val_steps=0,
    )
    
    if ckpt == None:
        # training from scratch
        print("Training from scratch")
        trainer.fit(model=model, datamodule=dm)
        trainer.test(datamodule=dm, ckpt_path="best")
    else:
        if cfg.trainer.max_epochs == 0:
            # only testing
            print(f"Testing {ckpt}")
            trainer.test(datamodule=dm, model=model)
        else:
            # fine-tuning
            print("Fine-tuning")
            trainer.fit(model=model, datamodule=dm)
            trainer.test(datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()
