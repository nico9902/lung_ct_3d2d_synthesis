import numpy as np
import torch
import torch.nn as nn
import src.lightning_model as lightning_model
from src.networks import SliceAttentionNetwork, BackboneClassifier2D, SelfPatch_softSlice, SelfEncoderLayerPatch_SoftSlice, DinoV2Patch_SoftSlice
from src.AFFNet import AFFNet
from src.detector import SwinRPNMIL
from src.dataset import LIDCIDRIDataset, LIDCRIDRIDataset2D
import torchvision.transforms as T

def build_lightning_model(cfg, ckpt=None):
    if cfg.stage == "classify":
        # Se non c'è checkpoint, crea istanza normale
        if ckpt is None:
            model = lightning_model.LungLitModel(cfg)
        else:
            # Se c'è checkpoint, carica dalla classe
            model = lightning_model.LungLitModel.load_from_checkpoint(ckpt, cfg=cfg)
    elif cfg.stage == "detect":
        if ckpt is None:
            model = lightning_model.DetectionLitModel(cfg)
        else:
            model = lightning_model.DetectionLitModel.load_from_checkpoint(ckpt, cfg=cfg)
    else:
        raise NotImplementedError(
            f"Lightning model not implemented for {cfg.stage} \n"
        )

    return model

def build_dataset(cfg, split, transforms):
    dataset_type = cfg.get("dataset_type", "3d")
    
    if dataset_type == "2d":
        return LIDCRIDRIDataset2D(
            csv_file=cfg.csv_file,
            processed_dir=cfg.processed_dir,
            split=split,
            transform=transforms
        )
    else:
        return LIDCIDRIDataset(
            root_dir=cfg.data_dir,
            processed_dir=cfg.processed_dir,
            csv_file=cfg.csv_file,
            split=split,
            transform=transforms,
            dicom=cfg.dicom,
            return_mask=cfg.get("return_mask", False)
        )

def build_model(cfg):
    # get model
    if cfg.network_name == "SliceAttentionNetwork":
        return SliceAttentionNetwork(
            num_classes=cfg.num_classes, 
            feature_dim=cfg.feature_dim, 
            backbone_name=cfg.backbone_name, 
            freeze_half=cfg.freeze_half,
            enable_segmentation=cfg.enable_segmentation,
            segmentation_feature_dim=cfg.segmentation_feature_dim,
            attention_type=cfg.attention_type,
            max_slices=cfg.max_slices,
        )
    elif cfg.network_name == "SelfPatch_softSlice":
        return SelfPatch_softSlice(
            backbone_name = cfg.backbone_name,
            patch_size = cfg.patch_size,
            img_size = cfg.img_size,
            num_classes = cfg.num_classes,
            freeze_half = cfg.freeze_half,
            num_heads = cfg.num_heads
        )
    elif cfg.network_name == "SelfEncoderLayerPatch_SoftSlice":
        return SelfEncoderLayerPatch_SoftSlice(
            backbone=cfg.backbone_name,
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            num_classes = cfg.num_classes,
            num_heads = cfg.num_heads
        )
    elif cfg.network_name == "DinoV2Patch_SoftSlice":
        return DinoV2Patch_SoftSlice(
            patch_size=cfg.patch_size,
            img_size=cfg.img_size,
            num_classes = cfg.num_classes,
        )
    elif cfg.network_name == "BackboneClassifier2D":
        return BackboneClassifier2D(num_classes=cfg.num_classes, model_name=cfg.backbone_name, freeze_half=cfg.freeze_half)
    elif cfg.network_name == "AFFNet":
        return AFFNet()
    elif cfg.network_name == "Detection3D":
        return SwinRPNMIL()

    # elif cfg.network_name == "DinoDetector":
    #     # Load Pre-trained DINOv2 Model
    #     dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    #     # dinov2_vits14.eval()

    #     # Compute feature_dim
    #     transform = T.Compose([
    #         T.ToPILImage(),
    #         T.Resize((224, 224)),
    #         T.ToTensor(),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
        
    #     dummy_slice = np.zeros((504, 504, 3), dtype=np.uint8)
    #     dummy_tensor = transform(dummy_slice).unsqueeze(0)
    #     with torch.no_grad():
    #         dummy_feature = dinov2_vits14(dummy_tensor).flatten()
    #     feature_dim = dummy_feature.shape[0]
        
    #     return DinoDetector(input_size=feature_dim, dinov2_model=dinov2_vits14)
    else:
        raise NotImplementedError(f"Model {cfg.network_name} is not implemented")

import src.losses as custom_losses

def build_loss(cfg):
    # get loss function
    if "loss" in cfg:
        loss_name = cfg.loss.name
        loss_kwargs = {k: v for k, v in cfg.loss.items() if k != 'name'}
        
        # Try custom losses first
        if hasattr(custom_losses, loss_name):
            loss_fn = getattr(custom_losses, loss_name)
        else:
            loss_fn = getattr(torch.nn, loss_name)
            
        loss_function = loss_fn(**loss_kwargs)
        return loss_function
    else:
        return None

def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]

    if "optimizer" in cfg:
        optimizer_name = cfg.optimizer.name
        optimizer_kwargs = {k: v for k, v in cfg.optimizer.items() if k != 'name'}
        
        # Ensure lr is passed from cfg.model.lr or optimizer_kwargs
        lr = cfg.get("lr", 1e-4) # default fallback
        
        optimizer_fn = getattr(torch.optim, optimizer_name)
        optimizer = optimizer_fn(params, lr=lr, **optimizer_kwargs)
        return optimizer
    else:
        return None

def build_scheduler(cfg, optimizer):
    if "scheduler" in cfg:
        scheduler_name = cfg.scheduler.name
        scheduler_kwargs = {k: v for k, v in cfg.scheduler.items() if k != 'name'}
        
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_name)
        scheduler = scheduler_fn(optimizer, **scheduler_kwargs)
        return scheduler
    else:
        return None
