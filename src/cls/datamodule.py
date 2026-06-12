import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from hydra.utils import instantiate
from src.collator import Collator
import src.builder as builder

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.collator = Collator()

        self.train_transforms = instantiate(cfg.train_transforms)
        self.test_transforms = instantiate(cfg.test_transforms)
        self.dicom = cfg.dicom

    def setup(self, stage=None):
        pass # Transforms are already instantiated by Hydra and passed to init

    def train_dataloader(self):
        dataset = builder.build_dataset(self.cfg, split='train', transforms=self.train_transforms)
        
        collate_fn = self.collator if self.cfg.get("dataset_type", "3d") != "2d" else None
            
        if self.cfg.get("dataset_type", "3d") != "2d" and self.cfg.weighted_sample:
            sampler = dataset.get_sampler()
        else:
            sampler = None

        return DataLoader(
            dataset, 
            sampler=sampler,
            batch_size=self.cfg.batch_size, 
            pin_memory=True,
            drop_last=True,
            shuffle=True if sampler is None else False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=True if self.cfg.num_workers > 0 else False
        )

    def val_dataloader(self):
        dataset = builder.build_dataset(self.cfg, split='val', transforms=self.train_transforms)
        collate_fn = self.collator if self.cfg.get("dataset_type", "3d") != "2d" else None

        return DataLoader(
            dataset, 
            batch_size=self.cfg.batch_size, 
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=True if self.cfg.num_workers > 0 else False
        )

    def test_dataloader(self):
        dataset = builder.build_dataset(self.cfg, split='test', transforms=self.test_transforms)
        collate_fn = self.collator if self.cfg.get("dataset_type", "3d") != "2d" else None

        return DataLoader(
            dataset, 
            batch_size=self.cfg.batch_size, 
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=4,
            persistent_workers=True if self.cfg.num_workers > 0 else False
        )

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("conf/data/default.yaml")
    device = "cpu"
    dm = DataModule(cfg, device)
    
    # Example: Iterating through the validation dataloader
    val_loader = dm.val_dataloader()
    for batch in val_loader:
        # Assuming the dataset returns a dictionary or tuple of (images, labels)
        print("Successfully loaded a batch")
        break
