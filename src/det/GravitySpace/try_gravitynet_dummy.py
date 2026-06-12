import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.det.GravitySpace.lightning_model import GravitySpaceLitModel
from src.det.GravitySpace.transforms import Custom_Resize, CustomToTensor, RepeatChannels, ExtractAnnotationCenter
from torchvision.transforms import Compose

def try_gravitynet_dummy():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Define image size and dummy data
    H, W = 352, 480
    S = 100 # Slices
    C = 1 # Initial channels
    
    # Create dummy volume [S, H, W]
    dummy_slices = np.zeros((S, H, W), dtype=np.float32)
    dummy_annotations = np.zeros((S, H, W), dtype=np.uint8)
    
    # Add a synthetic "nodule" (a 10x10 square) in the first slice
    dummy_annotations[0, 100:110, 150:160] = 1
    # Add another one in the second slice
    dummy_annotations[1, 200:210, 300:310] = 1
    # Add another one in the second slice
    dummy_annotations[1, 100:110, 150:160] = 1
    
    sample = {
        'slices': dummy_slices,
        'annotations': dummy_annotations,
        'slicenames': ["dummy_slice_1", "dummy_slice_2"]
    }
    
    # 2. Apply Transforms (same as in datamodule.py)
    transforms = Compose([
        # Custom_Resize((H, W)), # Already H, W
        CustomToTensor(),
        RepeatChannels(repeats=3), 
                                   # Actually MyResNet_models usually expects 3 if pretrained
        ExtractAnnotationCenter(max_nodules=5)
    ])
    
    # Let's check lightning_model.py for what it expects
    # The backbone might expect 3 channels.
    
    # Re-check lightning_model.py backbone init
    # In lightning_model.py:
    # self.model = GravitySpaceAttentionNet(..., backbone=backbone, ...)
    # If backbone is ResNet-18, MyResNet_models is called.
    
    sample = transforms(sample)
    
    # Add batch dimension
    slices = sample['slices'].unsqueeze(0).to(device) # [1, S, C, H, W]
    annotations = torch.from_numpy(sample['annotations']).unsqueeze(0).to(device) # [1, S, MaxNodules, 4]
    
    print(f"Input slices shape: {slices.shape}")
    print(f"Input annotations shape: {annotations.shape}")

    # 3. Instantiate Model
    model = GravitySpaceLitModel(
        backbone="ResNet-18",
        pretrained=False,
        attention="enhanced",
        window_size=3,
        sampling=1,
        hidden_dim=64,
        image_shape=(H, W),
        anchor_config="grid-10"
    ).to(device)
    
    # Let's log what fm_shape the model thinks it has
    print(f"Model FM shape: {model.fm_shape}")
    
    model.eval()
    
    # 4. Forward Pass
    with torch.no_grad():
        # Let's check the backbone output shape manually
        backbone_out = model.model.feature_extractor(slices)
        print(f"Backbone output shape: {backbone_out.shape}")
        
        classifications, regressions = model.model(slices)
        
        print("\nForward Pass Results:")
        print(f"Classifications shape: {classifications.shape}") # [B, S, A, 2]
        print(f"Regressions shape: {regressions.shape}")         # [B, S, A, 2]
        
        # 5. Compute Loss
        cls_loss, reg_loss = model.criterion(
            images_batch=slices,
            classifications_batch=classifications,
            regressions_batch=regressions,
            gravity_points=model.gravity_points,
            annotations_batch=annotations
        )
        
        print(f"\nLoss values:")
        print(f"Classification Loss: {cls_loss.item():.4f}")
        print(f"Regression Loss: {reg_loss.item():.4f}")
        print(f"Total Loss: {(cls_loss + reg_loss).item():.4f}")

if __name__ == "__main__":
    try_gravitynet_dummy()
