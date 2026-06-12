import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import math
import numpy as np
import timm
from peft import LoraConfig, get_peft_model
from transformers import Dinov2Model

def unfreeze_last_blocks(model, n_blocks=2):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze last n transformer blocks
    for p in model.blocks[-n_blocks:].parameters():
        p.requires_grad = True

    # unfreeze final norm
    for p in model.norm.parameters():
        p.requires_grad = True

    print(f"Unfroze last {n_blocks} blocks")

def freeze_all(model):
    """
    Freeze all parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = False

    print("All parameters frozen.")

def freeze_half_layers(model):
    """
    Freeze the first half of the parameters in the model.
    """
    params = list(model.parameters())
    num_params = len(params)
    half = math.ceil(num_params / 2)
    
    for i, p in enumerate(params):
        if i < half:
            p.requires_grad = False
        else:
            p.requires_grad = True
    
    print(f"Frozen {half} out of {num_params} parameter sets.")

def model_rgb2gray(model): 
   """
       Function to convert the first layer of a model to process grayscale images
   """
   # identify first layer (should be a Conv2d)
   first_layer = model 
   while len(list(first_layer.children())) > 0: 
      first_layer = list(first_layer.children())[0] 

   # convert first layer to process grayscale image 
   if hasattr(first_layer, 'in_channels') and first_layer.in_channels == 3:
       first_layer.in_channels = 1        
       first_layer.weight = torch.nn.Parameter(first_layer.weight.sum(1, keepdim=True))
       print(f"Converted first layer {type(first_layer)} to 1 channel input.")
   else:
       print(f"Could not convert layer {type(first_layer)} to grayscale or already converted.")

class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class TransformerEncoderWithWeights(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights_all_layers = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_all_layers.append(attn_weights)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_weights_all_layers
    
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None,
                 need_weights=True,
                 average_attn_weights=True):
        super(CustomTransformerEncoderLayer, self).__init__(d_model,
                                                            nhead,
                                                            dim_feedforward,
                                                            dropout,
                                                            activation,
                                                            layer_norm_eps,
                                                            batch_first,
                                                            norm_first,
                                                            device,
                                                            dtype)

        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        self.attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x, self.attn_weights = self.self_attn(x, x, x,
                                              attn_mask=attn_mask,
                                              key_padding_mask=key_padding_mask,
                                              need_weights=self.need_weights,
                                              average_attn_weights=self.average_attn_weights)
        return self.dropout1(x)

    def get_attn_weights(self):
        return self.attn_weights

def vgg16_features():
    model = torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    print("Model used: VGG16")

    model = nn.Sequential(*list(model.children())[:-1]) # Keep only features

    return model

def vgg16_features_truncated():
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    print("Model used: VGG16 truncated")

    features = model.features
    truncated = nn.Sequential(*list(features.children())[:16])

    return truncated

def efficientnetb0_features():
    model = torchvision.models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    print("Model used: EffcientNetB0")

    model = nn.Sequential(*list(model.children())[:-1]) # Keep only features

    return model

def resnet18_features():
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    print("Model used: ResNet18")

    # Remove the final fully connected layer and global avg pool
    model = nn.Sequential(*list(model.children())[:-2])  # Keep layers up to last conv

    return model

def get_backbone_feature_dim(backbone_name):
    """Get the output feature dimension for a given backbone."""
    feature_dims = {
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_b3": 1536,
        "efficientnet_b4": 1792,
        "efficientnet_b5": 2048,
        "efficientnet_b6": 2304,
        "efficientnet_b7": 2560,
        "efficientnet_lite0": 1280,
        "efficientnet_v2_s": 1280,
        "efficientnet_v2_m": 1280,
        "efficientnet_v2_l": 1280,
        "resnet18": 512,
        "resnet50": 2048,
        "vgg16": 512,
        "convnext_tiny": 768,
        "densenet121": 1024,
        "densenet161": 2208,
        "densenet169": 1664,
        "densenet201": 1920,
        "googlenet": 1024,
    }
    
    if backbone_name not in feature_dims:
        raise ValueError(f"Unknown backbone: {backbone_name}. Please add its feature dimension to get_backbone_feature_dim()")
    
    return feature_dims[backbone_name]

def get_backbone(backbone_name="efficientnet_b0", freeze_half=False, **kwargs):
    if backbone_name == "efficientnet_b0":
        model = efficientnetb0_features()
    elif backbone_name == "resnet18":
        model = resnet18_features()
    elif backbone_name == "vgg16":
        model = vgg16_features()
    elif backbone_name == "vgg16_truncated":
        model = vgg16_features_truncated()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Modify for grayscale images if needed
    model_rgb2gray(model)
    
    if freeze_half:
        freeze_half_layers(model)
    
    return model

class SliceAttentionNetwork(nn.Module):
    def __init__(self, num_classes=2, feature_dim=None, backbone_name="efficientnet_b0", freeze_half=False, 
                 enable_segmentation=False, segmentation_feature_dim=256, 
                 attention_type="soft", max_slices=1000, **kwargs):
        super(SliceAttentionNetwork, self).__init__()
        
        # Automatically get feature_dim from backbone if not provided
        if feature_dim is None:
            feature_dim = get_backbone_feature_dim(backbone_name)
        self.feature_dim = feature_dim
        self.attention_type = attention_type
        
        # Load pretrained backbone
        self.backbone = get_backbone(backbone_name, freeze_half=freeze_half, **kwargs) 
        
        # Attention mechanism selection
        if attention_type == "soft":
            self.attention = nn.Linear(feature_dim, 1)
        elif attention_type == "transformer_encoder":
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, max_slices + 1, feature_dim))
            encoder_layer = TransformerEncoderLayerWithWeights(d_model=feature_dim, nhead=8, batch_first=True)
            self.transformer_encoder = TransformerEncoderWithWeights(encoder_layer, num_layers=2)
        elif attention_type in ["self_attention_cls", "cross_attention_central"]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, max_slices + 1, feature_dim))
            self.mha = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")
        
        # Final MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes - 1 if num_classes == 2 else num_classes)
        )
        
        # Segmentation head (optional)
        self.enable_segmentation = enable_segmentation
        if self.enable_segmentation:
            # Segmentation decoder that adapts to different backbone output sizes
            # The decoder will upsample features back to original input resolution
            
            # Convolutional decoder with upsampling
            # Input features are (B*S, feature_dim, H', W')
            self.seg_conv_decoder = nn.Sequential(
                # First, reduce channels to save parameters
                nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # Upsample blocks
                # H', W' -> 2H', 2W'
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                # 2H', 2W' -> 4H', 4W'
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # 4H', 4W' -> 8H', 8W'
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                # 8H', 8W' -> 16H', 16W'
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # Final 1x1 conv to get single channel output
                nn.Conv2d(16, 1, kernel_size=1),
            )
            self.segmentation_feature_dim = feature_dim

    def forward(self, x, return_attn_weights=False):
        # Input shape: (Batch, Slices, Channels, Height, Width)
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        
        # Extract spatial backbone representations for each slice
        spatial_features = self.backbone(x)  # (B*S, feature_dim, H', W')
        
        # Global Average Pooling for classification
        features = F.adaptive_avg_pool2d(spatial_features, (1, 1))
        features = features.view(b, s, -1)  # (B, s, feature_dim)
        
        # Aggregate features across slices
        attn_weights = None
        if self.attention_type == "soft":
            # Soft attention aggregation for classification
            attn_weights = F.softmax(self.attention(features), dim=1)  # (B, S, 1)
            aggregated = torch.sum(attn_weights * features, dim=1)  # (B, feature_dim)
        
        elif self.attention_type == "transformer_encoder":
            # Transformer aggregation with CLS token
            cls_token = self.cls_token.expand(b, -1, -1)  # (B, 1, feature_dim)
            features = torch.cat([cls_token, features], dim=1)  # (B, S+1, feature_dim)
            
            # Add positional embedding
            pos_embed = self.pos_embed[:, :features.size(1), :]
            features = features + pos_embed
            
            # Transformer encoder forward
            features, attn_weights_all_layers = self.transformer_encoder(features)
            aggregated = features[:, 0, :]  # Extract CLS token output (B, feature_dim)
            attn_weights = attn_weights_all_layers # list of attn_weights per layer
        
        elif self.attention_type == "self_attention_cls":
            # Transformer-like self-attention with CLS token
            cls_token = self.cls_token.expand(b, -1, -1)  # (B, 1, feature_dim)
            features = torch.cat([cls_token, features], dim=1)  # (B, S+1, feature_dim)
            
            # Add positional embedding (truncate if necessary)
            pos_embed = self.pos_embed[:, :features.size(1), :]
            features = features + pos_embed
            
            # Multi-head attention
            features, attn_weights = self.mha(features, features, features, average_attn_weights=True)
            aggregated = features[:, 0, :]  # Extract CLS token output (B, feature_dim)
            attn_weights = attn_weights[:, 0, 1:]
            
        elif self.attention_type == "cross_attention_central":
            # Cross-attention: central slice as query, all slices as key/value
            
            central_idx = s // 2
            
            # Query = central slice
            query = features[:, central_idx:central_idx+1, :]  # (B, 1, feature_dim)
            
            # Key/Value = all slices (NO CLS)
            kv = features  # (B, S, feature_dim)
            
            # Positional embeddings (only for real slices)
            kv = kv + self.pos_embed[:, :s, :]
            query = query + self.pos_embed[:, central_idx:central_idx+1, :]
            
            # Cross-attention
            aggregated, attn_weights = self.mha(query, kv, kv, average_attn_weights=True)
            
            # Remove dimension 1
            aggregated = aggregated.squeeze(1)  # (B, feature_dim)
        
        # Classification output
        classification_logits = self.mlp(aggregated)
        
        # Segmentation output (if enabled)
        if self.enable_segmentation:
            # Upsample through decoder directly from spatial features
            seg_output = self.seg_conv_decoder(spatial_features)  # (B*S, 1, H", W")
            
            # Resize to match input resolution if needed
            if seg_output.shape[-2:] != (h, w):
                seg_output = F.interpolate(seg_output, size=(h, w), mode='bilinear', align_corners=False)
            
            seg_output = seg_output.view(b, s, 1, h, w)  # (B, S, 1, H, W)
            
            if return_attn_weights:
                return (classification_logits, seg_output), attn_weights
            
            return classification_logits, seg_output
        
        if return_attn_weights:
            return classification_logits, attn_weights
        
        return classification_logits

class SelfPatch_softSlice(nn.Module):
    def __init__(self, backbone_name, patch_size, img_size, num_classes, freeze_half, num_heads=8, **kwargs):
        super(SelfPatch_softSlice, self).__init__()
        
        # Load pretrained backbone
        self.backbone = get_backbone(backbone_name, freeze_half=freeze_half, **kwargs) 

        # self.p = patch_size
        # self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        with torch.no_grad():
            dummy = self.backbone(torch.rand(1,1,img_size[0],img_size[1]))
            self.embedding_dim = dummy.shape[1]
            self.p = int(img_size[0]/dummy.shape[2])
            self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice + 1, self.embedding_dim))

        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads, batch_first=True)

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.soft_attn = nn.Linear(self.embedding_dim, 1)

        # Custom init for soft attention layer
        with torch.no_grad():
            self.soft_attn.weight.fill_(0)
            self.soft_attn.bias.fill_(1)

        if num_classes == 2:
            num_classes = 1
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, c, h, w = x.shape

        # x = x.unfold(3, self.p, self.p).unfold(4, self.p, self.p)
        # # shape [batch_size, n_slice, h/p, w/p, p, p]

        # x = x.reshape(-1, 1, self.p, self.p)
        # # num_patch = n_slice*h*w/p^2
        # # shape [batch_size*num_patch, 1, p, p]

        # x = self.backbone(x)                     # [batch_size*num_patch, D, H', W']
        # x = F.adaptive_avg_pool2d(x, 1)          # [batch_size*num_patch, D, 1, 1]
        # x = x.flatten(1)                         # [batch_size*num_patch, D]

        x = x.view(batch_size*n_slice, c, h, w)
        x = self.backbone(x)   # [B*S, D, H', W']
        
        x = x.flatten(2).transpose(1,2)  # [B*S, num_patch, D]
        
        # x = x.view(batch_size*n_slice, self.patches_per_slice, self.embedding_dim)
        # # [batch_size*num_slices, patches_per_slice, D]

        class_token = self.class_token.expand(batch_size*n_slice, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        x, a_patches = self.attn(x, x, x, need_weights=return_attn_weights, average_attn_weights=True)

        x = x[:, 0].view(batch_size, n_slice, -1)

        a_slice = F.softmax(self.soft_attn(x), dim=1)
        # a shape [batch_size, n_slice, 1]

        x = torch.sum(x * a_slice, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, (a_patches, a_slice)
        else:
            return x

class SelfEncoderLayerPatch_SoftSlice(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_heads):
        super(SelfEncoderLayerPatch_SoftSlice, self).__init__()

        if backbone is not None:
            self.backbone = get_backbone(backbone, freeze_half=True)
            with torch.no_grad():
                dummy = self.backbone(torch.rand(1,1,img_size[0],img_size[1]))
                dummy = F.adaptive_avg_pool2d(dummy,1)
                self.embedding_dim = dummy.shape[1]
        else:
            self.backbone = None
            self.embedding_dim = 768
            self.embedding_layer = nn.Linear(patch_size*patch_size, self.embedding_dim)

        self.p = patch_size
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice + 1, self.embedding_dim))

        self.attn = TransformerEncoderLayerWithWeights(self.embedding_dim, num_heads, batch_first=True)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.soft_attn = nn.Linear(self.embedding_dim, 1)

        if num_classes == 2:
            num_classes = 1
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, c, h, w = x.shape

        x = x.unfold(3, self.p, self.p).unfold(4, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]

        if self.backbone is not None:
            chunk_size = 512
            x_ = []
            for i in range(0, x.shape[0], chunk_size):
                x_chunk = x[i:i+chunk_size]
                # shape [chunk_size, 1, p, p]
                x_chunk = self.backbone(x_chunk)
                # shape [chunk_size, embedding_dim, p-1, p-1]
                x_chunk = F.adaptive_avg_pool2d(x_chunk, 1)
                # shape [chunk_size, embedding_dim, 1, 1]
                x_chunk = x_chunk.flatten(1)
                # shape [chunk_size, embedding_dim]
                x_.append(x_chunk)
            x = torch.cat(x_, dim=0)
        else:
            x = x.flatten(2)
            # shape [batch_size*num_patch, patch_size*patch_size]
            x = self.embedding_layer(x)
            # shape [batch_size*num_patch, embedding_dim]

        x = x.view(batch_size * n_slice, self.patches_per_slice, self.embedding_dim)
        # shape [batch_size*num_slices, patches_per_slice, embedding_dim]

        class_token = self.class_token.expand(batch_size * n_slice, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        x, a_patches = self.attn(x) #, return_attn_weights=return_attn_weights)

        x = x[:, 0].view(batch_size, n_slice, -1)

        a_slice = F.softmax(self.soft_attn(x), dim=1)
        # a shape [batch_size, num_patch, 1]

        x = torch.sum(x * a_slice, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, (a_patches, a_slice)
        else:
            return x

class DinoV2Patch_SoftSlice(nn.Module):
    def __init__(self, patch_size, img_size, num_classes):
        super(DinoV2Patch_SoftSlice, self).__init__()

        # Load Pre-trained DINOv2 Model
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        freeze_all(self.dinov2_vits14)  

        # # Load Pre-trained DINOv2 Model
        # self.dinov2_vits14 = Dinov2Model.from_pretrained("facebook/dinov2-small")
        # # LoRA
        # freeze_all(self.dinov2_vits14)
        # config = LoraConfig(
        #     r=4,                      # rank
        #     lora_alpha=8,
        #     target_modules=["query", "value"],  # attention layers
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type="FEATURE_EXTRACTION"
        # )
        # self.dinov2_vits14 = get_peft_model(self.dinov2_vits14, config)
        # self.dinov2_vits14.print_trainable_parameters()

        # Compute feature_dim
        dummy_slice = torch.zeros((1, 3, 504, 504))
        with torch.no_grad():
            outputs = self.dinov2_vits14(pixel_values=dummy_slice, output_hidden_states=True)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            self.feature_dim = patch_tokens.shape[-1]  # solo dim delle features
            self.num_patches = patch_tokens.shape[1]   # numero di patch

        self.p = patch_size
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.soft_attn_patch = nn.Linear(self.feature_dim, 1)
        self.soft_attn_slice = nn.Linear(self.feature_dim, 1)

        if num_classes == 2:
            num_classes = 1
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_attn_weights=False):
        
        batch_size, n_slice, c, h, w = x.shape

        # Process slices in chunks to avoid OOM
        # A single slice 504x504 with DinoV2-ViT-S takes significant memory
        x = x.view(batch_size * n_slice, c, h, w)

        chunk_size = 4 # Process 4 slices at a time
        x_ = []
        
        for i in range(0, x.shape[0], chunk_size):
            x_chunk = x[i:i+chunk_size]
            # Pass full slices to the model
            #features_out = self.dinov2_vits14.base_model.forward_features(x_chunk)
            # x_norm_patchtokens shape: [chunk_size, num_patches, feature_dim]
            #features = features_out["x_norm_patchtokens"]
            features = self.dinov2_vits14(pixel_values=x_chunk, output_hidden_states=True).last_hidden_state[:, 1:, :]
            x_.append(features)
        
        # Concatenate all slice features: [B*S, num_patches, feature_dim]
        x = torch.cat(x_, dim=0)

        # Apply patch-level soft attention
        # x: [B*S, num_patches, feature_dim]
        a_patch = F.softmax(self.soft_attn_patch(x), dim=1)
        # a_patch: [B*S, num_patches, 1]
        
        # Aggregate patches within each slice
        x = torch.sum(x * a_patch, dim=1)
        # x: [B*S, feature_dim]

        # Reshape to separate batch and slices: [B, S, feature_dim]
        x = x.view(batch_size, n_slice, self.feature_dim)

        # Apply slice-level soft attention
        a_slice = F.softmax(self.soft_attn_slice(x), dim=1)
        # a_slice: [B, S, 1]
        
        # Aggregate across slices
        x = torch.sum(x * a_slice, dim=1)
        # x: [B, feature_dim]

        x = self.classifier(x)

        if return_attn_weights:
            return x, (a_patch, a_slice)
        else:
            return x

class BackboneClassifier2D(nn.Module):
    def __init__(self, num_classes=2, model_name="efficientnet_b0", freeze_half=False, **kwargs):
        super(BackboneClassifier2D, self).__init__()
        # Load model
        if model_name == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b1":
            self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b2":
            self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b3":
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b4":
            self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b5":
            self.model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b6":
            self.model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b7":
            self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        elif model_name == "convnext_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_lite0":
            self.model = models.efficientnet_lite0(weights=models.EfficientNet_Lite0_Weights.IMAGENET1K_V1)
        elif model_name == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_v2_s":
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_v2_m":
            self.model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_v2_l":
            self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        elif model_name == "densenet121":
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        elif model_name == "densenet161":
            self.model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        elif model_name == "densenet169":
            self.model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        elif model_name == "densenet201":
            self.model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        elif model_name == "googlenet":
            self.model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Final classifier
        out_features = num_classes - 1 if num_classes == 2 else num_classes
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Linear(self.model.fc.in_features, out_features)
        elif hasattr(self.model, "classifier"):
            if isinstance(self.model.classifier, nn.Sequential):
                self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, out_features)
            else:
                self.model.classifier = nn.Linear(self.model.classifier.in_features, out_features)


        # Freeze layers
        if freeze_half:
            freeze_half_layers(self.model)

    def forward(self, x):
        # Input shape: (Batch, Channels, Height, Width)
        logits = self.model(x)  # (B, num_classes)
        return logits

if __name__ == "__main__":
    import torch
    model = DinoV2Patch_SoftSlice(patch_size=14, img_size=[448,448], num_classes=2)
    x = torch.randn(1, 10, 1, 448, 448)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")