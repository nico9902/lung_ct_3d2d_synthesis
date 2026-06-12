import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR

def extract_patch(volume, centers, sizes, out_size=(8, 8, 8)):
    """
    Estrae patch 3D da un volume intorno ai centri e ridimensiona a dimensione fissa con interpolate.

    volume: [C, D, H, W]
    centers: tuple di 3 tensori (z, y, x) o tensor [num_props,3]
    sizes:   tuple di 3 tensori (dz, dy, dx) o tensor [num_props,3]
    out_size: dimensione finale delle patch (D,H,W)

    Ritorna:
        patches: [num_props, C, D_out, H_out, W_out]
    """
    # converte tuple/list in tensor se necessario
    if isinstance(centers, tuple) or isinstance(centers, list):
        centers = torch.stack(centers, dim=1)  # [num_props,3]
    if isinstance(sizes, tuple) or isinstance(sizes, list):
        sizes = torch.stack(sizes, dim=1)  # [num_props,3]

    C, D, H, W = volume.shape
    N, _ = centers.shape

    batch_patches = []
    for n in range(N):
        z, y, x = centers[n]
        dz, dy, dx = sizes[n]

        # converti in int e clamp
        z1 = int((z - dz // 2).clamp(0))
        y1 = int((y - dy // 2).clamp(0))
        x1 = int((x - dx // 2).clamp(0))
        z2 = int((z1 + dz).clamp(max=D))
        y2 = int((y1 + dy).clamp(max=H))
        x2 = int((x1 + dx).clamp(max=W))

        patch = volume[:, z1:z2, y1:y2, x1:x2].unsqueeze(0)  # aggiungi batch dim [1,C,D,H,W]

        # controlla che tutte le dimensioni siano >0
        if patch.shape[2] == 0 or patch.shape[3] == 0 or patch.shape[4] == 0:
            # opzione 1: salta il patch
            continue
            # opzione 2: fai padding minimo per renderlo almeno 1 voxel
            # patch = F.pad(patch, (0, max(1 - patch.shape[4], 0),
            #                       0, max(1 - patch.shape[3], 0),
            #                       0, max(1 - patch.shape[2], 0)))

        # ridimensiona patch a dimensione fissa
        patch_resized = F.interpolate(patch, size=out_size, mode='trilinear', align_corners=False)
        batch_patches.append(patch_resized.squeeze(0))  # rimuovi batch dim temporanea
    
    return torch.stack(batch_patches)  # [num_props, C, D_out, H_out, W_out]

def pad_to_multiple_3d(volume, size):
    # Standard 3D: [B, C, D, H, W]
    B, C, D, H, W = volume.shape
    pad_d = (size - D % size) % size
    pad_h = (size - H % size) % size
    pad_w = (size - W % size) % size
    # F.pad parte dall'ultima dimensione all'indietro: (W, H, D)
    return F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))

def split_volume_3d(volume, patch_size=64, stride=32, return_offsets=False):
    """
    volume: [B, C, D, H, W]
    ritorna: patches [B*N_patches, C, P, P, P] e opzionalmente offsets [B*N_patches, 3]
    """
    volume = pad_to_multiple_3d(volume, patch_size)
    B, C, D, H, W = volume.shape
    patches = []
    offsets = []

    for d in range(0, D - patch_size + 1, stride):
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = volume[:, :, d:d+patch_size, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                if return_offsets:
                    offsets.append([d, y, x])
    
    patches = torch.cat(patches, dim=0)  # [B*N_patches, C, P, P, P]
    if return_offsets:
        return patches, torch.tensor(offsets, dtype=torch.long)
    return patches

class SwinBackbone3D(nn.Module):
    def __init__(self, feature_size=48):
        super().__init__()
        self.swin = SwinUNETR(
            in_channels=1,
            out_channels=32,
            feature_size=feature_size,
        ).swinViT

    def forward(self, x):
        feats = self.swin(x)
        # Scegliamo lo strato con stride 16 (risoluzione 4x4x4 per input 64^3)
        # out_feats[3] ha forma [B, feature_size * 8, 4, 4, 4]
        return feats[3] 

class RPN3D(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_head = nn.Conv3d(256, num_anchors, kernel_size=1)
        self.reg_head = nn.Conv3d(256, num_anchors * 6, kernel_size=1)

    def forward(self, x):
        t = F.relu(self.conv(x))
        return self.cls_head(t), self.reg_head(t)

class AnchorGenerator3D:
    def __init__(self, sizes=[5, 10, 20], stride=32):
        """
        sizes: lista di dimensioni cubiche delle ancore
        stride: stride della feature map rispetto al volume originale
        """
        self.sizes = sizes
        self.stride = stride

    def generate(self, Df, Hf, Wf, offset=(0,0,0)):
        """
        Genera le ancore per una feature map 3D, con eventuale offset per patching.

        Df, Hf, Wf: dimensioni della feature map (profondità, altezza, larghezza)
        offset: (z_offset, y_offset, x_offset) posizione della patch nel volume originale
        """
        z_off, y_off, x_off = offset
        anchors = []

        for z in range(Df):
            for y in range(Hf):
                for x in range(Wf):
                    # centro dell’anchor nello spazio del volume originale
                    center_z = z * self.stride + self.stride // 2 + z_off
                    center_y = y * self.stride + self.stride // 2 + y_off
                    center_x = x * self.stride + self.stride // 2 + x_off
                    
                    for s in self.sizes:
                        # [z_center, y_center, x_center, dz, dy, dx]
                        anchors.append([center_z, center_y, center_x, s, s, s])

        return torch.tensor(anchors, dtype=torch.float32)

def apply_deltas_to_anchors(anchors, deltas):
    """
    anchors: [N, 6] -> [z, y, x, dz, dy, dx]
    deltas:  [N, 6] -> [Δz, Δy, Δx, Δdz, Δdy, Δdx]
    """
    z_a, y_a, x_a, dz_a, dy_a, dx_a = anchors.unbind(dim=-1)
    dz, dy, dx, ddz, ddy, ddx = deltas.unbind(dim=-1)

    z_pred = z_a + dz * dz_a
    y_pred = y_a + dy * dy_a
    x_pred = x_a + dx * dx_a

    dz_pred = dz_a * torch.exp(ddz)
    dy_pred = dy_a * torch.exp(ddy)
    dx_pred = dx_a * torch.exp(ddx)

    return torch.stack([z_pred, y_pred, x_pred, dz_pred, dy_pred, dx_pred], dim=-1)

def rpn_postprocess(cls_logits, reg_deltas, anchors, conf_thresh=0.5, nms_thresh=0.3):
    """
    cls_logits: [N, num_anchors, Df, Hf, Wf] -> logits
    reg_deltas: [N, num_anchors*6, Df, Hf, Wf] -> predicted offsets
    anchors: [N, num_anchors, 6]
    """
    N = cls_logits.shape[0]
    proposals = []

    for b in range(N):
        # 1. Flatten tutto
        cls_prob = cls_logits[b].sigmoid()        # [num_anchors, Df, Hf, Wf]
        cls_prob = cls_prob.flatten()             # [num_anchors*Df*Hf*Wf]
        
        deltas = reg_deltas[b].view(-1, 6)       # [num_anchors*Df*Hf*Wf, 6]
        anchors_b = anchors[b].view(-1, 6)
        # anchors_b = anchors.repeat(cls_logits.shape[2]*cls_logits.shape[3]*cls_logits.shape[4], 1)  # espanso
       
        # 2. Applica delta
        boxes = apply_deltas_to_anchors(anchors_b, deltas)  # [num_total, 6]

        # 3. Filtra per confidenza
        mask = cls_prob > conf_thresh
        boxes = boxes[mask]
        scores = cls_prob[mask]

        # 4. NMS 3D (sostituiamo con IoU cubico approssimativo)
        keep = nms_3d(boxes, scores, nms_thresh)
        
        # Filtriamo le ancore mantentendo solo le proposal post-NMS
        proposals.append(boxes[keep])

    return proposals

def nms_3d(boxes, scores, iou_thresh):
    """
    NMS 3D approssimativo: boxes = [z, y, x, dz, dy, dx]
    """
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    # calcolo min/max
    z1, y1, x1, dz, dy, dx = boxes.unbind(dim=1)
    z2, y2, x2 = z1 + dz, y1 + dy, x1 + dx

    order = scores.sort(descending=True)[1]
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        # intersezione cubica
        zz1 = torch.max(z1[i], z1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx1 = torch.max(x1[i], x1[rest])
        zz2 = torch.min(z2[i], z2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        xx2 = torch.min(x2[i], x2[rest])

        inter = ((zz2 - zz1).clamp(0) *
                 (yy2 - yy1).clamp(0) *
                 (xx2 - xx1).clamp(0))

        vol_i = dz[i]*dy[i]*dx[i]
        vol_rest = dz[rest]*dy[rest]*dx[rest]
        iou = inter / (vol_i + vol_rest - inter)

        order = rest[iou <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long)

def roi_align_3d(feature, boxes, size=4):
    rois = []
    for b in boxes:
        x,y,z,dx,dy,dz = b.int()
        crop = feature[:, z:z+dz, y:y+dy, x:x+dx]
        crop = F.interpolate(crop.unsqueeze(0), size=(size,size,size), mode="trilinear")
        rois.append(crop)

    return torch.cat(rois)

class NoduleEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = x.flatten(1)

        return self.fc(x)

class MILHead(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.cls = nn.Linear(dim, 1)

    def forward(self, feats):
        A = self.attn(feats)
        A = torch.softmax(A, dim=1)
        M = (A * feats).sum(dim=1)
        y = self.cls(M)

        return y

class SwinRPNMIL(nn.Module):
    def __init__(self, patch_size=64, stride=32):
        super().__init__()
        self.p = patch_size
        self.stride = stride
        
        # SwinUNETR con feature_size 48 produce 384 canali nel bottleneck (48 * 8)
        self.feat_dim = 384 
        self.backbone = SwinBackbone3D(feature_size=48)
        self.rpn = RPN3D(self.feat_dim)

        self.anchor_gen = AnchorGenerator3D()

        self.nodule_encoder = NoduleEncoder(512)
        self.mil = MILHead(128)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        B = x.shape[0]
        
        # 1. split in patch
        patches, offsets = split_volume_3d(x, self.p, self.stride, return_offsets=True)
        N_patches = int(patches.shape[0]/B)
        # patches: [B*N_patches, 1, P, P, P]
        # offsets: [B*N_patches, 3] -> posizione originale della patch nel volume
        
        # 2. backbone chunking
        chunk_size = 8
        all_feats = []
        for i in range(0, patches.shape[0], chunk_size):
            chunk = patches[i:i+chunk_size]
            all_feats.append(self.backbone(chunk))
        feats = torch.cat(all_feats, dim=0)  # [B*N_patches, C, Df, Hf, Wf]

        # 3. RPN
        cls_logits, reg_deltas = self.rpn(feats)

        # 4. genera anchors per patch
        anchors_list = []
        Df, Hf, Wf = feats.shape[-3:]
        for offset in offsets:
            anchors = self.anchor_gen.generate(Df, Hf, Wf, offset=tuple(offset.tolist()))
            anchors_list.append(anchors)
        anchors = torch.stack(anchors_list, dim=0)  # [B*N_patches, num_anchors, 6]
        N_anchors = anchors.shape[1]

        # 5. Post-processing RPN (applica delta + conf_thresh + NMS)
        # RPN output shapes: 
        # cls_logits: [B*N_patches, num_anchors_per_loc, Df, Hf, Wf]
        # reg_deltas: [B*N_patches, num_anchors_per_loc * 6, Df, Hf, Wf]
        num_anchors_per_loc = len(self.anchor_gen.sizes)
        
        # Bisogna permutare per far combaciare l'ordine con l'AnchorGenerator (che itera z, y, x, s)
        cls_logits = cls_logits.view(B * N_patches, num_anchors_per_loc, 1, Df, Hf, Wf)
        cls_logits = cls_logits.permute(0, 3, 4, 5, 1, 2).reshape(B * N_patches, -1, 1) # [B*N_patches, total_anchors, 1]
        
        reg_deltas = reg_deltas.view(B * N_patches, num_anchors_per_loc, 6, Df, Hf, Wf)
        reg_deltas = reg_deltas.permute(0, 3, 4, 5, 1, 2).reshape(B * N_patches, -1, 6) # [B*N_patches, total_anchors, 6]
        
        # Raggruppiamo al livello del batch
        cls_logits_b = cls_logits.view(B, N_patches * N_anchors, -1)
        reg_deltas_b = reg_deltas.view(B, N_patches * N_anchors, -1)
        anchors_b = anchors.view(B, N_patches * N_anchors, -1)

        # Genera list di proposals box per ogni elemento del batch
        proposals = rpn_postprocess(cls_logits_b, reg_deltas_b, anchors_b)
        
        # 6. Passa le proposals al NoduleEncoder / MIL
        nodule_feats = []
        for i, prop in enumerate(proposals):  # lista di box [num_props,6]
            if len(prop) == 0:
                # Nel caso NMS o conf_thresh rimuovano tutto, aggiriamo l'errore aggiungendo roba a (0,0,0)
                feat = torch.zeros((1, 128), device=x.device)
                nodule_feats.append(feat)
                continue
                
            z_, y_, x_, dz, dy, dx = prop.T
            patch = extract_patch(x[i], centers=(z_,y_,x_), sizes=(dz,dy,dx))
            feat = self.nodule_encoder(patch)
            nodule_feats.append(feat)
            
        nodule_feats = torch.cat(nodule_feats, dim=0)  # [num_all_props, 128] o se MIL aggrega per patch, [B*... 128]
        # ATTENZIONE: per le loss è più corretto tenere final_score a livello di volume (batch).
        # MILHead fa un pooling sulle ancore di un patch/volume.
        # Raggruppo per ogni volume:
        start_idx = 0
        final_scores = []
        for i, prop in enumerate(proposals):
            num_p = len(prop)
            if num_p == 0:
                final_scores.append(self.mil(torch.zeros((1, 128), device=x.device)))
            else:
                feats_i = nodule_feats[start_idx : start_idx + num_p].unsqueeze(0) # [1, num_p, 128]
                final_scores.append(self.mil(feats_i)) # [1, 1]
                start_idx += num_p
        final_score = torch.cat(final_scores, dim=0) # [B, 1]
        
        # Ritorno anche cls_logits_b, reg_deltas_b, anchors_b per poter calcolare la loss esternamente
        return proposals, cls_logits_b, reg_deltas_b, anchors_b, final_score

if __name__=="__main__":
    import monai
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from monai.networks.nets import SwinUNETR
    
    model = SwinRPNMIL()

    input = torch.randn((1, 1, 128, 128, 128))
    proposals, cls_logits, reg_deltas, anchors, out = model(input)
    print("Final score:", out.shape)
    print("Proposals (batch 0):", proposals[0].shape)
    print("Cls logits:", cls_logits.shape)
    print("Reg deltas:", reg_deltas.shape)
    print("Anchors:", anchors.shape)