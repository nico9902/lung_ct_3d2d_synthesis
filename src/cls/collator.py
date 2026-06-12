# import libraries
import torch
from torch.nn.utils.rnn import pad_sequence

class Collator(object):
    """
        Collate function for the DataLoader
    """
    def __init__(self):
        """
        Initialize the collator.
        """

    def __call__(self, batch):
        # Check if batch has masks (4 elements) or not (3 elements)
        has_masks = len(batch[0]) == 4
        
        # Get max number of slices in the batch
        if has_masks:
            max_slices = max(image.size(0) for image, label, pid, mask in batch)
        else:
            max_slices = max(image.size(0) for image, label, pid in batch)
        
        # Collate images - pad to max_slices
        images = []
        if has_masks:
            for image, _, _, _ in batch:
                # image shape: (num_slices, C, H, W)
                if image.size(0) < max_slices:
                    # Pad along slice dimension
                    padding = torch.zeros(max_slices - image.size(0), image.size(1), image.size(2), image.size(3), dtype=image.dtype)
                    image = torch.cat([image, padding], dim=0)
                images.append(image)
        else:
            for image, _, _ in batch:
                # image shape: (num_slices, C, H, W)
                if image.size(0) < max_slices:
                    # Pad along slice dimension
                    padding = torch.zeros(max_slices - image.size(0), image.size(1), image.size(2), image.size(3), dtype=image.dtype)
                    image = torch.cat([image, padding], dim=0)
                images.append(image)

        # Stack images
        images = torch.stack(images)  # (batch_size, max_slices, C, H, W)

        if has_masks:
            labels = torch.stack([label for _, label, _, _ in batch])
            pids = [pid for _, _, pid, _ in batch]
            
            # Collate masks
            masks = []
            for _, _, _, mask in batch:
                # mask shape: (num_slices, H, W)
                if mask.size(0) < max_slices:
                    # Pad masks to max_slices
                    padding = torch.zeros(max_slices - mask.size(0), mask.size(1), mask.size(2), dtype=mask.dtype)
                    mask = torch.cat([mask, padding], dim=0)
                masks.append(mask)
            
            masks = torch.stack(masks)  # (batch_size, max_slices, H, W)
            return images, labels, pids, masks
        else:
            labels = torch.stack([label for _, label, _ in batch])
            pids = [pid for _, _, pid in batch]
            return images, labels, pids

if __name__ == "__main__":
    collator = Collator()
    # Example batch: list of (image_sequences, label_tensor, pid)
    batch = [
        (torch.randn(2, 1, 8, 8), torch.tensor(0), "pid_1"),
        (torch.randn(3, 1, 8, 8), torch.tensor(1), "pid_2")
    ]
    
    images, labels, pids = collator(batch)
    print(f"Images shape: {images.shape}")  # (batch_size, max_num_sequences, max_seq

    print(images)