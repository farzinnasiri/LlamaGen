import torch
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add repo path
sys.path.append(os.getcwd())

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset

# Config (Match your reconstruction script!)
DATASET_PATH = "/datasets/imagenet/val"
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 8

def main():
    print("Building ImageNet Validation GT Reference...")
    
    # EXACT same transform as the reconstruction script
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        # Note: We don't need Normalize here because we want raw pixels for the NPZ
        # But we do need to convert to tensor to handle the batching, then back to uint8
        transforms.ToTensor(), 
    ])

    class DataArgs:
        dataset = "imagenet"
        data_path = DATASET_PATH
    
    dataset = build_dataset(DataArgs(), transform=transform)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    all_samples = []
    
    print(f"Processing {len(dataset)} images...")
    
    for x, _ in tqdm(loader):
        # x is [B, 3, 256, 256] in range [0, 1] (because ToTensor scales it)
        # We need [B, 256, 256, 3] in range [0, 255] uint8
        
        x = x.permute(0, 2, 3, 1).numpy() # [B, H, W, C]
        x = (x * 255).astype(np.uint8)
        all_samples.append(x)

    all_samples = np.concatenate(all_samples, axis=0)
    print(f"Collected shape: {all_samples.shape}")

    outfile = "imagenet_val_gt_256.npz"
    np.savez(outfile, arr_0=all_samples)
    print(f"Saved {outfile}")

if __name__ == "__main__":
    main()
