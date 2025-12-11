import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import os
import glob
import time
import json
import numpy as np
from PIL import Image
import sys

# Add current directory to path to find local modules
sys.path.append(os.getcwd())

from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models

# ==============================================================================
#                               CONSTANTS
# ==============================================================================
# Robustness Configuration
NOISE_STD_LOW = 0.1
NOISE_STD_MID = 0.2
NOISE_STD_HIGH = 0.5
MAX_SAMPLES = None # Set to None to run on all samples, or an integer like 1000

STAMP = int(time.time())
OUTDIR = f"{STAMP}_robustness_dataset_llamagen"

# Model Configuration
VQ_MODEL = "VQ-16"  # Choices: VQ-16, VQ-8
VQ_CKPT = "/checkpoints/vq_ds16_c2i.pt" # Example path, should be updated or passed as arg
CODEBOOK_SIZE = 16384
CODEBOOK_EMBED_DIM = 8

# Data Configuration
DATASET_PATH = "/datasets/imagenet/val"
IMAGE_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 8
SEED = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                           HELPER FUNCTIONS & CLASSES
# ==============================================================================

def gather_val_paths(root):
    exts = ("*.JPEG","*.JPG","*.jpg", "*.png")
    # Assuming ImageNet structure: root/class_id/image.ext
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    paths = []
    
    for wnid in classes:
        class_dir = os.path.join(root, wnid)
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(class_dir, e)))
        files = sorted(files)
        paths.extend(files)
    return paths

class ImageNetPathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

def batch_to_uint8_hwc(x):
    # x is [B, C, H, W] in [-1, 1] (from LlamaGen normalization)
    # Convert to [0, 1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    # Convert to [0, 255] uint8
    y = x.mul(255.0).round().to(torch.uint8)
    # [B, C, H, W] -> [B, H, W, C]
    y = y.permute(0, 2, 3, 1).contiguous()
    return y.detach().cpu().numpy()

class RobustnessDatasetGenerator:
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.metadata_path = os.path.join(output_dir, "metadata.jsonl")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Open metadata file in append mode
        self.metadata_file = open(self.metadata_path, "w")

    def close(self):
        self.metadata_file.close()

    def process_batch(self, batch, paths):
        # batch: [B, 3, H, W] in [-1, 1]
        x_clean = batch.to(self.device)
        
        # Generate noisy inputs
        # Noise should be added in the same normalized space [-1, 1]
        # LlamaGen normalizes to [-1, 1] with mean=0.5, std=0.5
        # 0.1 std in this space corresponds to 0.1 * 2 = 0.2 in [0,1] space?
        # The VQGAN script used 0.1 in [-1, 1] space directly. We will stick to the same logic.
        base_noise = torch.randn_like(x_clean)
        
        x_low  = torch.clamp(x_clean + base_noise * NOISE_STD_LOW,  -1.0, 1.0)
        x_mid  = torch.clamp(x_clean + base_noise * NOISE_STD_MID,  -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH, -1.0, 1.0)
        
        # Inference (Encode & Decode)
        with torch.no_grad():
            # Helper to encode/decode
            def run_inference(x):
                latent, _, [_, _, indices] = self.model.encode(x)
                rec = self.model.decode_code(indices, latent.shape)
                return rec, indices

            # Clean
            rec_clean, ind_clean = run_inference(x_clean)
            
            # Low
            rec_low, ind_low = run_inference(x_low)
            
            # Mid
            rec_mid, ind_mid = run_inference(x_mid)
            
            # High
            rec_high, ind_high = run_inference(x_high)

        # Convert to uint8 for saving
        img_clean_uint8 = batch_to_uint8_hwc(x_clean)
        rec_clean_uint8 = batch_to_uint8_hwc(rec_clean)
        
        img_low_uint8 = batch_to_uint8_hwc(x_low)
        rec_low_uint8 = batch_to_uint8_hwc(rec_low)
        
        img_mid_uint8 = batch_to_uint8_hwc(x_mid)
        rec_mid_uint8 = batch_to_uint8_hwc(rec_mid)
        
        img_high_uint8 = batch_to_uint8_hwc(x_high)
        rec_high_uint8 = batch_to_uint8_hwc(rec_high)

        # Iterate over batch to save files and write metadata
        for i, original_path in enumerate(paths):
            # Parse path to get class and filename
            # /datasets/imagenet/val/n01440764/ILSVRC2012_val_00000293.JPEG
            parts = original_path.split("/")
            if len(parts) >= 2:
                class_id = parts[-2]
                filename = parts[-1].split(".")[0]
            else:
                # Fallback if path structure is unexpected
                class_id = "unknown"
                filename = os.path.basename(original_path).split(".")[0]
            
            # Create directory structure
            # images/n01440764/ILSVRC2012_val_00000293/
            sample_dir = os.path.join(self.images_dir, class_id, filename)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save images
            Image.fromarray(img_clean_uint8[i]).save(os.path.join(sample_dir, "0_original.png"))
            Image.fromarray(rec_clean_uint8[i]).save(os.path.join(sample_dir, "1_recon_clean.png"))
            
            Image.fromarray(img_low_uint8[i]).save(os.path.join(sample_dir, "2_input_noise_low.png"))
            Image.fromarray(rec_low_uint8[i]).save(os.path.join(sample_dir, "3_recon_noise_low.png"))
            
            Image.fromarray(img_mid_uint8[i]).save(os.path.join(sample_dir, "4_input_noise_mid.png"))
            Image.fromarray(rec_mid_uint8[i]).save(os.path.join(sample_dir, "5_recon_noise_mid.png"))
            
            Image.fromarray(img_high_uint8[i]).save(os.path.join(sample_dir, "6_input_noise_high.png"))
            Image.fromarray(rec_high_uint8[i]).save(os.path.join(sample_dir, "7_recon_noise_high.png"))
            
            # Write metadata
            metadata = {
                "image_id": f"{class_id}/{filename}",
                "original_path": original_path,
                "indices_clean": ind_clean[i].cpu().numpy().tolist(),
                "indices_low": ind_low[i].cpu().numpy().tolist(),
                "indices_mid": ind_mid[i].cpu().numpy().tolist(),
                "indices_high": ind_high[i].cpu().numpy().tolist(),
                "noise_std": [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH]
            }
            self.metadata_file.write(json.dumps(metadata) + "\n")

# ==============================================================================
#                               MAIN
# ==============================================================================

def main():
    print(f"Starting LlamaGen Robustness Dataset Generation to {OUTDIR}")
    print(f"Noise Levels: Low={NOISE_STD_LOW}, Mid={NOISE_STD_MID}, High={NOISE_STD_HIGH}")
    
    # 1. Setup PyTorch and Device
    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Running on {device} with seed {SEED}")

    # 2. Load VQ-VAE Model
    print(f"Loading {VQ_MODEL} model...")
    vq_model = VQ_models[VQ_MODEL](
        codebook_size=CODEBOOK_SIZE,
        codebook_embed_dim=CODEBOOK_EMBED_DIM
    )
    vq_model.to(device)
    vq_model.eval()

    print(f"Loading checkpoint from {VQ_CKPT}...")
    checkpoint = torch.load(VQ_CKPT, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("Please check model weight keys.")
    vq_model.load_state_dict(model_weight)
    del checkpoint

    # 3. Setup Data Loading
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    print(f"Gathering paths from {DATASET_PATH}...")
    paths = gather_val_paths(DATASET_PATH)
    print(f"Found {len(paths)} images.")

    dataset = ImageNetPathDataset(paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # 4. Processing Loop
    generator = RobustnessDatasetGenerator(vq_model, device, OUTDIR)
    
    total_processed = 0
    
    try:
        for batch, batch_paths in tqdm(loader, desc="Processing Batches"):
            current_batch_size = len(batch_paths)
            
            # Check if we need to trim the batch to fit MAX_SAMPLES
            if MAX_SAMPLES is not None and (total_processed + current_batch_size) > MAX_SAMPLES:
                remaining = MAX_SAMPLES - total_processed
                if remaining <= 0:
                    break
                batch = batch[:remaining]
                batch_paths = batch_paths[:remaining]
                current_batch_size = remaining
            
            generator.process_batch(batch, batch_paths)
            total_processed += current_batch_size
            
            if MAX_SAMPLES is not None and total_processed >= MAX_SAMPLES:
                print(f"Reached MAX_SAMPLES limit ({MAX_SAMPLES}). Stopping.")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Closing gracefully...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.close()
        print(f"Done. Processed {total_processed} samples.")
        print(f"Output saved to {OUTDIR}")

if __name__ == "__main__":
    main()
