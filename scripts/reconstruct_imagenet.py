import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq_model import VQ_models
from tokenizer.tokenizer_image.lpips import LPIPS

# ==============================================================================
#                               CONSTANTS
# ==============================================================================
# Data Configuration
DATASET_PATH = "/path/to/imagenet_val"  # Update this path
SAMPLE_DIR = "reconstructions"
IMAGE_SIZE = 256
IMAGE_SIZE_EVAL = 256
BATCH_SIZE = 32
NUM_WORKERS = 4

# Model Configuration
VQ_MODEL = "VQ-16"  # Choices: VQ-16, VQ-8
VQ_CKPT = "/path/to/vq_ds16_c2i.pt"  # Update this path
CODEBOOK_SIZE = 16384
CODEBOOK_EMBED_DIM = 8

# Experiment Configuration
SEED = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    Adapted from: tokenizer/tokenizer_image/reconstruction_vq_ddp.py
    """
    samples = []
    # Find all png files, sort them to ensure order
    files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.png')])
    
    # If we have fewer samples than requested, adjust num
    if len(files) < num:
        print(f"Warning: Found {len(files)} samples, expected {num}. Using all found.")
        num = len(files)
    
    for i in tqdm(range(num), desc="Building .npz file"):
        # Assuming filenames are like 000000.png, but we use sorted list to be safe
        file_path = os.path.join(sample_dir, files[i])
        sample_pil = Image.open(file_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    
    samples = np.stack(samples)
    # assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

# ==============================================================================
#                               MAIN
# ==============================================================================

def main():
    # --------------------------------------------------------------------------
    # 1. Setup PyTorch and Device
    # --------------------------------------------------------------------------
    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.set_device(DEVICE)
        device = torch.device(DEVICE)
    else:
        device = torch.device("cpu")
    print(f"Running on {device} with seed {SEED}")

    # --------------------------------------------------------------------------
    # 2. Load VQ-VAE Model
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # 3. Setup Output Directory
    # --------------------------------------------------------------------------
    folder_name = (f"{VQ_MODEL}-imagenet-size-{IMAGE_SIZE}-size-{IMAGE_SIZE_EVAL}"
                   f"-codebook-size-{CODEBOOK_SIZE}-dim-{CODEBOOK_EMBED_DIM}-seed-{SEED}")
    sample_folder_dir = os.path.join(SAMPLE_DIR, folder_name)
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving samples to {sample_folder_dir}")

    # --------------------------------------------------------------------------
    # 4. Setup Data Loading
    # --------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # Mock args object for build_dataset
    class DataArgs:
        dataset = "imagenet"
        data_path = DATASET_PATH
    
    dataset = build_dataset(DataArgs(), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # Initialize LPIPS metric
    lpips_metric = LPIPS().to(device).eval()

    # --------------------------------------------------------------------------
    # 5. Reconstruction Loop
    # --------------------------------------------------------------------------
    psnr_val_rgb = []
    ssim_val_rgb = []
    mse_val_rgb = []
    lpips_val_rgb = []
    
    total_samples = 0
    
    print("Starting reconstruction...")
    for x, _ in tqdm(loader, desc="Processing Batches"):
        # Handle image size evaluation mismatch (interpolation)
        if IMAGE_SIZE_EVAL != IMAGE_SIZE:
            rgb_gts = F.interpolate(x, size=(IMAGE_SIZE_EVAL, IMAGE_SIZE_EVAL), mode='bicubic')
        else:
            rgb_gts = x
        
        # Prepare Ground Truth (GT) for metric calculation [0, 1]
        rgb_gts_numpy = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0
        
        # Move input to device
        x = x.to(device, non_blocking=True)
        
        # Inference
        with torch.no_grad():
            latent, _, [_, _, indices] = vq_model.encode(x)
            samples = vq_model.decode_code(indices, latent.shape)
            
            if IMAGE_SIZE_EVAL != IMAGE_SIZE:
                samples = F.interpolate(samples, size=(IMAGE_SIZE_EVAL, IMAGE_SIZE_EVAL), mode='bicubic')

            # Calculate LPIPS on GPU before moving to CPU
            # LPIPS expects inputs in [-1, 1]
            lpips_score = lpips_metric(rgb_gts.to(device), samples).mean().item()
            lpips_val_rgb.append(lpips_score)

        # Post-process samples for saving [0, 255] uint8
        samples_uint8 = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Calculate metrics and save images
        for i, (sample, rgb_gt) in enumerate(zip(samples_uint8, rgb_gts_numpy)):
            index = total_samples + i
            
            # Save image
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            
            # Prepare for metrics [0, 1] float32
            rgb_restored = sample.astype(np.float32) / 255.0
            
            # PSNR
            psnr = psnr_loss(rgb_restored, rgb_gt)
            psnr_val_rgb.append(psnr)
            
            # SSIM
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=1.0, channel_axis=-1)
            ssim_val_rgb.append(ssim)
            
            # MSE
            mse = np.mean((rgb_restored - rgb_gt) ** 2)
            mse_val_rgb.append(mse)
            
        total_samples += len(x)

    # --------------------------------------------------------------------------
    # 6. Summary and Results
    # --------------------------------------------------------------------------
    avg_psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
    avg_ssim = sum(ssim_val_rgb) / len(ssim_val_rgb)
    avg_mse = sum(mse_val_rgb) / len(mse_val_rgb)
    avg_lpips = sum(lpips_val_rgb) / len(lpips_val_rgb)

    print(f"Processed {total_samples} samples.")
    print(f"PSNR: {avg_psnr:.6f}")
    print(f"SSIM: {avg_ssim:.6f}")
    print(f"MSE: {avg_mse:.6f}")
    print(f"LPIPS: {avg_lpips:.6f}")

    result_file = f"{sample_folder_dir}_results.txt"
    print(f"Writing results to {result_file}")
    with open(result_file, 'w') as f:
        f.write(f"PSNR: {avg_psnr:.6f}\n")
        f.write(f"SSIM: {avg_ssim:.6f}\n")
        f.write(f"MSE: {avg_mse:.6f}\n")
        f.write(f"LPIPS: {avg_lpips:.6f}\n")

    # Create NPZ file (optional but requested to match reference)
    create_npz_from_sample_folder(sample_folder_dir, num=total_samples)
    print("Done.")

if __name__ == "__main__":
    main()
