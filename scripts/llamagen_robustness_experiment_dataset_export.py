import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import random
import math
import os
import glob
import time
import json
import numpy as np
from PIL import Image
import sys

import multiprocessing as mp

# Add current directory to path to find local modules
sys.path.append(os.getcwd())

from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models

# ==============================================================================
#                               CONSTANTS
# ==============================================================================
# Robustness Configuration
def get_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(val)
    if isinstance(default, float):
        return float(val)
    return val

NOISE_STD_LOW = 0.1
NOISE_STD_MID = 0.25
NOISE_STD_HIGH = 0.5
NOISE_STD_XHIGH = 1.0
MAX_SAMPLES = None # Set to None to run on all samples, or an integer like 1000

H2_DECODE_MAX_BATCH = get_env("H2_DECODE_MAX_BATCH", 64)
NUM_SAVE_WORKERS = 16

EXPERIMENT_MODE = get_env("EXPERIMENT_MODE", "h1_patch_noise_encoder")
PATCH_TOK_SIDE = 8
PATCH_FRACTION = 0.25
PATCH_PLACEMENT = "random"
TOKEN_EDIT_MODES = ["random_uniform", "closest", "farthest", "orthogonal"]
USE_BLACK_MASK_H1 = True

SEED = 0

STAMP = int(time.time())
OUTDIR = f"{STAMP}_robustness_dataset_llamagen_{EXPERIMENT_MODE}_patch{PATCH_TOK_SIDE}_seed{SEED}"

# Model Configuration
VQ_MODEL = "VQ-16"  # Choices: VQ-16, VQ-8
VQ_CKPT = "/checkpoints/vq_ds16_c2i.pt" # Example path, should be updated or passed as arg
CODEBOOK_SIZE = 16384
CODEBOOK_EMBED_DIM = 8
CODEBOOK_RELATIONS_NPZ_PATH = get_env("CODEBOOK_RELATIONS_NPZ_PATH", "llamagen_codebook_relations.npz")

# Data Configuration
DATASET_PATH = "/datasets/imagenet/val"
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#                           HELPER FUNCTIONS & CLASSES
# ==============================================================================

def save_worker(worker_id, queue, output_dir):
    """
    Worker process to save images and write metadata.
    Reads tasks from queue and writes to its own metadata file.
    """
    images_dir = os.path.join(output_dir, "images")
    metadata_path = os.path.join(output_dir, f"metadata_part_{worker_id}.jsonl")
    
    # Open unique metadata file for this worker
    with open(metadata_path, "w") as f:
        while True:
            task = queue.get()
            if task is None:
                break
            
            try:
                original_path = task['original_path']
                
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
                sample_dir = os.path.join(images_dir, class_id, filename)
                os.makedirs(sample_dir, exist_ok=True)
                
                for entry in task["images"]:
                    Image.fromarray(entry["array"]).save(os.path.join(sample_dir, entry["filename"]))

                metadata = dict(task["metadata"])
                metadata.update({
                    "image_id": f"{class_id}/{filename}",
                    "original_path": original_path,
                })
                f.write(json.dumps(metadata) + "\n")
                
            except Exception as e:
                print(f"Worker {worker_id} error processing {original_path}: {e}")

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
        self._codebook_relations = None
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize Workers
        self.queue = mp.Queue(maxsize=NUM_SAVE_WORKERS * 10) # Buffer size
        self.workers = []
        print(f"Starting {NUM_SAVE_WORKERS} save workers...")
        for i in range(NUM_SAVE_WORKERS):
            p = mp.Process(target=save_worker, args=(i, self.queue, output_dir))
            p.start()
            self.workers.append(p)

    def close(self):
        print("Waiting for workers to finish...")
        for _ in self.workers:
            self.queue.put(None)
        for p in self.workers:
            p.join()

    def _get_codebook_relations(self):
        if self._codebook_relations is not None:
            return self._codebook_relations

        if not os.path.exists(CODEBOOK_RELATIONS_NPZ_PATH):
            raise FileNotFoundError(f"Missing codebook relations NPZ: {CODEBOOK_RELATIONS_NPZ_PATH}")

        rel = np.load(CODEBOOK_RELATIONS_NPZ_PATH)
        required = ["min_dist_idx", "max_dist_idx", "ortho_idx"]
        missing = [k for k in required if k not in rel]
        if missing:
            raise KeyError(f"Codebook relations NPZ missing keys: {missing}")

        min_dist_idx = rel["min_dist_idx"]
        max_dist_idx = rel["max_dist_idx"]
        ortho_idx = rel["ortho_idx"]
        if min_dist_idx.shape[0] != CODEBOOK_SIZE or max_dist_idx.shape[0] != CODEBOOK_SIZE or ortho_idx.shape[0] != CODEBOOK_SIZE:
            raise ValueError(
                f"Codebook relations size mismatch: expected {CODEBOOK_SIZE}, got "
                f"min={min_dist_idx.shape[0]} max={max_dist_idx.shape[0]} ortho={ortho_idx.shape[0]}"
            )

        self._codebook_relations = {
            "min_dist_idx": torch.from_numpy(min_dist_idx).to(self.device),
            "max_dist_idx": torch.from_numpy(max_dist_idx).to(self.device),
            "ortho_idx": torch.from_numpy(ortho_idx).to(self.device),
        }
        return self._codebook_relations

    def sample_patch_bboxes_tok_square(self, batch_size, height_tok, width_tok, side_tok, placement):
        side_tok = max(1, min(int(side_tok), height_tok, width_tok))
        if placement == "center":
            j0 = (width_tok - side_tok) // 2
            i0 = (height_tok - side_tok) // 2
            j1 = j0 + side_tok
            i1 = i0 + side_tok
            return [(int(j0), int(i0), int(j1), int(i1)) for _ in range(batch_size)]
        if placement == "random":
            max_j0 = max(0, width_tok - side_tok)
            max_i0 = max(0, height_tok - side_tok)
            bboxes = []
            for _ in range(batch_size):
                j0 = random.randint(0, max_j0) if max_j0 > 0 else 0
                i0 = random.randint(0, max_i0) if max_i0 > 0 else 0
                j1 = j0 + side_tok
                i1 = i0 + side_tok
                bboxes.append((int(j0), int(i0), int(j1), int(i1)))
            return bboxes
        raise ValueError(f"Unknown placement: {placement}")

    def make_mask_from_bboxes_px(self, bboxes, height, width, device):
        mask = torch.zeros((len(bboxes), 1, height, width), device=device)
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            mask[i, :, y0:y1, x0:x1] = 1.0
        return mask

    def bbox_tok_to_bbox_px(self, bbox_tok, height_px, width_px, height_tok, width_tok):
        if (width_px % width_tok) != 0 or (height_px % height_tok) != 0:
            raise ValueError(
                f"Pixel/token grids are not evenly divisible: px=({height_px},{width_px}) tok=({height_tok},{width_tok})"
            )
        stride_x = width_px // width_tok
        stride_y = height_px // height_tok
        j0, i0, j1, i1 = bbox_tok
        x0 = j0 * stride_x
        y0 = i0 * stride_y
        x1 = j1 * stride_x
        y1 = i1 * stride_y
        return (int(x0), int(y0), int(x1), int(y1))

    def indices_to_flat_and_grid(self, indices, batch_size, height_tok, width_tok):
        ind_flat = indices.reshape(batch_size, -1)
        if ind_flat.shape[1] != (height_tok * width_tok):
            raise ValueError(
                f"Unexpected token count: got {ind_flat.shape[1]} expected {height_tok * width_tok}"
            )
        ind_grid = ind_flat.reshape(batch_size, height_tok, width_tok)
        return ind_flat, ind_grid

    def encode_decode(self, x):
        quant, _, (_, _, indices) = self.model.encode(x)
        rec = self.model.decode(quant)
        return rec, indices, quant

    def decode_from_indices_flat(self, ind_flat, quant_shape):
        shape = (ind_flat.shape[0], int(quant_shape[1]), int(quant_shape[2]), int(quant_shape[3]))
        rec = self.model.decode_code(ind_flat, shape)
        return rec

    def run_global_noise_experiment(self, x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok, quant_shape):
        base_noise = torch.randn_like(x_clean)
        x_low = torch.clamp(x_clean + base_noise * NOISE_STD_LOW, -1.0, 1.0)
        x_mid = torch.clamp(x_clean + base_noise * NOISE_STD_MID, -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH, -1.0, 1.0)
        x_xhigh = torch.clamp(x_clean + base_noise * NOISE_STD_XHIGH, -1.0, 1.0)

        with torch.no_grad():
            rec_low, ind_low_raw, _ = self.encode_decode(x_low)
            rec_mid, ind_mid_raw, _ = self.encode_decode(x_mid)
            rec_high, ind_high_raw, _ = self.encode_decode(x_high)
            rec_xhigh, ind_xhigh_raw, _ = self.encode_decode(x_xhigh)

        ind_low_flat, _ = self.indices_to_flat_and_grid(ind_low_raw, batch_size, height_tok, width_tok)
        ind_mid_flat, _ = self.indices_to_flat_and_grid(ind_mid_raw, batch_size, height_tok, width_tok)
        ind_high_flat, _ = self.indices_to_flat_and_grid(ind_high_raw, batch_size, height_tok, width_tok)
        ind_xhigh_flat, _ = self.indices_to_flat_and_grid(ind_xhigh_raw, batch_size, height_tok, width_tok)

        img_low_uint8 = batch_to_uint8_hwc(x_low)
        rec_low_uint8 = batch_to_uint8_hwc(rec_low)
        img_mid_uint8 = batch_to_uint8_hwc(x_mid)
        rec_mid_uint8 = batch_to_uint8_hwc(rec_mid)
        img_high_uint8 = batch_to_uint8_hwc(x_high)
        rec_high_uint8 = batch_to_uint8_hwc(rec_high)
        img_xhigh_uint8 = batch_to_uint8_hwc(x_xhigh)
        rec_xhigh_uint8 = batch_to_uint8_hwc(rec_xhigh)

        ind_clean_cpu = ind_clean_flat.detach().cpu().numpy()
        ind_low_cpu = ind_low_flat.detach().cpu().numpy()
        ind_mid_cpu = ind_mid_flat.detach().cpu().numpy()
        ind_high_cpu = ind_high_flat.detach().cpu().numpy()
        ind_xhigh_cpu = ind_xhigh_flat.detach().cpu().numpy()

        for i, original_path in enumerate(paths):
            images = [
                {"filename": "0_original.png", "array": img_clean_uint8[i]},
                {"filename": "1_recon_clean.png", "array": rec_clean_uint8[i]},
                {"filename": "2_input_noise_low.png", "array": img_low_uint8[i]},
                {"filename": "3_recon_noise_low.png", "array": rec_low_uint8[i]},
                {"filename": "4_input_noise_mid.png", "array": img_mid_uint8[i]},
                {"filename": "5_recon_noise_mid.png", "array": rec_mid_uint8[i]},
                {"filename": "6_input_noise_high.png", "array": img_high_uint8[i]},
                {"filename": "7_recon_noise_high.png", "array": rec_high_uint8[i]},
                {"filename": "8_input_noise_xhigh.png", "array": img_xhigh_uint8[i]},
                {"filename": "9_recon_noise_xhigh.png", "array": rec_xhigh_uint8[i]},
            ]
            metadata = {
                "experiment_mode": EXPERIMENT_MODE,
                "noise_std": [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH, NOISE_STD_XHIGH],
                "token_grid_hw": [int(height_tok), int(width_tok)],
                "indices_clean": ind_clean_cpu[i].tolist(),
                "indices_low": ind_low_cpu[i].tolist(),
                "indices_mid": ind_mid_cpu[i].tolist(),
                "indices_high": ind_high_cpu[i].tolist(),
                "indices_xhigh": ind_xhigh_cpu[i].tolist(),
            }
            task = {"original_path": original_path, "images": images, "metadata": metadata}
            self.queue.put(task)

    def run_h1_patch_noise_encoder_experiment(self, x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok, quant_shape):
        height_px, width_px = x_clean.shape[2], x_clean.shape[3]

        if PATCH_TOK_SIDE is None:
            side_tok = int(round(math.sqrt(max(0.0, min(1.0, PATCH_FRACTION))) * min(height_tok, width_tok)))
            patch_side_tok = max(1, min(side_tok, height_tok, width_tok))
        else:
            patch_side_tok = max(1, min(int(PATCH_TOK_SIDE), height_tok, width_tok))

        bboxes_tok = self.sample_patch_bboxes_tok_square(batch_size, height_tok, width_tok, patch_side_tok, PATCH_PLACEMENT)
        bboxes_px = [self.bbox_tok_to_bbox_px(b, height_px, width_px, height_tok, width_tok) for b in bboxes_tok]

        base_noise = torch.randn_like(x_clean)
        mask_px = self.make_mask_from_bboxes_px(bboxes_px, height_px, width_px, device=self.device)

        x_low = torch.clamp(x_clean + base_noise * NOISE_STD_LOW * mask_px, -1.0, 1.0)
        x_mid = torch.clamp(x_clean + base_noise * NOISE_STD_MID * mask_px, -1.0, 1.0)
        x_high = torch.clamp(x_clean + base_noise * NOISE_STD_HIGH * mask_px, -1.0, 1.0)
        x_xhigh = torch.clamp(x_clean + base_noise * NOISE_STD_XHIGH * mask_px, -1.0, 1.0)

        with torch.no_grad():
            rec_low, ind_low_raw, _ = self.encode_decode(x_low)
            rec_mid, ind_mid_raw, _ = self.encode_decode(x_mid)
            rec_high, ind_high_raw, _ = self.encode_decode(x_high)
            rec_xhigh, ind_xhigh_raw, _ = self.encode_decode(x_xhigh)

            if USE_BLACK_MASK_H1:
                x_masked = x_clean * (1.0 - mask_px) + (-1.0 * mask_px)
                rec_masked, ind_masked_raw, _ = self.encode_decode(x_masked)

        ind_low_flat, _ = self.indices_to_flat_and_grid(ind_low_raw, batch_size, height_tok, width_tok)
        ind_mid_flat, _ = self.indices_to_flat_and_grid(ind_mid_raw, batch_size, height_tok, width_tok)
        ind_high_flat, _ = self.indices_to_flat_and_grid(ind_high_raw, batch_size, height_tok, width_tok)
        ind_xhigh_flat, _ = self.indices_to_flat_and_grid(ind_xhigh_raw, batch_size, height_tok, width_tok)
        if USE_BLACK_MASK_H1:
            ind_masked_flat, _ = self.indices_to_flat_and_grid(ind_masked_raw, batch_size, height_tok, width_tok)

        img_low_uint8 = batch_to_uint8_hwc(x_low)
        rec_low_uint8 = batch_to_uint8_hwc(rec_low)
        img_mid_uint8 = batch_to_uint8_hwc(x_mid)
        rec_mid_uint8 = batch_to_uint8_hwc(rec_mid)
        img_high_uint8 = batch_to_uint8_hwc(x_high)
        rec_high_uint8 = batch_to_uint8_hwc(rec_high)
        img_xhigh_uint8 = batch_to_uint8_hwc(x_xhigh)
        rec_xhigh_uint8 = batch_to_uint8_hwc(rec_xhigh)
        if USE_BLACK_MASK_H1:
            img_masked_uint8 = batch_to_uint8_hwc(x_masked)
            rec_masked_uint8 = batch_to_uint8_hwc(rec_masked)

        ind_clean_cpu = ind_clean_flat.detach().cpu().numpy()
        ind_low_cpu = ind_low_flat.detach().cpu().numpy()
        ind_mid_cpu = ind_mid_flat.detach().cpu().numpy()
        ind_high_cpu = ind_high_flat.detach().cpu().numpy()
        ind_xhigh_cpu = ind_xhigh_flat.detach().cpu().numpy()
        if USE_BLACK_MASK_H1:
            ind_masked_cpu = ind_masked_flat.detach().cpu().numpy()

        for i, original_path in enumerate(paths):
            x0, y0, x1, y1 = bboxes_px[i]
            j0, i0, j1, i1 = bboxes_tok[i]
            images = [
                {"filename": "0_original.png", "array": img_clean_uint8[i]},
                {"filename": "1_recon_clean.png", "array": rec_clean_uint8[i]},
                {"filename": "2_input_patch_noise_low.png", "array": img_low_uint8[i]},
                {"filename": "3_recon_patch_noise_low.png", "array": rec_low_uint8[i]},
                {"filename": "4_input_patch_noise_mid.png", "array": img_mid_uint8[i]},
                {"filename": "5_recon_patch_noise_mid.png", "array": rec_mid_uint8[i]},
                {"filename": "6_input_patch_noise_high.png", "array": img_high_uint8[i]},
                {"filename": "7_recon_patch_noise_high.png", "array": rec_high_uint8[i]},
                {"filename": "8_input_patch_noise_xhigh.png", "array": img_xhigh_uint8[i]},
                {"filename": "9_recon_patch_noise_xhigh.png", "array": rec_xhigh_uint8[i]},
            ]
            if USE_BLACK_MASK_H1:
                images.extend([
                    {"filename": "10_input_patch_masked.png", "array": img_masked_uint8[i]},
                    {"filename": "11_recon_patch_masked.png", "array": rec_masked_uint8[i]},
                ])

            metadata = {
                "experiment_mode": EXPERIMENT_MODE,
                "noise_std": [NOISE_STD_LOW, NOISE_STD_MID, NOISE_STD_HIGH, NOISE_STD_XHIGH],
                "token_grid_hw": [int(height_tok), int(width_tok)],
                "patch_bbox_px": list(map(int, bboxes_px[i])),
                "patch_bbox_tok": list(map(int, bboxes_tok[i])),
                "patch_side_tok": int(patch_side_tok),
                "patch_top_left_tok": [int(j0), int(i0)],
                "patch_side_px": [int(x1 - x0), int(y1 - y0)],
                "patch_top_left_px": [int(x0), int(y0)],
                "indices_clean": ind_clean_cpu[i].tolist(),
                "indices_low": ind_low_cpu[i].tolist(),
                "indices_mid": ind_mid_cpu[i].tolist(),
                "indices_high": ind_high_cpu[i].tolist(),
                "indices_xhigh": ind_xhigh_cpu[i].tolist(),
            }
            if USE_BLACK_MASK_H1:
                metadata["indices_masked"] = ind_masked_cpu[i].tolist()
                metadata["has_masked_occlusion"] = True

            task = {"original_path": original_path, "images": images, "metadata": metadata}
            self.queue.put(task)

    def run_h2_patch_token_edit_decoder_experiment(self, x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, ind_clean_grid, height_tok, width_tok, quant_shape):
        height_px, width_px = x_clean.shape[2], x_clean.shape[3]

        if PATCH_TOK_SIDE is None:
            side_tok = int(round(math.sqrt(max(0.0, min(1.0, PATCH_FRACTION))) * min(height_tok, width_tok)))
            patch_side_tok = max(1, min(side_tok, height_tok, width_tok))
        else:
            patch_side_tok = max(1, min(int(PATCH_TOK_SIDE), height_tok, width_tok))

        bboxes_tok = self.sample_patch_bboxes_tok_square(batch_size, height_tok, width_tok, patch_side_tok, PATCH_PLACEMENT)
        bboxes_px = [self.bbox_tok_to_bbox_px(b, height_px, width_px, height_tok, width_tok) for b in bboxes_tok]

        if not isinstance(TOKEN_EDIT_MODES, (list, tuple)):
            raise ValueError("TOKEN_EDIT_MODES must be a list or tuple")
        token_edit_modes = [str(m) for m in TOKEN_EDIT_MODES]
        if len(set(token_edit_modes)) != len(token_edit_modes):
            raise ValueError("TOKEN_EDIT_MODES contains duplicates")

        valid_modes = {"random_uniform", "closest", "farthest", "orthogonal"}
        bad_modes = [m for m in token_edit_modes if m not in valid_modes]
        if bad_modes:
            raise ValueError(f"Unknown TOKEN_EDIT_MODES entries: {bad_modes}")

        relations = None
        if any(m in {"closest", "farthest", "orthogonal"} for m in token_edit_modes):
            relations = self._get_codebook_relations()

        ind_edits_flat = []
        ind_edits_cpu = {}

        for mode in token_edit_modes:
            ind_edit_grid = ind_clean_grid.clone()
            for bi, (j0, i0, j1, i1) in enumerate(bboxes_tok):
                if mode == "random_uniform":
                    rand_patch = torch.randint(
                        low=0,
                        high=int(CODEBOOK_SIZE),
                        size=(i1 - i0, j1 - j0),
                        device=ind_edit_grid.device,
                    )
                    ind_edit_grid[bi, i0:i1, j0:j1] = rand_patch
                else:
                    if mode == "closest":
                        map_key = "min_dist_idx"
                    elif mode == "farthest":
                        map_key = "max_dist_idx"
                    else:
                        map_key = "ortho_idx"
                    patch = ind_edit_grid[bi, i0:i1, j0:j1]
                    mapped = relations[map_key][patch]
                    if (mapped < 0).any():
                        raise RuntimeError("Codebook relations mapping produced invalid indices (<0)")
                    ind_edit_grid[bi, i0:i1, j0:j1] = mapped

            ind_edit_flat = ind_edit_grid.reshape(batch_size, -1)
            ind_edits_flat.append(ind_edit_flat)
            ind_edits_cpu[mode] = ind_edit_flat.detach().cpu().numpy()

        ind_all = torch.cat(ind_edits_flat, dim=0)
        max_decode_batch = int(H2_DECODE_MAX_BATCH)
        if max_decode_batch <= 0:
            raise ValueError("H2_DECODE_MAX_BATCH must be > 0")

        rec_chunks = []
        with torch.no_grad():
            for start in range(0, ind_all.shape[0], max_decode_batch):
                chunk = ind_all[start : start + max_decode_batch]
                rec_chunks.append(self.decode_from_indices_flat(chunk, quant_shape))
        rec_all = torch.cat(rec_chunks, dim=0)
        rec_all_uint8 = batch_to_uint8_hwc(rec_all)

        rec_by_mode = {}
        for mi, mode in enumerate(token_edit_modes):
            rec_by_mode[mode] = rec_all_uint8[mi * batch_size : (mi + 1) * batch_size]

        ind_clean_cpu = ind_clean_flat.detach().cpu().numpy()

        for i, original_path in enumerate(paths):
            images = [
                {"filename": "0_original.png", "array": img_clean_uint8[i]},
                {"filename": "1_recon_clean.png", "array": rec_clean_uint8[i]},
            ]
            for mode in token_edit_modes:
                images.append({"filename": f"2_recon_token_edit_{mode}.png", "array": rec_by_mode[mode][i]})

            metadata = {
                "experiment_mode": EXPERIMENT_MODE,
                "token_edit_modes": list(token_edit_modes),
                "token_grid_hw": [int(height_tok), int(width_tok)],
                "patch_bbox_px": list(map(int, bboxes_px[i])),
                "patch_bbox_tok": list(map(int, bboxes_tok[i])),
                "indices_clean": ind_clean_cpu[i].tolist(),
                "indices_edit_by_mode": {m: ind_edits_cpu[m][i].tolist() for m in token_edit_modes},
            }
            if len(token_edit_modes) == 1:
                metadata["token_edit_mode"] = token_edit_modes[0]
                metadata["indices_edit"] = ind_edits_cpu[token_edit_modes[0]][i].tolist()

            task = {"original_path": original_path, "images": images, "metadata": metadata}
            self.queue.put(task)

    def process_batch(self, batch, paths):
        x_clean = batch.to(self.device)
        batch_size = x_clean.shape[0]

        with torch.no_grad():
            rec_clean, ind_clean_raw, quant_clean = self.encode_decode(x_clean)

        height_tok, width_tok = int(quant_clean.shape[2]), int(quant_clean.shape[3])
        ind_clean_flat, ind_clean_grid = self.indices_to_flat_and_grid(ind_clean_raw, batch_size, height_tok, width_tok)

        img_clean_uint8 = batch_to_uint8_hwc(x_clean)
        rec_clean_uint8 = batch_to_uint8_hwc(rec_clean)

        if EXPERIMENT_MODE == "global_noise":
            self.run_global_noise_experiment(x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok, quant_clean.shape)
            return

        if EXPERIMENT_MODE == "h1_patch_noise_encoder":
            self.run_h1_patch_noise_encoder_experiment(x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, height_tok, width_tok, quant_clean.shape)
            return

        if EXPERIMENT_MODE == "h2_patch_token_edit_decoder":
            self.run_h2_patch_token_edit_decoder_experiment(x_clean, paths, batch_size, img_clean_uint8, rec_clean_uint8, ind_clean_flat, ind_clean_grid, height_tok, width_tok, quant_clean.shape)
            return

        raise ValueError(f"Unknown EXPERIMENT_MODE: {EXPERIMENT_MODE}")

# ==============================================================================
#                               MAIN
# ==============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Starting Robustness Dataset Generation to {OUTDIR}")
    print(f"Experiment Mode: {EXPERIMENT_MODE}")
    print(f"Noise Levels: Low={NOISE_STD_LOW}, Mid={NOISE_STD_MID}, High={NOISE_STD_HIGH}, XHigh={NOISE_STD_XHIGH}")
    if EXPERIMENT_MODE == "h1_patch_noise_encoder":
        print(f"H1 Black Mask (Occlusion): {USE_BLACK_MASK_H1}")
    print(f"Patch Fraction: {PATCH_FRACTION}, Patch Placement: {PATCH_PLACEMENT}")
    
    # 1. Setup PyTorch and Device
    torch.set_grad_enabled(False)
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

    os.makedirs(OUTDIR, exist_ok=True)

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
