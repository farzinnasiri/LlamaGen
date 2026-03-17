import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.lpips import LPIPS
from tokenizer.tokenizer_image.vq_model import VQ_models


VALID_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct an ImageNet-style dataset with LlamaGen VQ-VAE.")
    parser.add_argument("--data-path", type=str, required=True, help="Root directory containing one subdirectory per class.")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to the VQ checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="reconstructions", help="Directory where outputs are written.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output subdirectory name.")
    parser.add_argument("--image-size", type=int, default=256, help="Input crop size.")
    parser.add_argument("--image-size-eval", type=int, default=256, help="Evaluation size after optional resize.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--vq-model", type=str, default="VQ-16", choices=["VQ-16", "VQ-8"])
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-save-images", action="store_true", help="Do not save per-image PNG reconstructions.")
    parser.add_argument("--skip-export-npz", action="store_true", help="Do not export a stacked reconstruction NPZ.")
    return parser.parse_args()


def class_sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_dataset(root: str):
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    class_dirs = sorted([path for path in root_path.iterdir() if path.is_dir()], key=class_sort_key)
    if not class_dirs:
        raise RuntimeError(f"No class directories found under {root}")

    samples = []
    class_names = []
    next_label = 0
    for class_dir in class_dirs:
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS],
            key=lambda path: path.name,
        )
        if not image_paths:
            continue
        class_names.append(class_dir.name)
        samples.extend((str(path), next_label) for path in image_paths)
        next_label += 1

    if not samples:
        raise RuntimeError(f"No supported images found under {root}")

    return samples, class_names


class GenericImageFolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, path


def create_npz_from_sample_folder(sample_dir, num=None):
    samples = []
    files = sorted([file_name for file_name in os.listdir(sample_dir) if file_name.endswith(".png")])

    if num is None:
        num = len(files)
    if len(files) < num:
        print(f"Warning: Found {len(files)} samples, expected {num}. Using all found.")
        num = len(files)

    for index in tqdm(range(num), desc="Building .npz file"):
        file_path = os.path.join(sample_dir, files[index])
        sample_pil = Image.open(file_path)
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)

    stacked = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=stacked)
    print(f"Saved .npz file to {npz_path} [shape={stacked.shape}].")
    return npz_path


def metric_summary(values):
    if not values:
        return {"mean": None, "std": None, "count": 0}

    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": None, "std": None, "count": int(arr.size), "finite_count": 0}

    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }


def format_metric_line(name, summary):
    mean = summary["mean"]
    std = summary["std"]
    if mean is None or std is None:
        return f"{name}: unavailable"
    return f"{name}: mean={mean:.6f} std={std:.6f}"


def resolve_run_name(args):
    if args.run_name:
        return args.run_name
    dataset_name = Path(args.data_path).name
    return (
        f"{dataset_name}-{args.vq_model.lower()}-size-{args.image_size}"
        f"-eval-{args.image_size_eval}-codebook-{args.codebook_size}-dim-{args.codebook_embed_dim}-seed-{args.seed}"
    )


def main():
    args = parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")
    if not os.path.isfile(args.vq_ckpt):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.vq_ckpt}")

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is not available: {args.device}")
        torch.cuda.set_device(device)

    print(f"Running on {device} with seed {args.seed}")

    print(f"Loading {args.vq_model} model...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    vq_model.to(device).eval()

    print(f"Loading checkpoint from {args.vq_ckpt}...")
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise RuntimeError("Unsupported checkpoint format. Expected one of: ema, model, state_dict.")
    vq_model.load_state_dict(model_weight)
    del checkpoint

    run_name = resolve_run_name(args)
    sample_folder_dir = os.path.join(args.sample_dir, run_name)
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving outputs to {sample_folder_dir}")

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    samples, class_names = discover_dataset(args.data_path)
    dataset = GenericImageFolderDataset(samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    lpips_metric = LPIPS().to(device).eval()

    psnr_values = []
    ssim_values = []
    mse_values = []
    lpips_values = []

    total_samples = 0

    print(f"Discovered {len(samples)} images across {len(class_names)} classes.")
    print("Starting reconstruction...")
    for x, _, _ in tqdm(loader, desc="Processing Batches"):
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode="bicubic", align_corners=False)
        else:
            rgb_gts = x

        rgb_gts_numpy = (rgb_gts.permute(0, 2, 3, 1).cpu().numpy() + 1.0) / 2.0
        x = x.to(device, non_blocking=True)

        with torch.no_grad():
            latent, _, [_, _, indices] = vq_model.encode(x)
            samples_tensor = vq_model.decode_code(indices, latent.shape)

            if args.image_size_eval != args.image_size:
                samples_tensor = F.interpolate(
                    samples_tensor,
                    size=(args.image_size_eval, args.image_size_eval),
                    mode="bicubic",
                    align_corners=False,
                )

            lpips_batch = lpips_metric(rgb_gts.to(device), samples_tensor).view(-1).detach().cpu().numpy()
            lpips_values.extend(float(value) for value in lpips_batch)

        samples_01 = ((samples_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).permute(0, 2, 3, 1).cpu().numpy()
        samples_uint8 = np.clip(np.rint(samples_01 * 255.0), 0, 255).astype(np.uint8)

        for idx_in_batch, (sample_01, sample_uint8, rgb_gt) in enumerate(zip(samples_01, samples_uint8, rgb_gts_numpy)):
            index = total_samples + idx_in_batch

            if not args.skip_save_images:
                Image.fromarray(sample_uint8).save(os.path.join(sample_folder_dir, f"{index:06d}.png"))

            mse = float(np.mean((sample_01 - rgb_gt) ** 2))
            psnr = float(psnr_loss(rgb_gt, sample_01, data_range=1.0))
            ssim = float(ssim_loss(rgb_gt, sample_01, data_range=1.0, channel_axis=-1))

            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

        total_samples += len(samples_01)

    summary = {
        "timestamp": int(time.time()),
        "dataset_path": args.data_path,
        "checkpoint_path": args.vq_ckpt,
        "num_images": total_samples,
        "num_classes": len(class_names),
        "class_names": class_names,
        "metrics": {
            "mse": metric_summary(mse_values),
            "psnr": metric_summary(psnr_values),
            "ssim": metric_summary(ssim_values),
            "lpips": metric_summary(lpips_values),
        },
    }

    print(f"Processed {total_samples} samples.")
    print(format_metric_line("MSE", summary["metrics"]["mse"]))
    print(format_metric_line("PSNR", summary["metrics"]["psnr"]))
    print(format_metric_line("SSIM", summary["metrics"]["ssim"]))
    print(format_metric_line("LPIPS", summary["metrics"]["lpips"]))

    summary_json_path = os.path.join(sample_folder_dir, "summary.json")
    summary_txt_path = os.path.join(sample_folder_dir, "results.txt")
    print(f"Writing summary to {summary_json_path}")
    with open(summary_json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(summary_txt_path, "w") as handle:
        handle.write(f"Processed {total_samples} samples.\n")
        handle.write(f"{format_metric_line('MSE', summary['metrics']['mse'])}\n")
        handle.write(f"{format_metric_line('PSNR', summary['metrics']['psnr'])}\n")
        handle.write(f"{format_metric_line('SSIM', summary['metrics']['ssim'])}\n")
        handle.write(f"{format_metric_line('LPIPS', summary['metrics']['lpips'])}\n")

    if not args.skip_export_npz:
        if args.skip_save_images:
            raise RuntimeError("NPZ export requires saved PNG reconstructions. Remove --skip-save-images or use --skip-export-npz.")
        create_npz_from_sample_folder(sample_folder_dir, num=total_samples)

    print("Done.")


if __name__ == "__main__":
    main()
