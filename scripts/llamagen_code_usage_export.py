import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models


VALID_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Export LlamaGen tokenizer code usage for an image-folder dataset.")
    parser.add_argument("--data-path", type=str, required=True, help="Root directory containing one subdirectory per class.")
    parser.add_argument("--vq-ckpt", type=str, required=True, help="Path to the VQ checkpoint.")
    parser.add_argument("--outdir", type=str, required=True, help="Directory where usage outputs are written.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--vq-model", type=str, default="VQ-16", choices=["VQ-16", "VQ-8"])
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-indices", action="store_true", help="Also save per-image token grids to indices.npz.")
    return parser.parse_args()


def class_sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_dataset(root: str):
    root_path = Path(root)
    class_dirs = sorted([path for path in root_path.iterdir() if path.is_dir()], key=class_sort_key)

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

    return samples, class_names


class ImageFolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, path


def load_model(args, device):
    model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    )
    model.to(device).eval()

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise KeyError(f"Unexpected checkpoint keys: {list(checkpoint.keys())}")
    model.load_state_dict(model_weight)
    return model


def token_grid_from_indices(indices, batch_size):
    ind_flat = indices.reshape(batch_size, -1)
    side = int(round(ind_flat.shape[1] ** 0.5))
    ind_grid = ind_flat.reshape(batch_size, side, side)
    return ind_grid


def update_counts(ind_grid, global_counts, position_counts):
    ind_np = ind_grid.detach().cpu().numpy().astype(np.int64)
    global_counts += np.bincount(ind_np.ravel(), minlength=global_counts.shape[0])

    height, width = ind_np.shape[1], ind_np.shape[2]
    for row in range(height):
        for col in range(width):
            position_counts[:, row, col] += np.bincount(ind_np[:, row, col], minlength=global_counts.shape[0])

    return ind_np


def entropy_and_perplexity(counts):
    total = int(counts.sum())
    probs = counts[counts > 0].astype(np.float64) / float(total)
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy, float(np.exp(entropy))


def top_mass(counts, k):
    total = int(counts.sum())
    return float(np.sort(counts)[-k:].sum() / float(total))


def write_usage_csv(path, counts):
    order = np.argsort(-counts)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(counts) + 1)
    total = int(counts.sum())

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code_id", "count", "frequency", "is_active", "rank"])
        for code_id, count in enumerate(counts.tolist()):
            writer.writerow([code_id, count, count / total, int(count > 0), int(ranks[code_id])])


def write_top_codes_csv(path, counts, top_k=500):
    order = np.argsort(-counts)[:top_k]
    total = int(counts.sum())

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "code_id", "count", "frequency"])
        for rank, code_id in enumerate(order.tolist(), start=1):
            count = int(counts[code_id])
            writer.writerow([rank, code_id, count, count / total])


def main():
    args = parse_args()
    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    samples, class_names = discover_dataset(args.data_path)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    print(f"Dataset: {args.dataset_name}")
    print(f"Images: {len(samples)} across {len(class_names)} classes")
    print(f"Output: {outdir}")

    dataset = ImageFolderDataset(samples, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = load_model(args, device)

    global_counts = np.zeros(args.codebook_size, dtype=np.int64)
    position_counts = None
    all_indices = []
    paths = []

    for batch, _, batch_paths in tqdm(loader, desc="Encoding"):
        batch = batch.to(device)
        quant, _, (_, _, indices) = model.encode(batch)
        height_tok, width_tok = int(quant.shape[2]), int(quant.shape[3])
        ind_grid = token_grid_from_indices(indices, batch.shape[0])

        if position_counts is None:
            position_counts = np.zeros((args.codebook_size, height_tok, width_tok), dtype=np.int64)

        ind_np = update_counts(ind_grid, global_counts, position_counts)
        if args.save_indices:
            all_indices.append(ind_np.astype(np.int32))
            paths.extend(list(batch_paths))

    active_codes = int((global_counts > 0).sum())
    entropy, perplexity = entropy_and_perplexity(global_counts)

    np.save(outdir / "global_counts.npy", global_counts)
    np.save(outdir / "position_counts.npy", position_counts)
    write_usage_csv(outdir / "usage.csv", global_counts)
    write_top_codes_csv(outdir / "top_codes.csv", global_counts)

    if args.save_indices:
        np.savez_compressed(
            outdir / "indices.npz",
            indices=np.concatenate(all_indices, axis=0),
            paths=np.asarray(paths),
        )

    summary = {
        "model": "LlamaGen",
        "dataset": args.dataset_name,
        "data_path": args.data_path,
        "n_images": len(samples),
        "n_classes": len(class_names),
        "codebook_size": args.codebook_size,
        "token_grid_hw": [int(position_counts.shape[1]), int(position_counts.shape[2])],
        "total_tokens": int(global_counts.sum()),
        "active_codes": active_codes,
        "dead_codes": int(args.codebook_size - active_codes),
        "active_fraction_full": float(active_codes / args.codebook_size),
        "entropy": entropy,
        "perplexity": perplexity,
        "top_10_mass": top_mass(global_counts, 10),
        "top_100_mass": top_mass(global_counts, 100),
        "top_500_mass": top_mass(global_counts, 500),
        "class_names": class_names,
    }

    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
