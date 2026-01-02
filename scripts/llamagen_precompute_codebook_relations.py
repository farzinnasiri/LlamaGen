import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from tokenizer.tokenizer_image.vq_model import VQ_models


def load_model(vq_model_name, ckpt_path, codebook_size, codebook_embed_dim, device):
    model = VQ_models[vq_model_name](
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim,
    )
    model.to(device).eval()

    checkpoint = torch.load(ckpt_path, map_location="cpu")
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


def compute_maps_full(codebook, block_size):
    if codebook.ndim != 2:
        raise ValueError(f"Expected codebook 2D, got shape {tuple(codebook.shape)}")
    n_embed, embed_dim = codebook.shape

    device = codebook.device
    e = codebook.to(dtype=torch.float32)
    norms = (e * e).sum(dim=1)
    e_t = e.transpose(0, 1)

    eps = 1e-12
    e_norm = e / (e.norm(dim=1, keepdim=True) + eps)
    e_norm_t = e_norm.transpose(0, 1)

    min_dist_idx = np.empty((n_embed,), dtype=np.int32)
    max_dist_idx = np.empty((n_embed,), dtype=np.int32)
    ortho_idx = np.empty((n_embed,), dtype=np.int32)

    min_dist_val = np.empty((n_embed,), dtype=np.float32)
    max_dist_val = np.empty((n_embed,), dtype=np.float32)
    ortho_cos_val = np.empty((n_embed,), dtype=np.float32)

    for start in range(0, n_embed, int(block_size)):
        end = min(n_embed, start + int(block_size))
        b = e[start:end]
        b_norms = norms[start:end]
        dist2 = b_norms[:, None] + norms[None, :] - 2.0 * (b @ e_t)
        dist2 = dist2.clamp_min_(0.0)

        row = torch.arange(end - start, device=device)
        col = row + start

        dist2_min = dist2.clone()
        dist2_min[row, col] = float("inf")
        min_res = dist2_min.min(dim=1)
        min_j = min_res.indices
        min_v = torch.sqrt(min_res.values)

        dist2_max = dist2.clone()
        dist2_max[row, col] = -float("inf")
        max_res = dist2_max.max(dim=1)
        max_j = max_res.indices
        max_v = torch.sqrt(max_res.values)

        b_e_norm = e_norm[start:end]
        cos = b_e_norm @ e_norm_t
        abs_cos = cos.abs()
        abs_cos[row, col] = float("inf")
        ortho_j = abs_cos.argmin(dim=1)
        ortho_v = cos.gather(1, ortho_j.unsqueeze(1)).squeeze(1)

        min_dist_idx[start:end] = min_j.detach().cpu().numpy().astype(np.int32)
        max_dist_idx[start:end] = max_j.detach().cpu().numpy().astype(np.int32)
        ortho_idx[start:end] = ortho_j.detach().cpu().numpy().astype(np.int32)

        min_dist_val[start:end] = min_v.detach().cpu().numpy().astype(np.float32)
        max_dist_val[start:end] = max_v.detach().cpu().numpy().astype(np.float32)
        ortho_cos_val[start:end] = ortho_v.detach().cpu().numpy().astype(np.float32)

    if np.any(min_dist_idx == np.arange(n_embed, dtype=np.int32)):
        raise RuntimeError("min_dist_idx contains self-maps")
    if np.any(max_dist_idx == np.arange(n_embed, dtype=np.int32)):
        raise RuntimeError("max_dist_idx contains self-maps")
    if np.any(ortho_idx == np.arange(n_embed, dtype=np.int32)):
        raise RuntimeError("ortho_idx contains self-maps")

    alive_token_ids = np.arange(n_embed, dtype=np.int32)
    return (
        alive_token_ids,
        min_dist_idx,
        max_dist_idx,
        ortho_idx,
        min_dist_val,
        max_dist_val,
        ortho_cos_val,
        int(n_embed),
        int(embed_dim),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vq-model", type=str, default="VQ-16", choices=list(VQ_models.keys()))
    p.add_argument("--vq-ckpt", type=str, default="/checkpoints/vq_ds16_c2i.pt")
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--out", type=str, default="llamagen_codebook_relations.npz")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--block-size", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.vq_model, args.vq_ckpt, args.codebook_size, args.codebook_embed_dim, device)
    codebook = model.quantize.embedding.weight.detach()
    if getattr(model.quantize, "l2_norm", False):
        codebook = F.normalize(codebook, p=2, dim=-1)
    codebook = codebook.to(device=device)

    (
        alive_token_ids,
        min_dist_idx,
        max_dist_idx,
        ortho_idx,
        min_dist_val,
        max_dist_val,
        ortho_cos_val,
        n_embed,
        embed_dim,
    ) = compute_maps_full(codebook, args.block_size)

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        out_path,
        alive_token_ids=alive_token_ids.astype(np.int32),
        min_dist_idx=min_dist_idx,
        max_dist_idx=max_dist_idx,
        ortho_idx=ortho_idx,
        min_dist_val=min_dist_val,
        max_dist_val=max_dist_val,
        ortho_cos_val=ortho_cos_val,
        n_embed=np.int32(n_embed),
        embed_dim=np.int32(embed_dim),
    )

    print(f"Saved: {out_path}")
    print(f"n_embed={n_embed} embed_dim={embed_dim} alive={int(alive_token_ids.shape[0])}")


if __name__ == "__main__":
    main()
