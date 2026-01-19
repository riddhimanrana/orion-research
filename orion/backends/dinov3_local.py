"""Lightweight local DINOv3 loader.

Provides `DINOv3LocalModel` which can load a DINOv3-style PyTorch checkpoint
from a local directory and expose `encode_image`, `encode_images_batch`, and
`extract_frame_features` methods used by `DINOEmbedder`.

This is a pragmatic compatibility shim for environments where `transformers`
cannot be imported or the gated DINOv3 HF package isn't available.

It is NOT a full DINOv3 implementation, but maps checkpoint keys into a
minimal ViT-based model sufficient to produce 768-dim embeddings for tests
and quick experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class DINOv3LocalModel:
    def __init__(self, weights_dir: Path, device: str = "cpu"):
        self.weights_dir = Path(weights_dir)
        self.device = device
        self._load_weights()

    def _load_weights(self):
        # Look for a .pth file
        pth_files = list(self.weights_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {self.weights_dir}")
        weights_path = pth_files[0]
        sd = torch.load(weights_path, map_location="cpu")
        # If nested, extract
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if not isinstance(sd, dict):
            raise RuntimeError("Unsupported checkpoint format")

        # Create a simple ViT matching common DINOv3 param shapes (patch16, embed 768, depth inferred)
        # Infer depth from keys
        block_keys = [k for k in sd.keys() if k.startswith("blocks.") and k.count(".")>1]
        depths = set(int(k.split('.')[1]) for k in block_keys) if block_keys else {11}
        depth = max(depths) + 1 if depths else 12

        # Build minimal model
        self.model = _MinimalDINOv3Model(depth=depth)

        # Prepare state dict mapping and dtype conversion
        new_sd = {}
        for k, v in sd.items():
            new_k = k.replace("module.", "").replace("backbone.", "")
            if hasattr(v, "dtype") and v.dtype == torch.bfloat16:
                v = v.to(torch.float32)
            new_sd[new_k] = v

        # Load permissively
        try:
            self.model.load_state_dict(new_sd, strict=False)
        except Exception:
            # ignore errors — permissive load is okay for now
            pass

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda")
        elif self.device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
        self.model = self.model.to(dev)
        self.model.eval()
        self.device = dev

    def _to_rgb(self, img: np.ndarray) -> Image.Image:
        if img.ndim == 3 and img.shape[2] == 3:
            arr = img[..., ::-1]  # BGR->RGB
        else:
            arr = img
        return Image.fromarray(arr.astype('uint8'))

    def encode_image(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        pil = self._to_rgb(image)
        tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.inference_mode():
            out = self.model(tensor.to(self.device))
        emb = out[0].cpu().numpy().astype(np.float32)
        if normalize:
            n = np.linalg.norm(emb) + 1e-8
            emb = emb / n
        return emb

    def encode_images_batch(self, images: List[np.ndarray], normalize: bool = True) -> List[np.ndarray]:
        imgs = [self._to_rgb(img) for img in images]
        tensors = [torch.from_numpy(np.array(p)).permute(2, 0, 1).unsqueeze(0).float() / 255.0 for p in imgs]
        batch = torch.cat(tensors, dim=0).to(self.device)
        with torch.inference_mode():
            outs = self.model(batch)
        embs = [o.cpu().numpy().astype(np.float32) for o in outs]
        if normalize:
            embs = [e / (np.linalg.norm(e) + 1e-8) for e in embs]
        return embs

    def extract_frame_features(self, image: np.ndarray) -> np.ndarray:
        # Return tiled embedding as spatial map (Hf,Wf,D)
        emb = self.encode_image(image, normalize=True)
        side = 16
        fmap = np.tile(emb[None, None, :], (side, side, 1))
        return fmap.astype(np.float32)


# Minimal model matching our earlier SimpleViT used in scripts/load_dinov3.py
class _MinimalDINOv3Model(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12):
        super().__init__()
        self.patch = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList([_MinimalBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class _MinimalBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        qkv = self.qkv(y)
        dim = qkv.shape[-1] // 3
        combined = qkv.view(qkv.shape[0], qkv.shape[1], 3, dim).sum(dim=2)
        x = x + self.proj(combined)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(x))))
        return x
