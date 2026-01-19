"""Minimal DINOv3 loader + test

This script defines a small ViT-like class compatible with the checkpoint keys
present in `models/dinov3-vitb16/dinov3_vitb16_pretrain_*.pth` and attempts to
load the checkpoint into the model. It converts bfloat16 tensors to float32
and strips common prefixes.

Usage:
    python scripts/load_dinov3.py
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Minimal ViT block components (lightweight) to match common DINOv3 key naming.
# This is intentionally small: it aims to align parameter names and shapes so
# the checkpoint can be loaded. It is NOT a full DINOv3 reimplementation.

class SimplePatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

class SimpleMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Using qkv as a single linear to match many checkpoints
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SimpleMlp(dim)
        # accommodate ls1/ls2 gamma style params by exposing attributes
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.ls2 = nn.Parameter(torch.ones(dim))

class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_classes=0):
        super().__init__()
        self.patch_embed = SimplePatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # storage_tokens and mask_token from DINOv3: add as optional buffers/params
        self.storage_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList([SimpleBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # head (classification) which many checkpoints expect
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: B x 3 x H x W
        x = self.patch_embed.proj(x)  # B x C x H' x W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x N x C
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk.norm1(x)
            # qkv: B x N x (3*dim). Combine into a single representation
            qkv = blk.attn.qkv(x)
            dim = qkv.shape[-1] // 3
            # reshape to B x N x 3 x dim and sum the three parts as a simple proxy
            combined = qkv.view(qkv.shape[0], qkv.shape[1], 3, dim).sum(dim=2)
            x = x + blk.attn.proj(combined)
            x = x + blk.mlp(blk.norm2(x))
        x = self.norm(x)
        return x[:, 0]


def load_checkpoint_into_model(weights_path: Path, model: nn.Module):
    sd = torch.load(str(weights_path), map_location='cpu')
    # If the checkpoint is an OrderedDict directly, use it. If nested, try 'model'/'teacher'.
    if isinstance(sd, dict) and ('model' in sd or 'teacher' in sd):
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd['teacher']

    if not isinstance(sd, dict):
        raise RuntimeError('Unexpected checkpoint format: not a dict')

    # Convert bfloat16 to float32 if present and remove common prefixes
    new_sd = {}
    for k, v in sd.items():
        new_k = k.replace('module.', '').replace('backbone.', '')
        if hasattr(v, 'dtype') and v.dtype == torch.bfloat16:
            v = v.to(torch.float32)
        new_sd[new_k] = v

    # Load state dict permissively
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    return missing, unexpected


def main():
    weights = Path('models/dinov3-vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    print('Weights path:', weights)
    if not weights.exists():
        print('Weights not found. Exiting.')
        sys.exit(1)

    print('Instantiating SimpleViT...')
    model = SimpleViT()
    print('Loading checkpoint...')
    missing, unexpected = load_checkpoint_into_model(weights, model)
    print('Loaded checkpoint. Missing keys:', len(missing), 'Unexpected keys:', len(unexpected))
    if missing:
        print('Missing (first 10):', missing[:10])
    if unexpected:
        print('Unexpected (first 10):', unexpected[:10])

    # Quick forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
    print('Forward pass output shape:', out.shape)

if __name__ == '__main__':
    main()
