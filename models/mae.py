"""
Masked Autoencoder (MAE) — He et al., 2022 (https://arxiv.org/abs/2111.06377)

Architecture:
  - Asymmetric encoder-decoder
  - Encoder: full ViT on VISIBLE patches only (efficiency gain from masking)
  - Decoder: lightweight transformer that reconstructs ALL patch pixels
  - Loss: MSE on masked patches in pixel space (optionally normalized per patch)
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

from .vit import VisionTransformer, TransformerBlock, PatchEmbed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    2D sine-cosine positional embedding — no learnable params, generalizes to unseen resolutions.
    Returns (grid_size^2, embed_dim).
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')   # each: (grid_size, grid_size)
    grid = torch.stack(grid, dim=0).reshape(2, -1)          # (2, N)

    half = embed_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(half // 2, dtype=torch.float32) / (half // 2)))

    # Separate x and y encoding, then concatenate
    def encode(pos, omega):
        out = pos[:, None] * omega[None, :]   # (N, half//2)
        return torch.cat([out.sin(), out.cos()], dim=-1)   # (N, half)

    emb = torch.cat([encode(grid[0], omega), encode(grid[1], omega)], dim=-1)   # (N, embed_dim)
    return emb


class MAEEncoder(nn.Module):
    """
    MAE Encoder = standard ViT that only processes VISIBLE (unmasked) patches.
    No CLS token during pre-training (added during fine-tuning).
    """

    def __init__(self, image_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = image_size // patch_size

        # Fixed sincos positional embedding — registered as buffer (not trained)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))   # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def random_masking(self, x: torch.Tensor, mask_ratio: float):
        """
        Per-sample random masking via argsort of uniform noise.
        Returns:
          x_visible:   (B, N_vis, D)  — visible patch tokens
          mask:        (B, N)          — 1 = masked, 0 = visible
          ids_restore: (B, N)          — indices to un-shuffle tokens
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)           # ascending: lowest noise = kept
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :n_keep]
        x_visible = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0)                # 0 = visible

        return x_visible, mask, ids_restore

    def forward(self, x: torch.Tensor, mask_ratio: float):
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed             # add fixed pos embed to ALL patches

        x_vis, mask, ids_restore = self.random_masking(tokens, mask_ratio)

        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)

        return x_vis, mask, ids_restore


class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder that reconstructs pixels of ALL patches.
    Visible encoded tokens + learned mask tokens → pixel predictions.
    """

    def __init__(self, num_patches, encoder_dim, decoder_dim, decoder_depth, decoder_heads,
                 patch_size, in_channels, grid_size, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        patch_pixels = in_channels * patch_size * patch_size

        # Project encoder dim → decoder dim
        self.embed = nn.Linear(encoder_dim, decoder_dim, bias=True)

        # Learned mask token shared across all masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Fixed sincos pos embed for decoder (full N patches)
        pos_embed = get_2d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_pixels, bias=True)   # pixel reconstruction head

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pred.weight, std=0.02)

    def forward(self, x_vis: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        B, N_vis, _ = x_vis.shape
        x_vis = self.embed(x_vis)

        # Expand mask tokens to fill masked positions
        N = ids_restore.shape[1]
        n_masked = N - N_vis
        mask_tokens = self.mask_token.expand(B, n_masked, -1)

        # Concatenate visible + mask tokens, then un-shuffle to original order
        x_full = torch.cat([x_vis, mask_tokens], dim=1)
        x_full = x_full.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))

        x_full = x_full + self.pos_embed

        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)
        return self.pred(x_full)   # (B, N, patch_pixels)


class MaskedAutoencoder(nn.Module):
    """
    Full MAE model: encoder + decoder + reconstruction loss.
    During pre-training: forward() returns (loss, pred, mask).
    Encoder weights are reused for downstream fine-tuning.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 384,
        encoder_depth: int = 6,
        encoder_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_channels = in_channels

        num_patches = (image_size // patch_size) ** 2
        grid_size = image_size // patch_size

        self.encoder = MAEEncoder(image_size, patch_size, in_channels, embed_dim,
                                  encoder_depth, encoder_heads, mlp_ratio)
        self.decoder = MAEDecoder(num_patches, embed_dim, decoder_embed_dim,
                                  decoder_depth, decoder_heads, patch_size,
                                  in_channels, grid_size, mlp_ratio)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_pixels) — ground truth for reconstruction."""
        p = self.patch_size
        return rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)

    def unpatchify(self, patches: torch.Tensor, grid_size: int) -> torch.Tensor:
        """(B, N, patch_pixels) → (B, C, H, W) — for visualization."""
        p = self.patch_size
        return rearrange(patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=grid_size, w=grid_size, p1=p, p2=p, c=self.in_channels)

    def forward(self, imgs: torch.Tensor):
        x_vis, mask, ids_restore = self.encoder(imgs, self.mask_ratio)
        pred = self.decoder(x_vis, ids_restore)   # (B, N, patch_pixels)

        target = self.patchify(imgs)

        if self.norm_pix_loss:
            # Normalize each patch independently — prevents model from trivially
            # learning mean color; encourages structural reconstruction
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)   # (B, N)
        # Only supervise masked patches — not visible ones
        loss = (loss_per_patch * mask).sum() / mask.sum()

        return loss, pred, mask
