"""
ViT fine-tuning head for downstream classification.
Loads MAE encoder weights and adds a classification head.
Supports:
  - Linear probing (freeze encoder, train head only)
  - Full fine-tuning with layer-wise learning rate decay
"""

import torch
import torch.nn as nn
from .vit import VisionTransformer


class ViTClassifier(nn.Module):
    """
    ViT encoder + classification head.
    The encoder architecture must match the pre-trained MAE encoder.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        num_classes: int = 100,
        drop_path_rate: float = 0.1,
        global_pool: bool = True,
    ):
        super().__init__()
        self.global_pool = global_pool

        self.encoder = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            use_cls_token=not global_pool,
        )

        self.norm = nn.LayerNorm(embed_dim) if global_pool else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)   # (B, N+1, D) or (B, N, D)
        if self.global_pool:
            # Average all patch tokens, skip nothing (no CLS)
            features = features.mean(dim=1)
            features = self.norm(features)
        else:
            features = features[:, 0]  # CLS token
        return self.head(features)

    def load_mae_weights(self, checkpoint_path: str, device: torch.device):
        """
        Load encoder weights from MAE pre-training checkpoint.
        The MAEEncoder and VisionTransformer share identical block architecture,
        but positional embeddings differ (fixed sincos vs learnable).
        We load everything that matches by name.
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get('model_state', ckpt)

        # Keys from MAE encoder are prefixed 'encoder.*'
        encoder_state = {
            k.replace('encoder.', ''): v
            for k, v in state.items()
            if k.startswith('encoder.')
        }

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[load_mae_weights] Missing: {missing}")
        print(f"[load_mae_weights] Unexpected: {unexpected}")

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def get_layer_groups(self):
        """
        Return parameter groups ordered from input to output (for layer-wise LR decay).
        Group 0 = patch embedding, Group k = block k, last = norm + head.
        """
        groups = [
            {'params': list(self.encoder.patch_embed.parameters()), 'name': 'patch_embed'},
        ]
        for i, blk in enumerate(self.encoder.blocks):
            groups.append({'params': list(blk.parameters()), 'name': f'block_{i}'})
        groups.append({
            'params': (
                list(self.encoder.norm.parameters()) +
                list(self.norm.parameters()) +
                list(self.head.parameters())
            ),
            'name': 'head',
        })
        return groups
