# MAE-ViT: Masked Autoencoder with Vision Transformer

Self-supervised visual representation learning using Masked Autoencoders (He et al., 2022) on CIFAR-100.

## Results

| Metric | Value |
|---|---|
| **Top-1 Accuracy** (CIFAR-100 test) | **74.42%** |
| **Top-5 Accuracy** (CIFAR-100 test) | **93.92%** |
| Best pre-train val loss | 0.434 |
| Pre-training epochs | 200 |
| Fine-tuning epochs | 100 |
| Training hardware | Apple M4 Pro (MPS) |

## Architecture

```
Pre-training (MAE):
  Input image (32×32)
       ↓ Patch Embed (4×4 patches → 64 tokens)
       ↓ Random Mask 75% of patches
  ViT Encoder (6 blocks, dim=384) ← only sees 25% of patches
       ↓ 16 visible tokens
  Lightweight Decoder (4 blocks, dim=192)
       ↓ Mask tokens + sincos pos embed
  Pixel Reconstruction (MSE on masked patches)

Fine-tuning:
  Frozen/unfrozen ViT Encoder + Global Average Pool → Linear Head → 100 classes
  Layer-wise LR decay + Mixup + RandAugment
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

### Step 1 — MAE Pre-training (~3–5 hours on M4 Pro)
```bash
python train.py --mode pretrain --config configs/pretrain.yaml
```

Quick smoke test (5 epochs):
```bash
python train.py --mode pretrain --config configs/pretrain.yaml --epochs 5
```

### Step 2 — Fine-tuning (~1–2 hours on M4 Pro)
```bash
python train.py --mode finetune --config configs/finetune.yaml
```

### TensorBoard
```bash
tensorboard --logdir experiments/
```

## Project Structure

```
mae_vit/
├── models/
│   ├── vit.py          # Vision Transformer (patch embed, MHSA, transformer blocks)
│   ├── mae.py          # Masked Autoencoder (encoder + decoder + masking)
│   └── classifier.py   # Fine-tuning head with LLRD support
├── training/
│   ├── pretrain.py     # MAE pre-training loop
│   └── finetune.py     # Downstream classification fine-tuning
├── utils/
│   ├── data.py         # CIFAR-100 data loading + augmentation pipelines
│   ├── metrics.py      # AverageMeter, top-1/5 accuracy
│   ├── scheduler.py    # Cosine warmup LR scheduler
│   └── visualization.py # Reconstruction plots, training curves
├── configs/
│   ├── pretrain.yaml   # MAE pre-training hyperparameters
│   └── finetune.yaml   # Fine-tuning hyperparameters
├── notebooks/
│   └── analysis.ipynb  # Attention maps, t-SNE, reconstruction visualization
└── experiments/        # Checkpoints, logs, TensorBoard events (auto-created)
```

## Implementation Details

| Component | Detail |
|---|---|
| Encoder | ViT-Small (6 blocks, 384 dim, 6 heads) |
| Decoder | Lightweight (4 blocks, 192 dim, 6 heads) |
| Masking ratio | 75% — only 16/64 patches visible to encoder |
| Positional embedding | Fixed 2D sincos (encoder), learned (ViT fine-tune) |
| Reconstruction target | Normalized pixel values per patch |
| LR schedule | Cosine annealing with linear warmup |
| Optimizer | AdamW (β₁=0.9, β₂=0.95 for pre-train) |
| Fine-tune tricks | Layer-wise LR decay (decay=0.65), Mixup, RandAugment, Label smoothing |
| MPS support | Full Apple Silicon GPU acceleration via `torch.backends.mps` |

## Key Concepts 

**Why mask 75%?** High masking ratio forces the model to learn semantic understanding rather than memorizing local textures. Low-level interpolation is insufficient — the model must understand the global structure.

**Why asymmetric encoder-decoder?** The encoder (expensive ViT) only processes unmasked tokens → 4× speedup during pre-training. The decoder (cheap) handles reconstruction.

**Why normalize pixel targets?** Prevents the model from optimizing mean color prediction. Patch normalization removes per-patch intensity bias, encouraging structural learning.

**Layer-wise LR decay:** Layers closer to the input encode low-level features learned during pre-training that should not change much. Higher LR closer to the output (classification head) for task adaptation.

## References

- He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021) — https://arxiv.org/abs/2111.06377
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (2020) — https://arxiv.org/abs/2010.11929
- Chen et al., "A Simple Framework for Contrastive Learning" (2020)
