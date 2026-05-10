"""Visualization utilities for MAE reconstruction and training curves."""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) normalized → [0, 1] range."""
    mean = CIFAR100_MEAN.to(tensor.device)[:, None, None]
    std  = CIFAR100_STD.to(tensor.device)[:, None, None]
    return (tensor * std + mean).clamp(0, 1)


def visualize_reconstruction(model, imgs: torch.Tensor, save_path: str, device: torch.device, num_samples: int = 8):
    """
    Run MAE forward pass and plot: original | masked | reconstructed side by side.
    """
    model.eval()
    with torch.no_grad():
        imgs = imgs[:num_samples].to(device)
        _, pred, mask = model(imgs)

        grid_size = int(math.sqrt(mask.shape[1]))
        patch_size = model.patch_size

        # Reconstruct full image from predictions
        from einops import rearrange
        pred_pixels = rearrange(
            pred, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=grid_size, w=grid_size, p1=patch_size, p2=patch_size, c=3
        )

        # Build mask overlay — grey out visible patches so we see what was reconstructed
        mask_img = mask.reshape(num_samples, grid_size, grid_size)
        mask_img = mask_img.unsqueeze(1).repeat(1, 3, 1, 1)
        mask_img = torch.nn.functional.interpolate(mask_img.float(), scale_factor=patch_size)

        imgs_d    = denormalize(imgs).cpu()
        pred_d    = denormalize(pred_pixels).cpu()
        masked_d  = imgs_d * (1 - mask_img.cpu())   # grey = masked

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    for i in range(num_samples):
        for ax, img, title in zip(
            axes[i],
            [imgs_d[i], masked_d[i], pred_d[i]],
            ['Original', 'Masked', 'Reconstructed']
        ):
            ax.imshow(img.permute(1, 2, 0).numpy(), interpolation='bilinear')
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstruction visualization → {save_path}")


def plot_training_curves(log_path: str, save_path: str):
    """Parse a simple CSV log and plot loss / accuracy curves."""
    import csv
    epochs, train_loss, val_acc1, val_acc5 = [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row.get('train_loss', 0)))
            val_acc1.append(float(row.get('val_acc1', 0)))
            val_acc5.append(float(row.get('val_acc5', 0)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_acc1, label='Top-1 Acc')
    ax2.plot(epochs, val_acc5, label='Top-5 Acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Validation Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
