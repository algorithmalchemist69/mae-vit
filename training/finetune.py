"""
Fine-tuning loop with:
  - Layer-wise learning rate decay (LLRD)
  - Mixup / CutMix augmentation
  - Label smoothing
  - Top-1 and Top-5 accuracy tracking
"""

import os
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import ViTClassifier
from utils import build_dataset, build_dataloader, AverageMeter, accuracy, topk_accuracy
from utils.scheduler import CosineWarmupScheduler


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_param_groups_with_llrd(model: ViTClassifier, base_lr: float, layer_decay: float, weight_decay: float):
    """
    Layer-wise LR decay: each layer closer to input gets lr *= layer_decay.
    Bias / LayerNorm params get no weight decay (standard practice).
    """
    groups = model.get_layer_groups()
    n_layers = len(groups)
    param_groups = []
    for i, group in enumerate(groups):
        # Scale: head layer gets base_lr, earlier layers decay exponentially
        scale = layer_decay ** (n_layers - 1 - i)
        no_decay = [p for name, p in zip([], group['params'])
                    if p.ndim <= 1]   # bias and norm weights
        with_decay = [p for p in group['params'] if p.ndim > 1]

        param_groups += [
            {'params': with_decay, 'lr': base_lr * scale, 'weight_decay': weight_decay,
             'lr_scale': scale, 'name': group['name']},
            {'params': no_decay,   'lr': base_lr * scale, 'weight_decay': 0.0,
             'lr_scale': scale, 'name': f"{group['name']}_no_decay"},
        ]
    return param_groups


def mixup_data(x, y, alpha=0.8):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def run_finetune(cfg: dict):
    device = get_device()
    print(f"Fine-tuning on: {device}")

    exp_dir = Path('experiments') / cfg['experiment_name']
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / 'log.csv'

    writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))

    # Data
    train_ds = build_dataset('train', mode='finetune', image_size=cfg['data']['image_size'])
    val_ds   = build_dataset('val',   mode='finetune', image_size=cfg['data']['image_size'])
    train_dl = build_dataloader(train_ds, cfg['training']['batch_size'], cfg['data']['num_workers'])
    val_dl   = build_dataloader(val_ds, cfg['training']['batch_size'], cfg['data']['num_workers'], shuffle=False)

    # Model
    m_cfg = cfg['model']
    t_cfg = cfg['training']
    model = ViTClassifier(
        image_size=cfg['data']['image_size'],
        patch_size=m_cfg['patch_size'],
        embed_dim=m_cfg['embed_dim'],
        depth=m_cfg['encoder_depth'],
        num_heads=m_cfg['encoder_heads'],
        num_classes=m_cfg['num_classes'],
        drop_path_rate=m_cfg['drop_path_rate'],
        global_pool=m_cfg['global_pool'],
    ).to(device)

    # Load MAE pre-trained weights
    pretrain_ckpt = cfg.get('pretrained_checkpoint')
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        model.load_mae_weights(pretrain_ckpt, device)
        print(f"Loaded MAE weights from {pretrain_ckpt}")
    else:
        print("WARNING: No pre-trained checkpoint found — training from scratch")

    # Optimizer with LLRD
    base_lr = t_cfg['base_lr'] * t_cfg['batch_size'] / 256
    param_groups = build_param_groups_with_llrd(
        model, base_lr, t_cfg['layer_decay'], t_cfg['weight_decay']
    )
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr)

    steps_per_epoch = len(train_dl)
    total_steps  = t_cfg['epochs'] * steps_per_epoch
    warmup_steps = t_cfg['warmup_epochs'] * steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps, base_lr, t_cfg['min_lr'])

    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg['label_smoothing'])

    # Resume
    start_epoch = 0
    best_acc1 = 0.0
    latest = ckpt_dir / 'latest.pth'
    if latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.current_step = ckpt['step']
        start_epoch = ckpt['epoch'] + 1
        best_acc1 = ckpt.get('best_acc1', 0.0)
        print(f"Resumed from epoch {start_epoch}, best_acc1={best_acc1:.2f}%")

    write_header = not log_path.exists()
    log_file = open(log_path, 'a', newline='')
    csv_wr = csv.DictWriter(log_file, fieldnames=['epoch', 'train_loss', 'val_acc1', 'val_acc5', 'lr', 'time'])
    if write_header:
        csv_wr.writeheader()

    use_mixup = t_cfg.get('mixup_prob', 0) > 0
    mixup_alpha = t_cfg.get('mixup_alpha', 0.8)

    for epoch in range(start_epoch, t_cfg['epochs']):
        t0 = time.time()
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{t_cfg["epochs"]} [train]', leave=False)
        for step, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if use_mixup and torch.rand(1).item() < t_cfg['mixup_prob']:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, mixup_alpha)
                logits = model(imgs)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if t_cfg.get('clip_grad'):
                nn.utils.clip_grad_norm_(model.parameters(), t_cfg['clip_grad'])
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), imgs.size(0))
            if step % cfg['logging']['log_every'] == 0:
                global_step = epoch * steps_per_epoch + step
                writer.add_scalar('finetune/loss', loss.item(), global_step)
            pbar.set_postfix(loss=f'{loss_meter.avg:.4f}')

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        with torch.no_grad():
            for imgs, labels in tqdm(val_dl, desc=f'Epoch {epoch+1} [val]', leave=False):
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                logits = model(imgs)
                acc1_meter.update(accuracy(logits, labels), imgs.size(0))
                acc5_meter.update(topk_accuracy(logits, labels, k=5), imgs.size(0))

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d} | loss={loss_meter.avg:.4f} | "
              f"top1={acc1_meter.avg:.2f}% | top5={acc5_meter.avg:.2f}% | {elapsed:.0f}s")

        writer.add_scalar('finetune/val_acc1', acc1_meter.avg, epoch)
        writer.add_scalar('finetune/val_acc5', acc5_meter.avg, epoch)
        csv_wr.writerow({
            'epoch': epoch + 1,
            'train_loss': round(loss_meter.avg, 5),
            'val_acc1': round(acc1_meter.avg, 3),
            'val_acc5': round(acc5_meter.avg, 3),
            'lr': round(scheduler.get_last_lr(), 8),
            'time': round(elapsed, 1),
        })
        log_file.flush()

        # ── Checkpoints ──────────────────────────────────────────────────
        state = {
            'epoch': epoch,
            'step': scheduler.current_step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc1': best_acc1,
            'cfg': cfg,
        }
        torch.save(state, ckpt_dir / 'latest.pth')
        if acc1_meter.avg > best_acc1:
            best_acc1 = acc1_meter.avg
            torch.save(state, ckpt_dir / 'best.pth')
            print(f"  New best top-1: {best_acc1:.2f}%")
        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            torch.save(state, ckpt_dir / f'epoch_{epoch+1:04d}.pth')

    log_file.close()
    writer.close()
    print(f"\nFine-tuning complete. Best top-1 accuracy: {best_acc1:.2f}%")
