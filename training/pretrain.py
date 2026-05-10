"""MAE pre-training loop with MPS / CUDA / CPU support."""

import os
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import MaskedAutoencoder
from utils import build_dataset, build_dataloader, AverageMeter
from utils import visualize_reconstruction
from utils.scheduler import CosineWarmupScheduler


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(cfg) -> MaskedAutoencoder:
    return MaskedAutoencoder(
        image_size=cfg['data']['image_size'],
        patch_size=cfg['model']['patch_size'],
        embed_dim=cfg['model']['embed_dim'],
        encoder_depth=cfg['model']['encoder_depth'],
        encoder_heads=cfg['model']['encoder_heads'],
        decoder_embed_dim=cfg['model']['decoder_embed_dim'],
        decoder_depth=cfg['model']['decoder_depth'],
        decoder_heads=cfg['model']['decoder_heads'],
        mask_ratio=cfg['model']['mask_ratio'],
        norm_pix_loss=cfg['model']['norm_pix_loss'],
    )


def run_pretrain(cfg: dict):
    device = get_device()
    print(f"Training on: {device}")

    exp_dir = Path('experiments') / cfg['experiment_name']
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / 'log.csv'
    viz_dir  = exp_dir / 'visualizations'

    writer = SummaryWriter(log_dir=str(exp_dir / 'tensorboard'))

    # Data
    train_ds = build_dataset('train', mode='pretrain',
                              image_size=cfg['data']['image_size'])
    train_dl = build_dataloader(train_ds, cfg['training']['batch_size'],
                                cfg['data']['num_workers'])

    val_ds = build_dataset('val', mode='pretrain',
                            image_size=cfg['data']['image_size'])
    val_dl = build_dataloader(val_ds, cfg['training']['batch_size'],
                              cfg['data']['num_workers'], shuffle=False)

    # Model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"MAE parameters: {n_params:.1f}M")

    # Optimizer — AdamW with cosine LR
    t_cfg = cfg['training']
    effective_lr = t_cfg['base_lr'] * t_cfg['batch_size'] / 256
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=t_cfg['weight_decay'],
        betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_dl)
    total_steps = t_cfg['epochs'] * steps_per_epoch
    warmup_steps = t_cfg['warmup_epochs'] * steps_per_epoch
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps,
                                       effective_lr, t_cfg['min_lr'])

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    latest = ckpt_dir / 'latest.pth'
    if latest.exists():
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.current_step = ckpt['step']
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', best_loss)
        print(f"Resumed from epoch {start_epoch}")

    # CSV log
    write_header = not log_path.exists()
    log_file = open(log_path, 'a', newline='')
    csv_writer = csv.DictWriter(log_file, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr', 'time'])
    if write_header:
        csv_writer.writeheader()

    # Grab a fixed batch for visualization
    vis_imgs, _ = next(iter(val_dl))

    for epoch in range(start_epoch, t_cfg['epochs']):
        t0 = time.time()
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{t_cfg["epochs"]} [train]', leave=False)
        for step, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            loss, _, _ = model(imgs)

            optimizer.zero_grad()
            loss.backward()
            if t_cfg.get('clip_grad'):
                nn.utils.clip_grad_norm_(model.parameters(), t_cfg['clip_grad'])
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), imgs.size(0))
            global_step = epoch * steps_per_epoch + step

            if step % cfg['logging']['log_every'] == 0:
                writer.add_scalar('pretrain/loss', loss.item(), global_step)
                writer.add_scalar('pretrain/lr', scheduler.get_last_lr(), global_step)
            pbar.set_postfix(loss=f'{loss_meter.avg:.4f}')

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_meter = AverageMeter()
        with torch.no_grad():
            for imgs, _ in tqdm(val_dl, desc=f'Epoch {epoch+1} [val]', leave=False):
                imgs = imgs.to(device, non_blocking=True)
                loss, _, _ = model(imgs)
                val_meter.update(loss.item(), imgs.size(0))

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d} | train_loss={loss_meter.avg:.4f} | "
              f"val_loss={val_meter.avg:.4f} | lr={scheduler.get_last_lr():.6f} | {elapsed:.0f}s")

        writer.add_scalar('pretrain/val_loss', val_meter.avg, epoch)
        csv_writer.writerow({
            'epoch': epoch + 1,
            'train_loss': round(loss_meter.avg, 5),
            'val_loss': round(val_meter.avg, 5),
            'lr': round(scheduler.get_last_lr(), 8),
            'time': round(elapsed, 1),
        })
        log_file.flush()

        # ── Save checkpoints ─────────────────────────────────────────────
        state = {
            'epoch': epoch,
            'step': scheduler.current_step,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_loss': best_loss,
            'cfg': cfg,
        }
        torch.save(state, ckpt_dir / 'latest.pth')

        if val_meter.avg < best_loss:
            best_loss = val_meter.avg
            torch.save(state, ckpt_dir / 'best.pth')
            print(f"  New best val_loss: {best_loss:.4f}")

        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            torch.save(state, ckpt_dir / f'epoch_{epoch+1:04d}.pth')

        # ── Visualize reconstruction ──────────────────────────────────────
        if (epoch + 1) % 25 == 0 or epoch == 0:
            visualize_reconstruction(
                model, vis_imgs,
                save_path=str(viz_dir / f'epoch_{epoch+1:04d}.png'),
                device=device,
            )

    log_file.close()
    writer.close()
    print(f"\nPre-training complete. Best val_loss: {best_loss:.4f}")
    print(f"Checkpoint: {ckpt_dir / 'best.pth'}")
