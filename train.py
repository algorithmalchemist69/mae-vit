"""
Entry point for MAE pre-training and fine-tuning.

Usage:
  python train.py --mode pretrain  --config configs/pretrain.yaml
  python train.py --mode finetune  --config configs/finetune.yaml
  python train.py --mode finetune  --config configs/finetune.yaml --epochs 50
"""

import argparse
import yaml


def load_config(path: str, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Apply CLI overrides to nested keys using dot notation
    for key, val in overrides.items():
        parts = key.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = val
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   required=True, choices=['pretrain', 'finetune'])
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--epochs', type=int,   help='Override training.epochs')
    parser.add_argument('--batch_size', type=int, help='Override training.batch_size')
    parser.add_argument('--mask_ratio', type=float, help='Override model.mask_ratio')
    parser.add_argument('--experiment_name', type=str, help='Override experiment name')
    args = parser.parse_args()

    overrides = {}
    if args.epochs:        overrides['training.epochs']           = args.epochs
    if args.batch_size:    overrides['training.batch_size']       = args.batch_size
    if args.mask_ratio:    overrides['model.mask_ratio']          = args.mask_ratio
    if args.experiment_name: overrides['experiment_name']         = args.experiment_name

    cfg = load_config(args.config, overrides)

    if args.mode == 'pretrain':
        from training import run_pretrain
        run_pretrain(cfg)
    else:
        from training import run_finetune
        run_finetune(cfg)


if __name__ == '__main__':
    main()
