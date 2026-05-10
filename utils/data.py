"""
Data loading for CIFAR-100.
Pre-training uses lighter augmentation (random crop + flip).
Fine-tuning uses RandAugment + RandomErasing + Mixup/CutMix.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment, RandomErasing


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def build_pretrain_transforms(image_size: int, color_jitter: float = 0.4):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(color_jitter, color_jitter, color_jitter),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_finetune_train_transforms(image_size: int):
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        RandomErasing(p=0.25),
    ])


def build_eval_transforms(image_size: int):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_dataset(split: str, mode: str = 'pretrain', image_size: int = 32, data_root: str = './data'):
    """
    Args:
        split:  'train' or 'val'
        mode:   'pretrain' or 'finetune'
        image_size: spatial resolution
    """
    is_train = split == 'train'
    if mode == 'pretrain':
        transform = build_pretrain_transforms(image_size) if is_train else build_eval_transforms(image_size)
    else:
        transform = build_finetune_train_transforms(image_size) if is_train else build_eval_transforms(image_size)

    return datasets.CIFAR100(root=data_root, train=is_train, transform=transform, download=True)


def build_dataloader(dataset, batch_size: int, num_workers: int = 4, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
