import torch


class AverageMeter:
    """Running mean + sum tracker — used for loss and accuracy logging."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy."""
    pred = output.argmax(dim=1)
    return (pred == target).float().mean().item() * 100.0


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy."""
    _, pred = output.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.unsqueeze(1).expand_as(pred))
    return correct.any(dim=1).float().mean().item() * 100.0
