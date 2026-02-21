"""Independent reconstruction: binary pS129 segmentation from four annotation labels.

NOT the original EPFL code or a claim about its exact historical loss.
Requires PyTorch; MONAI is optional until model construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class LossConfig:
    background: float = 1.0
    manual_positive: float = 3.0
    hard_negative: float = 5.0
    approximate_positive: float = 0.5
    bce_coefficient: float = 1.0
    dice_coefficient: float = 1.0
    smooth: float = 1.0


def targets_and_weights(labels: torch.Tensor, cfg: LossConfig):
    """Input [N,H,W] labels 0..3; output binary targets and per-pixel weights."""
    if labels.ndim != 3 or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('labels must be integer tensor [N,H,W]')
    if not bool(((labels >= 0) & (labels <= 3)).all()):
        raise ValueError('label IDs must be in 0..3')
    weights = torch.tensor([cfg.background, cfg.manual_positive, cfg.hard_negative,
                            cfg.approximate_positive], device=labels.device, dtype=torch.float32)
    if not bool((weights > 0).all()):
        raise ValueError('class weights must be positive')
    return ((labels == 1) | (labels == 3)).float(), weights[labels.long()]


class WeightedBinaryDiceBCELoss(nn.Module):
    """Weighted BCE plus weighted soft Dice, both normalized per image.

    This weights Dice as well as BCE. A small class-3 weight reduces the effect
    of imprecise pixel-classifier labels; it does not model boundary uncertainty.
    """
    def __init__(self, cfg: LossConfig = LossConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.smooth <= 0 or cfg.bce_coefficient < 0 or cfg.dice_coefficient < 0:
            raise ValueError('invalid loss configuration')
        if cfg.bce_coefficient + cfg.dice_coefficient == 0:
            raise ValueError('at least one loss coefficient must be positive')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape[0] != labels.shape[0] or logits.shape[2:] != labels.shape[1:]:
            raise ValueError('logits [N,1,H,W] must align with labels [N,H,W]')
        targets, weights = targets_and_weights(labels, self.cfg)
        logit = logits[:, 0]
        bce = F.binary_cross_entropy_with_logits(logit, targets, reduction='none')
        bce = ((bce * weights).sum(dim=(1,2)) / weights.sum(dim=(1,2))).mean()
        probs = torch.sigmoid(logit)
        intersection = (weights * probs * targets).sum(dim=(1,2))
        denom = (weights * (probs + targets)).sum(dim=(1,2))
        dice_loss = (1 - (2 * intersection + self.cfg.smooth) / (denom + self.cfg.smooth)).mean()
        return self.cfg.bce_coefficient * bce + self.cfg.dice_coefficient * dice_loss


def group_disjoint_split(group_ids, train_fraction=0.75, val_fraction=0.15, seed=42):
    """Split samples by animal/group ID, never by correlated image tiles.

    At least three unique groups required. Fractions apply to groups, not tiles.
    """
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError('invalid fractions')
    ids = list(group_ids)
    if any(g is None for g in ids):
        raise ValueError('missing group ID')
    unique = sorted(set(ids), key=str)
    if len(unique) < 3:
        raise ValueError('at least three distinct groups needed')
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    group_to_split = {g: ('train' if i < n_train else 'val' if i < n_train+n_val else 'test') for i,g in enumerate(unique)}
    result = {'train': [], 'val': [], 'test': []}
    for i,g in enumerate(ids):
        result[group_to_split[g]].append(i)
    return result


def build_attention_unet(in_channels=1):
    """Create MONAI AttentionUnet for binary pathology segmentation."""
    try:
        from monai.networks.nets import AttentionUnet
    except ImportError as exc:
        raise ImportError('Install optional dependency: pip install monai') from exc
    return AttentionUnet(spatial_dims=2, in_channels=in_channels, out_channels=1,
                         channels=(16, 32, 64, 128), strides=(2, 2, 2))


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running, count = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        running += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    if count == 0:
        raise ValueError('empty training loader')
    return running / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, threshold=0.5):
    """Report manual-positive Dice and hard-negative false-positive rate.

    Pixels with label 3 are excluded from Dice ground truth due to imprecise
    boundaries. All results are tile-level and should be interpreted alongside
    animal-disjoint sampling and scientific quantification checks.
    """
    model.eval()
    loss_sum, samples = 0.0, 0
    tp = fp = fn = hard_fp = hard_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += float(loss_fn(logits, labels)) * images.shape[0]
        samples += images.shape[0]
        pred = torch.sigmoid(logits[:, 0]) >= threshold
        trusted = labels != 3
        positive = labels == 1
        tp += int((pred & positive).sum())
        fp += int((pred & (~positive) & trusted).sum())
        fn += int(((~pred) & positive).sum())
        hard_fp += int((pred & (labels == 2)).sum())
        hard_total += int((labels == 2).sum())
    if samples == 0:
        raise ValueError('empty evaluation loader')
    return {'loss': loss_sum/samples,
            'manual_dice': (2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'hard_negative_fpr': hard_fp/hard_total if hard_total else None}


def save_checkpoint(path, model, optimizer, epoch, validation_metrics, group_splits):
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'epoch': int(epoch), 'validation_metrics': validation_metrics,
                'group_splits': group_splits}, path)
