"""Reconstructed, synthetic-data-safe tile sampling example.

This is NOT verified original EPFL code. The historical target ratio is unknown.
Class IDs: 0 dark background; 1 manually traced positive; 2 hard negative;
3 approximate pixel-classifier positive.
"""
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class SamplingResult:
    selected_ids: tuple[str, ...]
    positive_ids: tuple[str, ...]
    negative_ids: tuple[str, ...]
    counts: tuple[int, int, int, int]
    negative_to_positive_ratio: float
    target_reached: bool


def sample_tiles(masks: Mapping[str, np.ndarray], *, target_negative_to_positive: float,
                 seed: int = 42) -> SamplingResult:
    """Retain all class-1/3 tiles; randomly draw class-2-only tiles to target ratio.

    Ratio is the count of ALL negative pixels (0+2) divided by ALL positive
    pixels (1+3) across retained tiles, including background in positive tiles.
    Class-0-only tiles are excluded. Negative candidates are added until the
    target is reached or candidates run out. If positives already exceed the
    target negative ratio, they cannot be discarded to enforce it.
    """
    if not np.isfinite(target_negative_to_positive) or target_negative_to_positive < 0:
        raise ValueError('target_negative_to_positive must be finite and >= 0')
    if not masks:
        raise ValueError('At least one mask is required')
    positives, negatives, histograms = [], [], {}
    for tile_id, mask in masks.items():
        a = np.asarray(mask)
        if a.ndim != 2 or a.size == 0 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f'{tile_id}: mask must be a nonempty 2D integer array')
        if np.any((a < 0) | (a > 3)):
            raise ValueError(f'{tile_id}: class IDs must be 0, 1, 2, 3')
        histograms[tile_id] = np.bincount(a.ravel(), minlength=4)
        if np.any((a == 1) | (a == 3)):
            positives.append(tile_id)
        elif np.any(a == 2):
            negatives.append(tile_id)
    if not positives:
        raise ValueError('No positive tiles (class 1 or 3) found')
    counts = sum((histograms[k] for k in positives), np.zeros(4, dtype=np.int64))
    chosen = []
    rng = np.random.default_rng(seed)
    for idx in rng.permutation(len(negatives)):
        pos_pixels = int(counts[1] + counts[3])
        neg_pixels = int(counts[0] + counts[2])
        if neg_pixels / pos_pixels >= target_negative_to_positive:
            break
        key = negatives[int(idx)]
        counts += histograms[key]
        chosen.append(key)
    ratio = float((counts[0] + counts[2]) / (counts[1] + counts[3]))
    return SamplingResult(tuple(positives + chosen), tuple(positives), tuple(chosen),
                          tuple(int(x) for x in counts), ratio,
                          ratio >= target_negative_to_positive)
