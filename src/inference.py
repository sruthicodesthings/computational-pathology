"""Reconstructed tiled inference primitives. No laboratory data or models included."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass(frozen=True)
class Tile:
    y: int
    x: int
    height: int
    width: int

def tile_grid(height: int, width: int, tile_size: int = 512, overlap: int = 64) -> list[Tile]:
    if min(height, width, tile_size) <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Invalid dimensions, tile size, or overlap")
    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        positions = list(range(0, length - tile_size + 1, tile_size - overlap))
        if positions[-1] != length - tile_size:
            positions.append(length - tile_size)
        return positions
    return [Tile(y, x, min(tile_size, height-y), min(tile_size, width-x))
            for y in starts(height) for x in starts(width)]

def predict_tiled(image: np.ndarray, predictor: Callable[[np.ndarray], np.ndarray],
                  tile_size: int = 512, overlap: int = 64) -> np.ndarray:
    """Average overlapping probability patches. Input HxW or HxWxC; predictor returns HxW."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("Expected HxW or HxWxC image")
    h, w = image.shape[:2]
    total = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.uint32)
    for tile in tile_grid(h, w, tile_size, overlap):
        patch = image[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
        probs = np.asarray(predictor(patch), dtype=np.float64)
        if probs.shape != (tile.height, tile.width):
            raise ValueError("Predictor must return a 2D map matching patch dimensions")
        if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError("Predictor must return finite probabilities in [0,1]")
        region = np.s_[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
        total[region] += probs
        count[region] += 1
    if not np.all(count):
        raise RuntimeError("Tiling left uncovered pixels")
    return (total/count).astype(np.float32)
