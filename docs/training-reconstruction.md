# Version 3 — Segmentation training reconstruction

**Status:** independently written demonstration; not original EPFL source code.

## Historical record vs new implementation

The report describes MONAI Attention U-Net, AdamW, cosine annealing, and Dice + BCE; the user confirms Dice and BCE were combined. The original code and guide describe four label IDs. The user clarified that the *scientific prediction objective* was binary: detect pathology, not four distinct object types. Therefore `src/training.py` implements a **new binary interpretation** with one logit per pixel; do not call it an exact reproduction of the historical four-output network.

| Mask label | Training target | Example weight (new, not historical) |
|---|---:|---:|
| 0 dark background | 0 | 1.0 |
| 1 manually outlined pathology | 1 | 3.0 |
| 2 pathology-like hard negative | 0 | 5.0 |
| 3 pixel-classifier approximate positive | 1 | 0.5 |

Both weighted BCE and soft Dice use per-pixel weights. Class-3 boundaries are not intrinsically uncertain in the formula; lower weight is only a crude proxy. The default weights and coefficients are illustrative and require tuning on genuinely independent animals.

## Evaluation and limitations

`group_disjoint_split` assigns all tiles from each animal to exactly one split. Supply **animal IDs**, not tile or slide IDs. The split is reproducible but not stratified; check that each split represents relevant acquisition batches, genotype and timepoint where possible. `evaluate` excludes weak class-3 pixels from Dice/precision/recall ground truth, and reports false-positive rate on class-2 pixels separately. It does not establish unbiased estimates of pathology burden or cell identity. In particular, annotation selection toward biologically expected regions can bias downstream regional comparisons.

The model builder lazily imports MONAI. To run it, install optional dependencies from `requirements-ml.txt`. No original microscopy images, model weights, infrastructure details or unpublished results are included.

## Usage

```python
from src.training import build_attention_unet, WeightedBinaryDiceBCELoss, LossConfig
import torch
model = build_attention_unet(in_channels=1)
loss_fn = WeightedBinaryDiceBCELoss(LossConfig())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

The module also supplies `train_one_epoch`, `evaluate`, and `save_checkpoint`. An actual research training run additionally requires a permitted image dataset, tile-to-animal metadata, validation strategy, data transforms, logging, and run configuration. This module is **not** a full end-to-end CLI and no trained model or empirical performance is claimed.
