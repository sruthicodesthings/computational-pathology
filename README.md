# Large-Scale Computational Pathology & Spatial Neuroscience

**Research engineering case study | EPFL | 6–7 months**

> **Sharing status:** PRIVATE DRAFT. This reconstruction contains no original laboratory data, credentials, or institutional code. Do not deploy without disclosure review.

## The scientific problem

Enable a biology research team to quantify the spatial burden of pS129 alpha-synuclein pathology in mouse brains, across anatomical regions and experimental timepoints. The role was to build computational infrastructure and hand over measurements; downstream biological interpretation remained with the laboratory.

## My contribution

Developed the computational pipeline in a biology-focused lab with existing QuPath, ABBA, OMERO imaging access and separate GPU infrastructure. The work included manual polygon annotation (~12,000), image preprocessing, iterative MONAI Attention U-Net training, storage-constrained GPU batch inference, atlas-coordinate export/correction, spatial reference matching, volumetric estimates, visualization, and operational handover.

## Constraints that shaped the system

- Several terabytes of microscopy data; some images approximately 50,000 × 50,000 pixels.
- Inconsistent exposure, damaged tissue, bright slide-edge artifacts, uneven sections and ongoing image acquisition.
- OMERO-hosted imaging data separated from GPU compute; temporary per-brain upload, processing, download, verification and cleanup.
- Manual precise polygons, hard-negative lookalikes, and coarser pixel-classifier-assisted positives.
- Atlas coordinate failures requiring root-annotation-based correction and anatomical review.
- Large 3D datasets that required reduced detail at overview zoom levels.

## Logical pipeline

OMERO / QuPath → manual labels & tile export → preprocessing → GPU training / inference → prediction polygons → QuPath / ABBA → coordinate QC → Allen CCFv3 + spatial reference matches → AP-based volume estimates → handover.

## Four annotation labels

| Label | Meaning | Caution |
|---|---|---|
| 0 | Dark background | Not equivalent to all possible negatives |
| 1 | Manually traced target pathology | High-confidence positive |
| 2 | Pathology-like non-target structures | Explicit hard negative, despite historical name `Ignore*` |
| 3 | Pixel-classifier positive location, imprecise boundary | Weak/approximate positive |

The fraction of precise manual annotations increased across retraining as new brains became available. Four development rounds were recalled; the original academic report summarizes three numerical rounds. Their numbering is not assumed equivalent.

## Reported results and limits

The academic report lists three numerical rounds with Dice scores of 0.6497, 0.8036 and 0.85. These are **reported historical figures, not independently validated here**. Tile-level splits can overestimate unseen-brain generalization; annotation uncertainty, selection of biologically likely regions, exposure differences, and manual removal of false positives can bias downstream quantification. Do not interpret spatial proximity to MERFISH reference cells as confirmed cell identity.

## Engineering deliverables

Source scripts, model weights, quantitative outputs, visualizations, and operational documentation were handed to the lab. This repository is an independently written demonstration of selected mathematical and engineering methods, **not a copy of the EPFL production system**.

## Explore

- [Architecture](docs/architecture.md)
- [Engineering challenges and decisions](docs/engineering-decisions.md)
- [Experiments and validation limitations](docs/experiments.md)
- [Scientific interpretation boundaries](docs/scientific-boundaries.md)
- [Sharing checklist](docs/sharing-checklist.md)
- [Reconstructed source modules](src/)

## Local verification

```bash
python -m unittest discover -s tests -v
```

Tests use small numerical fixtures, not microscopy images or scientific performance demonstrations. Python, NumPy and PyTorch are needed for the complete test suite.

### New: annotation-aware tile sampling reconstruction

See [`docs/tile-sampling-reconstruction.md`](docs/tile-sampling-reconstruction.md) and [`src/tile_sampling.py`](src/tile_sampling.py). This is newly written demonstrator code, **not the original lab implementation**; the historical pixel ratio is not known.

### New: Version 3 — binary segmentation training reconstruction

See [training reconstruction and methodological cautions](docs/training-reconstruction.md) and [`src/training.py`](src/training.py). This module includes weighted BCE + soft Dice, animal-disjoint splitting, training/evaluation functions, and checkpoint serialization. Defaults are **illustrative**, not verified EPFL hyperparameters. To run training tests, install PyTorch (`requirements-ml.txt`; MONAI needed only to instantiate Attention U-Net). No model was trained on EPFL data in this reconstruction.

### New: Version 4 — inference, registration and batch-processing code

See [`docs/v4-code-reconstruction.md`](docs/v4-code-reconstruction.md). Includes tiled probability inference, conservative GeoJSON component boxes, root-anchored coordinate translation, and safe per-brain local staging with verified outputs before cleanup. These are **independent reconstructions of engineering concepts**, not the original EPFL scripts. In particular, bounding boxes are not scientifically valid substitutes for pathology polygons and translation alone is not a substitute for full atlas registration.
