# System architecture — reconstructed

## Pre-existing infrastructure

OMERO (microscopy image storage), QuPath (annotations), ABBA (atlas registration), and GPUs existed before the internship. The segmentation and integrated analysis workflows needed to be developed.

## Data flow

1. Access OMERO imagery through QuPath; review exposure, missing/damaged sections, edge artifacts and channel quality.
2. Draw target polygons and pathology-like hard negatives; export patches and masks. Earlier pixel-classifier masks provided coarse positive examples.
3. Transfer one brain's working data to GPU compute because persistent compute-side storage was insufficient for the cohort.
4. Run Docker/RunAI jobs for MONAI Attention U-Net training or tiled inference; retrieve masks/polygons and remove temporary compute data after output verification.
5. Import predictions into QuPath; manually remove residual edge-artifact detections.
6. Register sections with ABBA; export detections and atlas coordinates. Diagnose and correct inconsistent coordinate frames using root-annotation reference geometry, then validate with 3D views and biologist review.
7. Assign Allen regions, approximate nearby lipidomic reference values and putative MERFISH cell-type neighborhoods; apply distance and region constraints.
8. Estimate regional pathology volume from areas and observed AP spacing, and hand over outputs for biological interpretation.

## Failure domains

- Image quality and sampling bias: variable intensity, incomplete acquisitions, hard-negative structures.
- Resource bottlenecks: OMERO/compute transfer, temporary storage, GPU job scheduling.
- Geometry: registration frame errors, uneven slice spacing, shape fidelity in annotations.
- Visualization: excessive point/mesh density; zoom-dependent detail reduction.
- Scientific validity: imperfect detections and uncertain spatial matches propagate into downstream quantities.

**Status:** conceptual reconstruction from user account and project documents; not an independently verified network topology.
