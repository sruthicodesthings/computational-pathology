# Computational Pathology and Spatial Neuroscience

**Research engineering case study | EPFL**
**Focus:** Computer vision, scientific image processing, GPU workflows, spatial registration, and quantitative analysis

## Project overview

The research team sought to quantify how pS129-positive α-synuclein pathology was distributed across mouse-brain tissue, anatomical regions, experimental groups, and post-injection timepoints. Microscopy images, [OMERO](https://omero.readthedocs.io/), [QuPath](https://qupath.readthedocs.io/), and [ABBA](https://github.com/BIOP/ijp-imagetoatlas) were already available. The missing component was an integrated computational workflow that could convert large, heterogeneous image collections into spatially registered measurements suitable for downstream biological analysis.

Biological interpretation and the broader longitudinal study continued after the internship.

> **Repository scope:** This repository documents historical engineering work and contains independently reconstructed code, not an export of the original EPFL repository. The case study is not a claim of independent biological discovery.

| Project aspect | Scope |
|---|---|
| Scientific question | Quantification of pathology burden and spatial distribution across brain regions, groups, and timepoints |
| Data | Several terabytes of multichannel fluorescence microscopy; some images approximately 50,000 × 50,000 pixels |
| Annotation effort | Approximately 12,000 manually traced pathology polygons, alongside hard-negative and classifier-assisted labels |
| ML approach | Iterative segmentation with [PyTorch](https://pytorch.org/docs/stable/) and [MONAI](https://docs.monai.io/) Attention U-Net |
| Computing | Separate OMERO image storage and GPU compute; [Docker](https://docs.docker.com/) and [Run:ai](https://run-ai-docs.nvidia.com/) |
| Spatial methods | QuPath/ABBA, Allen CCFv3, coordinate correction, KD-tree matching, AP-based volume estimation |
| Deliverables | Code, trained weights, processed data, visualizations, and an operating guide handed over to the laboratory |

## 1. Research objective and working hypothesis

The working scientific question was whether the location and burden of detected pathology differed across anatomical regions, experimental groups, and post-injection timepoints. The computational premise was that consistent segmentation, anatomical registration, and volumetric estimation could make these comparisons practical at whole-brain scale.

| Required output | Computational requirement | Scientific caveat |
|---|---|---|
| Pathology detections | Identify irregular pS129-positive structures in fluorescence images | Detection errors can bias burden estimates |
| Anatomical location | Map detections into Allen Mouse Brain CCFv3 | Registration error changes region assignments |
| Estimated pathology volume | Integrate detected area across serial sections using actual AP positions | Sparse or uneven section sampling limits precision |
| Lipid and cell-type context | Associate locations with reference lipidomics and MERFISH datasets | Spatial proximity does not establish cellular identity or causality |
| Comparisons across groups and time | Produce consistent quantitative tables | Biological inference requires completed cohorts and appropriate statistical analysis |

## 2. Initial setup and constraints

At the start of the work, the lab had acquired microscopy data and established biological workflows, but did not have an end-to-end segmentation-to-quantification pipeline.

| Existing resource | Initial difficulty | Practical response |
|---|---|---|
| OMERO microscopy server | Large images were stored separately from GPU compute | Move data in batches rather than maintaining the entire collection on compute storage |
| GPU infrastructure | GPUs were available, but storage was insufficient for all brains | Stage one brain, run inference, retrieve results, verify outputs, and remove temporary files |
| QuPath annotations | Manual tracing was slow, and boundaries could be ambiguous | Combine expert-reviewed polygons with approximate classifier annotations |
| Microscopy acquisitions | Exposure differences, damaged tissue, bright scan boundaries, incomplete imaging | Image review, filtering/normalization, selective annotation, and manual correction where needed |
| ABBA registration | Some section positions and exported coordinates were implausible | Use slide-root references for correction and inspect reconstructed spatial locations |

**Overall processing sequence**

```text
Microscopy acquisition / OMERO
           ↓
QuPath review, polygon annotation, image preprocessing
           ↓
Overlapping tile extraction and annotation-aware sampling
           ↓
Attention U-Net training and whole-slide inference
           ↓
Masks → detections → QuPath → ABBA / Allen CCFv3
           ↓
Coordinate validation and spatial reference matching
           ↓
AP-based volume estimates and 3D visualization
           ↓
Quantitative outputs, scripts, weights, and handover guide
```

## 3. Baseline method and annotation development

The initial [QuPath](https://qupath.readthedocs.io/) pixel classifier helped locate intensity-associated candidate pathology, but its coarse boundaries produced block-like shapes. This was insufficient where object geometry and area were important downstream. The workflow therefore moved toward supervised segmentation supported by manually traced polygons.

| Mask label | Interpretation | Intended training behavior |
|---|---|---|
| `0` | Dark, ordinary background | Negative, standard weight |
| `1` | Manually outlined pathology | Positive, high-confidence supervision |
| `2` | Similar-looking structures that were not targets | Hard negative, stronger false-positive penalty |
| `3` | Pixel-classifier localization with approximate boundaries | Positive, lower-confidence supervision |

Hard negatives were identified through model false-positive review, direct image inspection, and consultation with biologists. Across successive iterations, manually verified annotations increased while reliance on classifier-generated boundaries decreased.

**Tile sampling:** Overlapping patches were extracted; tiles containing class 1 or 3 were retained. Negative candidates were randomly subsampled while considering **pixel-level** class balance, since even pathology-positive tiles contained large areas of non-pathology. The exact historical target ratio is not available. An earlier operating guide describes a stricter class-1 occupancy filter; the two versions have not been conclusively reconciled.

See [sampling reconstruction](docs/tile-sampling-reconstruction.md) and [training reconstruction](docs/training-reconstruction.md).

## 4. Training rounds and progression

The imaging cohort continued to grow during model development. Newly acquired brains sometimes introduced exposure conditions, morphology, and artifacts on which an earlier model performed poorly. Retraining therefore responded to changing data availability as well as annotation refinement.

| Stage | Main approach or change | Observed limitation / motivation |
|---|---|---|
| Initial baseline | QuPath pixel classifier | Approximate localization, blocky object boundaries |
| Early deep-learning work | Attention U-Net trained on available labels | Sparse, heterogeneous pathology and visually similar non-target structures |
| Intermediate iterations | More manual polygons, hard-negative examples, less reliance on approximate labels | False positives, missed pathology, boundary errors, variable exposure and morphology |
| Later iterations | Larger curated tile sets and continued retraining as additional brains were imaged | Generalization to new acquisitions remained a practical concern |


| Reported quantitative round | Tiles | Tile size | Dice | Precision | Recall | Reported bleed rate |
|---|---:|---:|---:|---:|---:|---:|
| 1 | ~3,500 | 256 px | 0.6497 | 0.6711 | 0.5547 | ~2% |
| 2 | ~21,400 | 512 px | 0.8036 | 0.7740 | 0.8439 | 0.22% |
| 3 | ~92,000 | 512 px | 0.85 | 0.89 | 0.97 | 0.02% |

*Historical values reported in the project document; not independently reproduced. Training data volume and other conditions changed between rounds, so the improvement cannot be attributed to a single intervention. The original tile-level evaluation also requires caution about possible within-brain correlation across splits.*

**Evaluation priority:** Dice and precision were useful diagnostics, but the scientific endpoint was accurate **quantification**. False positives, missed pathology, and exposure-dependent detection could bias comparisons across regions or timepoints even when an aggregate segmentation score looked acceptable.

## 5. Engineering problems and resolutions

| Problem encountered | Response during the project | Remaining limitation |
|---|---|---|
| Bright scanning boundaries outside tissue | Manually exclude where possible; remove residual false detections after inference | Artifact removal was not fully automatic |
| Pathology-like non-target structures | Label class 2 hard negatives and weight their mistakes more strongly | Some ambiguous structures still required expert review |
| Approximate classifier shapes | Increase manual polygon supervision and move to Attention U-Net | Annotation remained time-intensive |
| New brains differed from earlier acquisitions | Review failures, expand annotations, retrain | Distribution shift was not comprehensively quantified |
| GPU compute lacked storage for the full cohort | Sequential transfer, inference, result retrieval, cleanup; batch workflow documentation | Transfer speed and per-brain processing time remained constraints |
| Incorrect ABBA coordinate transforms | Recalculate using slide-root references; check 3D anatomy and consult biologists | Reconstructed code does not implement the complete historical correction |
| Dense 3D geometry caused laptop slowdowns | Downsampling and zoom-dependent detail during exploration with several viewers | No claim of a fully automated level-of-detail renderer |
| Nearest MERFISH cell could still be distant | Region-constrained nearest-neighbour matching and distance limits | A putative nearby cell is not a proven host cell |

## 6. Spatial analysis and quantification

The segmentation outputs were converted into spatial measurements rather than treated as the final result.

| Stage | Method | Output |
|---|---|---|
| Detection export | QuPath object geometry and centroids | Detection tables / annotations |
| Atlas alignment | [ABBA](https://github.com/BIOP/ijp-imagetoatlas) with [Allen CCFv3](https://atlas.brain-map.org/) | Anatomical coordinates and region labels |
| Coordinate quality control | Root-reference correction, 3D inspection, biologist review | Corrected or flagged spatial positions |
| Lipid context | KD-tree nearest-neighbour queries against spatial lipidomics reference data | Nearest lipizone and associated reference features |
| Cell-type context | Anatomically constrained MERFISH neighbourhood matching | Putative nearby cell-type labels, with distance limits |
| Pathology burden | Integrate detected area along actual section AP coordinates using trapezoidal estimation | Estimated regional pathology volume |
| Visual review | [napari](https://napari.org/stable/), [Blender](https://docs.blender.org/manual/en/latest/), [Plotly](https://plotly.com/python/), [brainrender](https://brainglobe.info/documentation/brainrender/) | Regional overview and inspection of individual shapes |

The biological team received computational measurements for subsequent analysis. The project does **not** claim that the internship completed all genotype/timepoint comparisons or established biological causation.

## 7. Outcomes and handover

| Delivered | Purpose |
|---|---|
| Image processing, annotation, training, inference and analysis scripts | Continue and adapt computational workflows |
| Trained model weights and run configurations | Execute further inference / training within the lab |
| Spatial detections, tables, and visualizations | Support biological quantification and review |
| Docker / Run:ai operating procedures and troubleshooting guide | Reduce dependence on the original developer |
| Pipeline documentation | Support handover to laboratory members |

The computational pipeline and outputs were handed over after the internship. The wider biological study continued within the research team.


Additional technical notes: [Architecture](docs/architecture.md) · [Engineering decisions](docs/engineering-decisions.md) · [Experiments](docs/experiments.md) · [Scientific boundaries](docs/scientific-boundaries.md) · [Code limitations](docs/v4-code-reconstruction.md).

## 8. Future work

| Area | Proposed improvement | Reason |
|---|---|---|
| Segmentation evaluation | Animal-disjoint testing and group-stratified error analysis | Estimate generalization and possible bias in downstream quantification |
| Annotation quality | Boundary-aware confidence weighting and better hard-negative sampling audits | Reduce noise from approximate labels |
| Image processing | More systematic artifact masking and intensity-shift assessment | Improve robustness to new acquisitions |
| Registration | Landmark-based quantitative error checks and fuller transformation provenance | Make coordinate correction more auditable |
| Spatial analysis | Propagate registration and detection uncertainty into volume estimates | Avoid overstating precision |
| Visualization | Region-based filtering and scalable level-of-detail geometry | Preserve shape inspection without exhausting local resources |
| Engineering | Stronger logging, manifests, resumable transfer, and output checksums | Improve reliability of storage-constrained processing |

These are **proposed improvements**, not claims about work completed at EPFL.

## Software and documentation

| Software / reference | Role in workflow | Documentation |
|---|---|---|
| OMERO | Microscopy image management | [OMERO docs](https://omero.readthedocs.io/) |
| QuPath | Annotation, image review, Groovy scripts | [QuPath docs](https://qupath.readthedocs.io/) |
| ABBA | Histological atlas registration | [ABBA project](https://github.com/BIOP/ijp-imagetoatlas) |
| Allen Mouse Brain Atlas | Anatomical reference and CCFv3 | [Allen atlas](https://atlas.brain-map.org/) |
| PyTorch | Deep learning | [PyTorch docs](https://pytorch.org/docs/stable/) |
| MONAI | Medical imaging networks and transforms | [MONAI docs](https://docs.monai.io/) |
| Docker | Reproducible compute environments | [Docker docs](https://docs.docker.com/) |
| NVIDIA Run:ai | GPU job scheduling | [Run:ai docs](https://run-ai-docs.nvidia.com/) |
| napari | Multidimensional image inspection | [napari docs](https://napari.org/stable/) |
| Blender | 3D exploration | [Blender manual](https://docs.blender.org/manual/en/latest/) |
| Plotly | Interactive visualization | [Plotly Python](https://plotly.com/python/) |
| brainrender | Anatomical 3D visualization | [brainrender docs](https://brainglobe.info/documentation/brainrender/) |
| BrainGlobe Atlas API | Programmatic atlas access | [Atlas API docs](https://brainglobe.info/documentation/brainglobe-atlasapi/) |
| SciPy | KD-trees and numerical routines | [SciPy docs](https://docs.scipy.org/doc/scipy/) |

---

*Retrospective engineering case study. Institutional data and unpublished results are not distributed; access to this repository does not imply permission to reuse original laboratory materials.*
