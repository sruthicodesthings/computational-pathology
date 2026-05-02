# Version 4: Reconstructed computational modules

**Status:** independently written portfolio code, not original EPFL laboratory source.

This version adds:

- `src/inference.py`: overlapping tile generation and probability-map averaging. It accepts a caller-supplied prediction function, but does not load VSI or connect to a GPU.
- `src/annotations.py`: minimal 4-connected-component extraction and **bounding-box-only** GeoJSON output. Bounding boxes deliberately do not represent the historical polygon-preserving QuPath workflow. Do not use them for area/shape quantification.
- `src/registration.py`: reference-anchored translation and atlas-bound checks. The historical root-based correction is not completely specified in available documents; translation alone does **not** reproduce the historical transformation or ABBA registration.
- `src/batch_processing.py`: local staging, callback-based processing, nonempty-output verification, optional safe cleanup. No OMERO, RunAI, Docker registry, SSH or institutional credentials are included.

## Historical constraints reflected

Data acquisition and annotation were ongoing while the model was trained; the storage server and compute environment were separate; per-brain transfers were required. The code provides inspectable engineering primitives rather than a claim of end-to-end reproduction.

## Missing dependencies / non-goals

Original microscopy images, original model weights, QuPath projects, lab-specific scripts, GPU scheduling logs and confirmed performance benchmarks are unavailable. The historical report and guide contain some incompatible descriptions of dataset filtering and evaluation; they are not silently reconciled here. No biological validity is claimed for these standalone utilities.

## Verification

Run `python -m unittest discover -s tests -v` with NumPy and PyTorch installed (existing Version 3 tests use PyTorch). No image dataset is necessary. Tests use tiny arrays strictly as software fixtures, not portfolio demonstrations.

## Disclosure

Before sharing, review all case-study claims, unpublished results, and ownership of original work with applicable institutional rules. The new code contains no EPFL server paths or access details.
