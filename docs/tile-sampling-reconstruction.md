# Annotation-aware tile sampling (reconstruction)

**Status:** independently written example based on retrospective interview; not a verified copy of EPFL code. The historical pixel ratio and exact filtering implementation are unverified.

## Label semantics

| ID | Meaning | Binary target |
|---|---|---|
| 0 | Dark background | 0 |
| 1 | Manually traced pathology | 1 |
| 2 | Pathology-like non-target / hard negative | 0 |
| 3 | Pixel-classifier approximate pathology | 1 |

## Sampling strategy

1. Extract overlapping image/mask tiles (extraction code is not part of this module).
2. Keep every tile containing class 1 or 3.
3. Discard class-0-only tiles.
4. Randomly sample class-2-only tiles, using a fixed seed, until the **negative-to-positive pixel ratio** reaches a configurable target or candidates are exhausted.
5. Count class 0 and class 2 as negative, and class 1 and class 3 as positive, **including pixels within mixed tiles**.

The target is a *minimum* negative-to-positive ratio in this example, not an exact enforced balance. If the positive tiles already contain more negative pixels than the target allows, all positive tiles are still retained; the algorithm cannot remove their background pixels. This is a deliberately explicit interpretation of the interview description, not a recovered historical formula. A fixed maximum ratio or probabilistic sampling to approximate a ratio would be a separate policy.

**Crucial evaluation caveat:** create animal-disjoint train/validation/test partitions **before tiling and sampling**. This module selects training tiles only; it does not create partitions or establish the validity of reported segmentation metrics. Sampling alters prevalence, so calibration and whole-slide false-positive rates must be checked on representative untouched data.

## Running tests

From the repository root: `python -m unittest discover -s tests -v`.

## Example

```python
import numpy as np
from src.tile_sampling import sample_tiles
masks = {
    'positive_a': np.array([[1, 0], [0, 2]]),
    'positive_b': np.array([[3, 0], [0, 0]]),
    'hard_negative': np.array([[2, 2], [0, 0]]),
    'dark': np.zeros((2, 2), dtype=int),
}
result = sample_tiles(masks, target_negative_to_positive=3.0, seed=42)
print(result.selected_ids, result.counts, result.target_reached)
```

Use synthetic/public data only; do not include unpublished EPFL images, identifiers, or infrastructure credentials in a recruiter-facing copy without authorization.
