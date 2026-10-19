# Engineering decisions and failure log

| Problem | Response | Residual limitation |
|---|---|---|
| Pixel classifier captured intensity but produced blocky shapes | Manually trace target polygons and use Attention U-Net segmentation | Shape quality and annotation effort remain constraints |
| Pathology-like non-target structures | Label explicit hard negatives (`Ignore*`) | Historical label name may be confused with ignored loss pixels |
| Coarse automated positives | Keep pixel-classifier-assisted label distinct from precise manual positive | Approximate boundaries require separate handling |
| New brains differed from earlier acquisitions | Expand manual labels and retrain iteratively | Cohort/domain-shift generalization not established by tile split |
| Bright scan margins caused false positives | Exclude regions manually where possible and delete residual detections after inference | Manual correction affects reproducibility and burden estimates |
| GPU environment lacked dataset storage | Upload/compute/download/cleanup one brain at a time | Transfer throughput and job failures require recovery logic |
| ABBA exports occasionally mapped slices incorrectly | Root-reference coordinate correction and 3D/anatomical verification | Exact original correction parameters reconstructed |
| Full-resolution 3D scenes froze laptop | Downsample/adjust visible detail with zoom; test napari, Blender and other viewers | An automated level-of-detail engine |
| Nearby MERFISH cells were sometimes too distant to interpret | Region-constrained matching and radius checks | Nearest reference cell is not necessarily the cell of origin |
