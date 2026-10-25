# Experiments and validation

## What is known

- An initial QuPath pixel classifier found intensity-associated targets but produced coarse/blocky geometry.
- Four development rounds were recalled, with progressively more manually traced polygons and fewer classifier-assisted labels.
- New brain acquisitions prompted retraining because previous models sometimes failed on new images: false positives, false negatives, inaccurate boundaries, intensity shifts and morphology variation.
- The academic report lists three *numerical* training rounds: 3,500 tiles at 256px; 21,400 tiles at 512px; 92,000 tiles at 512px. Reported Dice: 0.6497 / 0.8036 / 0.85; precision: 0.6711 / 0.7740 / 0.89; recall: 0.5547 / 0.8439 / 0.97. Do not map these automatically to the four development rounds.

## Evaluation caveats

- A tile-random split may place correlated patches from the same brain into training and testing. Prefer animal-disjoint tests.
- Selection of biologically likely regions for annotation may bias prevalence estimates.
- Label 3 represents approximate positive regions; evaluate separately from manually traced ground truth.
- Quantification quality depends on false-positive and false-negative rates by brain, region, genotype, timepoint, imaging batch and morphology, not merely pooled Dice.
- Downstream manual false-positive deletion should be logged to make burden calculations auditable.
- No new results are claimed in this reconstructed portfolio.
