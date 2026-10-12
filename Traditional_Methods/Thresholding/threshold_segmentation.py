"""
Classic threshold-based lung segmentation -- the 1990s pipeline (after Hu,
Hoffman & Reinhardt, IEEE TMI 2001) -- run on the 2D CT slices of Chest_CT
and IQ-OTH_NCCD. LC25000_Lung is histopathology, not CT, so it is skipped.

The images are already-windowed PNG/JPGs, so there are no Hounsfield units.
Instead of a fixed HU cut-off, every threshold is picked per image with the
iterative isodata (Ridler-Calvard) method used by those early papers.

Steps per slice:
  1. Body mask  - isodata threshold over the whole image, keep the largest
                  bright region, close it slightly (bridges annotation lines
                  drawn across the body wall), fill its holes. This drops the
                  air outside the patient and the scanner table.
  2. Dark air   - isodata threshold recomputed on body pixels only, so the
                  lung / soft-tissue split adapts to each image's contrast.
  3. Lung pick  - drop dark regions smaller than MIN_LUNG_FRACTION of the
                  body (trachea, oesophagus, noise) and keep the two largest.
  4. Clean-up   - per lung, a disk closing (the "rolling ball") puts back
                  vessels and nodules touching the lung wall, then holes are
                  filled.

Outputs go to Results/<dataset>/ -- see common.py.

Usage:
    python threshold_segmentation.py                 # both CT datasets
    python threshold_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

import numpy as np
from skimage import measure
from skimage.filters import threshold_isodata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, body_mask, clean_lungs, main

RESULTS_DIR = Path(__file__).resolve().parent / "Results"


def segment(img):
    # 1. Body: bright pixels, largest region, closed, holes (lungs, airways) filled in.
    body, bright, body_threshold = body_mask(img)

    # 2. Dark air inside the body, with a threshold fitted to body pixels only.
    lung_threshold = threshold_isodata(img[body]) if body.any() else body_threshold
    dark = body & (img < lung_threshold)

    # 3. Keep the two largest dark regions that are big enough to be lungs.
    labels = measure.label(dark)
    min_area = MIN_LUNG_FRACTION * body.sum()
    regions = sorted(
        (r for r in measure.regionprops(labels) if r.area >= min_area),
        key=lambda r: r.area,
        reverse=True,
    )[:2]
    candidates = [labels == r.label for r in regions]

    # 4. Rolling-ball closing + hole filling, per lung.
    lungs = clean_lungs(candidates, body, img)

    panels = [
        ("Input", img),
        (f"1a. Bright > {body_threshold:.2f}", bright),
        ("1b. Body (largest, closed, filled)", body),
        (f"2. Dark inside body < {lung_threshold:.2f}", dark),
        ("3. Two largest dark regions", np.any(candidates, axis=0) if candidates else np.zeros_like(dark)),
        ("4. Closed + filled lungs", lungs),
    ]
    info = dict(body_threshold=round(body_threshold, 4), lung_threshold=round(lung_threshold, 4))
    return lungs, body, panels, info


if __name__ == "__main__":
    main(segment, RESULTS_DIR, "isodata threshold segmentation")
