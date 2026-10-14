import sys
from pathlib import Path

import numpy as np
from skimage import measure
from skimage.filters import threshold_isodata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, body_mask, clean_lungs, run

RESULTS_DIR = Path(__file__).resolve().parent / "Results"

def segment(img):
    body, bright, body_threshold = body_mask(img)

    lung_threshold = threshold_isodata(img[body]) if body.any() else body_threshold
    dark = body & (img < lung_threshold)

    labels = measure.label(dark)
    min_area = MIN_LUNG_FRACTION * body.sum()
    regions = sorted(
        (r for r in measure.regionprops(labels) if r.area >= min_area),
        key=lambda r: r.area,
        reverse=True,
    )[:2]
    candidates = [labels == r.label for r in regions]

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
    run(segment, RESULTS_DIR, "isodata threshold segmentation", "Thresholding")
