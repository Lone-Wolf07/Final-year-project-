"""
Edge-based lung segmentation -- the boundary-driven alternative to
thresholding. Instead of asking "is this pixel dark?", it asks "where does the
brightness change sharply?", and takes the lungs to be the dark regions
enclosed by those boundaries. Run on the 2D CT slices of Chest_CT and
IQ-OTH_NCCD (LC25000_Lung is histopathology, not CT, so it is skipped).

Steps per slice:
  1. Smooth     - Gaussian blur, so noise and fine texture don't create edges.
  2. Gradient   - Sobel gradient magnitude: how sharply brightness changes.
  3. Edges      - Canny: thin the gradient to 1-pixel ridges (non-maximum
                  suppression) and keep them by hysteresis -- strong edges
                  (top 1 - HIGH_QUANTILE of gradients) plus any weaker edge
                  (above LOW_QUANTILE) connected to a strong one.
  4. Link       - dilate the edges slightly to close small gaps, so lung
                  boundaries form closed curves (edge linking).
  5. Regions    - label the edge-free areas. Dark areas touching the border
                  are outside air; everything else, filled, is the body.
                  (Bright border areas are body cut off by the frame.)
  6. Lung pick  - inside the body, keep regions at least MIN_LUNG_FRACTION of
                  the body whose mean brightness is below the isodata cut of
                  the body pixels (lungs are the dark enclosed areas), the two
                  largest. Edges set the boundaries; this per-region statistic
                  only picks which enclosed regions are lung.
  7. Clean-up   - grow each region back by the linking radius (it was shrunk
                  by step 4), then the same rolling-ball closing + hole
                  filling as thresholding, so the two methods compare fairly.

Outputs go to Results/<dataset>/, scored against the lungmask reference masks
-- see common.py.

Usage:
    python edge_segmentation.py                 # both CT datasets
    python edge_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology
from skimage.filters import threshold_isodata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, clean_lungs, disk_radius, largest_component, main

RESULTS_DIR = Path(__file__).resolve().parent / "Results"
SIGMA_FRACTION = 0.004  # Gaussian sigma, as a fraction of the shorter image side (~2 px at 512)
LOW_QUANTILE = 0.80  # Canny hysteresis thresholds, as quantiles of the gradient magnitude
HIGH_QUANTILE = 0.90
LINK_RADIUS_FRACTION = 0.008  # edge dilation that closes small gaps in boundaries (~4 px at 512)


def segment(img):
    sigma = max(1.0, SIGMA_FRACTION * min(img.shape))

    # 1-2. Smooth, then Sobel gradient magnitude (shown for reference; Canny
    # computes its own gradient internally with the same sigma).
    smoothed = filters.gaussian(img, sigma=sigma)
    gradient = filters.sobel(smoothed)

    # 3. Canny edges with hysteresis thresholds set from gradient quantiles.
    edges = feature.canny(img, sigma=sigma, low_threshold=LOW_QUANTILE,
                          high_threshold=HIGH_QUANTILE, use_quantiles=True)

    # 4. Edge linking: thicken edges so small gaps in a boundary close up.
    link_radius = disk_radius(img, LINK_RADIUS_FRACTION)
    linked = ndi.binary_dilation(edges, structure=morphology.disk(link_radius))

    # 5. Edge-free regions. Dark ones touching the border are outside air (a
    # bright one touching it is body cut off by the image edge, e.g. an arm).
    regions = measure.label(~linked, connectivity=1)
    props = measure.regionprops(regions, intensity_image=img)
    border = set(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]])) - {0}
    image_cut = threshold_isodata(img)
    outside = np.isin(regions, [r.label for r in props if r.label in border and r.intensity_mean < image_cut])
    body = ndi.binary_fill_holes(largest_component(~outside))

    # 6. Enclosed regions inside the body that are dark on average and big
    # enough to be lungs. Edges decide where each boundary is; this one
    # brightness statistic per region only decides which regions are lung.
    body_cut = threshold_isodata(img[body]) if body.any() else image_cut
    min_area = MIN_LUNG_FRACTION * body.sum()
    lung_like = sorted(
        (r for r in props
         if r.label not in border and r.area >= min_area and r.intensity_mean < body_cut
         and body[tuple(r.coords[0])]),
        key=lambda r: r.area,
        reverse=True,
    )[:2]
    candidates = [regions == r.label for r in lung_like]

    # 7. Undo the linking shrink, then rolling-ball closing + hole filling.
    grow = morphology.disk(link_radius)
    lungs = clean_lungs([ndi.binary_dilation(c, structure=grow) for c in candidates], body, img)

    panels = [
        ("Input", img),
        (f"1-2. Sobel gradient (sigma {sigma:.1f})", gradient),
        (f"3. Canny edges (q {LOW_QUANTILE}/{HIGH_QUANTILE})", edges),
        (f"4. Linked edges (dilated {link_radius} px)", linked),
        ("5. Body (inside the outer boundary)", body),
        ("6-7. Dark enclosed regions -> lungs", lungs),
    ]
    info = dict(sigma=round(sigma, 2), edge_fraction=round(edges.mean(), 4), region_cut=round(body_cut, 4))
    return lungs, body, panels, info


if __name__ == "__main__":
    main(segment, RESULTS_DIR, "Canny edge-based segmentation")
