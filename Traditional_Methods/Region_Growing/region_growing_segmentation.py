"""
Seeded region growing for lung segmentation. Start from a point known to be
inside each lung and repeatedly absorb neighbouring pixels whose brightness is
close to the region's current mean, until nothing more can join. Run on the
2D CT slices of Chest_CT and IQ-OTH_NCCD (LC25000_Lung is histopathology, not
CT, so it is skipped).

Steps per slice:
  1. Body       - same body mask as thresholding (common.body_mask), so the
                  two methods differ only in how the lungs are found.
  2. Seeds      - blur heavily *inside the body only* (normalised convolution,
                  so the black outside air doesn't darken the body edge). Small
                  dark things -- vessels' gaps, the trachea -- wash out; big
                  dark areas -- the lungs -- stay dark. The darkest point in
                  each half of the body (left / right of its centre) is a seed,
                  kept only if it is darker than the body's median.
  3. Grow       - on a lightly smoothed image, a pixel joins the region if it
                  is connected to it and within TOLERANCE of the region's mean.
                  The mean is updated and the growth repeated until the region
                  stops changing (adaptive region growing). TOLERANCE is
                  TOLERANCE_FRACTION of the gap between the seed's brightness
                  and the body median, so it adapts to each image's contrast.
  4. Clean-up   - regions smaller than MIN_LUNG_FRACTION of the body are
                  dropped, then the same rolling-ball closing + hole filling
                  as the other methods.

Outputs go to Results/<dataset>/, scored against the lungmask reference masks
-- see common.py.

Usage:
    python region_growing_segmentation.py                 # both CT datasets
    python region_growing_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

import numpy as np
from skimage import filters, measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, body_mask, clean_lungs, main

RESULTS_DIR = Path(__file__).resolve().parent / "Results"
SEED_SIGMA_FRACTION = 0.02  # heavy blur for seed finding (~10 px at 512)
GROW_SIGMA = 1.0  # light blur for growing, so single noisy pixels don't stop it
TOLERANCE_FRACTION = 0.5  # join if within this fraction of (body median - seed level) of the mean
SEED_WINDOW = 3  # half-width of the window around a seed used for its starting level
MAX_ITERATIONS = 20


def body_blur(img, body, sigma):
    """Gaussian blur that only averages over body pixels."""
    weights = filters.gaussian(body.astype(float), sigma=sigma)
    blurred = filters.gaussian(img * body, sigma=sigma)
    return np.where(body, blurred / np.maximum(weights, 1e-6), np.inf)


def find_seeds(img, body):
    blurred = body_blur(img, body, SEED_SIGMA_FRACTION * min(img.shape))
    body_cols = np.nonzero(body.any(axis=0))[0]
    middle = (body_cols.min() + body_cols.max()) // 2
    body_median = np.median(img[body])
    seeds = []
    for c0, c1 in ((0, middle), (middle, img.shape[1])):
        half = blurred[:, c0:c1]
        r, c = np.unravel_index(np.argmin(half), half.shape)
        if half[r, c] < body_median:
            seeds.append((r, c + c0))
    return seeds, blurred


def seed_level(smooth, seed):
    """Starting brightness: median of a small window, in case the seed pixel
    itself sits on a vessel."""
    r, c = seed
    w = SEED_WINDOW
    window = (slice(max(r - w, 0), r + w + 1), slice(max(c - w, 0), c + w + 1))
    return np.median(smooth[window]), window


def grow(smooth, body, seed, body_median, tolerance_fraction=TOLERANCE_FRACTION):
    """Adaptive region growing from one seed; returns the grown region."""
    mean, window = seed_level(smooth, seed)
    tolerance = tolerance_fraction * (body_median - mean)
    region = np.zeros_like(body)
    for _ in range(MAX_ITERATIONS):
        labels = measure.label(body & (np.abs(smooth - mean) <= tolerance))
        # The component that covers most of the seed window (the seed pixel
        # itself might sit on a small vessel).
        hits = np.bincount(labels[window].ravel(), minlength=2)
        hits[0] = 0
        if hits.max() == 0:
            break
        grown = labels == hits.argmax()
        if np.array_equal(grown, region):
            break
        region = grown
        mean = smooth[region].mean()
    return region, tolerance


def segment(img):
    # 1. Body, shared with thresholding.
    body, _, _ = body_mask(img)
    if not body.any():
        empty = np.zeros_like(body)
        return empty, body, [("Input", img), ("No body found", empty)], dict(seeds=0, tolerance=0.0)
    body_median = np.median(img[body])

    # 2. One seed per lung: darkest point of each half of a heavily blurred body.
    seeds, blurred = find_seeds(img, body)

    # 3. Grow each seed on a lightly smoothed image. A second seed already
    # inside the first region (lungs touching) is skipped.
    smooth = filters.gaussian(img, sigma=GROW_SIGMA)
    regions, tolerances = [], []
    for seed in seeds:
        if any(region[seed] for region in regions):
            continue
        region, tolerance = grow(smooth, body, seed, body_median)
        if region.sum() >= MIN_LUNG_FRACTION * body.sum():
            regions.append(region)
            tolerances.append(tolerance)

    # 4. Rolling-ball closing + hole filling, per lung.
    lungs = clean_lungs(regions, body, img)

    seed_view = np.where(np.isfinite(blurred), blurred, 0.0)
    for r, c in seeds:
        seed_view[max(r - 4, 0):r + 5, max(c - 4, 0):c + 5] = seed_view.max()
    panels = [
        ("Input", img),
        ("1. Body (shared with thresholding)", body),
        ("2. Heavy body-only blur + seeds", seed_view),
        ("3. Grown regions", np.any(regions, axis=0) if regions else np.zeros_like(body)),
        ("4. Closed + filled lungs", lungs),
    ]
    info = dict(seeds=len(seeds), tolerance=round(float(np.mean(tolerances)), 4) if tolerances else 0.0)
    return lungs, body, panels, info


if __name__ == "__main__":
    main(segment, RESULTS_DIR, "seeded region growing")
