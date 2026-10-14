import sys
from pathlib import Path

import numpy as np
from skimage import filters, measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, body_mask, clean_lungs, run

RESULTS_DIR = Path(__file__).resolve().parent / "Results"
SEED_SIGMA_FRACTION = 0.02
GROW_SIGMA = 1.0
TOLERANCE_FRACTION = 0.5
SEED_WINDOW = 3
MAX_ITERATIONS = 20


def body_blur(img, body, sigma):
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
    r, c = seed
    w = SEED_WINDOW
    window = (slice(max(r - w, 0), r + w + 1), slice(max(c - w, 0), c + w + 1))
    return np.median(smooth[window]), window


def grow(smooth, body, seed, body_median, tolerance_fraction=TOLERANCE_FRACTION):
    mean, window = seed_level(smooth, seed)
    tolerance = tolerance_fraction * (body_median - mean)
    region = np.zeros_like(body)
    for _ in range(MAX_ITERATIONS):
        labels = measure.label(body & (np.abs(smooth - mean) <= tolerance))
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
    body, _, _ = body_mask(img)
    if not body.any():
        empty = np.zeros_like(body)
        return empty, body, [("Input", img), ("No body found", empty)], dict(seeds=0, tolerance=0.0)
    body_median = np.median(img[body])

    seeds, blurred = find_seeds(img, body)

    smooth = filters.gaussian(img, sigma=GROW_SIGMA)
    regions, tolerances = [], []
    for seed in seeds:
        if any(region[seed] for region in regions):
            continue
        region, tolerance = grow(smooth, body, seed, body_median)
        if region.sum() >= MIN_LUNG_FRACTION * body.sum():
            regions.append(region)
            tolerances.append(tolerance)

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
    run(segment, RESULTS_DIR, "seeded region growing", "Region growing")
