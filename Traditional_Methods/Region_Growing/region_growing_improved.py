"""
Improved seeded region growing -- the textbook method in
region_growing_segmentation.py plus fixes for its two failure modes on
Chest_CT: missed lungs (bad seeds) and leakage. Results go to
Results_Improved/ so both versions can be compared.

Fix 1, seeds from dark blobs, placed iteratively. The textbook version takes
  the darkest point in each half of the body, so a seed can land in the
  trachea or a small lung fragment and a lung is missed. Here the heavily
  blurred body is split into dark blobs (below the midpoint between its
  darkest value and the body median). A seed goes at the darkest point of the
  largest blob, that seed is grown, the grown region is removed from the blob
  map, and the next seed comes from the largest dark area left. This repeats
  until no blob of lung size remains (at most MAX_SEEDS), so two lungs that
  blur into one blob with the trachea still each get a seed.

Fix 2, leak guard. A region has leaked if it covers more than
  MAX_REGION_FRACTION of the body, or if its mean brightness has drifted from
  the seed's starting level by more than MAX_DRIFT_FRACTION of the gap
  between that level and the body median (adaptive growing creeping into
  brighter tissue such as liver). A leaked region is regrown with the
  tolerance halved, up to MAX_RETRIES times, and dropped if it still leaks.

Everything else -- body mask, adaptive growing, clean-up -- is unchanged.

Usage:
    python region_growing_improved.py                 # both CT datasets
    python region_growing_improved.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

import numpy as np
from skimage import filters, measure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MIN_LUNG_FRACTION, body_mask, clean_lungs, main
from region_growing_segmentation import (GROW_SIGMA, SEED_SIGMA_FRACTION, TOLERANCE_FRACTION, body_blur, grow,
                                         seed_level)

RESULTS_DIR = Path(__file__).resolve().parent / "Results_Improved"
MAX_SEEDS = 4  # seeding stops earlier once no lung-sized dark blob is left
MAX_REGION_FRACTION = 0.6  # a lung region covering more of the body than this has leaked
MAX_DRIFT_FRACTION = 0.3  # ...as has one whose mean drifted this far towards the body median
MAX_RETRIES = 3


def dark_blobs(img, body):
    blurred = body_blur(img, body, SEED_SIGMA_FRACTION * min(img.shape))
    cut = (blurred[body].min() + np.median(img[body])) / 2
    return body & (blurred < cut), blurred


def next_seed(blobs, blurred, min_area):
    """Darkest point of the largest remaining dark blob, or None."""
    regions = measure.regionprops(measure.label(blobs))
    if not regions:
        return None
    largest = max(regions, key=lambda r: r.area)
    if largest.area < min_area:
        return None
    coords = largest.coords
    r, c = coords[np.argmin(blurred[coords[:, 0], coords[:, 1]])]
    return r, c


def grow_with_leak_guard(smooth, body, seed, body_median):
    """Grow; if the region leaks (too big, or its mean drifted too far), halve
    the tolerance and regrow. Returns (region or None, retries used)."""
    start, _ = seed_level(smooth, seed)
    size_limit = MAX_REGION_FRACTION * body.sum()
    drift_limit = MAX_DRIFT_FRACTION * (body_median - start)
    fraction = TOLERANCE_FRACTION
    for retry in range(MAX_RETRIES + 1):
        region, _ = grow(smooth, body, seed, body_median, fraction)
        if region.any() and region.sum() <= size_limit and smooth[region].mean() - start <= drift_limit:
            return region, retry
        fraction /= 2
    return None, MAX_RETRIES


def segment(img):
    # Body, shared with thresholding.
    body, _, _ = body_mask(img)
    if not body.any():
        empty = np.zeros_like(body)
        return empty, body, [("Input", img), ("No body found", empty)], dict(seeds=0, leak_retries=0, leaks_dropped=0)
    body_median = np.median(img[body])
    min_area = MIN_LUNG_FRACTION * body.sum()
    smooth = filters.gaussian(img, sigma=GROW_SIGMA)

    # Fix 1 + 2: seed the largest dark blob, grow it with the leak guard,
    # remove what was grown (or the blob, if growing failed), repeat.
    blobs, blurred = dark_blobs(img, body)
    remaining = blobs.copy()
    seeds, regions, retries, dropped = [], [], 0, 0
    for _ in range(MAX_SEEDS):
        seed = next_seed(remaining, blurred, min_area)
        if seed is None:
            break
        seeds.append(seed)
        region, used = grow_with_leak_guard(smooth, body, seed, body_median)
        retries += used
        if region is None:
            dropped += 1
            region = measure.label(remaining) == measure.label(remaining)[seed]
        elif region.sum() >= min_area:
            regions.append(region)
        remaining &= ~region
        remaining[seed] = False

    # Like the other methods, keep the two largest regions as the lungs.
    regions = sorted(regions, key=lambda region: region.sum(), reverse=True)[:2]
    lungs = clean_lungs(regions, body, img)

    seed_view = np.where(np.isfinite(blurred), blurred, 0.0)
    for r, c in seeds:
        seed_view[max(r - 4, 0):r + 5, max(c - 4, 0):c + 5] = seed_view.max()
    panels = [
        ("Input", img),
        ("1. Body (shared with thresholding)", body),
        ("2a. Dark blobs in the blurred body", blobs),
        ("2b. Seeds (placed one at a time)", seed_view),
        ("3. Grown regions (leak-guarded)", np.any(regions, axis=0) if regions else np.zeros_like(body)),
        ("4. Closed + filled lungs", lungs),
    ]
    info = dict(seeds=len(seeds), leak_retries=retries, leaks_dropped=dropped)
    return lungs, body, panels, info


if __name__ == "__main__":
    main(segment, RESULTS_DIR, "improved seeded region growing")
