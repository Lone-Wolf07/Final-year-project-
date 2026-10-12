"""
Shared plumbing for the traditional lung-segmentation methods: image loading,
mask clean-up helpers, the per-dataset run loop, and the summary / figures
every method writes.

Each method script defines segment(img) -> (lungs, body, panels, info):
  lungs   bool mask of the segmented lungs
  body    bool mask of the patient's body (for the lung-fraction statistic)
  panels  [(title, array), ...] intermediate steps, drawn into steps.png
  info    dict of extra per-image values written to summary.csv
and calls main(segment, results_dir, title) from its __main__ block.

Every mask is scored against the same reference: the lungmask U-Net masks in
Datasets/<dataset>/ReferenceMasks (see Datasets/create_reference_masks.py).
score() is shared with the ML methods, so all methods are measured alike.

Outputs, under <method>/Results/<dataset>/:
  masks/<split>/<class>/<name>.png   binary lung mask (0/255) per image
  summary.csv                        per image: method values, lung regions,
                                     area, seconds, and the scores against the
                                     reference (plus raw tp/fp/fn/tn counts,
                                     so pooled scores can be recomputed)
  overlays.png                       random sample: mask (red), reference (green)
  steps.png                          every pipeline step on one image
"""

import argparse
import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color, img_as_float, io, measure, morphology
from skimage.filters import threshold_isodata

TRADITIONAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = TRADITIONAL_DIR.parent / "Datasets"
REFERENCE_DIRNAME = "ReferenceMasks"
CT_DATASETS = ["Chest_CT", "IQ-OTH_NCCD"]  # LC25000_Lung is histopathology, not CT
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

MIN_LUNG_FRACTION = 0.01  # a region must be >= 1% of the body to count as lung
CLOSING_RADIUS_FRACTION = 0.02  # rolling-ball radius, as a fraction of the shorter image side
BODY_CLOSING_RADIUS_FRACTION = 0.01  # bridges annotation lines that cut the body wall
N_OVERLAYS = 16
RANDOM_STATE = 42


def load_gray(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    return img_as_float(img)


def largest_component(mask):
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def disk_radius(img, fraction):
    return max(1, round(fraction * min(img.shape)))


def body_mask(img):
    """Isodata threshold, largest bright region, small closing (bridges
    annotation lines drawn across the body wall), holes filled."""
    body_threshold = threshold_isodata(img)
    bright = img > body_threshold
    footprint = morphology.disk(disk_radius(img, BODY_CLOSING_RADIUS_FRACTION))
    body = ndi.binary_fill_holes(morphology.closing(largest_component(bright), footprint))
    return body, bright, body_threshold


def close_and_fill(mask, radius):
    """Disk closing (the "rolling ball") then hole filling, done on the mask's
    bounding box only -- much faster than closing the whole image."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return mask
    pad = radius + 1
    r0, c0 = rows.min(), cols.min()
    crop = np.pad(mask[r0:rows.max() + 1, c0:cols.max() + 1], pad)
    crop = ndi.binary_fill_holes(ndi.binary_closing(crop, structure=morphology.disk(radius)))
    out = np.zeros_like(mask)
    r_lo, c_lo = max(r0 - pad, 0), max(c0 - pad, 0)
    r_hi, c_hi = min(rows.max() + 1 + pad, mask.shape[0]), min(cols.max() + 1 + pad, mask.shape[1])
    out[r_lo:r_hi, c_lo:c_hi] = crop[r_lo - (r0 - pad):r_hi - (r0 - pad), c_lo - (c0 - pad):c_hi - (c0 - pad)]
    return out


def clean_lungs(lung_regions, body, img):
    """Close + fill each lung on its own (so the two lungs are never bridged)."""
    radius = disk_radius(img, CLOSING_RADIUS_FRACTION)
    lungs = np.zeros(img.shape, dtype=bool)
    for region in lung_regions:
        lungs |= close_and_fill(region, radius)
    return lungs & body


def dice(a, b):
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2.0 * (a & b).sum() / total


def reference_mask(dataset, rel):
    return io.imread((DATASETS_DIR / dataset / REFERENCE_DIRNAME / rel).with_suffix(".png")) > 0


def score(mask, reference):
    """Per-image scores of a mask against the reference, plus the raw counts.
    When both masks are empty (e.g. an abdomen slice) every score is 1."""
    tp = int((mask & reference).sum())
    fp = int((mask & ~reference).sum())
    fn = int((~mask & reference).sum())
    tn = int(mask.size - tp - fp - fn)
    ratio = lambda num, den: round(num / den, 4) if den else 1.0
    return dict(dice=ratio(2 * tp, 2 * tp + fp + fn), iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                pixel_accuracy=ratio(tp + tn, mask.size), tp=tp, fp=fp, fn=fn, tn=tn)


def print_scores(rows):
    """Per split: mean / median Dice, failures, pooled pixel scores, time."""
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        d = np.array([r["dice"] for r in part])
        tp, fp, fn, tn = (sum(r[k] for r in part) for k in ("tp", "fp", "fn", "tn"))
        print(f"{split:>5} ({len(part)} images): Dice mean {d.mean():.4f} / median {np.median(d):.4f}, "
              f"< 0.8 on {(d < 0.8).sum()} | pooled IoU {tp / (tp + fp + fn):.4f}, "
              f"pixel accuracy {(tp + tn) / (tp + fp + fn + tn):.4f}, precision {tp / max(tp + fp, 1):.4f}, "
              f"recall {tp / max(tp + fn, 1):.4f} | {np.mean([r['seconds'] for r in part]):.2f}s per image")


def list_images(dataset):
    split_dir = DATASETS_DIR / dataset / "DataSplit"
    return sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS), split_dir


def save_overlays(samples, out_path, title):
    cols = 4
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (rel, img, mask, reference) in zip(axes.flat, samples):
        ax.imshow(img, cmap="gray")
        if reference.any():
            ax.contour(reference, levels=[0.5], colors="lime", linewidths=1)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="red", linewidths=1)
        ax.set_title(str(rel), fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def save_steps(panels, out_path, title):
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (name, panel) in zip(axes.flat, panels):
        ax.imshow(panel, cmap="gray")
        ax.set_title(name, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def run_dataset(name, segment, results_dir, title):
    print(f"\n{'#' * 50}\n# {name}\n{'#' * 50}")
    images, split_dir = list_images(name)
    out_dir = results_dir / name
    overlay_ids = set(random.Random(RANDOM_STATE).sample(range(len(images)), min(N_OVERLAYS, len(images))))

    t0 = time.time()
    rows, overlays = [], []
    for i, path in enumerate(images):
        rel = path.relative_to(split_dir)
        img = load_gray(path)
        t_image = time.perf_counter()
        lungs, body, panels, info = segment(img)
        seconds = time.perf_counter() - t_image
        reference = reference_mask(name, rel)

        mask_path = (out_dir / "masks" / rel).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(mask_path, (lungs * 255).astype(np.uint8), check_contrast=False)

        body_area = body.sum()
        row = dict(split=rel.parts[0], label=rel.parts[1], image=rel.name, **info)
        row["lung_regions"] = measure.label(lungs).max()
        row["lung_fraction_of_body"] = round(lungs.sum() / body_area, 4) if body_area else 0.0
        row["seconds"] = round(seconds, 4)
        row.update(score(lungs, reference))
        rows.append(row)

        if i in overlay_ids:
            overlays.append((rel, img, lungs, reference))
            if len(overlays) == 1:
                save_steps(panels, out_dir / "steps.png", f"{name}: {rel}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_overlays(overlays, out_dir / "overlays.png", f"{name}: {title} (red) vs. reference (green)")

    counts = np.bincount([min(r["lung_regions"], 3) for r in rows], minlength=4)
    fractions = np.array([r["lung_fraction_of_body"] for r in rows])
    print(f"Segmented {len(rows)} images in {time.time() - t0:.1f}s")
    print(f"Lung regions found (touching lungs count as 1): "
          f"3+ -> {counts[3]}, 2 -> {counts[2]}, 1 -> {counts[1]}, 0 -> {counts[0]}")
    print(f"Lung area as fraction of body: median {np.median(fractions):.3f}, "
          f"min {fractions.min():.3f}, max {fractions.max():.3f}")
    print_scores(rows)
    print(f"Saved masks, summary.csv, overlays.png and steps.png to {out_dir}")


def main(segment, results_dir, title):
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help=f"subset of {', '.join(CT_DATASETS)} (default: all)")
    args = parser.parse_args()
    names = args.datasets or CT_DATASETS
    unknown = set(names) - set(CT_DATASETS)
    if unknown:
        parser.error(f"unknown or non-CT dataset: {', '.join(sorted(unknown))}")
    for name in names:
        run_dataset(name, segment, results_dir, title)
