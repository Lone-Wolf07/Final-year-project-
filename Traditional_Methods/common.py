import argparse
import csv
import random
import sys
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
MANUAL_REFERENCE_DATASETS = ["COVID19_CT_Seg"]
CT_DATASETS = MANUAL_REFERENCE_DATASETS + ["Chest_CT", "IQ-OTH_NCCD"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

MIN_LUNG_FRACTION = 0.01
CLOSING_RADIUS_FRACTION = 0.02
BODY_CLOSING_RADIUS_FRACTION = 0.01
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
    body_threshold = threshold_isodata(img)
    bright = img > body_threshold
    footprint = morphology.disk(disk_radius(img, BODY_CLOSING_RADIUS_FRACTION))
    body = ndi.binary_fill_holes(morphology.closing(largest_component(bright), footprint))
    return body, bright, body_threshold


def close_and_fill(mask, radius):
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
    tp = int((mask & reference).sum())
    fp = int((mask & ~reference).sum())
    fn = int((~mask & reference).sum())
    tn = int(mask.size - tp - fp - fn)
    ratio = lambda num, den: round(num / den, 4) if den else 1.0
    return dict(dice=ratio(2 * tp, 2 * tp + fp + fn), iou=ratio(tp, tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                pixel_accuracy=ratio(tp + tn, mask.size), tp=tp, fp=fp, fn=fn, tn=tn)


def print_summary(rows, title, dataset):
    print()
    print("=" * 104)
    print(f"SUMMARY: {title} on {dataset} (vs. reference masks)")
    print("=" * 104)
    print(f"{'Split':<7}{'Images':>7}{'Dice mean':>11}{'median':>9}{'std':>8}{'<0.8':>6}{'IoU':>9}"
          f"{'PixAcc':>9}{'Precision':>11}{'Recall':>9}{'s/img':>8}")
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        d = np.array([r["dice"] for r in part])
        tp, fp, fn, tn = (sum(r[k] for r in part) for k in ("tp", "fp", "fn", "tn"))
        print(f"{split:<7}{len(part):>7}{d.mean():>11.4f}{np.median(d):>9.4f}{d.std():>8.3f}{(d < 0.8).sum():>6}"
              f"{tp / max(tp + fp + fn, 1):>9.4f}{(tp + tn) / max(tp + fp + fn + tn, 1):>9.4f}"
              f"{tp / max(tp + fp, 1):>11.4f}{tp / max(tp + fn, 1):>9.4f}"
              f"{np.mean([r['seconds'] for r in part]):>8.2f}")
    print("=" * 104)


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


def find_reference(path):
    parts = path.resolve().parts
    if "DataSplit" not in parts:
        return None
    i = parts.index("DataSplit")
    dataset, rel = parts[i - 1], Path(*parts[i + 1:])
    if not (DATASETS_DIR / dataset / REFERENCE_DIRNAME / rel).with_suffix(".png").exists():
        return None
    return reference_mask(dataset, rel)


def collect_images(targets):
    paths = []
    for target in targets:
        if target.is_dir():
            paths += sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(target)
    return paths


def open_windows():
    plt.switch_backend("TkAgg")


def save_side_by_side(path, img, lungs, method, out_dir, show):
    reference = find_reference(path)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.4), layout="constrained")
    left.imshow(img, cmap="gray", vmin=0, vmax=1)
    left.set_title("Input image")
    right.imshow(np.where(lungs, img, 0), cmap="gray", vmin=0, vmax=1)
    right_title = f"Segmented lungs ({lungs.mean():.1%} of image)"
    if reference is not None:
        right_title += f" - Dice {dice(lungs, reference):.4f}"
    right.set_title(right_title)
    for ax in (left, right):
        ax.axis("off")
    fig.suptitle(f"{method}: {path.name}")

    out_path = out_dir / f"{path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    print(f"{path.name}: {right_title} -> saved {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def view_images(segment, targets, show, method, out_dir):
    if show:
        open_windows()
    for path in collect_images(targets):
        img = load_gray(path)
        lungs = segment(img)[0]
        save_side_by_side(path, img, lungs, method, out_dir, show)


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
    print_summary(rows, title, name)
    print(f"Saved masks, summary.csv, overlays.png and steps.png to {out_dir}")


def main(segment, results_dir, title):
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*",
                        help=f"subset of {', '.join(CT_DATASETS)} (default: {', '.join(MANUAL_REFERENCE_DATASETS)})")
    args = parser.parse_args()
    names = args.datasets or MANUAL_REFERENCE_DATASETS
    unknown = set(names) - set(CT_DATASETS)
    if unknown:
        parser.error(f"unknown or non-CT dataset: {', '.join(sorted(unknown))}")
    for name in names:
        run_dataset(name, segment, results_dir, title)


def run(segment, results_dir, title, method):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--view", nargs="+", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args, rest = parser.parse_known_args()
    if args.view:
        view_images(segment, args.view, not args.no_show, method, results_dir / "side_by_side")
    else:
        sys.argv = [sys.argv[0]] + rest
        main(segment, results_dir, title)
