"""
Reference lung masks for the CT datasets, made with lungmask -- the published
U-Net of Hofmanninger et al. (Eur Radiol Exp 2020), model R231, trained on
hand-labelled CT scans covering a wide range of lung diseases. There are no
hand-traced masks for these datasets, so this independent model is the one
reference every segmentation method (traditional and ML) is scored against,
and the ML methods learn from its train-split masks.

lungmask expects Hounsfield units, but these images are already-windowed 8-bit
PNG/JPGs. Each image is mapped back to an approximate HU scale from its own
levels: the darkest 1% (outside air) -> -1000 HU and the median of the
bright, above-isodata pixels (soft tissue) -> +40 HU. This works whatever
display window the image was saved with: under a lung window the lungs land
around -700 HU, under a soft-tissue window they are already black (-1000).

Outputs, per dataset:
  <dataset>/ReferenceMasks/<split>/<class>/<name>.png   binary mask (0/255)
  <dataset>/reference_overlays.png                     random sample, outlined

Usage:
    python create_reference_masks.py                 # both CT datasets
    python create_reference_masks.py IQ-OTH_NCCD     # one dataset
"""

import logging
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, img_as_float, io
from skimage.filters import threshold_isodata

DATASETS_DIR = Path(__file__).resolve().parent
CT_DATASETS = ["Chest_CT", "IQ-OTH_NCCD"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MODEL_NAME = "R231"
AIR_HU, TISSUE_HU = -1000.0, 40.0
N_OVERLAYS = 16
RANDOM_STATE = 42


def load_gray(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])
    return img_as_float(img)


def to_hu(img):
    """Approximate Hounsfield units from a windowed image's own air and tissue levels."""
    air = np.percentile(img, 1)
    bright = img[img > threshold_isodata(img)]
    tissue = np.median(bright) if bright.size else img.max()
    scale = (TISSUE_HU - AIR_HU) / max(tissue - air, 1e-3)
    return (AIR_HU + (img - air) * scale).astype(np.float32)


def run_dataset(name, inferer):
    split_dir = DATASETS_DIR / name / "DataSplit"
    out_dir = DATASETS_DIR / name / "ReferenceMasks"
    images = sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    overlay_ids = set(random.Random(RANDOM_STATE).sample(range(len(images)), min(N_OVERLAYS, len(images))))
    print(f"=== {name}: {len(images)} images ===", flush=True)

    t0 = time.time()
    overlays, empty = [], 0
    for i, path in enumerate(images):
        rel = path.relative_to(split_dir)
        img = load_gray(path)
        mask = inferer.apply(to_hu(img)[None])[0] > 0  # labels: 1 = right lung, 2 = left lung
        empty += not mask.any()
        mask_path = (out_dir / rel).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(mask_path, (mask * 255).astype(np.uint8), check_contrast=False)
        if i in overlay_ids:
            overlays.append((rel, img, mask))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(images)} ({(time.time() - t0) / (i + 1):.2f}s per image)", flush=True)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (rel, img, mask) in zip(axes.flat, overlays):
        ax.imshow(img, cmap="gray")
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="lime", linewidths=1)
        ax.set_title(str(rel), fontsize=7)
    fig.suptitle(f"{name}: lungmask {MODEL_NAME} reference masks")
    fig.tight_layout()
    fig.savefig(DATASETS_DIR / name / "reference_overlays.png", dpi=80)
    plt.close(fig)
    print(f"  done in {time.time() - t0:.1f}s; {empty} images with no lung; masks in {out_dir}\n", flush=True)


def main():
    names = sys.argv[1:] or CT_DATASETS
    warnings.filterwarnings("ignore")
    logging.getLogger("lungmask").setLevel(logging.WARNING)
    from lungmask import LMInferer
    inferer = LMInferer(modelname=MODEL_NAME, tqdm_disable=True)
    for name in names:
        run_dataset(name, inferer)


if __name__ == "__main__":
    main()
