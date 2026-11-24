import argparse
import csv
import shutil
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from skimage.restoration import estimate_sigma
from sklearn.model_selection import train_test_split

PREPROCESSING_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PREPROCESSING_DIR.parent.parent / "Datasets" / "IQ-OTH_NCCD" / "DataSplit"
CLASSES = ["benign", "malignant", "normal"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
RANDOM_STATE = 42

IMG_SIZE = 224
NLM_H_FACTOR, NLM_MIN_H = 1.15, 3.0
NLM_TEMPLATE, NLM_SEARCH = 7, 21
ACLAHE_BASE_CLIP, ACLAHE_TARGET_STD, ACLAHE_CLIP_RANGE = 2.0, 60.0, (1.0, 4.0)
ACLAHE_TILES = (8, 8)
MAX_ROTATION, SCALE_RANGE, FLIP_PROB = 15.0, (0.9, 1.1), 0.5
BENIGN_COPIES = 3


def split_name(test_size, val_size=0.0):
    test_pct, val_pct = round(test_size * 100), round(val_size * 100)
    if val_pct:
        return f"split_{100 - val_pct - test_pct}_{val_pct}_{test_pct}"
    return f"split_{100 - test_pct}_{test_pct}"


def output_dir(test_size, val_size=0.0):
    return PREPROCESSING_DIR / "Output" / split_name(test_size, val_size)


def nlm_denoise(img):
    sigma = estimate_sigma(img)
    h = max(NLM_H_FACTOR * sigma, NLM_MIN_H)
    return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=NLM_TEMPLATE,
                                    searchWindowSize=NLM_SEARCH), h


def aclahe(img):
    clip = float(np.clip(ACLAHE_BASE_CLIP * ACLAHE_TARGET_STD / max(img.std(), 1.0), *ACLAHE_CLIP_RANGE))
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=ACLAHE_TILES).apply(img), clip


def to_square(img):
    h, w = img.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0)


def resize(img):
    return cv2.resize(to_square(img), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def enhance(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    denoised, h = nlm_denoise(img)
    enhanced, clip = aclahe(denoised)
    return resize(enhanced), h, clip


def z_score(img):
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return np.repeat(img[None], 3, axis=0)


def augment(img, rng):
    # On z-scored images the border fill of 0 is the image mean.
    channels_first = img.ndim == 3
    hw = img.transpose(1, 2, 0) if channels_first else img
    h, w = hw.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-MAX_ROTATION, MAX_ROTATION), rng.uniform(*SCALE_RANGE))
    out = cv2.warpAffine(np.ascontiguousarray(hw), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < FLIP_PROB:
        out = out[:, ::-1]
    if out.ndim == 2 and channels_first:
        out = out[..., None]
    return np.ascontiguousarray(out.transpose(2, 0, 1) if channels_first else out)


def collect_images():
    items = []
    for split in ("train", "test"):
        for label in CLASSES:
            items += [(p, label) for p in sorted((SOURCE_DIR / split / label).iterdir())
                      if p.suffix.lower() in IMAGE_EXTS]
    return items


def save_stages_figure(items, out_path):
    rng = np.random.default_rng(RANDOM_STATE)
    picks = [next(p for p, label in items if label == c) for c in CLASSES]
    titles = ["Original", "NLM denoised", "ACLAHE", "Z-score (224x224)", "Augmented"]
    fig, axes = plt.subplots(len(picks), len(titles), figsize=(3 * len(titles), 3 * len(picks)))
    for row, (path, label) in zip(axes, zip(picks, CLASSES)):
        original = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        denoised, _ = nlm_denoise(original)
        enhanced, clip = aclahe(denoised)
        normalised = z_score(resize(enhanced))
        for ax, img, title in zip(row, [original, denoised, enhanced, normalised[0], augment(normalised, rng)[0]],
                                  titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{label}: {title}" if title == "Original" else title, fontsize=9)
            ax.axis("off")
    fig.suptitle("IQ-OTH/NCCD preprocessing stages")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2, help="0.2 for the 80:20 split, 0.3 for 70:30")
    parser.add_argument("--val-size", type=float, default=0.0, help="fraction of all images held out for validation")
    args = parser.parse_args()

    items = collect_images()
    labels = [label for _, label in items]
    train_items, test_items = train_test_split(items, test_size=args.test_size, stratify=labels,
                                               random_state=RANDOM_STATE)
    splits = {"train": train_items, "test": test_items}
    if args.val_size:
        # Split the training pool again so the test set stays identical to the plain train/test split.
        train_items, val_items = train_test_split(train_items, test_size=args.val_size / (1 - args.test_size),
                                                  stratify=[label for _, label in train_items],
                                                  random_state=RANDOM_STATE)
        splits = {"train": train_items, "val": val_items, "test": test_items}
    out_dir = output_dir(args.test_size, args.val_size)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    print(f"{len(items)} de-duplicated images -> " + " / ".join(f"{len(v)} {k}" for k, v in splits.items()))

    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for split, split_items in splits.items():
        results = Parallel(n_jobs=-1)(delayed(enhance)(path) for path, _ in split_items)
        for (path, label), (img, h, clip) in zip(split_items, results):
            target = out_dir / split / label / f"{path.stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), img)
            rows.append(dict(split=split, label=label, file=target.relative_to(out_dir).as_posix(),
                             source=path.name, augmented=0, nlm_h=round(h, 2), aclahe_clip=round(clip, 2)))
            if split == "train" and label == "benign":
                for k in range(1, BENIGN_COPIES + 1):
                    copy = target.with_name(f"{path.stem}_aug{k}.png")
                    cv2.imwrite(str(copy), augment(img, rng))
                    rows.append(dict(split=split, label=label, file=copy.relative_to(out_dir).as_posix(),
                                     source=path.name, augmented=1, nlm_h=round(h, 2), aclahe_clip=round(clip, 2)))

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_stages_figure(items, out_dir / "preprocessing_stages.png")

    for split in splits:
        counts = {c: sum(r["split"] == split and r["label"] == c for r in rows) for c in CLASSES}
        print(f"{split}: {sum(counts.values())} images  " + "  ".join(f"{c}={n}" for c, n in counts.items()))
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
