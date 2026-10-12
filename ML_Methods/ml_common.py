"""
Shared plumbing for the machine-learning lung-segmentation methods (SVM, LR,
RF, KNN). Segmentation is treated as pixel classification: every pixel gets a
vector of hand-crafted features and a classifier labels it lung / not lung.
This is the "feature + classifier" stage that followed the rule-based methods
in Traditional_Methods/.

Labels and scoring. There are no hand-traced masks, so the reference is the
lungmask U-Net (Datasets/create_reference_masks.py), independent of every
method here. The classifiers learn from the reference masks of the training
split only and are scored on the test split with the same common.score() as
the traditional methods, so every method is measured against one reference.

Pipeline per dataset:
  1. Every image is resized to WORK_SIZE (speed, and one feature scale for
     all images).
  2. Pixel features: multiscale_basic_features -- Gaussian-smoothed
     intensity, Sobel edge strength and two Hessian texture values at
     N_SIGMAS scales from SIGMA_MIN to SIGMA_MAX -- plus position inside the
     body (row / column relative to the body's bounding box), distance to the
     body edge, and an inside-body flag. The body mask is common.body_mask.
  3. Training pixels: PIXELS_PER_IMAGE per training image, half lung, a
     quarter non-lung inside the body (the hard negatives), a quarter
     non-lung anywhere. Split by DataSplit/train.
  4. Hyperparameters: grid search with GroupKFold over images (pixels from
     one image never sit in both a training and a validation fold), on a
     CV_PIXELS subset, then a final fit on every training pixel.
  5. Prediction: score every pixel inside the body (outside scores -1),
     upsample the score map to the original image size, threshold. Raw
     output is noisy pixel-by-pixel, so the same clean-up as the traditional
     methods follows: two largest regions inside the body, rolling-ball
     closing, hole filling.

Outputs, under <method>/:
  Models/<dataset>_<tag>_segmenter.joblib   fitted model + feature settings
  Results/<dataset>/masks/<split>/<class>/<name>.png
  Results/<dataset>/summary.csv   per image: scores vs. the reference (cleaned
                                  mask, plus raw Dice), lung regions, area, time
  Results/<dataset>/overlays.png  random test images: mask (red), reference (green)
  Results/<dataset>/steps.png     label, features, score map, raw, cleaned
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage as ndi
from skimage import io, measure, transform
from skimage.feature import multiscale_basic_features
from sklearn.model_selection import GridSearchCV, GroupKFold

ML_DIR = Path(__file__).resolve().parent
TRADITIONAL_DIR = ML_DIR.parent / "Traditional_Methods"
sys.path.insert(0, str(TRADITIONAL_DIR))
from common import (CT_DATASETS, MIN_LUNG_FRACTION, N_OVERLAYS, RANDOM_STATE, body_mask, clean_lungs, dice,
                    list_images, load_gray, print_scores, reference_mask, save_overlays, save_steps, score)


WORK_SIZE = (128, 128)
SIGMA_MIN, SIGMA_MAX, N_SIGMAS = 1, 8, 4
PIXELS_PER_IMAGE = 40
CV_PIXELS = 10000
CV_FOLDS = 3


def pixel_features(small, body):
    """(H*W, n_features) matrix of per-pixel features for a WORK_SIZE image."""
    basic = multiscale_basic_features(small, intensity=True, edges=True, texture=True,
                                      sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, num_sigma=N_SIGMAS)
    rows, cols = np.indices(small.shape, dtype=float)
    if body.any():
        body_rows, body_cols = np.nonzero(body)
        rows = (rows - body_rows.min()) / max(np.ptp(body_rows), 1)
        cols = (cols - body_cols.min()) / max(np.ptp(body_cols), 1)
        edge_distance = ndi.distance_transform_edt(body)
        edge_distance /= edge_distance.max()
    else:
        rows, cols = rows / small.shape[0], cols / small.shape[1]
        edge_distance = np.zeros(small.shape)
    extra = np.stack([rows, cols, edge_distance, body.astype(float)], axis=-1)
    return np.concatenate([basic, extra], axis=-1).reshape(-1, basic.shape[-1] + extra.shape[-1])


def prepare(img):
    small = transform.resize(img, WORK_SIZE, anti_aliasing=True)
    body, _, _ = body_mask(small)
    return small, body, pixel_features(small, body)


def shrink_label(label):
    return transform.resize(label.astype(float), WORK_SIZE, order=1, anti_aliasing=True) > 0.5


def sample_pixels(label, body, rng):
    """Flat indices: half lung, a quarter non-lung inside the body, a quarter anywhere."""
    flat_label, flat_body = label.ravel(), body.ravel()
    pools = [
        (np.flatnonzero(flat_label), PIXELS_PER_IMAGE // 2),
        (np.flatnonzero(~flat_label & flat_body), PIXELS_PER_IMAGE // 4),
        (np.flatnonzero(~flat_label), PIXELS_PER_IMAGE // 4),
    ]
    picks = [rng.choice(pool, min(n, pool.size), replace=False) for pool, n in pools if pool.size]
    return np.concatenate(picks) if picks else np.array([], dtype=int)


def build_training_set(dataset, images, split_dir):
    rng = np.random.default_rng(RANDOM_STATE)
    X, y, groups = [], [], []
    for group, path in enumerate(images):
        rel = path.relative_to(split_dir)
        img = load_gray(path)
        small, body, features = prepare(img)
        label = shrink_label(reference_mask(dataset, rel))
        idx = sample_pixels(label, body, rng)
        X.append(features[idx])
        y.append(label.ravel()[idx])
        groups.append(np.full(idx.size, group))
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)


def fit_model(make_model, param_grid, X, y, groups):
    rng = np.random.default_rng(RANDOM_STATE)
    subset = rng.choice(len(y), min(CV_PIXELS, len(y)), replace=False)
    search = GridSearchCV(make_model(), param_grid, cv=GroupKFold(CV_FOLDS), scoring="f1", n_jobs=-1)
    search.fit(X[subset], y[subset], groups=groups[subset])
    print(f"Grid search on {subset.size} pixels: best {search.best_params_} (CV F1 {search.best_score_:.4f})")
    for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        print(f"  {params}: {score:.4f}")
    t0 = time.time()
    model = make_model().set_params(**search.best_params_).fit(X, y)
    print(f"Final fit on {len(y)} pixels in {time.time() - t0:.1f}s")
    return model, search.best_params_


def score_map(model, features):
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return model.predict_proba(features)[:, 1] - 0.5


def segment(model, img):
    """Return (cleaned lungs, raw prediction, body, steps panels) at full size."""
    small, small_body, features = prepare(img)
    # Only pixels inside the body can be lung (clean-up masks by the body
    # anyway), so only they are scored -- roughly halves prediction time.
    inside = small_body.ravel()
    scores = np.full(inside.size, -1.0)
    if inside.any():
        scores[inside] = score_map(model, features[inside])
    scores = scores.reshape(WORK_SIZE)
    raw = transform.resize(scores, img.shape, order=1) > 0

    body, _, _ = body_mask(img)
    labels = measure.label(raw & body)
    min_area = MIN_LUNG_FRACTION * body.sum()
    regions = sorted((r for r in measure.regionprops(labels) if r.area >= min_area),
                     key=lambda r: r.area, reverse=True)[:2]
    lungs = clean_lungs([labels == r.label for r in regions], body, img)

    feature_maps = features.reshape(*WORK_SIZE, -1)
    panels = [
        ("Input", img),
        (f"Feature: smoothed intensity (sigma {SIGMA_MAX})", feature_maps[..., (N_SIGMAS - 1) * 4]),
        ("Feature: distance to body edge", feature_maps[..., -2]),
        ("Classifier score (white = lung)", scores),
        ("Raw prediction (score > 0)", raw),
        ("Cleaned: 2 largest, closed, filled", lungs),
    ]
    return lungs, raw, body, panels


def run_dataset(name, make_model, param_grid, method_dir, tag, title):
    print(f"\n{'#' * 50}\n# {name}\n{'#' * 50}")
    images, split_dir = list_images(name)
    train_images = [p for p in images if p.relative_to(split_dir).parts[0] == "train"]
    out_dir = method_dir / "Results" / name

    t0 = time.time()
    X, y, groups = build_training_set(name, train_images, split_dir)
    print(f"Training pixels: {len(y)} from {len(train_images)} images ({y.mean():.0%} lung), "
          f"{X.shape[1]} features, built in {time.time() - t0:.1f}s")
    model, best_params = fit_model(make_model, param_grid, X, y, groups)

    model_path = method_dir / "Models" / f"{name}_{tag}_segmenter.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model, best_params=best_params, work_size=WORK_SIZE,
                     sigmas=(SIGMA_MIN, SIGMA_MAX, N_SIGMAS), labels="lungmask R231 reference, train split"),
               model_path)

    test_ids = [i for i, p in enumerate(images) if p.relative_to(split_dir).parts[0] == "test"]
    overlay_ids = set(random.Random(RANDOM_STATE).sample(test_ids, min(N_OVERLAYS, len(test_ids))))
    t0 = time.time()
    rows, overlays = [], []
    for i, path in enumerate(images):
        rel = path.relative_to(split_dir)
        img = load_gray(path)
        t_image = time.perf_counter()
        lungs, raw, body, panels = segment(model, img)
        seconds = time.perf_counter() - t_image
        reference = reference_mask(name, rel)

        mask_path = (out_dir / "masks" / rel).with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(mask_path, (lungs * 255).astype(np.uint8), check_contrast=False)

        body_area = body.sum()
        rows.append(dict(
            split=rel.parts[0], label=rel.parts[1], image=rel.name,
            raw_dice=round(dice(raw & body, reference), 4),
            lung_regions=measure.label(lungs).max(),
            lung_fraction_of_body=round(lungs.sum() / body_area, 4) if body_area else 0.0,
            seconds=round(seconds, 4),
            **score(lungs, reference),
        ))
        if i in overlay_ids:
            overlays.append((rel, img, lungs, reference))
            if len(overlays) == 1:
                save_steps([panels[0], ("Reference mask (lungmask)", reference)] + panels[1:],
                           out_dir / "steps.png", f"{name}: {rel}")
    predict_time = time.time() - t0

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_overlays(overlays, out_dir / "overlays.png", f"{name}: {title} (red) vs. reference (green), test images")

    print(f"Segmented {len(rows)} images in {predict_time:.1f}s ({predict_time / len(rows):.2f}s per image)")
    for split in ("train", "test"):
        raw = np.array([r["raw_dice"] for r in rows if r["split"] == split])
        print(f"{split:>5}: raw (before clean-up) Dice mean {raw.mean():.4f} / median {np.median(raw):.4f}")
    print_scores(rows)
    print(f"Saved model to {model_path} and masks, summary.csv, overlays.png, steps.png to {out_dir}")


def main(make_model, param_grid, method_dir, tag, title):
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help=f"subset of {', '.join(CT_DATASETS)} (default: all)")
    args = parser.parse_args()
    names = args.datasets or CT_DATASETS
    unknown = set(names) - set(CT_DATASETS)
    if unknown:
        parser.error(f"unknown or non-CT dataset: {', '.join(sorted(unknown))}")
    for name in names:
        run_dataset(name, make_model, param_grid, method_dir, tag, title)
