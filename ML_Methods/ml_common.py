import argparse
import csv
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import filters, io, measure, transform
from skimage.feature import multiscale_basic_features
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

ML_DIR = Path(__file__).resolve().parent
TRADITIONAL_DIR = ML_DIR.parent / "Traditional_Methods"
sys.path.insert(0, str(TRADITIONAL_DIR))
from common import (MANUAL_REFERENCE_DATASETS, MIN_LUNG_FRACTION, N_OVERLAYS, RANDOM_STATE, body_mask,
                    clean_lungs, collect_images, dice, list_images, load_gray, open_windows, print_summary,
                    reference_mask, save_overlays, save_side_by_side, save_steps, score)

WORK_SIZE = (256, 256)
SIGMA_MIN, SIGMA_MAX, N_SIGMAS = 0.5, 16, 6
CONTEXT_SIGMAS = (2, 4, 8, 16)
PIXELS_PER_IMAGE = 400
BOUNDARY_SHARE, BOUNDARY_BAND = 0.4, 3
CV_PIXELS = 15000
SMOOTH_SIGMA = 1.0
THRESHOLDS = np.round(np.arange(0.3, 0.71, 0.05), 2)
CLEANUPS = ("full", "light")
MIN_SPECK_FRACTION = 0.001


def patient_of(rel):
    return rel.parts[1]


def split_of(rel):
    return rel.parts[0]


def stage1_features(small, body):
    basic = multiscale_basic_features(small, intensity=True, edges=True, texture=True,
                                      sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, num_sigma=N_SIGMAS)
    cut = filters.threshold_isodata(small[body]) if body.sum() > 10 else filters.threshold_isodata(small)
    relative = [filters.gaussian(small, s) - cut for s in (1, 4)]
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
    extra = np.stack(relative + [rows, cols, np.abs(cols - 0.5), edge_distance], axis=-1)
    return np.concatenate([basic, extra], axis=-1).astype(np.float32)


def context_features(prob):
    lung = prob > 0.5
    signed = (ndi.distance_transform_edt(lung) - ndi.distance_transform_edt(~lung)) / WORK_SIZE[0]
    maps = [prob] + [filters.gaussian(prob, s) for s in CONTEXT_SIGMAS] + [signed]
    return np.stack(maps, axis=-1).astype(np.float32)


def prepare(img):
    small = transform.resize(img, WORK_SIZE, anti_aliasing=True)
    body, _, _ = body_mask(small)
    return small, body, stage1_features(small, body)


def shrink_label(label):
    return transform.resize(label.astype(float), WORK_SIZE, order=1, anti_aliasing=True) > 0.5


def lung_probability(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return 1 / (1 + np.exp(-2 * model.decision_function(features)))


def probability_map(model, features, body):
    prob = np.zeros(WORK_SIZE, dtype=np.float32)
    if body.any():
        prob[body] = lung_probability(model, features[body])
    return prob


def sample_pixels(label, body, rng):
    band = body & (ndi.binary_dilation(label, iterations=BOUNDARY_BAND)
                   ^ ndi.binary_erosion(label, iterations=BOUNDARY_BAND))
    n_band = int(PIXELS_PER_IMAGE * BOUNDARY_SHARE)
    picks = []
    for pool, n in ((np.flatnonzero(band), n_band), (np.flatnonzero(body), PIXELS_PER_IMAGE - n_band)):
        if pool.size:
            picks.append(rng.choice(pool, min(n, pool.size), replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=int)


def training_pixels(dataset, path, split_dir, seed):
    rel = path.relative_to(split_dir)
    small, body, features = prepare(load_gray(path))
    label = shrink_label(reference_mask(dataset, rel)) & body
    idx = sample_pixels(label, body, np.random.default_rng(seed))
    return features.reshape(-1, features.shape[-1])[idx], label.ravel()[idx], idx


def fit_subset(n, max_pixels, seed=RANDOM_STATE):
    if max_pixels is None or n <= max_pixels:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_pixels, replace=False))


def fit_stage(make_model, param_grid, X, y, groups, name, max_fit_pixels):
    rng = np.random.default_rng(RANDOM_STATE)
    subset = rng.choice(len(y), min(CV_PIXELS, len(y)), replace=False)
    search = GridSearchCV(make_model(), param_grid, cv=GroupKFold(3), scoring="f1", n_jobs=-1)
    search.fit(X[subset], y[subset], groups=groups[subset])
    print(f"{name}: grid search on {subset.size} pixels, best {search.best_params_} "
          f"(CV F1 {search.best_score_:.4f})")
    t0 = time.time()
    fit = fit_subset(len(y), max_fit_pixels)
    model = make_model().set_params(**search.best_params_).fit(X[fit], y[fit])
    print(f"{name}: final fit on {fit.size} pixels in {time.time() - t0:.1f}s")
    return model, search.best_params_


def _one_core(model):
    params = {k: 1 for k in model.get_params() if k.endswith("n_jobs")}
    return model.set_params(**params) if params else model


def stage1_map(model, path):
    small, body, features = prepare(load_gray(path))
    return probability_map(model, features, body)


def oof_stage1_maps(stage1, X, y, groups, train_images, split_dir, max_fit_pixels):
    patients = np.array([patient_of(p.relative_to(split_dir)) for p in train_images])
    maps = [None] * len(train_images)
    for fold, (_, held_out) in enumerate(GroupKFold(3).split(train_images, groups=patients)):
        held_patients = set(patients[held_out])
        keep = np.flatnonzero(~np.isin(groups, list(held_patients)))
        keep = keep[fit_subset(keep.size, max_fit_pixels, RANDOM_STATE + fold)]
        model = _one_core(clone(stage1).fit(X[keep], y[keep]))
        fold_maps = Parallel(n_jobs=-1)(delayed(stage1_map)(model, train_images[i]) for i in held_out)
        for i, m in zip(held_out, fold_maps):
            maps[i] = m
        print(f"  out-of-fold stage 1, fold {fold + 1}/3: held out {sorted(held_patients)}")
    return maps


def predict_prob(stage1, stage2, img):
    small, body, features = prepare(img)
    prob1 = probability_map(stage1, features, body)
    features2 = np.concatenate([features, context_features(prob1)], axis=-1)
    prob2 = probability_map(stage2, features2, body)
    return small, body, prob1, prob2, features


def to_mask(prob, img, threshold, cleanup):
    full = transform.resize(filters.gaussian(prob, SMOOTH_SIGMA), img.shape, order=1)
    body, _, _ = body_mask(img)
    raw = (full > threshold) & body
    labels = measure.label(raw)
    if cleanup == "light":
        keep = [r.label for r in measure.regionprops(labels) if r.area >= MIN_SPECK_FRACTION * body.sum()]
        return ndi.binary_fill_holes(np.isin(labels, keep)) & body, raw, body
    min_area = MIN_LUNG_FRACTION * body.sum()
    regions = sorted((r for r in measure.regionprops(labels) if r.area >= min_area),
                     key=lambda r: r.area, reverse=True)[:2]
    return clean_lungs([labels == r.label for r in regions], body, img), raw, body


def val_dice(stage1, stage2, path, rel, dataset):
    img = load_gray(path)
    _, _, _, prob2, _ = predict_prob(stage1, stage2, img)
    reference = reference_mask(dataset, rel)
    return [[dice(to_mask(prob2, img, t, c)[0], reference) for t in THRESHOLDS] for c in CLEANUPS]


def segment_one(stage1, stage2, threshold, cleanup, path, rel, dataset, out_dir):
    img = load_gray(path)
    t0 = time.perf_counter()
    small, small_body, prob1, prob2, features = predict_prob(stage1, stage2, img)
    lungs, raw, body = to_mask(prob2, img, threshold, cleanup)
    seconds = time.perf_counter() - t0
    reference = reference_mask(dataset, rel)

    mask_path = (out_dir / "masks" / rel).with_suffix(".png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(mask_path, (lungs * 255).astype(np.uint8), check_contrast=False)

    stage1_only = to_mask(prob1, img, 0.5, cleanup)[0]
    body_area = body.sum()
    row = dict(split=split_of(rel), label=rel.parts[1], image=rel.name,
               stage1_dice=round(dice(stage1_only, reference), 4),
               raw_dice=round(dice(raw, reference), 4),
               lung_regions=measure.label(lungs).max(),
               lung_fraction_of_body=round(lungs.sum() / body_area, 4) if body_area else 0.0,
               seconds=round(seconds, 4),
               **score(lungs, reference))
    panels = [
        ("Input", img),
        ("Reference mask (hand-drawn)", reference),
        (f"Feature: smoothed intensity (sigma {SIGMA_MAX})", features[..., (N_SIGMAS - 1) * 4]),
        ("Feature: distance to body edge", features[..., -1]),
        ("Stage 1 lung probability", prob1),
        ("Stage 2 (auto-context) probability", prob2),
        (f"Raw (prob > {threshold})", raw),
        (f"Cleaned ({cleanup})", lungs),
    ]
    return row, (img, lungs, reference, panels)


def run_dataset(name, make_model, param_grid, method_dir, tag, title, max_fit_pixels):
    print(f"\n{'#' * 50}\n# {name}\n{'#' * 50}")
    images, split_dir = list_images(name)
    rels = [p.relative_to(split_dir) for p in images]
    train_images = [p for p, r in zip(images, rels) if split_of(r) == "train"]
    out_dir = method_dir / "Results" / name
    t_start = time.time()

    t0 = time.time()
    parts = Parallel(n_jobs=-1)(delayed(training_pixels)(name, p, split_dir, RANDOM_STATE + i)
                                for i, p in enumerate(train_images))
    X1 = np.concatenate([x for x, _, _ in parts])
    y = np.concatenate([y for _, y, _ in parts])
    groups = np.concatenate([np.full(len(yy), patient_of(p.relative_to(split_dir)))
                             for p, (_, yy, _) in zip(train_images, parts)])
    print(f"Training pixels: {len(y)} from {len(train_images)} images of {len(set(groups))} patients "
          f"({y.mean():.0%} lung), {X1.shape[1]} features, built in {time.time() - t0:.1f}s")
    stage1, params1 = fit_stage(make_model, param_grid, X1, y, groups, "Stage 1", max_fit_pixels)

    maps = oof_stage1_maps(stage1, X1, y, groups, train_images, split_dir, max_fit_pixels)
    X2 = np.concatenate([np.concatenate([x, context_features(m).reshape(-1, len(CONTEXT_SIGMAS) + 2)[idx]], axis=1)
                         for (x, _, idx), m in zip(parts, maps)])
    stage2, params2 = fit_stage(make_model, param_grid, X2, y, groups, "Stage 2", max_fit_pixels)
    stage1, stage2 = _one_core(stage1), _one_core(stage2)

    val = [(p, r) for p, r in zip(images, rels) if split_of(r) == "val"]
    scores = np.array(Parallel(n_jobs=-1)(delayed(val_dice)(stage1, stage2, p, r, name) for p, r in val))
    means = scores.mean(axis=0)
    best_cleanup, best_threshold = np.unravel_index(means.argmax(), means.shape)
    cleanup, threshold = CLEANUPS[best_cleanup], float(THRESHOLDS[best_threshold])
    for c, row in zip(CLEANUPS, means):
        print(f"Val mean Dice, {c} clean-up, by threshold: " + ", ".join(f"{t}: {m:.4f}" for t, m in zip(THRESHOLDS, row)))
    print(f"Chosen: threshold {threshold}, {cleanup} clean-up")

    model_path = method_dir / "Models" / f"{name}_{tag}_segmenter.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(stage1=stage1, stage2=stage2, threshold=threshold, cleanup=cleanup, params=(params1, params2),
                     work_size=WORK_SIZE, sigmas=(SIGMA_MIN, SIGMA_MAX, N_SIGMAS), context_sigmas=CONTEXT_SIGMAS,
                     labels="hand-drawn lung masks, train patients"), model_path)

    t0 = time.time()
    results = Parallel(n_jobs=-1)(delayed(segment_one)(stage1, stage2, threshold, cleanup, p, r, name, out_dir)
                                  for p, r in zip(images, rels))
    rows = [row for row, _ in results]
    predict_time = time.time() - t0

    test_ids = [i for i, r in enumerate(rels) if split_of(r) == "test"]
    overlay_ids = sorted(random.Random(RANDOM_STATE).sample(test_ids, min(N_OVERLAYS, len(test_ids))))
    overlays = [(rels[i], *results[i][1][:3]) for i in overlay_ids]
    save_steps(results[overlay_ids[0]][1][3], out_dir / "steps.png", f"{name}: {rels[overlay_ids[0]]}")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_overlays(overlays, out_dir / "overlays.png", f"{name}: {title} (red) vs. reference (green), test images")

    print(f"Segmented {len(rows)} images in {predict_time:.1f}s; whole run {time.time() - t_start:.0f}s")
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        print(f"{split:>5}: stage-1 Dice mean {np.mean([r['stage1_dice'] for r in part]):.4f}, "
              f"raw (before clean-up) Dice mean {np.mean([r['raw_dice'] for r in part]):.4f}")
    print_summary(rows, f"{title} (threshold {threshold}, {cleanup} clean-up)", name)
    print(f"Saved model to {model_path} and masks, summary.csv, overlays.png, steps.png to {out_dir}")


def view_segment(model, path):
    img = load_gray(path)
    _, _, _, prob2, _ = predict_prob(model["stage1"], model["stage2"], img)
    return to_mask(prob2, img, model["threshold"], model["cleanup"])[0]


def view(method_dir, tag, method, targets, show, trained_on):
    model_path = method_dir / "Models" / f"{trained_on}_{tag}_segmenter.joblib"
    if not model_path.exists():
        sys.exit(f"No trained model at {model_path} - train it first by running this script without --view.")
    model = joblib.load(model_path)
    paths = collect_images(targets)
    print(f"Segmenting {len(paths)} image(s) with {model_path.name} "
          f"(threshold {model['threshold']}, {model['cleanup']} clean-up) ...")
    masks = Parallel(n_jobs=-1)(delayed(view_segment)(model, p) for p in paths)
    if show:
        open_windows()
    for path, lungs in zip(paths, masks):
        save_side_by_side(path, load_gray(path), lungs, method, method_dir / "Results" / "side_by_side", show)


def main(make_model, param_grid, method_dir, tag, title, max_fit_pixels=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*",
                        help=f"subset of {', '.join(MANUAL_REFERENCE_DATASETS)} (default: all)")
    args = parser.parse_args()
    names = args.datasets or MANUAL_REFERENCE_DATASETS
    unknown = set(names) - set(MANUAL_REFERENCE_DATASETS)
    if unknown:
        parser.error(f"unknown dataset, or one without hand-drawn masks: {', '.join(sorted(unknown))}")
    for name in names:
        run_dataset(name, make_model, param_grid, method_dir, tag, title, max_fit_pixels)


def run(make_model, param_grid, method_dir, tag, title, method, max_fit_pixels=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--view", nargs="+", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--trained-on", default=MANUAL_REFERENCE_DATASETS[0])
    args, rest = parser.parse_known_args()
    if args.view:
        view(method_dir, tag, method, args.view, not args.no_show, args.trained_on)
    else:
        sys.argv = [sys.argv[0]] + rest
        main(make_model, param_grid, method_dir, tag, title, max_fit_pixels)
