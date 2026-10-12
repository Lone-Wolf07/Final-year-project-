"""
Train + test the same five models as Task1 (SVM, Logistic Regression, Random
Forest, Gradient Boosting, KNN -- HOG features on 128x128 grayscale, same
hyperparameters and class-imbalance handling) on each dataset in
Datasets/<name>/DataSplit. Models are saved to Datasets/<name>/Models/.

Usage:
    python train_test_models.py                           # every dataset, every model
    python train_test_models.py IQ-OTH_NCCD               # one dataset
    python train_test_models.py --models rf,gb,knn        # chosen models only
"""

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from create_splits import DATASETS, DATASETS_DIR

IMG_SIZE = (128, 128)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

# name -> (factory, needs StandardScaler). Tree models split on raw thresholds
# so skip scaling, as in Task1. GB has no class_weight option, so it gets
# balanced per-sample weights at fit time instead (see run_dataset).
MODELS = {
    "svm": (lambda: SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42), True),
    "lr": (lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), True),
    "rf": (lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1), False),
    "gb": (lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42), False),
    "knn": (lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"), True),
}


def extract_features(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    return hog(img, **HOG_PARAMS)


def load_split(split_dir: Path):
    """DataSplit/<split>/<label>/... -> (features, labels)."""
    features, labels = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                features.append(extract_features(image_path))
                labels.append(class_dir.name)
    return np.array(features), np.array(labels)


def run_dataset(name: str, model_names: list):
    split_dir = DATASETS_DIR / name / "DataSplit"
    model_dir = DATASETS_DIR / name / "Models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 50}\n# {name}\n{'#' * 50}")
    t0 = time.time()
    X_train, y_train = load_split(split_dir / "train")
    X_test, y_test = load_split(split_dir / "test")
    print(f"Loaded {len(y_train)} train / {len(y_test)} test in {time.time() - t0:.1f}s")

    label_encoder = LabelEncoder().fit(sorted(set(DATASETS[name].values())))
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    results = {}
    for model_name in model_names:
        make_model, needs_scaling = MODELS[model_name]
        steps = [("scaler", StandardScaler())] if needs_scaling else []
        pipeline = Pipeline(steps + [(model_name, make_model())])
        fit_params = {}
        if model_name == "gb":
            fit_params["gb__sample_weight"] = compute_sample_weight("balanced", y_train_enc)

        print(f"\n--- {model_name.upper()} ---", flush=True)
        t0 = time.time()
        pipeline.fit(X_train, y_train_enc, **fit_params)
        train_time = time.time() - t0

        train_acc = accuracy_score(y_train_enc, pipeline.predict(X_train))
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test_enc, y_pred)
        macro_f1 = f1_score(y_test_enc, y_pred, average="macro")

        print(f"Trained in {train_time:.1f}s | train acc {train_acc:.4f} | test acc {test_acc:.4f} | macro F1 {macro_f1:.4f}")
        print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):", list(label_encoder.classes_))
        print(confusion_matrix(y_test_enc, y_pred))

        joblib.dump(
            {"pipeline": pipeline, "label_encoder": label_encoder, "img_size": IMG_SIZE, "hog_params": HOG_PARAMS},
            model_dir / f"{model_name}_lung_classifier.joblib",
        )
        results[model_name] = (train_time, train_acc, test_acc, macro_f1)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help=f"subset of {', '.join(DATASETS)} (default: all)")
    parser.add_argument("--models", default=",".join(MODELS), help=f"comma-separated subset of {','.join(MODELS)}")
    args = parser.parse_args()
    names = args.datasets or list(DATASETS)
    model_names = args.models.split(",")
    unknown = (set(names) - set(DATASETS)) | (set(model_names) - set(MODELS))
    if unknown:
        parser.error(f"unknown dataset/model: {', '.join(sorted(unknown))}")

    all_results = {name: run_dataset(name, model_names) for name in names}

    print(f"\n{'=' * 72}\nSUMMARY")
    print(f"{'Dataset':<15}{'Model':<7}{'Train time':>12}{'Train acc':>12}{'Test acc':>12}{'Macro F1':>12}")
    for name, results in all_results.items():
        for model_name, (t, tr, te, f1) in results.items():
            print(f"{name:<15}{model_name.upper():<7}{t:>11.1f}s{tr:>12.4f}{te:>12.4f}{f1:>12.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
