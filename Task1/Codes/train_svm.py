import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# --- Config -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "Datasets" / "Chest_CT" / "DataSplit"
MODEL_DIR = BASE_DIR / "Models"
MODEL_PATH = MODEL_DIR / "svm_lung_classifier.joblib"

IMG_SIZE = (128, 128)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))

#Giving simpler labels to the classification types
CANONICAL_CLASSES = [
    "normal",
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
]

"""Defining each folder into a different type"""
def canonical_label(folder_name: str) -> str:
    for label in CANONICAL_CLASSES:
        if folder_name.startswith(label):
            return label
    raise ValueError(f"Unrecognized class folder: {folder_name}")

"""Resizing the images into 128x128 size so that all the image vectors lineup properly"""
def extract_features(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    return hog(img, **HOG_PARAMS)

"""Goes into each folder and gives out two outputs, the features and the labels
for the respective features"""
def load_split(split_dir: Path):
    features, labels = [], []
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    for class_dir in class_dirs:
        label = canonical_label(class_dir.name)
        image_paths = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for image_path in image_paths:
            features.append(extract_features(image_path))
            labels.append(label)
    return np.array(features), np.array(labels)


def main():
    print("Loading training set...")
    t0 = time.time()
    X_train, y_train = load_split(DATA_DIR / "train")
    print(f"  {len(y_train)} images loaded in {time.time() - t0:.1f}s")
    for label in CANONICAL_CLASSES:
        print(f"    {label}: {np.sum(y_train == label)}")

    label_encoder = LabelEncoder().fit(CANONICAL_CLASSES)
    y_train_enc = label_encoder.transform(y_train)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=42),
            ),
        ]
    )

    print("Training SVM...")
    t0 = time.time()
    pipeline.fit(X_train, y_train_enc)
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")

    # Fit accuracy on the training set itself -- a sanity check only, NOT a
    # generalization measure (that's what test_svm.py's held-out set is for).
    train_accuracy = accuracy_score(y_train_enc, pipeline.predict(X_train))
    print(f"  Training-set accuracy (sanity check, not generalization): {train_accuracy:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "img_size": IMG_SIZE,
            "hog_params": HOG_PARAMS,
        },
        MODEL_PATH,
    )
    print(f"\nModel saved to {MODEL_PATH}")

    print("\n" + "=" * 40)
    print("FINAL TRAINING RESULT")
    print(f"  Training time     : {train_time:.1f}s")
    print(f"  Training accuracy : {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
