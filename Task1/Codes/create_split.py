"""
One-time setup: pool Data/train + Data/valid + Data/test into a single set
of images, then write ONE fresh random stratified 75/25 split to disk as
Datasets/Chest_CT/DataSplit/train/<class>/... and Datasets/Chest_CT/DataSplit/test/<class>/....
Byte-identical duplicate images are removed first (Datasets/create_splits.py
drop_duplicates), so no image is in both train and test.

This replaces the dataset's original train/valid/test folders (which come
from different scan batches and don't represent each other well -- see the
resplit diagnostic results) with a single consistent split that all models
can now train/test against.

Usage:
    python create_split.py
"""

import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR.parent / "Datasets"
DATASET_DIR = DATASETS_DIR / "Chest_CT"
sys.path.insert(0, str(DATASETS_DIR))
from create_splits import drop_duplicates
OLD_DATA_DIR = DATASET_DIR / "Data"
SPLIT_DIR = DATASET_DIR / "DataSplit"

TEST_SIZE = 0.25
RANDOM_STATE = 42

CANONICAL_CLASSES = [
    "normal",
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
]


def canonical_label(folder_name: str) -> str:
    for label in CANONICAL_CLASSES:
        if folder_name.startswith(label):
            return label
    raise ValueError(f"Unrecognized class folder: {folder_name}")


def collect_all_images():
    """Walk Data/train, Data/valid, Data/test and return (path, label) pairs."""
    items = []
    for source_split in ["train", "valid", "test"]:
        split_dir = OLD_DATA_DIR / source_split
        class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for class_dir in class_dirs:
            label = canonical_label(class_dir.name)
            image_paths = sorted(
                p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for image_path in image_paths:
                # Prefix with the source split so filenames can't collide
                # when copied into the new pooled folders below.
                items.append((image_path, label, f"{source_split}_{image_path.name}"))
    return items


def main():
    print("Collecting images from Data/train, Data/valid, Data/test...")
    items = collect_all_images()
    print(f"  {len(items)} images total")
    items = drop_duplicates(items)

    labels = [label for _, label, _ in items]
    print("Class distribution (pooled):")
    for label in CANONICAL_CLASSES:
        print(f"    {label}: {labels.count(label)}")

    train_items, test_items = train_test_split(
        items, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"\nFresh split: {len(train_items)} train / {len(test_items)} test")

    if SPLIT_DIR.exists():
        print(f"\nRemoving existing {SPLIT_DIR}...")
        shutil.rmtree(SPLIT_DIR)

    for split_name, split_items in [("train", train_items), ("test", test_items)]:
        for label in CANONICAL_CLASSES:
            (SPLIT_DIR / split_name / label).mkdir(parents=True, exist_ok=True)
        for src_path, label, dest_name in split_items:
            dest_path = SPLIT_DIR / split_name / label / dest_name
            shutil.copy2(src_path, dest_path)
        print(f"  Wrote {len(split_items)} images to {SPLIT_DIR / split_name}")

    print(f"\nDone. New split ready at {SPLIT_DIR}")


if __name__ == "__main__":
    main()
