"""
One-time setup: for each dataset under Datasets/, take the raw images in
<dataset>/Data/<original class folder>/... and write ONE random stratified
75/25 split to disk as <dataset>/DataSplit/train/<class>/... and
<dataset>/DataSplit/test/<class>/.... Byte-identical duplicate images are
removed first (see drop_duplicates), so no image is in both train and test.

Datasets:
  IQ-OTH_NCCD   - lung CT slices (benign / malignant / normal)
                  https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset
  LC25000_Lung  - lung histopathology (adenocarcinoma / squamous / normal);
                  lung subset only, colon images not used
                  https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Usage:
    python create_splits.py                 # split every dataset
    python create_splits.py IQ-OTH_NCCD     # split just one
"""

import hashlib
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

DATASETS_DIR = Path(__file__).resolve().parent

TEST_SIZE = 0.25
RANDOM_STATE = 42

# Dataset folder -> {original class folder: label}. Labels reuse Task1's
# names where the classes overlap. "Bengin" is IQ-OTH/NCCD's own spelling.
DATASETS = {
    "IQ-OTH_NCCD": {
        "Bengin cases": "benign",
        "Malignant cases": "malignant",
        "Normal cases": "normal",
    },
    "LC25000_Lung": {
        "lung_n": "normal",
        "lung_aca": "adenocarcinoma",
        "lung_scc": "squamous.cell.carcinoma",
    },
}


def collect_all_images(data_dir: Path, class_map: dict):
    """Walk Data/<class> and return (path, label) pairs."""
    items = []
    for folder_name, label in class_map.items():
        image_paths = sorted(
            p for p in (data_dir / folder_name).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        items.extend((image_path, label) for image_path in image_paths)
    return items


def drop_duplicates(items):
    """Remove byte-identical images so no image can sit in both train and test.

    items are tuples whose first two fields are (path, label). Copies that all
    share one label keep a single copy (the first by path); copies filed under
    different labels are all dropped, since the true label is unknown.
    """
    groups = {}
    for item in items:
        digest = hashlib.md5(item[0].read_bytes()).hexdigest()
        groups.setdefault(digest, []).append(item)
    kept, n_copies, n_conflicts = [], 0, 0
    for group in groups.values():
        group = sorted(group, key=lambda item: str(item[0]))
        if len({item[1] for item in group}) > 1:
            n_conflicts += 1
            print(f"    dropped (conflicting labels): {', '.join(f'{i[0].name} [{i[1]}]' for i in group)}")
            continue
        kept.append(group[0])
        n_copies += len(group) - 1
    print(f"  De-duplication: removed {n_copies} duplicate copies and {n_conflicts} image(s) with "
          f"conflicting labels -> {len(kept)} unique images")
    return sorted(kept, key=lambda item: str(item[0]))


def split_dataset(name: str, class_map: dict):
    data_dir = DATASETS_DIR / name / "Data"
    split_dir = DATASETS_DIR / name / "DataSplit"
    classes = list(class_map.values())

    print(f"=== {name} ===")
    items = collect_all_images(data_dir, class_map)
    print(f"  {len(items)} images total")
    items = drop_duplicates(items)

    labels = [label for _, label in items]
    for label in classes:
        print(f"    {label}: {labels.count(label)}")

    train_items, test_items = train_test_split(
        items, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"  Split: {len(train_items)} train / {len(test_items)} test")

    if split_dir.exists():
        shutil.rmtree(split_dir)

    for split_name, split_items in [("train", train_items), ("test", test_items)]:
        for label in classes:
            (split_dir / split_name / label).mkdir(parents=True, exist_ok=True)
        for src_path, label in split_items:
            shutil.copy2(src_path, split_dir / split_name / label / src_path.name)
    print(f"  Written to {split_dir}\n")


def main():
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        split_dataset(name, DATASETS[name])


if __name__ == "__main__":
    main()
