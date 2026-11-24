import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

QUANTUM_DIR = Path(__file__).resolve().parent
FEATURES_DIR = QUANTUM_DIR.parent / "Feature_Extraction" / "Features"


def tag(test_size=0.2, arch="tiny", val_size=0.0):
    test_pct, val_pct = round(test_size * 100), round(val_size * 100)
    split = f"{100 - val_pct - test_pct}_{val_pct}_{test_pct}" if val_pct else f"{100 - test_pct}_{test_pct}"
    return f"split_{split}_{arch}"


def load_features(test_size=0.2, arch="tiny", val_size=0.0):
    path = FEATURES_DIR / f"features_{tag(test_size, arch, val_size)}.npz"
    if not path.exists():
        raise SystemExit(f"No features at {path}: run CNN/Feature_Extraction/extract_features.py first.")
    f = np.load(path)
    d = dict(X_train=f["X_train"].astype(np.float32), y_train=f["y_train"],
             X_test=f["X_test"].astype(np.float32), y_test=f["y_test"],
             files_train=f["files_train"], files_test=f["files_test"], classes=[str(c) for c in f["classes"]])
    if "X_val" in f:
        d.update(X_val=f["X_val"].astype(np.float32), y_val=f["y_val"], files_val=f["files_val"])
    return d


def source_groups(files):
    stems = [re.sub(r"_aug\d+$", "", Path(f).stem) for f in files]
    return np.unique(stems, return_inverse=True)[1]


def scores(y, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), precision=precision, recall=recall, f1=f1,
                mcc=matthews_corrcoef(y, pred))
