"""
Random-forest lung segmentation: an ensemble of decision trees labels every
pixel lung / not lung from its hand-crafted features. Features, labels,
training-pixel sampling, clean-up and outputs are shared with the other ML
methods -- see ml_common.py.

Each tree asks a chain of yes/no questions ("intensity at sigma 8 < 0.3?",
"distance to body edge > 0.2?") on a random subset of pixels and features;
the forest votes, and the vote share is the score. Trees handle feature
interactions and unscaled features natively, so no scaler is needed.
class_weight is "balanced"; tree depth and leaf size are chosen by grouped
cross-validation.

Usage:
    python rf_segmentation.py                 # both CT datasets
    python rf_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_common import RANDOM_STATE, main

METHOD_DIR = Path(__file__).resolve().parent
N_TREES = 200
PARAM_GRID = {"max_depth": [None, 20], "min_samples_leaf": [1, 5]}


def make_model():
    return RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced", n_jobs=-1,
                                  random_state=RANDOM_STATE)


if __name__ == "__main__":
    main(make_model, PARAM_GRID, METHOD_DIR, "rf", "random forest pixel classifier")
