"""
Logistic-regression lung segmentation: a linear model labels every pixel
lung / not lung from its hand-crafted features. Features, labels,
training-pixel sampling, clean-up and outputs are shared with the other ML
methods -- see ml_common.py.

LR draws a single straight (linear) boundary in feature space: each feature
gets one weight, so it can learn "darker -> more lung" but not interactions
like "dark AND far from the body edge" unless the features already encode
them. That makes it the simplest, fastest and most interpretable of the four
classifiers. class_weight is "balanced"; C (inverse regularisation strength)
is chosen by grouped cross-validation.

Usage:
    python lr_segmentation.py                 # both CT datasets
    python lr_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_common import main

METHOD_DIR = Path(__file__).resolve().parent
PARAM_GRID = {"lr__C": [0.01, 0.1, 1, 10]}


def make_model():
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])


if __name__ == "__main__":
    main(make_model, PARAM_GRID, METHOD_DIR, "lr", "logistic regression pixel classifier")
