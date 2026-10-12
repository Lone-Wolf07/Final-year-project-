"""
SVM lung segmentation: an RBF-kernel support vector machine labels every
pixel lung / not lung from its hand-crafted features. Features, labels,
training-pixel sampling, clean-up and outputs are shared with the other ML
methods -- see ml_common.py.

The RBF kernel lets the SVM draw a curved boundary in feature space, e.g.
"dark AND far from the body edge AND low texture" -> lung. class_weight is
"balanced" so the lung / non-lung pixel ratio doesn't bias it. C (how hard it
tries to fit every training pixel) and gamma (how local each support vector's
influence is) are chosen by grouped cross-validation.

Usage:
    python svm_segmentation.py                 # both CT datasets
    python svm_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_common import main

METHOD_DIR = Path(__file__).resolve().parent
PARAM_GRID = {"svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1]}


def make_model():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", class_weight="balanced"))])


if __name__ == "__main__":
    main(make_model, PARAM_GRID, METHOD_DIR, "svm", "RBF SVM pixel classifier")
