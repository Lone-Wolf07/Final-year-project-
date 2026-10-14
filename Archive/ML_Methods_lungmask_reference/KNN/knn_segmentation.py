"""
k-nearest-neighbours lung segmentation: every pixel takes the majority label
of the k training pixels closest to it in feature space. Features, labels,
training-pixel sampling, clean-up and outputs are shared with the other ML
methods -- see ml_common.py.

KNN has no training phase beyond storing the training pixels, so all the
work happens at prediction time (a distance to every stored pixel), which
makes it the slowest to predict. Features are standardised so no single one
dominates the distance. k and the vote weighting (uniform, or by inverse
distance) are chosen by grouped cross-validation; k values are odd so
uniform votes never tie.

Usage:
    python knn_segmentation.py                 # both CT datasets
    python knn_segmentation.py IQ-OTH_NCCD     # one dataset
"""

import sys
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_common import main

METHOD_DIR = Path(__file__).resolve().parent
PARAM_GRID = {"knn__n_neighbors": [5, 15, 31], "knn__weights": ["uniform", "distance"]}


def make_model():
    return Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_jobs=-1))])


if __name__ == "__main__":
    main(make_model, PARAM_GRID, METHOD_DIR, "knn", "k-nearest-neighbours pixel classifier")
