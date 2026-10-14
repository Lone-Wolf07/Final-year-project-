import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_common import run

METHOD_DIR = Path(__file__).resolve().parent
PARAM_GRID = {"svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.01, 0.1]}
MAX_FIT_PIXELS = 20000


def make_model():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", cache_size=1000))])


if __name__ == "__main__":
    run(make_model, PARAM_GRID, METHOD_DIR, "svm", "RBF SVM, auto-context", "SVM", MAX_FIT_PIXELS)
