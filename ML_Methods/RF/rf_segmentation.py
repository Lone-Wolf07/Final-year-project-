import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml_common import RANDOM_STATE, run

METHOD_DIR = Path(__file__).resolve().parent
N_TREES = 200
PARAM_GRID = {"max_depth": [None, 25], "min_samples_leaf": [1, 5, 20]}

def make_model():
    return RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE)


if __name__ == "__main__":
    run(make_model, PARAM_GRID, METHOD_DIR, "rf", "random forest, auto-context", "Random forest")
