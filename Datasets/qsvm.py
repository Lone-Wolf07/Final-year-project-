"""
Quantum-kernel SVM (QSVM) on the same HOG features as the classical models.

Pipeline: HOG (128x128 grayscale) -> StandardScaler -> PCA to one feature per
qubit -> per-component StandardScaler -> x bandwidth -> ZZFeatureMap
statevector -> fidelity kernel K(a, b) = |<psi(a)|psi(b)>|^2 -> SVC with a
precomputed kernel (class_weight="balanced").

Simulating each kernel entry as its own circuit is O(N^2) circuit runs (~63M
for LC25000). Instead every image's statevector is simulated ONCE, batched in
torch, and the whole kernel is a single matrix product. The simulator is
checked against Qiskit's ZZFeatureMap + Statevector before anything runs.
Runs on the GPU automatically when torch has CUDA, otherwise on the CPU.

Hyperparameters (qubits, feature-map reps, entanglement, bandwidth, C) are
chosen by stratified K-fold CV on the training split only; the test split is
touched once, for the final score. A classical RBF SVM on the same PCA
features is tuned the same way, as a baseline for what the quantum kernel adds.

Usage:
    python qsvm.py                                   # every dataset
    python qsvm.py IQ-OTH_NCCD Chest_CT                # chosen datasets
    python qsvm.py LC25000_Lung --max-train 6000     # cap final training set (kernel memory is N^2)
"""

import argparse
import math
import time
from itertools import combinations, product

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from create_splits import DATASETS_DIR
from train_test_models import HOG_PARAMS, IMG_SIZE, extract_features

DATASET_DIRS = {
    "Chest_CT": DATASETS_DIR / "Chest_CT",
    "IQ-OTH_NCCD": DATASETS_DIR / "IQ-OTH_NCCD",
    "LC25000_Lung": DATASETS_DIR / "LC25000_Lung",
}
MODEL_DIRS = {
    "Chest_CT": DATASETS_DIR.parent / "Task1" / "Models",
    "IQ-OTH_NCCD": DATASETS_DIR / "IQ-OTH_NCCD" / "Models",
    "LC25000_Lung": DATASETS_DIR / "LC25000_Lung" / "Models",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

# Search grid. Bandwidth scales the features before encoding; small values
# stop the kernel from collapsing towards the identity as qubits increase.
QUBITS = [4, 6, 8, 10, 12, 14, 16]
REPS = [1, 2]
ENTANGLEMENT = ["linear", "full"]
BANDWIDTHS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
CS = [0.1, 1, 10, 100, 1000]
RBF_GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


# --- Quantum feature map ------------------------------------------------

def walsh_hadamard(psi: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Apply H to every qubit of a batch of statevectors (fast WHT, O(n 2^n))."""
    batch = psi.shape[0]
    for k in range(n_qubits):
        h = 1 << k
        psi = psi.reshape(batch, -1, 2, h)
        a, b = psi[:, :, 0, :], psi[:, :, 1, :]
        psi = torch.stack((a + b, a - b), dim=2)
    return psi.reshape(batch, -1) / 2 ** (n_qubits / 2)


class ZZFeatureMap:
    """Batched statevector simulation of Qiskit's zz_feature_map.

    Each rep is H on every qubit, then P(2 x_i) on qubit i and
    CX-P(2 (pi - x_i)(pi - x_j))-CX on each entangled pair. That second layer
    is diagonal, so it is a single elementwise phase on the statevector.
    Basis index bit i is qubit i (little-endian, as in Qiskit).
    """

    def __init__(self, n_qubits: int, reps: int, entanglement: str):
        self.n_qubits, self.reps = n_qubits, reps
        if entanglement == "full":
            pairs = list(combinations(range(n_qubits), 2))
        else:
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        self.pair_i = torch.tensor([i for i, _ in pairs], device=DEVICE)
        self.pair_j = torch.tensor([j for _, j in pairs], device=DEVICE)
        index = torch.arange(2 ** n_qubits, device=DEVICE)
        bits = (index[:, None] >> torch.arange(n_qubits, device=DEVICE)) & 1
        self.bits = bits.double()                                        # (2^n, n)
        self.parity = (bits[:, self.pair_i] ^ bits[:, self.pair_j]).double()  # (2^n, pairs)

    def states(self, X: np.ndarray) -> torch.Tensor:
        batch = max(1, min(512, 2 ** 22 >> self.n_qubits))  # ~200MB of temporaries whatever the qubit count
        out = torch.empty((len(X), 2 ** self.n_qubits), dtype=torch.complex64, device=DEVICE)
        for start in range(0, len(X), batch):
            x = torch.as_tensor(X[start:start + batch], dtype=torch.float64, device=DEVICE)
            pair_phi = (math.pi - x[:, self.pair_i]) * (math.pi - x[:, self.pair_j])
            theta = 2 * (x @ self.bits.T + pair_phi @ self.parity.T)
            phase = torch.polar(torch.ones_like(theta), theta).to(torch.complex64)
            psi = phase / 2 ** (self.n_qubits / 2)  # first rep acts on H|0..0> = uniform state
            for _ in range(self.reps - 1):
                psi = walsh_hadamard(psi, self.n_qubits) * phase
            out[start:start + batch] = psi
        return out


def fidelity_kernel(states_a: torch.Tensor, states_b: torch.Tensor, block: int = 1024) -> np.ndarray:
    """K[i, j] = |<a_i|b_j>|^2, computed in row blocks to bound memory."""
    kernel = np.empty((len(states_a), len(states_b)))
    states_b_dag = states_b.conj().T
    for start in range(0, len(states_a), block):
        overlap = states_a[start:start + block] @ states_b_dag
        kernel[start:start + block] = (overlap.abs() ** 2).cpu().numpy()
    return kernel


def verify_against_qiskit():
    """Check the simulator reproduces Qiskit's zz_feature_map statevectors."""
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(0)
    for n_qubits, reps, entanglement in [(3, 1, "linear"), (4, 2, "full"), (5, 2, "linear"), (4, 3, "full")]:
        X = rng.uniform(-2, 2, size=(3, n_qubits))
        ours = ZZFeatureMap(n_qubits, reps, entanglement).states(X).cpu().numpy()
        circuit = zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
        for x, psi in zip(X, ours):
            reference = Statevector(circuit.assign_parameters(x)).data
            fidelity = abs(np.vdot(reference, psi)) ** 2
            if fidelity < 1 - 1e-4:
                raise AssertionError(f"Simulator mismatch (n={n_qubits}, reps={reps}, {entanglement}): fidelity {fidelity:.6f}")
    print("Simulator matches Qiskit zz_feature_map statevectors.")


# --- Data -----------------------------------------------------------------

class Preprocessor:
    """HOG -> StandardScaler -> PCA(max qubits) -> per-component StandardScaler.
    PCA components are nested, so the first q columns are the q-qubit input."""

    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X):
        self.hog_scaler = StandardScaler().fit(X)
        self.pca = PCA(self.n_components, svd_solver="randomized", random_state=RANDOM_STATE)
        self.pc_scaler = StandardScaler().fit(self.pca.fit_transform(self.hog_scaler.transform(X)))
        return self

    def transform(self, X):
        return self.pc_scaler.transform(self.pca.transform(self.hog_scaler.transform(X)))


def load_split_float32(split_dir):
    """DataSplit/<split>/<label>/... -> (HOG features as float32, labels)."""
    features, labels = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                features.append(extract_features(image_path).astype(np.float32))
                labels.append(class_dir.name)
    return np.stack(features), np.array(labels)


# --- Hyperparameter search ---------------------------------------------------

def cross_validate(X, y, n_folds, qubits):
    """Score every (qubits, reps, entanglement, bandwidth, C) for QSVM and every
    (qubits, gamma, C) for the RBF baseline. Returns {config: [fold scores]}."""
    qsvm_scores, rbf_scores = {}, {}
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        t0 = time.time()
        prep = Preprocessor(max(qubits)).fit(X[train_idx])
        Z_train, Z_val = prep.transform(X[train_idx]), prep.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        for q in qubits:
            for gamma, C in product(RBF_GAMMAS, CS):
                svc = SVC(kernel="rbf", gamma=gamma, C=C, class_weight="balanced").fit(Z_train[:, :q], y_train)
                pred = svc.predict(Z_val[:, :q])
                rbf_scores.setdefault((q, gamma, C), []).append(
                    (accuracy_score(y_val, pred), f1_score(y_val, pred, average="macro")))

            for reps, entanglement, bandwidth in product(REPS, ENTANGLEMENT, BANDWIDTHS):
                fmap = ZZFeatureMap(q, reps, entanglement)
                S_train = fmap.states(bandwidth * Z_train[:, :q])
                S_val = fmap.states(bandwidth * Z_val[:, :q])
                K_train, K_val = fidelity_kernel(S_train, S_train), fidelity_kernel(S_val, S_train)
                for C in CS:
                    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K_train, y_train)
                    pred = svc.predict(K_val)
                    qsvm_scores.setdefault((q, reps, entanglement, bandwidth, C), []).append(
                        (accuracy_score(y_val, pred), f1_score(y_val, pred, average="macro")))
        print(f"  fold {fold}/{n_folds} done in {time.time() - t0:.1f}s", flush=True)
    return qsvm_scores, rbf_scores


def best_config(scores):
    """Highest mean CV accuracy, macro F1 as the tie-break."""
    means = {cfg: np.mean(s, axis=0) for cfg, s in scores.items()}
    cfg = max(means, key=lambda c: (round(means[c][0], 6), means[c][1]))
    return cfg, means[cfg]


# --- Final fit + test ----------------------------------------------------------

def predict_features(svc, fmap, bandwidth, S_train, Z, block=1024):
    """Predict from PCA features in blocks, so the test kernel is never held whole."""
    preds = []
    for start in range(0, len(Z), block):
        S_block = fmap.states(bandwidth * Z[start:start + block])
        preds.append(svc.predict(fidelity_kernel(S_block, S_train)))
    return np.concatenate(preds)


def qsvm_predict(bundle, X_hog):
    """Predict encoded labels for raw HOG features with a saved QSVM bundle."""
    cfg = bundle["config"]
    fmap = ZZFeatureMap(cfg["qubits"], cfg["reps"], cfg["entanglement"])
    S_train = fmap.states(cfg["bandwidth"] * bundle["train_features"])
    Z = bundle["preprocessor"].transform(X_hog)[:, :cfg["qubits"]]
    return predict_features(bundle["svc"], fmap, cfg["bandwidth"], S_train, Z)


def report(title, y_true, y_pred, classes):
    acc, f1 = accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")
    print(f"\n--- {title} ---\nTest acc {acc:.4f} | macro F1 {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):", list(classes))
    print(confusion_matrix(y_true, y_pred))
    return acc, f1


def run_dataset(name, n_folds, search_size, max_train, qubits):
    base_dir = DATASET_DIRS[name]
    print(f"\n{'#' * 50}\n# {name}\n{'#' * 50}")
    t0 = time.time()
    X_train, y_train = load_split_float32(base_dir / "DataSplit" / "train")
    X_test, y_test = load_split_float32(base_dir / "DataSplit" / "test")
    print(f"Loaded {len(y_train)} train / {len(y_test)} test in {time.time() - t0:.1f}s", flush=True)

    label_encoder = LabelEncoder().fit(y_train)
    y_train, y_test = label_encoder.transform(y_train), label_encoder.transform(y_test)

    def stratified_subset(size):
        if size is None or size >= len(y_train):
            return np.arange(len(y_train))
        idx, _ = train_test_split(np.arange(len(y_train)), train_size=size, stratify=y_train, random_state=RANDOM_STATE)
        return np.sort(idx)

    search_idx = stratified_subset(search_size)
    print(f"\nCV search: {n_folds}-fold on {len(search_idx)} training images "
          f"({len(qubits) * len(REPS) * len(ENTANGLEMENT) * len(BANDWIDTHS)} quantum kernels x {len(CS)} C values per fold)")
    t0 = time.time()
    qsvm_scores, rbf_scores = cross_validate(X_train[search_idx], y_train[search_idx], n_folds, qubits)
    search_time = time.time() - t0
    (q, reps, entanglement, bandwidth, C), (cv_acc, cv_f1) = best_config(qsvm_scores)
    (rbf_q, gamma, rbf_C), (rbf_cv_acc, rbf_cv_f1) = best_config(rbf_scores)
    print(f"Search took {search_time:.1f}s")
    print(f"Best QSVM: qubits={q} reps={reps} entanglement={entanglement} bandwidth={bandwidth} C={C} "
          f"| CV acc {cv_acc:.4f} | CV macro F1 {cv_f1:.4f}")
    print(f"Best RBF baseline: PCA dims={rbf_q} gamma={gamma} C={rbf_C} | CV acc {rbf_cv_acc:.4f} | CV macro F1 {rbf_cv_f1:.4f}")

    # Final QSVM on the (optionally capped) full training split.
    train_idx = stratified_subset(max_train)
    print(f"\nFinal fit on {len(train_idx)} training images", flush=True)
    prep = Preprocessor(max(q, rbf_q)).fit(X_train[train_idx])
    Z_train = prep.transform(X_train[train_idx])
    Z_test = prep.transform(X_test)
    del X_train, X_test

    t0 = time.time()
    fmap = ZZFeatureMap(q, reps, entanglement)
    S_train = fmap.states(bandwidth * Z_train[:, :q])
    K_train = fidelity_kernel(S_train, S_train)
    svc = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K_train, y_train[train_idx])
    del K_train
    train_time = time.time() - t0
    bundle = {
        "preprocessor": prep,
        "train_features": Z_train[:, :q],
        "svc": svc,
        "config": dict(qubits=q, reps=reps, entanglement=entanglement, bandwidth=bandwidth, C=C),
        "label_encoder": label_encoder,
        "img_size": IMG_SIZE,
        "hog_params": HOG_PARAMS,
    }
    y_pred = predict_features(svc, fmap, bandwidth, S_train, Z_test[:, :q])
    print(f"QSVM trained in {train_time:.1f}s ({svc.support_.size} support vectors)")
    test_acc, test_f1 = report(f"QSVM ({q} qubits)", y_test, y_pred, label_encoder.classes_)

    model_dir = MODEL_DIRS[name]
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "qsvm_lung_classifier.joblib")

    rbf = SVC(kernel="rbf", gamma=gamma, C=rbf_C, class_weight="balanced").fit(Z_train[:, :rbf_q], y_train[train_idx])
    rbf_acc, rbf_f1 = report(f"Classical RBF SVM on same {rbf_q} PCA features", y_test, rbf.predict(Z_test[:, :rbf_q]), label_encoder.classes_)

    return dict(qsvm=(q, cv_acc, test_acc, test_f1, train_time), rbf=(rbf_q, rbf_cv_acc, rbf_acc, rbf_f1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help=f"subset of {', '.join(DATASET_DIRS)} (default: all)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--search-size", type=int, default=3000, help="stratified training subset used for the CV search")
    parser.add_argument("--max-qubits", type=int, default=max(QUBITS), help="drop larger qubit counts from the search (statevector memory is 2^n)")
    parser.add_argument("--max-train", type=int, default=None, help="cap on training images for the final fit")
    args = parser.parse_args()
    names = args.datasets or list(DATASET_DIRS)
    unknown = set(names) - set(DATASET_DIRS)
    if unknown:
        parser.error(f"unknown dataset: {', '.join(sorted(unknown))}")

    qubits = [q for q in QUBITS if q <= args.max_qubits]
    print(f"Device: {DEVICE} | qubits searched: {qubits}")
    verify_against_qiskit()
    results = {name: run_dataset(name, args.folds, args.search_size, args.max_train, qubits) for name in names}

    print(f"\n{'=' * 80}\nSUMMARY (test split)")
    print(f"{'Dataset':<15}{'Model':<22}{'CV acc':>10}{'Test acc':>10}{'Macro F1':>10}")
    for name, r in results.items():
        q, cv_acc, test_acc, test_f1, _ = r["qsvm"]
        rbf_q, rbf_cv_acc, rbf_acc, rbf_f1 = r["rbf"]
        print(f"{name:<15}{f'QSVM ({q} qubits)':<22}{cv_acc:>10.4f}{test_acc:>10.4f}{test_f1:>10.4f}")
        print(f"{name:<15}{f'RBF SVM ({rbf_q} PCA)':<22}{rbf_cv_acc:>10.4f}{rbf_acc:>10.4f}{rbf_f1:>10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    # Run via the imported module so saved bundles pickle Preprocessor as
    # qsvm.Preprocessor (loadable elsewhere), not __main__.Preprocessor.
    import qsvm
    qsvm.main()
