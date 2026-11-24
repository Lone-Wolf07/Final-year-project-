import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
CNN_DIR = HERE.parent
for sub in ("Quantum_Classifier", "Feature_Extraction"):
    sys.path.insert(0, str(CNN_DIR / sub))
from adabelief import AdaBelief  # noqa: E402
from circuit import PQC, fidelity, ry, rz  # noqa: E402
from data import scores, source_groups  # noqa: E402
from fidelity import FidelityClassifier, same_source  # noqa: E402

FEATURES = CNN_DIR / "Comparison" / "Features" / "convnext_eca_features_split_80_20.npz"
OUT = HERE / "Results"
EPOCHS, BATCH, LR = 50, 32, 1e-4
SEEDS = [0, 1, 2]


class PaperPQC(PQC):
    """Encode once, then trainable RZ/RY layers with a CNOT ring and no further encoding."""

    def forward(self, angles):
        n = angles.shape[0]
        state = torch.zeros(n, 2 ** self.n_qubits, dtype=torch.complex64, device=angles.device)
        state[:, 0] = 1
        for q in range(self.n_qubits):
            state = self.apply(state, ry(angles[:, q]), q)
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                state = self.apply(state, rz(self.theta[layer, q, 0]), q)
                state = self.apply(state, ry(self.theta[layer, q, 1]), q)
            state = state[:, self.cnot_ring]
        return state


class LinearAngles(nn.Module):
    def __init__(self, n_features, n_qubits):
        super().__init__()
        self.linear = nn.Linear(n_features, n_qubits)

    def forward(self, f):
        return self.linear(f)


class PCAAngles(nn.Module):
    def __init__(self, X_train, n_qubits):
        super().__init__()
        pca = PCA(n_qubits, random_state=0).fit(X_train)
        z = pca.transform(X_train)
        self.register_buffer("mean", torch.tensor(pca.mean_, dtype=torch.float32))
        self.register_buffer("comp", torch.tensor(pca.components_.T, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(z.std(0), dtype=torch.float32))

    def forward(self, f):
        return math.pi * torch.tanh((f - self.mean) @ self.comp / self.scale)


VARIANTS = {
    "Ours (re-uploading, π·tanh, learned T, exclusion)": {},
    "Paper circuit (encode once, gates after)": dict(paper_circuit=True),
    "Unbounded angles (no tanh)": dict(no_tanh=True),
    "Plain softmax (T fixed at 1)": dict(temperature=1.0, learn_t=False),
    "Fixed temperature T = 10": dict(temperature=10.0, learn_t=False),
    "No same-source exclusion": dict(no_exclusion=True),
    "PCA to 8 angles (fixed projection)": dict(pca=True),
}
SIZE_VARIANTS = {
    "1 layer": dict(layers=1),
    "3 layers": dict(layers=3),
    "4 qubits": dict(qubits=4),
    "10 qubits": dict(qubits=10),
}


def build(d, cfg, device):
    n_feat = d["X_train"].shape[1]
    model = FidelityClassifier(n_feat, 3, n_qubits=cfg.get("qubits", 8), n_layers=cfg.get("layers", 2),
                               learn_temperature=cfg.get("learn_t", True), temperature=cfg.get("temperature", 10.0))
    if cfg.get("paper_circuit"):
        model.pqc = PaperPQC(cfg.get("qubits", 8), cfg.get("layers", 2))
    if cfg.get("no_tanh"):
        model.proj = LinearAngles(n_feat, cfg.get("qubits", 8))
    if cfg.get("pca"):
        model.proj = PCAAngles(d["X_train"], cfg.get("qubits", 8))
    return model.to(device)


def run(d, cfg, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(d, cfg, device)
    X_tr = torch.from_numpy(d["X_train"]).to(device)
    y_tr = torch.from_numpy(d["y_train"]).long().to(device)
    X_te = torch.from_numpy(d["X_test"]).to(device)
    groups = torch.from_numpy(source_groups(d["files_train"])).to(device)
    exclude = same_source(groups, groups)
    eye = torch.eye(len(X_tr), dtype=torch.bool, device=device)
    train_exclude = eye if cfg.get("no_exclusion") else exclude
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdaBelief(params, lr=LR)
    circuit_grad, start = [], time.time()
    for epoch in range(EPOCHS):
        model.train()
        order = torch.from_numpy(rng.permutation(len(X_tr))).to(device)
        for b in range(0, len(X_tr), BATCH):
            idx = order[b:b + BATCH]
            opt.zero_grad()
            ex = None if cfg.get("no_exclusion") else train_exclude[idx]
            loss = F.cross_entropy(model(X_tr[idx], X_tr, y_tr, ex), y_tr[idx])
            loss.backward()
            if epoch == 0 and b == 0:
                circuit_grad.append(model.pqc.theta.grad.norm().item())
            opt.step()
    train_s = time.time() - start
    model.eval()
    with torch.no_grad():
        ref = model.encode(X_tr)
        te = model.encode(X_te)
        S = model.class_scores(te, ref, y_tr)
        prob = torch.softmax(model.temperature * S, 1).cpu().numpy()
        fid = fidelity(te, ref)
        nn1 = y_tr[fid.argmax(1)].cpu().numpy()
        top5 = y_tr[fid.topk(5, dim=1).indices].cpu().numpy()
        knn5 = np.array([np.bincount(r, minlength=3).argmax() for r in top5])
        Stest = model.class_scores(te, ref, y_tr)
        margin = (Stest.max(1).values - Stest.median(1).values).mean().item()
    y = d["y_test"]
    pred = prob.argmax(1)
    s = scores(y, pred)
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0)
    return dict(seed=seed, accuracy=100 * s["accuracy"], f1=100 * s["f1"], mcc=100 * s["mcc"],
                recall_benign=100 * rec[0], mean_top_prob=100 * prob.max(1).mean(),
                temperature=model.temperature.item(), circuit_grad_first_step=circuit_grad[0],
                score_margin=margin, nn1_accuracy=100 * (nn1 == y).mean(), knn5_accuracy=100 * (knn5 == y).mean(),
                train_s=train_s)


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main():
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = np.load(FEATURES)
    d = dict(X_train=f["X_train"].astype(np.float32), y_train=f["y_train"], X_test=f["X_test"].astype(np.float32),
             y_test=f["y_test"], files_train=f["files_train"])
    rows = []
    plan = [(n, c, SEEDS) for n, c in VARIANTS.items()] + [(n, c, SEEDS[:1]) for n, c in SIZE_VARIANTS.items()]
    for name, cfg, seeds in plan:
        for seed in seeds:
            r = dict(variant=name, **run(d, cfg, seed, device))
            rows.append(r)
            print(f"{name:52s} seed {seed}: acc {r['accuracy']:6.2f}  benign {r['recall_benign']:5.1f}  "
                  f"top-p {r['mean_top_prob']:5.1f}  circuit grad {r['circuit_grad_first_step']:.2e}  "
                  f"({r['train_s']:.0f} s)", flush=True)
            write(OUT / "classifier_ablation_per_seed.csv", rows)
    summary = []
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        row = dict(variant=name, seeds=len(rs))
        for k in ("accuracy", "f1", "mcc", "recall_benign", "mean_top_prob", "circuit_grad_first_step",
                  "nn1_accuracy", "knn5_accuracy", "train_s"):
            v = [r[k] for r in rs]
            row[f"{k}_mean"], row[f"{k}_std"] = float(np.mean(v)), float(np.std(v))
        summary.append(row)
    write(OUT / "classifier_ablation.csv", summary)
    print("saved", OUT / "classifier_ablation.csv")


if __name__ == "__main__":
    main()
