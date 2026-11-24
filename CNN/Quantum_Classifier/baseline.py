import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from data import QUANTUM_DIR, load_features, scores, tag

sys.path.insert(0, str(QUANTUM_DIR.parent / "Feature_Extraction"))
from adabelief import AdaBelief  # noqa: E402

N_QUBITS = 8
SEEDS = [0, 1, 2, 3, 4]


def make_model(name, n_features, n_classes):
    if name == "768 -> 3":
        return nn.Linear(n_features, n_classes)
    return nn.Sequential(nn.Linear(n_features, N_QUBITS), nn.Tanh(), nn.Linear(N_QUBITS, n_classes))


def train(model, X, y, args, seed, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = AdaBelief(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    losses = []
    for _ in range(args.epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(X))).to(device)
        total = 0.0
        for b in range(0, len(X), args.batch_size):
            idx = order[b:b + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        losses.append(total / len(X))
    return losses


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    return model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_features(args.test_size, args.arch, args.val_size)
    classes, n_features = d["classes"], d["X_train"].shape[1]
    out_dir = QUANTUM_DIR / "Results" / f"baseline_{tag(args.test_size, args.arch, args.val_size)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(d['X_train'])} train / {len(d['X_test'])} test feature vectors of {n_features} on {device}")

    rows, curves, cms = [], {}, {}
    for name in ("768 -> 3", f"768 -> {N_QUBITS} -> 3"):
        for seed in SEEDS:
            model = make_model(name, n_features, len(classes)).to(device)
            losses = train(model, d["X_train"], d["y_train"], args, seed, device)
            for split in ("train", "test"):
                pred = predict(model, d[f"X_{split}"], device)
                rows.append(dict(model=name, seed=seed, split=split,
                                 **{k: round(v * 100, 2) for k, v in scores(d[f"y_{split}"], pred).items()}))
                if split == "test":
                    cms[name] = cms.get(name, 0) + confusion_matrix(d["y_test"], pred, labels=range(len(classes)))
            curves.setdefault(name, []).append(losses)

    with open(out_dir / "metrics_per_seed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = ["accuracy", "precision", "recall", "f1", "mcc"]
    summary = []
    print(f"\nMean ± std over {len(SEEDS)} seeds")
    for name in cms:
        for split in ("train", "test"):
            part = [r for r in rows if r["model"] == name and r["split"] == split]
            s = dict(model=name, split=split)
            for m in metrics:
                vals = np.array([r[m] for r in part])
                s[m] = f"{vals.mean():.2f} ± {vals.std():.2f}"
            summary.append(s)
            print(f"{name:<14} {split:<5} " + "  ".join(f"{m} {s[m]}" for m in metrics))
    with open(out_dir / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, runs in curves.items():
        axes[0].plot(np.mean(runs, axis=0), label=name)
    axes[0].set(title="Training loss (mean over seeds)", xlabel="epoch", yscale="log")
    axes[0].legend()
    for ax, (name, cm) in zip(axes[1:], cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(title=f"{name}: test, summed over {len(SEEDS)} seeds", xticks=range(len(classes)),
               yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted",
               ylabel="Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "baseline.png", dpi=100)
    plt.close(fig)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
