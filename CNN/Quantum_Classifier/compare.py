import argparse
import csv
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve

import baseline
from data import QUANTUM_DIR, load_features, scores, tag
from train import add_args, fit

STAGE1_MODELS = QUANTUM_DIR.parent / "Feature_Extraction" / "Models"
CNN_HEAD = "ConvNeXt head (Stage 1)"
QUANTUM = "8-qubit fidelity"


def cnn_head_probs(d, test_size, arch, val_size=0.0):
    state = torch.load(STAGE1_MODELS / f"convnext_eca_{tag(test_size, arch, val_size)}.pt", map_location="cpu")
    W, b = state["head.weight"].float(), state["head.bias"].float()
    return {split: torch.softmax(torch.from_numpy(d[f"X_{split}"]) @ W.T + b, 1).numpy() for split in ("train", "test")}


@torch.no_grad()
def baseline_probs(model, X, device):
    model.eval()
    return torch.softmax(model(torch.from_numpy(X).to(device)), 1).cpu().numpy()


def row(model, seed, y, prob, pred, classes):
    r = dict(model=model, seed=seed, **{k: round(v * 100, 2) for k, v in scores(y, pred).items()})
    r["auc"] = round(roc_auc_score(y, prob, multi_class="ovr", average="macro") * 100, 2)
    for c, rec in zip(classes, recall_score(y, pred, average=None, labels=range(len(classes)), zero_division=0)):
        r[f"recall_{c}"] = round(rec * 100, 2)
    return r


def main():
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_features(args.test_size, args.arch, args.val_size)
    classes, y_te = d["classes"], d["y_test"]
    out_dir = QUANTUM_DIR / "Results" / f"evaluation_{tag(args.test_size, args.arch, args.val_size)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    print(f"{len(d['X_train'])} train / {len(y_te)} test, {args.seeds} seeds on {device}")

    rows, probs, preds, quantum_logs = [], {}, {}, []
    p = cnn_head_probs(d, args.test_size, args.arch, args.val_size)["test"]
    rows.append(row(CNN_HEAD, "-", y_te, p, p.argmax(1), classes))
    probs[CNN_HEAD], preds[CNN_HEAD] = [p], [p.argmax(1)]

    base_args = SimpleNamespace(lr=args.baseline_lr, epochs=args.baseline_epochs, batch_size=args.batch_size)
    for name in ("768 -> 3", "768 -> 8 -> 3"):
        for seed in seeds:
            model = baseline.make_model(name, d["X_train"].shape[1], len(classes)).to(device)
            baseline.train(model, d["X_train"], d["y_train"], base_args, seed, device)
            p = baseline_probs(model, d["X_test"], device)
            rows.append(row(name, seed, y_te, p, p.argmax(1), classes))
            probs.setdefault(name, []).append(p)
            preds.setdefault(name, []).append(p.argmax(1))
        print(f"{name}: done")

    for seed in seeds:
        start = time.time()
        _, log, _, pred, p = fit(d, args, seed, device, verbose=False)
        rows.append(row(QUANTUM, seed, y_te, p, pred, classes))
        probs.setdefault(QUANTUM, []).append(p)
        preds.setdefault(QUANTUM, []).append(pred)
        quantum_logs += [dict(seed=seed, **r) for r in log]
        print(f"{QUANTUM} seed {seed}: test acc {rows[-1]['accuracy']:.2f}  auc {rows[-1]['auc']:.2f}  "
              f"({time.time() - start:.0f} s)")

    def write(path, data):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    write(out_dir / "metrics_per_seed.csv", rows)
    write(out_dir / "quantum_training_logs.csv", quantum_logs)
    metric_keys = [k for k in rows[0] if k not in ("model", "seed")]
    summary = []
    print(f"\nTest set, mean ± std over {args.seeds} seeds")
    for name in probs:
        part = [r for r in rows if r["model"] == name]
        s = dict(model=name, runs=len(part))
        for k in metric_keys:
            vals = np.array([r[k] for r in part])
            s[k] = f"{vals.mean():.2f} ± {vals.std():.2f}"
        summary.append(s)
        print(f"{name:<24} " + "  ".join(f"{k} {s[k]}" for k in metric_keys))
    write(out_dir / "comparison.csv", summary)

    fig, axes = plt.subplots(1, len(preds), figsize=(5 * len(preds), 4.5))
    for ax, (name, ps) in zip(axes, preds.items()):
        cm = sum(confusion_matrix(y_te, pr, labels=range(len(classes))) for pr in ps)
        ax.imshow(cm, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        runs = "1 run" if len(ps) == 1 else f"summed over {len(ps)} seeds"
        ax.set(title=f"{name}\n({runs})", xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.5))
    for c, ax in enumerate(axes):
        for name, ps in probs.items():
            fpr, tpr, _ = roc_curve(y_te == c, ps[0][:, c])
            aucs = [roc_auc_score(y_te == c, pr[:, c]) for pr in ps]
            ax.plot(fpr, tpr, label=f"{name} (AUC {np.mean(aucs):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=f"{classes[c]} vs rest (seed 0 curve)", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, ax, title in (("test_loss", axes[0], "Test loss"), ("test_acc", axes[1], "Test accuracy (%)")):
        curves = np.array([[r[key] for r in quantum_logs if r["seed"] == s] for s in seeds])
        epochs = np.arange(curves.shape[1])
        ax.plot(epochs, curves.mean(0))
        ax.fill_between(epochs, curves.min(0), curves.max(0), alpha=0.3)
        ax.set(title=f"{QUANTUM}: {title}, mean and range over {args.seeds} seeds", xlabel="epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "quantum_training.png", dpi=100)
    plt.close(fig)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
