"""
Stage 1b of QCF-ELCD (Sec. 3.2): fine-tune ConvNeXt + ECA on the preprocessed IQ-OTH/NCCD images, then
extract the 768-d feature vector of every image for the Stage 2 quantum classifier.

Training follows Table 1: AdaBelief (lr 1e-4, betas 0.9/0.999, eps 1e-8), batch 32, 50 epochs,
cross-entropy loss, z-score normalisation and random rotation/flip/scaling on the training images.
A temporary linear head (768 -> 3) supplies the loss; its test scores are a sanity check on the features,
not the paper's quantum fidelity classifier. The test split is only evaluated once, after training.

Run CNN/Preprocessing/preprocessing.py first.

Usage:
    python extract_features.py                    # 80:20 split, ConvNeXt-Tiny (768-d)
    python extract_features.py --test-size 0.3    # 70:30 split
    python extract_features.py --arch base        # ConvNeXt-Base (1024-d)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support)
from torch import nn

FEATURE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FEATURE_DIR.parent / "Preprocessing"))
from preprocessing import CLASSES, RANDOM_STATE, augment, output_dir, z_score  # noqa: E402

from adabelief import AdaBelief  # noqa: E402
from model import ConvNeXtECA  # noqa: E402

EPOCHS, BATCH_SIZE, LR = 5, 32, 1e-4
BETAS, EPS = (0.9, 0.999), 1e-8
EVAL_BATCH = 64


def load_split(data_dir, split):
    with open(data_dir / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    images = np.stack([z_score(cv2.imread(str(data_dir / r["file"]), cv2.IMREAD_GRAYSCALE))[0] for r in rows])
    labels = np.array([CLASSES.index(r["label"]) for r in rows])
    return images, labels, rows


def to_batch(images, device):
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)[:, None].expand(-1, 3, -1, -1)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate_split(model, images, labels, device):
    model.eval()
    loss, correct = 0.0, 0
    for b in range(0, len(images), EVAL_BATCH):
        y = torch.from_numpy(labels[b:b + EVAL_BATCH]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(to_batch(images[b:b + EVAL_BATCH], device)).float()
        loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
    return loss / len(images), correct / len(images)


def train(model, images, labels, args, device, log_path, val=None):
    optimizer = AdaBelief(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(RANDOM_STATE)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss, correct = time.time(), 0.0, 0
        order = rng.permutation(len(images))
        for b in range(0, len(order), BATCH_SIZE):
            batch = order[b:b + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            for mb in np.array_split(batch, max(1, int(np.ceil(len(batch) / args.micro_batch)))):
                x = to_batch(np.stack([augment(images[i], rng) for i in mb]), device)
                y = torch.from_numpy(labels[mb]).to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = loss_fn(logits.float(), y)
                scaler.scale(loss * len(mb) / len(batch)).backward()
                total_loss += loss.item() * len(mb)
                correct += (logits.argmax(1) == y).sum().item()
            scaler.step(optimizer)
            scaler.update()
        h = dict(epoch=epoch, loss=total_loss / len(images), accuracy=correct / len(images),
                 seconds=round(time.time() - start, 1))
        if val is not None:
            h["val_loss"], h["val_accuracy"] = evaluate_split(model, *val, device)
        history.append(h)
        print(f"epoch {epoch:>3}/{args.epochs}  loss {h['loss']:.4f}  train acc {h['accuracy']:.4f}"
              + (f"  val loss {h['val_loss']:.4f}  val acc {h['val_accuracy']:.4f}" if val is not None else "")
              + f"  ({h['seconds']}s)", flush=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return history


@torch.no_grad()
def extract(model, images, device):
    model.eval()
    feats, logits = [], []
    for b in range(0, len(images), EVAL_BATCH):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.features(to_batch(images[b:b + EVAL_BATCH], device))
            logits.append(model.head(f).float().cpu())
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).argmax(1).numpy()


def scores(y, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), precision=precision, recall=recall, f1=f1,
                mcc=matthews_corrcoef(y, pred))


def save_plots(history, cms, out_dir, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [h["epoch"] for h in history]
    ax[0].plot(epochs, [h["loss"] for h in history], label="train")
    ax[0].set(title="Loss", xlabel="epoch")
    ax[1].plot(epochs, [h["accuracy"] for h in history], label="train (augmented images)")
    ax[1].set(title="Accuracy", xlabel="epoch")
    if "val_loss" in history[0]:
        ax[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
        ax[1].plot(epochs, [h["val_accuracy"] for h in history], label="validation")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.5))
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(title=f"{name} confusion matrix", xticks=range(3), yticks=range(3), xticklabels=CLASSES,
               yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual")
    fig.suptitle(f"{title}: linear-head sanity check")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2, help="must match the preprocessing run")
    parser.add_argument("--val-size", type=float, default=0.0, help="must match the preprocessing run")
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--micro-batch", type=int, default=16,
                        help="images per forward pass; gradients are accumulated up to the batch size of 32")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = output_dir(args.test_size, args.val_size)
    tag = f"{data_dir.name}_{args.arch}"
    for sub in ("Models", "Features", "Results"):
        (FEATURE_DIR / sub).mkdir(exist_ok=True)
    results_dir = FEATURE_DIR / "Results" / tag
    results_dir.mkdir(exist_ok=True)

    train_x, train_y, train_rows = load_split(data_dir, "train")
    test_x, test_y, test_rows = load_split(data_dir, "test")
    val = load_split(data_dir, "val") if args.val_size else None
    print(f"{tag} on {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}: "
          f"{len(train_x)} train / " + (f"{len(val[0])} val / " if val else "") + f"{len(test_x)} test images",
          flush=True)

    model = ConvNeXtECA(args.arch, len(CLASSES)).to(device).to(memory_format=torch.channels_last)
    history = train(model, train_x, train_y, args, device, results_dir / "training_log.csv",
                    val=val[:2] if val else None)
    torch.save(model.state_dict(), FEATURE_DIR / "Models" / f"convnext_eca_{tag}.pt")

    train_f, train_pred = extract(model, train_x, device)
    test_f, test_pred = extract(model, test_x, device)
    splits = [("train", train_y, train_pred), ("test", test_y, test_pred)]
    extra = {}
    if val:
        val_f, val_pred = extract(model, val[0], device)
        splits.insert(1, ("val", val[1], val_pred))
        extra = dict(X_val=val_f, y_val=val[1], files_val=[r["file"] for r in val[2]])
    np.savez_compressed(
        FEATURE_DIR / "Features" / f"features_{tag}.npz",
        X_train=train_f, y_train=train_y, files_train=[r["file"] for r in train_rows],
        augmented_train=[int(r["augmented"]) for r in train_rows],
        X_test=test_f, y_test=test_y, files_test=[r["file"] for r in test_rows], classes=CLASSES, **extra)

    rows = [dict(split=name, **{k: round(v * 100, 2) for k, v in scores(y, p).items()}) for name, y, p in splits]
    with open(results_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_plots(history, {name.capitalize(): confusion_matrix(y, p, labels=range(len(CLASSES))) for name, y, p in splits},
               results_dir, f"ConvNeXt-{args.arch.capitalize()} + ECA, {data_dir.name}")

    print(f"\nFeatures: train {train_f.shape}, test {test_f.shape} -> Features/features_{tag}.npz")
    for r in rows:
        print(f"{r['split']:<5} acc {r['accuracy']}  prec {r['precision']}  rec {r['recall']}  "
              f"f1 {r['f1']}  mcc {r['mcc']}  (linear head)")


if __name__ == "__main__":
    main()
