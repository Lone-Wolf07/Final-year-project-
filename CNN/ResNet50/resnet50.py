import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

RESNET_DIR = Path(__file__).resolve().parent
CNN_DIR = RESNET_DIR.parent
sys.path.insert(0, str(CNN_DIR / "Feature_Extraction"))
sys.path.insert(0, str(CNN_DIR / "Preprocessing"))
from extract_features import EVAL_BATCH, load_split, scores, to_batch, train  # noqa: E402
from preprocessing import CLASSES, RANDOM_STATE, SOURCE_DIR, enhance, output_dir, split_name, z_score  # noqa: E402

EPOCHS = 50


def build(pretrained=True):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


@torch.no_grad()
def predict(model, images, device):
    model.eval()
    probs = []
    for b in range(0, len(images), EVAL_BATCH):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probs.append(torch.softmax(model(to_batch(images[b:b + EVAL_BATCH], device)).float(), 1).cpu())
    return torch.cat(probs).numpy()


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--single", type=int, default=20, help="images used to time one-image classification")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = output_dir(args.test_size)
    name = f"resnet50_{split_name(args.test_size)}_e{args.epochs}"
    out_dir = RESNET_DIR / "Results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (RESNET_DIR / "Models").mkdir(exist_ok=True)

    train_x, train_y, _ = load_split(data_dir, "train")
    print(f"{name} on {device}: {len(train_x)} train images, {args.epochs} epochs", flush=True)
    model = build().to(device).to(memory_format=torch.channels_last)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    history = train(model, train_x, train_y, args, device, out_dir / "training_log.csv")
    train_s = sum(h["seconds"] for h in history)
    torch.save(model.state_dict(), RESNET_DIR / "Models" / f"{name}.pt")
    del train_x

    with open(data_dir / "manifest.csv") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    sources = {q.name: q for q in SOURCE_DIR.rglob("*") if q.is_file()}
    paths = [sources[r["source"]] for r in test_rows]
    y = np.array([CLASSES.index(r["label"]) for r in test_rows])
    n = len(paths)

    def preprocess(ps):
        return np.stack([z_score(enhance(p)[0])[0] for p in ps])

    predict(model, preprocess(paths[:2]), device)
    images, t_pre = timed(lambda: preprocess(paths), torch.device("cpu"))
    probs, t_net = timed(lambda: predict(model, images, device), device)
    pred = probs.argmax(1)
    metrics = {k: round(v * 100, 2) for k, v in scores(y, pred).items()}
    metrics["auc"] = round(roc_auc_score(y, probs, multi_class="ovr", average="macro") * 100, 2)
    for c, rec in zip(CLASSES, recall_score(y, pred, average=None, labels=range(len(CLASSES)), zero_division=0)):
        metrics[f"recall_{c}"] = round(rec * 100, 2)

    single = []
    for q in paths[:args.single]:
        img, a = timed(lambda: preprocess([q]), torch.device("cpu"))
        _, b = timed(lambda: predict(model, img, device), device)
        single.append((a, b))
    single = np.array(single) * 1000

    test = f"Test set ({n} images)"
    one = f"One new image (mean of {len(single)})"
    timing = [dict(stage="Training", step="ResNet50 training (epochs only)", seconds=round(train_s, 1)),
              dict(stage=test, step="1. Preprocessing", seconds=round(t_pre, 3), per_image_ms=round(t_pre / n * 1000, 3)),
              dict(stage=test, step="2. ResNet50", seconds=round(t_net, 3), per_image_ms=round(t_net / n * 1000, 3)),
              dict(stage=test, step="Total", seconds=round(t_pre + t_net, 3),
                   per_image_ms=round((t_pre + t_net) / n * 1000, 3)),
              dict(stage=one, step="1. Preprocessing", per_image_ms=round(single[:, 0].mean(), 3)),
              dict(stage=one, step="2. ResNet50", per_image_ms=round(single[:, 1].mean(), 3)),
              dict(stage=one, step="Total", per_image_ms=round(single.sum(1).mean(), 3))]

    print("\nTest-set scores (%):")
    for k, v in metrics.items():
        print(f"  {k:<18} {v}")
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    print("Confusion matrix (rows actual, columns predicted: " + ", ".join(CLASSES) + "):")
    print(cm)
    print("\nTimes:")
    for r in timing:
        sec, ms = r.get("seconds"), r.get("per_image_ms")
        print(f"  {r['stage']:<27} {r['step']:<34}" + (f"{sec:>12.3f} s" if sec is not None else " " * 14)
              + (f"  {ms:>9.3f} ms/image" if ms is not None else ""))

    write_csv(out_dir / "metrics.csv", [metrics])
    write_csv(out_dir / "timing.csv", timing, ["stage", "step", "seconds", "per_image_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([h["epoch"] for h in history], [h["loss"] for h in history])
    axes[0].set(title="ResNet50 training loss", xlabel="epoch")
    axes[1].imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set(title=f"Test confusion matrix, accuracy {metrics['accuracy']}%", xticks=range(len(CLASSES)),
                yticks=range(len(CLASSES)), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "results.png", dpi=100)
    plt.close(fig)
    print(f"\nSaved {RESNET_DIR / 'Models' / f'{name}.pt'} and results to {out_dir}")


if __name__ == "__main__":
    main()
