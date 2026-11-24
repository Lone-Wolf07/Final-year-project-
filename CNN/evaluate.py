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

CNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CNN_DIR / "Quantum_Classifier"))
from bundle import Pipeline, choose_models  # noqa: E402
from data import QUANTUM_DIR, scores  # noqa: E402
from preprocessing import SOURCE_DIR, output_dir  # noqa: E402

TRAINING_STEPS = [("preprocessing_s", "1. Preprocessing (all images)"),
                  ("cnn_training_s", "2. ConvNeXt-ECA training (epochs only)"),
                  ("cnn_step_s", "2. ConvNeXt-ECA step (loading + training + extracting features)"),
                  ("quantum_step_s", "3. Quantum classifier training")]


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


def evaluate_model(path, n_single, device):
    p = Pipeline(path, device)
    cfg, classes = p.config, p.classes
    out_dir = QUANTUM_DIR / "Results" / "model_evaluation" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 100}\nModel: {path.name}")
    print("Settings: " + ", ".join(f"{k} {v}" for k, v in cfg.items()))

    with open(output_dir(cfg["test_size"]) / "manifest.csv") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    sources = {q.name: q for q in SOURCE_DIR.rglob("*") if q.is_file()}
    paths = [sources[r["source"]] for r in test_rows]
    y = np.array([classes.index(r["label"]) for r in test_rows])
    n = len(paths)
    print(f"Test set: {n} original images, run from the raw files through every stage")

    p.classify(p.features(p.preprocess(paths[:2])))
    images, t_pre = timed(lambda: p.preprocess(paths), torch.device("cpu"))
    feats, t_cnn = timed(lambda: p.features(images), device)
    (probs, _, _), t_q = timed(lambda: p.classify(feats), device)
    pred = probs.argmax(1)
    metrics = {k: round(v * 100, 2) for k, v in scores(y, pred).items()}
    metrics["auc"] = round(roc_auc_score(y, probs, multi_class="ovr", average="macro") * 100, 2)
    for c, rec in zip(classes, recall_score(y, pred, average=None, labels=range(len(classes)), zero_division=0)):
        metrics[f"recall_{c}"] = round(rec * 100, 2)

    single = []
    for q in paths[:n_single]:
        img, a = timed(lambda: p.preprocess([q]), torch.device("cpu"))
        feat, b = timed(lambda: p.features(img), device)
        _, c = timed(lambda: p.classify(feat), device)
        single.append((a, b, c))
    single = np.array(single) * 1000

    timing = [dict(stage="Training", step=label, seconds=p.times.get(key)) for key, label in TRAINING_STEPS]
    test = f"Test set ({n} images)"
    for step, s in (("1. Preprocessing", t_pre), ("2. ConvNeXt-ECA features", t_cnn), ("3. Quantum classifier", t_q),
                    ("Total", t_pre + t_cnn + t_q)):
        timing.append(dict(stage=test, step=step, seconds=round(s, 3), per_image_ms=round(s / n * 1000, 3)))
    one = f"One new image (mean of {len(single)})"
    for step, ms in (("1. Preprocessing", single[:, 0].mean()), ("2. ConvNeXt-ECA features", single[:, 1].mean()),
                     ("3. Quantum classifier", single[:, 2].mean()), ("Total", single.sum(1).mean())):
        timing.append(dict(stage=one, step=step, per_image_ms=round(ms, 3)))

    print("\nTest-set scores (%):")
    for k, v in metrics.items():
        print(f"  {k:<18} {v}")
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    print("Confusion matrix (rows actual, columns predicted: " + ", ".join(classes) + "):")
    print(cm)
    print("\nTimes:")
    for r in timing:
        sec, ms = r.get("seconds"), r.get("per_image_ms")
        if r["stage"] == "Training" and sec is None:
            shown = "  not recorded"
        else:
            shown = f"{sec:>12.3f} s" if sec is not None else " " * 14
        print(f"  {r['stage']:<27} {r['step']:<64}{shown}" + (f"  {ms:>9.3f} ms/image" if ms is not None else ""))

    write_csv(out_dir / "metrics.csv", [metrics])
    write_csv(out_dir / "timing.csv", timing, ["stage", "step", "seconds", "per_image_ms"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(title=f"{path.stem}\naccuracy {metrics['accuracy']}%", xticks=range(len(classes)),
           yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=100)
    plt.close(fig)
    print(f"Saved to {out_dir}")

    summary = dict(model=path.stem, split=cfg["split"], cnn_epochs=cfg["cnn_epochs"],
                   quantum_epochs=cfg["quantum_epochs"], layers=cfg["quantum_layers"], **metrics)
    summary.update({key: p.times.get(key) for key, _ in TRAINING_STEPS})
    summary.update(test_set_s=round(t_pre + t_cnn + t_q, 3), one_image_ms=round(single.sum(1).mean(), 3))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to a saved model; leave out to choose from the Models folder")
    parser.add_argument("--single", type=int, default=20, help="images used to time one-image classification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = [evaluate_model(path, args.single, device) for path in choose_models(args.model, many=True)]
    if len(summaries) > 1:
        out = QUANTUM_DIR / "Results" / "model_evaluation" / "comparison.csv"
        write_csv(out, summaries)
        cols = ["split", "cnn_epochs", "quantum_epochs", "accuracy", "f1", "mcc", "recall_benign",
                "cnn_training_s", "quantum_step_s", "one_image_ms"]
        print(f"\n{'=' * 100}\nComparison")
        print(f"{'model':<44}" + "".join(f"{c:>15}" for c in cols))
        for s in summaries:
            print(f"{s['model']:<44}" + "".join(f"{str(s[c]) if s[c] is not None else '-':>15}" for c in cols))
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
