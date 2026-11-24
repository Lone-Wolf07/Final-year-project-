import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from torch import nn

COMP_DIR = Path(__file__).resolve().parent
CNN_DIR = COMP_DIR.parent
for sub in ("Quantum_Classifier", "Feature_Extraction", "Preprocessing", "ResNet50"):
    sys.path.insert(0, str(CNN_DIR / sub))
import baseline  # noqa: E402
from data import scores  # noqa: E402
from extract_features import EVAL_BATCH, load_split, to_batch, train  # noqa: E402
from model import ConvNeXtECA  # noqa: E402
from preprocessing import CLASSES, SOURCE_DIR, enhance, output_dir, z_score  # noqa: E402
from resnet50 import build as build_resnet  # noqa: E402
from train import fit  # noqa: E402

TEST_SIZE = 0.2
SEEDS = [0, 1, 2, 3, 4]
N_SINGLE = 20
BENCH_EPOCHS = 2
QUANTUM = SimpleNamespace(epochs=50, batch_size=32, lr=1e-4, layers=2, temperature=10.0, fixed_temperature=False)
LINEAR = SimpleNamespace(epochs=100, batch_size=32, lr=1e-3)
CONVNEXT_PATH = CNN_DIR / "Archive_50_epochs" / "Feature_Extraction" / "Models" / "convnext_eca_split_80_20_tiny.pt"
CONVNEXT_LOG = CNN_DIR / "Archive_50_epochs" / "Feature_Extraction" / "Results" / "split_80_20_tiny" / "training_log.csv"
RESNET_PATH = CNN_DIR / "ResNet50" / "Models" / "resnet50_split_80_20_e50.pt"
RESNET_LOG = CNN_DIR / "ResNet50" / "Results" / "resnet50_split_80_20_e50" / "training_log.csv"
OUT = COMP_DIR / "Results"


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log_seconds(path):
    with open(path) as f:
        return sum(float(r["seconds"]) for r in csv.DictReader(f))


def metrics(y, probs):
    pred = probs.argmax(1)
    m = {k: round(v * 100, 2) for k, v in scores(y, pred).items()}
    m["auc"] = round(roc_auc_score(y, probs, multi_class="ovr", average="macro") * 100, 2)
    for c, r in zip(CLASSES, recall_score(y, pred, average=None, labels=range(len(CLASSES)), zero_division=0)):
        m[f"recall_{c}"] = round(r * 100, 2)
    return m


class Extractor:
    def __init__(self, name, device):
        self.name, self.device = name, device
        if name == "ConvNeXt-ECA":
            net = ConvNeXtECA("tiny", len(CLASSES), pretrained=False)
            net.load_state_dict(torch.load(CONVNEXT_PATH, map_location="cpu"))
            self.body, self.head = net.features, net.head
        else:
            net = build_resnet(pretrained=False)
            net.load_state_dict(torch.load(RESNET_PATH, map_location="cpu"))
            self.head, net.fc = net.fc, nn.Identity()
            self.body = net
        self.net = net.to(device).to(memory_format=torch.channels_last).eval()
        self.head.to(device).eval()

    @torch.no_grad()
    def features(self, images):
        out = []
        for b in range(0, len(images), EVAL_BATCH):
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                out.append(self.body(to_batch(images[b:b + EVAL_BATCH], self.device)).float())
        return torch.cat(out)


def classical_head(ext):
    @torch.no_grad()
    def classify(f):
        return torch.softmax(ext.head(f), 1)
    return classify


def quantum_classifier(model, X_ref, y_ref):
    with torch.no_grad():
        ref_states = model.encode(X_ref)

    @torch.no_grad()
    def classify(f):
        return torch.softmax(model.temperature * model.class_scores(model.encode(f), ref_states, y_ref), 1)
    return classify


def linear_classifier(model):
    @torch.no_grad()
    def classify(f):
        return torch.softmax(model(f), 1)
    return classify


def benchmark_epochs(train_x, train_y, device):
    rows = []
    for name, make in (("ConvNeXt-ECA", lambda: ConvNeXtECA("tiny", len(CLASSES))), ("ResNet50", build_resnet)):
        torch.manual_seed(0)
        net = make().to(device).to(memory_format=torch.channels_last)
        print(f"\nTiming {BENCH_EPOCHS} training epochs of {name}", flush=True)
        args = SimpleNamespace(epochs=BENCH_EPOCHS, micro_batch=16)
        h = train(net, train_x, train_y, args, device, OUT / f"benchmark_{name}.csv")
        per_epoch = float(np.mean([r["seconds"] for r in h]))
        rows.append(dict(extractor=name, seconds_per_epoch=round(per_epoch, 1),
                         estimate_50_epochs_s=round(per_epoch * 50, 1)))
        del net
        torch.cuda.empty_cache()
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    (COMP_DIR / "Features").mkdir(exist_ok=True)
    data_dir = output_dir(TEST_SIZE)

    train_x, y_train, train_rows = load_split(data_dir, "train")
    files_train = [r["file"] for r in train_rows]
    bench = benchmark_epochs(train_x, y_train, device)
    measured = {"ConvNeXt-ECA": log_seconds(CONVNEXT_LOG), "ResNet50": log_seconds(RESNET_LOG)}
    for r in bench:
        r["measured_50_epochs_s"] = round(measured[r["extractor"]], 1)
    write_csv(OUT / "epoch_benchmark.csv", bench)

    with open(data_dir / "manifest.csv") as f:
        test_rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    sources = {q.name: q for q in SOURCE_DIR.rglob("*") if q.is_file()}
    paths = [sources[r["source"]] for r in test_rows]
    y_test = np.array([CLASSES.index(r["label"]) for r in test_rows])
    n = len(paths)

    def preprocess(ps):
        return np.stack([z_score(enhance(p)[0])[0] for p in ps])

    test_x, t_pre = timed(lambda: preprocess(paths), torch.device("cpu"))
    single_pre = [timed(lambda: preprocess([p]), torch.device("cpu")) for p in paths[:N_SINGLE]]
    print(f"\nPreprocessing: {t_pre / n * 1000:.1f} ms per image over {n} test images", flush=True)

    cells, seed_rows, cms = [], [], {}
    y_tr_t = torch.from_numpy(y_train).long().to(device)
    for ext_name in ("ConvNeXt-ECA", "ResNet50"):
        ext = Extractor(ext_name, device)
        ext.features(test_x[:4])
        X_tr = ext.features(train_x)
        X_te, t_feat = timed(lambda: ext.features(test_x), device)
        np.savez_compressed(COMP_DIR / "Features" / f"{ext_name.replace('-', '_').lower()}_features_split_80_20.npz",
                            X_train=X_tr.cpu().numpy(), y_train=y_train, files_train=files_train,
                            X_test=X_te.cpu().numpy(), y_test=y_test, files_test=[r["source"] for r in test_rows])
        d = dict(X_train=X_tr.cpu().numpy(), y_train=y_train, X_test=X_te.cpu().numpy(), y_test=y_test,
                 files_train=files_train, classes=CLASSES)
        print(f"\n{ext_name}: {X_tr.shape[1]} features per image", flush=True)

        classifiers = [("Classical head (end-to-end)", classical_head(ext), measured[ext_name], 0.0, None)]
        quantum_runs, linear_runs = [], []
        for seed in SEEDS:
            (model, *_), t = timed(lambda: fit(d, QUANTUM, seed, device, verbose=False), device)
            clf = quantum_classifier(model.eval(), X_tr, y_tr_t)
            m = metrics(y_test, clf(X_te).cpu().numpy())
            seed_rows.append(dict(extractor=ext_name, classifier="Quantum fidelity", seed=seed, train_s=round(t, 1), **m))
            quantum_runs.append((clf, t, m))
            print(f"  quantum seed {seed}: accuracy {m['accuracy']}  ({t:.0f} s)", flush=True)

            lin = baseline.make_model("768 -> 3", X_tr.shape[1], len(CLASSES)).to(device)
            _, t = timed(lambda: baseline.train(lin, d["X_train"], y_train, LINEAR, seed, device), device)
            lin.eval()
            clf = linear_classifier(lin)
            m = metrics(y_test, clf(X_te).cpu().numpy())
            seed_rows.append(dict(extractor=ext_name, classifier="Linear probe", seed=seed, train_s=round(t, 1), **m))
            linear_runs.append((clf, t, m))
            print(f"  linear  seed {seed}: accuracy {m['accuracy']}  ({t:.1f} s)", flush=True)
        classifiers.append(("Quantum fidelity", quantum_runs[0][0], measured[ext_name], quantum_runs[0][1], quantum_runs))
        classifiers.append(("Linear probe", linear_runs[0][0], measured[ext_name], linear_runs[0][1], linear_runs))

        single_feat = []
        for img in [preprocess([p]) for p in paths[:N_SINGLE]]:
            single_feat.append(timed(lambda: ext.features(img), device))

        for clf_name, clf, t_ext_train, t_clf_train, runs in classifiers:
            clf(X_te[:4])
            probs, t_clf = timed(lambda: clf(X_te), device)
            probs = probs.cpu().numpy()
            m = metrics(y_test, probs)
            single_clf = [timed(lambda: clf(f), device)[1] for f, _ in single_feat]
            single = np.array([[a[1], b[1], c] for a, b, c in zip(single_pre, single_feat, single_clf)]) * 1000
            row = dict(extractor=ext_name, classifier=clf_name, features=X_tr.shape[1], **m)
            if runs:
                for k in ("accuracy", "f1", "mcc", "recall_benign"):
                    vals = [r[2][k] for r in runs]
                    row[f"{k}_mean"], row[f"{k}_std"] = round(np.mean(vals), 2), round(np.std(vals), 2)
            row.update(extractor_train_s=round(t_ext_train, 1), classifier_train_s=round(t_clf_train, 1),
                       batch_preprocess_ms=round(t_pre / n * 1000, 3), batch_extract_ms=round(t_feat / n * 1000, 3),
                       batch_classify_ms=round(t_clf / n * 1000, 4),
                       batch_total_ms=round((t_pre + t_feat + t_clf) / n * 1000, 3),
                       batch_model_ms=round((t_feat + t_clf) / n * 1000, 3),
                       single_preprocess_ms=round(single[:, 0].mean(), 2), single_extract_ms=round(single[:, 1].mean(), 2),
                       single_classify_ms=round(single[:, 2].mean(), 3), single_total_ms=round(single.sum(1).mean(), 2))
            cells.append(row)
            cms[(ext_name, clf_name)] = confusion_matrix(y_test, probs.argmax(1), labels=range(len(CLASSES)))
            print(f"  {clf_name:<28} accuracy {m['accuracy']:>6}  benign recall {m['recall_benign']:>6}  "
                  f"model {row['batch_model_ms']:.2f} ms/image  one image {row['single_total_ms']:.1f} ms", flush=True)
        del ext, X_tr, X_te
        torch.cuda.empty_cache()

    write_csv(OUT / "cells.csv", cells)
    write_csv(OUT / "per_seed.csv", seed_rows)
    np.savez(OUT / "confusion_matrices.npz", **{f"{e}|{c}": v for (e, c), v in cms.items()})

    main_cells = [(e, c) for e in ("ConvNeXt-ECA", "ResNet50") for c in ("Classical head (end-to-end)", "Quantum fidelity")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for ax, key in zip(axes.flat, main_cells):
        cm = cms[key]
        acc = next(r["accuracy"] for r in cells if (r["extractor"], r["classifier"]) == key)
        ax.imshow(cm, cmap="Blues")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(title=f"{key[0]} + {key[1].split(' (')[0]}\naccuracy {acc}%", xticks=range(3), yticks=range(3),
               xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=120)
    plt.close(fig)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
