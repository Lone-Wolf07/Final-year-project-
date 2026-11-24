import argparse
import csv
import sys
import time
from types import SimpleNamespace

import torch

import baseline
from data import QUANTUM_DIR, load_features, tag
from compare import QUANTUM
from train import add_args, fit

FEATURE_DIR = QUANTUM_DIR.parent / "Feature_Extraction"
sys.path.insert(0, str(FEATURE_DIR))
sys.path.insert(0, str(QUANTUM_DIR.parent / "Preprocessing"))
from extract_features import extract, load_split  # noqa: E402
from model import ConvNeXtECA  # noqa: E402
from preprocessing import CLASSES, SOURCE_DIR, enhance, output_dir  # noqa: E402


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def main():
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = tag(args.test_size, args.arch, args.val_size)
    d = load_features(args.test_size, args.arch, args.val_size)
    n_tr, n_te = len(d["X_train"]), len(d["X_test"])
    rows = []

    def add(stage, model, phase, seconds, n):
        rows.append(dict(stage=stage, model=model, phase=phase, images=n, total_s=round(seconds, 3),
                         per_image_ms=round(seconds / n * 1000, 3)))
        print(f"{stage:<26} {model:<24} {phase:<9} {n:>5} images  {seconds:9.3f} s  {seconds / n * 1000:9.3f} ms/image")

    data_dir = output_dir(args.test_size, args.val_size)
    with open(data_dir / "manifest.csv") as f:
        test_sources = [r["source"] for r in csv.DictReader(f) if r["split"] == "test"]
    paths = {p.name: p for p in SOURCE_DIR.rglob("*") if p.is_file()}
    _, t = timed(lambda: [enhance(paths[s]) for s in test_sources], torch.device("cpu"))
    add("1. Preprocessing", "NLM + ACLAHE + resize", "test", t, len(test_sources))

    with open(FEATURE_DIR / "Results" / name / "training_log.csv") as f:
        cnn_train = sum(float(r["seconds"]) for r in csv.DictReader(f))
    add("2. Feature extraction", "ConvNeXt-ECA", "training", cnn_train, n_tr)
    model = ConvNeXtECA(args.arch, len(CLASSES), pretrained=False).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(FEATURE_DIR / "Models" / f"convnext_eca_{name}.pt", map_location=device))
    images, _, _ = load_split(data_dir, "test")
    extract(model, images[:64], device)
    _, t = timed(lambda: extract(model, images, device), device)
    add("2. Feature extraction", "ConvNeXt-ECA", "test", t, len(images))
    del model
    torch.cuda.empty_cache()

    base_args = SimpleNamespace(lr=1e-3, epochs=100, batch_size=args.batch_size)
    for label in ("768 -> 3", "768 -> 8 -> 3"):
        clf = baseline.make_model(label, d["X_train"].shape[1], len(CLASSES)).to(device)
        _, t = timed(lambda: baseline.train(clf, d["X_train"], d["y_train"], base_args, 0, device), device)
        add("3. Classifier", label, "training", t, n_tr)
        baseline.predict(clf, d["X_test"], device)
        _, t = timed(lambda: baseline.predict(clf, d["X_test"], device), device)
        add("3. Classifier", label, "test", t, n_te)

    (clf, *_), t = timed(lambda: fit(d, args, 0, device, verbose=False), device)
    add("3. Classifier", QUANTUM, "training", t, n_tr)
    X_tr = torch.from_numpy(d["X_train"]).to(device)
    y_tr = torch.from_numpy(d["y_train"]).long().to(device)
    X_te = torch.from_numpy(d["X_test"]).to(device)

    @torch.no_grad()
    def predict():
        return clf(X_te, X_tr, y_tr).argmax(1).cpu().numpy()

    predict()
    _, t = timed(predict, device)
    add("3. Classifier", QUANTUM, "test", t, n_te)

    out_dir = QUANTUM_DIR / "Results" / f"evaluation_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_dir / 'timing.csv'}")


if __name__ == "__main__":
    main()
