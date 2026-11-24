import argparse
import csv
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from circuit import N_LAYERS
from data import QUANTUM_DIR, load_features, scores, source_groups, tag
from fidelity import INIT_TEMPERATURE, FidelityClassifier, same_source

sys.path.insert(0, str(QUANTUM_DIR.parent / "Feature_Extraction"))
from adabelief import AdaBelief  # noqa: E402


@torch.no_grad()
def evaluate(model, X, y, X_ref, y_ref, exclude=None):
    model.eval()
    logits = model(X, X_ref, y_ref, exclude)
    loss = F.cross_entropy(logits, y).item()
    return loss, logits.argmax(1).cpu().numpy(), torch.softmax(logits, 1).cpu().numpy()


def fit(d, args, seed, device, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X_tr = torch.from_numpy(d["X_train"]).to(device)
    y_tr = torch.from_numpy(d["y_train"]).long().to(device)
    X_te = torch.from_numpy(d["X_test"]).to(device)
    y_te = torch.from_numpy(d["y_test"]).long().to(device)
    has_val = "X_val" in d
    if has_val:
        X_va = torch.from_numpy(d["X_val"]).to(device)
        y_va = torch.from_numpy(d["y_val"]).long().to(device)
    groups = torch.from_numpy(source_groups(d["files_train"])).to(device)
    exclude = same_source(groups, groups)

    model = FidelityClassifier(X_tr.shape[1], len(d["classes"]), n_layers=args.layers,
                               learn_temperature=not args.fixed_temperature, temperature=args.temperature).to(device)
    optimizer = AdaBelief(model.parameters(), lr=args.lr)

    def snapshot(epoch, train_loss):
        _, tr_pred, _ = evaluate(model, X_tr, y_tr, X_tr, y_tr, exclude)
        te_loss, te_pred, _ = evaluate(model, X_te, y_te, X_tr, y_tr)
        row = dict(epoch=epoch, train_loss=round(train_loss, 5), test_loss=round(te_loss, 5),
                   train_acc=round(scores(d["y_train"], tr_pred)["accuracy"] * 100, 2),
                   test_acc=round(scores(d["y_test"], te_pred)["accuracy"] * 100, 2),
                   temperature=round(model.temperature.item(), 3))
        if has_val:
            va_loss, va_pred, _ = evaluate(model, X_va, y_va, X_tr, y_tr)
            row.update(val_loss=round(va_loss, 5), val_acc=round(scores(d["y_val"], va_pred)["accuracy"] * 100, 2))
        if verbose:
            print(f"epoch {epoch:>3}  train loss {row['train_loss']:.4f}  test loss {row['test_loss']:.4f}  "
                  f"train acc {row['train_acc']:6.2f}  test acc {row['test_acc']:6.2f}  "
                  + (f"val loss {row['val_loss']:.4f}  val acc {row['val_acc']:6.2f}  " if has_val else "")
                  + f"T {row['temperature']:.2f}")
        return row

    with torch.no_grad():
        init_loss = F.cross_entropy(model(X_tr, X_tr, y_tr, exclude), y_tr).item()
    log = [snapshot(0, init_loss)]
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(X_tr))).to(device)
        total = 0.0
        for b in range(0, len(X_tr), args.batch_size):
            idx = order[b:b + args.batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X_tr[idx], X_tr, y_tr, exclude[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        log.append(snapshot(epoch, total / len(X_tr)))

    _, tr_pred, _ = evaluate(model, X_tr, y_tr, X_tr, y_tr, exclude)
    _, te_pred, te_prob = evaluate(model, X_te, y_te, X_tr, y_tr)
    return model, log, tr_pred, te_pred, te_prob


def add_args(parser):
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--temperature", type=float, default=INIT_TEMPERATURE)
    parser.add_argument("--fixed-temperature", action="store_true")
    return parser


def main():
    parser = add_args(argparse.ArgumentParser())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_features(args.test_size, args.arch, args.val_size)
    classes = d["classes"]
    name = f"fidelity_{tag(args.test_size, args.arch, args.val_size)}"
    out_dir = QUANTUM_DIR / "Results" / name
    model_dir = QUANTUM_DIR / "Models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    print(f"{len(d['X_train'])} train / {len(d['X_test'])} test, {args.layers} layers, lr {args.lr:g}, "
          f"batch {args.batch_size}, {args.epochs} epochs on {device}")

    start = time.time()
    model, log, tr_pred, te_pred, _ = fit(d, args, args.seed, device)
    print(f"training took {time.time() - start:.0f} s")

    torch.save(dict(state_dict=model.state_dict(), args=vars(args), classes=classes), model_dir / f"{name}.pt")
    with open(out_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)

    metrics = []
    for split, y, pred in (("train", d["y_train"], tr_pred), ("test", d["y_test"], te_pred)):
        metrics.append(dict(split=split, **{k: round(v * 100, 2) for k, v in scores(y, pred).items()}))
        print(f"{split:<5} " + "  ".join(f"{k} {v}" for k, v in metrics[-1].items() if k != "split"))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    epochs = [r["epoch"] for r in log]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [r["train_loss"] for r in log], label="train")
    axes[0].plot(epochs, [r["test_loss"] for r in log], label="test")
    axes[0].set(title="Cross-entropy loss", xlabel="epoch")
    axes[0].legend()
    axes[1].plot(epochs, [r["train_acc"] for r in log], label="train")
    axes[1].plot(epochs, [r["test_acc"] for r in log], label="test")
    axes[1].set(title="Accuracy (%)", xlabel="epoch")
    axes[1].legend()
    axes[2].plot(epochs, [r["temperature"] for r in log])
    axes[2].set(title="Temperature", xlabel="epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=100)
    plt.close(fig)
    print(f"Saved {model_dir / f'{name}.pt'} and results to {out_dir}")


if __name__ == "__main__":
    main()
