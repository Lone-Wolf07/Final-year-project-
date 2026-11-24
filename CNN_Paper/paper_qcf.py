import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torchvision import models

PAPER_DIR = Path(__file__).resolve().parent
ROOT = PAPER_DIR.parent
sys.path.insert(0, str(ROOT / "CNN" / "Preprocessing"))
sys.path.insert(0, str(ROOT / "CNN" / "Feature_Extraction"))
sys.path.insert(0, str(ROOT / "CNN" / "Quantum_Classifier"))
from adabelief import AdaBelief  # noqa: E402
from circuit import ry, rz  # noqa: E402
from data import scores  # noqa: E402
from model import ECA  # noqa: E402
from preprocessing import augment, enhance, z_score  # noqa: E402

RAW_DIR = ROOT / "Datasets" / "IQ-OTH_NCCD" / "Data"
FOLDERS = {"benign": "Bengin cases", "malignant": "Malignant cases", "normal": "Normal cases"}
CLASSES = ["benign", "malignant", "normal"]
SEED = 42

EPOCHS, BATCH_SIZE, LR, BETAS, EPS = 50, 32, 1e-4, (0.9, 0.999), 1e-8
N_QUBITS = 8
ECA_KERNEL = 3
BENIGN_EXTRA = 360
EVAL_EVERY = 5
EVAL_BATCH = 32


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1, stochastic_depth_prob=0.0)
        self.stages, self.eca, self.pool, self.norm = net.features, ECA(ECA_KERNEL), nn.AdaptiveAvgPool2d(1), net.classifier[0]
        self.n_features = net.classifier[2].in_features

    def forward(self, x):
        return self.norm(self.pool(self.eca(self.stages(x)))).flatten(1)


def crz_ring_phases(phi, n_qubits):
    idx = torch.arange(2 ** n_qubits, device=phi.device)
    total = torch.zeros(2 ** n_qubits, device=phi.device)
    for q in range(n_qubits):
        c, t = (idx >> q) & 1, (idx >> ((q + 1) % n_qubits)) & 1
        total = total + c * torch.where(t == 1, phi[q] / 2, -phi[q] / 2)
    return torch.polar(torch.ones_like(total), total)


class PaperQFC(nn.Module):
    def __init__(self, n_features, n_qubits=N_QUBITS):
        super().__init__()
        self.n_qubits = n_qubits
        self.proj = nn.Linear(n_features, n_qubits)
        self.theta = nn.Parameter(torch.empty(n_qubits, 2).uniform_(-math.pi, math.pi))
        self.phi = nn.Parameter(torch.empty(n_qubits).uniform_(-math.pi, math.pi))

    def apply(self, state, gate, q):
        n = state.shape[0]
        state = state.view(n, 2 ** (self.n_qubits - 1 - q), 2, 2 ** q)
        eq = "nab,nxbz->nxaz" if gate.dim() == 3 else "ab,nxbz->nxaz"
        return torch.einsum(eq, gate, state).reshape(n, -1)

    def encode(self, f):
        z = self.proj(f)
        state = torch.zeros(len(z), 2 ** self.n_qubits, dtype=torch.complex64, device=z.device)
        state[:, 0] = 1
        for q in range(self.n_qubits):
            state = self.apply(state, ry(z[:, q]), q)
        for q in range(self.n_qubits):
            state = self.apply(state, ry(self.theta[q, 0]), q)
            state = self.apply(state, rz(self.theta[q, 1]), q)
        return state * crz_ring_phases(self.phi, self.n_qubits)

    @staticmethod
    def class_scores(states, ref_states, ref_labels, exclude_self=False):
        fid = (states @ ref_states.conj().T).abs() ** 2
        keep = torch.ones_like(fid)
        if exclude_self:
            keep = keep - torch.eye(len(fid), device=fid.device)
        onehot = F.one_hot(ref_labels, len(CLASSES)).float()
        return ((fid * keep) @ onehot) / (keep @ onehot).clamp(min=1)


def load_raw():
    items = []
    for label in CLASSES:
        items += [(p, label) for p in sorted((RAW_DIR / FOLDERS[label]).iterdir())
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return items


def enhanced_cache(items):
    path = PAPER_DIR / "Cache" / "enhanced_1097.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(exist_ok=True)
    print(f"Preprocessing {len(items)} raw images (NLM + ACLAHE + 224x224)", flush=True)
    imgs = np.stack([r[0] for r in Parallel(n_jobs=-1, prefer="threads")(delayed(enhance)(p) for p, _ in items)])
    np.save(path, imgs)
    return imgs


def benign_copies(imgs, n_extra, rng):
    copies = []
    for k in range(n_extra):
        copies.append(augment(imgs[k % len(imgs)].astype(np.float32), rng))
    return np.stack(copies)


def build_dataset(test_size, mode):
    items = load_raw()
    enhanced = enhanced_cache(items)
    labels = np.array([CLASSES.index(lbl) for _, lbl in items])
    names = np.array([p.name for p, _ in items])
    rng = np.random.default_rng(SEED)
    b = CLASSES.index("benign")
    if mode == "before":
        extra = benign_copies(enhanced[labels == b], BENIGN_EXTRA, rng)
        src = names[labels == b]
        x = np.concatenate([enhanced.astype(np.float32), extra])
        y = np.concatenate([labels, np.full(len(extra), b)])
        is_copy = np.r_[np.zeros(len(enhanced), bool), np.ones(len(extra), bool)]
        origin = np.concatenate([names, [src[k % len(src)] for k in range(len(extra))]])
        tr, te = train_test_split(np.arange(len(x)), test_size=test_size, stratify=y, random_state=SEED)
    else:
        tr0, te = train_test_split(np.arange(len(enhanced)), test_size=test_size, stratify=labels, random_state=SEED)
        ben = tr0[labels[tr0] == b]
        n_extra = round(BENIGN_EXTRA * len(ben) / (labels == b).sum())
        extra = benign_copies(enhanced[ben], n_extra, rng)
        x = np.concatenate([enhanced.astype(np.float32), extra])
        y = np.concatenate([labels, np.full(len(extra), b)])
        is_copy = np.r_[np.zeros(len(enhanced), bool), np.ones(len(extra), bool)]
        origin = np.concatenate([names, [names[ben][k % len(ben)] for k in range(len(extra))]])
        tr = np.r_[tr0, np.arange(len(enhanced), len(x))]
    x = np.stack([z_score(img.astype(np.float32))[0] for img in x])
    return dict(x=x, y=y, is_copy=is_copy, origin=origin, train=tr, test=te, items=items)


def to_batch(images, device):
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)[:, None].expand(-1, 3, -1, -1)
    return x.contiguous(memory_format=torch.channels_last)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def features(cnn, images, device):
    cnn.eval()
    out = []
    for b in range(0, len(images), EVAL_BATCH):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(cnn(to_batch(images[b:b + EVAL_BATCH], device)).float())
    return torch.cat(out)


@torch.no_grad()
def evaluate(cnn, qfc, d, device):
    qfc.eval()
    x, y = d["x"], torch.from_numpy(d["y"]).long().to(device)
    tr, te = d["train"], d["test"]
    ref = qfc.encode(features(cnn, x[tr], device))
    train_probs = torch.softmax(qfc.class_scores(ref, ref, y[tr], exclude_self=True), 1)
    test_states = qfc.encode(features(cnn, x[te], device))
    test_probs = torch.softmax(qfc.class_scores(test_states, ref, y[tr]), 1)
    return train_probs.cpu().numpy(), test_probs.cpu().numpy()


def full_metrics(y, probs):
    pred = probs.argmax(1)
    m = {k: round(v * 100, 2) for k, v in scores(y, pred).items()}
    m["auc"] = round(roc_auc_score(y, probs, multi_class="ovr", average="macro") * 100, 2)
    for c, r in zip(CLASSES, recall_score(y, pred, average=None, labels=range(3), zero_division=0)):
        m[f"recall_{c}"] = round(r * 100, 2)
    return m


def train_step(cnn, qfc, optimizer, xb, yb, micro, device):
    cnn.train()
    qfc.train()
    chunks = [xb[i:i + micro] for i in range(0, len(xb), micro)]
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        feats = torch.cat([cnn(to_batch(c, device)).float() for c in chunks])
    feats.requires_grad_(True)
    states = qfc.encode(feats)
    logits = qfc.class_scores(states, states, yb, exclude_self=True)
    loss = F.cross_entropy(logits, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = feats.grad.split(micro)
    for c, g in zip(chunks, grads):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = cnn(to_batch(c, device)).float()
        f.backward(g)
    circuit_grad = math.sqrt(sum(float(p.grad.norm() ** 2) for p in (qfc.theta, qfc.phi) if p.grad is not None))
    optimizer.step()
    correct = (logits.argmax(1) == yb).sum().item()
    return loss.item(), correct, circuit_grad


def save_figure(history, cm, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["loss"] for h in history])
    axes[0].set(title="Training loss (cross-entropy on softmax(S_c))", xlabel="epoch")
    axes[1].plot(ep, [h["batch_train_acc"] for h in history], label="train (in-batch)")
    ev = [h for h in history if h.get("test_acc") is not None]
    axes[1].plot([h["epoch"] for h in ev], [h["test_acc"] for h in ev], "o-", label="test")
    axes[1].set(title="Accuracy (%)", xlabel="epoch")
    axes[1].legend()
    axes[2].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[2].set(title=title, xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--mode", choices=["before", "after"], default="before",
                        help="augment benign before the split (Table 2) or after it (the text's claim)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--single", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"paper_{100 - round(args.test_size * 100)}_{round(args.test_size * 100)}_aug{args.mode}_e{args.epochs}"
    out = PAPER_DIR / "Results" / name
    out.mkdir(parents=True, exist_ok=True)
    (PAPER_DIR / "Models").mkdir(exist_ok=True)

    d = build_dataset(args.test_size, args.mode)
    tr, te = d["train"], d["test"]
    print(f"{name}: {len(d['x'])} images, {len(tr)} train / {len(te)} test "
          f"(test benign copies {int(d['is_copy'][te].sum())})", flush=True)

    cnn = Backbone().to(device).to(memory_format=torch.channels_last)
    qfc = PaperQFC(cnn.n_features).to(device)
    print(f"ConvNeXt-Base + ECA(k={ECA_KERNEL}): {cnn.n_features} features, "
          f"{sum(p.numel() for p in cnn.parameters()) + sum(p.numel() for p in qfc.parameters()):,} parameters", flush=True)
    optimizer = AdaBelief(list(cnn.parameters()) + list(qfc.parameters()), lr=LR, betas=BETAS, eps=EPS)
    y_all = torch.from_numpy(d["y"]).long().to(device)

    rng = np.random.default_rng(SEED)
    history, train_s = [], 0.0
    for epoch in range(1, args.epochs + 1):
        sync(device)
        start = time.perf_counter()
        order = rng.permutation(tr)
        total, correct, cg = 0.0, 0, []
        for b in range(0, len(order), BATCH_SIZE):
            idx = order[b:b + BATCH_SIZE]
            if len(idx) < 2:
                continue
            loss, c, g = train_step(cnn, qfc, optimizer, d["x"][idx], y_all[idx], args.micro_batch, device)
            total += loss * len(idx)
            correct += c
            cg.append(g)
        sync(device)
        sec = time.perf_counter() - start
        train_s += sec
        h = dict(epoch=epoch, loss=round(total / len(order), 5), batch_train_acc=round(correct / len(order) * 100, 2),
                 circuit_grad_norm=float(f"{np.mean(cg):.3e}"), seconds=round(sec, 1))
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            _, te_probs = evaluate(cnn, qfc, d, device)
            h["test_acc"] = round(float((te_probs.argmax(1) == d["y"][te]).mean() * 100), 2)
        history.append(h)
        print(f"epoch {epoch:>3}/{args.epochs}  loss {h['loss']:.4f}  in-batch acc {h['batch_train_acc']:6.2f}"
              + (f"  test acc {h['test_acc']:6.2f}" if "test_acc" in h else "")
              + f"  circuit grad {h['circuit_grad_norm']:.1e}  ({h['seconds']}s)", flush=True)
    write_csv(out / "training_log.csv", history)
    torch.save(dict(cnn=cnn.state_dict(), qfc=qfc.state_dict(), args=vars(args)), PAPER_DIR / "Models" / f"{name}.pt")

    sync(device)
    t0 = time.perf_counter()
    tr_probs, te_probs = evaluate(cnn, qfc, d, device)
    sync(device)
    t_eval = time.perf_counter() - t0
    y_te, y_tr = d["y"][te], d["y"][tr]
    rows = [dict(split="train", images=len(tr), **full_metrics(y_tr, tr_probs)),
            dict(split="test", images=len(te), **full_metrics(y_te, te_probs))]
    orig = ~d["is_copy"][te]
    rows.append(dict(split="test (original images only)", images=int(orig.sum()),
                     **full_metrics(y_te[orig], te_probs[orig])))
    leaked = d["is_copy"][te] & np.isin(d["origin"][te], d["origin"][tr])
    rows.append(dict(split="test benign copies whose source is in train", images=int(leaked.sum())))
    p = te_probs.max(1)
    rows.append(dict(split="test: mean top probability", images=len(te), accuracy=round(float(p.mean()) * 100, 2)))

    with torch.no_grad():
        ref = qfc.eval().encode(features(cnn, d["x"][tr], device))
    with torch.no_grad():
        qfc.class_scores(qfc.encode(features(cnn, d["x"][te[:1]], device)), ref, y_all[tr])
    single = []
    for i in te[:args.single]:
        path = d["items"][i][0] if i < len(d["items"]) else None
        if path is None:
            continue
        sync(device)
        a = time.perf_counter()
        img = z_score(enhance(path)[0])[0][None]
        b_ = time.perf_counter()
        with torch.no_grad():
            s = qfc.encode(features(cnn, img, device))
            qfc.class_scores(s, ref, y_all[tr])
        sync(device)
        single.append(((b_ - a) * 1000, (time.perf_counter() - b_) * 1000))
    single = np.array(single)
    timing = [dict(step="Training (all epochs, ConvNeXt-Base + ECA + quantum end to end)", seconds=round(train_s, 1)),
              dict(step=f"Batched inference, train refs + {len(te)} test images", seconds=round(t_eval, 2)),
              dict(step="One new image: preprocessing", ms=round(single[:, 0].mean(), 1)),
              dict(step="One new image: ConvNeXt-Base + quantum", ms=round(single[:, 1].mean(), 1)),
              dict(step="One new image: total", ms=round(single.sum(1).mean(), 1))]

    write_csv(out / "metrics.csv", rows)
    write_csv(out / "timing.csv", timing)
    cm = confusion_matrix(y_te, te_probs.argmax(1), labels=range(3))
    save_figure(history, cm, f"{name}\ntest accuracy {rows[1]['accuracy']}%", out / "results.png")

    print("\nResults (%):")
    for r in rows:
        print("  " + ", ".join(f"{k} {v}" for k, v in r.items()))
    print("Confusion matrix (rows actual, cols predicted: benign, malignant, normal):")
    print(cm)
    print("Times:")
    for r in timing:
        print("  " + ", ".join(f"{k} {v}" for k, v in r.items()))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
