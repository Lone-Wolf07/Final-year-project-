import argparse
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from circuit import N_LAYERS, PQC, fidelity
from data import load_features, scores, source_groups
from projection import N_QUBITS, Projection

INIT_TEMPERATURE = 10.0


class FidelityClassifier(nn.Module):
    def __init__(self, n_features=768, n_classes=3, n_qubits=N_QUBITS, n_layers=N_LAYERS,
                 learn_temperature=True, temperature=INIT_TEMPERATURE):
        super().__init__()
        self.n_classes = n_classes
        self.proj = Projection(n_features, n_qubits)
        self.pqc = PQC(n_qubits, n_layers)
        log_t = torch.tensor(math.log(temperature))
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_t)
        else:
            self.register_buffer("log_temperature", log_t)

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def encode(self, f):
        return self.pqc(self.proj(f))

    def class_scores(self, states, ref_states, ref_labels, exclude=None):
        fid = fidelity(states, ref_states)
        keep = torch.ones_like(fid) if exclude is None else (~exclude).float()
        onehot = F.one_hot(ref_labels, self.n_classes).float()
        return ((fid * keep) @ onehot) / ((keep @ onehot).clamp(min=1))

    def forward(self, f, ref_f, ref_labels, exclude=None):
        states = self.encode(f)
        ref_states = states if ref_f is f else self.encode(ref_f)
        return self.temperature * self.class_scores(states, ref_states, ref_labels, exclude)


def same_source(groups_a, groups_b):
    return groups_a[:, None] == groups_b[None, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_features(args.test_size, args.arch, args.val_size)
    classes = d["classes"]
    X_tr = torch.from_numpy(d["X_train"]).to(device)
    y_tr = torch.from_numpy(d["y_train"]).long().to(device)
    X_te = torch.from_numpy(d["X_test"]).to(device)
    groups = torch.from_numpy(source_groups(d["files_train"])).to(device)
    exclude = same_source(groups, groups)
    copies = exclude.sum(1) - 1
    print(f"training images compared with all others except themselves; "
          f"{(copies > 0).sum().item()} also skip their {copies.max().item()} augmented copies")

    model = FidelityClassifier(X_tr.shape[1], len(classes)).to(device)
    n_params = {name: sum(p.numel() for p in mod.parameters()) for name, mod in
                (("projection", model.proj), ("circuit", model.pqc))}
    print(f"parameters: projection {n_params['projection']}, circuit {n_params['circuit']}, temperature 1")

    with torch.no_grad():
        ref = model.encode(X_tr)
        S_te = model.class_scores(model.encode(X_te[:5]), ref, y_tr)
        S_self = model.class_scores(ref[:3], ref, y_tr)
        S_excl = model.class_scores(ref[:3], ref, y_tr, exclude[:3])
    print("\nclass-average fidelity S_c for 5 test images (benign, malignant, normal):")
    print(np.round(S_te.cpu().numpy(), 4))
    print("probabilities with the plain softmax (temperature 1):")
    print(np.round(torch.softmax(S_te, 1).cpu().numpy(), 3))
    print(f"probabilities with temperature {INIT_TEMPERATURE:g}:")
    print(np.round(torch.softmax(INIT_TEMPERATURE * S_te, 1).cpu().numpy(), 3))
    print(f"row sums: {torch.softmax(INIT_TEMPERATURE * S_te, 1).sum(1).cpu().numpy().round(6)}")
    print("\n3 training images scored against the training set, own class column:")
    own = [classes[c] for c in d["y_train"][:3]]
    for i in range(3):
        c = int(d["y_train"][i])
        print(f"  {own[i]:<9} including itself {S_self[i, c]:.4f}   excluding itself and copies {S_excl[i, c]:.4f}")

    print("\ngradient check (one batch of 32, cross-entropy):")
    for learn, t in ((False, 1.0), (True, INIT_TEMPERATURE)):
        torch.manual_seed(0)
        m = FidelityClassifier(X_tr.shape[1], len(classes), learn_temperature=learn, temperature=t).to(device)
        idx = torch.arange(32, device=device)
        logits = m(X_tr[idx], X_tr, y_tr, exclude[idx])
        loss = F.cross_entropy(logits, y_tr[idx])
        loss.backward()
        grads = {"projection": m.proj.linear.weight.grad.norm().item(), "circuit": m.pqc.theta.grad.norm().item()}
        if learn:
            grads["temperature"] = m.log_temperature.grad.abs().item()
        print(f"  temperature {t:>4g}: loss {loss.item():.4f}  "
              + "  ".join(f"{k} grad {v:.2e}" for k, v in grads.items()))

    with torch.no_grad():
        pred = (model.temperature * model.class_scores(model.encode(X_te), ref, y_tr)).argmax(1).cpu().numpy()
    s = scores(d["y_test"], pred)
    print(f"\nuntrained test accuracy {s['accuracy']:.2%} "
          f"(predicted counts: {dict(zip(classes, np.bincount(pred, minlength=len(classes)).tolist()))})")


if __name__ == "__main__":
    main()
