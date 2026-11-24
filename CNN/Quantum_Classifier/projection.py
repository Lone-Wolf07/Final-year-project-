import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from data import QUANTUM_DIR, load_features, tag

N_QUBITS = 8
SATURATION = 0.95


class Projection(nn.Module):
    def __init__(self, n_features=768, n_qubits=N_QUBITS):
        super().__init__()
        self.linear = nn.Linear(n_features, n_qubits)

    def forward(self, f):
        return math.pi * torch.tanh(self.linear(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    args = parser.parse_args()

    torch.manual_seed(0)
    d = load_features(args.test_size, args.arch, args.val_size)
    proj = Projection(d["X_train"].shape[1])
    with torch.no_grad():
        angles = proj(torch.from_numpy(d["X_train"]))

    n_params = sum(p.numel() for p in proj.parameters())
    print(f"weights {tuple(proj.linear.weight.shape)} + bias {tuple(proj.linear.bias.shape)} = {n_params} parameters")
    print(f"input {tuple(d['X_train'].shape)} -> angles {tuple(angles.shape)}")
    print(f"angle range [{angles.min():.3f}, {angles.max():.3f}], allowed [{-math.pi:.3f}, {math.pi:.3f}]")
    saturated = (angles.abs() > SATURATION * math.pi).float().mean()
    print(f"angles within {1 - SATURATION:.0%} of ±π (saturated tanh): {saturated:.2%}")
    for q in range(angles.shape[1]):
        a = angles[:, q]
        print(f"  qubit {q}: mean {a.mean():+.3f}  std {a.std():.3f}  min {a.min():+.3f}  max {a.max():+.3f}")

    out_dir = QUANTUM_DIR / "Results" / f"projection_{tag(args.test_size, args.arch, args.val_size)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for q, ax in enumerate(axes.flat):
        ax.hist(angles[:, q].numpy(), bins=40, range=(-math.pi, math.pi))
        ax.set_title(f"qubit {q}")
    fig.suptitle("Encoding angles of the training images (untrained projection)")
    fig.supxlabel("angle (radians)")
    fig.tight_layout()
    fig.savefig(out_dir / "angle_histograms.png", dpi=100)
    plt.close(fig)
    print(f"Saved {out_dir / 'angle_histograms.png'}")


if __name__ == "__main__":
    main()
