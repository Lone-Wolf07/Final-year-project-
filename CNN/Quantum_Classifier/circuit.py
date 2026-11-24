import argparse
import math
import time

import numpy as np
import torch
from torch import nn

from data import QUANTUM_DIR, load_features, tag
from projection import N_QUBITS, Projection

N_LAYERS = 2


def ry(a):
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)


def rz(a):
    zero = torch.zeros_like(a)
    neg, pos = torch.polar(torch.ones_like(a), -a / 2), torch.polar(torch.ones_like(a), a / 2)
    return torch.stack([torch.stack([neg, zero.to(neg.dtype)], -1), torch.stack([zero.to(neg.dtype), pos], -1)], -2)


def cnot_ring_permutation(n_qubits):
    positions = np.arange(2 ** n_qubits)
    perm = positions.copy()
    for q in range(n_qubits):
        control, target = q, (q + 1) % n_qubits
        perm = perm[np.where((positions >> control) & 1, positions ^ (1 << target), positions)]
    return torch.from_numpy(perm)


# Qubit q is bit q of the state index, Qiskit's ordering.
class PQC(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, final_encoding=True):
        super().__init__()
        self.n_qubits, self.n_layers, self.final_encoding = n_qubits, n_layers, final_encoding
        self.theta = nn.Parameter(torch.empty(n_layers, n_qubits, 2).uniform_(-math.pi, math.pi))
        self.register_buffer("cnot_ring", cnot_ring_permutation(n_qubits))

    def apply(self, state, gate, q):
        n = state.shape[0]
        state = state.view(n, 2 ** (self.n_qubits - 1 - q), 2, 2 ** q)
        if gate.dim() == 3:
            state = torch.einsum("nab,nxbz->nxaz", gate, state)
        else:
            state = torch.einsum("ab,nxbz->nxaz", gate, state)
        return state.reshape(n, -1)

    def forward(self, angles):
        n = angles.shape[0]
        state = torch.zeros(n, 2 ** self.n_qubits, dtype=torch.complex64, device=angles.device)
        state[:, 0] = 1
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                state = self.apply(state, ry(angles[:, q]), q)
                state = self.apply(state, rz(self.theta[layer, q, 0]), q)
                state = self.apply(state, ry(self.theta[layer, q, 1]), q)
            state = state[:, self.cnot_ring]
        # Gates after the last encoding cancel in the fidelity, so the circuit ends on an encoding.
        if self.final_encoding:
            for q in range(self.n_qubits):
                state = self.apply(state, ry(angles[:, q]), q)
        return state


def fidelity(a, b):
    return (a @ b.conj().T).abs() ** 2


def qiskit_circuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, angles=None, theta=None):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = angles if angles is not None else ParameterVector("x", n_qubits)
    t = theta if theta is not None else ParameterVector("θ", n_layers * n_qubits * 2)
    qc = QuantumCircuit(n_qubits)
    for layer in range(n_layers):
        for q in range(n_qubits):
            k = (layer * n_qubits + q) * 2
            qc.ry(float(x[q]) if angles is not None else x[q], q)
            qc.rz(float(t[layer][q][0]) if theta is not None else t[k], q)
            qc.ry(float(t[layer][q][1]) if theta is not None else t[k + 1], q)
        qc.barrier()
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
        qc.barrier()
    for q in range(n_qubits):
        qc.ry(float(x[q]) if angles is not None else x[q], q)
    return qc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    args = parser.parse_args()

    from qiskit.quantum_info import Statevector

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_features(args.test_size, args.arch, args.val_size)
    proj = Projection(d["X_train"].shape[1]).to(device)
    pqc = PQC(n_layers=args.layers).to(device)
    X = torch.from_numpy(d["X_train"]).to(device)

    with torch.no_grad():
        angles = proj(X)
        states = pqc(angles)
    print(f"{args.layers} layer(s), {pqc.theta.numel()} circuit parameters, "
          f"angles {tuple(angles.shape)} -> states {tuple(states.shape)}")
    print(f"state norms: min {states.abs().pow(2).sum(1).min():.6f}, max {states.abs().pow(2).sum(1).max():.6f}")

    diffs = []
    for i in range(5):
        qc = qiskit_circuit(n_layers=args.layers, angles=angles[i].cpu().numpy(), theta=pqc.theta.detach().cpu().numpy())
        diffs.append(np.abs(Statevector.from_instruction(qc).data - states[i].cpu().numpy()).max())
    print(f"max difference from Qiskit's Statevector on 5 images: {max(diffs):.2e}")

    F = fidelity(states[:6], states[:6]).cpu().numpy()
    print(f"self-fidelity (diagonal): {np.round(np.diag(F), 6)}")
    print("fidelity between the first 6 training images:")
    print(np.round(F, 3))

    for n_layers, final in ((1, False), (2, False), (1, True), (2, True)):
        torch.manual_seed(1)
        a = PQC(n_layers=n_layers, final_encoding=final).to(device)
        b = PQC(n_layers=n_layers, final_encoding=final).to(device)
        with torch.no_grad():
            fa = fidelity(a(angles[:50]), a(angles[:50]))
            dead = 0
            for layer in range(n_layers):
                c = PQC(n_layers=n_layers, final_encoding=final).to(device)
                c.theta.data = a.theta.data.clone()
                c.theta.data[layer] = b.theta.data[layer]
                if (fidelity(c(angles[:50]), c(angles[:50])) - fa).abs().max() < 1e-4:
                    dead += 1
        print(f"{n_layers} layer(s), final encoding {'on ' if final else 'off'}: "
              f"{n_layers - dead}/{n_layers} trainable layers change the fidelity")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        pqc(angles)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"simulating all {len(angles)} training images: {(time.time() - start) * 1000:.1f} ms on {device}")

    out_dir = QUANTUM_DIR / "Results" / f"circuit_{tag(args.test_size, args.arch, args.val_size)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc = qiskit_circuit(n_layers=args.layers)
    (out_dir / "circuit.txt").write_text(str(qc.draw("text", fold=-1)), encoding="utf-8")
    try:
        qc.draw("mpl", fold=-1, filename=str(out_dir / "circuit.png"))
        print(f"Saved {out_dir / 'circuit.png'} and circuit.txt")
    except Exception as e:
        print(f"Saved {out_dir / 'circuit.txt'} (PNG drawing unavailable: {e})")


if __name__ == "__main__":
    main()
