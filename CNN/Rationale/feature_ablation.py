import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import recall_score
from torch import nn
from torchvision import models

HERE = Path(__file__).resolve().parent
CNN_DIR = HERE.parent
for sub in ("Quantum_Classifier", "Feature_Extraction", "Preprocessing", "ResNet50"):
    sys.path.insert(0, str(CNN_DIR / sub))
import baseline  # noqa: E402
import extract_features as fe  # noqa: E402
from data import scores  # noqa: E402
from model import ConvNeXtECA  # noqa: E402
from preprocessing import CLASSES, output_dir  # noqa: E402
from resnet50 import build as build_resnet  # noqa: E402
from train import fit  # noqa: E402

OUT = HERE / "Results"
MODELS = HERE / "Models"
ARCHIVE = CNN_DIR / "Archive_50_epochs" / "Feature_Extraction" / "Models" / "convnext_eca_split_80_20_tiny.pt"
QUANTUM = SimpleNamespace(epochs=50, batch_size=32, lr=1e-4, layers=2, temperature=10.0, fixed_temperature=False)
LINEAR = SimpleNamespace(epochs=100, batch_size=32, lr=1e-3)
SEEDS = [0, 1, 2, 3, 4]


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})


def load_data():
    data_dir = output_dir(0.2)
    tr_x, tr_y, tr_rows = fe.load_split(data_dir, "train")
    te_x, te_y, _ = fe.load_split(data_dir, "test")
    return tr_x, tr_y, [r["file"] for r in tr_rows], te_x, te_y


def gradient_accumulation_check(tr_x, tr_y, device):
    rows = []
    device = torch.device("cpu")
    x = torch.from_numpy(np.ascontiguousarray(tr_x[:32]))[:, None].expand(-1, 3, -1, -1).contiguous().double()
    y = torch.from_numpy(tr_y[:32])
    for name, make in (("ConvNeXt-Tiny + ECA (LayerNorm)", lambda: ConvNeXtECA("tiny", 3)),
                       ("ResNet50 (BatchNorm)", build_resnet)):
        torch.manual_seed(0)
        net = make().to(device).double()
        net.train()
        for m in net.modules():
            if m.__class__.__name__ == "StochasticDepth" or isinstance(m, nn.Dropout):
                m.eval()
        state = {k: v.clone() for k, v in net.state_dict().items()}

        def grads(chunks):
            net.load_state_dict(state)
            net.zero_grad()
            for idx in np.array_split(np.arange(32), chunks):
                loss = nn.functional.cross_entropy(net(x[idx]), y[idx]) * len(idx) / 32
                loss.backward()
            return torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]).clone()

        g32, g16 = grads(1), grads(2)
        rel = ((g32 - g16).norm() / g32.norm()).item()
        rows.append(dict(network=name, relative_gradient_difference=rel,
                         cosine_similarity=nn.functional.cosine_similarity(g32, g16, 0).item()))
        print(f"{name}: relative difference batch 32 vs 2x16 = {rel:.2e}", flush=True)
        del net
        torch.cuda.empty_cache()
    write("grad_accumulation.csv", rows)


def memory_benchmark(tr_x, tr_y, device):
    rows = []
    nets = (("ConvNeXt-Tiny", lambda: ConvNeXtECA("tiny", 3)), ("ConvNeXt-Base", lambda: ConvNeXtECA("base", 3)),
            ("ResNet50", build_resnet))
    for name, make in nets:
        for batch, amp in ((32, False), (32, True), (16, False), (16, True)):
            torch.manual_seed(0)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            row = dict(network=name, batch=batch, mixed_precision=amp)
            try:
                net = make().to(device).to(memory_format=torch.channels_last)
                opt = torch.optim.AdamW(net.parameters(), 1e-4)
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                x = fe.to_batch(tr_x[:batch], device)
                y = torch.from_numpy(tr_y[:batch]).to(device)
                times = []
                for i in range(6):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    opt.zero_grad()
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        loss = nn.functional.cross_entropy(net(x).float(), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    torch.cuda.synchronize()
                    if i >= 2:
                        times.append(time.perf_counter() - t)
                row.update(peak_memory_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                           ms_per_image=1000 * np.mean(times) / batch, status="ok")
            except torch.OutOfMemoryError:
                row.update(status="out of memory")
            row["parameters_m"] = sum(p.numel() for p in make().parameters()) / 1e6
            print(row, flush=True)
            rows.append(row)
            net = opt = x = None
            torch.cuda.empty_cache()
    write("memory_benchmark.csv", rows)


@torch.no_grad()
def eca_analysis(tr_x, te_x, te_y, device):
    net = ConvNeXtECA("tiny", 3).to(device)
    net.load_state_dict(torch.load(ARCHIVE, map_location=device))
    net.eval()
    weights = []
    for b in range(0, len(te_x), 32):
        fmap = net.stages(fe.to_batch(te_x[b:b + 32], device))
        s = torch.sigmoid(net.eca.conv(fmap.mean(dim=(2, 3))[:, None]))[:, 0]
        weights.append(s.float().cpu())
    w = torch.cat(weights).numpy()
    per_class = np.stack([w[te_y == c].mean(0) for c in range(3)])
    rows = [dict(kernel=str(np.round(net.eca.conv.weight.flatten().cpu().numpy(), 4).tolist()),
                 attention_min=w.min(), attention_max=w.max(), attention_mean=w.mean(),
                 channel_spread=w.mean(0).std(), within_image_spread=w.std(1).mean(),
                 share_below_0_5=(w.mean(0) < 0.5).mean(), share_above_0_9=(w.mean(0) > 0.9).mean(),
                 max_class_difference=np.abs(per_class[:, None] - per_class[None]).max())]
    print(rows[0], flush=True)
    write("eca_analysis.csv", rows)
    np.save(OUT / "eca_weights_test.npy", w)


@torch.no_grad()
def features(net, x, device):
    net.eval()
    out = []
    for b in range(0, len(x), 64):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(net.features(fe.to_batch(x[b:b + 64], device)).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def head_accuracy(net, x, y, device):
    net.eval()
    pred = []
    for b in range(0, len(x), 64):
        with torch.autocast("cuda", dtype=torch.float16):
            pred.append(net(fe.to_batch(x[b:b + 64], device)).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    return 100 * (pred == y).mean(), 100 * recall_score(y, pred, average=None, labels=[0, 1, 2])[0]


def probe(name, Xtr, ytr, files, Xte, yte, device, train_s=None, head=None, quantum=True):
    lin = []
    for seed in SEEDS:
        m = baseline.make_model("768 -> 3", Xtr.shape[1], 3).to(device)
        baseline.train(m, Xtr, ytr, LINEAR, seed, device)
        pred = baseline.predict(m, Xte, device)
        lin.append((100 * (pred == yte).mean(), 100 * recall_score(yte, pred, average=None, labels=[0, 1, 2])[0]))
    row = dict(variant=name, linear_probe_mean=np.mean([a for a, _ in lin]), linear_probe_std=np.std([a for a, _ in lin]),
               linear_benign_recall=np.mean([b for _, b in lin]))
    if head:
        row.update(head_accuracy=head[0], head_benign_recall=head[1])
    if train_s:
        row["train_minutes"] = train_s / 60
    if quantum:
        d = dict(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, files_train=np.array(files), classes=CLASSES)
        _, _, _, te_pred, _ = fit(d, QUANTUM, 0, device, verbose=False)
        row.update(quantum_seed0=100 * scores(yte, te_pred)["accuracy"],
                   quantum_benign_recall=100 * recall_score(yte, te_pred, average=None, labels=[0, 1, 2])[0])
    print(row, flush=True)
    return row


def retrain(name, make, tr_x, tr_y, files, te_x, te_y, device, optimiser=None):
    torch.manual_seed(42)
    net = make().to(device).to(memory_format=torch.channels_last)
    if optimiser:
        fe.AdaBelief = optimiser
    start = time.time()
    fe.train(net, tr_x, tr_y, SimpleNamespace(epochs=50, micro_batch=16), device, OUT / f"log_{name}.csv")
    t = time.time() - start
    fe.AdaBelief = ADABELIEF
    torch.save(net.state_dict(), MODELS / f"{name}.pt")
    row = probe(name, features(net, tr_x, device), tr_y, files, features(net, te_x, device), te_y, device, t,
                head_accuracy(net, te_x, te_y, device))
    del net
    torch.cuda.empty_cache()
    return row


ADABELIEF = fe.AdaBelief


def no_eca():
    net = ConvNeXtECA("tiny", 3)
    net.eca = nn.Identity()
    return net


def adamw(params, lr, betas, eps):
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=0.05)


def main():
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    device = torch.device("cuda")
    tr_x, tr_y, files, te_x, te_y = load_data()
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step in ("all", "quick"):
        gradient_accumulation_check(tr_x, tr_y, device)
        memory_benchmark(tr_x, tr_y, device)
        eca_analysis(tr_x, te_x, te_y, device)
        torch.manual_seed(0)
        imagenet = ConvNeXtECA("tiny", 3).to(device)
        imagenet.eca = nn.Identity()
        write("transfer_imagenet.csv", [probe("ImageNet weights, no fine-tuning", features(imagenet, tr_x, device), tr_y,
                                              files, features(imagenet, te_x, device), te_y, device)])
        del imagenet
        torch.cuda.empty_cache()

    if step in ("all", "retrain"):
        rows = []
        for name, make, opt in (("no_eca", no_eca, None),
                                ("adamw", lambda: ConvNeXtECA("tiny", 3), adamw),
                                ("from_scratch", lambda: ConvNeXtECA("tiny", 3, pretrained=False), None)):
            rows.append(retrain(name, make, tr_x, tr_y, files, te_x, te_y, device, opt))
            write("feature_retrain.csv", rows)


if __name__ == "__main__":
    main()
