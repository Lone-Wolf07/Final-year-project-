import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, estimate_sigma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Preprocessing"))
from preprocessing import CLASSES, aclahe, collect_images, nlm_denoise  # noqa: E402

try:
    import bm3d
except ImportError:
    bm3d = None

OUT = Path(__file__).resolve().parent / "Results"
PER_CLASS = 40
ADDED_SIGMA = 10.0
SEED = 0


def sample():
    rng = np.random.default_rng(SEED)
    items = collect_images()
    picks = []
    for c in CLASSES:
        paths = [p for p, label in items if label == c]
        picks += [(paths[i], c) for i in rng.choice(len(paths), PER_CLASS, replace=False)]
    return picks, items


def u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


DENOISERS = {
    "None": lambda img: img,
    "Gaussian (5x5, s=1)": lambda img: cv2.GaussianBlur(img, (5, 5), 1.0),
    "Median (5x5)": lambda img: cv2.medianBlur(img, 5),
    "Bilateral (d=7)": lambda img: cv2.bilateralFilter(img, 7, 2 * max(estimate_sigma(img), 3), 5),
    "Total variation": lambda img: u8(255 * denoise_tv_chambolle(img / 255.0, weight=0.05)),
    "Wavelet (BayesShrink)": lambda img: u8(255 * denoise_wavelet(img / 255.0, method="BayesShrink", mode="soft")),
    "NLM, adaptive h (ours)": lambda img: nlm_denoise(img)[0],
}
if bm3d is not None:
    DENOISERS["BM3D"] = lambda img: u8(bm3d.bm3d(img.astype(np.float32), sigma_psd=max(estimate_sigma(img), 3.0)))


def grad(img):
    g = img.astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def edge_keep(ref, out):
    a, b = grad(ref).ravel(), grad(out).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def denoise_study(picks):
    rng = np.random.default_rng(SEED)
    rows = []
    for name, fn in DENOISERS.items():
        psnr, ssim, edge, noise, ms = [], [], [], [], []
        for path, _ in picks:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            noisy = u8(img + rng.normal(0, ADDED_SIGMA, img.shape))
            t = time.perf_counter()
            out = fn(noisy)
            ms.append(1000 * (time.perf_counter() - t))
            psnr.append(peak_signal_noise_ratio(img, out, data_range=255))
            ssim.append(structural_similarity(img, out, data_range=255))
            edge.append(edge_keep(img, out))
            noise.append(estimate_sigma(fn(img)))
        rows.append(dict(method=name, psnr_db=np.mean(psnr), ssim=np.mean(ssim), edge_corr=np.mean(edge),
                         residual_sigma=np.mean(noise), ms_per_image=np.mean(ms)))
        print(f"{name:26s} PSNR {np.mean(psnr):6.2f}  SSIM {np.mean(ssim):.4f}  edge {np.mean(edge):.4f}  "
              f"{np.mean(ms):8.1f} ms")
    return rows


def entropy(img):
    p = np.bincount(img.ravel(), minlength=256) / img.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def eme(img, k=8):
    h, w = img.shape
    vals = []
    for i in range(k):
        for j in range(k):
            b = img[i * h // k:(i + 1) * h // k, j * w // k:(j + 1) * w // k].astype(np.float32)
            vals.append(20 * np.log10((b.max() + 1) / (b.min() + 1)))
    return float(np.mean(vals))


def body_mask(img):
    return cv2.morphologyEx((img > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


ENHANCERS = {
    "None": lambda img: img,
    "Global HE": cv2.equalizeHist,
    "Gamma (0.8)": lambda img: u8(255 * (img / 255.0) ** 0.8),
    "CLAHE, fixed clip 2.0": lambda img: cv2.createCLAHE(2.0, (8, 8)).apply(img),
    "CLAHE, fixed clip 4.0": lambda img: cv2.createCLAHE(4.0, (8, 8)).apply(img),
    "ACLAHE, adaptive clip (ours)": lambda img: aclahe(img)[0],
}


def enhance_study(picks):
    base = [nlm_denoise(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))[0] for p, _ in picks]
    rows = []
    for name, fn in ENHANCERS.items():
        ent, rms, em, amb, noise_gain, spread, ms = [], [], [], [], [], [], []
        for img in base:
            t = time.perf_counter()
            out = fn(img)
            ms.append(1000 * (time.perf_counter() - t))
            m = body_mask(img)
            ent.append(entropy(out))
            rms.append(out[m].std() if m.any() else out.std())
            em.append(eme(out))
            amb.append(abs(float(out.mean()) - float(img.mean())))
            noise_gain.append(estimate_sigma(out) / max(estimate_sigma(img), 1e-3))
            spread.append(out.std())
        rows.append(dict(method=name, entropy_bits=np.mean(ent), rms_contrast_body=np.mean(rms), eme=np.mean(em),
                         brightness_shift=np.mean(amb), noise_gain=np.mean(noise_gain),
                         std_across_images=np.std(spread), ms_per_image=np.mean(ms)))
        print(f"{name:30s} entropy {np.mean(ent):.3f}  EME {np.mean(em):6.2f}  noise x{np.mean(noise_gain):.2f}  "
              f"std-of-std {np.std(spread):.2f}")
    return rows


def dataset_stats(items):
    rows = []
    for path, label in items:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rows.append(dict(label=label, height=img.shape[0], width=img.shape[1], sigma=estimate_sigma(img),
                         std=float(img.std()), mean=float(img.mean())))
    return rows


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def figures(picks, stats):
    path = picks[0][0]
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    noisy = u8(img + np.random.default_rng(SEED).normal(0, ADDED_SIGMA, img.shape))
    crop = (slice(img.shape[0] // 4, img.shape[0] // 4 + 160), slice(img.shape[1] // 4, img.shape[1] // 4 + 160))
    names = ["None"] + [n for n in DENOISERS if n != "None"]
    fig, axes = plt.subplots(2, (len(names) + 1) // 2 + 1, figsize=(16, 6.4))
    axes = axes.ravel()
    axes[0].imshow(img[crop], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (crop)", fontsize=9)
    for ax, n in zip(axes[1:], names):
        ax.imshow(DENOISERS[n](noisy)[crop], cmap="gray", vmin=0, vmax=255)
        ax.set_title("Noisy input" if n == "None" else n, fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "denoisers.png", dpi=130)
    plt.close(fig)

    den = nlm_denoise(img)[0]
    fig, axes = plt.subplots(2, len(ENHANCERS), figsize=(18, 6))
    for k, (n, fn) in enumerate(ENHANCERS.items()):
        out = fn(den)
        axes[0, k].imshow(out, cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_title(n, fontsize=9)
        axes[0, k].axis("off")
        axes[1, k].hist(out.ravel(), 64, color="dimgrey")
        axes[1, k].set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "enhancers.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in CLASSES:
        axes[0].hist([s["sigma"] for s in stats if s["label"] == c], 30, alpha=0.6, label=c)
        axes[1].hist([s["std"] for s in stats if s["label"] == c], 30, alpha=0.6, label=c)
    axes[0].set(xlabel="Estimated noise sigma (grey levels)", ylabel="Images", title="(a) Noise level per slice")
    axes[1].set(xlabel="Intensity standard deviation", title="(b) Global contrast per slice")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_variation.png", dpi=130)
    plt.close(fig)


def noise_shortcut(items, noisy=5.0, typical_frac=0.08):
    rng = np.random.default_rng(SEED)
    groups = {}
    for path, label in items:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        s0 = estimate_sigma(img)
        if s0 > noisy or rng.random() < typical_frac:
            den = nlm_denoise(img)[0]
            groups.setdefault((label, s0 > noisy), []).append((s0, estimate_sigma(den), estimate_sigma(aclahe(den)[0])))
    rows = []
    for (label, is_noisy), v in sorted(groups.items()):
        v = np.median(np.array(v), axis=0)
        rows.append(dict(label=label, group=f"noisy (raw sigma>{noisy:g})" if is_noisy else "typical (random sample)",
                         n=len(groups[(label, is_noisy)]), raw=v[0], after_nlm=v[1], after_aclahe=v[2]))
        print(rows[-1])
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    if "--shortcut-only" in sys.argv:
        write("noise_shortcut.csv", noise_shortcut(collect_images()))
        return
    picks, items = sample()
    write("noise_shortcut.csv", noise_shortcut(items))
    print(f"{len(picks)} sampled images, {len(items)} total; BM3D {'available' if bm3d else 'not installed'}")
    stats = dataset_stats(items)
    write("dataset_stats.csv", stats)
    write("denoising.csv", denoise_study(picks))
    write("enhancement.csv", enhance_study(picks))
    figures(picks, stats)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
