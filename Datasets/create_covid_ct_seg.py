import argparse
import csv
import random
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from skimage import io

DATASETS_DIR = Path(__file__).resolve().parent
OUT_DIR = DATASETS_DIR / "COVID19_CT_Seg"
ZENODO = "https://zenodo.org/api/records/3757476/files/{}/content"
ARCHIVES = {"cases": "COVID-19-CT-Seg_20cases.zip", "masks": "Lung_Mask.zip"}
LUNG_WINDOW = (-600, 1500)
SLICES_PER_PATIENT = 30
MIN_LUNG_PIXELS = 500
SPLIT_PATIENTS = {"coronacases": {"train": 5, "val": 2, "test": 3},
                  "radiopaedia": {"train": 4, "val": 1, "test": 3}}
RANDOM_STATE = 42


def fetch(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for sub, name in ARCHIVES.items():
        target = raw_dir / sub
        if target.exists() and any(target.rglob("*.nii.gz")):
            continue
        archive = raw_dir / name
        if not archive.exists():
            print(f"Downloading {name} ...", flush=True)
            urllib.request.urlretrieve(ZENODO.format(name), archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)


def patient_of(scan):
    m = re.match(r"(radiopaedia_\d+)_", scan)
    return m.group(1) if m else scan


def split_patients(patients):
    rng = random.Random(RANDOM_STATE)
    split_of = {}
    for source, counts in SPLIT_PATIENTS.items():
        group = sorted(p for p in patients if p.startswith(source))
        rng.shuffle(group)
        assert len(group) == sum(counts.values()), f"{source}: {len(group)} patients"
        i = 0
        for split, n in counts.items():
            for p in group[i:i + n]:
                split_of[p] = split
            i += n
    return split_of


def to_uint8(volume):
    if volume.min() >= 0 and volume.max() <= 255:
        return volume.astype(np.uint8)
    level, width = LUNG_WINDOW
    lo = level - width / 2
    return (np.clip((volume - lo) / width, 0, 1) * 255).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=OUT_DIR / "Raw", help="where the Zenodo files are / go")
    args = parser.parse_args()
    fetch(args.raw)

    case_paths = {p.name.removesuffix(".nii.gz"): p for p in (args.raw / "cases").rglob("*.nii.gz")}
    mask_paths = {p.name.removesuffix(".nii.gz"): p for p in (args.raw / "masks").rglob("*.nii.gz")}
    assert case_paths.keys() == mask_paths.keys(), "scans and masks don't match"
    scans_of = defaultdict(list)
    for scan in sorted(case_paths):
        scans_of[patient_of(scan)].append(scan)
    split_of = split_patients(scans_of)

    rows, samples = [], []
    for patient, scans in sorted(scans_of.items()):
        split = split_of[patient]
        per_scan = SLICES_PER_PATIENT // len(scans)
        n_slices = 0
        for scan in scans:
            volume = sitk.GetArrayFromImage(sitk.ReadImage(str(case_paths[scan])))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_paths[scan]))) > 0
            assert volume.shape == mask.shape, scan
            volume, mask = to_uint8(volume)[:, ::-1], mask[:, ::-1]
            with_lung = np.flatnonzero(mask.sum(axis=(1, 2)) >= MIN_LUNG_PIXELS)
            picks = with_lung[np.unique(np.linspace(0, len(with_lung) - 1, per_scan).round().astype(int))]
            for z in picks:
                rel = Path(split) / patient / f"{scan}_z{z:03d}.png"
                for root, arr in (("DataSplit", volume[z]), ("ReferenceMasks", mask[z] * np.uint8(255))):
                    path = OUT_DIR / root / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    io.imsave(path, np.ascontiguousarray(arr), check_contrast=False)
                samples.append((rel, volume[z], mask[z]))
            n_slices += len(picks)
        rows.append(dict(patient=patient, split=split, scans=" ".join(scans), slices=n_slices))
        print(f"{patient:<20} {split:<6} {len(scans)} scan(s), {n_slices} slices", flush=True)

    with open(OUT_DIR / "patients.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for ax, (rel, img, mask) in zip(axes.flat, random.Random(RANDOM_STATE).sample(samples, 16)):
        ax.imshow(img, cmap="gray")
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=1)
        ax.set_title(str(rel), fontsize=7)
        ax.axis("off")
    fig.suptitle("COVID19_CT_Seg: hand-drawn lung masks")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample.png", dpi=80)
    plt.close(fig)

    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        print(f"{split}: {len(part)} patients, {sum(r['slices'] for r in part)} slices")


if __name__ == "__main__":
    main()
