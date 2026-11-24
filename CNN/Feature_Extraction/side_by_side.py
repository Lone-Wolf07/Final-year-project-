import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

FEATURE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FEATURE_DIR.parent / "Preprocessing"))
from preprocessing import CLASSES, IMAGE_EXTS, SOURCE_DIR, enhance, output_dir, z_score  # noqa: E402

from model import ConvNeXtECA  # noqa: E402

STAGE_1, STAGE_4 = 1, 7
TITLES = ["Original", "Preprocessed (224x224)", "Stage 1, strongest channel", "Stage 4 + ECA feature map",
          "feature vector"]


def find_original(source_name, label):
    for split in ("train", "test"):
        path = SOURCE_DIR / split / label / source_name
        if path.exists():
            return path
    raise FileNotFoundError(source_name)


def split_items(data_dir, split):
    with open(data_dir / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and r["augmented"] == "0"]
    for r in rows:
        yield dict(name=r["source"], label=r["label"], original=find_original(r["source"], r["label"]),
                   enhanced=lambda r=r: cv2.imread(str(data_dir / r["file"]), cv2.IMREAD_GRAYSCALE),
                   nlm_h=float(r["nlm_h"]), clip=float(r["aclahe_clip"]))


def folder_items(folder):
    for path in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        label = path.parent.name.lower()
        yield dict(name=path.relative_to(folder).as_posix(), label=label if label in CLASSES else None,
                   original=path, enhanced=lambda path=path: enhance(path), nlm_h=None, clip=None)


def image_item(image):
    path = image if image.exists() else next(SOURCE_DIR.rglob(image.name), None)
    if path is None:
        sys.exit(f"{image} not found, as a path or as a file name under {SOURCE_DIR}")
    label = path.parent.name.lower()
    yield dict(name=path.name, label=label if label in CLASSES else None,
               original=path, enhanced=lambda: enhance(path), nlm_h=None, clip=None)


def feature_maps(model, x):
    maps = {}
    hook = model.stages[STAGE_1].register_forward_hook(lambda m, i, o: maps.__setitem__("stage1", o))
    with torch.no_grad():
        stage4 = model.stages(x)
        refined = model.eca(stage4)
        feats = model.norm(model.pool(refined)).flatten(1)
        probs = torch.softmax(model.head(feats), 1)
    hook.remove()
    assert len(model.stages) == STAGE_4 + 1
    stage1 = maps["stage1"][0]
    # Stage 1's 96 channels cancel out when averaged, so show the most structured one.
    stage1 = stage1[stage1.flatten(1).std(1).argmax()]
    return (stage1.cpu().numpy(), refined[0].mean(0).cpu().numpy(),
            feats[0].cpu().numpy(), probs[0].cpu().numpy())


def draw_row(row, item, model, device):
    original = cv2.imread(str(item["original"]), cv2.IMREAD_GRAYSCALE)
    enhanced = item["enhanced"]()
    if isinstance(enhanced, tuple):  # enhance() also returns the NLM h and ACLAHE clip
        enhanced, item["nlm_h"], item["clip"] = enhanced
    preprocessed = z_score(enhanced)
    stage1, stage4, feats, probs = feature_maps(model, torch.from_numpy(preprocessed)[None].to(device))
    pred = CLASSES[int(probs.argmax())]

    truth = f", true: {item['label']}" if item["label"] else ""
    row[0].imshow(original, cmap="gray")
    row[0].set_title(f"{item['name']}\n{original.shape[1]}x{original.shape[0]}{truth}", fontsize=9)
    row[1].imshow(preprocessed[0], cmap="gray")
    row[1].set_title(f"NLM h={item['nlm_h']:.2f}, ACLAHE clip={item['clip']:.2f}", fontsize=9)
    row[2].imshow(stage1, cmap="inferno")
    row[2].set_title(f"{TITLES[2]} ({stage1.shape[0]}x{stage1.shape[1]})", fontsize=9)
    row[3].imshow(cv2.resize(stage4, (224, 224), interpolation=cv2.INTER_CUBIC), cmap="inferno")
    row[3].set_title(f"{TITLES[3]} ({stage4.shape[0]}x{stage4.shape[1]}, upsampled)", fontsize=9)
    for ax in row[:4]:
        ax.axis("off")
    row[4].bar(range(len(feats)), feats, width=1.0, color="tab:blue")
    row[4].set_xlim(0, len(feats))
    verdict = "" if item["label"] is None else (", correct" if pred == item["label"] else ", WRONG")
    row[4].set_title(f"predicted {pred} ({probs.max():.0%}{verdict}); "
                     + ", ".join(f"{c} {p:.0%}" for c, p in zip(CLASSES, probs)), fontsize=9)
    row[4].set_xlabel(f"{len(feats)}-d feature vector (feature index)", fontsize=8)
    return dict(image=item["name"], true=item["label"] or "", predicted=pred,
                correct="" if item["label"] is None else int(pred == item["label"]),
                **{f"p_{c}": round(float(p), 4) for c, p in zip(CLASSES, probs)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2, help="which preprocessed split's model to use")
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=["train", "val", "test"], default="test", help="every image of this split")
    source.add_argument("--folder", type=Path, help="every image in this folder instead of a split")
    source.add_argument("--image", type=Path, help="just this image (a path, or a file name in the dataset)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = output_dir(args.test_size, args.val_size)
    tag = f"{data_dir.name}_{args.arch}"
    model_path = FEATURE_DIR / "Models" / f"convnext_eca_{tag}.pt"
    if not model_path.exists():
        sys.exit(f"No model at {model_path}: run extract_features.py first.")
    if not args.folder and not args.image and not (data_dir / "manifest.csv").exists():
        sys.exit(f"No preprocessed images in {data_dir}: run CNN/Preprocessing/preprocessing.py first.")

    device = torch.device(args.device)
    model = ConvNeXtECA(args.arch, len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    if args.image:
        set_name, items = "single", image_item(args.image)
    elif args.folder:
        set_name, items = args.folder.name, folder_items(args.folder)
    else:
        set_name, items = args.split, split_items(data_dir, args.split)
    out_dir = FEATURE_DIR / "Results" / tag / "side_by_side" / set_name
    if out_dir.exists() and not args.image:
        shutil.rmtree(out_dir)

    results = []
    for n, item in enumerate(items, 1):
        fig, axes = plt.subplots(1, len(TITLES), figsize=(20, 3.8), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.8]))
        results.append(draw_row(axes[0], item, model, device))
        fig.suptitle(f"ConvNeXt-{args.arch.capitalize()} + ECA feature extraction", fontsize=12)
        fig.tight_layout()
        out = out_dir / (item["label"] or "") / f"{Path(item['name']).stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=80)
        plt.close(fig)
        r = results[-1]
        verdict = "" if r["correct"] == "" else ("ok" if r["correct"] else "WRONG")
        print(f"{n:>4}  {r['image']:<32} true {r['true'] or '?':<9} pred {r['predicted']:<9} {verdict}", flush=True)
    if not results:
        sys.exit("No images found.")
    if args.image:
        print(f"Saved {out}")
        return

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    labelled = [r for r in results if r["correct"] != ""]
    if labelled:
        print(f"\n{sum(r['correct'] for r in labelled)}/{len(labelled)} correct")
    print(f"{len(results)} side-by-side images + predictions.csv in {out_dir}")


if __name__ == "__main__":
    main()
