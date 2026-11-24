import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "Quantum_Classifier"))
from bundle import Pipeline, choose_model  # noqa: E402
from preprocessing import IMAGE_EXTS, SOURCE_DIR  # noqa: E402


def find_images(targets):
    paths = []
    for t in map(Path, targets):
        if t.is_dir():
            paths += sorted(p for p in t.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif t.exists():
            paths.append(t)
        else:
            matches = list(SOURCE_DIR.rglob(t.name))
            if not matches:
                sys.exit(f"Not found: {t}")
            paths.append(matches[0])
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", help="image paths, folders, or file names in the dataset")
    parser.add_argument("--model", help="path to a saved model; leave out to choose from the Models folder")
    parser.add_argument("--similar", type=int, default=3, help="most similar training scans to list")
    args = parser.parse_args()

    model_path = choose_model(args.model)
    if not args.images:
        args.images = [input("Path to the image: ").strip().lstrip("\ufeff").strip('"').strip("'")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = Pipeline(model_path, device)
    print(f"Model: {model_path.name}  (" + ", ".join(f"{k} {v}" for k, v in p.config.items() if k != "created") + ")")
    ref_stems = [Path(f).stem for f in p.ref_files]
    train_stems = {s.split("_aug")[0] for s in ref_stems}

    paths = find_images(args.images)
    probs, scores, fid = p.classify(p.features(p.preprocess(paths)))
    for path, prob, s, f in zip(paths, probs, scores, fid):
        print(f"\n{path.name}")
        if path.stem in train_stems:
            print("  note: this image is in the training set, so it is compared with itself")
        print(f"  prediction: {p.classes[prob.argmax()]}  ("
              + ", ".join(f"{c} {q:.1%}" for c, q in zip(p.classes, prob)) + ")")
        print("  class similarity S_c: " + ", ".join(f"{c} {v:.4f}" for c, v in zip(p.classes, s)))
        print("  most similar training scans:")
        for i in np.argsort(-f)[:args.similar]:
            print(f"    {ref_stems[i]:<28} {p.classes[int(p.y_ref[i])]:<9} fidelity {f[i]:.3f}")


if __name__ == "__main__":
    main()
