import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FEATURE_DIR.parent / "Preprocessing"))
from preprocessing import split_name  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2, help="which split's features to export")
    parser.add_argument("--val-size", type=float, default=0.0)
    parser.add_argument("--arch", default="tiny", choices=["tiny", "small", "base"])
    args = parser.parse_args()

    tag = f"{split_name(args.test_size, args.val_size)}_{args.arch}"
    npz_path = FEATURE_DIR / "Features" / f"features_{tag}.npz"
    if not npz_path.exists():
        sys.exit(f"No features at {npz_path}: run extract_features.py first.")

    f = np.load(npz_path)
    classes = list(f["classes"])
    for split in [s for s in ("train", "val", "test") if f"X_{s}" in f]:
        X = f[f"X_{split}"]
        df = pd.DataFrame(X, columns=[f"f{j}" for j in range(X.shape[1])])
        df.insert(0, "file", f[f"files_{split}"])
        df.insert(1, "label", [classes[k] for k in f[f"y_{split}"]])
        df.insert(2, "augmented", f["augmented_train"] if split == "train" else 0)
        out = FEATURE_DIR / "Features" / f"{split}_features_{tag}.csv"
        df.to_csv(out, index=False)
        print(f"{split}: {X.shape[0]} images x {X.shape[1]} features -> {out}")


if __name__ == "__main__":
    main()
