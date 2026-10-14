import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METHODS = {
    "Thresholding": ROOT / "Traditional_Methods" / "Thresholding" / "Results",
    "Edge detection": ROOT / "Traditional_Methods" / "Edge_Detection" / "Results",
    "Region growing": ROOT / "Traditional_Methods" / "Region_Growing" / "Results",
    "Region growing (improved)": ROOT / "Traditional_Methods" / "Region_Growing" / "Results_Improved",
    "SVM": ROOT / "ML_Methods" / "SVM" / "Results",
    "Random forest": ROOT / "ML_Methods" / "RF" / "Results",
}
DATASETS = ["COVID19_CT_Seg"]
ML_METHODS = {"SVM", "Random forest"}
OUT_CSV = Path(__file__).resolve().parent / "segmentation_results.csv"


def summarise(rows):
    dice = np.array([float(r["dice"]) for r in rows])
    tp, fp, fn, tn = (sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn", "tn"))
    missed = sum(int(r["tp"]) + int(r["fp"]) == 0 and int(r["tp"]) + int(r["fn"]) > 0 for r in rows)
    return dict(
        images=len(rows),
        dice_mean=round(dice.mean(), 4), dice_median=round(np.median(dice), 4), dice_std=round(dice.std(), 4),
        dice_below_0_8=int((dice < 0.8).sum()),
        iou=round(tp / (tp + fp + fn), 4), pixel_accuracy=round((tp + tn) / (tp + fp + fn + tn), 4),
        precision=round(tp / max(tp + fp, 1), 4), recall=round(tp / max(tp + fn, 1), 4),
        empty_masks=missed,
        seconds_per_image=round(np.mean([float(r["seconds"]) for r in rows]), 3),
    )


def main():
    results = []
    for method, results_dir in METHODS.items():
        for dataset in DATASETS:
            path = results_dir / dataset / "summary.csv"
            if not path.exists():
                print(f"missing: {path}")
                continue
            rows = list(csv.DictReader(open(path)))
            for split in ("test", "val", "train"):
                part = [r for r in rows if r["split"] == split]
                results.append(dict(method=method, dataset=dataset, split=split, **summarise(part)))

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    for dataset in DATASETS:
        print(f"\n{dataset} -- test split, all methods vs. the hand-drawn reference")
        print(f"{'Method':<27}{'Dice mean':>10}{'median':>8}{'std':>7}{'<0.8':>6}{'IoU':>8}{'PixAcc':>8}"
              f"{'Prec':>8}{'Recall':>8}{'Empty':>7}{'s/img':>8}")
        for r in (r for r in results if r["dataset"] == dataset and r["split"] == "test"):
            print(f"{r['method']:<27}{r['dice_mean']:>10.4f}{r['dice_median']:>8.4f}{r['dice_std']:>7.3f}"
                  f"{r['dice_below_0_8']:>6}{r['iou']:>8.4f}{r['pixel_accuracy']:>8.4f}{r['precision']:>8.4f}"
                  f"{r['recall']:>8.4f}{r['empty_masks']:>7}{r['seconds_per_image']:>8.3f}")

        test = sorted((r for r in results if r["dataset"] == dataset and r["split"] == "test"),
                      key=lambda r: r["dice_mean"], reverse=True)
        print(f"\n{dataset} -- ranking by test mean Dice")
        for rank, r in enumerate(test, 1):
            kind = "ML" if r["method"] in ML_METHODS else "traditional"
            print(f"  {rank}. {r['method']:<27}{kind:<13}Dice {r['dice_mean']:.4f}  IoU {r['iou']:.4f}")
        best_trad = next((r for r in test if r["method"] not in ML_METHODS), None)
        best_ml = next((r for r in test if r["method"] in ML_METHODS), None)
        if best_trad and best_ml:
            print(f"Best traditional: {best_trad['method']} (Dice {best_trad['dice_mean']:.4f}) | "
                  f"best ML: {best_ml['method']} (Dice {best_ml['dice_mean']:.4f}) | "
                  f"ML - traditional: {best_ml['dice_mean'] - best_trad['dice_mean']:+.4f}")
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
