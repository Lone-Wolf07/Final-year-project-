import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- settings
TEST_SIZE = 0.2              # 0.2 -> 80:20 split, 0.3 -> 70:30
ARCH = "tiny"                # ConvNeXt size: "tiny", "small" or "base"

CNN_EPOCHS = 10
CNN_MICRO_BATCH = 16         # lower this if the GPU runs out of memory

QUANTUM_EPOCHS = 10
QUANTUM_LAYERS = 2
QUANTUM_LR = 1e-4
QUANTUM_BATCH_SIZE = 32
TEMPERATURE = 10.0           # starting softmax temperature
LEARN_TEMPERATURE = True
SEED = 0

REUSE_PREPROCESSING = True   # skip preprocessing when this split's images already exist
EVALUATE_AFTER = True        # run evaluate.py on the new model at the end
MODEL_NAME = ""              # leave empty for an automatic label
# -------------------------------------------------------------------------

CNN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CNN_DIR / "Preprocessing"))
sys.path.insert(0, str(CNN_DIR / "Quantum_Classifier"))
from preprocessing import output_dir, split_name  # noqa: E402


def run(step, script, *args):
    print(f"\n=== {step} ===", flush=True)
    start = time.perf_counter()
    subprocess.run([sys.executable, "-u", script.name, *map(str, args)], cwd=script.parent, check=True)
    seconds = time.perf_counter() - start
    print(f"=== {step}: {seconds:.1f} s ===", flush=True)
    return seconds


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    from bundle import MODELS_DIR, PREFIX, save_bundle

    split = split_name(TEST_SIZE)
    tag = f"{split}_{ARCH}"
    times = {}

    if REUSE_PREPROCESSING and (output_dir(TEST_SIZE) / "manifest.csv").exists():
        print(f"Reusing preprocessed images in {output_dir(TEST_SIZE)}")
        times["preprocessing_s"] = None
    else:
        times["preprocessing_s"] = run("1. Preprocessing", CNN_DIR / "Preprocessing" / "preprocessing.py",
                                       "--test-size", TEST_SIZE)

    times["cnn_step_s"] = run("2. ConvNeXt-ECA training + feature extraction",
                              CNN_DIR / "Feature_Extraction" / "extract_features.py", "--test-size", TEST_SIZE,
                              "--arch", ARCH, "--epochs", CNN_EPOCHS, "--micro-batch", CNN_MICRO_BATCH)
    cnn_log = read_csv(CNN_DIR / "Feature_Extraction" / "Results" / tag / "training_log.csv")
    times["cnn_training_s"] = round(sum(float(r["seconds"]) for r in cnn_log), 1)

    quantum_args = ["--test-size", TEST_SIZE, "--arch", ARCH, "--epochs", QUANTUM_EPOCHS, "--layers", QUANTUM_LAYERS,
                    "--lr", QUANTUM_LR, "--batch-size", QUANTUM_BATCH_SIZE, "--temperature", TEMPERATURE,
                    "--seed", SEED] + ([] if LEARN_TEMPERATURE else ["--fixed-temperature"])
    times["quantum_step_s"] = run("3. Quantum fidelity classifier training",
                                  CNN_DIR / "Quantum_Classifier" / "train.py", *quantum_args)
    times = {k: (round(v, 1) if v is not None else None) for k, v in times.items()}

    metrics = {r["split"]: {k: float(v) for k, v in r.items() if k != "split"}
               for r in read_csv(CNN_DIR / "Quantum_Classifier" / "Results" / f"fidelity_{tag}" / "metrics.csv")}
    config = dict(test_size=TEST_SIZE, split=split, arch=ARCH, cnn_epochs=CNN_EPOCHS, cnn_micro_batch=CNN_MICRO_BATCH,
                  quantum_epochs=QUANTUM_EPOCHS, quantum_layers=QUANTUM_LAYERS, quantum_lr=QUANTUM_LR,
                  quantum_batch_size=QUANTUM_BATCH_SIZE, temperature=TEMPERATURE,
                  learn_temperature=LEARN_TEMPERATURE, seed=SEED, created=datetime.now().isoformat(timespec="seconds"))
    name = MODEL_NAME or (f"{PREFIX}{ARCH}_{split.removeprefix('split_')}_cnn{CNN_EPOCHS}_q{QUANTUM_EPOCHS}"
                          f"_L{QUANTUM_LAYERS}_{datetime.now():%Y%m%d-%H%M}")
    if not name.startswith(PREFIX):
        name = PREFIX + name
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{name.removesuffix('.pt')}.pt"
    save_bundle(model_path, config,
                cnn_path=CNN_DIR / "Feature_Extraction" / "Models" / f"convnext_eca_{tag}.pt",
                quantum_path=CNN_DIR / "Quantum_Classifier" / "Models" / f"fidelity_{tag}.pt",
                features_path=CNN_DIR / "Feature_Extraction" / "Features" / f"features_{tag}.npz",
                metrics=metrics, times=times)

    print(f"\nSaved model: {model_path}")
    print("Step times (s): " + ", ".join(f"{k} {v if v is not None else 'reused'}" for k, v in times.items()))
    print("Test: " + ", ".join(f"{k} {v}" for k, v in metrics["test"].items()))

    if EVALUATE_AFTER:
        run("4. Evaluation", CNN_DIR / "evaluate.py", "--model", model_path)


if __name__ == "__main__":
    main()
