import sys
from pathlib import Path

import numpy as np
import torch

from data import QUANTUM_DIR
from fidelity import FidelityClassifier

FEATURE_DIR = QUANTUM_DIR.parent / "Feature_Extraction"
sys.path.insert(0, str(FEATURE_DIR))
sys.path.insert(0, str(QUANTUM_DIR.parent / "Preprocessing"))
from extract_features import extract  # noqa: E402
from model import ConvNeXtECA  # noqa: E402
from preprocessing import CLASSES, enhance, z_score  # noqa: E402

MODELS_DIR = QUANTUM_DIR / "Models"
PREFIX = "qcf_"


def list_models():
    return sorted(MODELS_DIR.glob(f"{PREFIX}*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


def resolve(path):
    p = Path(path.strip().lstrip("\ufeff").strip('"').strip("'"))
    if not p.exists() and (MODELS_DIR / p.name).exists():
        p = MODELS_DIR / p.name
    if not p.exists():
        sys.exit(f"No model at {p}")
    return p


def choose_models(path=None, many=False):
    if path:
        return [resolve(path)]
    models = list_models()
    if not models:
        sys.exit(f"No saved models in {MODELS_DIR}: run run_pipeline.py first.")
    print("Saved models (newest first):")
    for i, p in enumerate(models, 1):
        print(f"  {i}. {p.name}")
    hint = ", several like 1,3, 'a' for all," if many else ""
    choice = input(f"Choose a model [1-{len(models)}{hint} Enter for 1] or paste a path: ").strip().lstrip("\ufeff")
    if not choice:
        return [models[0]]
    if many and choice.lower() == "a":
        return models
    parts = [c.strip() for c in choice.split(",")] if many else [choice]
    if all(c.isdigit() and 1 <= int(c) <= len(models) for c in parts):
        return [models[int(c) - 1] for c in parts]
    return [resolve(choice)]


def choose_model(path=None):
    return choose_models(path)[0]


class Pipeline:
    def __init__(self, path, device):
        self.path, self.device = Path(path), device
        b = torch.load(self.path, map_location="cpu", weights_only=False)
        self.config, self.metrics, self.times = b["config"], b.get("metrics", {}), b.get("times", {})
        self.classes = b["classes"]
        self.cnn = ConvNeXtECA(self.config["arch"], len(self.classes), pretrained=False)
        self.cnn.load_state_dict(b["cnn"])
        self.cnn.to(device).to(memory_format=torch.channels_last).eval()
        q = b["quantum_args"]
        self.clf = FidelityClassifier(b["X_ref"].shape[1], len(self.classes), n_layers=q["layers"],
                                      learn_temperature=not q["fixed_temperature"], temperature=q["temperature"])
        self.clf.load_state_dict(b["quantum"])
        self.clf.to(device).eval()
        self.X_ref = torch.from_numpy(b["X_ref"]).to(device)
        self.y_ref = torch.from_numpy(b["y_ref"]).long().to(device)
        self.ref_files = [str(f) for f in b["ref_files"]]
        with torch.no_grad():
            self.ref_states = self.clf.encode(self.X_ref)

    @staticmethod
    def preprocess(paths):
        return np.stack([z_score(enhance(p)[0])[0] for p in paths])

    def features(self, images):
        return extract(self.cnn, images, self.device)[0]

    @torch.no_grad()
    def classify(self, feats):
        states = self.clf.encode(torch.from_numpy(feats).to(self.device))
        scores = self.clf.class_scores(states, self.ref_states, self.y_ref)
        probs = torch.softmax(self.clf.temperature * scores, 1).cpu().numpy()
        fid = (states @ self.ref_states.conj().T).abs().pow(2).cpu().numpy()
        return probs, scores.cpu().numpy(), fid


def save_bundle(path, config, cnn_path, quantum_path, features_path, metrics, times):
    q = torch.load(quantum_path, map_location="cpu", weights_only=False)
    f = np.load(features_path)
    torch.save(dict(config=config, classes=CLASSES, cnn=torch.load(cnn_path, map_location="cpu"),
                    quantum=q["state_dict"], quantum_args=q["args"], X_ref=f["X_train"].astype(np.float32),
                    y_ref=f["y_train"], ref_files=[str(x) for x in f["files_train"]], metrics=metrics, times=times),
               path)
