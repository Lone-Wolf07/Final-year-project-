import time

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_lr import DATA_DIR, MODEL_PATH, load_split


def main():
    print(f"Loading model from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]

    print("Loading test set...")
    t0 = time.time()
    X_test, y_test = load_split(DATA_DIR / "test")
    print(f"  {len(y_test)} images loaded in {time.time() - t0:.1f}s")

    y_test_enc = label_encoder.transform(y_test)

    print("Running predictions...")
    y_pred = pipeline.predict(X_test)

    print("\nTest results:")
    test_accuracy = accuracy_score(y_test_enc, y_pred)
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(
        classification_report(
            y_test_enc, y_pred, target_names=label_encoder.classes_, zero_division=0
        )
    )
    print("Confusion matrix (rows=true, cols=pred):")
    print(label_encoder.classes_)
    print(confusion_matrix(y_test_enc, y_pred))

    print("\n" + "=" * 40)
    print("FINAL TEST RESULT")
    print(f"  Test images   : {len(y_test)}")
    print(f"  Test accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()
