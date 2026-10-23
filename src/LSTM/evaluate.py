"""Evaluation script for the LSTM baseline.

Loads the saved Keras checkpoint and scores it on the requested split 
(test by default). It computes the headline metrics (Macro-F1, Accuracy, 
AUC-ROC, Confusion Matrix) and saves a JSON report to the runs/ directory,
mimicking the Workshop 1 and peers' reporting style.

Usage:
    poetry run python -m src.evaluate
    poetry run python -m src.evaluate --split validation --checkpoint checkpoints/mi_modelo_lstm.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tf_keras as keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .data import prepare_lstm_data, LABELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "model_best.keras"


def compute_metrics(y_true_series, y_true_encoded, probabilities) -> dict:
    """Calculates all metrics required for the report and JSON export."""
    y_pred_indices = np.argmax(probabilities, axis=1)
    y_pred = [LABELS[idx] for idx in y_pred_indices]

    accuracy = float(np.mean(y_pred_indices == y_true_encoded))
    macro_f1 = float(f1_score(y_true_series, y_pred, average="macro", labels=LABELS))
    auc = float(roc_auc_score(
        y_true_series, probabilities, multi_class="ovr", average="macro", labels=LABELS
    ))
    
    cm = confusion_matrix(y_true_series, y_pred, labels=LABELS)
    
    # Calculate recall per class dynamically
    report = classification_report(y_true_series, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    recall = {label: float(report[label]["recall"]) for label in LABELS}

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "auc_roc": auc,
        "recall": recall,
        "confusion_matrix": cm.tolist() # Lists are JSON serializable; numpy arrays are not
    }


def print_report(title: str, metrics: dict, y_true_series, probabilities):
    """Prints the metrics and the confusion matrix to the console."""
    y_pred_indices = np.argmax(probabilities, axis=1)
    y_pred = [LABELS[idx] for idx in y_pred_indices]

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"AUC-ROC  : {metrics['auc_roc']:.4f} (one-vs-rest, macro)")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}  <- headline metric\n")
    
    print("Classification Report:")
    print(classification_report(y_true_series, y_pred, labels=LABELS, zero_division=0))
    
    print("Confusion Matrix (rows = true, cols = predicted):")
    header = "            " + "".join(f"{lab[:8]:>10}" for lab in LABELS)
    print(header)
    for lab, row in zip(LABELS, metrics["confusion_matrix"]):
        print(f"{lab:>10}  " + "".join(f"{v:>10,}" for v in row))


def main() -> int:
    """Loads the model and evaluates it on the specified split."""
    parser = argparse.ArgumentParser(description="Evaluate the trained LSTM model.")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data pipeline...")
    data = prepare_lstm_data()

    # Select the correct split based on CLI arguments
    if args.split == "test":
        x_eval = data["x_test"]
        y_eval_encoded = data["y_test"]
        y_eval_series = data["y_test_series"]
    else:
        x_eval = data["x_val"]
        y_eval_encoded = data["y_val"]
        y_eval_series = data["y_val_series"]

    # 2. Load Model
    print(f"\nLoading model from {args.checkpoint}...")
    try:
        model = keras.models.load_model(args.checkpoint)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return 1

    # 3. Predict and evaluate
    print(f"Predicting on the '{args.split}' split...")
    probabilities = model.predict(x_eval)

    metrics = compute_metrics(y_eval_series, y_eval_encoded, probabilities)
    title = f"LSTM Keras — {args.split.upper()} SPLIT"
    print_report(title, metrics, y_eval_series, probabilities)

    # 4. Save results to JSON file
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(args.checkpoint).stem
    destination = RUNS_DIR / f"eval_{args.split}_{checkpoint_name}.json"
    
    payload = {
        "model": "lstm-keras",
        "split": args.split,
        "checkpoint": checkpoint_name,
        **metrics
    }
    
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Results successfully saved to {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())