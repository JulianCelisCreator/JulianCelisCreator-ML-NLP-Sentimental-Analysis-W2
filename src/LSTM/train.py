"""Training script for the LSTM model.

This script trains the LSTM model using the processed data, implements
callbacks for early stopping and checkpointing, logs metrics to TensorBoard,
and evaluates the best model on the validation set.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tf_keras as keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Import custom modules
from .data import prepare_lstm_data, LABELS
from .model import build_lstm_model

# Setup paths (adjusting to the Workshop's repository structure)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_CHECKPOINT = CHECKPOINT_DIR / "model_best_LSTM.keras"


def evaluate_validation(model, x_val, y_val_series, y_val_encoded):
    """Calculates and prints comprehensive metrics for the validation set."""
    print(f"\n{'=' * 60}\nVALIDATION RESULTS (BEST MODEL)\n{'=' * 60}")
    
    # Predict probabilities
    probabilities = model.predict(x_val)
    y_pred_indices = np.argmax(probabilities, axis=1)
    
    # Convert predictions to text labels
    y_pred = [LABELS[idx] for idx in y_pred_indices]
    
    # Calculate metrics
    accuracy = np.mean(y_pred_indices == y_val_encoded)
    macro_f1 = f1_score(y_val_series, y_pred, average="macro", labels=LABELS)
    auc = roc_auc_score(
        y_val_series, probabilities, multi_class="ovr", average="macro", labels=LABELS
    )
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC-ROC  : {auc:.4f} (one-vs-rest, macro)")
    print(f"Macro-F1 : {macro_f1:.4f}  <- headline metric\n")
    
    # Detailed report (Precision, Recall, F1 per class)
    print("Classification Report:")
    print(classification_report(y_val_series, y_pred, labels=LABELS, zero_division=0))
    
    # Confusion Matrix
    print("Confusion Matrix (rows = true, cols = predicted):")
    matrix = confusion_matrix(y_val_series, y_pred, labels=LABELS)
    header = "            " + "".join(f"{lab[:8]:>10}" for lab in LABELS)
    print(header)
    for lab, row in zip(LABELS, matrix):
        print(f"{lab:>10}  " + "".join(f"{v:>10,}" for v in row))


def main():
    """Main training loop."""
    # Ensure directories exist
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a unique directory for TensorBoard logs
    run_name = f"lstm_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = RUNS_DIR / run_name
    
    # 1. Load and prepare data
    print("Loading and preparing data...")
    data = prepare_lstm_data()
    
    # 2. Build model
    print("Building LSTM architecture...")
    model = build_lstm_model(
        vocab_size=data["vocab_size"], 
        max_len=data["max_len"], 
        num_classes=len(LABELS)
    )
    
    # 3. Define Callbacks
    # TensorBoard: tracks train/val loss and accuracy across epochs
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=1
    )
    
    # ModelCheckpoint: saves the weights only when val_loss improves
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_CHECKPOINT),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )
    
    # EarlyStopping: halts training if val_loss doesn't improve for 3 epochs
    # restore_best_weights=True ensures the final model has the best weights
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )
    
    # 4. Train Model
    epochs = 15  # Set a high number, EarlyStopping will stop it if needed
    batch_size = 32
    print(f"\nStarting training... Logs will be saved to {log_dir}")
    
    model.fit(
        data["x_train"],
        data["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data["x_val"], data["y_val"]),
        class_weight=data["class_weights"],
        callbacks=[tensorboard_cb, checkpoint_cb, early_stopping_cb]
    )
    
    # 5. Evaluate on Validation Set
    print("\nTraining completed. Evaluating the best checkpoint on the validation set...")
    evaluate_validation(
        model=model, 
        x_val=data["x_val"], 
        y_val_series=data["y_val_series"], 
        y_val_encoded=data["y_val"]
    )
    
    print(f"\n[SUCCESS] Best model saved to: {BEST_CHECKPOINT}")
    print(f"[INFO] To view the loss graphs, run in your terminal:")
    print(f"       poetry run tensorboard --logdir runs")


if __name__ == "__main__":
    main()