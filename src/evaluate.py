"""Evaluation: the same metrics as Workshop #1, on any split, from the saved checkpoint.

The test split is the final anchor and is meant to be scored ONCE, after model
selection on validation is finished:

    poetry run python -m src.evaluate                      # test split
    poetry run python -m src.evaluate --split validation   # re-check validation

Results are printed and written to runs/eval_<split>_<checkpoint name>.json.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from .config import BEST_CHECKPOINT, LABELS, RUNS_DIR, get_device, set_seed
from .data import build_loaders
from .model import load_checkpoint


def autocast(device: torch.device, bf16: bool):
    """bf16 autocast on the GPU; full precision elsewhere."""
    if bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict(model, loader, device, bf16=True, criterion=None, desc="eval"):
    """Return (class probabilities, true labels, mean loss or None)."""
    model.eval()
    probabilities, labels, total_loss, seen = [], [], 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["labels"].to(device)
        with autocast(device, bf16):
            logits = model(input_ids, attention_mask)
        logits = logits.float()
        if criterion is not None:
            total_loss += criterion(logits, target).item() * len(target)
            seen += len(target)
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(target.cpu().numpy())
    mean_loss = total_loss / seen if criterion is not None else None
    return np.concatenate(probabilities), np.concatenate(labels), mean_loss


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Headline metrics in Workshop #1's definitions: macro-F1, accuracy, OvR macro AUC."""
    y_pred = probabilities.argmax(axis=1)
    label_ids = list(range(len(LABELS)))
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    metrics = {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=label_ids)),
        "auc_roc": None,
        "recall": {label: float(r) for label, r in zip(LABELS, recall)},
        "confusion": matrix.tolist(),
    }
    # AUC needs every class present; a tiny debug subset may lack one.
    if len(np.unique(y_true)) == len(LABELS):
        metrics["auc_roc"] = float(
            roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro",
                          labels=label_ids)
        )
    return metrics


def print_report(name: str, y_true: np.ndarray, probabilities: np.ndarray, metrics: dict):
    """Human-readable report, laid out like Workshop #1's baselines output."""
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"accuracy : {metrics['accuracy']:.4f}")
    auc = metrics["auc_roc"]
    print(f"AUC-ROC  : {auc:.4f} (one-vs-rest, macro)" if auc is not None else "AUC-ROC  : n/a")
    print(f"macro-F1 : {metrics['macro_f1']:.4f}  <- headline metric\n")
    print(classification_report(y_true, probabilities.argmax(axis=1),
                                labels=list(range(len(LABELS))), target_names=LABELS,
                                zero_division=0))
    print("confusion matrix (rows = true, cols = predicted):")
    print("            " + "".join(f"{label[:8]:>10}" for label in LABELS))
    for label, row in zip(LABELS, metrics["confusion"]):
        print(f"{label:>10}  " + "".join(f"{value:>10,}" for value in row))


def main() -> int:
    """Score the best checkpoint on one split and save the numbers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device(args.device)
    model, tokenizer, config, checkpoint = load_checkpoint(args.checkpoint, device)
    config.limit = 0  # always the full split, whatever the training run used
    loaders, _ = build_loaders(tokenizer, config, splits=(args.split,))

    probabilities, y_true, _ = predict(model, loaders[args.split], device, config.bf16)
    metrics = compute_metrics(y_true, probabilities)
    print_report(f"{config.model_name} — {args.split} split "
                 f"(best epoch {checkpoint.get('epoch')})", y_true, probabilities, metrics)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RUNS_DIR / f"eval_{args.split}_{Path(args.checkpoint).stem}.json"
    destination.write_text(json.dumps({"split": args.split, "model": config.model_name,
                                       "checkpoint": Path(args.checkpoint).name,
                                       "best_epoch": checkpoint.get("epoch"), **metrics},
                                      indent=2), encoding="utf-8")
    print(f"\nsaved to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
