"""Evaluation for the BiLSTM: the same metrics as Workshop #1, from the checkpoint.

The metric definitions (macro-F1 headline, accuracy, one-vs-rest macro AUC, per
-class recall, confusion matrix) and their printed report are the transformer's,
imported from ``src.evaluate`` so both models are scored identically. Only the
prediction loop differs: the BiLSTM takes (input_ids, lengths) and runs in plain
float32, with no autocast.

The test split is the final anchor, scored ONCE after model selection:

    poetry run python -m src.bilstm.evaluate                    # test split
    poetry run python -m src.bilstm.evaluate --split validation # re-check validation

Results are printed and written to runs/eval_<split>_<checkpoint name>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Metrics and reporting are shared with the transformer, so numbers line up.
from ..evaluate import compute_metrics, print_report
from .config import BEST_CHECKPOINT, RUNS_DIR, get_device, set_seed
from .data import build_loaders
from .model import load_checkpoint


@torch.no_grad()
def predict(model, loader, device, criterion=None, desc="eval"):
    """Return (class probabilities, true labels, mean loss or None)."""
    model.eval()
    probabilities, labels, total_loss, seen = [], [], 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"]
        target = batch["labels"].to(device)
        logits = model(input_ids, lengths).float()
        if criterion is not None:
            total_loss += criterion(logits, target).item() * len(target)
            seen += len(target)
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.append(target.cpu().numpy())
    mean_loss = total_loss / seen if criterion is not None else None
    return np.concatenate(probabilities), np.concatenate(labels), mean_loss


def main() -> int:
    """Score the best checkpoint on one split and save the numbers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed()
    device = get_device(args.device)
    model, vocab, config, checkpoint = load_checkpoint(args.checkpoint, device)
    config.limit = 0  # always the full split, whatever the training run used
    loaders, _, _ = build_loaders(config, vocab=vocab, splits=(args.split,))

    probabilities, y_true, _ = predict(model, loaders[args.split], device)
    metrics = compute_metrics(y_true, probabilities)
    print_report(f"BiLSTM + GloVe — {args.split} split "
                 f"(best epoch {checkpoint.get('epoch')})", y_true, probabilities, metrics)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RUNS_DIR / f"eval_{args.split}_{Path(args.checkpoint).stem}.json"
    destination.write_text(json.dumps({"split": args.split, "model": "bilstm-glove",
                                       "checkpoint": Path(args.checkpoint).name,
                                       "best_epoch": checkpoint.get("epoch"), **metrics},
                                      indent=2), encoding="utf-8")
    print(f"\nsaved to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
