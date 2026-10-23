"""Train the BiLSTM + GloVe model on Workshop #1's split. Validation only; test untouched.

Two phases, the recurrent analogue of the transformer's transfer-learning recipe:

1. GloVe frozen; only the BiLSTM and the head learn (``frozen_epochs``). The
   pretrained word vectors stay put while the model learns to read sequences.
2. GloVe unfrozen at a smaller learning rate (``finetune_epochs``), letting the
   embeddings adapt to the reviews, with early stopping on validation macro-F1.

Within each phase a ReduceLROnPlateau scheduler halves the learning rate when
validation macro-F1 stalls. The checkpoint with the best validation macro-F1,
from either phase, is saved to checkpoints/bilstm_best.pt. Losses and metrics go
to TensorBoard under runs/.

    poetry run python -m src.bilstm.train                       # full run (use the GPU)
    poetry run python -m src.bilstm.train --limit 64 --device cpu --run-name smoke
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import BEST_CHECKPOINT, RUNS_DIR, BiLSTMConfig, get_device, set_seed
from .data import build_embedding_matrix, build_loaders, class_weights
from .evaluate import compute_metrics, predict
from .model import BiLSTMClassifier, count_parameters, save_checkpoint

LOG_EVERY = 50  # optimiser steps between TensorBoard points for the training loss


def parse_args() -> tuple[BiLSTMConfig, argparse.Namespace]:
    """CLI overrides for the most-tuned hyperparameters; the rest live in BiLSTMConfig."""
    defaults = BiLSTMConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--max-len", type=int, default=defaults.max_len)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--finetune-lr", type=float, default=defaults.finetune_lr)
    parser.add_argument("--frozen-epochs", type=int, default=defaults.frozen_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=defaults.finetune_epochs)
    parser.add_argument("--limit", type=int, default=0, help="rows per split (debug)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT))
    args = parser.parse_args()

    config = BiLSTMConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        finetune_lr=args.finetune_lr,
        frozen_epochs=args.frozen_epochs,
        finetune_epochs=args.finetune_epochs,
        limit=args.limit,
    )
    return config, args


def make_optimizer(model: BiLSTMClassifier, lr: float, weight_decay: float):
    """AdamW over the currently trainable parameters (embeddings may be frozen)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, criterion, device, config, writer, step, desc):
    """One pass over the training loader. Returns (mean loss, global step)."""
    model.train()
    total_loss, seen = 0.0, 0
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"]
        target = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, lengths)
        loss = criterion(logits.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        step += 1
        total_loss += loss.item() * len(target)
        seen += len(target)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if step % LOG_EVERY == 0:
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
    return total_loss / seen, step


def main() -> int:  # pylint: disable=too-many-locals,too-many-statements
    """Run both phases, log every epoch, keep the best validation checkpoint."""
    config, args = parse_args()
    set_seed(config.seed)
    device = get_device(args.device)
    run_name = args.run_name or f"bilstm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    writer = SummaryWriter(run_dir)

    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                 if device.type == "cuda" else ""))
    print(f"run: {run_dir}")

    loaders, y_train, vocab = build_loaders(config, splits=("train", "validation"))
    print(f"train {len(loaders['train'].dataset):,} | "
          f"validation {len(loaders['validation'].dataset):,} reviews")

    embedding_weights = build_embedding_matrix(vocab, config, config.seed)
    model = BiLSTMClassifier(len(vocab), config, embedding_weights=embedding_weights).to(device)
    weights = class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("class weights:", [round(w, 3) for w in weights.tolist()])
    print("parameters:", {k: f"{v:,}" for k, v in count_parameters(model).items()})
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    phases = [("frozen", config.frozen_epochs, config.lr, True),
              ("finetune", config.finetune_epochs, config.finetune_lr, False)]
    history, best_f1, best_epoch, step, epoch = [], -1.0, None, 0, 0
    started = time.time()

    for phase, n_epochs, lr, frozen in phases:
        if n_epochs == 0:
            continue
        model.freeze_embeddings(frozen)
        optimizer = make_optimizer(model, lr, config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=config.lr_factor,
                                      patience=config.lr_patience, min_lr=config.min_lr)
        print(f"\n--- phase '{phase}': {n_epochs} epoch(s), lr {lr:g}, "
              f"trainable {count_parameters(model)['trainable']:,} params")
        stale = 0

        for _ in range(n_epochs):
            epoch += 1
            epoch_start = time.time()
            train_loss, step = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, config,
                writer, step, desc=f"epoch {epoch} [{phase}]",
            )
            probabilities, y_val, val_loss = predict(
                model, loaders["validation"], device, criterion, desc="validation"
            )
            metrics = compute_metrics(y_val, probabilities)
            scheduler.step(metrics["macro_f1"])
            minutes = (time.time() - epoch_start) / 60

            record = {"epoch": epoch, "phase": phase, "train_loss": train_loss,
                      "val_loss": val_loss, "minutes": round(minutes, 2), **metrics}
            history.append(record)
            writer.add_scalars("loss", {"train": train_loss, "validation": val_loss}, epoch)
            for key in ("accuracy", "macro_f1", "auc_roc"):
                if metrics[key] is not None:
                    writer.add_scalar(f"validation/{key}", metrics[key], epoch)
            for label, value in metrics["recall"].items():
                writer.add_scalar(f"validation/recall_{label}", value, epoch)
            writer.flush()

            auc = metrics["auc_roc"]
            auc_text = f"{auc:.4f}" if auc is not None else "n/a"
            recall_text = " ".join(f"{k[:3]} {v:.3f}" for k, v in metrics["recall"].items())
            print(f"epoch {epoch} [{phase}] {minutes:.1f} min | train loss {train_loss:.4f} | "
                  f"val loss {val_loss:.4f} | acc {metrics['accuracy']:.4f} | AUC {auc_text} | "
                  f"macro-F1 {metrics['macro_f1']:.4f} | recall {recall_text}")

            if metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
                save_checkpoint(model, config, vocab, Path(args.checkpoint), epoch=epoch,
                                phase=phase, val_metrics=metrics, run=run_name)
                print(f"  new best -> {args.checkpoint}")
            else:
                stale += 1
                if stale > config.patience:
                    print(f"  no improvement for {stale} epochs: early stop")
                    break

    summary = {"run": run_name, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
               "minutes": round((time.time() - started) / 60, 1), "history": history,
               "config": config.to_dict()}
    (run_dir / "history.json").write_text(json.dumps(summary, indent=2))
    writer.close()
    print(f"\ndone in {summary['minutes']} min | best epoch {best_epoch} | "
          f"val macro-F1 {best_f1:.4f} (Workshop #1 best: 0.6258)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
