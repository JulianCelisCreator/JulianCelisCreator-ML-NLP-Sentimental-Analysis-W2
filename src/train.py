"""Fine-tune the transformer on Workshop #1's split. Validation only; test is untouched.

Two phases, adapted from the brief's transfer-learning recipe:

1. Encoder frozen, only the classification head learns (``frozen_epochs``).
2. Whole network unfrozen at a small learning rate with linear warmup and decay
   (``finetune_epochs``), stopping early when validation macro-F1 stops improving.

The brief's 5-10 + 10-20 epochs is a CNN-on-ImageNet recipe; BERT overfits well
before that, so the defaults are 1 + 3 (see README.md).

The checkpoint with the best validation macro-F1, from either phase, is saved to
checkpoints/model_best.pt. Losses and metrics go to TensorBoard under runs/.

    poetry run python -m src.train                        # full run (use the GPU)
    poetry run python -m src.train --limit 64 --device cpu --run-name smoke
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import BEST_CHECKPOINT, RUNS_DIR, TrainConfig, get_device, set_seed
from .data import build_loaders, class_weights
from .evaluate import autocast, compute_metrics, predict
from .model import SentimentClassifier, count_parameters, save_checkpoint

LOG_EVERY = 50  # optimiser steps between TensorBoard points for the training loss


def parse_args() -> tuple[TrainConfig, argparse.Namespace]:
    """CLI overrides for the most-tuned hyperparameters; the rest live in TrainConfig."""
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=defaults.model_name)
    parser.add_argument("--max-len", type=int, default=defaults.max_len)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--frozen-epochs", type=int, default=defaults.frozen_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=defaults.finetune_epochs)
    parser.add_argument("--limit", type=int, default=0, help="rows per split (debug)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT))
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model,
        max_len=args.max_len,
        head_tokens=min(defaults.head_tokens, args.max_len // 4),
        batch_size=args.batch_size,
        lr=args.lr,
        frozen_epochs=args.frozen_epochs,
        finetune_epochs=args.finetune_epochs,
        limit=args.limit,
    )
    return config, args


def make_optimizer(model: SentimentClassifier, lr: float, weight_decay: float, head_only: bool):
    """AdamW; no weight decay on biases and LayerNorm weights, as in BERT's recipe."""
    params = model.classifier.named_parameters() if head_only else model.named_parameters()
    decay, no_decay = [], []
    for name, parameter in params:
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "LayerNorm" in name else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, config,
                    writer, step, desc):
    """One pass over the training loader. Returns (mean loss, global step)."""
    model.train()
    total_loss, seen = 0.0, 0
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, config.bf16):
            logits = model(input_ids, attention_mask)
        loss = criterion(logits.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        total_loss += loss.item() * len(target)
        seen += len(target)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if step % LOG_EVERY == 0:
            writer.add_scalar("train/loss_step", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
    return total_loss / seen, step


def main() -> int:  # pylint: disable=too-many-locals,too-many-statements
    """Run both phases, log every epoch, keep the best validation checkpoint."""
    config, args = parse_args()
    set_seed(config.seed)
    device = get_device(args.device)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_name
    writer = SummaryWriter(run_dir)

    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})"
                                 if device.type == "cuda" else ""))
    print(f"run: {run_dir}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    loaders, y_train = build_loaders(tokenizer, config, splits=("train", "validation"))
    print(f"train {len(loaders['train'].dataset):,} | "
          f"validation {len(loaders['validation'].dataset):,} reviews")

    model = SentimentClassifier(config.model_name, dropout=config.dropout).to(device)
    weights = class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("class weights:", [round(w, 3) for w in weights.tolist()])
    print("parameters:", {k: f"{v:,}" for k, v in count_parameters(model).items()})
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    phases = [("frozen", config.frozen_epochs, config.head_lr, True),
              ("finetune", config.finetune_epochs, config.lr, False)]
    history, best_f1, best_epoch, step, epoch = [], -1.0, None, 0, 0
    started = time.time()

    for phase, n_epochs, lr, frozen in phases:
        if n_epochs == 0:
            continue
        model.freeze_encoder(frozen)
        optimizer = make_optimizer(model, lr, config.weight_decay, head_only=frozen)
        total_steps = n_epochs * len(loaders["train"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(config.warmup_ratio * total_steps), total_steps
        )
        print(f"\n--- phase '{phase}': {n_epochs} epoch(s), lr {lr:g}, "
              f"trainable {count_parameters(model)['trainable']:,} params")
        stale = 0

        for _ in range(n_epochs):
            epoch += 1
            epoch_start = time.time()
            train_loss, step = train_one_epoch(
                model, loaders["train"], optimizer, scheduler, criterion, device, config,
                writer, step, desc=f"epoch {epoch} [{phase}]",
            )
            probabilities, y_val, val_loss = predict(
                model, loaders["validation"], device, config.bf16, criterion, desc="validation"
            )
            metrics = compute_metrics(y_val, probabilities)
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
                save_checkpoint(model, config, Path(args.checkpoint), epoch=epoch,
                                phase=phase, val_metrics=metrics, run=run_name)
                print(f"  new best -> {args.checkpoint}")
            else:
                stale += 1
                if phase == "finetune" and stale > config.patience:
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
