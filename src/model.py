"""Model architecture: a pretrained BERT-family encoder plus a classification head.

    input_ids (batch, seq) ──► encoder (12 layers, hidden 768, GELU, 12 heads)
                                   │ last_hidden_state[:, 0]  ([CLS] token)
                                   ▼
                            dropout ─► Linear(768 → 3) ─► logits (batch, 3)

The head is deliberately small: the encoder already produces a sentence
representation, and a deeper head would only add parameters to overfit.

Run it to check shapes and parameter counts (the brief asks for both):

    poetry run python -m src.model
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .config import LABELS, TrainConfig


class SentimentClassifier(nn.Module):
    """Pretrained transformer encoder with a dropout + linear head."""

    def __init__(self, model_name: str, num_labels: int = len(LABELS), dropout: float = 0.1,
                 pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            # Same architecture, random weights: used when loading a checkpoint,
            # whose weights replace these anyway.
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores of shape (batch, num_labels)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = hidden.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Freeze (phase 1) or unfreeze (phase 2) every encoder weight."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not frozen


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts, overall and per block."""
    def count(module, trainable_only=False):
        return sum(p.numel() for p in module.parameters()
                   if p.requires_grad or not trainable_only)

    return {
        "total": count(model),
        "trainable": count(model, trainable_only=True),
        "encoder": count(model.encoder),
        "head": count(model.classifier),
    }


def save_checkpoint(model: SentimentClassifier, config: TrainConfig, path: Path,
                    **metadata) -> None:
    """Save weights plus everything needed to rebuild the model without train.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "labels": LABELS,
            **metadata,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    """Return (model in eval mode, tokenizer, TrainConfig, raw checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    model = SentimentClassifier(config.model_name, dropout=config.dropout, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return model, tokenizer, config, checkpoint


def _shape_check() -> None:
    """Forward pass on a sample batch, printing shapes and parameter counts."""
    config = TrainConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = SentimentClassifier(config.model_name, dropout=config.dropout).eval()

    sample = ["The rides were amazing!", "Too crowded, long queues, never again.",
              "It was fine, nothing special."]
    batch = tokenizer(sample, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["attention_mask"])

    print(f"model         : {config.model_name}")
    print(f"input_ids     : {tuple(batch['input_ids'].shape)}  (batch, seq_len)")
    print(f"logits        : {tuple(logits.shape)}  (batch, {len(LABELS)} classes)")
    assert logits.shape == (len(sample), len(LABELS)), "unexpected output shape"

    encoder_config = model.encoder.config
    print(f"layers        : {encoder_config.num_hidden_layers}")
    print(f"hidden size   : {encoder_config.hidden_size}")
    print(f"attention     : {encoder_config.num_attention_heads} heads")
    print(f"activation    : {encoder_config.hidden_act}")
    for name, value in count_parameters(model).items():
        print(f"params {name:<9}: {value:>12,}")
    model.freeze_encoder(True)
    print(f"trainable when frozen: {count_parameters(model)['trainable']:,}")


if __name__ == "__main__":
    _shape_check()
