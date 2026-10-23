"""Model architecture: GloVe embeddings + a BiLSTM + a small classification head.

    input_ids (batch, seq) ──► Embedding (GloVe-initialised, row 0 = <pad>)
                                   │ masked by lengths
                                   ▼
                            BiLSTM (num_layers, hidden H, bidirectional)
                                   │ per-timestep states (batch, seq, 2H)
                                   ▼
                 masked mean-pool ⊕ masked max-pool over time  → (batch, 4H)
                                   ▼
                            dropout ─► Linear(4H → 3) ─► logits (batch, 3)

Why BiLSTM, and why this head. A review's verdict can sit anywhere in the text,
so reading it in both directions and pooling over all timesteps captures more
than the last hidden state alone. Concatenating mean- and max-pool is a cheap,
well-worn trick: mean summarises the whole review, max picks out its most
sentiment-bearing word. The head is deliberately tiny — the recurrent layers
already produce the sentence representation.

Transfer learning lives in the embedding layer: it is initialised from GloVe and
frozen in phase 1, then unfrozen in phase 2 (see train.py). <pad> is masked
throughout, so its all-zero row never reaches the loss.

Run it to check shapes and parameter counts (the brief asks for both):

    poetry run python -m src.bilstm.model
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import LABELS, PAD_ID, BiLSTMConfig


class BiLSTMClassifier(nn.Module):
    """GloVe embedding layer, a bidirectional LSTM, mean+max pooling, linear head."""

    def __init__(self, vocab_size: int, config: BiLSTMConfig,
                 embedding_weights: torch.Tensor | None = None,
                 num_labels: int = len(LABELS)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        self.freeze_embeddings(config.freeze_embeddings)

        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        # mean-pool ⊕ max-pool over a bidirectional LSTM → 2 * (2 * hidden).
        self.classifier = nn.Linear(4 * config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores of shape (batch, num_labels)."""
        embedded = self.embedding(input_ids)

        # Pack so the LSTM skips padding; lengths must be on the CPU for packing.
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Mask padding before pooling: mean over real tokens only, max with padding
        # pushed to -inf so it can never be selected.
        mask = (input_ids != PAD_ID).unsqueeze(-1)
        summed = (outputs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        mean_pool = summed / counts
        max_pool = outputs.masked_fill(~mask, float("-inf")).max(dim=1).values

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(self.dropout(pooled))

    def freeze_embeddings(self, frozen: bool = True) -> None:
        """Freeze (phase 1) or unfreeze (phase 2) the GloVe embedding matrix."""
        self.embedding.weight.requires_grad = not frozen


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts, overall and per block."""
    def count(module, trainable_only=False):
        return sum(p.numel() for p in module.parameters()
                   if p.requires_grad or not trainable_only)

    return {
        "total": count(model),
        "trainable": count(model, trainable_only=True),
        "embedding": count(model.embedding),
        "lstm": count(model.lstm),
        "head": count(model.classifier),
    }


def save_checkpoint(model: BiLSTMClassifier, config: BiLSTMConfig, vocab, path: Path,
                    **metadata) -> None:
    """Save weights plus everything needed to rebuild the model without train.py.

    The vocabulary is part of the checkpoint: without it the saved embedding rows
    cannot be matched back to words, so inference would be impossible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "vocab": vocab.to_dict(),
            "labels": LABELS,
            **metadata,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    """Return (model in eval mode, vocab, BiLSTMConfig, raw checkpoint dict)."""
    from .data import Vocab  # local import avoids a model <-> data import cycle

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = BiLSTMConfig(**checkpoint["config"])
    vocab = Vocab.from_dict(checkpoint["vocab"])
    # No GloVe needed here: the trained embedding rows come from the state dict.
    model = BiLSTMClassifier(len(vocab), config, embedding_weights=None)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, vocab, config, checkpoint


def _shape_check() -> None:
    """Forward pass on a toy batch, printing shapes and parameter counts."""
    from .data import make_collate, tokenize

    config = BiLSTMConfig()
    sample = ["The rides were amazing!", "Too crowded, long queues, never again.",
              "It was fine, nothing special."]
    # A tiny stand-in vocab so the check runs without building the real one.
    tokens = {tok for text in sample for tok in tokenize(text)}
    itos = ["<pad>", "<unk>", *sorted(tokens)]
    stoi = {tok: i for i, tok in enumerate(itos)}
    sequences = [[stoi.get(t, 1) for t in tokenize(text)] for text in sample]
    batch = make_collate()([(ids, 0) for ids in sequences])

    model = BiLSTMClassifier(len(itos), config).eval()
    with torch.no_grad():
        logits = model(batch["input_ids"], batch["lengths"])

    print(f"embedding dim : {config.embedding_dim} (GloVe {config.glove_name})")
    print(f"input_ids     : {tuple(batch['input_ids'].shape)}  (batch, seq_len)")
    print(f"logits        : {tuple(logits.shape)}  (batch, {len(LABELS)} classes)")
    assert logits.shape == (len(sample), len(LABELS)), "unexpected output shape"

    print(f"hidden size   : {config.hidden_size} x {config.num_layers} layers, bidirectional")
    for name, value in count_parameters(model).items():
        print(f"params {name:<9}: {value:>12,}")
    model.freeze_embeddings(True)
    print(f"trainable when embeddings frozen: {count_parameters(model)['trainable']:,}")


if __name__ == "__main__":
    _shape_check()
