"""Paths and default hyperparameters for the BiLSTM + GloVe model.

The data contract (silver parquet, the hash-checked Workshop #1 split, the label
order and the seeds) is shared with the transformer and imported from
``src.config`` rather than copied, so the two models cannot drift apart and the
comparison in the report stays honest. Only what is specific to the recurrent
model lives here: its own checkpoint, the GloVe settings and ``BiLSTMConfig``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Re-exported so every BiLSTM module imports the shared contract from one place.
from ..config import (  # noqa: F401
    CHECKPOINT_DIR,
    DATA_DIR,
    LABEL_TO_ID,
    LABELS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RUNS_DIR,
    SILVER_PATH,
    SILVER_ROWS,
    SPLITS_PATH,
    SPLITS_SHA256,
    TARGET_COL,
    TEXT_COL,
    get_device,
    set_seed,
)

# The recurrent model's own checkpoint, next to the transformer's model_best.pt.
BEST_CHECKPOINT = CHECKPOINT_DIR / "bilstm_best.pt"

# GloVe lives under data/ alongside silver and gold; it is downloaded once and
# cached (gitignored, like the rest of data/). gensim exposes the Wikipedia +
# Gigaword vectors as "glove-wiki-gigaword-<dim>".
GLOVE_DIR = DATA_DIR / "glove"
GLOVE_DIM = 200
GLOVE_NAME = f"glove-wiki-gigaword-{GLOVE_DIM}"

# Reserved rows 0 and 1 of the embedding matrix. <pad> must stay all-zeros and is
# masked everywhere; <unk> catches words missing from both the corpus and GloVe.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


@dataclass
class BiLSTMConfig:
    """Every hyperparameter of a run. Saved inside the checkpoint and the run log."""

    # --- embeddings / transfer learning ---------------------------------------
    glove_name: str = GLOVE_NAME
    embedding_dim: int = GLOVE_DIM
    # Phase 1 freezes GloVe (transfer learning: only the recurrent layers and the
    # head learn). Phase 2 unfreezes it at a smaller lr to adapt to the reviews.
    freeze_embeddings: bool = True

    # --- vocabulary -----------------------------------------------------------
    # Built from the TRAIN split only. A word must appear at least this many times
    # to earn a row; rarer words fall back to <unk>, which keeps the matrix small
    # and avoids memorising hapax legomena.
    min_freq: int = 2
    max_vocab: int = 40_000
    max_len: int = 300  # reviews are truncated to this many word tokens

    # --- recurrent core -------------------------------------------------------
    hidden_size: int = 256
    num_layers: int = 2
    # Dropout between stacked LSTM layers (PyTorch applies it only when num_layers > 1).
    lstm_dropout: float = 0.3
    # Dropout on the pooled sentence vector, just before the linear head.
    dropout: float = 0.5

    # --- optimisation ---------------------------------------------------------
    batch_size: int = 64
    eval_batch_size: int = 128
    # Phase 1: GloVe frozen, recurrent layers + head learn.
    frozen_epochs: int = 6
    lr: float = 1e-3
    # Phase 2: whole network, GloVe unfrozen at a smaller lr.
    finetune_epochs: int = 8
    finetune_lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # ReduceLROnPlateau on validation macro-F1 (mode="max").
    lr_patience: int = 1
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    # Early stopping: stop a phase when macro-F1 has not improved this many epochs.
    patience: int = 3

    seed: int = RANDOM_STATE
    # Debug knob: keep only the first N rows of each split (0 = all of them).
    limit: int = 0

    def to_dict(self) -> dict:
        """Plain-dict view, for checkpoints and logs."""
        return asdict(self)
