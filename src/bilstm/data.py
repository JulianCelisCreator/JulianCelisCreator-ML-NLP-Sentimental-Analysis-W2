"""Data for the BiLSTM: Workshop #1's split, tokenised into words, plus GloVe.

The split, the silver layer, the label order and the balanced class weights are
the transformer's, imported from ``src.data`` unchanged, so both models see
exactly the same rows and the comparison is fair. What is new here is everything
the recurrent model needs that a subword tokenizer gave the transformer for free:

* a word-level tokenizer,
* a vocabulary built from the TRAIN split only,
* an embedding matrix whose rows are GloVe vectors, aligned to that vocabulary.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# The data contract is shared with the transformer; reuse it rather than copy it.
from ..data import class_weights, encode_labels, load_split_frames
from .config import (
    PAD_ID,
    PAD_TOKEN,
    TARGET_COL,
    TEXT_COL,
    UNK_ID,
    UNK_TOKEN,
    BiLSTMConfig,
)

SPLIT_NAMES = ("train", "validation", "test")

# Words, contractions and standalone punctuation that carries sentiment ("!", "?").
# Lowercased to match GloVe's lowercase vocabulary.
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?]")


def tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenizer aligned to GloVe's vocabulary.

    GloVe (Wikipedia + Gigaword) is lowercase and has no subwords, so we lowercase
    and split on words, keeping "!" and "?" because they carry sentiment. Anything
    out of vocabulary becomes <unk> at encoding time.
    """
    return _TOKEN_RE.findall(str(text).lower())


class Vocab:
    """Word-to-id map built from the training split, with <pad>=0 and <unk>=1."""

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {token: index for index, token in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map tokens to ids, sending anything unseen to <unk>."""
        unk = UNK_ID
        return [self.stoi.get(token, unk) for token in tokens]

    @classmethod
    def build(cls, token_lists, min_freq: int, max_vocab: int) -> "Vocab":
        """Vocabulary from TRAIN tokens: frequent words first, capped at max_vocab.

        <pad> and <unk> take rows 0 and 1; the remaining budget goes to the most
        frequent words that clear ``min_freq``. Ordering by frequency means a
        smaller ``max_vocab`` simply drops the rarest words.
        """
        counter: Counter[str] = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        itos = [PAD_TOKEN, UNK_TOKEN]
        budget = max_vocab - len(itos)
        for token, count in counter.most_common():
            if count < min_freq or budget <= 0:
                break
            itos.append(token)
            budget -= 1
        return cls(itos)

    def to_dict(self) -> dict:
        """Serialisable form stored in the checkpoint (itos is enough to rebuild)."""
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocab":
        """Rebuild a vocab saved by ``to_dict``."""
        return cls(list(payload["itos"]))


def _load_glove_keyed_vectors(glove_name: str):
    """Download (once, cached) and return GloVe vectors via gensim's downloader."""
    import gensim.downloader as api  # imported lazily: only train/eval need it

    return api.load(glove_name)


def build_embedding_matrix(vocab: Vocab, config: BiLSTMConfig, seed: int) -> torch.Tensor:
    """A (vocab, dim) matrix: GloVe where available, else a small random vector.

    This is where transfer learning enters. Rows for words GloVe knows are copied
    from GloVe; words it does not know (and <unk>) get a small random vector so
    the model can still learn something for them. <pad> stays all-zeros and is
    masked everywhere, so its row never affects the loss.
    """
    rng = np.random.default_rng(seed)
    dim = config.embedding_dim
    # Small random init on the scale GloVe vectors live at, for the unknown rows.
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0

    glove = _load_glove_keyed_vectors(config.glove_name)
    if glove.vector_size != dim:
        msg = f"{config.glove_name} is {glove.vector_size}d, config expects {dim}d."
        raise ValueError(msg)

    found = 0
    for index, token in enumerate(vocab.itos):
        if index in (PAD_ID, UNK_ID):
            continue
        if token in glove:
            matrix[index] = glove[token]
            found += 1
    coverage = found / max(len(vocab) - 2, 1)
    print(f"GloVe coverage: {found:,}/{len(vocab) - 2:,} vocab words ({coverage:.1%})")
    return torch.from_numpy(matrix)


class ReviewDataset(Dataset):
    """Pre-tokenised, pre-encoded reviews and integer labels."""

    def __init__(self, sequences: list[list[int]], labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], int(self.labels[index])


def make_collate():
    """Pad each batch to its own longest review; return ids, lengths and labels.

    Lengths are kept so the model can pack the sequence and so pooling can ignore
    the padding. Padding uses PAD_ID (0), whose embedding row is all-zeros.
    """

    def collate(batch):
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(ids) for ids in sequences], dtype=torch.long)
        width = int(lengths.max())
        input_ids = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
        for row, ids in enumerate(sequences):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _encode_frame(frame: pd.DataFrame, vocab: Vocab, max_len: int) -> list[list[int]]:
    """Tokenise, truncate to max_len words, and map to ids. Empty reviews keep <unk>."""
    sequences = []
    for text in frame[TEXT_COL]:
        tokens = tokenize(text)[:max_len]
        ids = vocab.encode(tokens) if tokens else [UNK_ID]
        sequences.append(ids)
    return sequences


def build_loaders(config: BiLSTMConfig, vocab: Vocab | None = None, splits=SPLIT_NAMES):
    """Return ({split: DataLoader}, train labels, vocab).

    When ``vocab`` is None it is built from the training split (training path);
    evaluation and inference pass the vocab saved in the checkpoint so the ids
    line up with the embedding matrix the model was trained with.
    """
    frames = load_split_frames(config.limit)
    if vocab is None:
        train_tokens = [tokenize(text) for text in frames["train"][TEXT_COL]]
        vocab = Vocab.build(train_tokens, config.min_freq, config.max_vocab)
        print(f"vocabulary: {len(vocab):,} words "
              f"(min_freq {config.min_freq}, cap {config.max_vocab:,})")

    collate = make_collate()
    generator = torch.Generator().manual_seed(config.seed)

    loaders = {}
    for name in splits:
        frame = frames[name]
        dataset = ReviewDataset(
            _encode_frame(frame, vocab, config.max_len),
            encode_labels(frame[TARGET_COL]),
        )
        is_train = name == "train"
        loaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size if is_train else config.eval_batch_size,
            shuffle=is_train,
            collate_fn=collate,
            generator=generator if is_train else None,
        )
    return loaders, encode_labels(frames["train"][TARGET_COL]), vocab


__all__ = [
    "Vocab",
    "tokenize",
    "build_embedding_matrix",
    "build_loaders",
    "make_collate",
    "class_weights",
]
