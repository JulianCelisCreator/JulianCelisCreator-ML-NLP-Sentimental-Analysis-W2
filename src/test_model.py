"""Load the saved checkpoint and predict on new reviews.

Proves checkpoints/model_best.pt is self-contained: nothing from train.py is
needed to rebuild the model and classify text it has never seen.

    poetry run python -m src.test_model
    poetry run python -m src.test_model "Loved it!" "Queues were endless."
"""

from __future__ import annotations

import argparse

import torch

from .config import BEST_CHECKPOINT, LABELS, get_device
from .data import encode_head_tail, make_collate
from .model import load_checkpoint

# Written for this script, not taken from the corpus, so they are truly unseen.
DEFAULT_REVIEWS = [
    "Absolutely magical day, the kids loved every ride and the staff were wonderful.",
    "It was okay. Some rides were fun but the queues were long and food was overpriced.",
    "Worst theme park experience ever. Rude staff, broken rides, and filthy toilets.",
]


def predict_texts(texts, checkpoint=BEST_CHECKPOINT, device=None):
    """Return a list of (label, {label: probability}) for each text."""
    device = get_device(device)
    model, tokenizer, config, _ = load_checkpoint(checkpoint, device)
    sequences = encode_head_tail(texts, tokenizer, config.max_len, config.head_tokens)
    batch = make_collate(tokenizer.pad_token_id)([(ids, 0) for ids in sequences])
    with torch.no_grad():
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    probabilities = torch.softmax(logits.float(), dim=-1).cpu()
    return [
        (LABELS[int(row.argmax())], {label: float(p) for label, p in zip(LABELS, row)})
        for row in probabilities
    ]


def main() -> int:
    """Classify the reviews given on the command line (or three built-in ones)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reviews", nargs="*", default=DEFAULT_REVIEWS)
    parser.add_argument("--checkpoint", default=str(BEST_CHECKPOINT))
    parser.add_argument("--device", default="cpu", help="cpu is enough for a few reviews")
    args = parser.parse_args()

    predictions = predict_texts(args.reviews, args.checkpoint, args.device)
    assert len(predictions) == len(args.reviews), "one prediction per review expected"
    for text, (label, probabilities) in zip(args.reviews, predictions):
        assert abs(sum(probabilities.values()) - 1) < 1e-4, "probabilities must sum to 1"
        shown = " ".join(f"{k} {v:.2f}" for k, v in probabilities.items())
        print(f"{label:>8}  [{shown}]  {text[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
