"""Load the saved Keras LSTM checkpoint and predict on new raw reviews.

Proves the model can classify completely unseen text. Note: since the tokenizer
was not serialized separately during training, this script quickly rebuilds it
from the training data before running inference to ensure exact vocabulary match.

Usage:
    poetry run python -m src.test_model
    poetry run python -m src.test_model "Loved it!" "Queues were endless."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tf_keras as keras
from tf_keras.preprocessing.sequence import pad_sequences

# Import dependencies from our standalone data pipeline
from .data import prepare_lstm_data, LABELS, normalize_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "model_best.keras"

# Default reviews exactly as used by peers, to prove it works out-of-the-box
DEFAULT_REVIEWS = [
    "Absolutely magical day, the kids loved every ride and the staff were wonderful.",
    "It was okay. Some rides were fun but the queues were long and food was overpriced.",
    "Worst theme park experience ever. Rude staff, broken rides, and filthy toilets.",
]


def predict_texts(texts: list[str], model, tokenizer, max_len: int) -> list[tuple[str, dict]]:
    """Normalizes, tokenizes, pads, and predicts the sentiment of raw texts."""
    
    # 1. Normalize text (lowercase, remove URLs)
    cleaned_texts = [normalize_text(text) for text in texts]
    
    # 2. Convert to numeric sequences based on the fitted vocabulary
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    
    # 3. Apply padding to match the model's expected input shape
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    
    # 4. Predict probabilities (verbose=0 to keep console clean)
    probabilities = model.predict(padded, verbose=0)
    
    # 5. Format results into (Label, {class: probability})
    results = []
    for row in probabilities:
        predicted_idx = int(np.argmax(row))
        label = LABELS[predicted_idx]
        probs_dict = {lbl: float(p) for lbl, p in zip(LABELS, row)}
        results.append((label, probs_dict))
        
    return results


def main() -> int:
    """Classify the reviews given on the command line (or three built-in ones)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reviews", nargs="*", default=DEFAULT_REVIEWS)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    # 1. Get the fitted tokenizer from data.py
    # (In a production system, you'd load the tokenizer from a .pkl file,
    # but rebuilding it here from the training split guarantees vocabulary sync).
    print("Loading vocabulary/tokenizer from training data (this might take a few seconds)...")
    data = prepare_lstm_data()
    tokenizer = data["tokenizer"]
    max_len = data["max_len"]

    # 2. Load the trained Keras model
    print(f"Loading Keras model from {args.checkpoint}...\n")
    try:
        model = keras.models.load_model(args.checkpoint)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return 1

    # 3. Predict on unseen reviews
    predictions = predict_texts(args.reviews, model, tokenizer, max_len)
    
    # 4. Print results matching the peers' elegant format
    assert len(predictions) == len(args.reviews), "one prediction per review expected"
    for text, (label, probabilities) in zip(args.reviews, predictions):
        # Verify probabilities sum to 1 (accounting for floating point errors)
        assert abs(sum(probabilities.values()) - 1) < 1e-4, "probabilities must sum to 1"
        
        # Format the probabilities dictionary into a clean string
        shown = " ".join(f"{k} {v:.2f}" for k, v in probabilities.items())
        print(f"{label:>8}  [{shown}]  {text[:70]}")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())