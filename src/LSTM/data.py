"""Standalone data preparation module for the LSTM model.

This module handles loading the silver data, splitting based on the gold layer,
text normalization, sequential tokenization, padding, and calculating class
weights for handling class imbalance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.preprocessing.text import Tokenizer

# -----------------------------------------------------------------------------
# PATHS AND CONSTANTS
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SILVER_PATH = DATA_DIR / "silver" / "reviews.parquet"
SPLITS_PATH = DATA_DIR / "gold" / "splits.json"

TEXT_COL = "Review_Text"  # Ensure this matches your parquet column name exactly
TARGET_COL = "sentiment"
LABELS = ["negative", "neutral", "positive"]

MAX_WORDS = 10000  # Vocabulary size (most frequent words)
MAX_LEN = 200      # Maximum review length in words


# -----------------------------------------------------------------------------
# LOADING AND NORMALIZATION FUNCTIONS
# -----------------------------------------------------------------------------

def normalize_text(text):
    """Converts to lowercase and removes URLs."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    return text


def load_raw_text_splits() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """Loads the Silver layer and splits it using the Gold layer indices."""
    
    # 1. Read the Parquet file (Silver layer)
    try:
        clean = pd.read_parquet(SILVER_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Silver file not found at {SILVER_PATH}. Make sure to run Workshop 1 baseline first."
        )

    # Restore data types
    if "period" in clean.columns:
        clean["period"] = clean["period"].dt.to_period("M")
    clean[TARGET_COL] = clean[TARGET_COL].astype("category")

    # 2. Read the partition indices (Gold layer)
    try:
        with SPLITS_PATH.open(encoding="utf-8") as file:
            splits = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Gold splits file not found at {SPLITS_PATH}.")

    def subset(name):
        rows = clean.loc[splits[name]]
        return rows[TEXT_COL].values, rows[TARGET_COL]

    # Extract texts and labels
    x_train_text, y_train = subset("train")
    x_val_text, y_val = subset("validation")
    x_test_text, y_test = subset("test")

    # Normalize the text arrays individually
    x_train_norm = np.array([normalize_text(text) for text in x_train_text])
    x_val_norm = np.array([normalize_text(text) for text in x_val_text])
    x_test_norm = np.array([normalize_text(text) for text in x_test_text])

    return x_train_norm, x_val_norm, x_test_norm, y_train, y_val, y_test


# -----------------------------------------------------------------------------
# LSTM PREPARATION PIPELINE
# -----------------------------------------------------------------------------

def compute_class_weights(y_train_encoded: np.ndarray) -> dict[int, float]:
    """Computes class weights to mitigate dataset imbalance."""
    class_counts = np.bincount(y_train_encoded)
    total_samples = len(y_train_encoded)
    n_classes = len(LABELS)
    
    class_weights = {
        i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)
    }
    return class_weights


def prepare_lstm_data() -> dict:
    """Executes the entire pipeline and returns tensors ready for model.fit() and evaluate()."""
    
    # 1. Load normalized base data
    print("Loading and normalizing data from parquet (Silver) and JSON (Gold)...")
    x_train_text, x_val_text, x_test_text, y_train, y_val, y_test = load_raw_text_splits()

    # 2. Encode labels to integers (0, 1, 2)
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    y_train_encoded = y_train.map(label_to_index).to_numpy()
    y_val_encoded = y_val.map(label_to_index).to_numpy()
    y_test_encoded = y_test.map(label_to_index).to_numpy()

    # 3. Compute class weights (ONLY with train data)
    print("Computing class weights for training...")
    class_weights_numeric = compute_class_weights(y_train_encoded)

    # 4. Tokenization (Fit ONLY on train data)
    print("Fitting tokenizer and converting texts to sequences...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_text)

    # Convert to sequences
    x_train_seq = tokenizer.texts_to_sequences(x_train_text)
    x_val_seq = tokenizer.texts_to_sequences(x_val_text)
    x_test_seq = tokenizer.texts_to_sequences(x_test_text)

    # 5. Apply Padding to 200 tokens
    print(f"Applying padding to standardize sequences to {MAX_LEN} tokens...")
    X_train_pad = pad_sequences(x_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_val_pad = pad_sequences(x_val_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(x_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    print("Data pipeline completed successfully.\n")

    return {
        "x_train": X_train_pad,
        "y_train": y_train_encoded,
        "x_val": X_val_pad,
        "y_val": y_val_encoded,
        "x_test": X_test_pad,
        "y_test": y_test_encoded,
        "y_val_series": y_val,   # Original string-type labels
        "y_test_series": y_test, # Original string-type labels
        "class_weights": class_weights_numeric,
        "tokenizer": tokenizer,
        "vocab_size": min(MAX_WORDS, len(tokenizer.word_index) + 1),
        "max_len": MAX_LEN
    }