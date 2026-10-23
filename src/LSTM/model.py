"""Model architecture definition for the LSTM baseline.

This module defines the Deep Learning architecture used for the sentiment
analysis task. It implements a Sequential model with an Embedding layer,
Spatial Dropout, an LSTM core, and a deep Dense classifier.

Framework: tf_keras (TensorFlow/Keras)
"""

from __future__ import annotations

import tf_keras as keras
from tf_keras import layers, models

def build_lstm_model(
    vocab_size: int, 
    max_len: int, 
    num_classes: int = 3
) -> models.Sequential:
    """Builds and compiles the LSTM neural network architecture.

    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        max_len (int): The maximum length of the padded input sequences.
        num_classes (int): The number of output classes for classification.

    Returns:
        models.Sequential: A compiled Keras Sequential model ready for training.
    """
    model = models.Sequential([
        # 1. Embedding Layer: Maps word indices to dense vectors of fixed size
        layers.Embedding(
            input_dim=vocab_size, 
            output_dim=128, 
            input_length=max_len
        ),
        
        # 2. Spatial Dropout: Drops entire 1D feature maps to promote independence
        layers.SpatialDropout1D(0.2),
        
        # 3. Recurrent Core: LSTM layer with both standard and recurrent dropout
        layers.LSTM(
            units=64, 
            dropout=0.2, 
            recurrent_dropout=0.2
        ),
        
        # 4. Deep Classifier Head: Gradually reduces dimensionality
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        
        # 5. Output Layer: Softmax for multi-class probability distribution
        layers.Dense(num_classes, activation="softmax")
    ])

    # Compile the model with Adam optimizer and sparse categorical crossentropy
    # since labels are provided as integers (0, 1, 2) instead of one-hot vectors.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    # Quick shape and parameter check (can be run directly to verify the architecture)
    dummy_vocab_size = 10000
    dummy_max_len = 200
    
    print("Building LSTM model for shape check...")
    test_model = build_lstm_model(vocab_size=dummy_vocab_size, max_len=dummy_max_len)
    test_model.summary()