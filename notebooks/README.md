
# Notebooks
Experimentation and learning-curve figures. Training itself lives in `src/train.py`
and `src/bilstm/train.py`, so every reported number is reproducible from the command line.

- `bilstm_experiments.ipynb` — the BiLSTM + GloVe model: architecture summary, learning
  curves (loss and metrics per epoch), per-class recall, the test confusion matrix, and the
  comparison against the Workshop #1 baselines and the Transformer. It reads the artifacts in
  `runs/` (it does not retrain) and writes its figures to `notebooks/figures/`.es in `src/train.py`
so every reported number is reproducible from the command line.
