# Team 4 — Workshop #2: Fine-tuned Transformer for Review Sentiment

Machine Learning (Semester 2026-III) — Systems Engineering,
**Universidad Distrital Francisco José de Caldas**. Instructor: Eng. Carlos Andrés Sierra, M.Sc.

Team: Xiomara Salome Arias Arias · Julian David Celis Giraldo · Alejandro Gomez Moyano

Two deep-learning models classify 42,632 Disneyland reviews as
**negative / neutral / positive**, on exactly the split and metrics of Workshop #1.
The number to beat is the Workshop #1 Logistic Regression: **macro-F1 0.6258** on validation.

Two complementary architectures were implemented and compared:

- **Transformer** (`src/`): a pretrained BERT/RoBERTa encoder, fine-tuned. The headline model.
- **BiLSTM + GloVe** (`src/bilstm/`): a bidirectional LSTM over pretrained GloVe word
  embeddings — a recurrent alternative, trained and evaluated on the same split and metrics
  so the two can be compared line for line.

## Repository structure

```
src/
├── config.py       paths, seeds, label order, all hyperparameters (TrainConfig)
├── data.py         loads Workshop #1's split (hash-checked), head+tail tokenisation
├── model.py        Transformer architecture; `python -m src.model` runs the shape check
├── train.py        two-phase fine-tuning, TensorBoard logging, best checkpoint
├── evaluate.py     test-set (or validation) metrics from the checkpoint
├── test_model.py   loads the checkpoint and classifies new reviews
└── bilstm/         BiLSTM + GloVe model (recurrent alternative)
    ├── config.py       BiLSTMConfig; reuses the shared data contract from src.config
    ├── data.py         word-level tokeniser, vocabulary, GloVe embedding matrix
    ├── model.py        BiLSTM architecture; `python -m src.bilstm.model` shape check
    ├── train.py        two-phase training (GloVe frozen -> unfrozen), best checkpoint
    ├── evaluate.py     test-set metrics (reuses src.evaluate's metric definitions)
    └── test_model.py   loads bilstm_best.pt and classifies new reviews
scripts/train_tty.sh   launches training from a text console (see "GPU" below)
checkpoints/        model_best.pt (RoBERTa, ~499 MB) and bilstm_best.pt (~25 MB), via Git LFS
runs/               TensorBoard logs, history.json, eval_<split>_<checkpoint>.json
notebooks/          experimentation — bilstm_experiments.ipynb has the learning curves
data/               silver + gold copied from Workshop #1 (gitignored)
reports/            the workshop briefs
```

## Setup

Python 3.12–3.14, Poetry, and **Git LFS** (the checkpoint is stored with it):

```bash
git lfs install            # once per machine, before cloning
git clone git@github.com:JulianCelisCreator/JulianCelisCreator-ML-NLP-Sentimental-Analysis-W2.git
```

```bash
# AMD GPU (ROCm 7.2) — the configuration the results were produced on:
poetry install --with rocm

# NVIDIA or CPU: skip the ROCm build and install the matching torch yourself, e.g.
poetry install
poetry run pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Data

This repo reuses Workshop #1's cleaned corpus and frozen split instead of re-deriving
them, so the comparison with the baselines is on identical rows. From the Workshop #1
repository, after running its pipeline (`python -m src.preprocessing`):

```bash
mkdir -p data/silver data/gold
cp <workshop1>/ML_NPL/data/silver/reviews.parquet data/silver/
cp <workshop1>/ML_NPL/data/gold/splits.json       data/gold/
```

`src/data.py` refuses to run unless `splits.json` hashes to
`ab3b8e6826127908e471fc8c1dc5f49ab619ac7780fc8e17847b3e26632267cb` and silver has
42,632 rows (29,842 train / 6,395 validation / 6,395 test).

## Reproducing training

**Transformer:**

```bash
poetry run python -m src.model        # forward pass on a sample batch, parameter counts
poetry run python -m src.train        # fine-tune; writes checkpoints/model_best.pt
poetry run python -m src.evaluate     # test-set metrics — run once, at the very end
poetry run python -m src.test_model   # inference on unseen reviews from the checkpoint
poetry run tensorboard --logdir runs  # learning curves
```

**BiLSTM + GloVe:**

```bash
poetry run python -m src.bilstm.model       # forward pass on a sample batch, parameter counts
poetry run python -m src.bilstm.train       # two-phase training; writes checkpoints/bilstm_best.pt
poetry run python -m src.bilstm.evaluate    # test-set metrics — run once, at the very end
poetry run python -m src.bilstm.test_model  # inference on unseen reviews from the checkpoint
```

On first run `src.bilstm.train` downloads the GloVe vectors (`glove-wiki-gigaword-200`,
~250 MB) through gensim and caches them under `~/gensim-data/`.

Either model takes `--limit 64 --device cpu` to run the whole loop on 64 reviews per
split in about a minute: a smoke test, not a result.

### GPU on a machine whose GPU also drives the display (AMD, Linux)

A long training job starves the desktop compositor, the driver resets the GPU, and the
graphical session crashes. Train from a text console instead:

1. `Ctrl+Alt+F3`, log in.
2. `cd` into this repository and run `bash scripts/train_tty.sh`
   (it refuses to start while the desktop is the active session).
3. When it finishes, `Ctrl+Alt+F2` (or F1) returns to the desktop.

## Method

| | |
| --- | --- |
| Model | `bert-base-uncased` encoder + dropout + `Linear(768 → 3)` on `[CLS]`; 109.5M parameters |
| Input | 256 tokens; reviews longer than that keep the first 64 and last 190 tokens (head+tail) |
| Loss | cross-entropy with balanced class weights from the training labels |
| Phase 1 | encoder frozen, head only: 1 epoch, AdamW lr 1e-3 |
| Phase 2 | full network: up to 3 epochs, AdamW lr 2e-5, weight decay 0.01, 10% linear warmup then linear decay |
| Regularisation | dropout 0.1, weight decay, gradient clipping at 1.0, early stopping on validation macro-F1 |
| Precision | bf16 autocast on the GPU |
| Model selection | best validation macro-F1 across both phases; test is never seen during training |
| Seeds | 42 for Python, numpy and torch |

The brief suggests 5–10 frozen epochs then 10–20 fine-tuning epochs. That schedule
comes from CNNs on ImageNet; BERT fine-tuning overfits within a few epochs, so the
two phases are kept but shortened.

## Method — BiLSTM + GloVe (`src/bilstm/`)

| | |
| --- | --- |
| Model | GloVe-200d embeddings → BiLSTM (2 layers, hidden 256, bidirectional) → masked mean+max pooling (1024-d) → dropout → `Linear(1024 → 3)`; ~6.2M parameters |
| Transfer learning | embedding layer initialised from `glove-wiki-gigaword-200`; the recurrent layers and head are trained from scratch (GloVe gives word representations, not a pretrained BiLSTM) |
| Vocabulary | 18,219 words, built from the TRAIN split only (min frequency 2, cap 40k); `<pad>`=0, `<unk>`=1 |
| Input | up to 150 word tokens; dynamic padding per batch, masked everywhere |
| Loss | cross-entropy with balanced class weights from the training labels |
| Phase 1 | GloVe frozen, BiLSTM + head only: 6 epochs, AdamW lr 1e-3 |
| Phase 2 | whole network, GloVe unfrozen: up to 8 epochs, AdamW lr 3e-4 |
| Regularisation | dropout 0.5, weight decay 1e-4, gradient clipping at 5.0, `ReduceLROnPlateau` on validation macro-F1, early stopping |
| Model selection | best validation macro-F1 across both phases; test is never seen during training |
| Seeds | 42 for Python, numpy and torch |

The two phases mirror the Transformer's transfer-learning recipe: first learn to read
sequences while the pretrained vectors stay put, then let the embeddings adapt to the
review vocabulary at a lower learning rate.

## Results — test split (6,395 reviews, scored once after model selection)

| Model | accuracy | AUC-ROC | **macro-F1** |
| --- | --- | --- | --- |
| **RoBERTa-base (Workshop #2, `checkpoints/model_best.pt`)** | **0.8460** | **0.9357** | **0.7130** |
| BERT-base-uncased (Workshop #2, alternative) | 0.8394 | 0.9265 | 0.6973 |
| **BiLSTM + GloVe (Workshop #2, `checkpoints/bilstm_best.pt`)** | 0.7706 | 0.8791 | 0.6137 |
| Logistic Regression (Workshop #1) | 0.7787 | 0.8879 | 0.6169 |
| Multinomial Naive Bayes (Workshop #1) | 0.8013 | 0.8286 | 0.4299 |
| Random Forest (Workshop #1) | 0.7962 | 0.8621 | 0.3037 |
| Majority class (Workshop #1) | 0.7952 | 0.5000 | 0.2953 |

Each model was chosen on validation before its test split was opened (RoBERTa 0.7350,
BERT 0.7187, BiLSTM 0.6414 macro-F1). Reproduce: `python -m src.evaluate` and
`python -m src.bilstm.evaluate` here, and `python -m src.baselines --split test` in the
Workshop #1 repository.

**Reading the comparison.** The BiLSTM reaches macro-F1 0.614 on test, matching the best
Workshop #1 baseline (Logistic Regression, 0.617) and far above Naive Bayes, Random Forest
and the majority-class floor. It does not reach the Transformers (0.71 / 0.70): self-attention
captures long-range context that a recurrent encoder over static word vectors cannot, and it
fine-tunes far more pretrained parameters. Both Workshop #2 models overfit during fine-tuning
(train loss falling while validation loss rises); both were saved at their best-validation
epoch, so the delivered checkpoints generalise (BiLSTM validation 0.641 → test 0.614). The
shared weak spot across every model is the `neutral` class, which leaks into `positive` — see
`notebooks/bilstm_experiments.ipynb` for the learning curves and confusion matrix.

## Related

Workshop #1 (EDA, preprocessing pipeline, classical baselines):
[JulianCelisCreator/ML-NLP-Sentimental-Analysis](https://github.com/JulianCelisCreator/ML-NLP-Sentimental-Analysis).
The silver layer and frozen split this repository trains on are produced there.
