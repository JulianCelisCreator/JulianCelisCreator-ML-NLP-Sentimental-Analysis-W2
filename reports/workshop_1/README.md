# Disneyland Reviews — Sentiment Analysis

The team's problem domain is sentiment analysis on a corpus of **42,656 Disneyland park reviews**, each with a 1-5
rating, a visit month, a reviewer country and the park visited.

## Getting started

Requires Python 3.12 and Poetry, plus a Kaggle API token to download the corpus.

```bash
cd ML_NPL
poetry install

# Kaggle credentials: kaggle.com -> Settings -> API -> Create New Token
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

poetry run python download_dataset.py     # corpus -> data/bronze/ (31 MB, gitignored)
poetry run python -m src.preprocessing    # -> data/silver/ and data/gold/
```

Then open `notebooks/eda.ipynb`, or re-run it headlessly:

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

## Layout

```
ML_NPL/
├── ml_npl/
│   ├── dataset.py   # Kaggle download, schema validation, load_reviews()
│   ├── eda.py       # statistics: quality report, text features, n-grams
│   └── plots.py     # figures on a colour-blind-validated palette
├── notebooks/eda.ipynb
├── download_dataset.py
└── tests/
```

The three modules do not import each other: each takes a DataFrame and returns something,
and the notebook wires them together. Later workshops import `load_reviews()` and
`sentiment_from_rating()` so every stage trains on exactly the same data and labels.

`data/` and `reports/` hold derived output and are gitignored — both regenerate from a
command.

## Progress

| Workshop | Focus | Status |
| --- | --- | --- |
| 0 | Setup, acquisition, EDA | **Done** |
| 1 | Supervised classification | EDA and plan done; preprocessing, split and models pending |
| 2 | RNNs / LSTMs / transformers | Not started |
| 3 | RL for dialogue systems | Not started |
| 4 | Autoencoders for text representation | Not started |
| 5 | GANs for text generation | Not started |
| 6 | Medical text mining | Not started |

## What the EDA established

* A majority-class predictor reaches **79.5% accuracy** without reading a word, so
  **macro-F1** is the headline metric, not accuracy.
* `Review_ID` correlates **+0.9927** with the visit month — it is a timestamp in disguise
  and must be excluded as a feature.
* The park confounds the label: **13.7%** of Paris reviews are negative against
  California's **6.4%**.
* **24 duplicated review texts** must be resolved before splitting, or identical text lands
  on both sides and inflates the score invisibly.
* **No user column exists**, so user leakage cannot be assessed. This is a standing
  limitation of any result from this corpus.

Details, figures and the full leakage analysis are in `notebooks/eda.ipynb`.
