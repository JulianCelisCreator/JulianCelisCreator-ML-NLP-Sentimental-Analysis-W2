# Notebooks

Report material for Workshop #2. Training itself lives in `src/train.py`, so every number
here is read from saved artifacts (`runs/`, `checkpoints/model_best.pt`) and is
reproducible from the command line.

| Notebook | Brief section it covers |
| --- | --- |
| `01_model_architecture.ipynb` | Model description: layers, dimensions, parameter counts, forward pass with shape checks, hyperparameters |
| `02_training_curves.ipynb` | Training results: loss and validation metrics per epoch, BERT vs RoBERTa |
| `03_test_evaluation.ipynb` | Test-set evaluation and the comparison table against the Workshop #1 baselines |
| `04_error_analysis.ipynb` | Analysis: where the model succeeds and fails (rating, length, park, confidence, examples) |

Each notebook ends with a **Findings** section. Figures are saved to `figures/w2_*.png`
(200 dpi) for the report.

Re-run them all:

```bash
cd notebooks
for nb in 0*.ipynb; do poetry run jupyter nbconvert --to notebook --execute --inplace "$nb"; done
```

`04_error_analysis.ipynb` runs inference on the test split the first time (about 1 minute
on a GPU, about 10 on a CPU) and caches it in `runs/predictions_test_model_best.csv`.
