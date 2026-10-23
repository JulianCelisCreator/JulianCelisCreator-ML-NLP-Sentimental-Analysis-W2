# Checkpoints

`model_best.pt` is the final Workshop #2 model: RoBERTa-base, best validation macro-F1
(0.7350, epoch 4). It holds the weights plus the training config and label order, so
`python -m src.test_model` and `python -m src.evaluate` rebuild it without `train.py`.

It is ~499 MB and stored with **Git LFS**: install `git-lfs` before cloning, or run
`git lfs pull` afterwards. Other checkpoints written by `src.train` stay local.

`bilstm_best.pt` is the recurrent alternative: BiLSTM + GloVe-200d, best validation
macro-F1 (0.6414, epoch 8). It holds the weights, the training config, the vocabulary
and the label order, so `python -m src.bilstm.test_model` and `python -m src.bilstm.evaluate`
rebuild it without `train.py` and without re-downloading GloVe. At ~25 MB it is small
enough that Git LFS is optional.
