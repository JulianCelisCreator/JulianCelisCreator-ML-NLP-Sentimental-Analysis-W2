# Checkpoints

`model_best.pt` is the final Workshop #2 model: RoBERTa-base, best validation macro-F1
(0.7350, epoch 4). It holds the weights plus the training config and label order, so
`python -m src.test_model` and `python -m src.evaluate` rebuild it without `train.py`.

It is ~499 MB and stored with **Git LFS**: install `git-lfs` before cloning, or run
`git lfs pull` afterwards. Other checkpoints written by `src.train` stay local.
