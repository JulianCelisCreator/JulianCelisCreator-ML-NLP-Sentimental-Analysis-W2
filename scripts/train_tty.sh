#!/usr/bin/env bash
# Launch GPU training from a text console (Ctrl+Alt+F3), never from the desktop.
#
# The RX 9060 XT also drives the display. A long GPU job while the desktop is
# active starves gnome-shell, the driver resets the GPU and the session crashes.
# From a text console the desktop is paused and the GPU is free for training.
#
#   bash scripts/train_tty.sh                 # full run, default hyperparameters
#   bash scripts/train_tty.sh --lr 3e-5       # any src.train option passes through
#   FORCE=1 bash scripts/train_tty.sh         # skip the desktop check (not advised)
#
# Back to the desktop afterwards: Ctrl+Alt+F2 (or F1, depending on the machine).

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python

# 1. Refuse to start while a graphical session is the active one.
if [[ "${FORCE:-0}" != "1" ]]; then
    for session in $(loginctl list-sessions --no-legend | awk '{print $1}'); do
        type=$(loginctl show-session "$session" -p Type --value)
        active=$(loginctl show-session "$session" -p Active --value)
        if [[ "$type" =~ ^(wayland|x11)$ && "$active" == "yes" ]]; then
            echo "A desktop session ($type) is active. Switch to a text console first:"
            echo "  Ctrl+Alt+F3, log in, then run this script again."
            exit 1
        fi
    done
fi

mkdir -p runs
LOG="runs/train_$(date +%Y%m%d-%H%M%S).log"

# 2. Short GPU sanity check: compute must work and agree with the CPU.
"$PYTHON" - <<'EOF' 2>&1 | grep -v "Failed to initialize NumPy" | tee "$LOG"
import torch
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
x = torch.randn(1024, 1024)
gpu = (x.cuda() @ x.cuda()).cpu()
assert torch.allclose(gpu, x @ x, rtol=1e-3, atol=1e-2), "GPU result differs from CPU"
print(f"GPU OK: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
EOF

# 3. Train. The log keeps everything; TensorBoard data goes to runs/<run>/.
echo "training... log: $LOG"
"$PYTHON" -m src.train "$@" 2>&1 | tee -a "$LOG"

echo
echo "Done. Back to the desktop with Ctrl+Alt+F2 (or F1)."
echo "Learning curves: .venv/bin/tensorboard --logdir runs"
