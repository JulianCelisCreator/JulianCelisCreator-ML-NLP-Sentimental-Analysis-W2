#!/usr/bin/env bash
# Train on the GPU WITH the desktop active, killing the run at the first GPU reset.
#
# The RX 9060 XT also drives the display. A short test (2026-10-12) trained with the
# desktop active and no resets, but that is not a guarantee for a long run: this
# script watches the kernel log and stops training the moment the driver reports a
# ring timeout, before the desktop session goes down. If it ever fires, train from a
# text console with scripts/train_tty.sh instead.
#
#   bash scripts/train_watchdog.sh              # full run
#   bash scripts/train_watchdog.sh --lr 3e-5    # any src.train option passes through
#
# Avoid GPU-heavy desktop use (video, games) while it runs.

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
LOG="runs/train_$(date +%Y%m%d-%H%M%S).log"
START=$(date '+%Y-%m-%d %H:%M:%S')

# systemd-inhibit blocks suspend while training: a suspended GPU loses the run.
systemd-inhibit --what=sleep --who="segundo_taller" --why="Training a model on the GPU" \
    .venv/bin/python -m src.train "$@" > "$LOG" 2>&1 &
PID=$!
echo "training (pid $PID), log: $LOG"

while kill -0 "$PID" 2>/dev/null; do
    if journalctl -k --since "$START" --no-pager 2>/dev/null \
            | grep -qE 'ring .* timeout|device wedged'; then
        # $PID is systemd-inhibit; kill the training process under it too.
        kill -9 $(pgrep -P "$PID") "$PID" 2>/dev/null
        echo "ABORTED: the GPU driver reported a reset. Use scripts/train_tty.sh." | tee -a "$LOG"
        exit 2
    fi
    sleep 1
done

wait "$PID"
CODE=$?
echo "training exited with code $CODE"
tr '\r' '\n' < "$LOG" | grep -aE '^(epoch [0-9]+ \[|done in|  new best|  no improvement)' || true
exit "$CODE"
