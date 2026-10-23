"""Shared figure style for the Workshop #2 notebooks.

One palette and one set of helpers, so every figure in the report reads as the
same system. Colours come from a palette validated for colour-vision
deficiency; identity is never carried by colour alone (legends or direct
labels are always present).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import LABELS, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b3"  # de-emphasised marks (e.g. baselines next to the final model)

# Categorical slots in fixed order. The first three validate on every pair, so
# they are safe for any chart with up to three series.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MODEL_COLORS = {"RoBERTa-base": SERIES[0], "BERT-base": SERIES[1]}
CLASS_COLORS = dict(zip(LABELS, SERIES))

# Sequential single-hue ramp for confusion matrices: counts are magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list("w2_seq", ["#fcfcfb", "#86b6ef", "#184f95"])


def apply_theme() -> None:
    """Recessive axes and grid, thin marks, readable text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    """Write a figure to notebooks/figures/<name>.png for the report."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")


def _percent(share: float) -> str:
    """'<1%' rather than a misleading '0%' for small but non-empty cells."""
    return "<1%" if 0 < share < 0.005 else f"{share:.0%}"


def confusion_heatmap(matrix, ax, title: str, show_ylabel: bool = True):
    """Row-normalised shading (reads as recall), raw counts printed in each cell."""
    matrix = np.asarray(matrix)
    totals = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals != 0)
    image = ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    if show_ylabel:
        ax.set_ylabel("True")
    ax.grid(False)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{matrix[i, j]:,}\n{_percent(share[i, j])}", ha="center", va="center",
                    fontsize=9, color="white" if share[i, j] > 0.55 else INK)
    ax.set_title(title)
    return image
