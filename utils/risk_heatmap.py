# utils/risk_heatmap.py
from pathlib import Path
from typing import List, Dict, Tuple
import math

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Map common textual levels → numeric (1..5)
_LEVEL_MAP = {
    "vlow": 1, "very low": 1, "very-low": 1, "1": 1,
    "low": 2, "2": 2,
    "medium": 3, "med": 3, "moderate": 3, "3": 3,
    "high": 4, "4": 4,
    "vhigh": 5, "very high": 5, "very-high": 5, "critical": 5, "5": 5
}

def _to_1_5(v) -> int:
    if isinstance(v, (int, float)):
        return max(1, min(5, int(round(v))))
    if isinstance(v, str):
        s = v.strip().lower()
        return _LEVEL_MAP.get(s, 3)
    return 3

def _matrix_counts(risks: List[Dict]) -> Tuple[np.ndarray, Dict[Tuple[int,int], list]]:
    grid = np.zeros((5, 5), dtype=int)  # [impact-1, likelihood-1]
    bucket: Dict[Tuple[int,int], list] = {}
    for r in risks or []:
        i = _to_1_5(r.get("impact"))
        l = _to_1_5(r.get("likelihood"))
        grid[i-1, l-1] += 1
        bucket.setdefault((i, l), []).append(r)
    return grid, bucket

def generate_risk_heatmap(
    risks: List[Dict],
    out_path: str,
    *,
    title: str = "Risk Heatmap (Impact × Likelihood)"
) -> str:
    """
    Builds a 5x5 heatmap image and writes it to out_path.
    Returns the out_path on success.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grid, buckets = _matrix_counts(risks)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(grid, origin="lower")  # default colormap

    # Axis labels (1..5)
    ax.set_xticks(range(5))
    ax.set_xticklabels([1,2,3,4,5])
    ax.set_yticks(range(5))
    ax.set_yticklabels([1,2,3,4,5])

    ax.set_xlabel("Likelihood →")
    ax.set_ylabel("Impact →")
    ax.set_title(title)

    # Annotate counts in each cell
    for i in range(5):
        for j in range(5):
            val = int(grid[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)