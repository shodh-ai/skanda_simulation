"""
Shared percolation utilities — used by both cbd_binder.py (Step 4)
and percolation.py (Step 7).

Kept in a separate module to avoid the circular import that would arise
from cbd_binder importing percolation (which already imports cbd_binder
for CBDBinderResult).

Both public functions use scipy.ndimage.label (C-level union-find)
for consistent behaviour and speed (~50ms on 128³ vs ~seconds for
pure-Python BFS).
"""

from __future__ import annotations

from typing import Set

import numpy as np
from scipy.ndimage import label as scipy_label

# 6-connectivity structure (shared, module-level constant)
STRUCT6 = np.zeros((3, 3, 3), dtype=np.int8)
STRUCT6[1, 1, 0] = STRUCT6[1, 1, 2] = 1
STRUCT6[1, 0, 1] = STRUCT6[1, 2, 1] = 1
STRUCT6[0, 1, 1] = STRUCT6[2, 1, 1] = 1


def check_percolates_z(mask: np.ndarray) -> bool:
    """
    Fast True/False check: does `mask` contain at least one connected
    component that spans from Z=0 to Z=nz-1?

    Used by CBD GRF retry loop (Step 4) where only the boolean result
    matters and full diagnostics are not needed.

    Args:
        mask: bool array (nx, ny, nz)

    Returns:
        True if any 6-connected component touches both Z faces.
    """
    if not mask.any():
        return False

    labeled, _ = scipy_label(mask, structure=STRUCT6)

    z0_labels: Set[int] = set(labeled[:, :, 0].ravel()) - {0}
    znz_labels: Set[int] = set(labeled[:, :, -1].ravel()) - {0}

    return bool(z0_labels & znz_labels)


def run_percolation(
    mask: np.ndarray,
) -> tuple[float, int, int, bool]:
    """
    Full percolation analysis with diagnostics.

    Used by PercolationValidator (Step 7) for both electronic and
    ionic network checks.

    Args:
        mask: bool array (nx, ny, nz)

    Returns:
        (percolating_fraction, n_total_components,
         n_percolating_components, percolates)

        percolating_fraction:
            voxels in percolating components / total True voxels.
        n_total_components:
            total number of 6-connected components in mask.
        n_percolating_components:
            number of components touching both Z=0 and Z=nz-1.
        percolates:
            True iff n_percolating_components > 0.
    """
    N_total = int(mask.sum())
    if N_total == 0:
        return 0.0, 0, 0, False

    labeled, n_comp = scipy_label(mask, structure=STRUCT6)

    z0_labels: Set[int] = set(labeled[:, :, 0].ravel()) - {0}
    znz_labels: Set[int] = set(labeled[:, :, -1].ravel()) - {0}

    percolating_labels = z0_labels & znz_labels
    n_perc = len(percolating_labels)

    if n_perc == 0:
        return 0.0, n_comp, 0, False

    perc_mask = np.isin(labeled, list(percolating_labels))
    frac = float(perc_mask.sum()) / N_total

    return frac, n_comp, n_perc, True
