"""Logit / inverse-logit transforms (ported from ``shared/plot_utilities.py``).

These live in :mod:`bench.stats` because they had no home in ``bench`` and are
used by the logit-scale strength adjustment (``bench/adjust/strength.py``). The
clip ``eps`` is pinned to the legacy ``1e-5`` so the adjust stage reproduces the
old ``turn_predicted`` per-row values byte-for-byte (the ``block:"none"`` parity
requirement).
"""

from __future__ import annotations

import numpy as np


# The legacy clip bound (shared/plot_utilities.logit). Pin it so a finite,
# parity-stable logit_strength is produced for relative_strength == 1.0 etc.
LOGIT_EPS = 1e-5


def logit(p, eps: float = LOGIT_EPS):
    """Probability → log-odds, clipping to ``[eps, 1-eps]`` to stay finite."""
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def inv_logit(x):
    """Log-odds → probability."""
    return 1 / (1 + np.exp(-x))
