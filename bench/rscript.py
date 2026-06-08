"""Cross-platform ``Rscript`` discovery.

The old repo's ``_find_rscript()`` hardcoded a Windows ``C:``/``D:\\Program Files\\R``
scan that silently fails on Linux/macOS (plans/stage0.md, D5). Here we locate
``Rscript`` via the ``PATH`` and an explicit ``CIV_BENCH_RSCRIPT`` override only —
no platform-specific guesses. Used by the R-backed ratings analyses.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional


RSCRIPT_ENV = "CIV_BENCH_RSCRIPT"


def find_rscript(required: bool = False) -> Optional[str]:
    """Return a path to ``Rscript``, or ``None`` if not found.

    Resolution order:
      1. the ``CIV_BENCH_RSCRIPT`` environment override (a full path to the binary), then
      2. ``Rscript`` on the ``PATH``.

    With ``required=True``, a missing ``Rscript`` raises ``RuntimeError`` with an
    install hint (no graceful degradation — see AGENTS.md §Dependencies).
    """
    override = os.environ.get(RSCRIPT_ENV)
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        if required:
            raise RuntimeError(
                f"{RSCRIPT_ENV}={override!r} does not point at an executable Rscript."
            )
        # Fall through to PATH if the override is unusable but not required.

    found = shutil.which("Rscript")
    if found:
        return found

    if required:
        raise RuntimeError(
            "Rscript not found. Install R (https://cran.r-project.org/) and ensure "
            f"Rscript is on PATH, or set {RSCRIPT_ENV} to its full path. The R-backed "
            "ratings analyses (BradleyTerry2 / PlackettLuce) require it."
        )
    return None
