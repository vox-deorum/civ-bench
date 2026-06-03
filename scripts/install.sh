#!/usr/bin/env bash
# Install every civ-bench dependency and verify it imports. Fails loudly if anything is missing.
#
# There is no graceful degradation in civ-bench: torch / xgboost / optuna / R are all required.
# Run this once before `civ-bench run`. Re-run after pulling new dependencies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${CIV_BENCH_PYTHON:-python3}"

echo "Using ${PYTHON} ($("${PYTHON}" --version 2>&1))"

# --- Python dependencies (all required — no optional extras) ----------------
PY_DEPS=(
    pandas numpy scipy statsmodels matplotlib seaborn plotly
    scikit-learn tabulate
    torch xgboost optuna imbalanced-learn
)

echo
echo "[1/4] Upgrading pip"
"${PYTHON}" -m pip install --upgrade pip

echo
echo "[2/4] Installing Python dependencies"
"${PYTHON}" -m pip install --upgrade "${PY_DEPS[@]}"

if [ -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "Installing civ-bench (editable)"
    "${PYTHON}" -m pip install -e "${REPO_ROOT}"
else
    echo "No pyproject.toml yet — skipping editable install of civ-bench."
fi

# --- Verify every Python import --------------------------------------------
echo
echo "[3/4] Verifying Python imports"
"${PYTHON}" - <<'PYEOF'
import importlib, sys
mods = ['pandas','numpy','scipy','statsmodels','matplotlib','seaborn','plotly',
        'sklearn','tabulate','torch','xgboost','optuna','imblearn']
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f'{m}: {e}')
if missing:
    print('MISSING PYTHON DEPENDENCIES:', file=sys.stderr)
    for x in missing:
        print('  - ' + x, file=sys.stderr)
    sys.exit(1)
print('All Python dependencies import OK.')
PYEOF

# --- R rating packages ------------------------------------------------------
echo
echo "[4/4] Installing & verifying R packages (BradleyTerry2, PlackettLuce)"
if ! command -v Rscript >/dev/null 2>&1; then
    echo "Rscript not found on PATH. Install R (https://cran.r-project.org/) then re-run." >&2
    echo "ratings.* analyses require BradleyTerry2 and PlackettLuce." >&2
    exit 1
fi
Rscript - <<'REOF'
pkgs <- c("BradleyTerry2", "PlackettLuce")
missing <- pkgs[!(pkgs %in% rownames(installed.packages()))]
if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
still <- pkgs[!(pkgs %in% rownames(installed.packages()))]
if (length(still) > 0) {
  cat("MISSING R PACKAGES:", paste(still, collapse = ", "), "\n", file = stderr())
  quit(status = 1)
}
cat("R packages OK:", paste(pkgs, collapse = ", "), "\n")
REOF

echo
echo "All dependencies installed and verified."
