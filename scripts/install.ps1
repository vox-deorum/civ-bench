#requires -Version 5.1
<#
.SYNOPSIS
    Install every civ-bench dependency and verify it imports. Fails loudly if anything is missing.
.DESCRIPTION
    There is no graceful degradation in civ-bench: torch / xgboost / optuna / R are all required.
    Run this once before `civ-bench run`. Re-run after pulling new dependencies.
#>

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

# --- Python interpreter -----------------------------------------------------
$Python = if ($env:CIV_BENCH_PYTHON) { $env:CIV_BENCH_PYTHON } else { 'python' }
$pyVersion = & $Python --version
Write-Host "Using $Python ($pyVersion)" -ForegroundColor Cyan

# --- Python dependencies (all required, no optional extras) ------------------
$PyDeps = @(
    'pandas', 'numpy', 'scipy', 'statsmodels', 'matplotlib', 'seaborn', 'plotly',
    'scikit-learn', 'tabulate',
    'torch', 'xgboost', 'optuna', 'imbalanced-learn'
)

Write-Host "`n[1/4] Upgrading pip" -ForegroundColor Cyan
& $Python -m pip install --upgrade pip

Write-Host "`n[2/4] Installing Python dependencies" -ForegroundColor Cyan
& $Python -m pip install --upgrade @PyDeps

# Install the package itself in editable mode once pyproject.toml exists.
if (Test-Path (Join-Path $RepoRoot 'pyproject.toml')) {
    Write-Host "Installing civ-bench (editable)" -ForegroundColor Cyan
    & $Python -m pip install -e $RepoRoot
} else {
    Write-Host "No pyproject.toml yet; skipping editable install of civ-bench." -ForegroundColor Yellow
}

# --- Verify every Python import --------------------------------------------
Write-Host "`n[3/4] Verifying Python imports" -ForegroundColor Cyan
$ImportNames = @(
    'pandas', 'numpy', 'scipy', 'statsmodels', 'matplotlib', 'seaborn', 'plotly',
    'sklearn', 'tabulate', 'torch', 'xgboost', 'optuna', 'imblearn'
)
$ModList = ($ImportNames | ForEach-Object { "'$_'" }) -join ', '
$checkPy = @"
import importlib, sys
missing = []
for m in [$ModList]:
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
"@
& $Python -c $checkPy
if ($LASTEXITCODE -ne 0) { throw "Python dependency verification failed." }

# --- R rating packages ------------------------------------------------------
Write-Host "`n[4/4] Installing & verifying R packages (BradleyTerry2, PlackettLuce)" -ForegroundColor Cyan
$Rscript = Get-Command Rscript -ErrorAction SilentlyContinue
if (-not $Rscript) {
    throw "Rscript not found on PATH. Install R (https://cran.r-project.org/) then re-run. " +
          "ratings.* analyses require BradleyTerry2 and PlackettLuce."
}
$rCode = @'
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
'@
& $Rscript.Source -e $rCode
if ($LASTEXITCODE -ne 0) { throw "R dependency verification failed." }

Write-Host "`nAll dependencies installed and verified." -ForegroundColor Green
