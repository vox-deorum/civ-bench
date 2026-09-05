"""civ-bench: a modular, JSON-configurable benchmark harness for LLM strategists
in Civilization V: Vox Populi (via the Vox Deorum platform).

Keep this module import light: it must not pull in heavy analysis dependencies
(torch / statsmodels / matplotlib) so that `civ-bench run --dry-run` can load and
validate a config + print its DAG without importing any stage implementation.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
