"""Config-backed strategist + experiment catalog.

Ports ``shared/model_catalog.py`` + ``shared/experiments.py`` into a single
config-driven object. Unlike the old module-global helpers (which read a fixed
``shared/config`` dir), :class:`Catalog` is constructed from explicit file paths
(normally resolved from the run-spec's ``catalogs`` block), so nothing is
hardcoded (invariant 1: config over code).

The headline addition over the old code is the **orthodox ``player_type``
composition** (benchmark.md §3.3): :meth:`Catalog.compose_player_type` builds the
identity from the per-player game metadata (``model-{id}`` + ``strategist-{id}``)
via a ``player_type_template`` + alias maps, then applies the unified
``player_type_labels`` map. The legacy per-seat ``condition_player_mapping`` is
demoted to :meth:`Catalog.fallback_player_type`, used only for games that
predate the metadata.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Optional


class Catalog:
    def __init__(self, models: dict, experiments: dict):
        self._models = models
        self._experiments = experiments
        self._strategist_models = list(models.get("strategist_models", []))
        self._strategist_variants = dict(models.get("strategist_variants", {}))
        self._prediction_models = list(models.get("prediction_models", []))
        self._prompt_patterns = dict(models.get("prompt_patterns", {}))
        self._vanilla_aliases = {
            a.lower() for a in models.get("vanilla_model_aliases", [])
        }
        self._null_aliases = {
            a.lower() for a in models.get("null_strategist_aliases", [])
        }
        self._variant_map = dict(models.get("strategist_variant_map", {}))
        self._labels = dict(experiments.get("player_type_labels", {}))
        self._strategist_model_lookup = self._build_strategist_model_lookup()
        self._model_alias_candidates = self._build_model_alias_candidates()
        self._variant_suffix_order = sorted(
            self._strategist_variants.items(),
            key=lambda item: len(item[1].get("suffix", "")),
            reverse=True,
        )

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_paths(cls, models_path: str | Path, experiments_path: str | Path) -> "Catalog":
        with open(models_path, "r", encoding="utf-8") as h:
            models = json.load(h)
        with open(experiments_path, "r", encoding="utf-8") as h:
            experiments = json.load(h)
        return cls(models, experiments)

    @classmethod
    def from_run_config(cls, cfg) -> "Catalog":
        """Build from a :class:`bench.config.RunConfig` (lazy sibling paths)."""
        return cls.from_paths(cfg.catalog_path("models"), cfg.catalog_path("experiments"))

    # ── raw config accessors ────────────────────────────────────────────────
    @property
    def models_config(self) -> dict:
        return self._models

    @property
    def experiments_config(self) -> dict:
        return self._experiments

    def strategist_models(self) -> list[dict]:
        return list(self._strategist_models)

    def strategist_variants(self) -> dict:
        return dict(self._strategist_variants)

    def prediction_models(self) -> list[dict]:
        return list(self._prediction_models)

    def prompt_patterns(self) -> dict:
        return dict(self._prompt_patterns)

    # ── orthodox player_type composition (§3.3) ─────────────────────────────
    @property
    def player_type_template(self) -> str:
        return self._models.get("player_type_template", "{model}-{variant}{suffix}")

    @property
    def vanilla_label(self) -> str:
        return self._models.get("vanilla_label", "Vanilla")

    @property
    def null_label(self) -> str:
        return self._models.get("null_label", "Null")

    @property
    def _vanilla_model_aliases(self) -> set[str]:
        return set(self._vanilla_aliases)

    @property
    def _null_strategist_aliases(self) -> set[str]:
        return set(self._null_aliases)

    @property
    def _strategist_variant_map(self) -> dict:
        return dict(self._variant_map)

    @property
    def player_type_labels(self) -> dict:
        return dict(self._labels)

    def compose_player_type(
        self,
        model: Optional[str],
        strategist: Optional[str] = None,
        condition: Optional[str] = None,
        config_slot: Optional[int] = None,
    ) -> str:
        """Compose the orthodox ``player_type`` from per-player metadata.

        ``model`` is alias-normalized; ``strategist`` maps to a variant; a
        ``VPAI``/vanilla model resolves to the baseline label (``Null`` when the
        strategist is the null-strategist, else ``Vanilla``). The unified
        ``player_type_labels`` map is then applied: a value starting with ``-`` is
        a suffix (skipped for baselines so they pool across conditions); otherwise
        it is a full override (applies even to baselines). ``(condition, slot)``
        beats ``condition``.
        """
        if not model:
            return "N/A"

        base = self.canonicalize_model_name(model)
        is_baseline = self._is_baseline_model(model, base)

        if is_baseline:
            composed = self._baseline_label(base, strategist)
        else:
            variant = self._variant_map.get(strategist) if strategist else None
            if variant and variant in self._strategist_variants:
                composed = self.player_type_template.format(
                    model=base, variant=variant, suffix=""
                )
            else:
                composed = base

        label = self._lookup_label(condition, config_slot)
        if label:
            if label.startswith("-"):
                if not is_baseline:
                    composed = composed + label
            else:
                composed = label
        return composed

    def vanilla_model_aliases(self) -> set[str]:
        """Lower-cased raw-model aliases that denote the Vanilla VPAI engine.

        Used by the controlled-design baseline (``adjust/strength.py``) to keep
        "Vanilla VPAI baseline" narrow: a row counts as baseline evidence only
        when its raw ``model`` is one of these (e.g. ``VPAI``).
        """
        return set(self._vanilla_aliases)

    def is_vanilla_model(self, model: Optional[str]) -> bool:
        """True when a raw ``model`` value is the Vanilla/VPAI engine."""
        return bool(model) and str(model).lower() in self._vanilla_aliases

    def _is_baseline_model(self, raw_model: str, base: str) -> bool:
        if base in (self.vanilla_label, self.null_label):
            return True
        return bool(raw_model) and raw_model.lower() in self._vanilla_model_aliases

    def _is_null_strategist(self, strategist: Optional[str]) -> bool:
        if not strategist:
            return False
        if strategist.lower() in self._null_aliases:
            return True
        return self._variant_map.get(strategist) == self.null_label

    def _baseline_label(self, base: str, strategist: Optional[str]) -> str:
        if base == self.null_label or self._is_null_strategist(strategist):
            return self.null_label
        return self.vanilla_label

    def _lookup_label(self, condition: Optional[str], slot: Optional[int]) -> Optional[str]:
        if condition is None:
            return None
        labels = self.player_type_labels
        entry = labels.get(condition)
        if entry is None:
            entry = self._match_wildcard_label(condition, labels)
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if slot is not None and str(slot) in entry:
                return entry[str(slot)]
            return entry.get("_default")
        return None

    @staticmethod
    def _match_wildcard_label(condition: str, labels: dict):
        """Resolve a glob-pattern condition key (e.g. ``*-per-5``) for a condition.

        Consulted only after an exact key miss, so a literal condition key always
        wins. Among the wildcard keys (those containing ``*``) that match, the
        **most specific** wins (the one with the most non-wildcard characters,
        ties broken lexicographically), so a broad ``*`` never shadows a narrower
        ``oss-*-per-5``. Returns the matched entry (a string or ``(slot)`` dict,
        interpreted exactly like an exact-key entry) or ``None``.
        """
        matches = [
            (key, value)
            for key, value in labels.items()
            if "*" in key and fnmatch.fnmatchcase(condition, key)
        ]
        if not matches:
            return None
        _, value = max(matches, key=lambda kv: (len(kv[0].replace("*", "")), kv[0]))
        return value

    # ── alias normalization (ported) ────────────────────────────────────────
    @staticmethod
    def normalize_model_base(model_raw: Optional[str]) -> Optional[str]:
        if not model_raw:
            return None
        return model_raw.split("@", 1)[0].strip()

    def canonicalize_model_name(self, model_base: Optional[str]) -> str:
        normalized = self.normalize_model_base(model_base)
        if not normalized:
            return "N/A"
        lowered = normalized.lower()
        # Exact alias match wins outright, so a short alias (e.g. "minimax")
        # can never shadow a longer model spelling that contains it
        # ("minimax-m2.7") when both are registered.
        for candidate, model_id in self._model_alias_candidates:
            if candidate == lowered:
                return model_id
        # Otherwise the most specific (longest) substring alias wins;
        # _model_alias_candidates is sorted longest-first at build time.
        for candidate, model_id in self._model_alias_candidates:
            if candidate and candidate in lowered:
                return model_id
        return normalized

    def get_strategist_model(self, name: Optional[str]) -> Optional[dict]:
        if not name:
            return None
        return self._strategist_model_lookup.get(name.lower())

    def _build_strategist_model_lookup(self) -> dict:
        out: dict = {}
        for model in self._strategist_models:
            out[model["id"].lower()] = model
            for alias in model.get("aliases", []):
                out[alias.lower()] = model
        return out

    def _build_model_alias_candidates(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for model in self._strategist_models:
            for candidate in [model["id"], *model.get("aliases", [])]:
                if candidate:
                    out.append((candidate.lower(), model["id"]))
        # Sort longest-first so substring matching in canonicalize_model_name
        # prefers the most specific alias (a short alias never shadows a longer
        # spelling that contains it). Ties keep config order for determinism.
        out.sort(key=lambda pair: len(pair[0]), reverse=True)
        return out

    def get_variant_config(self, name: Optional[str]) -> Optional[dict]:
        if not name:
            return None
        return self._strategist_variants.get(name)

    def get_variant_style_key(self, name: Optional[str]) -> Optional[str]:
        config = self.get_variant_config(name)
        if not config:
            return None
        return config.get("style_key", name)

    def split_player_type(self, player_type: Optional[str]) -> dict:
        if not player_type:
            return {"model_id": None, "variant": None}
        model = self.get_strategist_model(player_type)
        if model is not None:
            return {"model_id": model["id"], "variant": None}
        for variant_name, config in self._variant_suffix_order:
            suffix = config.get("suffix", "")
            if suffix and player_type.endswith(suffix):
                model_id = player_type[: -len(suffix)]
                base_model = self.get_strategist_model(model_id)
                resolved_id = base_model["id"] if base_model is not None else model_id
                return {"model_id": resolved_id, "variant": variant_name}
        return {"model_id": player_type, "variant": None}

    def condition_suffixes(self) -> list[str]:
        """Return configured suffix-style condition labels, longest first.

        Both direct ``player_type_labels`` values and per-slot dictionary values
        participate. Full-identity overrides are deliberately excluded.
        """
        suffixes: set[str] = set()
        for entry in self._labels.values():
            values = entry.values() if isinstance(entry, dict) else (entry,)
            for value in values:
                if isinstance(value, str) and value.startswith("-"):
                    suffixes.add(value)
        return sorted(suffixes, key=lambda value: (-len(value), value))

    def split_condition_suffix(
        self,
        player_type: Optional[str],
        suffixes: Optional[list[str]] = None,
    ) -> tuple[Optional[str], str]:
        """Split a display-condition suffix from an orthodox player identity.

        Vanilla and Null are pooled baselines and therefore never split. An
        explicit ``suffixes`` list restricts recognition; otherwise suffixes are
        derived from the experiment catalog. Longest suffix wins.
        """
        if not player_type or player_type in (self.vanilla_label, self.null_label):
            return player_type, ""
        effective = self.condition_suffixes() if suffixes is None else list(suffixes)
        for suffix in sorted(set(effective), key=lambda value: (-len(value), value)):
            if suffix and player_type.endswith(suffix) and len(player_type) > len(suffix):
                return player_type[: -len(suffix)], suffix
        return player_type, ""

    # ── pricing + colors (ported) ───────────────────────────────────────────
    def pricing_per_million(self) -> dict:
        pricing: dict = {}
        for model in self.strategist_models():
            p = model.get("pricing")
            if p:
                pricing[model["id"]] = p
        return pricing

    def strategist_model_colors(self) -> dict:
        return {m["id"]: m["color"] for m in self.strategist_models()}

    def prediction_model_order(self) -> list[str]:
        models = sorted(self.prediction_models(), key=lambda m: m.get("predict_order", 0))
        return [m["id"] for m in models]

    def prediction_model_colors(self) -> dict:
        return {m["id"]: m["color"] for m in self.prediction_models()}

    # ── experiment groups + legacy fallback (ported) ────────────────────────
    def vanilla_experiments(self) -> list[str]:
        return list(self._experiments.get("vanilla_experiments", []))

    def null_ai_experiments(self) -> list[str]:
        return list(self._experiments.get("null_ai_experiments", []))

    def non_llm_experiments(self) -> list[str]:
        return self.vanilla_experiments() + self.null_ai_experiments()

    def default_excluded_experiments(self) -> list[str]:
        return list(self._experiments.get("default_excluded_experiments", []))

    def _raw_condition_player_mapping(self) -> dict:
        return dict(self._experiments.get("condition_player_mapping", {}))

    def _expand_seat(self, seat) -> str:
        if isinstance(seat, str):
            return seat
        if isinstance(seat, dict):
            model_id = self.canonicalize_model_name(seat.get("model"))
            variant = seat.get("variant")
            if not variant or model_id in (self.vanilla_label, self.null_label):
                return model_id
            cfg = self.get_variant_config(variant)
            return f"{model_id}{cfg['suffix']}" if cfg else model_id
        return str(seat)

    def condition_player_mapping(self) -> dict:
        return {
            condition: [self._expand_seat(seat) for seat in seats]
            for condition, seats in self._raw_condition_player_mapping().items()
        }

    def llm_experiments(self) -> list[str]:
        mapping = self.condition_player_mapping()
        return [
            exp for exp, players in mapping.items()
            if any(p not in (self.vanilla_label, self.null_label) for p in players)
        ]

    def fallback_player_type(
        self, condition: Optional[str], player_id: int
    ) -> str:
        """Legacy static ``(condition, slot)`` → player_type fallback (§3.3).

        Used only for games with no per-player metadata. Returns ``Player {id}``
        when the condition/slot is unknown.
        """
        mapping = self.condition_player_mapping()
        seats = mapping.get(condition)
        if seats is not None and 0 <= player_id < len(seats):
            return seats[player_id]
        return f"Player {player_id}"
