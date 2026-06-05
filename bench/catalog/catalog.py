"""Config-backed strategist + experiment catalog.

Ports ``shared/model_catalog.py`` + ``shared/experiments.py`` into a single
config-driven object. Unlike the old module-global helpers (which read a fixed
``shared/config`` dir), :class:`Catalog` is constructed from explicit file paths
— normally resolved from the run-spec's ``catalogs`` block — so nothing is
hardcoded (invariant 1: config over code).

The headline addition over the old code is the **orthodox ``player_type``
composition** (benchmark.md §3.3): :meth:`Catalog.compose_player_type` builds the
identity from the per-player game metadata (``model-{id}`` + ``strategist-{id}``)
via a ``player_type_template`` + alias maps, then applies the unified
``player_type_labels`` map. The legacy per-seat ``condition_player_mapping`` is
demoted to :meth:`Catalog.fallback_player_type` — used only for games that
predate the metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class Catalog:
    def __init__(self, models: dict, experiments: dict):
        self._models = models
        self._experiments = experiments

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
        return list(self._models.get("strategist_models", []))

    def strategist_variants(self) -> dict:
        return dict(self._models.get("strategist_variants", {}))

    def prediction_models(self) -> list[dict]:
        return list(self._models.get("prediction_models", []))

    def prompt_patterns(self) -> dict:
        return dict(self._models.get("prompt_patterns", {}))

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
        return {a.lower() for a in self._models.get("vanilla_model_aliases", [])}

    @property
    def _null_strategist_aliases(self) -> set[str]:
        return {a.lower() for a in self._models.get("null_strategist_aliases", [])}

    @property
    def _strategist_variant_map(self) -> dict:
        return dict(self._models.get("strategist_variant_map", {}))

    @property
    def player_type_labels(self) -> dict:
        return dict(self._experiments.get("player_type_labels", {}))

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
            variant = self._strategist_variant_map.get(strategist) if strategist else None
            if variant and variant in self.strategist_variants():
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

    def _is_baseline_model(self, raw_model: str, base: str) -> bool:
        if base in (self.vanilla_label, self.null_label):
            return True
        return bool(raw_model) and raw_model.lower() in self._vanilla_model_aliases

    def _is_null_strategist(self, strategist: Optional[str]) -> bool:
        if not strategist:
            return False
        if strategist.lower() in self._null_strategist_aliases:
            return True
        return self._strategist_variant_map.get(strategist) == self.null_label

    def _baseline_label(self, base: str, strategist: Optional[str]) -> str:
        if base == self.null_label or self._is_null_strategist(strategist):
            return self.null_label
        return self.vanilla_label

    def _lookup_label(self, condition: Optional[str], slot: Optional[int]) -> Optional[str]:
        labels = self.player_type_labels
        entry = labels.get(condition)
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if slot is not None and str(slot) in entry:
                return entry[str(slot)]
            return entry.get("_default")
        return None

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
        for model in self.strategist_models():
            candidates = [model["id"], *model.get("aliases", [])]
            if any(c and c.lower() in lowered for c in candidates):
                return model["id"]
        return normalized

    def get_strategist_model(self, name: Optional[str]) -> Optional[dict]:
        if not name:
            return None
        lookup = self._strategist_models_by_name()
        return lookup.get(name.lower())

    def _strategist_models_by_name(self) -> dict:
        out: dict = {}
        for model in self.strategist_models():
            out[model["id"].lower()] = model
            for alias in model.get("aliases", []):
                out[alias.lower()] = model
        return out

    def get_variant_config(self, name: Optional[str]) -> Optional[dict]:
        if not name:
            return None
        return self.strategist_variants().get(name)

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
        for variant_name, config in sorted(
            self.strategist_variants().items(),
            key=lambda item: len(item[1].get("suffix", "")),
            reverse=True,
        ):
            suffix = config.get("suffix", "")
            if suffix and player_type.endswith(suffix):
                model_id = player_type[: -len(suffix)]
                base_model = self.get_strategist_model(model_id)
                resolved_id = base_model["id"] if base_model is not None else model_id
                return {"model_id": resolved_id, "variant": variant_name}
        return {"model_id": player_type, "variant": None}

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
