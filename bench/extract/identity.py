"""Orthodox ``player_type`` composition at extract time (benchmark.md §3.3).

The game runner records each seat's identity in ``GameMetadata`` as
``model-{player_id}`` (e.g. ``Sonnet-4.5``, or ``VPAI`` for vanilla) and
``strategist-{player_id}`` (e.g. ``simple-strategist-briefed``). Because the
identity travels with the player, the composed ``player_type`` stays correct even
when controlled seating rotates a model through different seats.

This is the single source of truth for ``player_type``: the same map feeds
``panel_data``, ``turn_data`` (broadcast per (game, player)), and
``model_token_usage`` — replacing the old load-time ``(condition, player_id)``
seat merge, which survives only as a fallback for legacy games with no metadata.
"""

from __future__ import annotations

from typing import Optional

from ..catalog import Catalog
from .utilities import SeedingInfo


def compose_identities(
    metadata: dict,
    player_ids: list[int],
    experiment: str,
    catalog: Optional[Catalog],
    seeding: SeedingInfo,
) -> dict[int, dict]:
    """Compose ``{player_id: {player_type, model, strategist, config_slot}}``.

    For each player, ``model-{id}`` / ``strategist-{id}`` are read from
    ``metadata`` and composed via :meth:`Catalog.compose_player_type` (with the
    player's ``config_slot`` so ``(condition, slot)`` label overrides apply). A
    player with no ``model-{id}`` metadata is a **legacy seat**: its
    ``player_type`` falls back to the catalog's static ``(condition, slot)`` map.
    """
    identities: dict[int, dict] = {}
    for pid in player_ids:
        model = metadata.get(f"model-{pid}")
        strategist = metadata.get(f"strategist-{pid}")
        config_slot = seeding.config_slot(pid)

        if catalog is None:
            player_type = None
        elif model:
            player_type = catalog.compose_player_type(
                model, strategist, condition=experiment, config_slot=config_slot
            )
        else:
            # Legacy game (no per-player metadata) → static seat-map fallback.
            player_type = catalog.fallback_player_type(experiment, pid)

        identities[pid] = {
            "player_type": player_type,
            "model": model if model is not None else "N/A",
            "strategist": strategist if strategist is not None else "N/A",
            "config_slot": config_slot,
        }
    return identities
