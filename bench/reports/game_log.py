"""Render individual games and shared replay links from saved analysis tables."""

from __future__ import annotations

import html
import json
from urllib.parse import quote, urlencode, urljoin, urlsplit, urlunsplit

import pandas as pd

from bench.config.schema import REPORT_DEFAULT_VIEWER_URL
from bench.reports.content import render_footer_html, resolve_footer
from bench.reports.context import ReportBuildContext
from bench.reports.errors import ReportError
from bench.reports.model import Download, GameLogDocument, ReplayOptions


def game_log_document(ctx: ReportBuildContext) -> GameLogDocument | None:
    sections = [s for s in ctx.sections if s.module == "performance.game_log"]
    if not sections:
        return None
    if len(sections) > 1:
        raise ReportError("The report requires a single performance.game_log section.")
    section = sections[0]
    options = ctx.meta.get("replay") or {}
    replay = None
    if options.get("enabled", False):
        replay = ReplayOptions(
            viewer_url=options.get("viewer_url", REPORT_DEFAULT_VIEWER_URL),
            base_url=options.get("base_url"),
            latest_game=options.get("latest_game", True),
        )
    downloads = list(section.downloads)
    downloads.extend(Download(f"Table: {t.name} (CSV)", t.rel_csv)
                     for t in section.tables if t.rel_csv)
    return GameLogDocument(
        title=ctx.meta["title"], section_id=section.id,
        games=ctx.load_table(section.id, "games") if not section.empty else pd.DataFrame(),
        game_players=ctx.load_table(section.id, "game_players") if not section.empty else pd.DataFrame(),
        metadata=dict(section.metadata), downloads=downloads, replay=replay,
        footer=resolve_footer(ctx.meta.get("footer")),
    )


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _number(value) -> str:
    return "-" if not _text(value) else str(int(value))


def _flag(value) -> bool:
    return _text(value).lower() in {"true", "1", "1.0"}


def _esc(value) -> str:
    return html.escape(_text(value), quote=True)


def game_players(doc: GameLogDocument, game_id: str) -> list[dict]:
    if doc.game_players.empty:
        return []
    return doc.game_players.loc[
        doc.game_players["game_id"].astype(str) == str(game_id)
    ].sort_values("player_id", kind="mergesort").to_dict("records")


def player_label(player: dict) -> str:
    label = _text(player.get("strategist")) or _text(player.get("player_type"))
    condition = _text(player.get("condition"))
    return f"{label} | {condition}" if condition else label


def _viewer_url(viewer: str, params: list[tuple[str, str]]) -> str:
    parts = urlsplit(viewer)
    query = urlencode(params, quote_via=quote)
    return urlunsplit(parts._replace(query="&".join(filter(None, (parts.query, query)))))


def replay_target(doc: GameLogDocument, game: dict, players: list[dict], prefix: str = "") -> tuple[str, str]:
    replay = doc.replay
    path = replay.save_paths.get(str(game["game_id"])) if replay else None
    if path is None:
        return "", ""
    params = [(f"player{_number(p['player_id'])}", player_label(p))
              for p in players if not _flag(p.get("is_vanilla"))]
    if _text(game.get("winner_player_id")):
        params.append(("winner", _number(game["winner_player_id"])))
    query = urlencode(params, quote_via=quote)
    encoded_path = quote(path, safe="/")
    if replay.base_url:
        save_url = urljoin(replay.base_url.rstrip("/") + "/", encoded_path)
        return _viewer_url(replay.viewer_url, [("file", save_url), *params]), query
    return prefix + encoded_path, query


def replay_link_html(doc: GameLogDocument, game: dict, players: list[dict],
                     prefix: str = "", label: str = "▶ Watch") -> str:
    href, query = replay_target(doc, game, players, prefix)
    if not href:
        return '<span class="muted">no replay</span>'
    direct = ' data-direct="true" target="_blank" rel="noopener"' if doc.replay.base_url else ""
    return (f'<a class="replay-link" href="{_esc(href)}" '
            f'data-viewer="{_esc(doc.replay.viewer_url)}" data-query="{_esc(query)}"'
            f'{direct}>{_esc(label)}</a>')


def seats_text(game: dict, players: list[dict], *, include_winner: bool = True) -> str:
    seats = []
    listed_winner = False
    for player in players:
        if _flag(player.get("is_vanilla")):
            continue
        won = _flag(player.get("is_winner"))
        listed_winner |= won
        detail = _text(player.get("civilization")) or "Unknown"
        if won:
            detail += ", Won"
        seats.append(f"Player {_number(player['player_id'])} ({detail})")
    result = " | ".join(seats)
    if include_winner and not listed_winner and _text(game.get("winner_player_id")):
        winner = winner_text(game)
        result += (" · " if result else "") + winner
    return result or "-"


def winner_text(game: dict) -> str:
    if not _text(game.get("winner_player_id")):
        return "-"
    detail = _text(game.get("winner_civilization")) or "Unknown"
    if _flag(game.get("winner_is_vanilla")):
        detail += ", VPAI"
    return f"Winner: Player {_number(game['winner_player_id'])} ({detail})"


def games_url(prefix: str = "", **filters) -> str:
    query = urlencode({k: v for k, v in filters.items() if v is not None}, quote_via=quote)
    return prefix + "games.html" + ("?" + query if query else "")


def latest_game_card_html(doc: GameLogDocument) -> str:
    if not doc.replay or not doc.replay.latest_game or doc.games.empty:
        return ""
    game = doc.games.iloc[0].to_dict()
    players = game_players(doc, game["game_id"])
    return ('<aside class="announcement latest-game">'
            f'<p class="eyebrow">Latest game · {_esc(game.get("date_utc"))} '
            '(<a href="games.html">All recent games</a>)</p>'
            f'<p>{_esc(game["label"])} · '
            f'{replay_link_html(doc, game, players, label="▶ Watch replay")}</p>'
            f'<p>{_esc(seats_text(game, players))}</p></aside>')


def game_log_markdown(doc: GameLogDocument) -> list[str]:
    lines = ["## Game Log", ""]
    if any(d.rel_path == "games.html" for d in doc.downloads):
        lines.extend(["[Browse recent games](games.html)", ""])
    if not doc.games.empty:
        game = doc.games.iloc[0].to_dict()
        players = game_players(doc, game["game_id"])
        latest = f"Latest game: {game['date_utc']} · {game['label']}. {seats_text(game, players)}"
        href, _ = replay_target(doc, game, players)
        if href and doc.replay.base_url:
            latest += f" [Watch replay]({href})"
        lines.extend([latest, ""])
    lines.extend(f"- [{d.label}]({d.rel_path})" for d in doc.downloads if d.rel_path != "games.html")
    return lines + [""]


def _identities(players: list[dict], vanilla: str) -> list[dict]:
    return [{"strategist": _text(p.get("strategist")),
             "condition": _text(p.get("condition")),
             "player": _number(p["player_id"]),
             "vanilla": _flag(p.get("is_vanilla"))} for p in players]


def render_game_log_page(doc: GameLogDocument, navigation: list[str]) -> str:
    records = doc.games.to_dict("records")
    vanilla = str(doc.metadata.get("vanilla_label", "Vanilla"))
    rows = [(game, game_players(doc, game["game_id"])) for game in records]
    options = {key: set() for key in ("strategist", "condition", "seed", "player", "victory")}
    for game, players in rows:
        if _flag(game.get("controlled")):
            options["seed"].add(_number(game.get("seed")))
        if _text(game.get("victory_type")):
            options["victory"].add(_text(game["victory_type"]))
        for identity in _identities(players, vanilla):
            for key in ("strategist", "condition", "player"):
                if identity[key]:
                    options[key].add(identity[key])
    parts = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             f'<title>Game Log | {_esc(doc.title)}</title>',
             '<link rel="stylesheet" href="assets/report.css">',
             '<script src="assets/report-common.js" defer></script>',
             '<script src="assets/game-log.js" defer></script></head><body>',
             '<a class="skip-link" href="#main-content">Skip to content</a>', *navigation,
             '<main class="content game-log" id="main-content">',
             f'<p class="eyebrow">{_esc(doc.title)}</p><h1>Game Log</h1>']
    available = len(doc.replay.save_paths) if doc.replay else 0
    parts.append(f'<p>{len(records)} games · {available} with a replay</p>')
    parts.append('<form class="game-log-filters" id="game-log-filters">')
    for key, label in (("strategist", "Strategist"), ("condition", "Condition"),
                       ("seed", "Seed"), ("player", "Seat"), ("victory", "Victory")):
        parts.append(f'<label>{label} <select name="{key}"><option value="">All</option>')
        ordered = sorted(options[key], key=int if key in {"seed", "player"} else str)
        preferred = doc.metadata.get(key + "_order", [])
        ordered = [v for v in preferred if v in options[key]] + [v for v in ordered if v not in preferred]
        for value in ordered:
            label_value = "VPAI" if value == vanilla else ("Player " + value if key == "player" else value)
            parts.append(f'<option value="{_esc(value)}">{_esc(label_value)}</option>')
        parts.append('</select></label>')
    parts.append('<label><input type="checkbox" name="controlled" checked> Controlled only</label>'
                 f'<span id="game-log-count" aria-live="polite">Showing {len(records)} of {len(records)}</span>'
                 '<button type="reset">Reset</button></form>')
    parts.append('<div class="table-scroll"><table id="game-log-table" class="no-auto-sort"><thead><tr>')
    for key, label in (("timestamp", "Date"), ("label", "Strategist | Condition"), ("seed", "Seed"),
                       ("rotation", "Rot"), ("turns", "Turns"), (None, "Seats"),
                       ("victory", "Victory"), (None, "Replay")):
        content = f'<button type="button" data-sort="{key}">{label}</button>' if key else label
        parts.append(f'<th scope="col">{content}</th>')
    parts.append('</tr></thead><tbody>')
    for game, players in rows:
        controlled = _flag(game.get("controlled"))
        identities = _identities(players, vanilla)
        attrs = {"identities": json.dumps(identities, ensure_ascii=False, separators=(",", ":")),
                 "strategist": json.dumps(sorted({p["strategist"] for p in identities})),
                 "condition": json.dumps(sorted({p["condition"] for p in identities})),
                 "player": json.dumps([p["player"] for p in identities]),
                 "seed": _number(game.get("seed")) if controlled else "-",
                 "rotation": _number(game.get("seating_rotation")) if controlled else "-",
                 "controlled": "true" if controlled else "false",
                 "timestamp": _number(game.get("timestamp")), "turns": _number(game.get("turns")),
                 "label": game["label"], "victory": game.get("victory_type", "")}
        parts.append('<tr ' + ' '.join(f'data-{k}="{_esc(v)}"' for k, v in attrs.items()) + '>')
        cells = [game.get("date_utc"), game["label"], attrs["seed"], attrs["rotation"],
                 attrs["turns"], seats_text(game, players, include_winner=False)]
        parts.extend(f'<td>{_esc(cell) or "-"}</td>' for cell in cells)
        victory = _esc(game.get("victory_type")) or "-"
        if _text(game.get("winner_player_id")) and not any(
            _flag(player.get("is_winner")) and not _flag(player.get("is_vanilla"))
            for player in players
        ):
            victory += f'<span class="game-winner">{_esc(winner_text(game))}</span>'
        parts.append(f'<td>{victory}</td>')
        parts.append(f'<td>{replay_link_html(doc, game, players)}</td></tr>')
    parts.append('</tbody></table></div>')
    links = [f'<a href="{_esc(d.rel_path)}">{_esc(d.label)}</a>' for d in doc.downloads if d.rel_path != "games.html"]
    if links:
        parts.append('<p class="caption">Source tables: ' + ' · '.join(links) + '</p>')
    parts.extend([render_footer_html(doc.footer), '</main></body></html>'])
    return "\n".join(parts) + "\n"


def render_seat_games(doc: GameLogDocument, seed: int, player_id: int, prefix: str = "../") -> str:
    if doc.games.empty:
        return ""
    rows = []
    for game in doc.games.loc[doc.games["seed"] == seed].to_dict("records"):
        players = game_players(doc, game["game_id"])
        seat = next((p for p in players if int(p["player_id"]) == player_id), None)
        if seat is not None:
            rows.append((game, players, seat))
    parts = ['<section aria-labelledby="games-heading"><h2 id="games-heading">Games</h2>',
             '<div class="table-scroll"><table class="seat-games"><thead><tr>',
             '<th>Date</th><th>Strategist | Condition</th><th>Rot</th><th>This seat</th><th>Victory</th><th>Replay</th>',
             '</tr></thead><tbody>']
    for game, players, seat in rows:
        parts.append(f'<tr class="game-row" data-strategist="{_esc(seat["strategist"])}" '
                     f'data-vanilla="{str(_flag(seat.get("is_vanilla"))).lower()}">')
        outcome = "Won" if _flag(seat.get("is_winner")) else winner_text(game)
        label = "VPAI" if _flag(seat.get("is_vanilla")) else player_label(seat)
        parts.extend(f'<td>{_esc(value) or "-"}</td>' for value in
                     (game.get("date_utc"), label, _number(game.get("seating_rotation")), outcome, game.get("victory_type")))
        parts.append(f'<td>{replay_link_html(doc, game, players, prefix)}</td></tr>')
    parts.append('</tbody></table></div>')
    parts.append(f'<p><a href="{_esc(games_url(prefix, seed=seed, player=player_id))}">Show all {len(rows)} →</a></p></section>')
    return "\n".join(parts)


GAME_LOG_JS = """/* Game Log filters and stable sorting. */
(function () {
  "use strict";
  var form = document.getElementById("game-log-filters");
  var table = document.getElementById("game-log-table");
  if (!form || !table) { return; }
  var rows = Array.from(table.tBodies[0].rows);
  var keys = ["strategist", "condition", "seed", "player", "victory"];
  var query = new URLSearchParams(window.location.search);
  var sort = query.get("sort") || "timestamp";
  var descending = query.get("direction") !== "asc";
  keys.forEach(function (key) {
    var value = query.get(key) || "";
    var select = form.elements[key];
    if (value && !Array.from(select.options).some(function (option) { return option.value === value; })) {
      select.add(new Option(value, value));
    }
    select.value = value;
  });
  form.elements.controlled.checked = query.get("controlled") !== "false";
  function apply() {
    var values = {};
    keys.forEach(function (key) { values[key] = form.elements[key].value; });
    var shown = 0;
    rows.forEach(function (row) {
      var data = row.dataset;
      var identities = JSON.parse(data.identities);
      var match = (!form.elements.controlled.checked || data.controlled === "true") &&
        (!values.seed || values.seed === data.seed) && (!values.victory || values.victory === data.victory) &&
        identities.some(function (identity) {
          return (!values.strategist || values.strategist === identity.strategist) &&
            (!values.condition || values.condition === identity.condition) &&
            (!values.player || values.player === identity.player);
        });
      row.hidden = !match;
      if (match) { shown += 1; }
    });
    document.getElementById("game-log-count").textContent = "Showing " + shown + " of " + rows.length;
    var numeric = ["timestamp", "seed", "rotation", "turns"].indexOf(sort) >= 0;
    rows.map(function (row, index) { return {row: row, index: index}; }).sort(function (a, b) {
      var x = a.row.dataset[sort] || "", y = b.row.dataset[sort] || "";
      var diff = numeric ? (Number(x === "-" ? -1 : x) - Number(y === "-" ? -1 : y)) : x.localeCompare(y);
      return (descending ? -diff : diff) || a.index - b.index;
    }).forEach(function (entry) { table.tBodies[0].appendChild(entry.row); });
    table.querySelectorAll("[data-sort]").forEach(function (button) {
      button.parentElement.setAttribute("aria-sort", button.dataset.sort === sort ? (descending ? "descending" : "ascending") : "none");
    });
    var url = new URL(window.location.href);
    keys.forEach(function (key) {
      if (values[key]) { url.searchParams.set(key, values[key]); } else { url.searchParams.delete(key); }
    });
    if (form.elements.controlled.checked) { url.searchParams.delete("controlled"); }
    else { url.searchParams.set("controlled", "false"); }
    if (sort === "timestamp" && descending) {
      url.searchParams.delete("sort"); url.searchParams.delete("direction");
    } else { url.searchParams.set("sort", sort); url.searchParams.set("direction", descending ? "desc" : "asc"); }
    try { window.history.replaceState(null, "", url); } catch (error) { /* Local file pages may disallow history updates. */ }
  }
  form.addEventListener("submit", function (event) { event.preventDefault(); });
  form.addEventListener("change", apply);
  form.addEventListener("reset", function (event) {
    event.preventDefault();
    keys.forEach(function (key) { form.elements[key].value = ""; });
    form.elements.controlled.checked = true;
    sort = "timestamp"; descending = true; apply();
  });
  table.querySelectorAll("[data-sort]").forEach(function (button) {
    button.addEventListener("click", function () {
      descending = sort === button.dataset.sort ? !descending : false;
      sort = button.dataset.sort; apply();
    });
  });
  apply();
}());
"""
