"""Shared static report assets (stage 5).

The report site is rendered server-side; a page that scripts its own content
loads :data:`REPORT_COMMON_JS` (written to ``assets/report-common.js``) before
its page script. The util is deterministic vanilla JavaScript: no packages, no
network. It owns the ``window.civBench`` namespace so forthcoming report pages
reuse the same helpers instead of growing per-report copies.

Currently the namespace carries one util: ``civBench.distinguishColors``.
Catalog colors are per model family, so several strategists on one page can
share a color; the util keeps the first member's color and steps every extra
member around the hue wheel (grays step lightness instead), so same-family
series stay distinguishable.
"""

from __future__ import annotations

REPORT_HELP_JS = """/* Shared report help: hover, keyboard focus, click, and Escape.
   Also switches section views: the first view of each group is shown, and a
   choice applies to every group on the page that offers the same view. And it
   shows the text of any [data-tip] element (heatmap cells and labels) in one
   floating tooltip on hover or keyboard focus. A tip's first line is its bold
   title when there are more; "label<TAB>value<TAB>note" lines form a grid with
   the value and the note's numbers in an accent color, and other lines are
   muted subtitles. */
(function () {
  "use strict";
  var tooltip = null;
  function span(cls, text) {
    var node = document.createElement("span");
    node.className = cls;
    node.textContent = text;
    return node;
  }
  /* A note's numbers (a CI range, a game count) share the value color. */
  function noteSpan(text) {
    var node = span("tip-note", "");
    text.split(/([+\\-\\u2212]?\\d[\\d,.]*%?)/).forEach(function (part, index) {
      if (!part) { return; }
      node.appendChild(index % 2 ? span("tip-number", part) : document.createTextNode(part));
    });
    return node;
  }
  function renderTip(text) {
    tooltip.textContent = "";
    var lines = text.split("\\n");
    var grid = null;
    lines.forEach(function (line, index) {
      var cells = line.split("\\t");
      if (cells.length > 1) {
        if (!grid) {
          grid = document.createElement("div");
          grid.className = "tip-grid";
          tooltip.appendChild(grid);
        }
        grid.appendChild(span("tip-label", cells[0]));
        var value = document.createElement("span");
        value.className = "tip-value";
        if (cells[1]) { value.appendChild(span("tip-number", cells[1])); }
        if (cells[2]) { value.appendChild(noteSpan(cells[2])); }
        grid.appendChild(value);
        return;
      }
      grid = null;
      var cls = index === 0 && lines.length > 1 ? "tip-title" : (index === 0 ? "tip-text" : "tip-sub");
      tooltip.appendChild(span(cls, line));
    });
  }
  function showTip(target) {
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.id = "heat-tooltip";
      tooltip.className = "heat-tooltip";
      tooltip.setAttribute("role", "tooltip");
      document.body.appendChild(tooltip);
    }
    renderTip(target.getAttribute("data-tip") || "");
    tooltip.style.display = "block";
    var rect = target.getBoundingClientRect();
    var top = rect.top - tooltip.offsetHeight - 6;
    if (top < 4) { top = rect.bottom + 6; }
    var left = Math.min(rect.left, document.documentElement.clientWidth - tooltip.offsetWidth - 8);
    tooltip.style.top = (window.scrollY + top) + "px";
    tooltip.style.left = (window.scrollX + Math.max(4, left)) + "px";
  }
  function hideTip() {
    if (tooltip) { tooltip.style.display = "none"; }
  }
  function tipTarget(event) {
    return event.target.closest ? event.target.closest("[data-tip]") : null;
  }
  document.addEventListener("mouseover", function (event) {
    var target = tipTarget(event);
    if (target) { showTip(target); }
  });
  document.addEventListener("mouseout", function (event) {
    if (tipTarget(event)) { hideTip(); }
  });
  document.addEventListener("focusin", function (event) {
    var target = tipTarget(event);
    if (target) { showTip(target); }
  });
  document.addEventListener("focusout", hideTip);
  var groups = document.querySelectorAll(".view-group");
  function showView(group, name) {
    var panels = group.querySelectorAll(".view-panel");
    var known = Array.prototype.some.call(panels, function (panel) {
      return panel.getAttribute("data-view") === name;
    });
    if (!known) { return; }
    panels.forEach(function (panel) {
      panel.hidden = panel.getAttribute("data-view") !== name;
    });
    group.querySelectorAll(".view-button").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.getAttribute("data-view") === name));
    });
  }
  if (groups.length) {
    document.documentElement.classList.add("views-ready");
    groups.forEach(function (group) {
      var first = group.querySelector(".view-panel");
      if (first) { showView(group, first.getAttribute("data-view")); }
      group.querySelectorAll(".view-button").forEach(function (button) {
        button.addEventListener("click", function () {
          var name = button.getAttribute("data-view");
          groups.forEach(function (other) { showView(other, name); });
        });
      });
    });
  }
  document.querySelectorAll(".report-help").forEach(function (help) {
    var button = help.querySelector(".help-toggle");
    function dismiss() {
      help.classList.remove("help-open");
      help.classList.add("help-dismissed");
      button.setAttribute("aria-expanded", "false");
    }
    function reveal() {
      help.classList.remove("help-dismissed");
      button.setAttribute("aria-expanded", "true");
      var tip = help.querySelector(".help-text");
      tip.style.left = "0px";
      var rect = tip.getBoundingClientRect();
      tip.style.left = Math.min(0, window.innerWidth - rect.right - 12) + "px";
    }
    button.addEventListener("click", function () {
      if (help.classList.contains("help-open")) {
        dismiss();
      } else {
        help.classList.add("help-open");
        reveal();
      }
    });
    button.addEventListener("focus", reveal);
    help.addEventListener("mouseenter", reveal);
    help.addEventListener("mouseleave", function () {
      if (!help.contains(document.activeElement) && !help.classList.contains("help-open")) {
        dismiss();
      }
    });
    help.addEventListener("focusout", function (event) {
      if (!help.contains(event.relatedTarget)) { dismiss(); }
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { dismiss(); }
    });
    document.addEventListener("click", function (event) {
      if (!help.contains(event.target)) { dismiss(); }
    });
  });
}());
"""

REPORT_COMMON_JS = """/* civ-bench shared report utilities.
   Deterministic vanilla JavaScript: no packages, no network. Loaded by every
   report page that scripts its own content; defines window.civBench. */
(function () {
  "use strict";

  var HUE_STEP = 30;       // degrees per extra member of a same-color group
  var GRAY_L_STEP = 0.14;  // lightness step when a color has no hue to rotate

  function normalizeHex(color) {
    if (typeof color !== "string") {
      return null;
    }
    var value = color.trim().toLowerCase();
    if (/^#[0-9a-f]{6}$/.test(value)) {
      return value;
    }
    if (/^#[0-9a-f]{3}$/.test(value)) {
      return "#" + value.charAt(1) + value.charAt(1) + value.charAt(2) +
        value.charAt(2) + value.charAt(3) + value.charAt(3);
    }
    return null;
  }

  function hexToHsl(hex) {
    var r = parseInt(hex.slice(1, 3), 16) / 255;
    var g = parseInt(hex.slice(3, 5), 16) / 255;
    var b = parseInt(hex.slice(5, 7), 16) / 255;
    var max = Math.max(r, g, b);
    var min = Math.min(r, g, b);
    var l = (max + min) / 2;
    var h = 0;
    var s = 0;
    if (max !== min) {
      var d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      if (max === r) {
        h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
      } else if (max === g) {
        h = ((b - r) / d + 2) / 6;
      } else {
        h = ((r - g) / d + 4) / 6;
      }
    }
    return { h: h * 360, s: s, l: l };
  }

  function hueToRgb(p, q, t) {
    if (t < 0) { t += 1; }
    if (t > 1) { t -= 1; }
    if (t < 1 / 6) { return p + (q - p) * 6 * t; }
    if (t < 1 / 2) { return q; }
    if (t < 2 / 3) { return p + (q - p) * (2 / 3 - t) * 6; }
    return p;
  }

  function hslToHex(h, s, l) {
    h = (((h % 360) + 360) % 360) / 360;
    var rgb;
    if (s === 0) {
      var v = Math.round(l * 255);
      rgb = [v, v, v];
    } else {
      var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      var p = 2 * l - q;
      rgb = [
        Math.round(hueToRgb(p, q, h + 1 / 3) * 255),
        Math.round(hueToRgb(p, q, h) * 255),
        Math.round(hueToRgb(p, q, h - 1 / 3) * 255)
      ];
    }
    return "#" + rgb.map(function (channel) {
      var clamped = Math.max(0, Math.min(255, channel));
      var text = clamped.toString(16);
      return text.length === 1 ? "0" + text : text;
    }).join("");
  }

  /* distinguishColors(items) takes one { key, color } entry per series or
     model in display order and returns a key-to-color map. A color used by
     only one entry is passed through untouched; the first member of a shared
     color also keeps it, and every extra member rotates the hue by HUE_STEP
     degrees (colors too gray to rotate step lightness instead), so same-family
     models stay distinguishable. Keys repeat only by accident; the first
     occurrence wins. Colors outside #rgb/#rrggbb pass through unchanged. */
  function distinguishColors(items) {
    var out = {};
    var groups = {};
    var seen = {};
    (items || []).forEach(function (item) {
      if (!item || seen[item.key]) {
        return;
      }
      var hex = normalizeHex(item.color);
      if (hex === null) {
        seen[item.key] = true;
        out[item.key] = item.color;
        return;
      }
      seen[item.key] = hex;
      out[item.key] = hex;
      if (groups[hex]) {
        groups[hex].push(item.key);
      } else {
        groups[hex] = [item.key];
      }
    });
    Object.keys(groups).forEach(function (hex) {
      var keys = groups[hex];
      if (keys.length < 2) {
        return;
      }
      var hsl = hexToHsl(hex);
      keys.forEach(function (key, index) {
        if (index === 0) {
          return;  // the first member keeps the catalog color
        }
        if (hsl.s >= 0.1) {
          out[key] = hslToHex(hsl.h + HUE_STEP * index, hsl.s, hsl.l);
        } else {
          var l = hsl.l + (index - (keys.length - 1) / 2) * GRAY_L_STEP;
          out[key] = hslToHex(hsl.h, hsl.s, Math.max(0.12, Math.min(0.92, l)));
        }
      });
    });
    return out;
  }

  window.civBench = {
    distinguishColors: distinguishColors
  };
  document.querySelectorAll("a.replay-link").forEach(function (link) {
    if (link.dataset.direct === "true") { return; }
    if (window.location.protocol !== "http:" && window.location.protocol !== "https:") {
      link.title = "Open this report from a web server to launch the viewer, or drop the downloaded save onto it";
      return;
    }
    var viewer = new URL(link.dataset.viewer);
    viewer.searchParams.set("file", new URL(link.getAttribute("href"), window.location.href).href);
    new URLSearchParams(link.dataset.query).forEach(function (value, key) {
      viewer.searchParams.set(key, value);
    });
    link.href = viewer.href;
    link.target = "_blank";
    link.rel = "noopener";
  });
})();
"""


def render_common_script() -> str:
    """Return the shared, deterministic browser util script for report pages."""
    return REPORT_COMMON_JS
