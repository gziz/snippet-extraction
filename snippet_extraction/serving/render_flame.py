"""Render measured spans (/tmp/flame_spans.json) as an icicle/flame-graph SVG.

No external deps — emits hand-built SVG so it stays crisp at any zoom and the
phase labels are real text. One panel per doc size; each panel is normalized to
its own total compute time (so you read PROPORTION of the bottleneck), with
absolute ms on every bar and a category summary on the right.

    python render_flame.py            # -> /tmp/flame.svg
"""

from __future__ import annotations

import json
from pathlib import Path

# category -> fill colour
CAT = {
    "Segmentation": "#e8743b",  # syntok sentence splitter (pure Python)
    "Tokenization": "#f6c244",  # HF fast tokenizer
    "GPU forward": "#2f6df6",  # the encoder forward pass
    "Transfer": "#8e7cff",  # H2D / D2H copies
    "Pooling": "#19a979",  # numpy per-unit pooling
    "Other": "#9aa3af",  # build spans / select+render
}
CONTAINER = "#e9edf4"  # parent/container spans (muted)
CONTAINER_STROKE = "#c4ccd8"


def classify(name: str) -> str | None:
    if name.startswith(("request total", "score", "minibatch loop", "minibatch ")):
        return None  # container
    if "segment" in name:
        return "Segmentation"
    if "tokenize" in name or "plan windows" in name:
        return "Tokenization"
    if "forward" in name:
        return "GPU forward"
    if "H2D" in name or "D2H" in name or "sigmoid" in name:
        return "Transfer"
    if "pool" in name:
        return "Pooling"
    return "Other"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(size: str, res: dict, x0: float, y0: float, W: float) -> tuple[str, float]:
    spans = res["spans"]
    total = res["compute_ms"]
    row_h = 30.0
    gap = 1.0
    max_depth = max(s["depth"] for s in spans)
    px_per_ms = W / total

    # category totals from leaves
    cats: dict[str, float] = {}
    for s in spans:
        c = classify(s["name"])
        if c:
            cats[c] = cats.get(c, 0.0) + s["dur"]

    out = []
    # title
    out.append(
        f'<text x="{x0}" y="{y0 - 14}" font-size="17" font-weight="700" '
        f'fill="#1a2230">{esc(size.upper())} doc — {res["n_units"]:,} sentences '
        f"→ {res['n_windows']} windows, {res['n_minibatches']} GPU passes "
        f"· {total:.0f} ms total</text>"
    )

    for s in spans:
        c = classify(s["name"])
        x = x0 + s["start"] * px_per_ms
        w = max(0.4, s["dur"] * px_per_ms - gap)
        y = y0 + s["depth"] * row_h
        h = row_h - gap
        fill = CONTAINER if c is None else CAT[c]
        stroke = CONTAINER_STROKE if c is None else "#ffffff"
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="2.5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        # label if wide enough
        if w > 46:
            txt_fill = "#1a2230" if c in (None, "Tokenization", "Pooling") else "#ffffff"
            label = s["name"]
            ms = f"{s['dur']:.0f}ms"
            # trim label to fit
            maxchars = int(w / 6.6)
            shown = (
                label
                if len(label) + len(ms) + 2 <= maxchars
                else label[: max(0, maxchars - len(ms) - 2)].rstrip()
            )
            out.append(
                f'<text x="{x + 6:.1f}" y="{y + h / 2 + 4:.1f}" font-size="11.5" '
                f'fill="{txt_fill}" font-family="monospace">{esc(shown)} '
                f'<tspan font-weight="700">{ms}</tspan></text>'
            )

    chart_bottom = y0 + (max_depth + 1) * row_h

    # category summary bar (stacked) under the icicle
    sy = chart_bottom + 24
    out.append(
        f'<text x="{x0}" y="{sy - 8}" font-size="12" font-weight="700" '
        f'fill="#56607a">where the {total:.0f} ms goes</text>'
    )
    bx = x0
    order = ["GPU forward", "Segmentation", "Tokenization", "Pooling", "Transfer", "Other"]
    for c in order:
        v = cats.get(c, 0.0)
        if v <= 0:
            continue
        w = v / total * W
        out.append(f'<rect x="{bx:.1f}" y="{sy:.1f}" width="{w:.1f}" height="26" fill="{CAT[c]}"/>')
        if w > 60:
            pct = v / total * 100
            out.append(
                f'<text x="{bx + 8:.1f}" y="{sy + 17:.1f}" font-size="11.5" '
                f'fill="#fff" font-family="monospace" font-weight="700">'
                f"{pct:.0f}%</text>"
            )
        bx += w
    return "\n".join(out), sy + 26 + 30


def main():
    data = json.loads(Path("/tmp/flame_spans.json").read_text())
    W = 1180.0
    margin = 30.0
    panels = []
    y = 56.0
    for size in ("long", "medium"):
        if size not in data:
            continue
        body, next_y = panel(size, data[size], margin, y, W)
        panels.append(body)
        y = next_y + 36

    height = y
    legend_y = 24
    legend = [
        f'<text x="{margin}" y="{legend_y}" font-size="20" font-weight="800" '
        f'fill="#0e1420">compress() request flame graph — L40S, warm</text>'
    ]
    # simple horizontal legend row
    lx = margin + 560
    for c, col in CAT.items():
        legend.append(
            f'<rect x="{lx}" y="{legend_y - 13}" width="13" height="13" rx="2.5" fill="{col}"/>'
        )
        legend.append(
            f'<text x="{lx + 18}" y="{legend_y - 2}" font-size="12" fill="#56607a">{c}</text>'
        )
        lx += 22 + len(c) * 7.2 + 14

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W + 2 * margin:.0f}" '
        f'height="{height:.0f}" font-family="-apple-system,Segoe UI,Roboto,sans-serif" '
        f'viewBox="0 0 {W + 2 * margin:.0f} {height:.0f}">'
        f'<rect width="100%" height="100%" fill="#fbfcfe"/>'
        + "\n".join(legend)
        + "\n"
        + "\n".join(panels)
        + "</svg>"
    )
    Path("/tmp/flame.svg").write_text(svg)
    print("wrote /tmp/flame.svg", f"({W + 2 * margin:.0f}x{height:.0f})")


if __name__ == "__main__":
    main()
