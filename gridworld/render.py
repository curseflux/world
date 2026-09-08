"""Render grid-world maps as SVG.

Each panel is one map drawn as straight lines, one line per directed edge, and
nothing else -- no arrowheads, no node dots, no gridlines. Everything on the page
is a road. `mapping/make_maps.py` also draws edges only (it assigns a `radius`
and never uses it); a dot at every intersection makes a lattice read as graph
paper, which is precisely the wrong impression, since the regular-looking lines
*are* the roads.

Two palettes:

  TRUE          the ground-truth map, black
  RECONSTRUCTED the map a model's sequences imply, red -- every edge the
                reconstruction used, whether or not it exists in reality

Drawn side by side, the comparison is the whole point: extra red streets, or
missing ones, are read off against the black map beside it. `highlight_false`
splits the reconstructed panel back into black for real streets and red for
invented ones, which is useful zoomed in on a particular neighbourhood.

This departs from the paper in one respect. `make_maps.py` draws invented edges
as Bezier curves whose control point leans along the edge's own *label*, so an
edge labelled NW that runs east bulges northwest before swinging back -- its way
of showing "impossible physical orientations" and "flyovers". At 10x10 that
mostly produces a tangle, and the same signal is reported exactly as
`impossible_orientation_rate`, so edges are drawn straight here.
"""

TRUE = {"colour": "#111827", "width": 1.4, "alpha": 0.8}
RECONSTRUCTED = {"colour": "#d7263d", "width": 1.4, "alpha": 0.8}
UNUSED = {"colour": "#d4dae1", "width": 1.2, "alpha": 0.9}


def panel(graph, coords, cell=64, origin=(0, 0), palette=TRUE,
          highlight_false=False, show_unused=False, show_nodes=False):
    """SVG elements for one map, plus the size of the box they occupy."""
    ox, oy = origin
    rows = max(r for r, _ in coords.values()) + 1
    cols = max(c for _, c in coords.values()) + 1

    def xy(node):
        r, c = coords[node]
        return ox + c * cell, oy + r * cell

    def line(x1, y1, x2, y2, style):
        return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{style["colour"]}" stroke-width="{style["width"]}" '
                f'opacity="{style["alpha"]}"/>')

    parts = []
    # Draw missed and real edges first so invented ones sit on top of them.
    order = {"true_unused": 0, "true": 1, "new": 2}
    edges = sorted(graph.out_edges(keys=True, data=True),
                   key=lambda e: order.get(e[3].get("edge_type", "true"), 1))

    for u, v, _, data in edges:
        kind = data.get("edge_type", "true")
        if kind == "true_unused":
            if not show_unused:
                continue
            style = UNUSED
        elif highlight_false:
            style = RECONSTRUCTED if kind == "new" else TRUE
        else:
            style = palette
        x1, y1 = xy(u)
        x2, y2 = xy(v)
        parts.append(line(x1, y1, x2, y2, style))

    if show_nodes:
        for node in coords:
            x, y = xy(node)
            parts.append(f'<circle cx="{x}" cy="{y}" r="2.2" fill="#111827" opacity="0.5"/>')

    return parts, ((cols - 1) * cell, (rows - 1) * cell)


def _document(width, height, body):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        *body,
        '</svg>',
    ])


def _caption(x, y, text, size=12, colour="#4b5563", weight="normal"):
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" fill="{colour}" '
            f'font-weight="{weight}">{text}</text>')


def render(graph, coords, path, title="", subtitle="", cell=64, pad=40,
           palette=TRUE, highlight_false=False, show_unused=False, show_nodes=False):
    """Render a single map to `path`."""
    top = pad + (30 if title else 0) + (16 if subtitle else 0)
    body, (w, h) = panel(graph, coords, cell, origin=(pad, top), palette=palette,
                         highlight_false=highlight_false, show_unused=show_unused,
                         show_nodes=show_nodes)
    width, height = w + 2 * pad, top + h + pad
    head = []
    if title:
        head.append(_caption(pad, pad + 4, title, 15, "#111827", "600"))
    if subtitle:
        head.append(_caption(pad, pad + 22, subtitle, 12))
    with open(path, "w") as f:
        f.write(_document(width, height, head + body))


def render_pair(left, right, coords, path, left_title, right_title,
                left_sub="", right_sub="", cell=64, pad=40, gap=54,
                highlight_false=False, show_unused=False, show_nodes=False):
    """True map beside reconstructed map, in the layout of the paper's Figure 3."""
    top = pad + 46
    left_body, (w, h) = panel(left, coords, cell, origin=(pad, top), palette=TRUE,
                              show_nodes=show_nodes)
    right_x = pad + w + gap
    right_body, _ = panel(right, coords, cell, origin=(right_x, top),
                          palette=RECONSTRUCTED, highlight_false=highlight_false,
                          show_unused=show_unused, show_nodes=show_nodes)
    width, height = right_x + w + pad, top + h + pad

    head = [
        _caption(pad, pad + 2, left_title, 14, "#111827", "600"),
        _caption(pad, pad + 20, left_sub, 11.5),
        _caption(right_x, pad + 2, right_title, 14, "#111827", "600"),
        _caption(right_x, pad + 20, right_sub, 11.5),
    ]
    with open(path, "w") as f:
        f.write(_document(width, height, head + left_body + right_body))
