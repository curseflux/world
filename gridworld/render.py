"""Render grid-world maps in the convention the paper uses for its Manhattan figures.

Read off mapping/make_maps.py, which is what produced Figures 3 and 9:

  * true edges the reconstruction used  -- straight, thin, black, no direction shown
  * false edges the reconstruction added -- CURVED, and coloured with a
    lightsalmon -> firebrick gradient running source to target. That gradient is
    what the Figure 9 caption means by "a darkening gradient indicating the
    directionality of the edge".
  * true edges never used -- skipped entirely (`continue` in make_map)
  * nodes -- not drawn at all. make_map assigns a `radius` and never uses it; it
    only ever adds edge lines. A dot at every intersection makes a lattice read
    as graph paper, which is exactly the wrong impression: the regular-looking
    lines are roads, not gridlines.

No arrowheads anywhere: on a graph this dense they bury the signal. Direction is
carried by the gradient, and only on the edges where it matters.

The curve is the other half of the trick. Its Bezier control point is offset from
the *source* in the direction of the edge's own label, so an edge labelled NW that
actually runs east bulges northwest before swinging back. That is how the paper's
"streets with impossible physical orientations" and "flyovers above other streets"
become visible rather than merely tabulated.

One deliberate deviation: make_map draws true edges at alpha 0.3, which suits
9,846 edges packed into Manhattan. A 10x10 grid has ~290, so they are drawn more
solidly here or the map would read as blank.
"""
import math

# lightsalmon -> firebrick, the cmap_new of mapping/make_maps.py
FALSE_EDGE_FROM = (255, 160, 122)
FALSE_EDGE_TO = (178, 34, 34)

STYLE = {
    "true": {"color": "#111827", "width": 1.4, "alpha": 0.75},
    "true_unused": {"color": "#d4dae1", "width": 1.2, "alpha": 0.9},
}

_LABEL_OFFSET = {
    "N": (0, -1), "NE": (1, -1), "E": (1, 0), "SE": (1, 1),
    "S": (0, 1), "SW": (-1, 1), "W": (-1, 0), "NW": (-1, -1),
}


def _blend(t):
    r = FALSE_EDGE_FROM[0] + (FALSE_EDGE_TO[0] - FALSE_EDGE_FROM[0]) * t
    g = FALSE_EDGE_FROM[1] + (FALSE_EDGE_TO[1] - FALSE_EDGE_FROM[1]) * t
    b = FALSE_EDGE_FROM[2] + (FALSE_EDGE_TO[2] - FALSE_EDGE_FROM[2]) * t
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _curve_points(x1, y1, x2, y2, direction, bulge, steps=14):
    """Quadratic Bezier whose control point leans along the edge's *label*."""
    dx, dy = _LABEL_OFFSET.get(direction, (0, 0))
    norm = math.hypot(dx, dy) or 1
    cx = x1 + bulge * dx / norm
    cy = y1 + bulge * dy / norm
    points = []
    for i in range(steps + 1):
        t = i / steps
        points.append((
            (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2,
            (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2,
        ))
    return points


def panel(graph, coords, cell=64, show_unused=False, origin=(0, 0), show_nodes=False):
    """SVG elements for one map, plus the size of the box they occupy."""
    ox, oy = origin
    rows = max(r for r, _ in coords.values()) + 1
    cols = max(c for _, c in coords.values()) + 1

    def xy(node):
        r, c = coords[node]
        return ox + c * cell, oy + r * cell

    parts = []
    order = {"true_unused": 0, "true": 1, "new": 2}
    edges = sorted(graph.out_edges(keys=True, data=True),
                   key=lambda e: order.get(e[3].get("edge_type", "true"), 1))

    for u, v, _, data in edges:
        kind = data.get("edge_type", "true")
        if kind == "true_unused" and not show_unused:
            continue
        x1, y1 = xy(u)
        x2, y2 = xy(v)
        if kind == "new":
            points = _curve_points(x1, y1, x2, y2, data.get("direction"), bulge=cell * 0.55)
            for i in range(len(points) - 1):
                (px, py), (qx, qy) = points[i], points[i + 1]
                colour = _blend(i / (len(points) - 2) if len(points) > 2 else 1.0)
                parts.append(
                    f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{qx:.1f}" y2="{qy:.1f}" '
                    f'stroke="{colour}" stroke-width="1.7" stroke-linecap="round" opacity="0.85"/>')
        else:
            style = STYLE[kind]
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{style["color"]}" stroke-width="{style["width"]}" '
                f'opacity="{style["alpha"]}"/>')

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
           show_unused=False, show_nodes=False):
    """Render a single map to `path`."""
    top = pad + (30 if title else 0) + (16 if subtitle else 0)
    body, (w, h) = panel(graph, coords, cell, show_unused, origin=(pad, top),
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
                show_unused=False, show_nodes=False):
    """Two maps side by side, in the layout of the paper's Figure 3."""
    top = pad + 46
    left_body, (w, h) = panel(left, coords, cell, show_unused, origin=(pad, top),
                              show_nodes=show_nodes)
    right_x = pad + w + gap
    right_body, _ = panel(right, coords, cell, show_unused, origin=(right_x, top),
                          show_nodes=show_nodes)
    width, height = right_x + w + pad, top + h + pad

    head = [
        _caption(pad, pad + 2, left_title, 14, "#111827", "600"),
        _caption(pad, pad + 20, left_sub, 11.5),
        _caption(right_x, pad + 2, right_title, 14, "#111827", "600"),
        _caption(right_x, pad + 20, right_sub, 11.5),
    ]
    with open(path, "w") as f:
        f.write(_document(width, height, head + left_body + right_body))
