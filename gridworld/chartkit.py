"""Minimal SVG chart primitives shared by the analysis scripts.

Palette and mark specs follow the project's data-viz conventions: categorical
slots assigned in fixed order and never cycled, 2px lines with >=8px end markers,
hairline recessive grid, text in ink tokens rather than series colours, and a
direct label on every series so identity is never carried by colour alone.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CHROME = "#0b0b0b"

# Categorical slots, in fixed order. The first three validate on the all-pairs
# list in both modes, which is the cap for a chart with markers.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def document(width, height, body):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', *body, '</svg>'])


def text(x, y, body, size=11, fill=SECONDARY, weight="normal", anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{body}</text>')


def line_chart(series, x_labels, path, title, subtitle="", y_label="",
               reference=None, width=760, height=420, y_max=1.0):
    """Plot several series over a shared ordinal x axis.

    `series` is [(name, [values...]), ...]; x positions are evenly spaced, so a
    doubling budget reads as a log axis without a log scale to misread.
    `reference` is an optional (value, label) drawn as chrome, not as a series.
    """
    left, right, top, bottom = 62, 96, 78, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    x0, y0 = left, top

    def px(i):
        if len(x_labels) == 1:
            return x0 + plot_w / 2
        return x0 + plot_w * i / (len(x_labels) - 1)

    def py(v):
        return y0 + plot_h - (v / y_max) * plot_h

    body = [text(30, 30, title, 16, INK, "600"),
            text(30, 50, subtitle, 11.5, SECONDARY)]

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        value = tick * y_max
        ty = py(value)
        body.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0 + plot_w}" y2="{ty:.1f}" '
                    f'stroke="{GRID}" stroke-width="1"/>')
        body.append(text(x0 - 8, ty + 3.5, f"{value:.2f}", 9.5, MUTED, anchor="end"))
    body.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" '
                f'y2="{y0 + plot_h}" stroke="{AXIS}" stroke-width="1"/>')
    if y_label:
        body.append(text(x0 - 54, y0 - 10, y_label, 9.5, MUTED))

    for i, label in enumerate(x_labels):
        body.append(text(px(i), y0 + plot_h + 16, str(label), 9.5, MUTED, anchor="middle"))

    if reference is not None:
        value, label = reference
        ry = py(value)
        body.append(f'<line x1="{x0}" y1="{ry:.1f}" x2="{x0 + plot_w}" y2="{ry:.1f}" '
                    f'stroke="{CHROME}" stroke-width="1" opacity="0.45"/>')
        body.append(text(x0 + plot_w - 4, ry - 5, label, 9.5, MUTED, anchor="end"))

    # Legend: identity is never colour-alone, so every series is also labelled at
    # its own line end below.
    lx = x0
    for slot, (name, _) in enumerate(series):
        colour = SERIES[slot % len(SERIES)]
        body.append(f'<line x1="{lx}" y1="{y0 - 22}" x2="{lx + 16}" y2="{y0 - 22}" '
                    f'stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        body.append(text(lx + 22, y0 - 18.5, name, 10.5, SECONDARY))
        lx += 26 + len(name) * 6.4

    taken = []
    for slot, (name, values) in enumerate(series):
        colour = SERIES[slot % len(SERIES)]
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        body.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                    f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, value in enumerate(values):
            body.append(f'<g><title>{name} at {x_labels[i]}: {value:.3f}</title>'
                        f'<circle cx="{px(i):.1f}" cy="{py(value):.1f}" r="4" '
                        f'fill="{colour}" stroke="{SURFACE}" stroke-width="1.5"/></g>')
        # Direct label at the line end, nudged clear of its neighbours.
        ly = py(values[-1])
        while any(abs(ly - other) < 12 for other in taken):
            ly += 12
        taken.append(ly)
        body.append(text(x0 + plot_w + 10, ly + 3.5,
                         f"{name} {values[-1]:.2f}", 10.5, SECONDARY))

    with open(path, "w") as f:
        f.write(document(width, height, body))
