"""Break a model's generation errors down by position, path length and state.

The headline `token_error_rate` is one number; this says *where* the errors fall,
which is what separates two very different failure modes:

  flat hazard    errors arrive at a constant per-token rate regardless of how far
                 into a path the model is -- transcription noise over a map it
                 basically has
  rising hazard  the error rate climbs with position, i.e. the model loses track
                 of where it is as the path lengthens -- a state-tracking failure,
                 which is the paper's actual claim

Because the walk's true state is undefined after the first illegal token, per-
position rates are *hazard* rates: among sequences still legal at position k, the
fraction whose k-th token is illegal. That conditioning is what makes flat-versus-
rising the right question to ask of them.

Writes error_analysis.json (the numbers, and the table view) and
error_analysis.svg (six panels). Bars carry Wilson 95% intervals, so a bin with
few samples is visibly uncertain rather than silently noisy.

    python gridworld/analyze_errors.py --map-dir gridworld/maps/nyc10 \\
        --samples world-model-evaluation-main/results/<run>/samples.txt \\
        --out-dir world-model-evaluation-main/results/<run>
"""
import argparse
import collections
import json
import math
import os
import pickle

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
# Uncertainty whiskers are chrome, not a second series: they are held to
# legibility over both the surface and the bar they cross, not to the
# categorical lightness band. A surface-coloured casing under a dark stroke is
# the documented way to keep an overlapping mark crisp.
WHISKER = "#0b0b0b"


def wilson(successes, total, z=1.96):
    """95% interval for a rate. A bin of 3 samples should look like one."""
    if not total:
        return 0.0, 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_map(map_dir):
    with open(f"{map_dir}/valid_turns.pkl", "rb") as f:
        valid_turns = pickle.load(f)
    with open(f"{map_dir}/node_and_direction_to_neighbor.pkl", "rb") as f:
        n2n = pickle.load(f)
    return valid_turns, n2n


def hop_distances(valid_turns, n2n):
    """(origin, destination) -> fewest hops, for the planning panel."""
    distances = {}
    for origin in valid_turns:
        seen, frontier, depth = {origin}, [origin], 0
        while frontier:
            depth += 1
            nxt = []
            for node in frontier:
                for direction in valid_turns[node]:
                    neighbour = n2n[(node, direction)]
                    if neighbour not in seen:
                        seen.add(neighbour)
                        distances[(origin, neighbour)] = depth
                        nxt.append(neighbour)
            frontier = nxt
    return distances


def walk(sequence, valid_turns, n2n):
    """Replay one generated sequence against the true map.

    Returns the per-token observations up to and including the first illegal
    move, plus whether the sequence finished legally at its stated destination.
    """
    origin, destination = sequence[0], sequence[1]
    node = origin
    observations = []  # (position, direction, out_degree, illegal)
    for position, direction in enumerate(sequence[2:], start=1):
        legal = direction in valid_turns.get(node, ())
        observations.append((position, direction, len(valid_turns.get(node, ())), not legal))
        if not legal:
            return observations, False, False
        node = n2n[(node, direction)]
    return observations, True, node == destination


def analyse(sequences, valid_turns, n2n, distances):
    """Every breakdown in one pass."""
    by_position = collections.defaultdict(lambda: [0, 0])      # hazard
    by_out_degree = collections.defaultdict(lambda: [0, 0])
    by_direction = collections.defaultdict(lambda: [0, 0])
    by_length = collections.defaultdict(lambda: [0, 0])        # invalid-sequence rate
    by_od_distance = collections.defaultdict(lambda: [0, 0])   # reached-destination rate
    first_error = collections.Counter()
    tokens = errors = legal_sequences = arrived = 0

    for sequence in sequences:
        observations, legal, reached = walk(sequence, valid_turns, n2n)
        length = len(sequence) - 2
        for position, direction, degree, illegal in observations:
            tokens += 1
            errors += illegal
            for table, key in ((by_position, position), (by_out_degree, degree),
                               (by_direction, direction)):
                table[key][1] += 1
                table[key][0] += illegal
            if illegal:
                first_error[position] += 1

        by_length[length][1] += 1
        by_length[length][0] += not legal
        legal_sequences += legal
        arrived += reached

        distance = distances.get((sequence[0], sequence[1]))
        if distance is not None:
            by_od_distance[distance][1] += 1
            by_od_distance[distance][0] += reached

    return {
        "sequences": len(sequences),
        "tokens_examined": tokens,
        "illegal_tokens": errors,
        "token_error_rate": round(errors / tokens, 6) if tokens else 0.0,
        "legal_sequence_rate": round(legal_sequences / len(sequences), 4) if sequences else 0.0,
        "reached_destination_rate": round(arrived / len(sequences), 4) if sequences else 0.0,
        "by_position": {k: v for k, v in sorted(by_position.items())},
        "by_sequence_length": {k: v for k, v in sorted(by_length.items())},
        "by_out_degree": {k: v for k, v in sorted(by_out_degree.items())},
        "by_direction": {k: v for k, v in sorted(by_direction.items())},
        "by_od_distance": {k: v for k, v in sorted(by_od_distance.items())},
        "first_error_position": {k: v for k, v in sorted(first_error.items())},
    }


def rebin(table, target=14):
    """Merge adjacent numeric bins until at most `target` remain."""
    keys = sorted(table)
    if len(keys) <= target:
        return [(str(k), table[k][0], table[k][1]) for k in keys]
    step = math.ceil(len(keys) / target)
    out = []
    for i in range(0, len(keys), step):
        chunk = keys[i:i + step]
        hits = sum(table[k][0] for k in chunk)
        total = sum(table[k][1] for k in chunk)
        label = str(chunk[0]) if len(chunk) == 1 else f"{chunk[0]}-{chunk[-1]}"
        out.append((label, hits, total))
    return out


def bars(bins, x, y, width, height, title, subtitle, ylabel, counts_only=False,
         reference=None):
    """One panel: title, hairline grid, capped bars, Wilson whiskers."""
    pad_left, pad_bottom, pad_top = 46, 30, 44
    plot_w = width - pad_left - 10
    plot_h = height - pad_top - pad_bottom
    x0, y0 = x + pad_left, y + pad_top

    values = []
    for label, hits, total in bins:
        if counts_only:
            values.append((label, hits, hits, hits, total))
        else:
            rate, low, high = wilson(hits, total)
            values.append((label, rate, low, high, total))

    ceiling = max([v[3] for v in values] + [1e-9])
    ceiling = max(ceiling * 1.15, 1e-9)
    if counts_only:
        ticks = [0, ceiling / 2, ceiling]
        fmt = lambda t: f"{int(round(t))}"
    else:
        ticks = [0, ceiling / 2, ceiling]
        fmt = lambda t: f"{t:.1%}" if ceiling < 0.25 else f"{t:.0%}"

    parts = [
        f'<text x="{x}" y="{y + 16}" font-size="13" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="{x}" y="{y + 32}" font-size="10.5" fill="{SECONDARY}">{subtitle}</text>',
    ]
    for tick in ticks:
        ty = y0 + plot_h - (tick / ceiling) * plot_h
        parts.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0 + plot_w}" y2="{ty:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 6}" y="{ty + 3.5:.1f}" font-size="9.5" fill="{MUTED}" '
                     f'text-anchor="end" font-variant-numeric="tabular-nums">{fmt(tick)}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" '
                 f'stroke="{AXIS}" stroke-width="1"/>')

    # The whole question on a hazard panel is "do the bars track this line or
    # climb across it", which is unanswerable without the line drawn.
    if reference is not None:
        value, label = reference
        ry = y0 + plot_h - (value / ceiling) * plot_h
        if y0 <= ry <= y0 + plot_h:
            parts.append(f'<line x1="{x0}" y1="{ry:.1f}" x2="{x0 + plot_w}" y2="{ry:.1f}" '
                         f'stroke="{WHISKER}" stroke-width="1" opacity="0.45"/>')
            parts.append(f'<text x="{x0 + plot_w}" y="{ry - 4:.1f}" font-size="9" '
                         f'fill="{MUTED}" text-anchor="end">{label}</text>')
    parts.append(f'<text x="{x0 - 38}" y="{y0 - 8}" font-size="9.5" fill="{MUTED}">{ylabel}</text>')

    band = plot_w / max(len(values), 1)
    bar_w = min(24.0, band - 2)  # 2px surface gap between neighbours
    for i, (label, value, low, high, total) in enumerate(values):
        cx = x0 + band * i + band / 2
        bx = cx - bar_w / 2
        bh = (value / ceiling) * plot_h
        by = y0 + plot_h - bh
        radius = min(4.0, bar_w / 2, max(bh, 0))
        if bh <= 0.5:
            path = ""
        elif bh <= radius:
            path = f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{SERIES}"/>'
        else:
            path = (f'<path d="M{bx:.1f},{y0 + plot_h:.1f} L{bx:.1f},{by + radius:.1f} '
                    f'Q{bx:.1f},{by:.1f} {bx + radius:.1f},{by:.1f} '
                    f'L{bx + bar_w - radius:.1f},{by:.1f} '
                    f'Q{bx + bar_w:.1f},{by:.1f} {bx + bar_w:.1f},{by + radius:.1f} '
                    f'L{bx + bar_w:.1f},{y0 + plot_h:.1f} Z" fill="{SERIES}"/>')
        readable = f"{value:.0f}" if counts_only else f"{value:.2%}"
        parts.append(f'<g><title>{label}: {readable} (n={total})</title>{path}')
        if not counts_only and total:
            ly = y0 + plot_h - (low / ceiling) * plot_h
            hy = y0 + plot_h - (high / ceiling) * plot_h
            cap = min(5.0, bar_w / 2.5)
            spine = (f'M{cx:.1f},{hy:.1f} L{cx:.1f},{ly:.1f} '
                     f'M{cx - cap:.1f},{hy:.1f} L{cx + cap:.1f},{hy:.1f} '
                     f'M{cx - cap:.1f},{ly:.1f} L{cx + cap:.1f},{ly:.1f}')
            parts.append(f'<path d="{spine}" stroke="{SURFACE}" stroke-width="3.4" '
                         f'fill="none" stroke-linecap="round"/>')
            parts.append(f'<path d="{spine}" stroke="{WHISKER}" stroke-width="1.2" '
                         f'fill="none" opacity="0.72" stroke-linecap="round"/>')
        parts.append('</g>')
        if len(values) <= 16:
            parts.append(f'<text x="{cx:.1f}" y="{y0 + plot_h + 13:.1f}" font-size="9" '
                         f'fill="{MUTED}" text-anchor="middle">{label}</text>')
    if len(values) > 16:
        for i in (0, len(values) - 1):
            cx = x0 + band * i + band / 2
            parts.append(f'<text x="{cx:.1f}" y="{y0 + plot_h + 13:.1f}" font-size="9" '
                         f'fill="{MUTED}" text-anchor="middle">{values[i][0]}</text>')
    return parts


def render(report, path, heading):
    overall = report["token_error_rate"]
    arrived = report["reached_destination_rate"]
    panels = [
        ("by_position", "1. Error rate vs how deep into the path",
         "FLAT = steady noise   RISING = losing track of state",
         "wrong turns", False, (overall, f"overall {overall:.2%}")),
        ("by_sequence_length", "2. Broken paths vs path length",
         "share of generated paths of that length with any wrong turn",
         "broken paths", False, None),
        ("by_od_distance", "3. Arrived at destination vs how far it was",
         "short trips easy, long trips hard = a planning limit, not a legality one",
         "arrived", False, (arrived, f"overall {arrived:.0%}")),
        ("first_error_position", "4. Where the first wrong turn happens",
         "count of paths whose first wrong turn is at that step",
         "paths", True, None),
        ("by_out_degree", "5. Error rate vs junction complexity",
         "out-degree of the node the model was standing on",
         "wrong turns", False, (overall, f"overall {overall:.2%}")),
        ("by_direction", "6. Error rate vs direction emitted",
         "is one token disproportionately wrong?",
         "wrong turns", False, (overall, f"overall {overall:.2%}")),
    ]
    pw, ph, gap = 400, 210, 34
    body = [f'<text x="30" y="30" font-size="16" font-weight="600" fill="{INK}">{heading}</text>',
            f'<text x="30" y="50" font-size="11.5" fill="{SECONDARY}">'
            f'{report["illegal_tokens"]} illegal of {report["tokens_examined"]} tokens examined '
            f'({report["token_error_rate"]:.3%} per token) across {report["sequences"]} sequences; '
            f'{report["legal_sequence_rate"]:.1%} of paths fully legal, '
            f'{report["reached_destination_rate"]:.1%} reached their destination. '
            f'Taller bar = more wrong turns. Whiskers are Wilson 95% intervals, so a '
            f'tall whisker means too few samples to trust that bar; hover for n.</text>']
    for index, (key, title, subtitle, ylabel, counts_only, reference) in enumerate(panels):
        table = report[key]
        if not table:
            continue
        numeric = {int(k) if str(k).lstrip("-").isdigit() else k: v for k, v in table.items()}
        if counts_only:
            binned = [(str(k), v, v) for k, v in sorted(numeric.items())][:20]
        else:
            binned = rebin(numeric)
        col, row = index % 2, index // 2
        body += bars(binned, 30 + col * (pw + gap), 74 + row * (ph + gap),
                     pw, ph, title, subtitle, ylabel, counts_only, reference)

    width = 30 * 2 + pw * 2 + gap
    height = 74 + 3 * (ph + gap)
    with open(path, "w") as f:
        f.write("\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" '
            f'font-family="system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
            f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', *body, '</svg>']))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", default=None, help="heading for the figure")
    args = parser.parse_args()

    valid_turns, n2n = load_map(args.map_dir)
    with open(args.samples) as f:
        raw = [line for line in f.read().split("\n") if line.strip()]
    sequences = []
    for line in raw:
        tokens = line.split(" ")
        if len(tokens) > 3 and tokens[-1] == "end":
            sequences.append([int(tokens[0]), int(tokens[1])] + tokens[2:-1])
    if not sequences:
        raise SystemExit("no well-formed sequences in the samples file")

    report = analyse(sequences, valid_turns, n2n, hop_distances(valid_turns, n2n))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/error_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    render(report, f"{args.out_dir}/error_analysis.svg",
           args.label or os.path.basename(os.path.abspath(args.out_dir)))

    print(f"{report['illegal_tokens']}/{report['tokens_examined']} tokens illegal "
          f"({report['token_error_rate']:.3%} per token)")
    print(f"{report['legal_sequence_rate']:.1%} of paths fully legal, "
          f"{report['reached_destination_rate']:.1%} reached the destination")
    positions = report["by_position"]
    if len(positions) > 6:
        keys = sorted(positions)
        early = keys[:len(keys) // 3]
        late = keys[-len(keys) // 3:]

        def rate(subset):
            hits = sum(positions[k][0] for k in subset)
            total = sum(positions[k][1] for k in subset)
            return hits / total if total else 0.0

        print(f"hazard early {rate(early):.3%} vs late {rate(late):.3%} -- "
              f"{'rising: state tracking degrades with path length' if rate(late) > 2 * rate(early) else 'flat: errors look position-independent'}")
    print(f"-> {args.out_dir}/error_analysis.svg")


if __name__ == "__main__":
    main()
