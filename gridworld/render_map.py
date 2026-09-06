"""Render a grid world's true map, in the paper's figure convention.

    python gridworld/render_map.py --map-dir gridworld/maps/nyc10 --out true_map.svg

For the true-versus-reconstructed pair, reconstruct_map.py writes that directly
as true_vs_reconstructed.svg.
"""
import argparse
import os
import pickle

import networkx as nx

import render


def load_graph(map_dir):
    with open(f"{map_dir}/node_and_direction_to_neighbor.pkl", "rb") as f:
        n2n = pickle.load(f)
    with open(f"{map_dir}/coords.pkl", "rb") as f:
        coords = pickle.load(f)
    graph = nx.MultiDiGraph()
    for node, position in coords.items():
        graph.add_node(node, lat_long=position)
    for (node, direction), neighbor in n2n.items():
        graph.add_edge(node, neighbor, direction=direction, edge_type="true")
    return graph, coords


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="True world model")
    parser.add_argument("--cell", type=int, default=64)
    args = parser.parse_args()

    graph, coords = load_graph(args.map_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    degrees = [graph.out_degree(n) for n in graph.nodes]
    subtitle = (f"{graph.number_of_nodes()} intersections, {graph.number_of_edges()} "
                f"one-way streets, mean out-degree "
                f"{sum(degrees) / len(degrees):.2f}")
    render.render(graph, coords, args.out, title=args.title, subtitle=subtitle,
                  cell=args.cell)
    print(subtitle)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
