"""Self-consistency checks on a generated dataset.

Mirrors `is_valid_sequence` from world-model-evaluation-main/utils.py (which we
cannot import directly -- that module pulls in torch).  If any of these fail,
every downstream world-model metric is measuring a broken dataset rather than a
broken model, so run this before training anything.
"""
import argparse
import collections
import json
import pickle


def is_valid_sequence(sample, valid_turns, n2n):
    parts = sample.split(" ")
    if len(parts) < 3:
        return False
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    node = start
    for direction in parts[2:]:
        if direction == "end":
            return node == end
        if direction not in valid_turns.get(node, ()):
            return False
        node = n2n[(node, direction)]
    return False


def read(path):
    with open(path) as f:
        return [line for line in f.read().split("\n") if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    with open(f"{args.data_dir}/valid_turns.pkl", "rb") as f:
        valid_turns = pickle.load(f)
    with open(f"{args.data_dir}/node_and_direction_to_neighbor.pkl", "rb") as f:
        n2n = pickle.load(f)

    train = read(f"{args.data_dir}/train_sequences.txt")
    heldout = read(f"{args.data_dir}/heldout_sequences.txt")

    report = {}
    for name, split in [("train", train), ("heldout", heldout)]:
        invalid = [s for s in split if not is_valid_sequence(s, valid_turns, n2n)]
        report[f"{name}_sequences"] = len(split)
        report[f"{name}_invalid"] = len(invalid)
        if invalid:
            report[f"{name}_invalid_example"] = invalid[0][:120]
        report[f"{name}_max_tokens"] = max((len(s.split(" ")) for s in split), default=0)

    def pairs(split):
        return {tuple(s.split(" ")[:2]) for s in split}

    report["od_pair_leakage"] = len(pairs(train) & pairs(heldout))

    # Coverage: can the model even see every state and every edge?
    seen_nodes, seen_edges = set(), set()
    for seq in train:
        parts = seq.split(" ")
        node = int(parts[0])
        seen_nodes.add(node)
        for direction in parts[2:]:
            if direction == "end":
                break
            seen_edges.add((node, direction))
            node = n2n[(node, direction)]
            seen_nodes.add(node)
    total_edges = sum(len(v) for v in valid_turns.values())
    report["nodes_covered"] = f"{len(seen_nodes)}/{len(valid_turns)}"
    report["edges_covered"] = f"{len(seen_edges)}/{total_edges}"

    lengths = collections.Counter(len(s.split(" ")) for s in train)
    report["median_train_length"] = sorted(lengths.elements())[len(train) // 2] if train else 0

    print(json.dumps(report, indent=2))
    failures = [k for k in report if k.endswith("_invalid") and report[k]]
    if failures or report["od_pair_leakage"]:
        print("\nFAILED:", ", ".join(failures + (["od_pair_leakage"] if report["od_pair_leakage"] else [])))
        raise SystemExit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
