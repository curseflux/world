"""Sample traversals from a trained model, for map reconstruction.

The paper's generate_sequences_for_map.py does this but imports osmnx at module
level and writes into the Manhattan-specific samples/ layout. This is the same
procedure without the geo dependency: prompt with (origin, destination) pairs,
sample greedily-free continuations, and write one sequence per line in the
`<origin> <destination> <dir> ... end` format reconstruct_map.py reads.
"""
import argparse
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "world-model-evaluation-main"))
import torch
from tqdm import tqdm

from utils import is_valid_sequence, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset name under data/")
    parser.add_argument("--out", required=True, help="output samples .txt")
    parser.add_argument("--num-sequences", type=int, default=6400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use-untrained-model", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    model = load_model(args.data, args.use_untrained_model)
    tokenizer = model.tokenizer
    valid_turns = tokenizer.valid_turns
    n2n = tokenizer.node_and_direction_to_neighbor
    eos_token_id = tokenizer.word_to_id["end"]

    with open(f"data/{args.data}/all_pairs.pkl", "rb") as f:
        all_pairs = pickle.load(f)

    samples, valid = [], 0
    batches = (args.num_sequences + args.batch_size - 1) // args.batch_size
    bar = tqdm(range(batches))
    for _ in bar:
        pairs = [random.choice(all_pairs) for _ in range(args.batch_size)]
        prefix = torch.tensor(
            [tokenizer.encode(f"{origin} {destination}") for origin, destination in pairs]
        ).to(model.device)
        with torch.no_grad():
            generated = model.model.generate(
                prefix, max_length=args.max_length, num_return_sequences=1,
                eos_token_id=eos_token_id, pad_token_id=tokenizer.pad_token_id,
                do_sample=True, temperature=args.temperature)
        for row in generated:
            sample = tokenizer.decode(row)
            samples.append(sample)
            valid += is_valid_sequence(sample, valid_turns, n2n)
        bar.set_description(f"valid {valid}/{len(samples)} ({valid/len(samples):.3f})")

    samples = samples[:args.num_sequences]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(samples))
    print(f"\nwrote {len(samples)} samples to {args.out}")
    print(f"valid under the true world model: {valid}/{len(samples)} "
          f"({valid / max(len(samples), 1):.3f})")
    print("\nReconstruct and score the implied map with:")
    print(f"  python gridworld/reconstruct_map.py --map-dir <map> "
          f"--samples {args.out} --out-dir results/{args.data}/map")


if __name__ == "__main__":
    main()
