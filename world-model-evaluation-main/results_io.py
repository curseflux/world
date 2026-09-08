"""Persist evaluation results, one directory per run.

The metric scripts originally reported only through `tqdm.set_description`, so a
number lived in a progress-bar line on stderr and was gone as soon as the
terminal scrolled. Each script now also writes a JSON record here and echoes a
final line to stdout.

Results are keyed on the *run* rather than the dataset, so several architectures
trained on one dataset land in separate directories instead of overwriting each
other. Every record carries the architecture it was produced with, which is what
lets compare_runs.py build a table without re-reading any checkpoints.
"""
import json
import os


def save_result(script, data, metrics, use_untrained_model=False, extra=None,
                run=None, architecture=None):
    """Write results/<run>/<script>[-untrained].json and echo a summary."""
    run = run or data
    name = f"{script}-untrained" if use_untrained_model else script
    out_dir = os.path.join("results", run)
    os.makedirs(out_dir, exist_ok=True)
    record = {"script": script, "run": run, "data": data,
              "untrained": bool(use_untrained_model)}
    if architecture:
        record.update({k: architecture[k] for k in sorted(architecture)})
    record.update(metrics)
    if extra:
        record.update(extra)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print("\n" + json.dumps(record, indent=2, default=str))
    print(f"-> {path}")
    return path
