"""Persist evaluation results.

The metric scripts originally reported only through `tqdm.set_description`, so a
number lived in a progress-bar line on stderr and was gone as soon as the
terminal scrolled. Each script now also writes a JSON record here and echoes a
final line to stdout, so a sweep can be collected afterwards.
"""
import json
import os


def save_result(script, data, metrics, use_untrained_model=False, extra=None):
    """Write results/<data>/<script>[-untrained].json and echo a summary."""
    name = f"{script}-untrained" if use_untrained_model else script
    out_dir = os.path.join("results", data)
    os.makedirs(out_dir, exist_ok=True)
    record = {"script": script, "data": data,
              "untrained": bool(use_untrained_model), **metrics}
    if extra:
        record.update(extra)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print("\n" + json.dumps(record, indent=2, default=str))
    print(f"-> {path}")
    return path
