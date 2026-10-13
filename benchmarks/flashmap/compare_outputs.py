"""Compare greedy outputs of bench_sglang.sh servers with the target alone's.

Speculative decoding is lossless at temperature 0: each server should
reproduce the target's own greedy continuation. Batched verification runs the
target's kernels on different shapes than plain decoding, so bf16 rounding can
flip a near-tied argmax; this reports, per cell, how many outputs are
identical and, for the rest, where they first diverge.

    python benchmarks/flashmap/compare_outputs.py <OUT dir> [reference server]
"""

import glob
import json
import os
import sys


def load(out: str):
    cells = {}
    for path in glob.glob(os.path.join(out, "results-*.jsonl")):
        for line in open(path):
            row = json.loads(line)
            tag = dict(kv.split("=", 1) for kv in row["tag"].split(","))
            if "generated_texts" not in row or float(tag["temperature"]) != 0:
                continue
            key = (tag["task"], int(tag["concurrency"]))
            cells.setdefault(key, {})[tag["server"]] = row["generated_texts"]
    return cells


def first_divergence(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main(out: str, reference: str = "target") -> None:
    for (task, concurrency), runs in sorted(load(out).items()):
        if reference not in runs:
            continue
        ref = runs[reference]
        for server, texts in sorted(runs.items()):
            if server == reference:
                continue
            same = sum(a == b for a, b in zip(ref, texts))
            diverge = [
                f"{first_divergence(a, b)}/{len(a)}"
                for a, b in zip(ref, texts)
                if a != b
            ]
            print(
                f"{task:10s} c{concurrency:<3d} {server:16s} identical {same}/{len(ref)}"
                + (
                    f"; first divergence (char/len): {' '.join(diverge)}"
                    if diverge
                    else ""
                )
            )


if __name__ == "__main__":
    main(*sys.argv[1:])
