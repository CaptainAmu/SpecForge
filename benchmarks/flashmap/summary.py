"""Summarize bench_sglang.sh results: per cell, each server's acceptance length,
output throughput, mean TPOT and end-to-end latency, and its speedup over the
target alone.

    python benchmarks/flashmap/summary.py <OUT dir>
"""

import glob
import json
import os
import sys


def main(out: str) -> None:
    cells = {}
    for path in sorted(glob.glob(os.path.join(out, "results-*.jsonl"))):
        if path.endswith("-warmup.jsonl"):
            continue
        for line in open(path):
            row = json.loads(line)
            tag = dict(kv.split("=", 1) for kv in row["tag"].split(","))
            key = (tag["task"], int(tag["concurrency"]), tag["temperature"])
            cells.setdefault(key, {})[tag["server"]] = row
    header = (
        f"{'server':16s} {'accept':>7s} {'tok/s':>8s} {'speedup':>8s} "
        f"{'TPOT ms':>8s} {'E2E s':>7s} {'out tok':>8s}"
    )
    for task, concurrency, temperature in sorted(cells):
        runs = cells[(task, concurrency, temperature)]
        print(f"\n{task}, concurrency {concurrency}, T={temperature}")
        print(header)
        base = runs.get("target", {}).get("output_throughput")
        for server, row in runs.items():
            tps = row["output_throughput"]
            accept = row.get("accept_length") or float("nan")
            speedup = f"{tps / base:7.2f}x" if base else f"{'-':>8s}"
            print(
                f"{server:16s} {accept:7.2f} {tps:8.1f} {speedup} "
                f"{row.get('mean_tpot_ms', float('nan')):8.2f} "
                f"{row.get('mean_e2e_latency_ms', float('nan')) / 1000:7.2f} "
                f"{row['total_output_tokens']:8d}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
