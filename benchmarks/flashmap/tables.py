"""Markdown tables of bench_sglang.sh results: one table per temperature and
concurrency, a row per server (acceptance length, output tok/s and speedup over
the target alone) and a column per task.

    python benchmarks/flashmap/tables.py <OUT dir> [server=label ...]

Servers are listed in the given order with the given labels (default: all, by
name).
"""

import glob
import json
import os
import sys


def load(out: str):
    cells = {}
    for path in glob.glob(os.path.join(out, "results-*.jsonl")):
        if path.endswith("-warmup.jsonl"):
            continue
        for line in open(path):
            row = json.loads(line)
            tag = dict(kv.split("=", 1) for kv in row["tag"].split(","))
            key = (float(tag["temperature"]), int(tag["concurrency"]))
            cells.setdefault(key, {}).setdefault(tag["task"], {})[tag["server"]] = row
    return cells


def main(out: str, *labels: str) -> None:
    cells = load(out)
    servers = [label.split("=", 1) for label in labels]
    for (temperature, concurrency), tasks in sorted(cells.items()):
        names = servers or [
            (s, s) for s in sorted({s for runs in tasks.values() for s in runs})
        ]
        task_names = sorted(tasks)
        print(f"\nT={temperature:g}, concurrency {concurrency}\n")
        print("| method | " + " | ".join(task_names) + " |")
        print("|---|" + "---|" * len(task_names))
        for server, label in names:
            row = []
            for task in task_names:
                run = tasks[task].get(server)
                base = tasks[task].get("target")
                if run is None:
                    row.append("-")
                    continue
                tps = run["output_throughput"]
                text = f"{tps:.0f} tok/s"
                if base and server != "target":
                    text += f" ({tps / base['output_throughput']:.2f}x)"
                if run.get("accept_length"):
                    text = f"{run['accept_length']:.2f} / " + text
                row.append(text)
            print(f"| {label} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:])
