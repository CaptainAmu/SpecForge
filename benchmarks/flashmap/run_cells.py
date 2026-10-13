"""Run sglang.benchmark.serving on several cells in one process (each separate
``python -m sglang.bench_serving`` pays sglang's import again, which takes
minutes on a network filesystem). Used by bench_sglang.sh.

    python benchmarks/flashmap/run_cells.py --name fm1 --target <model> \\
        --prompts <dir> --out <dir> --cells "gsm8k:1:16:0 mt-bench:4:32:0"
"""

import argparse
import os
import sys
import traceback

from sglang.benchmark.serving import cli_main


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cells", required=True, help="task:concurrency:prompts:temp ..."
    )
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-details", action="store_true")
    args = parser.parse_args()

    def run(name, task, concurrency, count, temperature):
        prompts = os.path.join(args.out, f"prompts-{task}-{count}.jsonl")
        with open(os.path.join(args.prompts, f"{task}.jsonl")) as src:
            lines = src.readlines()[:count]
        with open(prompts, "w") as dst:
            dst.writelines(lines)
        tag = (
            f"server={name},task={task},concurrency={concurrency},"
            f"prompts={count},temperature={temperature}"
        )
        sys.argv = [
            "bench_serving",
            "--backend", "sglang",
            "--host", "127.0.0.1",
            "--port", str(args.port),
            "--model", args.target,
            "--tokenizer", args.target,
            "--dataset-name", "custom",
            "--dataset-path", prompts,
            "--num-prompts", str(count),
            "--sharegpt-output-len", str(args.max_tokens),
            "--disable-ignore-eos",
            "--temperature", str(temperature),
            "--max-concurrency", str(concurrency),
            "--request-rate", "inf",
            "--flush-cache",
            "--seed", "0",
            "--disable-tqdm",
            "--tag", tag,
            "--output-file", os.path.join(args.out, f"results-{name}.jsonl"),
        ] + (["--output-details"] if args.output_details else [])  # fmt: skip
        print(f"== {tag}", flush=True)
        try:
            cli_main()
        except Exception:  # one failed cell does not stop the others
            traceback.print_exc()

    run(f"{args.name}-warmup", "gsm8k", 4, 8, 0)
    for cell in args.cells.split():
        task, concurrency, count, temperature = cell.split(":")
        run(args.name, task, int(concurrency), int(count), float(temperature))


if __name__ == "__main__":
    main()
