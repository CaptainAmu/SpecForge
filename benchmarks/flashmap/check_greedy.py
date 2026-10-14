"""Check that a server's greedy (temperature 0) outputs are the target's own
greedy continuations, by teacher forcing: the target scores prompt + output in
one forward pass, and every output token must be its argmax given the tokens
before it.

The outputs are the token ids run_cells.py saved with ``--greedy-ids``
(``greedy-<server>.jsonl``). Exact equality with a target-only server's text is
too strict a test: the speculative server verifies 17 tokens per forward pass,
the target alone decodes one, and their bf16 kernels round differently, so a
near-tied argmax can flip and the two continuations part ways (DSPARK's own
drafts diverge from the target-only text at the same places, whatever the
draft). Here a token that is not the reference argmax must be within ``--tie``
of the best logit.

    python benchmarks/flashmap/check_greedy.py --out <bench OUT dir> --server fm1 \\
        --target Qwen/Qwen3-8B
"""

import argparse
import collections
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--tie", type=float, default=0.5, help="logit gap of a near-tie"
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16)
    model = model.cuda().eval()
    totals = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    with open(os.path.join(args.out, f"greedy-{args.server}.jsonl")) as handle:
        for line in handle:
            row = json.loads(line)
            prompt_ids = tokenizer.encode(row["prompt"])
            output_ids = row["output_ids"]
            ids = torch.tensor([prompt_ids + output_ids], device="cuda")
            with torch.no_grad():
                logits = model(ids).logits[0, len(prompt_ids) - 1 : -1].float()
            out = torch.tensor(output_ids, device="cuda")
            gap = logits.max(-1).values - logits.gather(-1, out[:, None])[:, 0]
            total = totals[row["task"]]
            total[0] += out.numel()
            total[1] += int((gap == 0).sum())
            total[2] += int(((gap > 0) & (gap <= args.tie)).sum())
            total[3] = max(total[3], float(gap.max()))
    for task, (tokens, argmax, ties, worst) in totals.items():
        print(
            f"{task:10s} {args.server}: {tokens} output tokens, {argmax} the target's "
            f"argmax, {ties} within {args.tie} of it, {tokens - argmax - ties} not; "
            f"largest gap {worst:.3f}"
        )


if __name__ == "__main__":
    main()
