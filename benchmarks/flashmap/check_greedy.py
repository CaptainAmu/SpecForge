"""Check that a server's greedy (temperature 0) outputs are the target's own
greedy continuations, by teacher forcing: the target scores prompt + output
in one forward pass, and every output token must be its argmax given the
tokens before it.

Exact equality with a target-only server's text is too strict a test: the
speculative server verifies 17 tokens per forward pass, the target alone
decodes one, and their bf16 kernels round differently, so a near-tied argmax
can flip and the two continuations part ways (DSPARK's own drafts diverge from
the target-only text at the same places, whatever the draft). Here a token
that is not the reference argmax must be within ``--tie`` of the best logit.

    python benchmarks/flashmap/check_greedy.py --out <bench OUT dir> --server fm1 \\
        --target Qwen/Qwen3-8B --prompts <bench prompts dir>
"""

import argparse
import glob
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def bench_order(prompts_file: str, count: int, seed: int = 0):
    """The prompts in the order sglang.bench_serving sent them (it shuffles the
    custom dataset after seeding ``random`` with ``--seed``)."""
    rows = [json.loads(line) for line in open(prompts_file) if line.strip()]
    prompts = [row["conversations"][0]["content"] for row in rows]
    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--tie", type=float, default=0.5, help="logit gap of a near-tie"
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    model = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16)
    model = model.cuda().eval()
    for line in open(os.path.join(args.out, f"results-{args.server}.jsonl")):
        row = json.loads(line)
        tag = dict(kv.split("=", 1) for kv in row["tag"].split(","))
        if (
            float(tag["temperature"]) != 0
            or int(tag["concurrency"]) != args.concurrency
        ):
            continue
        count = int(tag["prompts"])
        (path,) = glob.glob(
            os.path.join(args.out, f"prompts-{tag['task']}-{count}.jsonl")
        )
        prompts = bench_order(path, count)
        lens = [len(tokenizer.encode(p)) for p in prompts]
        assert lens == row["input_lens"], "could not recover the bench's prompt order"
        tokens = argmax = ties = worst = 0
        for prompt, text in zip(prompts, row["generated_texts"]):
            prompt_ids = tokenizer.encode(prompt)
            output_ids = tokenizer.encode(text)
            ids = torch.tensor([prompt_ids + output_ids], device="cuda")
            with torch.no_grad():
                logits = model(ids).logits[0, len(prompt_ids) - 1 : -1].float()
            out = torch.tensor(output_ids, device="cuda")
            gap = logits.max(-1).values - logits.gather(-1, out[:, None])[:, 0]
            tokens += out.numel()
            argmax += int((gap == 0).sum())
            ties += int(((gap > 0) & (gap <= args.tie)).sum())
            worst = max(worst, float(gap.max()))
        print(
            f"{tag['task']:10s} {args.server}: {tokens} output tokens, {argmax} the target's "
            f"argmax, {ties} within {args.tie} of it, {tokens - argmax - ties} not; "
            f"largest gap {worst:.3f}"
        )


if __name__ == "__main__":
    main()
