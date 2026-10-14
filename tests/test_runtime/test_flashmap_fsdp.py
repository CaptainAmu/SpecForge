# coding=utf-8
"""FlashMap under two-rank FSDP (CPU/gloo), wrapped as the trainer backend
wraps a draft: per-block units only for the draft's ``_no_split_modules``."""

import functools
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

VOCAB, HIDDEN = 64, 32


def _config(flow_map):
    from specforge.modeling.draft.flashmap import FlashMapConfig

    return FlashMapConfig(
        target_hidden_size=HIDDEN,
        vocab_size=VOCAB,
        target_layer_ids=[0, 1],
        block_size=4,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        cond_dim=16,
        time_embed_dim=16,
        flow_map=flow_map,
    )


def _wrapped(stage, sharding):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from specforge.algorithms.flashmap.model import OnlineFlashMapModel
    from specforge.algorithms.flashmap.providers import initialize_map_from_flow
    from specforge.modeling.draft.flashmap import FlashMapDraftModel

    torch.manual_seed(0)
    embed = nn.Embedding(VOCAB, HIDDEN).requires_grad_(False)
    head = nn.Linear(HIDDEN, VOCAB, bias=False).requires_grad_(False)
    draft = FlashMapDraftModel(_config(stage == "map"))
    flow = None
    if stage == "map":
        flow = FlashMapDraftModel(_config(False))
        initialize_map_from_flow(draft, flow)
    model = OnlineFlashMapModel(
        draft,
        head,
        embed,
        stage=stage,
        flow=flow,
        attention_backend="sdpa",
        num_anchors=6,
        objective_chunk_blocks=2,
    )
    # FSDPTrainingBackend.prepare_model's per-block policy.
    block_classes = {
        type(module)
        for module in model.modules()
        if type(module).__name__ in set(draft._no_split_modules or ())
    }
    kwargs = {}
    if block_classes:
        kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=block_classes
        )
    return FSDP(
        model,
        use_orig_params=True,
        sharding_strategy=getattr(ShardingStrategy, sharding),
        ignored_modules=(head, embed),
        device_id=torch.device("cpu"),
        **kwargs,
    )


def _run(rank, world_size, port):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(2)
    inputs = (
        torch.randint(0, VOCAB, (1, 24)),
        torch.randn(1, 24, 2 * HIDDEN),
        torch.ones(1, 24),
    )
    for stage in ("flow", "map"):
        losses = {}
        for sharding in ("NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"):
            model = _wrapped(stage, sharding)
            torch.manual_seed(3)
            loss, _, metrics = model(*inputs)
            metrics["loss_terms"][0].backward()
            losses[sharding] = float(loss)
        assert len(set(losses.values())) == 1, (stage, losses)

    from specforge.algorithms.flashmap.providers import _decorrelate_ranks

    _decorrelate_ranks(42)
    draws = torch.rand(4)
    gathered = [torch.empty_like(draws) for _ in range(world_size)]
    dist.all_gather(gathered, draws)
    assert not torch.equal(gathered[0], gathered[1]), "ranks share their draws"
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


class FlashMapFSDPTest(unittest.TestCase):
    def test_sharded_training_and_per_rank_draws(self):
        mp.spawn(_run, nprocs=2, args=(2, _free_port()))


if __name__ == "__main__":
    unittest.main()
