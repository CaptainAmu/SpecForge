# coding=utf-8
"""Matrix-only weight decay under two-rank sharded FSDP (CPU/gloo): the trainer
backend wraps the draft before the optimizer groups its parameters, and sharded
FSDP flattens their data, so the grouping must use the shapes as built."""

import functools
import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


class _Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)
        self.fc1 = nn.Linear(8, 16)
        self.norm = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 8, bias=False)


def _run(rank, world_size, port, sharding, result):
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    import torch.distributed.fsdp as fsdp

    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig

    # The backend wraps with FSDP on the model's accelerator; these ranks are CPU.
    fsdp.FullyShardedDataParallel = functools.partial(
        fsdp.FullyShardedDataParallel, device_id=torch.device("cpu")
    )
    torch.manual_seed(0)
    backend = FSDPTrainingBackend(
        ParallelConfig(
            world_size=world_size,
            sharding_strategy=sharding,
            param_dtype=torch.float32,
            fsdp_process_group=dist.group.WORLD,
        ),
        optimizer_factory=lambda module: BF16Optimizer(
            module, lr=1e-3, weight_decay=0.1, decay_matrices_only=True
        ),
    )
    draft = _Draft()
    backend.prepare_model(draft, optimizer_target=draft)
    decay, no_decay = backend.optimizer.optimizer.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    # fc1.weight and fc2.weight; not the embedding, the biases or the norm gains.
    result[rank] = (len(decay["params"]), len(no_decay["params"]))
    if sharding != "NO_SHARD":
        # Grouping by the shapes FSDP leaves (flat shards) would decay nothing.
        for param in draft.parameters():
            del param._specforge_shape
        groups = BF16Optimizer(
            draft, lr=1e-3, weight_decay=0.1, decay_matrices_only=True
        ).optimizer.param_groups
        assert len(groups[0]["params"]) == 0, groups
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


class MatrixDecayUnderFSDPTest(unittest.TestCase):
    def test_sharded_wrapping_keeps_the_matrices_decayed(self):
        for sharding in ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding):
                result = mp.Manager().dict()
                mp.spawn(_run, nprocs=2, args=(2, _free_port(), sharding, result))
                self.assertEqual({0: (2, 4), 1: (2, 4)}, dict(result))


if __name__ == "__main__":
    unittest.main()
