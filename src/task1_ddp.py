"""DistributedDataParallel support for the Task 1 notebooks.

The notebooks stay single-process notebooks. This module is inert unless the process was
started by `torchrun`, which is the only way to get one process per GPU: `torch.distributed`
needs a process group, and `mp.spawn` cannot pickle a function defined in a notebook cell.
So `initialise()` reads the environment `torchrun` sets and returns a world of size one when
it is absent -- an interactive run keeps exactly the behaviour it has today, and the DDP path
is reached only by launching the exported script under `torchrun`.

Why the global batch is held fixed rather than scaled by device count:

  `BATCH_SIZE` is hashed into `RUN_FINGERPRINT`, and a checkpoint whose fingerprint differs is
  refused on load. Scaling the global batch with the number of visible GPUs would therefore
  make a checkpoint's identity depend on the hardware that happened to train it, and the
  parallel run's whole premise -- eight jobs trained on whatever machines are free, combined
  later -- would stop holding. So `BATCH_SIZE` remains the *global* batch and each rank takes
  `BATCH_SIZE // world_size` of it.

That choice is also what makes the result trustworthy. With the global batch fixed, DDP's
gradient all-reduce averages the per-rank gradients, which is arithmetically the same
gradient a single process would compute over the whole batch. Combined with
`SyncBatchNorm` -- which computes normalisation statistics across ranks instead of over each
rank's shard -- an N-GPU run is equivalent to a 1-GPU run rather than merely similar to it.
`nn.DataParallel` cannot make that claim: it leaves BatchNorm to see batch/N per device.

One honest exception: the augmentation RNG. Each rank augments its own shard, so the sequence
of random draws differs from a single process working through the same rows in one stream.
The augmentation *policy* and its distribution are unchanged, and every rank is seeded
deterministically from the run seed, so the run is reproducible -- but it is not bit-identical
to a single-GPU run, and nothing can make it so while the shards are augmented in parallel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class World:
    """The process group this run belongs to, or a world of one when not distributed."""

    rank: int = 0
    local_rank: int = 0
    size: int = 1
    backend: str = ""

    @property
    def enabled(self):
        """True only when there is really more than one process to synchronise."""
        return self.size > 1

    @property
    def is_main(self):
        """Rank 0. The only rank that writes files or prints progress."""
        return self.rank == 0

    def device(self):
        """The device this rank owns.

        Each rank binds to exactly one GPU. Sharing a GPU between ranks is possible but
        pointless: it splits the card's memory without adding throughput.
        """
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")


def launched_by_torchrun():
    """Whether the launcher populated the rendezvous variables."""
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def initialise(timeout_minutes=30):
    """Join the process group, or report a world of one.

    Returns a `World` either way, so callers have no branch to write: an interactive notebook
    gets `size == 1` and every helper below degrades to a no-op.
    """
    if not launched_by_torchrun():
        return World()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])

    if size == 1:
        # torchrun with one process. Skipping the process group avoids paying for
        # collectives that would each have a single participant.
        return World(rank=rank, local_rank=local_rank, size=1)

    # NCCL on GPUs, gloo elsewhere. NCCL is the only backend with real GPU collectives; gloo
    # exists here so the code path can be exercised on a CPU-only machine.
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        # Bind before init_process_group: NCCL selects the device from the current context,
        # and every rank landing on cuda:0 is the classic way to hang a run at the first
        # collective.
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(backend=backend,
                                timeout=timedelta(minutes=timeout_minutes))

    return World(rank=rank, local_rank=local_rank, size=size, backend=backend)


def shutdown():
    """Leave the process group. Safe to call when there is none."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(world):
    """Wait for every rank. A no-op in a world of one."""
    if world.enabled and dist.is_initialized():
        dist.barrier()


def per_rank_batch(global_batch, world):
    """Split the global batch across ranks, refusing a split that would change the recipe.

    The global batch is the quantity the fingerprint pins and the optimiser was tuned for, so
    it has to survive the split exactly. A world size that does not divide it would leave
    ranks with unequal batches and an effective batch that is no longer `global_batch`, which
    is a silently different experiment -- so it is an error rather than a rounding.
    """
    if not world.enabled:
        return global_batch
    if global_batch % world.size:
        raise ValueError(
            f"BATCH_SIZE {global_batch} is not divisible by the world size {world.size}, so "
            f"the global batch could not be preserved exactly.\n"
            f"Launch with a process count that divides it -- {sorted(n for n in range(1, global_batch + 1) if global_batch % n == 0 and n <= 8)} "
            f"are the usable sizes up to 8 -- or change BATCH_SIZE, which changes "
            f"RUN_FINGERPRINT and invalidates existing checkpoints."
        )
    return global_batch // world.size


def wrap_model(model, world, device, sync_batchnorm=True):
    """Wrap a model for distributed training, preserving what BatchNorm would have seen.

    `convert_sync_batchnorm` is the reason this is worth doing over DataParallel. Ordinary
    BatchNorm in a distributed run normalises over one rank's shard, so the statistics depend
    on how many GPUs happened to be available. SyncBatchNorm all-reduces the batch statistics,
    which restores exactly the normalisation a single process would have computed over the
    whole global batch.
    """
    if not world.enabled:
        return model

    # CUDA only: SyncBatchNorm's collectives are implemented for NCCL, and DDP rejects a
    # converted module on CPU outright ("SyncBatchNorm layers only work with GPU modules").
    # A gloo run therefore keeps ordinary BatchNorm and its per-shard statistics -- which is
    # acceptable only because the CPU path exists to exercise the code, never to train with.
    if sync_batchnorm and device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    device_ids = [world.local_rank] if device.type == "cuda" else None
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=world.local_rank if device.type == "cuda" else None,
        # Every parameter reached by `forward` receives a gradient in the training loops that
        # go through DDP, so the extra graph traversal that `True` costs would buy nothing.
        # The decoupled stage is the one place with frozen parameters, and it calls `embed`
        # and `fc` directly rather than `forward`, so it is trained unwrapped instead.
        find_unused_parameters=False,
        broadcast_buffers=True,
    )


def unwrap(model):
    """The underlying module, past any DDP, DataParallel or torch.compile wrapper.

    Written as a loop because the wrappers compose: a compiled model inside DDP presents as
    `module._orig_mod`, and code that reaches for `.fc` or `.embed` needs the real module
    whichever combination is in play.
    """
    seen = 0
    while seen < 8:
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
        else:
            break
        seen += 1
    return model


def shard(order, world):
    """This rank's slice of one epoch's row order.

    Strided rather than contiguous so each rank sees the same class mixture; a contiguous
    split of an ordering that is only shuffled once would hand different ranks different
    distributions. Every rank must receive the same number of batches or the collectives
    deadlock at the end of the epoch, so the tail that does not divide evenly is dropped --
    at most `world_size - 1` rows out of tens of thousands, and a different few each epoch
    because the order is reshuffled.
    """
    if not world.enabled:
        return order
    usable = (len(order) // world.size) * world.size
    return order[world.rank:usable:world.size]


def all_reduce_mean(value, world, device=None):
    """Average a scalar metric across ranks.

    Validation loss and accuracy are computed on each rank's shard, so reporting rank 0's
    number alone would report a fraction of the validation set as if it were the whole.
    """
    if not world.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64,
                          device=device or torch.device("cpu"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / world.size)


def all_reduce_sum(value, world, device=None):
    """Total a count across ranks, for metrics that must be pooled rather than averaged."""
    if not world.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64,
                          device=device or torch.device("cpu"))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def gather_concat(tensor, world):
    """Collect a per-rank tensor onto every rank, concatenated along dim 0.

    Used where the notebook needs the whole validation set's predictions rather than a
    reduced statistic -- the confusion matrix and the per-class F1 tables, which cannot be
    reconstructed from an average.
    """
    if not world.enabled:
        return tensor
    sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device)
             for _ in range(world.size)]
    dist.all_gather(sizes, torch.tensor([tensor.shape[0]], dtype=torch.long,
                                        device=tensor.device))
    largest = int(max(int(size.item()) for size in sizes))

    # all_gather needs equal shapes, so short ranks are padded and the padding is cut after.
    padded = tensor
    if tensor.shape[0] < largest:
        padding = torch.zeros((largest - tensor.shape[0], *tensor.shape[1:]),
                              dtype=tensor.dtype, device=tensor.device)
        padded = torch.cat([tensor, padding], dim=0)

    parts = [torch.zeros_like(padded) for _ in range(world.size)]
    dist.all_gather(parts, padded)
    return torch.cat([part[:int(size.item())] for part, size in zip(parts, sizes)], dim=0)


def report(world, device):
    """One line per rank at start-up, printed by rank 0 only."""
    if not world.is_main:
        return
    if world.enabled:
        print(f"DDP: {world.size} processes over {torch.cuda.device_count()} visible GPU(s) "
              f"| backend {world.backend} | SyncBatchNorm on")
    elif torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs visible but this is a single process, so "
              f"only {device} is used.\n"
              "  Launch the exported script with torchrun to use them all -- see the DDP "
              "cell in the notebook.")
