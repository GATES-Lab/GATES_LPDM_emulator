"""Single-node multi-GPU (data-parallel) support for the dual-head trainers.

Design, and why it looks like this
----------------------------------
The dual model is small (~270k parameters) and a training step is kernel-launch bound: on a
GH200 the step takes ~27 ms for ANY batch size from 1 to 10. Splitting one batch across GPUs
therefore buys nothing; the speed-up comes from every GPU processing its OWN batch each step:

* ``dataloader.batch_size`` is the PER-GPU batch size. With ``num_gpus = W`` each optimizer
  step averages the gradients of ``W`` consecutive batches, so the EFFECTIVE batch size is
  ``batch_size * W`` and an epoch has ``W`` times fewer optimizer steps. This changes the
  optimisation (like any batch-size change): a multi-GPU run is NOT directly comparable with
  a single-GPU run of the same parameter file, and learning rates may need re-tuning. The
  resolved values are recorded in ``parameters["distributed_resolved"]`` (and so in
  ``training_settings_*.json`` / the W&B config).
* The expensive data load happens ONCE, in the parent process (which is also rank 0 and keeps
  all I/O: W&B, checkpoints, metrics, plots, NetCDF export). The already-batched tensors are
  placed in shared memory (:func:`materialise_batches`) and handed to ``W - 1`` spawned worker
  processes, so the dataset is never duplicated per GPU (a 3-year run peaks at ~135 GB host
  memory; four copies would not fit a node). This also keeps the shared-data experiment
  drivers working unchanged: they still load once and run the arms one after another.
* Every rank runs the SAME trainer code (``build_model_and_context`` + ``run_full_training`` of
  the trainer module). Gradients are averaged with one flat all-reduce per step
  (:func:`average_gradients`) rather than ``DistributedDataParallel``: the trainers freeze /
  unfreeze heads and swap optimizers mid-run (bg freeze, refit), which DDP's static reducer
  handles poorly, and for a model this small a flat all-reduce is just as fast. The ``model``
  object stays the bare module, so checkpoints keep their exact single-GPU format.
* Validation is sharded too; per-batch losses and predictions are gathered back in the
  original order (:func:`gather_sharded_batches`), so every rank sees identical epoch losses
  and therefore takes identical control-flow decisions (early stopping, bg freeze, refit).

With ``num_gpus = 1`` (the default) none of this is active and the trainers behave exactly as
before. Configuration (all optional)::

    "distributed": {"num_gpus": 1}        # int, or "auto" = all GPUs visible to the job
    "dataloader": {"in_memory_batches": false}   # single-GPU opt-in to the in-memory
                                                 # batch loader (always on for num_gpus > 1)
"""

import copy
import dataclasses
import datetime
import importlib
import os
import socket
import sys
import threading
import time

import torch
import torch.distributed as dist


# ---------------------------------------------------------------
# Process-group state helpers (all are no-ops / trivial when not distributed)
# ---------------------------------------------------------------

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    """True on rank 0 and in any non-distributed run. Guards all file / W&B output."""
    return get_rank() == 0


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

def resolve_num_gpus(parameters):
    """Validate ``parameters["distributed"]`` and return the number of GPUs to train on.

    Also records the resolved setup under ``parameters["distributed_resolved"]`` so it is
    saved with the training settings. Call before ``wandb.init`` / the data load so a bad
    config fails early.
    """
    cfg = parameters.get("distributed", None) or {}
    unknown = set(cfg) - {"num_gpus"}
    if unknown:
        raise ValueError(f"distributed: unknown keys {sorted(unknown)}; allowed: ['num_gpus']")
    requested = cfg.get("num_gpus", 1)
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if requested == "auto":
        num_gpus = max(1, visible)
    elif isinstance(requested, int) and not isinstance(requested, bool) and requested >= 1:
        num_gpus = requested
    else:
        raise ValueError(f"distributed.num_gpus must be a positive int or 'auto', got {requested!r}")
    if num_gpus > 1 and num_gpus > visible:
        raise ValueError(f"distributed.num_gpus={num_gpus} but only {visible} GPU(s) are visible "
                         "to this job (check --gres=gpu:N)")
    if visible > num_gpus:
        print(f"FLAG: {visible} GPUs are visible but only {num_gpus} will be used "
              f"(set \"distributed\": {{\"num_gpus\": \"auto\"}} to use them all).")

    batch_size = parameters.get("dataloader", {}).get("batch_size", 5)
    parameters["distributed_resolved"] = {
        "num_gpus": num_gpus,
        "per_gpu_batch_size": batch_size,
        "effective_batch_size": batch_size * num_gpus,
    }
    if num_gpus > 1:
        print(f"Multi-GPU training on {num_gpus} GPUs: per-GPU batch {batch_size} -> EFFECTIVE "
              f"batch {batch_size * num_gpus}, {num_gpus}x fewer optimizer steps per epoch. "
              "Not directly comparable with single-GPU runs of the same parameter file.")
    return num_gpus


def use_in_memory_batches(parameters, num_gpus):
    return num_gpus > 1 or bool(parameters.get("dataloader", {}).get("in_memory_batches", False))


# ---------------------------------------------------------------
# In-memory, shardable batch loader
# ---------------------------------------------------------------

def materialise_batches(loader, description="batches"):
    """Iterate ``loader`` once and stack its batches into shared-memory tensors.

    ``loader`` yields tuples of equally-shaped tensors (the dual loaders trim the data so
    every batch is full). Returns one tensor per tuple position with a leading
    ``n_batches`` dim, holding exactly the batches — same content, same order — that the
    loader yields every epoch (the dual loaders fix their shuffle once, at construction).

    The torch CPU RNG state is restored afterwards (starting a DataLoader iterator draws from
    it), so the model built next gets the same initial weights as without this step.
    """
    start = time.perf_counter()
    n_batches = len(loader)
    stacked = None
    rng_state = torch.get_rng_state()
    for i, batch in enumerate(loader):
        if stacked is None:
            stacked = tuple(torch.empty((n_batches, *t.shape), dtype=t.dtype).share_memory_()
                            for t in batch)
        for dst, src in zip(stacked, batch):
            if src.shape != dst.shape[1:]:
                raise ValueError(f"batch {i} has shape {tuple(src.shape)}, expected "
                                 f"{tuple(dst.shape[1:])}: in-memory batches need equal-sized batches")
            dst[i].copy_(src)
    torch.set_rng_state(rng_state)
    if stacked is None:
        raise ValueError(f"cannot materialise {description}: the loader is empty")
    gib = sum(t.numel() * t.element_size() for t in stacked) / 2**30
    print(f"Materialised {n_batches} {description} in shared memory "
          f"({gib:.1f} GiB, {time.perf_counter() - start:.0f} s)")
    return stacked


class InMemoryBatchLoader:
    """Iterate pre-built batches held in (shared) memory, optionally sharded across ranks.

    Rank ``r`` of ``W`` yields batches ``r, r + W, r + 2W, ...``: one optimizer step consumes
    ``W`` consecutive batches of the single-GPU order, one per GPU.

    * ``pad=False`` (training): the trailing ``n_batches % W`` batches are dropped so every
      rank takes the same number of steps.
    * ``pad=True`` (validation): ranks that run out of batches repeat the last batch so all
      ranks stay in step; :func:`gather_sharded_batches` discards the repeats, so every
      batch is evaluated exactly once.
    """

    def __init__(self, tensors, rank=0, world_size=1, pad=False):
        self.tensors = tuple(tensors)
        self.n_total = int(self.tensors[0].shape[0])
        if self.n_total < world_size:
            raise ValueError(f"only {self.n_total} batches for {world_size} GPUs")
        if pad:
            per_rank = -(-self.n_total // world_size)  # ceil
            self.indices = [min(rank + j * world_size, self.n_total - 1) for j in range(per_rank)]
        else:
            per_rank = self.n_total // world_size
            self.indices = [rank + j * world_size for j in range(per_rank)]
        self.n_dropped = 0 if pad else self.n_total - per_rank * world_size

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield tuple(t[idx] for t in self.tensors)


def gather_sharded_batches(local, n_total):
    """Reassemble per-batch results of a ``pad=True`` sharded loader in the original order.

    ``local`` is this rank's ``(n_local_batches, ...)`` tensor (same shape on all ranks).
    Returns the ``(n_total, ...)`` tensor on EVERY rank (identity when not distributed).
    """
    if not is_distributed():
        return local
    world_size = get_world_size()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local.contiguous())
    # gathered[r][j] is global batch r + j * W  ->  stack on dim 1 and flatten to interleave
    interleaved = torch.stack(gathered, dim=1).reshape(-1, *local.shape[1:])
    return interleaved[:n_total]


# ---------------------------------------------------------------
# Gradient / metric synchronisation
# ---------------------------------------------------------------

def average_gradients(model):
    """Average the gradients across ranks with one flat all-reduce (no-op if not distributed).

    Parameters without a gradient (frozen heads, detached loss terms) are skipped; all ranks
    follow the same schedule, so the set of parameters with gradients is identical everywhere.
    """
    if not is_distributed():
        return
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(get_world_size())
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset:offset + n].view_as(g))
        offset += n


def all_reduce_mean(values, device):
    """Mean over ranks of a sequence of python floats (returned unchanged if not distributed)."""
    if not is_distributed():
        return tuple(values)
    t = torch.tensor(list(values), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return tuple((t / get_world_size()).tolist())


def sync_model_from_main(model):
    """Broadcast rank 0's parameters and buffers so every rank starts from identical weights."""
    if not is_distributed():
        return
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(t, src=0)


# ---------------------------------------------------------------
# Worker processes
# ---------------------------------------------------------------

# Rank 0 does CPU-only work between collectives (metrics, plots, W&B artifact uploads, the
# NetCDF export) while the workers wait, so the collective timeout must be generous.
_COLLECTIVE_TIMEOUT = datetime.timedelta(hours=2)


def _init_process_group(rank, world_size, init_method):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method=init_method, rank=rank,
                            world_size=world_size, timeout=_COLLECTIVE_TIMEOUT,
                            device_id=torch.device("cuda", rank))


def _shutdown_process_group():
    # Tearing down NCCL is collective: every rank must arrive here before any rank waits for
    # another process to exit (rank 0 joining the workers first would deadlock).
    dist.barrier()
    dist.destroy_process_group()


def _worker(local_index, world_size, init_method, trainer_module_name, parameters,
            training_ctx, paths_ctx, train_tensors, test_tensors, run_kwargs):
    """Entry point of ranks 1..W-1: same model build + training loop as rank 0, no output."""
    from gates.training.training_helperfuns import set_reproducibility, enable_deterministic_algorithms
    from gates.training.training_dual import initialise_dual_losses

    rank = local_index + 1
    sys.stdout = open(os.devnull, "w")  # rank 0 owns the log; errors still reach stderr
    torch.cuda.set_device(rank)

    seed = parameters.get("seed", 34)
    set_reproducibility(seed)
    enable_deterministic_algorithms(parameters)
    _init_process_group(rank, world_size, init_method)

    parameters = copy.deepcopy(parameters)
    parameters["use_wandb"] = False
    training_ctx = dataclasses.replace(training_ctx, parameters=parameters, use_wandb=False,
                                       device=torch.device("cuda", rank))

    trainer = importlib.import_module(trainer_module_name)
    model, model_ctx = trainer.build_model_and_context(parameters, training_ctx, paths_ctx)
    sync_model_from_main(model)
    torch.manual_seed(seed + rank)  # decorrelate dropout across ranks (weights are synced above)

    trainer.run_full_training(
        model, model_ctx, training_ctx, paths_ctx,
        InMemoryBatchLoader(train_tensors, rank, world_size),
        InMemoryBatchLoader(test_tensors, rank, world_size, pad=True),
        None, initialise_dual_losses(), **run_kwargs)
    _shutdown_process_group()


class DistributedRun:
    """Handle on the spawned workers; rank 0 (the parent) calls :meth:`finish` when done."""

    def __init__(self, context):
        self.context = context
        self._stop = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _watch(self):
        # A dead worker would leave rank 0 blocked in a collective until the (long) timeout;
        # fail fast and loudly instead.
        while not self._stop.wait(5.0):
            for i, proc in enumerate(self.context.processes):
                if proc.exitcode not in (None, 0):
                    print(f"FATAL: multi-GPU worker rank {i + 1} died with exit code "
                          f"{proc.exitcode}; aborting the run (see its traceback above).",
                          file=sys.stderr, flush=True)
                    os._exit(1)

    def finish(self):
        """Tear the process group down together with the workers, then wait for them to exit."""
        _shutdown_process_group()
        self._stop.set()
        self.context.join()

    def abort(self):
        """Kill the workers after a failure on rank 0."""
        self._stop.set()
        for proc in self.context.processes:
            if proc.is_alive():
                proc.terminate()
        if is_distributed():
            dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_workers(trainer_module_name, num_gpus, parameters, training_ctx, paths_ctx,
                  train_tensors, test_tensors, run_kwargs):
    """Spawn ranks 1..num_gpus-1 and join the process group as rank 0.

    ``trainer_module_name`` must be importable and provide ``build_model_and_context`` and
    ``run_full_training``; the workers call them with the same arguments as rank 0 (W&B off,
    ``test_fp_dataset=None``). Returns ``(DistributedRun, train_loader, test_loader)`` with
    rank 0's shard loaders.
    """
    import torch.multiprocessing as mp

    init_method = f"tcp://127.0.0.1:{_free_port()}"
    context = mp.spawn(
        _worker, nprocs=num_gpus - 1, join=False,
        args=(num_gpus, init_method, trainer_module_name, parameters, training_ctx, paths_ctx,
              train_tensors, test_tensors, run_kwargs))
    run = DistributedRun(context)
    try:
        _init_process_group(0, num_gpus, init_method)
    except BaseException:
        run.abort()
        raise
    train_loader = InMemoryBatchLoader(train_tensors, 0, num_gpus)
    test_loader = InMemoryBatchLoader(test_tensors, 0, num_gpus, pad=True)
    if train_loader.n_dropped:
        print(f"Multi-GPU: dropping the last {train_loader.n_dropped} of {train_loader.n_total} "
              f"training batches so every GPU takes {len(train_loader)} steps per epoch.")
    return run, train_loader, test_loader


def setup_training_loaders(trainer_module_name, num_gpus, parameters, training_ctx, paths_ctx,
                           train_loader, test_loader, run_kwargs):
    """Swap in the in-memory batch loaders when configured and start the multi-GPU workers.

    The one call a trainer's ``train_and_save_model`` makes between building its DataLoaders
    and building its model. Returns ``(dist_run, train_loader, test_loader)``:

    * single GPU, ``dataloader.in_memory_batches`` off (default): the loaders are returned
      untouched and ``dist_run`` is None — the run is exactly the pre-multi-GPU code path;
    * single GPU, ``in_memory_batches`` on: same batches in the same order, served from memory;
    * ``num_gpus > 1``: workers spawned (see :func:`start_workers`), ``dist_run`` must be
      ``finish()``-ed after training (or ``abort()``-ed on failure).
    """
    if not use_in_memory_batches(parameters, num_gpus):
        return None, train_loader, test_loader
    train_tensors = materialise_batches(train_loader, "training batches")
    test_tensors = materialise_batches(test_loader, "test batches")
    del train_loader, test_loader
    if num_gpus == 1:
        return None, InMemoryBatchLoader(train_tensors), InMemoryBatchLoader(test_tensors)
    return start_workers(trainer_module_name, num_gpus, parameters, training_ctx, paths_ctx,
                         train_tensors, test_tensors, run_kwargs)
