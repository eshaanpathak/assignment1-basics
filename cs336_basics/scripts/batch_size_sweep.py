"""
Modal launcher for a batch-size sweep. Infra only: it fans out one train_modal.train_remote call per batch size,
all in parallel (one GPU container each). The training logic lives in train.py.

Fill in BATCH_SIZES below, then run:
    uv run modal run --detach cs336_basics/scripts/batch_size_sweep.py --prefix ts_bs_sweep
    uv run modal run --detach cs336_basics/scripts/batch_size_sweep.py --prefix ts_bs_sweep --lr 1e-3

Every run processes the same number of tokens (BASE_BATCH_SIZE * BASE_MAX_ITERS * context_length, i.e. 32 * 40000 * 256
= 327.68M with the defaults in config.py), so max_iters is scaled as BASE_BATCH_SIZE * BASE_MAX_ITERS / batch_size.
The other step-count knobs are scaled with it so the schedule shape and the number of evals/checkpoints stay the same:
  - cosine_cycle_iters = max_iters (one full cosine cycle per run)
  - warmup_iters is scaled by tokens (floored at MIN_WARMUP_ITERS)
  - eval_interval and checkpoint_interval keep the same number of evals/checkpoints per run

Learning rate is held fixed across the sweep (--lr, min_lr = lr * MIN_LR_RATIO). The best LR usually shifts with batch
size, so treat this as a batch-size comparison at one LR and re-sweep the LR afterwards.

Each run is named <prefix>_bs<value> (e.g. ts_bs_sweep_bs256), logs to wandb under that name, and writes checkpoints
to /vol/assignment1/checkpoints/<run-name>/. Compare them in the wandb UI on val/loss, against tokens seen.
Pass --extra-args for anything else; it applies to every run. Don't override max_iters/context_length there, or the
runs will no longer see equal tokens.
"""

from cs336_basics.training.train_modal import app, train_remote

BATCH_SIZES: list[int] = [16, 64, 128, 256, 512, 800, 1024]

# The reference run whose token budget every other run matches: train.py's defaults (batch 32, 40k iters).
BASE_BATCH_SIZE = 32
BASE_MAX_ITERS = 40_000
BASE_WARMUP_ITERS = 200
BASE_EVAL_INTERVAL = 500
BASE_CHECKPOINT_INTERVAL = 1000

# Warmup is scaled by tokens, but never shorter than this (large batches have few steps).
MIN_WARMUP_ITERS = 20

# train.py's default is lr=3e-4, min_lr=3e-5, i.e. a ratio of 0.1.
LEARNING_RATE = 3e-4
MIN_LR_RATIO = 0.1


def _run_name(prefix: str, batch_size: int) -> str:
    return f"{prefix}_bs{batch_size}"


def _scaled_iters(batch_size: int) -> dict[str, int]:
    """Step counts for a run at batch_size that sees the same tokens as the base run."""
    total_sequences = BASE_BATCH_SIZE * BASE_MAX_ITERS
    if total_sequences % batch_size:
        raise SystemExit(
            f"batch size {batch_size} does not divide {total_sequences} sequences, "
            "so the token budget could not be matched exactly"
        )
    max_iters = total_sequences // batch_size
    scale = max_iters / BASE_MAX_ITERS
    return {
        "max_iters": max_iters,
        "cosine_cycle_iters": max_iters,
        "warmup_iters": max(MIN_WARMUP_ITERS, round(BASE_WARMUP_ITERS * BASE_BATCH_SIZE / batch_size)),
        "eval_interval": max(1, round(BASE_EVAL_INTERVAL * scale)),
        "checkpoint_interval": max(1, round(BASE_CHECKPOINT_INTERVAL * scale)),
    }


def _train_args(batch_size: int, lr: float, min_lr_ratio: float, extra_args: str, wandb: bool) -> str:
    it = _scaled_iters(batch_size)
    parts = [
        f"--optim.batch_size {batch_size}",
        f"--optim.lr {lr:g}",
        f"--optim.min_lr {lr * min_lr_ratio:g}",
        f"--optim.max_iters {it['max_iters']}",
        f"--optim.cosine_cycle_iters {it['cosine_cycle_iters']}",
        f"--optim.warmup_iters {it['warmup_iters']}",
        f"--train.eval_interval {it['eval_interval']}",
        f"--train.checkpoint_interval {it['checkpoint_interval']}",
    ]
    if wandb:
        parts.append("--wandb.enabled")
    if extra_args:
        parts.append(extra_args)
    return " ".join(parts)


@app.local_entrypoint()
def sweep(
    prefix: str = "ts_bs",
    lr: float = LEARNING_RATE,
    min_lr_ratio: float = MIN_LR_RATIO,
    extra_args: str = "",
    wandb: bool = True,
):
    if not BATCH_SIZES:
        raise SystemExit("BATCH_SIZES is empty: fill it in at the top of batch_size_sweep.py")
    if len(set(BATCH_SIZES)) != len(BATCH_SIZES):
        raise SystemExit(f"duplicate batch sizes in BATCH_SIZES: {BATCH_SIZES}")

    # spawn() returns immediately, so every run starts at once rather than one after another.
    calls = {}
    for batch_size in BATCH_SIZES:
        name = _run_name(prefix, batch_size)
        args = _train_args(batch_size, lr, min_lr_ratio, extra_args, wandb)
        calls[name] = train_remote.spawn(name, args)
        print(f"launched {name}: {args}  ({calls[name].object_id})")

    # Wait for all runs. An OOM or crashed run is reported but does not stop the others.
    failed = []
    for name, call in calls.items():
        try:
            call.get()
            print(f"finished {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAILED {name}: {e!r}")

    print(f"\n{len(calls) - len(failed)}/{len(calls)} runs finished")
    if failed:
        print(f"failed: {failed}")
