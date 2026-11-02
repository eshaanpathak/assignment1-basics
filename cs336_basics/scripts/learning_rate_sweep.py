"""
Modal launcher for a learning-rate sweep. Infra only: it fans out one train_modal.train_remote call per learning
rate, all in parallel (one GPU container each). The training logic lives in train.py.

Fill in LEARNING_RATES below, then run:
    uv run modal run --detach cs336_basics/scripts/learning_rate_sweep.py --prefix ts_lr_sweep
    uv run modal run --detach cs336_basics/scripts/learning_rate_sweep.py --prefix ts_lr_sweep --extra-args "--optim.max_iters 5000 --optim.cosine_cycle_iters 5000"

Each run is named <prefix>_lr<value> (e.g. ts_lr_sweep_lr0.0003), logs to wandb under that name, and writes checkpoints to
/vol/assignment1/checkpoints/<run-name>/. Compare them in the wandb UI on val/loss.

min_lr is set to lr * MIN_LR_RATIO for every run, so the schedule shape stays the same across the sweep
(override with --min-lr-ratio). Pass --extra-args for anything else; it applies to every run.
"""

from cs336_basics.training.train_modal import app, train_remote

LEARNING_RATES: list[float] = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

# train.py's default is lr=3e-4, min_lr=3e-5, i.e. a ratio of 0.1.
MIN_LR_RATIO = 0.1


def _run_name(prefix: str, lr: float) -> str:
    return f"{prefix}_lr{lr:g}"


def _train_args(lr: float, min_lr_ratio: float, extra_args: str, wandb: bool) -> str:
    parts = [f"--optim.lr {lr:g}", f"--optim.min_lr {lr * min_lr_ratio:g}"]
    if wandb:
        parts.append("--wandb.enabled")
    if extra_args:
        parts.append(extra_args)
    return " ".join(parts)


@app.local_entrypoint()
def sweep(
    prefix: str = "ts_lr",
    min_lr_ratio: float = MIN_LR_RATIO,
    extra_args: str = "",
    wandb: bool = True,
):
    if not LEARNING_RATES:
        raise SystemExit("LEARNING_RATES is empty: fill it in at the top of learning_rate_sweep.py")
    if len(set(LEARNING_RATES)) != len(LEARNING_RATES):
        raise SystemExit(f"duplicate learning rates in LEARNING_RATES: {LEARNING_RATES}")

    # spawn() returns immediately, so every run starts at once rather than one after another.
    calls = {}
    for lr in LEARNING_RATES:
        name = _run_name(prefix, lr)
        args = _train_args(lr, min_lr_ratio, extra_args, wandb)
        calls[name] = train_remote.spawn(name, args)
        print(f"launched {name}: {args}  ({calls[name].object_id})")

    # Wait for all runs. A diverged or crashed run is reported but does not stop the others.
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
