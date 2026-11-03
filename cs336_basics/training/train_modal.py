"""
Modal launcher for cs336_basics/training/train.py. Infra only: the training logic lives in train.py.

Same volume as the tokenizer launchers (one volume, one top-level directory per assignment):
    /assignment1/data/         tokenized .bin files (ts_train.bin, ts_valid.bin, ...)
    /assignment1/checkpoints/  checkpoints written by train.py

If the .bin files are not on the volume yet (they are normally produced there by encode_dataset_modal.py):
    uv run modal volume put cs336-data data/ts_train.bin /assignment1/data/ts_train.bin
    uv run modal volume put cs336-data data/ts_valid.bin /assignment1/data/ts_valid.bin

Required once (the function mounts it): create a Modal secret holding your wandb key.
Log to wandb by passing --wandb.enabled in --train-args.
    uv run modal secret create wandb-secret WANDB_API_KEY=<your key>

Run. --train-args is passed to train.py verbatim (same flags as parse_config in config.py). These defaults are
prepended, so anything you pass overrides them:
    --data.train_path /vol/assignment1/data/ts_train.bin
    --data.valid_path /vol/assignment1/data/ts_valid.bin
    --train.checkpoint_dir /vol/assignment1/checkpoints/<run-name>
    --train.device cuda
    --train.run_name <run-name>

    uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_default
    uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_lr1e-3 --train-args "--optim.lr 1e-3 --optim.min_lr 1e-4"
    uv run modal run --detach cs336_basics/training/train_modal.py --run-name owt_default --train-args "--data.train_path /vol/assignment1/data/owt_train.bin --data.valid_path /vol/assignment1/data/owt_valid.bin --model.vocab_size 32000"

Run commands for ablations:
uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_no_rmsnorm --train-args "--no-model.use_rmsnorm"
uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_no_rmsnorm_lr1e4 --train-args "--no-model.use_rmsnorm --optim.lr 1e-4 --optim.min_lr 1e-5"
uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_post_norm --train-args "--model.post_norm"
uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_post_norm_lr1e4 --train-args "--model.post_norm --optim.lr 1e-4 --optim.min_lr 1e-5"
uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_no_rope --train-args "--no-model.use_rope"
uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_silu_ffn --train-args "--model.ffn silu"

Resume (checkpoint path is on the volume):
    uv run modal run --detach cs336_basics/training/train_modal.py --run-name ts_default --train-args "--train.resume_from /vol/assignment1/checkpoints/ts_default/checkpoint_5000.pt"

Download checkpoints:
    uv run modal volume get cs336-data /assignment1/checkpoints/ts_default checkpoints/
"""

import shlex
import sys

import modal

app = modal.App("cs336-train")

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
VOLUME_PATH = "/vol"
ASSIGNMENT = "assignment1"
DATA_DIR = f"{VOLUME_PATH}/{ASSIGNMENT}/data"
CHECKPOINT_ROOT = f"{VOLUME_PATH}/{ASSIGNMENT}/checkpoints"

GPU = "B200"
TIMEOUT_HOURS = 6

# Mirrors the torch pin in pyproject.toml.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.11.0", "numpy", "einops", "einx", "jaxtyping", "wandb", "tqdm", "regex")
    .add_local_python_source("cs336_basics")
)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    gpu=GPU,
    timeout=TIMEOUT_HOURS * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_remote(run_name: str, train_args: str):
    from cs336_basics.training.train import main

    volume.reload()
    defaults = [
        "--data.train_path", f"{DATA_DIR}/ts_train.bin",
        "--data.valid_path", f"{DATA_DIR}/ts_valid.bin",
        "--train.checkpoint_dir", f"{CHECKPOINT_ROOT}/{run_name}",
        "--train.device", "cuda",
        "--train.run_name", run_name,
    ]
    # argparse keeps the last occurrence of a flag, so user args override the defaults.
    sys.argv = ["train.py", *defaults, *shlex.split(train_args)]
    try:
        main()
    finally:
        # Persist checkpoints even if training crashes or is stopped partway.
        volume.commit()


@app.local_entrypoint()
def main(run_name: str, train_args: str = ""):
    train_remote.remote(run_name, train_args)
