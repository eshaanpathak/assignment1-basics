import os
import time

import numpy as np
import torch

from cs336_basics.model.transformer import Transformer
from cs336_basics.training.config import parse_config
from cs336_basics.training.optimizer import AdamW
from cs336_basics.training.utils import load_checkpoint_with_extra, save_checkpoint
from cs336_basics.training.wandb_logger import WandBLogger


def main():
    cfg = parse_config()
    # np.memmap on train and valid data
    # instantiate transformer
    # instantiate optimizer
    # instantiate wandblogger
    # train loop with optimizer.zero_grad(), model.forward(), loss.backward(), clip_gradient(), optimizer.step(), update_lr()
    # load and save checkpoints
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    model = Transformer(
        cfg.model.d_model,
        cfg.model.num_heads,
        cfg.model.d_ff,
        cfg.model.rope_theta,
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.num_layers,
        device=cfg.train.device,
        dtype=cfg.train.dtype
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps
    )

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    if cfg.run.resume_from is not None:
        iteration, extra_dict = load_checkpoint_with_extra(cfg.run.resume_from, model, optimizer)
    else:
        iteration, extra_dict = 0, {}

    wandb_logger = WandBLogger(cfg, extra_dict.get("run_id"), extra_dict.get("elapsed_offset"))

    train_data = np.memmap(cfg.data.train_path, dtype=cfg.data.token_dtype, mode="r")
    valid_data = np.memmap(cfg.data.valid_path, dtype=cfg.data.token_dtype, mode="r")


if __name__ == "__main__":
    main()
