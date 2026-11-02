import os
import time

import numpy as np
import torch

from cs336_basics.model.transformer import Transformer
from cs336_basics.training.config import parse_config
from cs336_basics.training.optimizer import AdamW, clip_gradients, cosine_lr_schedule, update_lr
from cs336_basics.training.utils import cross_entropy, get_batch, load_checkpoint_with_extra, save_checkpoint
from cs336_basics.training.wandb_logger import WandBLogger


def main():
    cfg = parse_config()
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

    wandb_logger = WandBLogger(cfg, extra_dict.get("run_id"), extra_dict.get("elapsed_offset", 0))

    train_data = np.memmap(cfg.data.train_path, dtype=cfg.data.token_dtype, mode="r")
    valid_data = np.memmap(cfg.data.valid_path, dtype=cfg.data.token_dtype, mode="r")

    valid_loss = extra_dict.get("valid_loss", float("inf"))
    best_valid_loss = extra_dict.get("best_valid_loss", float("inf"))

    def make_extra() -> dict:
        return {
            "run_id": wandb_logger.run_id,
            "elapsed_offset": wandb_logger.elapsed(),
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
        }

    t = iteration  # stays defined (and equal to the resumed step) if the loop below never runs
    for t in range(iteration + 1, cfg.optim.max_iters + 1):
        model.train()
        batch_data, target_data = get_batch(
            train_data,
            cfg.optim.batch_size,
            cfg.model.context_length,
            cfg.train.device
        )
        batch_preds = model(batch_data)
        loss = cross_entropy(batch_preds, target_data)

        cosine_lr = cosine_lr_schedule(
            t,
            cfg.optim.lr,
            cfg.optim.min_lr,
            cfg.optim.warmup_iters,
            cfg.optim.cosine_cycle_iters
        )
        update_lr(optimizer, cosine_lr)
        optimizer.zero_grad()

        loss.backward()
        parameters, total_norm = clip_gradients(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()

        if t % cfg.train.log_interval == 0:
            wandb_logger.log_train(t, loss.item(), cosine_lr, total_norm.item())

        if t % cfg.train.eval_interval == 0:
            model.eval()
            with torch.inference_mode():
                batch_data, target_data = get_batch(
                    valid_data,
                    cfg.optim.eval_batch_size,
                    cfg.model.context_length,
                    cfg.train.device
                )
                batch_preds = model(batch_data)
                valid_loss = cross_entropy(batch_preds, target_data).item()
                wandb_logger.log_val(t, valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(
                    model,
                    optimizer,
                    t,
                    os.path.join(cfg.train.checkpoint_dir, "checkpoint_best.pt"),
                    extra=make_extra()
                )

        if t % cfg.train.checkpoint_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                t,
                os.path.join(cfg.train.checkpoint_dir, f"checkpoint_{t}.pt"),
                extra=make_extra()
            )

    if t > iteration and t % cfg.train.checkpoint_interval != 0:
        save_checkpoint(
            model,
            optimizer,
            t,
            os.path.join(cfg.train.checkpoint_dir, f"checkpoint_{t}.pt"),
            extra=make_extra()
        )

    wandb_logger.finish()


if __name__ == "__main__":
    main()
