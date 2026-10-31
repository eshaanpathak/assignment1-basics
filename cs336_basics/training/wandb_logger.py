import os
import time

import wandb

from cs336_basics.training.config import Config


class WandBLogger:
    def __init__(self, cfg: Config, run_id: str | None = None, elapsed_offset: float = 0.0):
        self.run = self._init_run(cfg, run_id) if cfg.wandb.enabled else None
        self.elapsed_offset = elapsed_offset  # wall-clock seconds spent in earlier runs of a resumed job
        self.start_time = time.perf_counter()  # after init, so wandb startup does not count as training time

    @staticmethod
    def _init_run(cfg: Config, run_id: str | None):
        if cfg.wandb.dir:
            os.makedirs(cfg.wandb.dir, exist_ok=True)

        # If online init fails, retry offline so the run is still recorded locally for `wandb sync`.
        modes = [cfg.wandb.mode]
        if cfg.wandb.mode != "offline":
            modes.append("offline")
        for mode in modes:
            try:
                return wandb.init(
                    project=cfg.wandb.project,
                    name=cfg.train.run_name,
                    config=cfg.to_dict(),
                    id=run_id,
                    resume="allow" if run_id else None,
                    mode=mode,
                    dir=cfg.wandb.dir,
                )
            except Exception as e:
                print(f"wandb init failed (mode={mode}): {e}")
        print("continuing without logging")
        return None

    @property
    def run_id(self) -> str | None:
        """Save this in checkpoints so a resumed job continues"""
        return self.run.id if self.run else None

    def elapsed(self) -> float:
        """Wall-clock seconds across all resumes. Save this in checkpoints and pass it back as elapsed_offset."""
        return self.elapsed_offset + time.perf_counter() - self.start_time

    def log_train(self, step: int, loss: float, lr: float, grad_norm: float) -> None:
        self._log(step, {"train/loss": loss, "train/lr": lr, "train/grad_norm": grad_norm})

    def log_val(self, step: int, loss: float) -> None:
        self._log(step, {"val/loss": loss})

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as e:
            print(f"wandb finish failed: {e}")

    def _log(self, step: int, metrics: dict[str, float]) -> None:
        if self.run is None:
            return
        try:
            self.run.log({**metrics, "wallclock_s": self.elapsed()}, step=step)
        except Exception as e:
            print(f"wandb log failed at step {step}: {e}")
