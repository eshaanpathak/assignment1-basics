import os
from typing import IO, BinaryIO

import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # -1 * o_target + log(sum(exp(o_a)))
    log_sum_exp = torch.logsumexp(inputs, dim=-1)
    # we unsqueeze to solect only that index in inputs
    # we squeeze to ensure log_sum_exp.shape == indexed_inputs.shape
    indexed_inputs = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_sum_exp - indexed_inputs)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint_dict, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint_dict = torch.load(src)
    model.load_state_dict(checkpoint_dict["model"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    return checkpoint_dict["iteration"]
