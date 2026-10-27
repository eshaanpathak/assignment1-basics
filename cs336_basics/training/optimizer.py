import math
from collections.abc import Callable, Iterable

import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # -1 * o_target + log(sum(exp(o_a)))
    log_sum_exp = torch.logsumexp(inputs, dim=-1)
    # we unsqueeze to solect only that index in inputs
    # we squeeze to ensure log_sum_exp.shape == indexed_inputs.shape
    indexed_inputs = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_sum_exp - indexed_inputs)

def cosine_lr_schedule(t: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    if t < warmup_iters:
        return (t / warmup_iters) * lr_max
    if t <= cosine_cycle_iters:
        return lr_min + 0.5 * (1 + math.cos(((t - warmup_iters) / (cosine_cycle_iters - warmup_iters)) * math.pi)) * (lr_max - lr_min)
    return lr_min

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters = list(parameters)

    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            total_norm += torch.linalg.vector_norm(p.grad.data) ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= max_l2_norm / (total_norm + 10e-6)

    return parameters


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, weight_decay, betas, eps = group["lr"], group["weight_decay"], group["betas"], group["eps"]
            beta_1, beta_2 = betas[0], betas[1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                first_moment = state.get("first_moment",
                    torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                )
                second_moment = state.get("second_moment", 
                    torch.zeros(p.shape, dtype=p.dtype, device=p.device)
                )
                t = state.get("t", 1)

                grad = p.grad.data
                lr_t = lr * (1 - beta_2**t)**0.5 / (1 - beta_1**t)
                p.data -= lr * weight_decay * p.data
                first_moment = beta_1 * first_moment + (1 - beta_1) * grad
                second_moment = beta_2 * second_moment + (1 - beta_2) * grad**2
                p.data -= lr_t * first_moment / (second_moment**0.5 + eps)

                state["first_moment"] = first_moment
                state["second_moment"] = second_moment
                state["t"] = t + 1
        return loss


if __name__ == "__main__":
    torch.manual_seed(42)
    init_weights = 5 * torch.randn((10, 10))
    for lr in [1, 1e1, 1e2, 1e3]:
        weights = torch.nn.Parameter(init_weights.clone())
        opt = SGD([weights], lr=lr)
        print(f"\nSGD optimizer: {opt.defaults}")
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
