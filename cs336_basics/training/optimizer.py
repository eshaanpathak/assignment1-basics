import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    # -1 * o_target + log(sum(exp(o_a)))
    log_sum_exp = torch.logsumexp(inputs, dim=-1)
    # we unsqueeze to solect only that index in inputs
    # we squeeze to ensure log_sum_exp.shape == indexed_inputs.shape
    indexed_inputs = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_sum_exp - indexed_inputs)
