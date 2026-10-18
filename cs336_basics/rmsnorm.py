import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(size=(d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast to prevent overflow
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # we sum across the last dim of d_model size.
        # we keep dim so the result has consistent dim of (..., d_model)
        rms = ((torch.sum(x**2, dim=-1, keepdim=True) / self.d_model) + self.eps)**0.5
        result = torch.mul(x / rms, self.gain)
        # downcast
        return result.to(in_dtype)
