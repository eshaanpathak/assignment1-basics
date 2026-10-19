import torch
from torch import nn

from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_ff is None:
            d_ff = (8 / 3) * d_model
            d_ff = max(round(d_ff / 64), 1) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu = torch.mul(w1_x, torch.sigmoid(w1_x))
        silu_w3 = torch.mul(silu, self.w3(x))
        result = self.w2(silu_w3)
        return result

if __name__ == "__main__":
    swiglu_layer = SwiGLU(512)
    print(swiglu_layer.state_dict())
