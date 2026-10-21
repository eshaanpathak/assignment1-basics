import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(size=(out_features, in_features), dtype=dtype, device=device)
        )
        std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return torch.matmul(x, self.weight.T)


if __name__ == "__main__":
    linear_layer = Linear(2, 2)
    print(linear_layer.state_dict())
