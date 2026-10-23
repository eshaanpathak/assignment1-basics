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


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.embeddings = nn.Parameter(
            torch.empty(size=(num_embeddings, embedding_dim), dtype=dtype, device=device)
        )
        nn.init.trunc_normal_(self.embeddings, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


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
        # make sure gain is same precision type as x
        gain = self.gain.to(torch.float32)
        # we sum across the last dim of d_model size.
        # we keep dim so the result has consistent dim of (..., d_model)
        rms = ((torch.sum(x**2, dim=-1, keepdim=True) / self.d_model) + self.eps)**0.5
        result = torch.mul(x / rms, gain)
        # downcast
        return result.to(in_dtype)


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


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        positions = torch.arange(max_seq_len, device=device)
        inv_freqs = 1 / (theta**((2 * torch.arange(d_k // 2, device=device)) / d_k))
        angles = torch.outer(positions, inv_freqs) # (max_seq_len, d_k/2)

        cos_angles = torch.cos(angles)
        self.register_buffer("cos_angles", cos_angles, persistent=False)

        sin_angles = torch.sin(angles)
        self.register_buffer("sin_angles", sin_angles, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        cos_thetas = self.cos_angles[token_positions]
        sin_thetas = self.sin_angles[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        # x_even and x_odd:          (batch, heads, seq, d_k/2)
        # cos_thetas and sin_thetas: (batch,        seq, d_k/2)
        num_dim_insert = x_even.ndim - cos_thetas.ndim
        for _ in range(num_dim_insert):
            cos_thetas = cos_thetas.unsqueeze(-3)
            sin_thetas = sin_thetas.unsqueeze(-3)

        rot_even = x_even * cos_thetas - x_odd * sin_thetas
        rot_odd = x_odd * cos_thetas + x_even * sin_thetas

        stacked_rots = torch.stack((rot_even, rot_odd), dim=-1) # (..., d_k/2, 2)
        roped_x = torch.flatten(stacked_rots, start_dim=-2, end_dim=-1) # (..., d_k)

        return roped_x.to(x.dtype)


if __name__ == "__main__":
    linear_layer = Linear(2, 2)
    print(linear_layer.state_dict())
    
    swiglu_layer = SwiGLU(512)
    print(swiglu_layer.state_dict())

    rope_layer = RotaryPositionEmbedding(10000, 4, 8)
    x = torch.Tensor([
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    token_positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    print(f"cos_angles: {rope_layer.cos_angles}")
    print(f"sin_angles: {rope_layer.sin_angles}")
    print(f"roped_x: {rope_layer(x, token_positions)}")
