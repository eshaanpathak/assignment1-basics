import torch
from torch import nn


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
