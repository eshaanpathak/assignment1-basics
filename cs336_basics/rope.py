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
        x_dtype = x.dtype
        cos_positions = self.cos_angles[token_positions]
        sin_positions = self.sin_angles[token_positions]

        first_values = x[..., ::2]
        second_values = x[..., 1::2]

        first_row = first_values * cos_positions - second_values * sin_positions
        second_row = second_values * cos_positions + first_values * sin_positions

        stacked_rows = torch.stack((first_row, second_row), dim=-1) # (..., d_k/2, 2)
        roped_x = torch.flatten(stacked_rows, start_dim=-2, end_dim=-1) # (..., d_k)

        return roped_x.to(x_dtype)


if __name__ == "__main__":
    rope_layer = RotaryPositionEmbedding(10000, 4, 8)
    x = torch.Tensor([
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    token_positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    print(rope_layer(x, token_positions))
