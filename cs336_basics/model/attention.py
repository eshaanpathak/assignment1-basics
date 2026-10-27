import torch
from torch import nn

from cs336_basics.model.layers import Linear, RotaryPositionEmbedding


def softmax(x: torch.Tensor, dim_idx: int) -> torch.Tensor:
    max_values = torch.max(x, dim_idx, keepdim=True).values
    exp_x = torch.exp(x - max_values)
    softmax_x = exp_x / torch.sum(exp_x, dim=dim_idx, keepdim=True)
    return softmax_x


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
):
    scaled_dot_product = torch.matmul(Q, K.transpose(-2, -1)) / K.size(-1)**0.5
    if mask is not None:
        scaled_dot_product = torch.where(mask, scaled_dot_product, float("-inf"))
    sdp_attention = torch.matmul(softmax(scaled_dot_product, -1), V)
    return sdp_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: int | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # d_k == d_v == d_model / num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by num_heads={self.num_heads}"
            )

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionEmbedding(
                theta, self.d_model // self.num_heads, max_seq_len, device
            )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ):
        x_shape_until_last = x.shape[:-1]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # x shape is (..., seq_len, d_model)
        # q, k, and v are (..., seq_len, d_model)
        # we want (..., num_heads, seq_len, d_k)
        # we infer d_k with -1
        new_size = (*x_shape_until_last, self.num_heads, -1)
        q = q.view(new_size).transpose(-3, -2)
        k = k.view(new_size).transpose(-3, -2)
        v = v.view(new_size).transpose(-3, -2)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).to(torch.bool)

        # (..., num_heads, seq_len, d_k)
        multihead = scaled_dot_product_attention(q, k, v, mask=mask)
        multihead = multihead.transpose(-3, -2).reshape((*x_shape_until_last, self.d_model))

        return self.o_proj(multihead)


if __name__ == "__main__":
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    print(torch.allclose(softmax(x, 1), torch.softmax(x, 1)))
