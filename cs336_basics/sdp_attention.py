import torch

from cs336_basics.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
):
    scaled_dot_product = torch.matmul(Q, K.transpose(-1, -2)) / K.size(-1)**0.5
    if mask is not None:
        scaled_dot_product = torch.where(mask, scaled_dot_product, float("-inf"))
    sdp_attention = torch.matmul(softmax(scaled_dot_product, -1), V)
    return sdp_attention
