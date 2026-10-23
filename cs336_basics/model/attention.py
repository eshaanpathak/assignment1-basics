import torch


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
    scaled_dot_product = torch.matmul(Q, K.transpose(-1, -2)) / K.size(-1)**0.5
    if mask is not None:
        scaled_dot_product = torch.where(mask, scaled_dot_product, float("-inf"))
    sdp_attention = torch.matmul(softmax(scaled_dot_product, -1), V)
    return sdp_attention


if __name__ == "__main__":
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    print(torch.allclose(softmax(x, 1), torch.softmax(x, 1)))
