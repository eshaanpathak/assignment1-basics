import torch
from torch import nn

from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.layers import Embedding, Linear, RMSNorm, SiLU, SwiGLU


def make_norm(d_model: int, use_rmsnorm: bool, device=None, dtype=None) -> nn.Module:
    """RMSNorm, or a no-op when the norm-removal ablation is on."""
    if use_rmsnorm:
        return RMSNorm(d_model, device=device, dtype=dtype)
    return nn.Identity()


class TransformerBlock(nn.Module):
    """Defaults give the baseline block (pre-norm, RMSNorm, RoPE, SwiGLU). The flags switch on the ablations:
    use_rmsnorm=False, post_norm=True, use_rope=False, ffn="silu"."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn: str = "swiglu",
    ):
        super().__init__()

        self.post_norm = post_norm
        self.ln1 = make_norm(d_model, use_rmsnorm, device, dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, theta=theta if use_rope else None, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ln2 = make_norm(d_model, use_rmsnorm, device, dtype)
        if ffn == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn == "silu":
            self.ffn = SiLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"ffn={ffn} must be 'swiglu' or 'silu'")

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    ):
        if self.post_norm:
            x = self.ln1(torch.add(x, self.attn(x, token_positions)))
            return self.ln2(torch.add(x, self.ffn(x)))
        x = torch.add(x, self.attn(self.ln1(x), token_positions))
        return torch.add(x, self.ffn(self.ln2(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn: str = "swiglu",
    ):
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype,
                    use_rmsnorm=use_rmsnorm, post_norm=post_norm, use_rope=use_rope, ffn=ffn,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = make_norm(d_model, use_rmsnorm and not post_norm, device, dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, sequence_length = x.shape[0], x.shape[1]
        seq_len_range = torch.arange(sequence_length, dtype=torch.int64, device=x.device)
        token_positions = seq_len_range.unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
