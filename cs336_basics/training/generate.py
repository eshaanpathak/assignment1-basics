import torch

from cs336_basics.model.attention import softmax


def sample_from_top_p(token_probabilities: torch.Tensor, p: float):
    values, indices = torch.sort(token_probabilities, descending=True)
    cumsum_values = torch.cumsum(values, dim=-1)
    cumsum_values_bool = (cumsum_values - values) < p
    top_p_probs = values[cumsum_values_bool]
    indices = indices[:top_p_probs.shape[0]]
    top_p_probs_norm = top_p_probs / top_p_probs.sum()
    idx = torch.multinomial(top_p_probs_norm, num_samples=1).item()
    return indices[idx]


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    end_of_text_id: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0
):  
    num_dims = prompt_ids.ndim
    if num_dims != 1:
        raise ValueError(f"prompt_ids.ndim=={num_dims} must be 1")
    num_tokens = prompt_ids.numel()
    if num_tokens == 0:
        raise ValueError(f"prompt_ids.numel()=={num_tokens} must be at least 1")
    
    model_training = model.training
    model.eval()

    device = next(model.parameters()).device
    all_token_ids = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        context_token_ids = all_token_ids[-model.context_length:]
        logits = model(context_token_ids.unsqueeze(0))[0, -1, :]
        token_probs = softmax(logits, -1, temperature)

        if top_p == 1.0:
            generated_token_id = torch.multinomial(token_probs, num_samples=1)
        else:
            generated_token_id = sample_from_top_p(token_probs, top_p)
        
        if end_of_text_id == generated_token_id.item():
            break

        all_token_ids = torch.cat((all_token_ids, generated_token_id.unsqueeze(-1)))
    
    if model_training:
        model.train()
    
    return all_token_ids[prompt_ids.shape[0]:]
