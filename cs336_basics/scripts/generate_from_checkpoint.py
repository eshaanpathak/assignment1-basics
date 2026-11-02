"""
Load a training checkpoint and sample text from it with generate() in cs336_basics/training/generate.py, either on
Modal (reads the checkpoint straight off the volume, no download) or locally. Glue only: the model, tokenizer and
decoding logic live in their own modules.

A checkpoint stores the weights but not the architecture, so pass the same --model.* flags the run was trained with
(same names and defaults as config.py; anything you leave out falls back to the config.py default).

On Modal (same volume as train_modal.py; run name = the checkpoint directory, i.e. the --run-name you trained with):
    uv run modal run cs336_basics/scripts/generate_from_checkpoint.py --run-name ts_default --prompt "Once upon a time,"
    uv run modal run cs336_basics/scripts/generate_from_checkpoint.py --run-name ts_bs_sweep_bs256 --checkpoint checkpoint_5000.pt \
        --temperature 0.8 --top-p 0.9 --num-samples 3
    uv run modal run cs336_basics/scripts/generate_from_checkpoint.py --run-name ts_big --model-args "--model.d_model 768 --model.num_layers 8"

    Defaults: --checkpoint checkpoint_best.pt, --vocab-name ts_bpe_vocab_10000.json, --merges-name ts_bpe_merges_10000.txt
    (files in /vol/assignment1/data, as written by train_bpe_modal.py). Use the owt_* vocab/merges for OpenWebText runs.

Locally (CPU or --device mps; download the files first, ~270 MB for a default-size checkpoint):
    uv run modal volume get cs336-data /assignment1/checkpoints/ts_default/checkpoint_best.pt checkpoints/ts_default/
    uv run modal volume get cs336-data /assignment1/data/ts_bpe_vocab_10000.json data/
    uv run modal volume get cs336-data /assignment1/data/ts_bpe_merges_10000.txt data/
    uv run python -m cs336_basics.scripts.generate_from_checkpoint \
        --checkpoint checkpoints/ts_default/checkpoint_best.pt \
        --vocab data/ts_bpe_vocab_10000.json --merges data/ts_bpe_merges_10000.txt \
        --prompt "Once upon a time," --temperature 0.8 --top-p 0.9
"""

import argparse
import shlex

import torch

from cs336_basics.model.transformer import Transformer
from cs336_basics.tokenizer.bpe_tokenizer import BPETokenizer
from cs336_basics.training.config import parse_config
from cs336_basics.training.generate import generate
from cs336_basics.training.train_modal import CHECKPOINT_ROOT, DATA_DIR, GPU, VOLUME_PATH, app, image, volume

END_OF_TEXT = "<|endoftext|>"


def sample(
    checkpoint: str,
    vocab: str,
    merges: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    seed: int,
    device: str,
    model_argv: list[str],
) -> tuple[str, list[tuple[int, str]]]:
    """Returns (a one-line description of the loaded checkpoint, [(num_new_tokens, prompt + completion), ...])."""
    cfg = parse_config(model_argv)  # --model.* flags and defaults, exactly as in training
    torch.manual_seed(seed)

    model = Transformer(
        cfg.model.d_model,
        cfg.model.num_heads,
        cfg.model.d_ff,
        cfg.model.rope_theta,
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.num_layers,
        device=device,
        dtype=cfg.train.torch_dtype,
    )
    # Only the weights are needed; skip the optimizer state in the file.
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    info = f"loaded {checkpoint} (iteration {ckpt['iteration']}, extra={ckpt.get('extra', {})})"

    tokenizer = BPETokenizer.from_files(vocab, merges, [END_OF_TEXT])
    end_of_text_id = tokenizer.bytes_to_ids[END_OF_TEXT.encode("utf-8")]
    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)

    samples = []
    for _ in range(num_samples):
        new_ids = generate(
            model,
            prompt_ids,
            end_of_text_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        samples.append((new_ids.numel(), prompt + tokenizer.decode(new_ids.tolist())))
    return info, samples


def _print_samples(info: str, samples: list[tuple[int, str]]):
    print(info)
    for i, (num_tokens, text) in enumerate(samples, 1):
        print(f"\n--- sample {i} ({num_tokens} new tokens) ---")
        print(text)


# ---- Modal ----

@app.function(image=image, volumes={VOLUME_PATH: volume}, gpu=GPU, timeout=30 * 60)
def generate_remote(
    run_name: str,
    checkpoint_name: str,
    vocab_name: str,
    merges_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    seed: int,
    model_args: str,
) -> tuple[str, list[tuple[int, str]]]:
    volume.reload()  # pick up checkpoints committed by training runs that finished after this container image started
    return sample(
        f"{CHECKPOINT_ROOT}/{run_name}/{checkpoint_name}",
        f"{DATA_DIR}/{vocab_name}",
        f"{DATA_DIR}/{merges_name}",
        prompt,
        max_new_tokens,
        temperature,
        top_p,
        num_samples,
        seed,
        "cuda",
        shlex.split(model_args),
    )


@app.local_entrypoint()
def generate_modal(
    run_name: str,
    checkpoint: str = "checkpoint_best.pt",
    vocab_name: str = "ts_bpe_vocab_10000.json",
    merges_name: str = "ts_bpe_merges_10000.txt",
    prompt: str = "Once upon a time,",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_samples: int = 1,
    seed: int = 0,
    model_args: str = "",
):
    info, samples = generate_remote.remote(
        run_name, checkpoint, vocab_name, merges_name, prompt,
        max_new_tokens, temperature, top_p, num_samples, seed, model_args,
    )
    _print_samples(info, samples)


# ---- Local ----

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a checkpoint_*.pt written by train.py")
    parser.add_argument("--vocab", required=True, help="tokenizer vocab json used for training")
    parser.add_argument("--merges", required=True, help="tokenizer merges txt used for training")
    parser.add_argument("--prompt", default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    # Everything else (--model.d_model, --model.num_layers, ...) goes through the training config parser.
    args, model_argv = parser.parse_known_args(argv)
    info, samples = sample(
        args.checkpoint, args.vocab, args.merges, args.prompt, args.max_new_tokens,
        args.temperature, args.top_p, args.num_samples, args.seed, args.device, model_argv,
    )
    _print_samples(info, samples)


if __name__ == "__main__":
    main()
