"""
Tokenize a text file with a trained BPETokenizer and stream the IDs to a flat binary file.
Infra only: the tokenization logic lives in cs336_basics.tokenizer.bpe_tokenizer.

The output is raw token IDs in one fixed dtype with no header, so training must read it with the same
dtype (DataConfig.token_dtype), e.g. np.memmap(path, dtype=token_dtype, mode="r").

Local run:
    uv run python -m cs336_basics.tokenizer.encode_dataset \
        --input data/TinyStoriesV2-GPT4-valid.txt --output data/ts_valid.bin \
        --vocab data/ts_bpe_vocab_10000.json --merges data/ts_bpe_merges_10000.txt
"""

import argparse
import os
from itertools import batched

import numpy as np

from cs336_basics.tokenizer.bpe_tokenizer import BPETokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]


def encode_file_to_bin(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    token_dtype: str = "uint16",
    buffer_tokens: int = 1_000_000,
) -> int:
    """Stream-encode input_path into output_path and return the number of tokens written."""
    dtype = np.dtype(token_dtype)
    total = 0
    # newline="" keeps \r\n as-is, so the tokens match encoding the raw file contents.
    with open(input_path, encoding="utf-8", newline="") as src, open(output_path, "wb") as dst:
        token_ids = tokenizer.encode_iterable(src)
        for batch in batched(token_ids, buffer_tokens):
            np.array(batch, dtype=dtype).tofile(dst)
            total += len(batch)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--dtype", default="uint16")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, SPECIAL_TOKENS)
    total = encode_file_to_bin(tokenizer, args.input, args.output, args.dtype)
    print(f"wrote {total} tokens ({args.dtype}) to {args.output}")


if __name__ == "__main__":
    main()
