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
import time
from itertools import batched
from multiprocessing import Pool

import numpy as np

from cs336_basics.tokenizer.bpe_tokenizer import BPETokenizer


def encode_file_to_bin(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    token_dtype: str = "uint16",
    buffer_tokens: int = 1_000_000,
) -> int:
    t0 = time.perf_counter()
    dtype = np.dtype(token_dtype)
    total = 0
    # newline="" keeps \r\n as-is, so the tokens match encoding the raw file contents.
    with open(input_path, encoding="utf-8", newline="") as src, open(output_path, "wb") as dst:
        token_ids = tokenizer.encode_iterable(src)
        for batch in batched(token_ids, buffer_tokens):
            np.array(batch, dtype=dtype).tofile(dst)
            total += len(batch)
    t1 = time.perf_counter()
    print(f"Wrote {total} tokens ({token_dtype}) to {output_path} in {t1 - t0} seconds")
    return total

def encode_chunk(args):
    tokenizer, text = args
    return tokenizer.encode(text)

def encode_file_to_bin_with_multiprocessing(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    token_dtype: str = "uint16",
    buffer_tokens: int = 1_000_000,
    cpu_core_count: int | None = None
) -> int:
    t0 = time.perf_counter()
    dtype = np.dtype(token_dtype)
    total = 0

    if cpu_core_count is None:
        cpu_core_count = os.cpu_count()

    with open(input_path, encoding="utf-8", newline="") as src, open(output_path, "wb") as dst, Pool(cpu_core_count) as pool:
        chunks = iter(lambda: src.read(buffer_tokens), "")
        for tokens in pool.imap(
            encode_chunk,
            ((tokenizer, chunk) for chunk in chunks)
        ):
            np.asarray(tokens, dtype=dtype).tofile(dst)
            total += len(tokens)

    t1 = time.perf_counter()
    print(f"Wrote {total} tokens ({token_dtype}) to {output_path} in {t1 - t0} seconds")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--dtype", default="uint16")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    total = encode_file_to_bin_with_multiprocessing(tokenizer, args.input, args.output, args.dtype)


if __name__ == "__main__":
    main()
