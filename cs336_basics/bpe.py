import regex
import os
from collections import Counter
from multiprocessing import Pool

from pretokenization_example import find_chunk_boundaries


def retrieve_text_from_file(input_path: str | os.PathLike) -> str:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    return text

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if special_tokens == []:
        return [text]
    special_tokens = sorted(special_tokens, key=lambda x: -len(x))
    escaped_special_tokens = []
    for special_token in special_tokens:
        escaped_special_token = regex.escape(special_token)
        escaped_special_tokens.append(escaped_special_token)
    special_tokens_pat = "|".join(escaped_special_tokens)
    documents  = regex.split(special_tokens_pat, text)
    return documents

def pretokenize(text: str) -> dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = regex.finditer(PAT, text)
    token_counts = Counter()
    for match in matches:
        pretokenized_text = match.group().encode("utf-8")
        token_counts[tuple(bytes([b]) for b in pretokenized_text)] += 1
    return token_counts

def get_merge_candidate(token_counts: dict[tuple[bytes, ...], int]) -> tuple[bytes, bytes]:
    pair_counts = Counter()
    for token, counts in token_counts.items():
        for i in range(len(token) - 1):
            window = token[i : i + 2]
            pair_counts[window] += counts
    candidate = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
    return candidate

def merge(token_counts: dict[tuple[bytes, ...], int], candidate: tuple[bytes, ...]) -> dict[tuple[bytes, ...], int]:
    new_token_counts = Counter()
    for token, counts in token_counts.items():
        new_token_bytes = []
        i = 0
        while i < len(token):
            if i + 1 < len(token) and (token[i], token[i + 1]) == candidate:
                new_token_bytes.append(b"".join((token[i], token[i + 1])))
                i += 2
            else:
                new_token_bytes.append(token[i])
                i += 1
        new_token = tuple(new_token_bytes)
        new_token_counts[new_token] += counts
    return new_token_counts

def merge_to_vocab_size(
    token_counts: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    vocab_size: int
):
    merges = []
    while len(vocab) < vocab_size:
        merge_candidate = get_merge_candidate(token_counts)
        token_counts = merge(token_counts, merge_candidate)
        vocab[len(vocab)] = b"".join(merge_candidate)
        merges.append(merge_candidate)
    return vocab, merges

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = retrieve_text_from_file(input_path)
    documents = split_on_special_tokens(text, special_tokens)
    num_special_tokens = len(special_tokens)
    vocab = {i: special_tokens[i].encode("utf-8") for i in range(num_special_tokens)}
    vocab.update({i + num_special_tokens: bytes([i]) for i in range(256)})
    token_counts = Counter()
    for document in documents:
        token_counts.update(pretokenize(document))
    vocab, merges = merge_to_vocab_size(token_counts, vocab, vocab_size)
    return vocab, merges

def process_chunk(args):
    file_path, start, end, special_tokens, vocab, vocab_size = args
    token_counts = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode(
            "utf-8", errors="ignore"
        )
        split_chunks = split_on_special_tokens(chunk, special_tokens)
        for chunk in split_chunks:
            token_counts.update(pretokenize(chunk))
        vocab, merges = merge_to_vocab_size(token_counts, vocab, vocab_size)
    return token_counts, vocab, merges

def train_bpe_with_multiprocessing(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_special_tokens = len(special_tokens)
    vocab = {i: special_tokens[i].encode("utf-8") for i in range(num_special_tokens)}
    vocab.update({i + num_special_tokens: bytes([i]) for i in range(256)})

    with open(..., "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    tasks = [
        (input_path, start, end, special_tokens, vocab, vocab_size)
        for start, end in zip(
            boundaries[:-1], boundaries[1:]
        )
    ]

    with Pool(num_processes) as pool:
        token_counts, vocab, merges = pool.map(process_chunk, tasks)
    
    
    return vocab, merges

import time
start = time.time()
vocab, merges = train_bpe_with_multiprocessing("data/TinyStoriesV2-GPT4-valid.txt", 10_000, ["<|endoftext|>"])
end = time.time() - start
print(vocab)
print()
print(merges)
print()
print(end)
