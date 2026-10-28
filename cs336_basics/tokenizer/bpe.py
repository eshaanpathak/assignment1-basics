import json
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import regex

from cs336_basics.tokenizer.pretokenization_example import find_chunk_boundaries
from tests.common import gpt2_bytes_to_unicode


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
    documents = regex.split(special_tokens_pat, text)
    return documents

def pretokenize(text: str) -> dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    matches = regex.finditer(PAT, text)
    token_counts = Counter()
    for match in matches:
        pretokenized_text = match.group().encode("utf-8")
        token_counts[tuple(bytes([b]) for b in pretokenized_text)] += 1
    return token_counts

def get_merge_candidate(
    pair_counts: dict[tuple[bytes, ...], int]
) -> tuple[bytes, bytes] | None:
    if not pair_counts:
        return
    return max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

def merge(
    token_counts: dict[tuple[bytes, ...], int],
    candidate: tuple[bytes, ...],
    pair_counts: dict[tuple[bytes, ...], int],
    pair_tokens: dict[tuple[bytes, ...], list[set]],
) -> dict[tuple[bytes, ...], int]:
    new_token_counts = Counter()
    pair_deltas = Counter()
    affected_tokens = pair_tokens[candidate].copy()

    for token in affected_tokens:
        counts = token_counts.pop(token)

        # remove the old token's pair contributions
        for pair, freq in Counter(zip(token, token[1:])).items():
            pair_deltas[pair] -= freq * counts
            pair_tokens[pair].discard(token)

        # merge the selected pair
        i = 0
        new_token = []
        while i < len(token):
            if i + 1 < len(token) and (token[i], token[i + 1]) == candidate:
                new_token.append(b"".join((token[i], token[i + 1])))
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_token_counts[tuple(new_token)] += counts

    # add the new token sequences and their pair contributions
    for token, counts in new_token_counts.items():
        token_counts[token] += counts

        for pair, freq in Counter(zip(token, token[1:])).items():
            pair_deltas[pair] += freq * counts
            pair_tokens[pair].add(token)

    # apply only the pair-count changes
    for pair, delta in pair_deltas.items():
        pair_counts[pair] += delta

        if pair_counts[pair] <= 0:
            pair_counts.pop(pair, None)

    return token_counts

def merge_to_vocab_size(
    token_counts: dict[tuple[bytes, ...], int],
    vocab: dict[int, bytes],
    vocab_size: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pair_counts = Counter()
    pair_tokens = defaultdict(set)

    for token, counts in token_counts.items():
        for pair in zip(token, token[1:]):
            pair_counts[pair] += counts
            pair_tokens[pair].add(token)

    merges = []
    while len(vocab) < vocab_size:
        merge_candidate = get_merge_candidate(pair_counts)
        if not merge_candidate:
            break
        token_counts = merge(token_counts, merge_candidate, pair_counts, pair_tokens)
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

def process_chunk(args: tuple) -> dict[tuple[bytes, ...], int]:
    file_path, start, end, special_tokens = args
    token_counts = Counter()
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        split_chunks = split_on_special_tokens(chunk, special_tokens)
        for split_chunk in split_chunks:
            token_counts.update(pretokenize(split_chunk))
    return token_counts

def train_bpe_with_multiprocessing(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_special_tokens = len(special_tokens)
    vocab = {i: special_tokens[i].encode("utf-8") for i in range(num_special_tokens)}
    vocab.update({i + num_special_tokens: bytes([i]) for i in range(256)})

    cpu_core_count = os.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, cpu_core_count * 10, special_tokens[0].encode("utf-8"))

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(
            boundaries[:-1], boundaries[1:]
        )
    ]

    with Pool(processes=cpu_core_count) as pool:
        results = pool.map(process_chunk, tasks)

    token_counts = Counter()
    for result in results:
        token_counts.update(result)

    vocab, merges = merge_to_vocab_size(token_counts, vocab, vocab_size)
    return vocab, merges

def bytes_to_str(b: bytes, byte_decoder: dict[int, str]) -> str:
    return "".join(byte_decoder[x] for x in b)

def train_and_save_bpe(
    input_path: str | os.PathLike,
    acronym: str,
    vocab_size: int,
    special_tokens: list[str],
    output_dir: str = "data/",
):
    t0 = time.perf_counter()
    vocab, merges = train_bpe_with_multiprocessing(input_path, vocab_size, special_tokens)
    t1 = time.perf_counter()

    os.makedirs(output_dir, exist_ok=True)
    gpt2_byte_decoder = gpt2_bytes_to_unicode()

    vocab_path = os.path.join(output_dir, f"{acronym}_bpe_vocab_{vocab_size}.json")
    serializable_vocab = {bytes_to_str(tok, gpt2_byte_decoder): idx for idx, tok in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False)

    merges_path = os.path.join(output_dir, f"{acronym}_bpe_merges_{vocab_size}.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        merge_writes = []
        for left, right in merges:
            merge_str = f"{bytes_to_str(left, gpt2_byte_decoder)} {bytes_to_str(right, gpt2_byte_decoder)}\n"
            merge_writes.append(merge_str)
        f.writelines(merge_writes)

    longest_id, longest_bytes = max(vocab.items(), key=lambda x: len(x[1]))
    longest_str = longest_bytes.decode("utf-8", errors="replace")

    print(f"{input_path} Duration: {t1 - t0} seconds")
    print(f"{input_path} Longest Token | Decoded: {longest_str!r} | ID: {longest_id} | Bytes: {len(longest_bytes)}\n")


if __name__ == "__main__":
    train_and_save_bpe("data/TinyStoriesV2-GPT4-train.txt", "ts", 10_000, ["<|endoftext|>"])
    # train_and_save_bpe("data/owt_train.txt", "owt", 32_000, ["<|endoftext|>"])
