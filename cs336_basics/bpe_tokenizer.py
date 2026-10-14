import json
from collections.abc import Iterable, Iterator

import regex


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens

        if self.special_tokens:
            vocab_bytes = set(self.vocab.values())
            for special_token in special_tokens:
                encoded_special_token = special_token.encode("utf-8")
                if encoded_special_token not in vocab_bytes:
                    self.vocab[len(self.vocab)] = encoded_special_token

        self.bytes_to_ids = {value: key for key, value in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath) as vocab_f:
            vocab = json.load(vocab_f)
        vocab = {value: get_gpt2_unicode_to_bytes_string(key) for key, value in vocab.items()}

        merges = []
        with open(merges_filepath) as merges_f:
            for line in merges_f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(get_gpt2_unicode_to_bytes_string(b) for b in cleaned_line.split(" ")))
        return cls(vocab, merges, special_tokens)

    def split_on_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        special_tokens = sorted(self.special_tokens, key=lambda x: -len(x))
        escaped_special_tokens = []
        for special_token in special_tokens:
            escaped_special_token = regex.escape(special_token)
            escaped_special_tokens.append(escaped_special_token)
        special_tokens_pat = "|".join(escaped_special_tokens)
        documents = regex.split(f"({special_tokens_pat})", text)
        return documents

    def pretokenize(self, splitted_text: list[str]) -> list[tuple[bytes, ...]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        all_pretokenized_bytes = []
        for text in splitted_text:
            if self.special_tokens and text in self.special_tokens:
                all_pretokenized_bytes.append((text.encode("utf-8"),))
            else:
                matches = regex.finditer(PAT, text)
                for match in matches:
                    pretokenized_text = match.group().encode("utf-8")
                    pretokenized_bytes = tuple(bytes([b]) for b in pretokenized_text)
                    all_pretokenized_bytes.append(pretokenized_bytes)
        return all_pretokenized_bytes

    def merge(
        self,
        all_pretokenized_bytes: list[tuple[bytes, ...]],
        candidate: tuple[bytes, ...]
    ) -> list[tuple[bytes, ...]]:
        # new_all_pretokenized_bytes is recreated each time we call merge() which is inefficient
        new_all_pretokenized_bytes = []
        for token in all_pretokenized_bytes:
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
            new_all_pretokenized_bytes.append(new_token)
        return new_all_pretokenized_bytes

    def encode(self, text: str) -> list[int]:
        splitted_text = self.split_on_special_tokens(text)
        all_pretokenized_bytes = self.pretokenize(splitted_text)
        for candidate in self.merges:
            all_pretokenized_bytes = self.merge(all_pretokenized_bytes, candidate)
        encoded_ids = []
        for tuple_bytes in all_pretokenized_bytes:
            for byte in tuple_bytes:
                byte_id = self.bytes_to_ids[byte]
                encoded_ids.append(byte_id)
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            encoded_s = self.encode(s)
            yield from encoded_s

    def decode(self, ids: list[int]) -> str:
        decoded_tokens = []
        for bpe_id in ids:
            assert bpe_id in self.vocab, f"{bpe_id} is not in vocab"
            token = self.vocab[bpe_id]
            decoded_tokens.append(token)
        return b"".join(decoded_tokens).decode("utf-8", errors="replace")

def gpt2_unicode_to_bytes() -> dict[str, int]:
    """
    Returns a mapping between a printable unicode string character representation 
    and every possible byte (an integer from 0 to 255). This function is taken
    from the GPT-2 code and was modified from the version in `tests/common.py`.
    See that file for the entire docstring.
    """
    # These 188 integers can be used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # characters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(characters, bs))
    return d

def get_gpt2_unicode_to_bytes_string(text: str) -> bytes:
    bytes_string = []
    for c in text:
        bytes_string.append(gpt2_unicode_to_bytes_dict[c])
    return bytes(bytes_string)

gpt2_unicode_to_bytes_dict = gpt2_unicode_to_bytes()

if __name__ == "__main__":
    from tests.test_tokenizer import MERGES_PATH, VOCAB_PATH

    bpe_tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"])
    print(bpe_tokenizer.encode("hellooo i am <|endoftext|> hungry"))
