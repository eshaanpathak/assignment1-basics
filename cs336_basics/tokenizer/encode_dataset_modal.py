"""
Modal launcher for encoding a text file to a flat token-ID .bin. Infra only: the logic lives in
cs336_basics.tokenizer.encode_dataset (which calls the BPETokenizer in bpe_tokenizer.py).

Volume layout is the same as train_bpe_modal.py (one volume, one top-level directory per assignment):
    /assignment1/data/   inputs (raw text, vocab json, merges txt) and outputs (.bin)

The text file is split at <|endoftext|> boundaries and encoded in parallel across CPUS processes.
Run train_bpe_modal.py first so the vocab and merges files for the input are on the volume.

Upload the validation data once per file:
    uv run modal volume put cs336-data data/TinyStoriesV2-GPT4-valid.txt /assignment1/data/TinyStoriesV2-GPT4-valid.txt
    uv run modal volume put cs336-data data/owt_valid.txt /assignment1/data/owt_valid.txt

Run (all names are relative to /assignment1/data). TinyStories first as a cheap check of the setup:
    uv run modal run --detach cs336_basics/tokenizer/encode_dataset_modal.py --input-name TinyStoriesV2-GPT4-valid.txt --output-name ts_valid.bin --vocab-name ts_bpe_vocab_10000.json --merges-name ts_bpe_merges_10000.txt
    uv run modal run --detach cs336_basics/tokenizer/encode_dataset_modal.py --input-name TinyStoriesV2-GPT4-train.txt --output-name ts_train.bin --vocab-name ts_bpe_vocab_10000.json --merges-name ts_bpe_merges_10000.txt
    uv run modal run --detach cs336_basics/tokenizer/encode_dataset_modal.py --input-name owt_valid.txt --output-name owt_valid.bin --vocab-name owt_bpe_vocab_32000.json --merges-name owt_bpe_merges_32000.txt
    uv run modal run --detach cs336_basics/tokenizer/encode_dataset_modal.py --input-name owt_train.txt --output-name owt_train.bin --vocab-name owt_bpe_vocab_32000.json --merges-name owt_bpe_merges_32000.txt

Download results:
    uv run modal volume get cs336-data /assignment1/data/ts_valid.bin data/
    uv run modal volume get cs336-data /assignment1/data/ts_train.bin data/
"""

import modal

from cs336_basics.tokenizer.bpe_tokenizer import BPETokenizer

app = modal.App("cs336-encode-dataset")

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
VOLUME_PATH = "/vol"
ASSIGNMENT = "assignment1"
DATA_DIR = f"{VOLUME_PATH}/{ASSIGNMENT}/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex", "numpy")
    # bpe_tokenizer.py may import tests.common for the GPT-2 byte<->unicode helper, so ship tests/ too.
    .add_local_python_source("cs336_basics", "tests")
)

CPUS = 32
MEMORY_GB = 32


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    cpu=CPUS,
    memory=MEMORY_GB * 1024,
    timeout=6 * 60 * 60,
)
def encode_remote(
    input_name: str,
    output_name: str,
    vocab_name: str,
    merges_name: str,
    token_dtype: str,
):
    from cs336_basics.tokenizer.encode_dataset import encode_file_to_bin_with_multiprocessing

    volume.reload()
    tokenizer = BPETokenizer.from_files(f"{DATA_DIR}/{vocab_name}", f"{DATA_DIR}/{merges_name}", ["<|endoftext|>"])
    total = encode_file_to_bin_with_multiprocessing(
        tokenizer=tokenizer,
        input_path=f"{DATA_DIR}/{input_name}",
        output_path=f"{DATA_DIR}/{output_name}",
        token_dtype=token_dtype,
        cpu_core_count=CPUS,
    )
    volume.commit()


@app.local_entrypoint()
def main(
    input_name: str,
    output_name: str,
    vocab_name: str,
    merges_name: str,
    token_dtype: str = "uint16",
):
    encode_remote.remote(input_name, output_name, vocab_name, merges_name, token_dtype)
