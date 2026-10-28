"""Modal launcher for BPE training. Infra only: the training logic lives in cs336_basics.tokenizer.bpe.

Volume layout (one volume, one top-level directory per assignment):
    /assignment1/data/   inputs (raw text) and outputs (vocab json, merges txt)

Create the volume, if it does not already exist:
    uv run modal volume create cs336-data
    uv run modal volume list

Upload the data once per file (OWT is ~12GB, so that one takes a while):
    uv run modal volume put cs336-data data/TinyStoriesV2-GPT4-train.txt /assignment1/data/TinyStoriesV2-GPT4-train.txt
    uv run modal volume put cs336-data data/owt_train.txt /assignment1/data/owt_train.txt

Run (all three flags are required; --input-name is relative to /assignment1/data).
TinyStories first as a cheap check of the setup:
    uv run modal run cs336_basics/tokenizer/train_bpe_modal.py --input-name TinyStoriesV2-GPT4-train.txt --acronym ts --vocab-size 10000
    uv run modal run cs336_basics/tokenizer/train_bpe_modal.py --input-name owt_train.txt --acronym owt --vocab-size 32000

Download results:
    uv run modal volume get cs336-data /assignment1/data/ts_bpe_vocab_10000.json data/
    uv run modal volume get cs336-data /assignment1/data/ts_bpe_merges_10000.txt data/
    uv run modal volume get cs336-data /assignment1/data/owt_bpe_vocab_32000.json data/
    uv run modal volume get cs336-data /assignment1/data/owt_bpe_merges_32000.txt data/

`TinyStoriesV2-GPT4-train.txt` took 59 seconds locally and 104 seconds on Modal.
`owt_train.txt` took __ seconds on Modal.
"""

import modal

app = modal.App("cs336-bpe")

volume = modal.Volume.from_name("cs336-data", create_if_missing=True)
VOLUME_PATH = "/vol"
ASSIGNMENT = "assignment1"
DATA_DIR = f"{VOLUME_PATH}/{ASSIGNMENT}/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("regex")
    # bpe.py imports tests.common for the GPT-2 byte<->unicode helper, so ship tests/ too.
    .add_local_python_source("cs336_basics", "tests")
)

CPUS = 32
MEMORY_GB = 128


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    cpu=CPUS,
    memory=MEMORY_GB * 1024,
    timeout=6 * 60 * 60,
)
def train_bpe_remote(
    input_name: str,
    acronym: str,
    vocab_size: int,
    special_tokens: list[str],
):
    from cs336_basics.tokenizer.bpe import train_and_save_bpe

    volume.reload()
    train_and_save_bpe(
        input_path=f"{DATA_DIR}/{input_name}",
        acronym=acronym,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        cpu_core_count=CPUS,
        output_dir=DATA_DIR,
    )
    volume.commit()


@app.local_entrypoint()
def main(
    input_name: str,
    acronym: str,
    vocab_size: int,
):
    train_bpe_remote.remote(input_name, acronym, vocab_size, ["<|endoftext|>"])
