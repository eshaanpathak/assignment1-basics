import argparse
from dataclasses import asdict, dataclass, field, fields
from types import UnionType
from typing import Union, get_args

import numpy as np


@dataclass
class DataConfig:
    train_path: str = "data/ts_train.bin"
    valid_path: str = "data/ts_valid.bin"
    token_dtype: str = "uint16"

    def __post_init__(self):
        np.dtype(self.token_dtype)  # raises TypeError on an unknown dtype name


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int | None = None  # derived from d_model when left as None
    rope_theta: float = 10_000.0

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        if self.d_ff is None:
            self.d_ff = 64 * round(8 * self.d_model / 3 / 64)


@dataclass
class OptimConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    max_iters: int = 40_000  # also the end of the cosine schedule
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    batch_size: int = 32

    def __post_init__(self):
        assert self.min_lr <= self.lr, "min_lr must not exceed lr"
        assert self.warmup_iters < self.max_iters, "warmup_iterers"


@dataclass
class TrainConfig:
    eval_interval: int = 500
    eval_iters: int = 20
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    resume_from: str | None = None
    run_name: str = "run"
    wandb: bool = False
    wandb_project: str = "cs336-a1"
    device: str = "cpu"
    seed: int = 0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        max_id = np.iinfo(np.dtype(self.data.token_dtype)).max
        assert self.model.vocab_size - 1 <= max_id, "token_dtyp"

    def tokens_per_step(self) -> int:
        return self.optim.batch_size * self.model.context_length

    def total_tokens(self) -> int:
        return self.optim.max_iters * self.tokens_per_step()

    def to_dict(self) -> dict:
        return asdict(self)


def _base_type(t):
    """Strip `X | None` down to X."""
    if isinstance(t, UnionType) or getattr(t, "__origin__", None) is Union:
        non_none = [a for a in get_args(t) if a is not type(None)]
        return non_none[0]
    return t

def _add_section_args(parser: argparse.ArgumentParser, section_name: str, section_cls) -> None:
    for f in fields(section_cls):
        flag = f"--{section_name}.{f.name}"
        base = _base_type(f.type)
        if base is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=base, default=f.default)

def parse_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser()
    sections = {f.name: f.type for f in fields(Config)}
    for name, cls in sections.items():
        _add_section_args(parser, name, cls)
    args = vars(parser.parse_args(argv))
    built = {name: cls(**{f.name: args[f"{name}.{f.name}"] for f in fields(cls)}) for name, cls in sections.items()}
    return Config(**built)
