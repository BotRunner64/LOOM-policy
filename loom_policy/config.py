"""One flat config for data, model and training; stored in every checkpoint."""
from dataclasses import asdict, dataclass, fields
import json


@dataclass
class Config:
    # Segments. All counts are in 30 fps frames.
    image_size: tuple = (192, 256)   # H, W every camera is resized to
    obs_stride: int = 10             # observation subsampling step (open question in the design)
    current_frames: int = 2          # observed frames of the current segment, ending at t
    ref_frames: int = 32             # observed frames of one reference segment, at most
    max_refs: int = 2                # training draws 0..max_refs reference segments per sample
    horizon: int = 50                # action slots: frames t+1 .. t+horizon
    max_text_bytes: int = 256        # longer instructions are an error, never truncated

    # Model.
    patch: int = 16
    dim: int = 512
    heads: int = 8
    num_latents: int = 64            # L, latents per segment (open question in the design)
    reader_blocks: int = 2           # each: one cross-attention + reader_self_layers self-attention
    reader_self_layers: int = 2
    trunk_layers: int = 8

    # Training.
    batch_size: int = 32
    steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup: int = 1000
    grad_clip: float = 1.0
    workers: int = 8
    seed: int = 0
    val_fraction: float = 0.02
    stats_samples: int = 2000        # samples used to estimate the normalization statistics
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 16
    save_every: int = 1000

    def __post_init__(self):
        self.image_size = tuple(self.image_size)
        if any(size % self.patch for size in self.image_size) or self.dim % self.heads:
            raise ValueError('image_size must be a multiple of patch, dim a multiple of heads')

    @classmethod
    def load(cls, path=None, overrides=()):
        """Defaults, then a JSON file, then `key=value` overrides (values parsed as JSON)."""
        values = json.loads(open(path, encoding='utf-8').read()) if path else {}
        for item in overrides:
            key, _, value = item.partition('=')
            values[key] = json.loads(value)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        return cls(**values)

    def to_dict(self):
        return asdict(self)
