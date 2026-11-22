"""One flat config for data, model and training; stored in every checkpoint."""
from dataclasses import asdict, dataclass, fields
import json


@dataclass
class Config:
    # Segments. All counts are in 30 fps frames.
    image_size: tuple = (192, 256)   # base position grid (pixels); its area bounds each adaptive view
    adaptive_rgb: bool = True
    obs_stride: int = 10             # observation subsampling step (open question in the design)
    current_frames: int = 2          # observed frames of the current segment, ending at t
    absolute_state: bool = False     # encode the observed 26D contract state at each frame
    ref_frames: int = 32             # observed frames of one reference segment, at most
    max_refs: int = 2                # training draws 0..max_refs reference segments per sample
    horizon: int = 50                # action slots: frames t+1 .. t+horizon
    max_text_bytes: int = 256        # longer instructions are an error, never truncated

    # Model.
    patch: int = 16
    dim: int = 512
    heads: int = 8
    num_latents: int = 256           # one budget for current, another for all references together
    share_reference_budget: bool = True
    reader_blocks: int = 2           # each: one cross-attention + reader_self_layers self-attention
    reader_self_layers: int = 2
    trunk_layers: int = 25
    trunk_dim: int | None = 2560
    trunk_heads: int | None = 40

    # Training.
    max_prefix_len: int = 4          # upper bound on known actions, always leaving one frame to predict
    batch_size: int = 32
    steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 0.05
    adam_beta2: float = 0.999
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
        if type(self.max_prefix_len) is not int or self.max_prefix_len < 0 or self.horizon < 1:
            raise ValueError('max_prefix_len must be a nonnegative integer and horizon must be positive')
        if self.trunk_dim is None:
            self.trunk_dim = self.dim
        if self.trunk_heads is None:
            self.trunk_heads = self.heads
        if self.trunk_dim < 1 or self.trunk_heads < 1 or self.trunk_dim % self.trunk_heads:
            raise ValueError('trunk_dim must be positive and divisible by a positive trunk_heads')
        if self.num_latents < 1 or (self.share_reference_budget and self.max_refs > self.num_latents):
            raise ValueError('num_latents must be positive and cover at least one latent per sampled reference')
        if (len(self.image_size) != 2 or self.patch < 1 or any(size < 1 or size % self.patch for size in self.image_size)
                or self.dim % self.heads):
            raise ValueError('image_size must be a multiple of patch, dim a multiple of heads')

    @classmethod
    def from_checkpoint(cls, values):
        # Older checkpoints inherited the reader width and head count for the trunk.
        return cls(**{'adaptive_rgb': False, 'num_latents': 64, 'share_reference_budget': False,
                      'trunk_dim': None, 'trunk_heads': None, 'trunk_layers': 8, **values})

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
