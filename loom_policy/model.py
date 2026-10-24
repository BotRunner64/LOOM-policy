"""Observation tokens -> shared Transformer -> direct action-chunk regression."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .events import PolicyInput, ChunkConfig


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    heads: int = 4
    layers: int = 4
    mlp_ratio: int = 4
    patch_size: int = 16
    dropout: float = 0.0
    max_segments: int = 16
    max_streams: int = 32
    max_tokens: int = 4096
    randomize_ids: bool = True
    chunk_size: int = 8
    action_interval_ns: int = 50_000_000

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (
                self.d_model, self.heads, self.layers, self.mlp_ratio, self.patch_size,
                self.max_segments, self.max_streams, self.max_tokens)):
            raise ValueError("Model sizes and token budget must be positive integers")
        if self.d_model % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("d_model must be divisible by heads; dropout must be in [0,1)")
        self.chunk  # validate the fixed output timing

    @property
    def chunk(self):
        return ChunkConfig(self.chunk_size, self.action_interval_ns)


def text_positions(length, width, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000) / width))
    out = torch.zeros(length, width, device=device, dtype=dtype)
    out[:, 0::2] = torch.sin(position * frequency)
    out[:, 1::2] = torch.cos(position * frequency[:width // 2])
    return out


class LoomPolicy(nn.Module):
    state_dim = 26
    action_dim = 20

    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig()
        d = c.d_model
        self.patch = nn.Conv2d(3, d, kernel_size=c.patch_size, stride=c.patch_size)
        self.state = nn.Sequential(nn.Linear(self.state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.text = nn.Embedding(256, d)
        self.segment = nn.Embedding(c.max_segments, d)
        self.stream = nn.Embedding(c.max_streams, d)
        self.modality = nn.Embedding(3, d)  # image, state, text only
        self.meta = nn.Sequential(nn.Linear(4 + 3 * d, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, c.heads, c.mlp_ratio * d, c.dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, c.layers, nn.LayerNorm(d), enable_nested_tensor=False)
        self.action_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                         nn.Linear(d, c.chunk_size * self.action_dim))
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)

    def metadata(self, continuous, segment, stream, modality):
        n = continuous.shape[0]
        return self.meta(torch.cat((continuous, self.segment.weight[segment].expand(n, -1),
                                   self.stream.weight[stream].expand(n, -1),
                                   self.modality.weight[modality].expand(n, -1)), dim=-1))

    def tokenize(self, inputs: PolicyInput):
        c = self.config
        device, dtype = self.patch.weight.device, self.patch.weight.dtype
        segments = inputs.visible()
        if len(segments) > c.max_segments:
            raise ValueError("Too many segments; increase max_segments")
        seg_ids = torch.arange(len(segments), device=device)
        if self.training and c.randomize_ids:
            seg_ids = torch.randperm(c.max_segments, device=device)[:len(segments)]
        pieces = []
        count = 0

        def append(content, tau, xy, sid, rid, modality):
            nonlocal count
            n = content.shape[0]
            count += n
            if count > c.max_tokens:
                raise ValueError(f"{count} tokens exceed max_tokens={c.max_tokens}; adjust explicit data sampling")
            continuous = torch.zeros((n, 4), device=device, dtype=dtype)
            continuous[:, 0] = tau
            if xy is not None:
                continuous[:, 1:3] = xy
                continuous[:, 3] = 1
            pieces.append(content + self.metadata(continuous, sid, rid, modality))

        current_span = None
        for index, seg in enumerate(segments):
            sid = seg_ids[index]
            current = seg.segment_id == inputs.current.segment_id
            anchor = inputs.time if current else max((e.time for e in seg.events), default=0)
            streams = sorted({e.stream_id for e in seg.events})
            if len(streams) > c.max_streams:
                raise ValueError("Too many streams; increase max_streams")
            stream_ids = torch.arange(len(streams), device=device)
            if self.training and c.randomize_ids:
                stream_ids = torch.randperm(c.max_streams, device=device)[:len(streams)]
            mapping = dict(zip(streams, stream_ids))
            start = count
            for event in seg.events:
                tau = event.time - anchor
                rid = mapping[event.stream_id]
                if event.modality == "image":
                    value = torch.as_tensor(event.value, device=device, dtype=dtype)
                    if value.ndim != 3 or value.shape[0] != 3 or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                        raise ValueError("image must be finite RGB CHW float in [0,1]")
                    if value.shape[1] % c.patch_size or value.shape[2] % c.patch_size:
                        raise ValueError("Image dimensions must be divisible by patch_size")
                    feature = self.patch(value[None])[0]
                    height, width = feature.shape[1:]
                    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
                    xy = torch.stack(((x.flatten() + .5) / width, (y.flatten() + .5) / height), dim=-1)
                    append(feature.flatten(1).T, tau, xy, sid, rid, 0)
                elif event.modality == "state":
                    value = torch.as_tensor(event.value, device=device, dtype=dtype)
                    if value.shape != (self.state_dim,) or not torch.isfinite(value).all():
                        raise ValueError("state must be finite [26]")
                    append(self.state(value[None]), tau, None, sid, rid, 1)
                else:
                    if not isinstance(event.value, str) or not event.value:
                        raise ValueError("text must be nonempty UTF-8 text")
                    ids = torch.tensor(list(event.value.encode("utf-8")), device=device, dtype=torch.long)
                    content = self.text(ids) + text_positions(len(ids), c.d_model, device, dtype)
                    append(content, tau, None, sid, rid, 2)
            if current:
                current_span = slice(start, count)
        return torch.cat(pieces), current_span

    def forward(self, inputs: list[PolicyInput]):
        """One forward pass -> [batch, chunk_size,20] continuous actions."""
        if not inputs:
            raise ValueError("Empty batch")
        tokenized = [self.tokenize(sample) for sample in inputs]
        longest = max(len(tokens) for tokens, _ in tokenized)
        tokens = self.patch.weight.new_zeros(len(inputs), longest, self.config.d_model)
        padding = torch.ones((len(inputs), longest), device=tokens.device, dtype=torch.bool)
        for i, (value, _) in enumerate(tokenized):
            tokens[i, :len(value)] = value
            padding[i, :len(value)] = False
        hidden = self.backbone(tokens, src_key_padding_mask=padding)
        # Pool real current-trajectory observations AFTER shared multimodal
        # attention. References condition them through the same backbone.
        pooled = torch.stack([hidden[i, span].mean(dim=0) for i, (_, span) in enumerate(tokenized)])
        return self.action_head(pooled).reshape(len(inputs), self.config.chunk_size, self.action_dim)


def masked_l1_loss(predictions, samples):
    """OFT-style continuous L1 objective; missing fields have no gradients."""
    if len(predictions) != len(samples) or not samples:
        raise ValueError("A nonempty aligned batch is required")
    total = predictions.new_zeros(())
    count = 0
    for pred, sample in zip(predictions, samples, strict=True):
        target = sample["target"].to(pred)
        mask = sample["mask"].to(device=pred.device, dtype=torch.bool)
        if target.shape != pred.shape or mask.shape != pred.shape:
            raise ValueError("Target/mask must match the configured [chunk_size,20]")
        if not torch.isfinite(target[mask]).all():
            raise ValueError("Nonfinite supervised target")
        total = total + (pred[mask] - target[mask]).abs().sum()
        count += int(mask.sum())
    if count == 0:
        raise ValueError("No supervised action fields")
    return total / count
