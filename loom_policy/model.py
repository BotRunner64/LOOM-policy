"""Observations + action output slots -> shared Transformer -> L1 action chunks."""
from dataclasses import dataclass
import torch
from torch import nn
from .events import PolicyInput, ChunkConfig
from .encoders import ViTEncoder, BertTextEncoder, sequence_positions, reset_transformer


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    heads: int = 4
    layers: int = 4
    vision_layers: int = 2
    text_layers: int = 2
    max_text_tokens: int = 512
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
                self.d_model, self.heads, self.layers, self.vision_layers, self.text_layers,
                self.max_text_tokens, self.mlp_ratio, self.patch_size,
                self.max_segments, self.max_streams, self.max_tokens)):
            raise ValueError("Model sizes and token budget must be positive integers")
        if self.d_model < 2 or self.d_model % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("d_model must be >=2 and divisible by heads; dropout must be in [0,1)")
        self.chunk  # validate the fixed output timing

    @property
    def chunk(self):
        return ChunkConfig(self.chunk_size, self.action_interval_ns)


class ActionHead(nn.Module):
    """Shared MLP: one hidden state per timestep -> the complete action vector."""
    def __init__(self, width, action_dim):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(width),
                                   nn.Linear(width, width), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU())
            for _ in range(2)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

    def forward(self, states):
        # [B,H,D] -> [B,H,action_dim]; the same head is applied to every step.
        value = self.input(states)
        for block in self.blocks:
            value = value + block(value)
        return self.output(value)


class LoomPolicy(nn.Module):
    state_dim = 26
    action_dim = 20

    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig()
        d = c.d_model
        self.image_encoder = ViTEncoder(d, c.heads, c.vision_layers, c.mlp_ratio, c.patch_size, c.dropout)
        self.text_encoder = BertTextEncoder(d, c.heads, c.text_layers, c.mlp_ratio, c.max_text_tokens, c.dropout)
        self.state = nn.Sequential(nn.Linear(self.state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.segment = nn.Embedding(c.max_segments, d)
        self.stream = nn.Embedding(c.max_streams, d)
        self.modality = nn.Embedding(4, d)  # image, state, text, action output
        self.meta = nn.Sequential(nn.Linear(4 + 3 * d, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, c.heads, c.mlp_ratio * d, c.dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, c.layers, nn.LayerNorm(d), enable_nested_tensor=False)
        self.action_head = ActionHead(d, self.action_dim)
        reset_transformer(self.backbone)

    def metadata(self, continuous, segment, stream, modality):
        n = continuous.shape[0]
        return self.meta(torch.cat((continuous, self.segment.weight[segment].expand(n, -1),
                                   self.stream.weight[stream].expand(n, -1),
                                   self.modality.weight[modality].expand(n, -1)), dim=-1))

    def tokenize(self, inputs: PolicyInput):
        c = self.config
        device, dtype = self.segment.weight.device, self.segment.weight.dtype
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

        action_segment = action_stream = None
        for index, seg in enumerate(segments):
            sid = seg_ids[index]
            current = seg.segment_id == inputs.current.segment_id
            anchor = inputs.time if current else max((e.time for e in seg.events), default=0)
            streams = sorted({e.stream_id for e in seg.events})
            count_streams = len(streams) + int(current)
            if count_streams > c.max_streams:
                raise ValueError("Too many streams; current segment also needs one action-output stream")
            stream_ids = torch.arange(count_streams, device=device)
            if self.training and c.randomize_ids:
                stream_ids = torch.randperm(c.max_streams, device=device)[:count_streams]
            mapping = dict(zip(streams, stream_ids))
            for event in seg.events:
                tau = event.time - anchor
                rid = mapping[event.stream_id]
                if event.modality == "image":
                    value = torch.as_tensor(event.value, device=device, dtype=dtype)
                    content, xy = self.image_encoder(value, token_budget=c.max_tokens - count - c.chunk_size)
                    append(content, tau, xy, sid, rid, 0)
                elif event.modality == "state":
                    value = torch.as_tensor(event.value, device=device, dtype=dtype)
                    if value.shape != (self.state_dim,) or not torch.isfinite(value).all():
                        raise ValueError("state must be finite [26]")
                    append(self.state(value[None]), tau, None, sid, rid, 1)
                else:
                    content = self.text_encoder(event.value, token_budget=c.max_tokens - count - c.chunk_size)
                    append(content, tau, None, sid, rid, 2)
            if current:
                action_segment, action_stream = sid, stream_ids[-1]
        start = count
        # Empty content, distinguished by fixed slot position. Our backbone
        # has no built-in positional encoding, so supply it explicitly.
        slots = sequence_positions(c.chunk_size, c.d_model, device, dtype)
        append(slots, 0., None, action_segment, action_stream, 3)
        return torch.cat(pieces), slice(start, count)

    def forward(self, inputs: list[PolicyInput]):
        """One forward pass -> [batch, chunk_size,20] continuous actions."""
        if not inputs:
            raise ValueError("Empty batch")
        tokenized = [self.tokenize(sample) for sample in inputs]
        longest = max(len(tokens) for tokens, _ in tokenized)
        tokens = self.segment.weight.new_zeros(len(inputs), longest, self.config.d_model)
        padding = torch.ones((len(inputs), longest), device=tokens.device, dtype=torch.bool)
        for i, (value, _) in enumerate(tokenized):
            tokens[i, :len(value)] = value
            padding[i, :len(value)] = False
        hidden = self.backbone(tokens, src_key_padding_mask=padding)
        states = torch.stack([hidden[i, span] for i, (_, span) in enumerate(tokenized)])
        return self.action_head(states)


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
