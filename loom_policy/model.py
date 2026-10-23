"""Shallow content maps + shared metadata MLP + one Transformer.

Observation events are filtered before *any* mixing layer. Action queries
contain the flow state x_t and flow time, never future observation events.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .events import PolicyInput
from .flow import FlowMatchingHead, conditional_path, masked_velocity_loss


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
    flow_steps: int = 10

    def __post_init__(self):
        if any(x < 1 for x in (self.d_model, self.heads, self.layers, self.mlp_ratio,
                               self.patch_size, self.max_segments, self.max_streams, self.max_tokens)):
            raise ValueError("Model sizes and token budget must be positive")
        if self.d_model % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("d_model must be divisible by heads; dropout must be in [0,1)")
        if type(self.flow_steps) is not int or self.flow_steps < 1:
            raise ValueError("flow_steps must be a positive integer")


def text_positions(length, width, device, dtype):
    """Byte order, independent of event ordering and physical event time."""
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
        # Download-free, shallow UTF-8 byte baseline; replaceable by a future
        # text input map without changing event/segment/query semantics.
        self.text = nn.Embedding(256, d)
        self.segment = nn.Embedding(c.max_segments, d)
        self.stream = nn.Embedding(c.max_streams, d)
        self.modality = nn.Embedding(4, d)  # image, state, text, action query
        self.meta = nn.Sequential(nn.Linear(4 + 3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.flow_head = FlowMatchingHead(d, self.action_dim)
        layer = nn.TransformerEncoderLayer(d, c.heads, c.mlp_ratio * d, c.dropout,
                                          batch_first=True, norm_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, c.layers, nn.LayerNorm(d), enable_nested_tensor=False)
        # TransformerEncoder otherwise clones identical initial layer weights.
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

    def tokenize(self, inputs: PolicyInput, noisy_actions, flow_time):
        c = self.config
        device, dtype = self.flow_head.query_bias.device, self.flow_head.query_bias.dtype
        noisy_actions = torch.as_tensor(noisy_actions, device=device, dtype=dtype)
        if noisy_actions.shape != (len(inputs.query.offsets), self.action_dim):
            raise ValueError("Noisy action horizon must match query offsets")
        query_content = self.flow_head.condition(noisy_actions, flow_time)
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

        target_sid = None
        for index, seg in enumerate(segments):
            sid = seg_ids[index]
            target = seg.segment_id == inputs.query.segment_id
            # Target anchor=query time: preserves observation-to-query latency.
            # Reference anchor=last visible event: independent local clocks.
            anchor = inputs.query.time if target else max((e.time for e in seg.events), default=0)
            streams = sorted({e.stream_id for e in seg.events})
            if len(streams) + 1 > c.max_streams:
                raise ValueError("Too many streams; reserve one stream for the action query")
            stream_ids = torch.arange(len(streams) + 1, device=device)
            if self.training and c.randomize_ids:
                stream_ids = torch.randperm(c.max_streams, device=device)[:len(streams) + 1]
            mapping = dict(zip(streams, stream_ids))
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
            if target:
                target_sid = sid
                query_stream = stream_ids[-1]
        start = count
        for h, offset in enumerate(inputs.query.offsets):
            append(query_content[h:h + 1], offset, None, target_sid, query_stream, 3)
        return torch.cat(pieces), slice(start, count)

    def forward(self, inputs: list[PolicyInput], noisy_actions, flow_times):
        """Predict velocity, not clean actions, at the supplied flow state."""
        if not inputs:
            raise ValueError("Empty batch")
        if len(noisy_actions) != len(inputs) or len(flow_times) != len(inputs):
            raise ValueError("One noisy chunk and flow time are required per sample")
        tokenized = [self.tokenize(sample, noisy, time)
                     for sample, noisy, time in zip(inputs, noisy_actions, flow_times, strict=True)]
        longest = max(len(tokens) for tokens, _ in tokenized)
        tokens = self.flow_head.query_bias.new_zeros(len(inputs), longest, self.config.d_model)
        padding = torch.ones((len(inputs), longest), device=tokens.device, dtype=torch.bool)
        for i, (value, _) in enumerate(tokenized):
            tokens[i, :len(value)] = value
            padding[i, :len(value)] = False
        hidden = self.backbone(tokens, src_key_padding_mask=padding)
        return [self.flow_head.velocity(hidden[i, span]) for i, (_, span) in enumerate(tokenized)]

    def flow_matching_loss(self, samples, generator=None):
        """One uniformly sampled flow time per complete action chunk."""
        if not samples:
            raise ValueError("Empty batch")
        parameter = self.flow_head.query_bias
        noisy, velocities, masks, times = [], [], [], []
        for sample in samples:
            target = sample["target"].to(parameter)
            mask = sample["mask"].to(device=parameter.device, dtype=torch.bool)
            noise = torch.randn(target.shape, device=parameter.device, dtype=parameter.dtype, generator=generator)
            time = torch.rand((), device=parameter.device, dtype=parameter.dtype, generator=generator)
            state, velocity = conditional_path(target, mask, noise, time)
            noisy.append(state)
            velocities.append(velocity)
            masks.append(mask)
            times.append(time)
        predictions = self([sample["input"] for sample in samples], noisy, times)
        return masked_velocity_loss(predictions, velocities, masks)

    @torch.no_grad()
    def sample_actions(self, inputs, *, steps=None, generator=None, initial_noise=None):
        """Euler-integrate dx/dt=v_theta(obs,x,t) from Gaussian t=0 to t=1."""
        if self.training:
            raise ValueError("Call model.eval() before flow sampling")
        steps = self.config.flow_steps if steps is None else steps
        if type(steps) is not int or steps < 1:
            raise ValueError("Flow sampling steps must be a positive integer")
        if not inputs or (initial_noise is not None and len(initial_noise) != len(inputs)):
            raise ValueError("Sampling needs inputs and one initial noise chunk per sample")
        parameter = self.flow_head.query_bias
        state = []
        for i, sample in enumerate(inputs):
            shape = (len(sample.query.offsets), self.action_dim)
            noise = (torch.randn(shape, device=parameter.device, dtype=parameter.dtype, generator=generator)
                     if initial_noise is None else torch.as_tensor(initial_noise[i], device=parameter.device, dtype=parameter.dtype).clone())
            if noise.shape != shape or not torch.isfinite(noise).all():
                raise ValueError("Initial noise must be finite and match [H,20]")
            state.append(noise)
        for i in range(steps):
            velocity = self(inputs, state, [i / steps] * len(inputs))
            state = [x + v / steps for x, v in zip(state, velocity, strict=True)]
        if any(not torch.isfinite(x).all() for x in state):
            raise FloatingPointError("Nonfinite flow sample")
        return state
