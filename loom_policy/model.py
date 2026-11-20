"""Reader + shared transformer + action slots.

1. Every segment goes through the same reader (Perceiver IO): its image patches,
   state tokens and text bytes are read by cross-attention into L latents.
2. The latents of all segments of a sample plus `horizon` action slots go through
   one full-attention transformer.
3. Each action slot regresses the action of one future frame; the loss is L1.

Trunk cost does not depend on reference length, and the latents of a reference
do not depend on the current segment, so deployment encodes a reference once.

Token facts: frame offset (action slot k is offset +k), camera, segment kind;
plus what a token needs to be read at all (modality, patch position, byte position).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from loom_policy.geometry import FEATURE_DIM


def pad_groups(values, group, count):
    """values [T, ...] with group ids [T] -> padded [count, longest, ...] and its mask [count, longest]."""
    sizes = torch.bincount(group, minlength=count)
    order = torch.argsort(group, stable=True)
    position = torch.arange(len(group), device=group.device) - (sizes.cumsum(0) - sizes)[group[order]]
    padded = values.new_zeros(count, int(sizes.max()), *values.shape[1:])
    mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=group.device)
    padded[group[order], position] = values[order]
    mask[group[order], position] = True
    return padded, mask


class FrameOffset(nn.Module):
    """Frame offset (any real number of frames) -> embedding, by Fourier features."""

    def __init__(self, dim, bands=32):
        super().__init__()
        self.register_buffer('frequency', 2 * math.pi / torch.logspace(math.log10(2), math.log10(20000), bands),
                             persistent=False)
        self.project = nn.Linear(2 * bands, dim)

    def forward(self, offset):
        angle = offset.float()[..., None] * self.frequency
        return self.project(torch.cat((angle.sin(), angle.cos()), -1))


class Block(nn.Module):
    """Pre-LN attention + MLP; attends to `context` when given, to itself otherwise."""

    def __init__(self, dim, heads, cross=False):
        super().__init__()
        self.heads = heads
        self.norm, self.norm_context = nn.LayerNorm(dim), nn.LayerNorm(dim) if cross else None
        self.query, self.key_value, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, context=None, mask=None):
        """mask [B, N] marks the real tokens of what is attended to."""
        query = self.norm(x)
        context = query if context is None else self.norm_context(context)
        split = lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        key, value = self.key_value(context).chunk(2, -1)
        attended = F.scaled_dot_product_attention(
            split(self.query(query)), split(key), split(value),
            attn_mask=None if mask is None else mask[:, None, None])
        x = x + self.out(attended.transpose(1, 2).flatten(2))
        return x + self.mlp(self.norm_mlp(x))


class LoomPolicy(nn.Module):
    def __init__(self, config, stats=None):
        super().__init__()
        self.config = config
        dim, heads = config.dim, config.heads
        patches = (config.image_size[0] // config.patch) * (config.image_size[1] // config.patch)
        parameter = lambda *shape: nn.Parameter(torch.randn(*shape) * 0.02)

        # Tokens.
        self.patch = nn.Conv2d(3, dim, config.patch, config.patch)
        self.patch_position = parameter(patches, dim)
        self.state = nn.Sequential(nn.Linear(FEATURE_DIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.byte = nn.Embedding(256, dim)
        self.byte_position = nn.Embedding(config.max_text_bytes, dim)
        self.offset = FrameOffset(dim)
        self.modality, self.camera, self.kind = nn.Embedding(3, dim), nn.Embedding(3, dim), nn.Embedding(2, dim)

        # Reader.
        self.latents = parameter(config.num_latents, dim)
        self.reader = nn.ModuleList(
            nn.ModuleList([Block(dim, heads, cross=True), *(Block(dim, heads) for _ in range(config.reader_self_layers))])
            for _ in range(config.reader_blocks))
        self.reader_norm = nn.LayerNorm(dim)

        # Trunk.
        self.latent_position = parameter(config.num_latents, dim)
        self.latent_kind = nn.Embedding(2, dim)
        self.slot = parameter(dim)
        self.trunk = nn.ModuleList(Block(dim, heads) for _ in range(config.trunk_layers))
        self.trunk_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, FEATURE_DIM))

        for name in ('input_mean', 'input_std', 'action_mean', 'action_std'):
            value = torch.tensor(stats[name]) if stats else torch.full((FEATURE_DIM,), float(name.endswith('std')))
            self.register_buffer(name, value.float())

    def encode(self, batch):
        """Segments of a collated batch -> latents [segments, L, dim]."""
        kind = batch['seg_kind']
        tokens, group = [], []
        if 'images' in batch:
            image = self.patch(batch['images'].float() / 127.5 - 1).flatten(2).transpose(1, 2)
            fact = self.offset(batch['img_off']) + self.camera(batch['img_cam']) + self.modality.weight[0]
            tokens.append((image + self.patch_position + fact[:, None]).flatten(0, 1))
            group.append(batch['img_seg'].repeat_interleave(image.shape[1]))
        state = (batch['state'] - self.input_mean) / self.input_std
        tokens.append(self.state(state) + self.offset(batch['st_off']) + self.modality.weight[1])
        group.append(batch['st_seg'])
        if 'text' in batch:
            tokens.append(self.byte(batch['text']) + self.byte_position(batch['txt_pos'])
                          + self.offset(batch['txt_off']) + self.modality.weight[2])
            group.append(batch['txt_seg'])
        group = torch.cat(group)
        tokens = torch.cat(tokens) + self.kind(kind)[group]

        # Read the segments of each kind as one padded batch: current segments are much shorter than
        # references, and padding them to reference length would waste most of the reader's memory.
        latents, order = [], []
        for value in kind.unique():
            segments = (kind == value).nonzero()[:, 0]
            renumber = torch.zeros_like(kind).index_put_((segments,), torch.arange(len(segments), device=kind.device))
            chosen = kind[group] == value
            padded, mask = pad_groups(tokens[chosen], renumber[group[chosen]], len(segments))
            read = self.latents.expand(len(segments), -1, -1)
            for cross, *layers in self.reader:
                read = cross(read, padded, mask)
                for layer in layers:
                    read = layer(read)
            latents.append(read)
            order.append(segments)
        return self.reader_norm(torch.cat(latents)[torch.argsort(torch.cat(order))])

    def decode(self, latents, seg_sample, seg_kind, samples):
        """Latents of all segments -> normalized actions [samples, horizon, 20]."""
        horizon = self.config.horizon
        latents = latents + self.latent_position + self.latent_kind(seg_kind)[:, None]
        latents, mask = pad_groups(latents, seg_sample, samples)
        mask = mask.repeat_interleave(latents.shape[2], 1)
        slots = self.slot + self.offset(torch.arange(1, horizon + 1, device=latents.device))
        x = torch.cat((latents.flatten(1, 2), slots.expand(samples, -1, -1)), 1)
        mask = torch.cat((mask, mask.new_ones(samples, horizon)), 1)
        for layer in self.trunk:
            x = layer(x, mask=mask)
        return self.head(self.trunk_norm(x[:, -horizon:]))

    def forward(self, batch):
        """Collated batch -> (L1 loss in normalized space, actions [B, horizon, 20] in feature units)."""
        samples = int(batch['seg_sample'].max()) + 1
        prediction = self.decode(self.encode(batch), batch['seg_sample'], batch['seg_kind'], samples).float()
        actions = prediction * self.action_std + self.action_mean
        if 'target' not in batch:
            return None, actions
        return F.l1_loss(prediction, (batch['target'] - self.action_mean) / self.action_std), actions
