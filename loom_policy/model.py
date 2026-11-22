"""Reader + shared transformer + action slots.

1. Every segment goes through the same reader (Perceiver IO): its image patches,
   state tokens and text bytes are read by cross-attention. The current segment
   and the set of references each get one latent budget per sample.
2. The latents of all segments of a sample plus `horizon` action slots go through
   one full-attention transformer.
3. Each action slot regresses the action of one future frame; the loss is L1.

Trunk cost does not depend on reference length, and the latents of a reference
do not depend on the current segment, so deployment encodes a reference once.

Token facts: frame offset (action slot k is offset +k), camera, segment kind;
plus what a token needs to be read at all (modality, patch position, byte position).
"""
from collections import defaultdict
import math

import torch
from torch import nn
import torch.nn.functional as F

from loom_policy.data import REFERENCE
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
        self.prefix_proj = nn.Linear(FEATURE_DIM, dim)
        width = config.trunk_dim
        self.trunk_in = nn.Identity() if dim == width else nn.Linear(dim, width)
        self.trunk = nn.ModuleList(Block(width, config.trunk_heads) for _ in range(config.trunk_layers))
        self.trunk_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, FEATURE_DIM))

        for name in ('input_mean', 'input_std', 'action_mean', 'action_std'):
            value = torch.tensor(stats[name]) if stats else torch.full((FEATURE_DIM,), float(name.endswith('std')))
            self.register_buffer(name, value.float())

    def _reader_groups(self, seg_kind, seg_sample):
        """Group segments by kind and actual query count, splitting each sample's reference budget."""
        budget = self.config.num_latents
        groups, references = defaultdict(list), defaultdict(list)
        for index, (kind, sample) in enumerate(zip(seg_kind.tolist(), seg_sample.tolist(), strict=True)):
            if kind == REFERENCE and self.config.share_reference_budget:
                references[sample].append(index)
            else:
                groups[kind, budget].append(index)
        for indices in references.values():
            count, extra = divmod(budget, len(indices))
            if count == 0:
                raise ValueError(f'{len(indices)} references exceed the shared budget of {budget} latents')
            for rank, index in enumerate(indices):
                groups[REFERENCE, count + (rank < extra)].append(index)
        return groups

    def encode(self, batch):
        """One [query count, dim] tensor per segment, in input order. Legacy checkpoints return a stack."""
        kind = batch['seg_kind']
        tokens, group = [], []
        base_grid = tuple(size // self.config.patch for size in self.config.image_size)
        for views in batch['image_groups']:
            image = self.patch(views['pixels'].float() / 127.5 - 1)
            position = self.patch_position
            if image.shape[-2:] != base_grid:
                grid = position.T.reshape(1, self.config.dim, *base_grid)
                position = F.interpolate(grid, size=image.shape[-2:], mode='bicubic',
                                         align_corners=False).flatten(2)[0].T
            image = image.flatten(2).transpose(1, 2)
            fact = self.offset(views['off']) + self.camera(views['cam']) + self.modality.weight[0]
            tokens.append((image + position + fact[:, None]).flatten(0, 1))
            group.append(views['seg'].repeat_interleave(image.shape[1]))
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
        latents = [None] * len(kind)
        for (_, count), indices in self._reader_groups(kind, batch['seg_sample']).items():
            segments = kind.new_tensor(indices)
            renumber = torch.full_like(kind, -1)
            renumber[segments] = torch.arange(len(indices), device=kind.device)
            local_group = renumber[group]
            chosen = local_group >= 0
            padded, mask = pad_groups(tokens[chosen], local_group[chosen], len(indices))
            read = self.latents[:count].expand(len(indices), -1, -1)
            for cross, *layers in self.reader:
                read = cross(read, padded, mask)
                for layer in layers:
                    read = layer(read)
            for index, result in zip(indices, self.reader_norm(read).unbind(), strict=True):
                latents[index] = result
        return tuple(latents) if self.config.share_reference_budget else torch.stack(latents)

    def decode(self, latents, seg_sample, seg_kind, samples, prefix_norm=None, prefix_mask=None):
        """Decode actions, optionally replacing slots selected by prefix_mask with normalized actions."""
        horizon = self.config.horizon
        counts = seg_sample.new_tensor([len(segment) for segment in latents])
        latents = torch.cat(tuple(latents))
        segment = torch.repeat_interleave(torch.arange(len(counts), device=latents.device), counts,
                                          output_size=len(latents))
        position = torch.arange(len(latents), device=latents.device) - (counts.cumsum(0) - counts)[segment]
        latents = latents + self.latent_position[position] + self.latent_kind(seg_kind)[segment]
        latents, mask = pad_groups(latents, seg_sample[segment], samples)
        content = self.slot.expand(samples, horizon, -1)
        if prefix_norm is not None:
            content = torch.where(prefix_mask[..., None], self.prefix_proj(prefix_norm), content)
        slots = content + self.offset(torch.arange(1, horizon + 1, device=latents.device))
        x = self.trunk_in(torch.cat((latents, slots), 1))
        mask = torch.cat((mask, mask.new_ones(samples, horizon)), 1)
        for layer in self.trunk:
            x = layer(x, mask=mask)
        return self.head(self.trunk_norm(x[:, -horizon:]))

    def forward(self, batch):
        """Collated batch -> (L1 loss in normalized space, actions [B, horizon, 20] in feature units)."""
        samples = int(batch['seg_sample'].max()) + 1
        target = None
        prefix_norm = prefix_mask = None
        if 'target' in batch:
            target = (batch['target'] - self.action_mean) / self.action_std
            if self.training:
                max_len = min(self.config.max_prefix_len, self.config.horizon - 1)
                lengths = torch.randint(max_len + 1, (samples,), device=target.device)
                prefix_mask = torch.arange(self.config.horizon, device=target.device) < lengths[:, None]
                prefix_norm = target.masked_fill(~prefix_mask[..., None], 0)
        prediction = self.decode(self.encode(batch), batch['seg_sample'], batch['seg_kind'], samples,
                                 prefix_norm, prefix_mask).float()
        actions = prediction * self.action_std + self.action_mean
        if target is None:
            return None, actions
        if prefix_mask is not None:
            error = (prediction - target).abs().mean(dim=-1)
            return error.masked_fill(prefix_mask, 0).sum() / (~prefix_mask).sum(), actions
        return F.l1_loss(prediction, target), actions
