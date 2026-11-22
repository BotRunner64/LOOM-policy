"""Deployment wrapper: encode references once, then predict action chunks from the current observation."""
import numpy as np
import torch

from loom_policy.data import CAMERAS, CURRENT, REFERENCE, batch_to, collate, make_segment, resize
from loom_policy.eval import load_model
from loom_policy.geometry import STATE_DIM, compose, relative


class Policy:
    def __init__(self, checkpoint, device='cuda'):
        self.device = torch.device(device)
        self.model = load_model(checkpoint, self.device)
        self.config = self.model.config
        self.references = None

    def segment(self, kind, states, images, cameras, instruction, previous=None):
        """Observed frames, obs_stride apart and oldest first, as a model segment.

        states [F, 26] contract states, images [F][C] of H x W x 3 uint8, cameras: C names
        from CAMERAS, previous: the state one obs_stride before states[0], if known.
        """
        states = np.asarray(states, dtype=np.float64)
        features = states
        if not self.config.absolute_state:
            before = np.concatenate(([states[0] if previous is None else previous], states[:-1]))
            features = relative(before, states)
        offsets = (np.arange(len(states)) - len(states) + 1) * self.config.obs_stride
        patch = self.config.patch if self.config.adaptive_rgb else None
        images = [[resize(view, self.config.image_size, patch) for view in frame] for frame in images]
        return make_segment(kind, features, offsets, [(offsets[0], instruction)], images,
                            [CAMERAS.index(name) for name in cameras], self.config.max_text_bytes)

    @torch.no_grad()
    def encode(self, segments):
        batch = collate([{'segments': segments}])
        batch = batch_to(batch, self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            return self.model.encode(batch)

    def set_references(self, references):
        """references: list of dicts with the arguments of `segment` (states, images, cameras, instruction)."""
        segments = [self.segment(REFERENCE, **reference) for reference in references]
        self.references = self.encode(segments) if segments else None

    @torch.no_grad()
    def act(self, states, images, cameras, instruction, previous=None, *, prefix_states=None):
        """Predict t+1..t+horizon, preserving the supplied [d, 26] committed targets at t+1..t+d."""
        prefix_norm = prefix_mask = None
        length = 0
        if prefix_states is not None:
            prefix_states = np.asarray(prefix_states)
            if prefix_states.ndim != 2 or prefix_states.shape[1] != STATE_DIM:
                raise ValueError('prefix_states must have shape [d, 26]')
            length = len(prefix_states)
            if length > self.config.horizon:
                raise ValueError('prefix_states exceeds the action horizon')
            if length == self.config.horizon:
                return prefix_states.copy()
            if length:
                prefix = torch.as_tensor(relative(states[-1], prefix_states), device=self.device)
                normalized = (prefix - self.model.action_mean) / self.model.action_std
                prefix_norm = normalized.new_zeros(1, self.config.horizon, normalized.shape[-1])
                prefix_norm[0, :length] = normalized
                prefix_mask = torch.arange(self.config.horizon, device=self.device)[None, :] < length
        latents = self.encode([self.segment(CURRENT, states, images, cameras, instruction, previous)])
        kinds = [CURRENT]
        if self.references is not None:
            latents = (*latents, *self.references)
            kinds += [REFERENCE] * len(self.references)
        kinds = torch.tensor(kinds, device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            actions = self.model.decode(latents, torch.zeros_like(kinds), kinds, 1, prefix_norm, prefix_mask).float()
        actions = actions * self.model.action_std + self.model.action_mean
        suffix = compose(states[-1], actions[0, length:].cpu().numpy())
        return np.concatenate((prefix_states, suffix)) if length else suffix
