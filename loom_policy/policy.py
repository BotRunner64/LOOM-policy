"""Deployment wrapper: encode references once, then predict action chunks from the current observation."""
import numpy as np
import torch

from loom_policy.data import CAMERAS, CURRENT, REFERENCE, collate, make_segment, resize
from loom_policy.eval import load_model
from loom_policy.geometry import compose, relative


class Policy:
    def __init__(self, checkpoint, device='cuda'):
        self.device = torch.device(device)
        self.model = load_model(checkpoint, self.device)
        self.config = self.model.config
        self.references = None

    def segment(self, kind, states, images, cameras, instruction, previous=None):
        """Observed frames, obs_stride apart and oldest first, as a model segment.

        states [F, 26] contract states, images [F, C, H, W, 3] uint8, cameras: C names
        from CAMERAS, previous: the state one obs_stride before states[0], if known.
        """
        states = np.asarray(states, dtype=np.float64)
        before = np.concatenate(([states[0] if previous is None else previous], states[:-1]))
        offsets = (np.arange(len(states)) - len(states) + 1) * self.config.obs_stride
        images = np.stack([[resize(view, self.config.image_size) for view in frame] for frame in images])
        return make_segment(kind, relative(before, states), offsets, [(offsets[0], instruction)], images,
                            [CAMERAS.index(name) for name in cameras], self.config.max_text_bytes)

    @torch.no_grad()
    def encode(self, segments):
        batch = collate([{'segments': segments}])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            return self.model.encode(batch)

    def set_references(self, references):
        """references: list of dicts with the arguments of `segment` (states, images, cameras, instruction)."""
        segments = [self.segment(REFERENCE, **reference) for reference in references]
        self.references = self.encode(segments) if segments else None

    @torch.no_grad()
    def act(self, states, images, cameras, instruction, previous=None):
        """Target contract states [horizon, 26] for frames t+1..t+horizon, in the frame of states[-1]."""
        latents = self.encode([self.segment(CURRENT, states, images, cameras, instruction, previous)])
        kinds = [CURRENT]
        if self.references is not None:
            latents = torch.cat((latents, self.references))
            kinds += [REFERENCE] * len(self.references)
        kinds = torch.tensor(kinds, device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            actions = self.model.decode(latents, torch.zeros_like(kinds), kinds, 1).float()
        actions = actions * self.model.action_std + self.model.action_mean
        return compose(states[-1], actions[0].cpu().numpy())
