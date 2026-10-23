"""Conditional flow matching on a straight Gaussian-noise -> action path."""
import math
import torch
from torch import nn


class FlowMatchingHead(nn.Module):
    """Map noisy actions + flow time to query tokens; decode a velocity field.

    The queries mix with image/state/text tokens in the existing shared
    Transformer to predict a joint conditional velocity field.
    Flow time t in [0,1] is unrelated to physical event/query time in seconds.
    """
    def __init__(self, d_model, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.query_bias = nn.Parameter(torch.randn(d_model) * .02)
        self.noisy_action = nn.Linear(action_dim, d_model)
        self.register_buffer("frequencies", torch.logspace(0, 3, 16), persistent=False)
        self.time = nn.Sequential(nn.Linear(33, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.velocity = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                      nn.SiLU(), nn.Linear(d_model, action_dim))

    def condition(self, noisy_actions, flow_time):
        if noisy_actions.ndim != 2 or noisy_actions.shape[1] != self.action_dim or not torch.isfinite(noisy_actions).all():
            raise ValueError(f"Noisy actions must be finite [H,{self.action_dim}]")
        t = torch.as_tensor(flow_time, device=noisy_actions.device, dtype=noisy_actions.dtype)
        if t.numel() != 1 or not torch.isfinite(t).all() or not 0 <= t.item() <= 1:
            raise ValueError("Each action chunk needs one finite flow time in [0,1]")
        t = t.reshape(1)
        phase = 2 * math.pi * t * self.frequencies.to(noisy_actions)
        time_features = torch.cat((t, phase.sin(), phase.cos()))
        return self.query_bias + self.noisy_action(noisy_actions) + self.time(time_features)


def conditional_path(target, mask, noise, flow_time):
    """x_t=(1-t)*z+t*a, u=a-z; unknown target fields stay independent noise.

    Sanitize *before* conditioning the joint action head. A masked NaN or
    placeholder must never contaminate valid fields through attention.
    """
    if target.shape != noise.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("Target, boolean mask and noise must have identical shapes")
    if not torch.isfinite(target[mask]).all() or not torch.isfinite(noise).all():
        raise ValueError("Nonfinite supervised target or noise")
    t = torch.as_tensor(flow_time, device=target.device, dtype=target.dtype)
    if t.numel() != 1 or not torch.isfinite(t).all() or not 0 <= t.item() <= 1:
        raise ValueError("Flow time must be a scalar in [0,1]")
    endpoint = torch.where(mask, target, noise)
    velocity = endpoint - noise
    return noise + t * velocity, velocity


def masked_velocity_loss(predictions, targets, masks):
    total = predictions[0].new_zeros(())
    count = 0
    for pred, target, mask in zip(predictions, targets, masks, strict=True):
        if target.shape != pred.shape or mask.shape != pred.shape:
            raise ValueError("Velocity target/mask must match [H,20]")
        if not torch.isfinite(target[mask]).all():
            raise ValueError("Nonfinite supervised velocity")
        total = total + (pred[mask] - target[mask]).square().sum()
        count += int(mask.sum())
    if count == 0:
        raise ValueError("No supervised action fields")
    return total / count
