from pathlib import Path

import numpy as np
import torch

from loom_policy.config import Config
from loom_policy.data import Samples, collate
from loom_policy.model import LoomPolicy, pad_groups


def test_pad_groups():
    values = torch.arange(5.)[:, None]
    padded, mask = pad_groups(values, torch.tensor([2, 0, 2, 2, 0]), 3)
    assert padded[..., 0].tolist() == [[1, 4, 0], [0, 0, 0], [0, 2, 3]]
    assert mask.tolist() == [[True, True, False], [False] * 3, [True] * 3]


def test_patch_matmul_matches_the_convolution(config):
    model = LoomPolicy(config)
    pixels = torch.randint(0, 256, (2, 3, 32, 48), dtype=torch.uint8)
    expected = model.patch(pixels.float() / 127.5 - 1).permute(0, 2, 3, 1)
    torch.testing.assert_close(model.embed_patches(pixels), expected, atol=1e-5, rtol=1e-5)


def make(dataset, config, refs):
    torch.manual_seed(0)
    samples = Samples([dataset], config)
    rng = np.random.default_rng(0)
    drawn = []
    for count in refs:
        samples.num_refs = count
        drawn.append(samples.sample(rng))
    return LoomPolicy(config).eval(), drawn


def test_forward_and_backward(dataset, config):
    model, drawn = make(dataset, config, [0, 2, 1])
    model.train()
    loss, actions = model(collate(drawn))
    assert actions.shape == (3, 8, 20) and loss.item() > 0
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_prefix_replaces_slot_content_and_only_suffix_is_supervised(dataset, config, monkeypatch):
    model, drawn = make(dataset, config, [0, 1])
    model.train()
    batch = collate(drawn)
    model.action_center.fill_(0.5)
    model.action_scale.fill_(2)
    lengths = torch.tensor([0, 4])
    monkeypatch.setattr(torch, 'randint', lambda high, size, device: lengths.to(device))
    inputs = []
    hook = model.trunk_in.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
    loss, actions = model(batch)
    hook.remove()
    offsets = model.offset(torch.arange(1, config.horizon + 1))
    target = (batch['target'] - model.action_center) / model.action_scale
    slots = inputs[0][:, -config.horizon:]
    torch.testing.assert_close(slots[0], model.slot + offsets)
    torch.testing.assert_close(slots[1, :4], model.prefix_proj(target[1, :4]) + offsets[:4])
    torch.testing.assert_close(slots[1, 4:], model.slot + offsets[4:])

    suffix = torch.arange(config.horizon) >= lengths[:, None]
    error = (actions - batch['target']).abs() / model.action_scale
    torch.testing.assert_close(loss, error[suffix].mean())
    # With the same observations and known prefix, changing suffix labels cannot change predictions.
    batch['target'][suffix] += 10
    torch.testing.assert_close(model(batch)[1], actions)


def test_a_sample_does_not_depend_on_its_batch(dataset, config):
    model, drawn = make(dataset, config, [0, 2, 1])
    with torch.no_grad():
        together = model(collate(drawn))[1]
        for i, sample in enumerate(drawn):
            torch.testing.assert_close(model(collate([sample]))[1][0], together[i], atol=1e-5, rtol=1e-4)


def test_reference_changes_the_prediction_and_can_be_encoded_once(dataset, config):
    model, (sample,) = make(dataset, config, [2])
    current, references = sample['segments'][:1], sample['segments'][1:]
    with torch.no_grad():
        full = model(collate([sample]))[1]
        alone = model(collate([{'segments': current}]))[1]
        assert (full - alone).abs().max() > 1e-4
        # Latents of the references are computed without the current segment and reused.
        cached = model.encode(collate([{'segments': references}]))
        latents = (*model.encode(collate([{'segments': current}])), *cached)
        kinds = torch.tensor([0, 1, 1])
        split = model.decode(latents, torch.zeros(3, dtype=torch.long), kinds, 1)
        torch.testing.assert_close(split * model.action_scale + model.action_center, full, atol=1e-5, rtol=1e-4)


def test_current_and_all_references_each_get_one_budget(dataset, config):
    config.num_latents = Config().num_latents
    model, drawn = make(dataset, config, [0, 1, 2, 3])
    batch = collate(drawn)
    with torch.no_grad():
        latents = model.encode(batch)
        assert [len(x) for x in latents] == [256, 256, 256, 256, 128, 128, 256, 86, 85, 85]
        trunk_masks = []
        hook = model.trunk[0].register_forward_pre_hook(
            lambda module, args, kwargs: trunk_masks.append(kwargs['mask']), with_kwargs=True)
        model.decode(latents, batch['seg_sample'], batch['seg_kind'], len(drawn))
        hook.remove()
    mask, = trunk_masks
    assert mask.shape == (4, 512 + config.horizon)
    assert mask.sum(dim=1).tolist() == [n + config.horizon for n in (256, 512, 512, 512)]


def test_default_model_is_2b():
    config = Config.load()
    assert config == Config.load(Path(__file__).resolve().parents[1] / 'configs/2b.json')
    with torch.device('meta'):
        model = LoomPolicy(config)
    assert model.latents.shape == (256, 512)
    assert model.trunk_in.weight.shape == (2560, 512)
    assert len(model.trunk) == 25 and all(layer.heads == 40 for layer in model.trunk)
    assert sum(p.numel() for p in model.parameters()) == 1_995_094_548
