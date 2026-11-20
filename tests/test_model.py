import numpy as np
import torch

from loom_policy.data import Samples, collate
from loom_policy.model import LoomPolicy, pad_groups


def test_pad_groups():
    values = torch.arange(5.)[:, None]
    padded, mask = pad_groups(values, torch.tensor([2, 0, 2, 2, 0]), 3)
    assert padded[..., 0].tolist() == [[1, 4, 0], [0, 0, 0], [0, 2, 3]]
    assert mask.tolist() == [[True, True, False], [False] * 3, [True] * 3]


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
    loss, actions = model(collate(drawn))
    assert actions.shape == (3, 8, 20) and loss.item() > 0
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


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
        latents = torch.cat((model.encode(collate([{'segments': current}])), cached))
        kinds = torch.tensor([0, 1, 1])
        split = model.decode(latents, torch.zeros(3, dtype=torch.long), kinds, 1)
        torch.testing.assert_close(split * model.action_std + model.action_mean, full, atol=1e-5, rtol=1e-4)
