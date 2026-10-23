from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from loom_policy import Event, Segment, Query, PolicyInput, LoomPolicy, ModelConfig


def sample():
    return PolicyInput([
        Segment("video", "human-video", [Event(100, "camera", "image", torch.ones(3, 16, 16) * .2)]),
        Segment("robot", "robot-episode", [Event(2, "camera", "image", torch.ones(3, 16, 16) * .4),
                Event(2, "state", "state", np.ones(26, dtype=np.float32)),
                Event(1, "instruction", "text", "ab方")])], Query("robot", 2, (.02, .1)))


def model(**kwargs):
    torch.manual_seed(4)
    return LoomPolicy(ModelConfig(d_model=16, heads=2, layers=2, patch_size=8, **kwargs))


def test_visibility_future_values_and_modalities_cannot_change_prediction():
    m = model().eval()
    a = sample()
    dirty = copy.deepcopy(a)
    dirty.segments[-1].events.extend([Event(2.1, "future", "state", np.full(26, np.nan)),
                                      Event(3, "future", "text", "secret answer"),
                                      Event(4, "future", "image", torch.full((3, 16, 16), 99))])
    with torch.no_grad():
        noise = [torch.randn(2, 20)]
        torch.testing.assert_close(m([a], noise, [.4])[0], m([dirty], noise, [.4])[0], atol=0, rtol=0)
        torch.testing.assert_close(m.sample_actions([a], initial_noise=noise)[0],
                                   m.sample_actions([dirty], initial_noise=noise)[0], atol=0, rtol=0)


def test_padded_batch_matches_singleton_and_has_all_modality_gradients():
    m = model().eval()
    a = sample()
    b = PolicyInput([Segment("only", "goal", [Event(0, "cam", "image", torch.rand(3, 16, 16))])], Query("only", 0, (.1,)))
    noise = [torch.randn(2, 20), torch.randn(1, 20)]
    batch = m([a, b], noise, [.2, .7])
    torch.testing.assert_close(batch[0], m([a], noise[:1], [.2])[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batch[1], m([b], noise[1:], [.7])[0], atol=1e-6, rtol=1e-5)
    assert batch[0].shape == (2, 20) and batch[1].shape == (1, 20)
    batch[0].square().sum().backward()
    for parameter in (m.patch.weight, m.state[0].weight, m.text.weight, m.meta[0].weight,
                      m.flow_head.noisy_action.weight, m.flow_head.time[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_physical_time_translation_and_text_order():
    m = model().eval()
    a = sample()
    shifted = PolicyInput([replace(s, events=[replace(e, time=e.time + 10000) for e in s.events])
                           for s in a.segments], replace(a.query, time=a.query.time + 10000))
    with torch.no_grad():
        noise = [torch.randn(2, 20)]
        torch.testing.assert_close(m([a], noise, [.5])[0], m([shifted], noise, [.5])[0], atol=1e-5, rtol=1e-5)
        swapped_words = copy.deepcopy(a)
        swapped_words.segments[-1].events[-1] = replace(a.segments[-1].events[-1], value="ba方")
        assert not torch.allclose(m([a], noise, [.5])[0], m([swapped_words], noise, [.5])[0])


def test_budget_failures():
    with pytest.raises(ValueError, match="max_tokens"):
        model(max_tokens=3)([sample()], [torch.zeros(2, 20)], [.5])
    with pytest.raises(ValueError, match="divisible"):
        model()([PolicyInput([Segment("a", "source", [Event(0, "cam", "image", torch.zeros(3, 15, 16))])], Query("a", 0, (.1,)))],
                [torch.zeros(1, 20)], [.5])
