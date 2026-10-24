from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from torch import nn
from loom_policy import Event, Segment, PolicyInput, LoomPolicy, ModelConfig, ChunkConfig
from loom_policy.model import masked_l1_loss


def sample():
    return PolicyInput(
        current=Segment("robot", "robot-episode", [
            Event(2, "camera", "image", torch.ones(3, 16, 16) * .4),
            Event(2, "state", "state", np.ones(26, dtype=np.float32)),
            Event(1, "instruction", "text", "ab方")]), time=2,
        references=[Segment("video", "human-video", [
            Event(100, "camera", "image", torch.ones(3, 16, 16) * .2)])])


def model(**kwargs):
    torch.manual_seed(4)
    return LoomPolicy(ModelConfig(d_model=16, heads=2, layers=2, patch_size=8, **kwargs))


def test_future_observations_cannot_change_direct_prediction():
    m = model().eval()
    a = sample()
    dirty = copy.deepcopy(a)
    dirty.current.events.extend([Event(2.1, "future", "state", np.full(26, np.nan)),
                                 Event(3, "future", "text", "secret answer"),
                                 Event(4, "future", "image", torch.full((3, 16, 16), 99))])
    with torch.no_grad():
        torch.testing.assert_close(m([a]), m([dirty]), atol=0, rtol=0)


def test_padding_batch_singleton_and_all_modalities_receive_gradients():
    m = model(chunk_size=3).eval()
    a = sample()
    b = PolicyInput(Segment("only", "goal", [Event(0, "cam", "image", torch.rand(3, 16, 16))]), time=0)
    batch = m([a, b])
    assert batch.shape == (2, 3, 20)
    torch.testing.assert_close(batch[:1], m([a]), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(batch[1:], m([b]), atol=1e-6, rtol=1e-5)
    batch[0].square().sum().backward()
    for parameter in (m.patch.weight, m.state[0].weight, m.text.weight, m.meta[0].weight,
                      m.action_head[1].weight, m.action_head[-1].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_only_observation_tokens_independent_of_output_chunk_size():
    for size in (1, 8):
        m = model(chunk_size=size).eval()
        tokens, span = m.tokenize(sample())
        # Two 2x2-patch images, one state, five UTF-8 bytes; nothing appended.
        assert len(tokens) == 4 + 4 + 1 + len("ab方".encode("utf-8"))
        assert span == slice(4, len(tokens))
        assert m([sample()]).shape == (1, size, 20)


def test_pool_current_segment_after_attention_references_condition_prediction():
    m = model().eval()
    a = sample()
    b = copy.deepcopy(a)
    b.references[0].events[0] = replace(b.references[0].events[0], value=torch.ones(3, 16, 16) * .9)
    assert not torch.allclose(m([a]), m([b]))

    class IdentityEncoder(nn.Module):
        def forward(self, tokens, src_key_padding_mask):
            return tokens

    m.backbone = IdentityEncoder()
    # With mixing removed, changing reference content must not affect the
    # readout: it pools current-trajectory tokens, not reference tokens.
    torch.testing.assert_close(m([a]), m([b]), atol=0, rtol=0)


def test_relative_physical_time_and_text_order():
    m = model().eval()
    a = sample()
    shift = lambda s: replace(s, events=[replace(e, time=e.time + 10000) for e in s.events])
    shifted = PolicyInput(shift(a.current), a.time + 10000, [shift(s) for s in a.references])
    with torch.no_grad():
        torch.testing.assert_close(m([a]), m([shifted]), atol=1e-5, rtol=1e-5)
        swapped = copy.deepcopy(a)
        swapped.current.events[-1] = replace(a.current.events[-1], value="ba方")
        assert not torch.allclose(m([a]), m([swapped]))


def test_single_forward_and_deterministic_inference():
    m = model().eval()
    calls = []
    handle = m.backbone.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        first = m([sample()])
        assert len(calls) == 1
        second = m([sample()])
    handle.remove()
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_l1_masks_missing_labels_and_normalizes_valid_field_count():
    pred = torch.tensor([[[2., 3.], [4., 5.]]], requires_grad=True)
    target = torch.tensor([[1., float("nan")], [2., 1.]])
    mask = torch.tensor([[True, False], [True, True]])
    loss = masked_l1_loss(pred, [{"target": target, "mask": mask}])
    torch.testing.assert_close(loss, torch.tensor(7 / 3))
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[1 / 3, 0.], [1 / 3, 1 / 3]]]))


def test_current_observations_required_sources_distinct_and_chunk_validated():
    empty = PolicyInput(Segment("current", "one", [Event(1, "text", "text", "future")]), time=0)
    with pytest.raises(ValueError, match="visible observation"):
        model()([empty])
    a = sample()
    with pytest.raises(ValueError, match="physical source"):
        model()([replace(a, references=[replace(a.current, segment_id="reference")])])
    with pytest.raises(ValueError, match="positive integers"):
        ChunkConfig(size=0)
    with pytest.raises(ValueError, match="positive integers"):
        ModelConfig(action_interval_ns=0)
    with pytest.raises(ValueError, match="Target/mask"):
        masked_l1_loss(torch.zeros(1, 2, 20), [{"target": torch.zeros(3, 20), "mask": torch.ones(3, 20, dtype=torch.bool)}])


def test_explicit_token_budget_and_patch_size_validation():
    with pytest.raises(ValueError, match="max_tokens"):
        model(max_tokens=3)([sample()])
    with pytest.raises(ValueError, match="divisible"):
        model()([PolicyInput(Segment("a", "source", [Event(0, "cam", "image", torch.zeros(3, 15, 16))]), time=0)])
