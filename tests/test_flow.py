import copy
import types
import pytest
import torch
from loom_policy.flow import conditional_path, masked_velocity_loss
from test_model import model, sample


def test_linear_path_endpoints_and_exact_conditional_velocity():
    target = torch.tensor([[2., -1.]])
    noise = torch.tensor([[-3., 4.]])
    mask = torch.ones_like(target, dtype=torch.bool)
    for t in (0., .25, 1.):
        state, velocity = conditional_path(target, mask, noise, t)
        torch.testing.assert_close(state, (1 - t) * noise + t * target)
        torch.testing.assert_close(velocity, target - noise)
    # A correct oracle velocity gives zero loss throughout the interpolant.
    assert masked_velocity_loss([target - noise], [velocity], [mask]) == 0


def test_masked_targets_cannot_enter_noisy_conditioning_or_loss():
    target = torch.tensor([[2., float("nan")]])
    noise = torch.tensor([[.1, -.7]])
    mask = torch.tensor([[True, False]])
    state, velocity = conditional_path(target, mask, noise, .5)
    torch.testing.assert_close(state, torch.tensor([[1.05, -.7]]))
    torch.testing.assert_close(velocity, torch.tensor([[1.9, 0.]]))
    pred = torch.tensor([[2., 5.]], requires_grad=True)
    loss = masked_velocity_loss([pred], [velocity], [mask])
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[.2, 0.]]))


def test_joint_head_poisoned_missing_targets_do_not_change_valid_loss():
    m = model().eval()
    clean = {"input": sample(), "target": torch.randn(2, 20), "mask": torch.ones(2, 20, dtype=torch.bool)}
    clean["mask"][1, 10:] = False
    dirty = copy.deepcopy(clean)
    dirty["target"][~dirty["mask"]] = float("nan")
    loss = m.flow_matching_loss([clean], generator=torch.Generator().manual_seed(25))
    poisoned = m.flow_matching_loss([dirty], generator=torch.Generator().manual_seed(25))
    torch.testing.assert_close(loss, poisoned, atol=0, rtol=0)
    loss.backward()
    assert m.flow_head.velocity[-1].weight.grad.abs().sum() > 0


def test_velocity_depends_on_noisy_action_and_flow_time():
    m = model().eval()
    z = torch.randn(2, 20)
    a = m([sample()], [z], [.2])[0]
    b = m([sample()], [z], [.8])[0]
    c = m([sample()], [z + .5], [.2])[0]
    assert not torch.allclose(a, b)
    assert not torch.allclose(a, c)


def test_euler_solver_direction_steps_and_joint_chunk_shape():
    m = model().eval()
    calls = []

    def constant_velocity(self, inputs, state, times):
        calls.append(times)
        return [torch.ones_like(x) * 2 for x in state]

    m.forward = types.MethodType(constant_velocity, m)
    noise = torch.randn(2, 20)
    result = m.sample_actions([sample()], steps=4, initial_noise=[noise])[0]
    torch.testing.assert_close(result, noise + 2)
    assert calls == [[0.], [.25], [.5], [.75]]


def test_flow_sampling_reproducible_for_fixed_initial_noise_or_seed():
    m = model().eval()
    a = m.sample_actions([sample()], generator=torch.Generator().manual_seed(15))[0]
    b = m.sample_actions([sample()], generator=torch.Generator().manual_seed(15))[0]
    c = m.sample_actions([sample()], generator=torch.Generator().manual_seed(16))[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert not torch.allclose(a, c)


def test_fixed_validation_generator_does_not_advance_training_rng():
    m = model().eval()
    payload = {"input": sample(), "target": torch.randn(2, 20), "mask": torch.ones(2, 20, dtype=torch.bool)}
    state = torch.get_rng_state().clone()
    m.flow_matching_loss([payload], generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(torch.get_rng_state(), state, atol=0, rtol=0)


def test_invalid_flow_times_states_and_empty_supervision_are_rejected():
    m = model().eval()
    for t in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="flow time"):
            m([sample()], [torch.randn(2, 20)], [t])
    with pytest.raises(ValueError, match="horizon"):
        m([sample()], [torch.randn(1, 20)], [.5])
    with pytest.raises(ValueError, match="positive integer"):
        m.sample_actions([sample()], steps=0)
    with pytest.raises(ValueError, match="No supervised"):
        m.flow_matching_loss([{"input": sample(), "target": torch.zeros(2, 20),
                               "mask": torch.zeros(2, 20, dtype=torch.bool)}])
    with pytest.raises(ValueError, match="eval"):
        m.train().sample_actions([sample()])
