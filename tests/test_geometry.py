import numpy as np

from conftest import states
from loom_policy.geometry import compose, errors, relative


def test_relative_and_compose_are_inverse():
    state = states(20, np.random.default_rng(1))
    feature = relative(state[0], state[1:])
    assert feature.shape == (19, 20)
    np.testing.assert_allclose(compose(state[0], feature), state[1:], atol=1e-5)


def test_relative_ignores_the_reference_frame():
    state = states(10, np.random.default_rng(2)).astype(np.float64)
    moved = state.copy()
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    for base in (0, 13):
        rotation = state[:, base + 3:base + 12].reshape(-1, 3, 3).transpose(0, 2, 1)
        moved[:, base:base + 3] = state[:, base:base + 3] @ turn.T + [1, 2, 3]
        moved[:, base + 3:base + 12] = (turn @ rotation).transpose(0, 2, 1).reshape(-1, 9)
    np.testing.assert_allclose(relative(moved[0], moved[1:]), relative(state[0], state[1:]), atol=1e-5)


def test_identity_feature_and_errors():
    state = states(3, np.random.default_rng(3))
    same = relative(state[1], state[1])
    np.testing.assert_allclose(same.reshape(2, 10)[:, :9], [[0, 0, 0, 1, 0, 0, 0, 1, 0]] * 2, atol=1e-6)
    assert max(errors(same, same).values()) < 1e-3
    turned = same.reshape(2, 10).copy()
    turned[:, 3:9] = [0, 1, 0, -1, 0, 0]
    assert abs(errors(turned.reshape(20), same)['rotation_deg'] - 90) < 1e-3
