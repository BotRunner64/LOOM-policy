import numpy as np
import pytest
from loom_policy.geometry import (pose_to_matrix, matrix_to_pose, relative,
                                  decode_actions, transform_to_action, rotation6d_to_matrix)


def transform(angle, position):
    t = np.eye(4)
    t[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    t[:3, 3] = position
    return t


def test_column_major_and_body_frame_world_invariance():
    a = transform(.7, [1, 2, 3])
    b = transform(-.4, [2, 1, 4])
    g = transform(1.2, [20, -3, 7])
    np.testing.assert_allclose(pose_to_matrix(matrix_to_pose(a)), a)
    np.testing.assert_allclose(matrix_to_pose(a)[3:6], a[:3, 0])
    np.testing.assert_allclose(relative(g @ a, g @ b), relative(a, b), atol=1e-14)
    # A world-axis translation difference would fail under this rotation.
    np.testing.assert_allclose(relative(a, b)[:3, 3], a[:3, :3].T @ (b[:3, 3] - a[:3, 3]))


def test_chunk_decode_uses_one_anchor_per_arm_and_horizon():
    anchors = {"left": transform(.2, [1, 0, 0]), "right": transform(-.8, [5, 0, 0])}
    delta = transform(.4, [.3, -.2, .1])
    action = np.r_[transform_to_action(delta, .6), transform_to_action(delta, .2)]
    result = decode_actions(np.stack([action, action]), anchors)
    for side in anchors:
        predicted = pose_to_matrix(result[f"{side}_eef"])
        np.testing.assert_allclose(predicted[0], anchors[side] @ delta, atol=1e-7)
        np.testing.assert_allclose(predicted[0], predicted[1])


def test_rotation_degeneracy_and_invalid_export():
    for x in [np.zeros(6), np.ones(6), [1, 0, 0, 2, 0, 0]]:
        r = rotation6d_to_matrix(x)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1)
    with pytest.raises(ValueError, match="SO"):
        pose_to_matrix(np.zeros(12))
