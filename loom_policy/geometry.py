"""Body-frame SE(3); export rotations are flattened COLUMN-major."""
import numpy as np


def pose_to_matrix(pose):
    p = np.asarray(pose, dtype=np.float64)
    if p.shape[-1:] != (12,) or not np.isfinite(p).all():
        raise ValueError("pose must be finite [...,12]")
    out = np.zeros(p.shape[:-1] + (4, 4), dtype=np.float64)
    out[..., :3, :3] = p[..., 3:].reshape(p.shape[:-1] + (3, 3)).swapaxes(-1, -2)
    out[..., :3, 3] = p[..., :3]
    out[..., 3, 3] = 1
    r = out[..., :3, :3]
    if not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(r), 1, atol=1e-4):
        raise ValueError("pose rotation must be in SO(3)")
    return out


def matrix_to_pose(t):
    t = np.asarray(t)
    return np.concatenate((t[..., :3, 3], t[..., :3, :3].swapaxes(-1, -2).reshape(t.shape[:-2] + (9,))), axis=-1)


def relative(anchor, pose):
    """anchor^-1 @ pose. Each arm has its own unrelated fixed world frame."""
    a, p = np.asarray(anchor), np.asarray(pose)
    out = np.zeros(np.broadcast_shapes(a.shape, p.shape))
    r = a[..., :3, :3].swapaxes(-1, -2)
    out[..., :3, :3] = r @ p[..., :3, :3]
    out[..., :3, 3] = (r @ (p[..., :3, 3] - a[..., :3, 3])[..., None])[..., 0]
    out[..., 3, 3] = 1
    return out


def transform_to_action(t, openness):
    """Per arm: local translation(3), first two rotation columns(6), gripper(1)."""
    return np.concatenate((t[:3, 3], t[:3, :2].T.reshape(6), [openness])).astype(np.float32)


def rotation6d_to_matrix(x):
    """Gram-Schmidt with deterministic fallbacks for degenerate predictions."""
    x = np.asarray(x, dtype=np.float64).reshape(2, 3)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite rotation prediction")
    a = x[0]
    a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-8 else np.array([1., 0., 0.])
    b = x[1] - np.dot(a, x[1]) * a
    if np.linalg.norm(b) < 1e-8:
        basis = np.eye(3)[np.argmin(np.abs(a))]
        b = basis - np.dot(a, basis) * a
    b /= np.linalg.norm(b)
    return np.stack((a, b, np.cross(a, b)), axis=1)


def decode_actions(actions, anchors):
    """[H,20] -> export-format absolute target poses + clipped openness.

    Every horizon shares the same observation anchor (no action accumulation).
    anchors contains separate left/right 4x4 matrices measured at current time.
    """
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != 20 or not np.isfinite(a).all():
        raise ValueError("actions must be finite [H,20]")
    result = {}
    for side, offset in (("left", 0), ("right", 10)):
        poses = []
        for row in a[:, offset:offset + 10]:
            delta = np.eye(4)
            delta[:3, 3] = row[:3]
            delta[:3, :3] = rotation6d_to_matrix(row[3:9])
            poses.append(matrix_to_pose(anchors[side] @ delta))
        result[f"{side}_eef"] = np.asarray(poses).tolist()
        result[f"{side}_gripper"] = a[:, offset + 9].clip(0, 1).tolist()
    return result
