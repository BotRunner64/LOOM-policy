"""Relative-pose features of the 26-d contract state.

Contract state: per arm a pose12 `[x, y, z, R column-major]` (T_reference_eef) and a
gripper openness; left arm at 0:13, right arm at 13:26.

Feature (20-d, 10 per arm): `[translation 3, rotation 6D, gripper 1]` of the body-frame
transform `T_base^-1 · T_target`, with the gripper openness of the target taken as is.
The reference frame of the dataset cancels out. The same feature serves as input
(base = previous observed frame) and as action (base = frame t for the whole chunk).
"""
import numpy as np

STATE_DIM = 26
FEATURE_DIM = 20


def split(state):
    """[..., 26] -> translation [..., 2, 3], rotation [..., 2, 3, 3], gripper [..., 2]."""
    state = np.asarray(state, dtype=np.float64)
    pose = np.stack((state[..., 0:12], state[..., 13:25]), axis=-2)
    rotation = np.swapaxes(pose[..., 3:].reshape(*pose.shape[:-1], 3, 3), -1, -2)
    return pose[..., :3], rotation, state[..., [12, 25]]


def relative(base, target):
    """Features [..., 20] of `target` relative to `base`; the two broadcast against each other."""
    base_t, base_r, _ = split(base)
    target_t, target_r, gripper = split(target)
    inverse = np.swapaxes(base_r, -1, -2)
    rotation = inverse @ target_r
    translation = (inverse @ (target_t - base_t)[..., None])[..., 0]
    six = np.swapaxes(rotation[..., :, :2], -1, -2).reshape(*rotation.shape[:-2], 6)
    feature = np.concatenate((translation, six, gripper[..., None]), axis=-1)
    return feature.reshape(*feature.shape[:-2], FEATURE_DIM).astype(np.float32)


def rotation_from_6d(six):
    """[..., 6] (first two columns, need not be orthonormal) -> rotation matrices [..., 3, 3]."""
    six = np.asarray(six, dtype=np.float64)
    x = six[..., :3] / np.linalg.norm(six[..., :3], axis=-1, keepdims=True)
    z = np.cross(x, six[..., 3:])
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    return np.stack((x, np.cross(z, x), z), axis=-1)


def compose(base, feature):
    """Inverse of `relative`: contract states [..., 26] of `feature` applied to `base` [26]."""
    base_t, base_r, _ = split(base)
    feature = np.asarray(feature, dtype=np.float64).reshape(*np.shape(feature)[:-1], 2, 10)
    rotation = base_r @ rotation_from_6d(feature[..., 3:9])
    translation = base_t + (base_r @ feature[..., :3, None])[..., 0]
    columns = np.swapaxes(rotation, -1, -2).reshape(*rotation.shape[:-2], 9)
    arms = np.concatenate((translation, columns, feature[..., 9:]), axis=-1)
    return arms.reshape(*arms.shape[:-2], STATE_DIM).astype(np.float32)


def errors(prediction, target):
    """Mean physical errors between two feature arrays: translation (m), rotation (deg), gripper."""
    prediction = np.asarray(prediction, dtype=np.float64).reshape(*np.shape(prediction)[:-1], 2, 10)
    target = np.asarray(target, dtype=np.float64).reshape(prediction.shape)
    delta = np.swapaxes(rotation_from_6d(prediction[..., 3:9]), -1, -2) @ rotation_from_6d(target[..., 3:9])
    cosine = np.clip((np.trace(delta, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return {'translation_m': float(np.linalg.norm(prediction[..., :3] - target[..., :3], axis=-1).mean()),
            'rotation_deg': float(np.degrees(np.arccos(cosine)).mean()),
            'gripper': float(np.abs(prediction[..., 9] - target[..., 9]).mean())}
