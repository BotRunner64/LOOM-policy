import copy
import numpy as np
import torch
import pytest
from loom_policy.predict import predict
from loom_policy.train import train, load_checkpoint


def config(manifest, output):
    return {"manifest": str(manifest), "output": str(output),
            "model": {"d_model": 16, "heads": 2, "layers": 1, "patch_size": 8, "dropout": .1, "chunk_size": 3},
            "sampling": {"image_size": [16, 16], "max_image_frames": 2, "max_state_events": 4},
            "train": {"device": "cpu", "steps": 6, "batch_size": 1, "save_every": 2,
                      "lr": .001, "seed": 11, "cpu_threads": 2}}


def test_checkpoint_resume_exact_optimizer_data_rng_and_prediction(manifest, tmp_path):
    full_config = config(manifest, tmp_path / "full")
    full = train(full_config)
    resumed_config = config(manifest, tmp_path / "resumed")
    partial = train(resumed_config, steps_override=3)
    checkpoint = train(resumed_config, resume=partial)
    a, b = load_checkpoint(full), load_checkpoint(checkpoint)
    assert a["step"] == b["step"] == 6
    assert b["format_version"] == 3
    assert a["epoch"] == b["epoch"] and a["cursor"] == b["cursor"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    result = predict(checkpoint, manifest)
    assert result["offsets_ns"] == (50_000_000, 100_000_000, 150_000_000)
    assert result == predict(checkpoint, manifest)
    assert np.asarray(result["relative_actions"]).shape == (3, 20)
    from loom_policy.geometry import pose_to_matrix
    assert pose_to_matrix(result["absolute_targets"]["left_eef"]).shape == (3, 4, 4)
    changed = copy.deepcopy(resumed_config)
    changed["model"]["layers"] = 2
    with pytest.raises(ValueError, match="unchanged"):
        train(changed, resume=checkpoint)


def test_validation_source_overlap_rejected(manifest, tmp_path):
    cfg = config(manifest, tmp_path / "overlap")
    cfg["validation_manifest"] = str(manifest)
    with pytest.raises(ValueError, match="overlap"):
        train(cfg)
