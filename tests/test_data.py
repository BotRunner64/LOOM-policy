import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loom_policy.data import ExportDataset, ManifestDataset, SamplingConfig, load_input
from loom_policy.geometry import matrix_to_pose, relative
from loom_policy.index import build_manifest
from loom_policy.events import ChunkConfig


def test_async_contract_times_no_fps_and_no_future_input(exported, manifest):
    sample = ManifestDataset(manifest)[1]
    inputs = sample["input"]
    target = inputs.current
    reference = inputs.references[0]
    assert all(e.time <= .22 + 1e-12 for e in target.events)
    assert any(e.time > .22 for e in reference.events)
    assert [e.time for e in target.events if e.modality == "image"] == pytest.approx([0, .15])
    assert [e.value for e in target.events if e.modality == "text"] == ["pick 方块"]
    # Four streams don't share row times. At current time .22, left=.20/right=.22.
    ep = ExportDataset(exported).episode("target")
    np.testing.assert_allclose(sample["anchors"]["left"][:3, 3], [.2, 0, 0])
    np.testing.assert_allclose(sample["anchors"]["right"][:3, 3], [1.2, 0, 0])
    expected = relative(sample["anchors"]["left"], ep.streams["left_eef"][1][3])
    np.testing.assert_allclose(sample["target"][1, :3].numpy(), expected[:3, 3], atol=1e-7)
    assert not sample["mask"][-1].any()  # no extrapolation after episode end


def test_causal_join_delta_before_sampling_and_first_identity(exported):
    ep = ExportDataset(exported).episode("target")
    events = ep.state_events(220_000_000, 250_000_000)
    assert events[0].time == pytest.approx(.02)  # wait for all four native streams
    identity = matrix_to_pose(np.eye(4))
    np.testing.assert_allclose(events[0].value[:12], identity)
    np.testing.assert_allclose(events[0].value[13:25], identity)
    # At a gripper-only arrival the pose did not move: both body deltas are I.
    at_gripper = next(e for e in events if abs(e.time - .11) < 1e-12)
    np.testing.assert_allclose(at_gripper.value[:12], identity, atol=1e-14)
    thin = ep.events("target", 220_000_000, SamplingConfig(max_state_events=1))
    np.testing.assert_allclose([e.value for e in thin.events if e.modality == "state"][0], events[-1].value)


def test_future_field_mask_and_stale_anchor(exported):
    ep = ExportDataset(exported).episode("target")
    _, mask = ep.labels(220_000_000, [50_000_000, 100_000_000], 60_000_000)
    assert not mask[0, :9].any()  # left age=.07
    assert mask[0, 10:19].all()  # right age=.05
    with pytest.raises(ValueError, match="anchor"):
        ep.anchors(219_000_000, 1)


def test_duplicate_trajectory_cannot_become_reference(manifest):
    row = json.loads(manifest.read_text().splitlines()[0])
    row["references"] = [{"episode_id": "target"}]
    with pytest.raises(ValueError, match="physical source"):
        load_input(row, SamplingConfig())


def test_invalid_arrow_and_path_escape(exported):
    with pytest.raises(ValueError, match="directory"):
        from loom_policy.data import safe_child
        safe_child(exported, "../escape")
    path = exported / "episodes/target/state/left_eef.parquet"
    table = pq.read_table(path)
    wrong = pa.table({"timestamp_ns": table["timestamp_ns"],
                      "pose": pa.array(table["pose"].to_pylist(), type=pa.list_(pa.float64()))})
    pq.write_table(wrong, path)
    with pytest.raises(ValueError, match="Arrow type"):
        ExportDataset(exported).episode("target")


def test_index_and_relative_manifest_paths(exported, tmp_path):
    output = tmp_path / "index.jsonl"
    chunk = ChunkConfig(size=2)
    count = build_manifest(exported, output, episodes=["target"], stride=1, chunk=chunk,
                           references=["demo"])
    assert count > 1
    assert len(ManifestDataset(output, chunk=chunk)) == count
    row = json.loads(output.read_text().splitlines()[0])
    row["dataset"] = "export"
    output.write_text(json.dumps(row) + "\n")
    assert ManifestDataset(output, chunk=chunk)[0]["target"].shape == (2, 20)
    assert ManifestDataset(output, chunk=chunk)[0]["mask"].any()


def test_duplicate_native_timestamp_uses_last_arrival(exported):
    path = exported / "episodes/target/state/left_gripper.parquet"
    table = pq.read_table(path)
    times = table["timestamp_ns"].to_pylist()
    values = table["openness"].to_pylist()
    times.insert(3, times[2])
    values.insert(3, .99)
    pq.write_table(pa.table({"timestamp_ns": pa.array(times, type=pa.int64()),
                             "openness": pa.array(values, type=pa.float32())}), path)
    ep = ExportDataset(exported).episode("target")
    assert ep.at("left_gripper", 220_000_000, 250_000_000) == pytest.approx(.99)


def test_future_raw_pose_changes_do_not_alter_visible_state(exported):
    before = ExportDataset(exported).episode("target").state_events(220_000_000, 250_000_000)
    path = exported / "episodes/target/state/left_eef.parquet"
    table = pq.read_table(path)
    values = table["pose"].to_pylist()
    for row in values[3:]:
        row[0] += 1000
    pq.write_table(pa.table({"timestamp_ns": table["timestamp_ns"],
                             "pose": pa.array(values, type=pa.list_(pa.float64(), 12))}), path)
    after = ExportDataset(exported).episode("target").state_events(220_000_000, 250_000_000)
    np.testing.assert_array_equal(np.stack([e.value for e in before]), np.stack([e.value for e in after]))
