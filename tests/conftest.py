import json
from pathlib import Path
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from loom_policy.geometry import matrix_to_pose

torch.set_num_threads(2)


def make_export(root, episodes=("demo", "target")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = []
    for eid in episodes:
        folder = root / "episodes" / eid
        (folder / "state").mkdir(parents=True)
        (folder / "rgb").mkdir()
        for name, shift in (("left_eef", 0), ("right_eef", 20_000_000),
                            ("left_gripper", 10_000_000), ("right_gripper", 0)):
            times = np.arange(5, dtype=np.int64) * 100_000_000 + shift
            if name.endswith("eef"):
                values = []
                for i in range(5):
                    t = np.eye(4)
                    angle = i * .1
                    t[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
                    t[0, 3] = i * .1 + (1 if name.startswith("right") else 0)
                    values.append(matrix_to_pose(t))
                table = pa.table({"timestamp_ns": pa.array(times, type=pa.int64()),
                                  "pose": pa.array(values, type=pa.list_(pa.float64(), 12))})
            else:
                table = pa.table({"timestamp_ns": pa.array(times, type=pa.int64()),
                                  "openness": pa.array(np.linspace(.1, .9, 5), type=pa.float32())})
            pq.write_table(table, folder / "state" / f"{name}.parquet")
        with av.open(str(folder / "rgb" / "top.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for i in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 50, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        pq.write_table(pa.table({"frame_index": pa.array(range(4), type=pa.int64()),
                                 "timestamp_ns": pa.array([0, 150_000_000, 350_000_000, 450_000_000], type=pa.int64())}),
                       folder / "rgb" / "top.parquet")
        metadata.append({"episode_id": eid, "cameras": ["top"], "instructions": [
            {"start_ns": 0, "end_ns": 300_000_000, "text": "pick 方块"},
            {"start_ns": 300_000_000, "end_ns": 450_000_000, "text": "future instruction"}]})
    (root / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in metadata), encoding="utf-8")
    return root


@pytest.fixture
def exported(tmp_path):
    return make_export(tmp_path / "export")


@pytest.fixture
def manifest(exported, tmp_path):
    path = tmp_path / "samples.jsonl"
    rows = [{"dataset": str(exported), "episode_id": "target", "time_ns": time,
             "references": [{"episode_id": "demo"}]} for time in [120_000_000, 220_000_000]]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path
