"""Optional integration against the real adjacent LOOM-data writers."""
import os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pytest
from loom_policy.data import ExportDataset


def test_actual_loom_data_writers_are_consumed(exported, monkeypatch):
    upstream = os.environ.get("LOOM_DATA_REPO")
    if not upstream:
        pytest.skip("Set LOOM_DATA_REPO to run the upstream-writer integration")
    monkeypatch.syspath_prepend(str(Path(upstream).resolve()))
    from scripts.convert.export_common import write_state, write_video_index
    folder = exported / "episodes/target"
    for name in ("left_eef", "right_eef", "left_gripper", "right_gripper"):
        path = folder / "state" / f"{name}.parquet"
        table = pq.read_table(path)
        pose = name.endswith("eef")
        write_state(path, table["timestamp_ns"].to_pylist(),
                    table["pose" if pose else "openness"].to_pylist(), 0, pose=pose)
    index = folder / "rgb/top.parquet"
    table = pq.read_table(index)
    write_video_index(index, table["timestamp_ns"].to_pylist(), 0)
    episode = ExportDataset(exported).episode("target")
    values, mask = episode.labels(220_000_000, [50_000_000], 250_000_000)
    assert values.shape == (1, 20) and mask.all()
