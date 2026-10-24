"""Consumer of LOOM-data/docs/contract/export-contract.md.

No source-specific field names, FPS-derived timestamps or external data-hub
imports. Four native state streams are joined causally, only inside this
policy adapter; the export itself is never rewritten/resampled.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import json
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from .events import Event, Segment, ChunkConfig, PolicyInput
from .geometry import pose_to_matrix, matrix_to_pose, relative, transform_to_action

STREAMS = ("left_eef", "right_eef", "left_gripper", "right_gripper")
INT64_MAX = 2**63 - 1


def ns(value):
    if type(value) is not int or not 0 <= value <= INT64_MAX:
        raise ValueError("Timestamps must be nonnegative int64 nanoseconds")
    return value


def safe_child(root, name):
    if not isinstance(name, str) or not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("episode_id must be a single directory name")
    path = (root / "episodes" / name).resolve()
    if path.parent != (root / "episodes").resolve():
        raise ValueError("episode_id escapes dataset")
    return path


def read_table(path, value_column, expected_type):
    table = pq.read_table(path)
    for key, expected in (("timestamp_ns", pa.int64()), (value_column, expected_type)):
        if key not in table.column_names or table.schema.field(key).type != expected:
            raise ValueError(f"{path}: {key} must have Arrow type {expected}")
        if table[key].null_count:
            raise ValueError(f"{path}: null {key}")
    time = table["timestamp_ns"].to_numpy().astype(np.int64, copy=False)
    values = np.asarray(table[value_column].to_pylist())
    if not len(time) or np.any(time < 0) or np.any(time[1:] < time[:-1]):
        raise ValueError(f"{path}: timestamps must be nonempty, nonnegative, nondecreasing")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: nonfinite {value_column}")
    return time, values


class ExportDataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata = {}
        with (self.root / "episodes.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                eid = row["episode_id"]
                safe_child(self.root, eid)
                if eid in self.metadata:
                    raise ValueError(f"Duplicate episode_id: {eid}")
                cameras = row["cameras"]
                if not cameras or len(set(cameras)) != len(cameras) or any(not isinstance(c, str) or not re.fullmatch(r"[a-z0-9_]+", c) for c in cameras):
                    raise ValueError(f"Invalid cameras: {cameras}")
                instructions = row["instructions"]
                if not instructions:
                    raise ValueError("At least one instruction is required")
                end = -1
                for instruction in instructions:
                    start, stop = ns(instruction["start_ns"]), ns(instruction["end_ns"])
                    if start < end or stop <= start or not isinstance(instruction["text"], str) or not instruction["text"].strip():
                        raise ValueError("Invalid/overlapping instruction intervals")
                    end = stop
                self.metadata[eid] = row
        if not self.metadata:
            raise ValueError("Empty episodes.jsonl")
        directories = {p.name for p in (self.root / "episodes").iterdir() if p.is_dir()}
        if directories != set(self.metadata):
            raise ValueError("episodes.jsonl and episodes/ directories must correspond exactly")

    @lru_cache(maxsize=8)
    def episode(self, eid):
        row = self.metadata[eid]
        path = safe_child(self.root, eid)
        streams = {}
        for name in STREAMS:
            pose = name.endswith("eef")
            col = "pose" if pose else "openness"
            times, values = read_table(path / "state" / f"{name}.parquet", col,
                                      pa.list_(pa.float64(), 12) if pose else pa.float32())
            if pose:
                values = pose_to_matrix(values)
            elif np.any((values < 0) | (values > 1)):
                raise ValueError("openness must be in [0,1]")
            streams[name] = (times, values)
        cameras = {}
        actual = {p.name for p in (path / "rgb").iterdir() if p.suffix in (".mp4", ".parquet")}
        expected = {f"{c}.{ext}" for c in row["cameras"] for ext in ("mp4", "parquet")}
        if actual != expected:
            raise ValueError("Camera files do not match metadata")
        for camera in row["cameras"]:
            times, indices = read_table(path / "rgb" / f"{camera}.parquet", "frame_index", pa.int64())
            if not np.array_equal(indices, np.arange(len(indices))):
                raise ValueError("frame_index must be display-order indices 0..N-1")
            cameras[camera] = (times, indices)
        if min(t[0] for t, _ in list(streams.values()) + list(cameras.values())) != 0:
            raise ValueError("Earliest native stream sample must be timestamp_ns=0")
        return Episode(path, row, streams, cameras)


@dataclass
class Episode:
    path: Path
    metadata: dict
    streams: dict
    cameras: dict
    _state_cache: dict = field(default_factory=dict, repr=False)

    @property
    def source_id(self):
        return str(self.path.resolve())

    def at(self, stream, time_ns, max_age_ns, require_coverage=False):
        times, values = self.streams[stream]
        index = int(np.searchsorted(times, time_ns, side="right")) - 1
        if index < 0 or time_ns - int(times[index]) > max_age_ns:
            return None
        # Future labels may not extrapolate beyond a stream's final timestamp.
        if require_coverage and time_ns > int(times[-1]):
            return None
        return values[index]

    def anchors(self, time_ns, max_age_ns):
        result = {name: self.at(name, time_ns, max_age_ns) for name in STREAMS}
        if any(value is None for value in result.values()):
            raise ValueError("Current time has missing/stale state anchor")
        return {"left": result["left_eef"], "right": result["right_eef"]}

    def state_events(self, end_ns, max_age_ns):
        """Native union timeline, backward-as-of join; no future lookup.

        Emit only when all four streams have arrived and are fresh. Delta is
        relative to the preceding emitted composite observation, computed
        BEFORE token-budget sampling. Unchanged arms have identity motion.
        """
        if max_age_ns not in self._state_cache:
            timeline = np.unique(np.concatenate([times for times, _ in self.streams.values()]))
            valid = np.ones(len(timeline), dtype=bool)
            joined = []
            for name in STREAMS:
                times, values = self.streams[name]
                indices = np.searchsorted(times, timeline, side="right") - 1
                valid &= (indices >= 0) & (timeline - times[indices.clip(0)] <= max_age_ns)
                joined.append(values[indices.clip(0)])
            timeline = timeline[valid]
            result = []
            if len(timeline):
                left, right, lg, rg = [v[valid] for v in joined]
                previous_left = np.concatenate((left[:1], left[:-1]))
                previous_right = np.concatenate((right[:1], right[:-1]))
                states = np.concatenate((matrix_to_pose(relative(previous_left, left)), lg[:, None],
                                         matrix_to_pose(relative(previous_right, right)), rg[:, None]), axis=1).astype(np.float32)
                result = [Event(int(t) * 1e-9, "state", "state", state) for t, state in zip(timeline, states)]
            # One staleness policy per cached episode; bound memory when configs change.
            self._state_cache.clear()
            self._state_cache[max_age_ns] = (timeline, result)
        timeline, result = self._state_cache[max_age_ns]
        return result[:int(np.searchsorted(timeline, end_ns, side="right"))]

    def labels(self, time_ns, offsets_ns, max_age_ns):
        anchors = self.anchors(time_ns, max_age_ns)
        target = np.zeros((len(offsets_ns), 20), dtype=np.float32)
        mask = np.zeros_like(target, dtype=bool)
        for h, offset in enumerate(offsets_ns):
            time = time_ns + offset
            ns(time)
            for side, base in (("left", 0), ("right", 10)):
                pose = self.at(f"{side}_eef", time, max_age_ns, require_coverage=True)
                gripper = self.at(f"{side}_gripper", time, max_age_ns, require_coverage=True)
                if pose is not None:
                    target[h, base:base + 9] = transform_to_action(relative(anchors[side], pose), 0)[:9]
                    mask[h, base:base + 9] = True
                if gripper is not None:
                    target[h, base + 9] = gripper
                    mask[h, base + 9] = True
        return target, mask

    def events(self, segment_id, end_ns, sampling):
        import av
        events = self.state_events(end_ns, sampling.max_age_ns)
        events = thin(events, sampling.max_state_events)
        for camera, (times, _) in sorted(self.cameras.items()):
            wanted = thin(np.flatnonzero(times <= end_ns).tolist(), sampling.max_image_frames)
            if not wanted:
                continue
            wanted = set(wanted)
            count = 0
            with av.open(str(self.path / "rgb" / f"{camera}.mp4")) as container:
                if (len(container.streams.video) != 1 or container.streams.audio
                        or container.streams.video[0].codec_context.name != "h264"
                        or "mp4" not in container.format.name.split(",")):
                    raise ValueError("Expected one silent H.264 MP4 video stream")
                for i, frame in enumerate(container.decode(video=0)):
                    if i in wanted:
                        image = frame.to_image().resize((sampling.image_size[1], sampling.image_size[0]))
                        value = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
                        events.append(Event(int(times[i]) * 1e-9, f"camera:{camera}", "image", value))
                    count = i + 1
            if count != len(times):
                raise ValueError("Decoded frame count differs from camera index")
        for instruction in self.metadata["instructions"]:
            if instruction["start_ns"] <= end_ns:
                events.append(Event(instruction["start_ns"] * 1e-9, "instruction", "text", instruction["text"]))
        return Segment(segment_id, self.source_id, events)


def thin(sequence, limit):
    if limit is None or len(sequence) <= limit:
        return sequence
    # Sampling is an explicit data policy, not an architectural time window.
    # Always preserve the most recent event.
    return [sequence[i] for i in np.linspace(0, len(sequence) - 1, limit, dtype=int)] if limit > 1 else [sequence[-1]]


@dataclass(frozen=True)
class SamplingConfig:
    image_size: tuple[int, int] = (64, 64)
    max_image_frames: int | None = 4  # per camera per segment
    max_state_events: int | None = 32
    max_age_ns: int = 250_000_000

    def __post_init__(self):
        if len(self.image_size) != 2 or any(type(x) is not int or x < 1 for x in self.image_size):
            raise ValueError("image_size must be two positive integers")
        if self.max_age_ns < 0 or any(x is not None and x < 1 for x in (self.max_image_frames, self.max_state_events)):
            raise ValueError("Invalid sampling budget or max_age_ns")


@lru_cache(maxsize=8)
def export_dataset(root):
    """Process-local immutable dataset view; restart to consume changed exports."""
    return ExportDataset(root)


def load_input(row, sampling):
    root = export_dataset(str(Path(row["dataset"]).resolve()))
    episode = root.episode(row["episode_id"])
    time_ns = ns(row["time_ns"])
    references = []
    for i, ref in enumerate(row.get("references", [])):
        ref_data = export_dataset(str(Path(ref.get("dataset", row["dataset"])).resolve()))
        ref_ep = ref_data.episode(ref["episode_id"])
        end = ns(ref["end_ns"]) if "end_ns" in ref else max(int(t[-1]) for t, _ in list(ref_ep.streams.values()) + list(ref_ep.cameras.values()))
        references.append(ref_ep.events(f"reference_{i}", end, sampling))
    policy_input = PolicyInput(episode.events("current", time_ns, sampling), time_ns * 1e-9, references)
    policy_input.visible()  # catches target relabelled as a reference
    anchors = episode.anchors(time_ns, sampling.max_age_ns)
    return policy_input, episode, anchors


class ManifestDataset(Dataset):
    """One JSONL observation cutoff per sample; output timing is fixed by model config."""
    def __init__(self, manifest, sampling=None, chunk=None):
        self.path = Path(manifest).resolve()
        self.sampling = sampling or SamplingConfig()
        self.chunk = chunk or ChunkConfig()
        self.rows = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["dataset"] = str((self.path.parent / row["dataset"]).resolve())
                    for ref in row.get("references", []):
                        if "dataset" in ref:
                            ref["dataset"] = str((self.path.parent / ref["dataset"]).resolve())
                    self.rows.append(row)
        if not self.rows:
            raise ValueError("Empty sample manifest")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        inputs, episode, anchors = load_input(row, self.sampling)
        target, mask = episode.labels(row["time_ns"], self.chunk.offsets_ns, self.sampling.max_age_ns)
        if not mask.any():
            raise ValueError("Sample has no valid future labels")
        return {"input": inputs, "target": torch.from_numpy(target), "mask": torch.from_numpy(mask), "anchors": anchors}
