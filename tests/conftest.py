"""A small synthetic export-contract dataset with landscape, portrait and square cameras."""
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from loom_policy.config import Config

CAMERAS = ('top', 'left_wrist', 'right_wrist')
CAMERA_SIZES = ((48, 64), (64, 48), (32, 32))


def color(frame, camera):
    """Uniform RGB color that identifies a frame and a camera, far enough apart to survive H.264."""
    return 30 + 12 * (frame % 16), 30 + 40 * (frame // 16), 40 + 60 * camera


def identify(image):
    """(frame, camera) of an image [..., 3] or [3, ...] made by `color`."""
    image = np.asarray(image, dtype=np.float64)
    red, green, blue = (image if image.shape[0] == 3 else np.moveaxis(image, -1, 0)).reshape(3, -1).mean(1)
    return round((red - 30) / 12) + 16 * round((green - 30) / 40), round((blue - 40) / 60)


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def states(length, rng):
    """Smooth random motion of both arms as contract states [length, 26]."""
    out = np.zeros((length, 26), dtype=np.float32)
    for arm in range(2):
        position = np.cumsum(rng.normal(0, 0.002, (length, 3)), axis=0) + rng.normal(0, 0.3, 3)
        angle = np.cumsum(rng.normal(0, 0.01, length)) + rng.uniform(-3, 3)
        base = 13 * arm
        for k in range(length):
            out[k, base:base + 3] = position[k]
            out[k, base + 3:base + 12] = rotation(angle[k]).T.reshape(-1)
        out[:, base + 12] = np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, length)), 0, 1)
    return out


@pytest.fixture(scope='session')
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('export') / 'source' / 'top+left_wrist+right_wrist_64x48'
    rng = np.random.default_rng(0)
    lengths = [40, 52, 45, 60, 48]
    rows, episodes, index = [], [], 0
    for episode, length in enumerate(lengths):
        state = states(length, rng)
        tasks = [0 if k < length // 2 else 1 for k in range(length)] if episode == 0 else [episode % 2] * length
        rows.append(pa.table({
            'observation.state': pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), pa.float32()), 26),
            'timestamp': pa.array(np.arange(length) / 30, pa.float32()),
            'frame_index': np.arange(length), 'episode_index': np.full(length, episode),
            'index': index + np.arange(length), 'task_index': np.array(tasks)}))
        row = {'episode_index': episode, 'episode_id': f'ep_{episode}', 'length': length,
               'data/chunk_index': 0, 'data/file_index': 0}
        for number, camera in enumerate(CAMERAS):
            key = f'observation.images.{camera}'
            path = root / f'videos/{key}/chunk-000/file-{episode:03d}.mp4'
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('libx264', rate=30, options={'g': '2', 'crf': '10'})
                h, w = CAMERA_SIZES[number]
                stream.width, stream.height, stream.pix_fmt = w, h, 'yuv420p'
                for k in range(length):
                    image = np.full((h, w, 3), color(k, number), dtype=np.uint8)
                    container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')))
                container.mux(stream.encode())
            row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': episode,
                        f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': length / 30})
        episodes.append(row)
        index += length
    (root / 'data/chunk-000').mkdir(parents=True)
    with pq.ParquetWriter(root / 'data/chunk-000/file-000.parquet', rows[0].schema) as writer:
        for table in rows:
            writer.write_table(table)
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), root / 'meta/episodes/chunk-000/file-000.parquet')
    pq.write_table(pa.table({'task_index': [0, 1], 'task': ['pick the cup', '把杯子放下']}), root / 'meta/tasks.parquet')
    features = {'observation.state': {'dtype': 'float32', 'shape': [26]},
                **{f'observation.images.{camera}': {'dtype': 'video', 'shape': [*size, 3]}
                   for camera, size in zip(CAMERAS, CAMERA_SIZES)}}
    (root / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'fps': 30, 'features': features,
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}))
    return root


@pytest.fixture(scope='session')
def packed(dataset, tmp_path_factory):
    """The dataset's videos in the LOOM-data packed format, written straight from its contract."""
    root = tmp_path_factory.mktemp('packed')
    directory = root / dataset.parent.name / dataset.name
    (directory / 'shards').mkdir(parents=True)
    keys = [f'observation.images.{camera}' for camera in ('top', 'left_wrist', 'right_wrist')]
    episodes = pq.read_table(dataset / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    frames, rows, offset = [], [], 0
    with open(directory / 'shards/000000.bin', 'wb') as shard:
        for episode in episodes:
            streams, extradata = [], []
            for key in keys:
                path = dataset / f'videos/{key}/chunk-000/file-{episode["episode_index"]:03d}.mp4'
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    streams.append([(bytes(p), p.is_keyframe) for p in container.demux(stream) if p.size])
                    extradata.append(bytes(stream.codec_context.extradata))
            rows.append({'episode_index': episode['episode_index'], 'episode_id': episode['episode_id'],
                         'length': episode['length'], 'shard': 0,
                         **{f'{key}.extradata': data for key, data in zip(keys, extradata)}})
            for k in range(episode['length']):
                row = {'episode_index': episode['episode_index'], 'frame_index': k, 'offset': offset}
                for key, stream in zip(keys, streams):
                    data, keyframe = stream[k]
                    shard.write(data)
                    offset += len(data)
                    row[f'{key}.size'], row[f'{key}.keyframe'] = len(data), keyframe
                frames.append(row)
    pq.write_table(pa.Table.from_pylist(rows), directory / 'episodes.parquet')
    pq.write_table(pa.Table.from_pylist(frames), directory / 'frames.parquet')
    (directory / 'info.json').write_text(json.dumps({
        'format': 'loom-packed', 'version': 1, 'codec': 'h264', 'fps': 30, 'cameras': keys,
        'episodes': len(rows), 'frames': len(frames), 'shards': [{'bytes': offset, 'sha256': ''}]}))
    return root


@pytest.fixture
def config():
    return Config(image_size=(32, 48), obs_stride=3, current_frames=2, ref_frames=5, max_refs=2, horizon=8,
                  dim=32, heads=2, num_latents=4, reader_blocks=1, reader_self_layers=1,
                  trunk_dim=64, trunk_heads=4, trunk_layers=2,
                  stats_samples=20, batch_size=3, workers=0)
