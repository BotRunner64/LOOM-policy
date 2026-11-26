"""Read export-contract datasets (LeRobot v3.0, synchronous 30 fps) and build training samples.

A sample is one current segment (observed frames ending at t), 0..max_refs reference
segments (windows of other episodes of the same dataset) and the action target
(frames t+1 .. t+horizon relative to t). A segment holds images, state features and
instruction text; every token carries its frame offset from the segment anchor,
which is the last frame of the segment.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from loom_policy.geometry import FEATURE_DIM, STATE_DIM, relative

FPS = 30
STATE_KEY = 'observation.state'
CAMERAS = ('top', 'left_wrist', 'right_wrist')
CURRENT, REFERENCE = 0, 1
LOADER_THREADS = 16    # threads of each DataLoader worker reading the samples of one batch


def find_datasets(paths):
    """Dataset roots at or up to two levels below the given paths (export root / source / layout)."""
    roots = []
    for path in map(Path, paths):
        found = [p.parents[1] for pattern in ('meta/info.json', '*/meta/info.json', '*/*/meta/info.json')
                 for p in sorted(path.glob(pattern))]
        if not found:
            raise ValueError(f'{path}: no dataset (meta/info.json) found')
        roots += [root for root in found if root not in roots]
    return roots


def rgb_size(height, width, size, patch):
    """Choose a patch-aligned H x W, retaining native scale up to the configured area."""
    max_pixels = math.prod(size)
    budget = max_pixels // patch**2
    scale = min(1., math.sqrt(max_pixels / (height * width))) / patch
    rows, cols = max(1, round(height * scale)), max(1, round(width * scale))
    if rows * cols > budget:
        # Floor after scaling when rounding up would exceed the token budget.
        # Clamping each axis also covers panoramas with a one-patch short side.
        rows = min(budget, max(1, math.floor(height * scale)))
        cols = min(budget, max(1, math.floor(width * scale)))
    return rows * patch, cols * patch


def resize(image, size, patch):
    """Resize a decoded frame or a deployment RGB array with the same swscale operation."""
    if not isinstance(image, av.VideoFrame):
        image = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format='rgb24')
    height, width = rgb_size(image.height, image.width, size, patch)
    return image.to_ndarray(width=width, height=height, format='rgb24')


def read_frames(path, indices, size, patch):
    """Selected frames of a single camera as [N, H, W, 3] uint8."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Loader threads already decode many files at once; per-file codec threads only add overhead.
        stream.codec_context.thread_count = 1
        for index in indices:
            pts = round(index / FPS / stream.time_base)
            container.seek(pts, stream=stream)
            for frame in container.decode(stream):
                if frame.pts >= pts - 0.5 / FPS / stream.time_base:
                    frames.append(resize(frame, size, patch))
                    break
            else:
                raise ValueError(f'{path}: no frame {index}')
    return np.stack(frames)


class PackedVideos:
    """Videos of one dataset in the LOOM-data packed format (docs/contract/packed-contract.md there).

    Frames are read straight from large shard files by byte offset: no file is opened per sample
    and no MP4 is parsed, which is what makes random frame access fast on network filesystems.
    """

    def __init__(self, directory, source):
        self.directory = Path(directory)
        info = json.loads((self.directory / 'info.json').read_text(encoding='utf-8'))
        if info.get('format') != 'loom-packed' or info.get('version') != 1 or info.get('codec') != 'h264':
            raise ValueError(f'{directory}: not a version 1 loom-packed directory')
        self.keys = info['cameras']
        if sorted(self.keys) != sorted(f'observation.images.{CAMERAS[c]}' for c in source.cameras):
            raise ValueError(f'{directory}: cameras differ from {source.root}')
        episodes = pq.read_table(self.directory / 'episodes.parquet')
        identity = list(zip(*(episodes[c].to_pylist() for c in ('episode_index', 'episode_id', 'length'))))
        if identity != [(row['episode_index'], row['episode_id'], row['length']) for row in source.episodes]:
            raise ValueError(f'{directory}: packed from a different version of {source.root}; pack it again')
        self.shard = episodes['shard'].to_numpy()
        self.extradata = [episodes[f'{key}.extradata'].to_pylist() for key in self.keys]
        frames = pq.read_table(self.directory / 'frames.parquet',
                               columns=['offset', *(f'{key}.{field}' for key in self.keys for field in ('size', 'keyframe'))])
        if frames.num_rows != source.offsets[-1]:
            raise ValueError(f'{directory}: frame count differs from {source.root}')
        self.offset = frames['offset'].to_numpy()
        self.sizes = np.stack([frames[f'{key}.size'].to_numpy() for key in self.keys], axis=1).astype(np.int64)
        self.keyframe = np.stack([frames[f'{key}.keyframe'].to_numpy() for key in self.keys], axis=1)
        self.offsets = source.offsets
        self._files = {}

    def _file(self, shard):
        key = os.getpid(), shard  # DataLoader workers fork; each process opens its own descriptors
        if key not in self._files:
            self._files[key] = os.open(self.directory / 'shards' / f'{shard:06d}.bin', os.O_RDONLY)
        return self._files[key]

    def frames(self, episode, frame):
        """One av.VideoFrame per camera, in packed camera order, for one frame of an episode."""
        row = self.offsets[episode] + frame
        # Each camera decodes from its own latest keyframe; the cameras need not key on the same frames.
        firsts = first_keyframes(self.keyframe, self.offsets[episode], row)
        base = self.offset[firsts.min()]
        data = os.pread(self._file(self.shard[episode]), int(self.offset[row] + self.sizes[row].sum() - base), int(base))
        decoded = []
        for camera, first in enumerate(firsts):
            codec = av.CodecContext.create('h264', 'r')
            codec.extradata = self.extradata[camera][episode]
            codec.thread_count = 1
            output = []
            for packet_row in range(first, row + 1):
                start = int(self.offset[packet_row] - base + self.sizes[packet_row, :camera].sum())
                output += codec.decode(av.Packet(data[start:start + int(self.sizes[packet_row, camera])]))
            output += codec.decode(None)
            if len(output) != row - first + 1:
                raise ValueError(f'{self.directory}: episode {episode} frame {frame} decoded {len(output)} frames')
            decoded.append(output[-1])
        return decoded

    def __getstate__(self):
        return {**self.__dict__, '_files': {}}


def first_keyframes(keyframe, start, row):
    """For each camera, the latest row in [start, row] holding a keyframe (keyframe [rows, cameras])."""
    window = keyframe[start:row + 1][::-1]
    if not window.any(axis=0).all():
        raise ValueError('an episode does not start with a keyframe')
    return row - window.argmax(axis=0)


def packed_directory(packed, root):
    """<packed>/<source>/<layout> for a dataset at <export>/<source>/<layout>."""
    directory = Path(packed) / Path(root).parent.name / Path(root).name
    if not (directory / 'info.json').is_file():
        raise ValueError(f'{root}: no packed videos at {directory}')
    return directory


class Source:
    """One dataset root. `packed` is its packed video directory, read instead of the MP4s."""

    def __init__(self, root, packed=None):
        self.root = Path(root)
        self.info = json.loads((self.root / 'meta/info.json').read_text(encoding='utf-8'))
        features = self.info['features']
        if (self.info.get('codebase_version') != 'v3.0' or self.info.get('fps') != FPS
                or features.get(STATE_KEY, {}).get('shape') != [STATE_DIM]):
            raise ValueError(f'{root}: not a contract dataset (LeRobot v3.0, {FPS} fps, {STATE_DIM}-d state)')
        self.cameras = [i for i, name in enumerate(CAMERAS) if f'observation.images.{name}' in features]
        if not self.cameras:
            raise ValueError(f'{root}: no contract cameras')
        tasks = pq.read_table(self.root / 'meta/tasks.parquet', columns=['task_index', 'task'])
        self.tasks = dict(zip(tasks['task_index'].to_pylist(), tasks['task'].to_pylist()))
        files = sorted((self.root / 'meta/episodes').rglob('*.parquet'))
        names = pq.read_schema(files[0]).names
        columns = [name for name in names if name in ('episode_index', 'episode_id', 'length')
                   or name.startswith(('data/', 'videos/'))]
        table = pa.concat_tables(pq.read_table(file, columns=columns) for file in files)
        self.episodes = sorted(table.to_pylist(), key=lambda row: row['episode_index'])
        self._load_states()
        self.packed = PackedVideos(packed, self) if packed is not None else None

    def _load_states(self):
        """Read the states and task indices of all episodes once; DataLoader workers share them after fork."""
        lengths = np.array([row['length'] for row in self.episodes])
        indices = np.array([row['episode_index'] for row in self.episodes])
        self.offsets = np.concatenate(([0], np.cumsum(lengths)))
        files = sorted({(row['data/chunk_index'], row['data/file_index']) for row in self.episodes})
        paths = [self.root / self.info['data_path'].format(chunk_index=c, file_index=f) for c, f in files]
        columns = [STATE_KEY, 'episode_index', 'frame_index', 'task_index']

        def read(path):
            table = pq.read_table(path, columns=columns)
            episode = np.searchsorted(indices, table['episode_index'].to_numpy())
            frame = table['frame_index'].to_numpy()
            if (episode >= len(indices)).any() or (indices[episode] != table['episode_index'].to_numpy()).any() \
                    or (frame < 0).any() or (frame >= lengths[episode]).any():
                raise ValueError(f'{path}: frame rows differ from episode metadata')
            states = table[STATE_KEY].combine_chunks().flatten().to_numpy(zero_copy_only=False)
            return self.offsets[episode] + frame, states.reshape(-1, STATE_DIM), table['task_index'].to_numpy()

        # Fill preallocated arrays a few files at a time, to bound the peak memory.
        self.states, self.task_index, filled = None, np.zeros(self.offsets[-1], np.int32), 0
        with ThreadPoolExecutor(8) as pool:
            for first in range(0, len(paths), 8):
                for rows, states, tasks in pool.map(read, paths[first:first + 8]):
                    if self.states is None:
                        self.states = np.full((self.offsets[-1], STATE_DIM), np.nan, states.dtype)
                    self.states[rows], self.task_index[rows] = states, tasks
                    filled += len(rows)
        if filled != self.offsets[-1] or np.isnan(self.states[:, 0]).any():
            raise ValueError(f'{self.root}: frame rows differ from episode metadata')

    def episode_id(self, episode):
        row = self.episodes[episode]
        return f"{self.root.parent.name}/{self.root.name}/{row.get('episode_id', row['episode_index'])}"

    def frames(self, episode):
        """(states [N, 26] float64, task_index [N]) of one episode."""
        start, stop = self.offsets[episode], self.offsets[episode + 1]
        return self.states[start:stop].astype(np.float64), self.task_index[start:stop]

    def images(self, episode, indices, size, patch):
        """RGB views in frame, camera order: one tuple of H x W x 3 arrays per frame; shapes can differ."""
        if self.packed is not None:
            order = [self.packed.keys.index(f'observation.images.{CAMERAS[c]}') for c in self.cameras]
            return [tuple(resize(frames[i], size, patch) for i in order)
                    for frames in (self.packed.frames(episode, index) for index in indices)]
        row = self.episodes[episode]
        views = []
        for camera in self.cameras:
            key = f'observation.images.{CAMERAS[camera]}'
            path = self.root / self.info['video_path'].format(
                video_key=key, chunk_index=row[f'videos/{key}/chunk_index'],
                file_index=row[f'videos/{key}/file_index'])
            start = round(row[f'videos/{key}/from_timestamp'] * FPS)
            views.append(read_frames(path, [start + i for i in indices], size, patch))
        return list(zip(*views, strict=True))


def make_segment(kind, features, offsets, texts, images=None, cameras=(), max_text_bytes=256):
    """One segment as tensors.

    features [F, D] absolute states or relative state features, offsets [F] frame offsets from the anchor,
    texts [(offset, instruction)], images [F][C] of H x W x 3 uint8 or None, cameras [C] ids.
    """
    encoded = []
    for offset, text in texts:
        data = text.encode('utf-8')
        if not 0 < len(data) <= max_text_bytes:
            raise ValueError(f'Instruction of {len(data)} bytes outside 1..{max_text_bytes}: {text[:80]!r}')
        encoded.append((offset, torch.frombuffer(bytearray(data), dtype=torch.uint8).long()))
    segment = {'kind': kind, 'state': torch.as_tensor(np.asarray(features, dtype=np.float32)),
               'offsets': torch.as_tensor(np.asarray(offsets, dtype=np.float32)), 'texts': encoded}
    if images is not None:
        segment['images'] = [[torch.as_tensor(view).permute(2, 0, 1) for view in frame] for frame in images]
        segment['cameras'] = list(cameras)
    return segment


def collate(samples):
    """Flatten samples into token groups; every token names its segment, every segment its sample."""
    image_groups = defaultdict(list)
    parts = {key: [] for key in ('state', 'st_seg', 'st_off',
                                 'text', 'txt_seg', 'txt_off', 'txt_pos', 'seg_sample', 'seg_kind')}
    index = 0
    for number, sample in enumerate(samples):
        for segment in sample['segments']:
            count = len(segment['state'])
            if 'images' in segment:
                for frame, offset in zip(segment['images'], segment['offsets'].tolist(), strict=True):
                    for image, camera in zip(frame, segment['cameras'], strict=True):
                        image_groups[image.shape].append((image, index, offset, camera))
            parts['state'].append(segment['state'])
            parts['st_seg'].append(torch.full((count,), index))
            parts['st_off'].append(segment['offsets'])
            for offset, data in segment['texts']:
                parts['text'].append(data)
                parts['txt_seg'].append(torch.full((len(data),), index))
                parts['txt_off'].append(torch.full((len(data),), float(offset)))
                parts['txt_pos'].append(torch.arange(len(data)))
            parts['seg_sample'].append(torch.tensor([number]))
            parts['seg_kind'].append(torch.tensor([segment['kind']]))
            index += 1
    batch = {key: torch.cat(value) for key, value in parts.items() if value}
    batch['image_groups'] = []
    for views in image_groups.values():
        pixels, segments, offsets, cameras = zip(*views)
        batch['image_groups'].append({'pixels': torch.stack(pixels), 'seg': torch.tensor(segments),
                                      'off': torch.tensor(offsets), 'cam': torch.tensor(cameras)})
    if 'target' in samples[0]:
        batch['target'] = torch.stack([sample['target'] for sample in samples])
    return batch


def batch_to(batch, device):
    """Move collated tensors and resolution-grouped views to the model device."""
    moved = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'image_groups'}
    moved['image_groups'] = [
        {k: v.to(device, non_blocking=True) for k, v in group.items()} for group in batch['image_groups']]
    return moved


class Samples(torch.utils.data.Dataset):
    """Action windows indexed by (draw position, anchor) over all episodes longer than the horizon.

    `num_refs` fixes the number of reference segments (None draws 0..max_refs);
    `images=False` skips video decoding, for statistics; `packed` is a packed export folder
    (<packed>/<source>/<layout> for each root) whose videos are read instead of the MP4s.
    """

    def __init__(self, roots, config, seed=0, num_refs=None, images=True, packed=None):
        self.config, self.seed, self.num_refs, self.with_images = config, seed, num_refs, images
        self._pool = self._pool_pid = None
        self.sources = [Source(root, packed_directory(packed, root) if packed and images else None) for root in roots]
        # Reference segments are windows of the other episodes of the same dataset.
        self.references = [[(episode, row['length']) for episode, row in enumerate(source.episodes)]
                           for source in self.sources]
        self.episodes = [(number, episode, row['length']) for number, source in enumerate(self.sources)
                         for episode, row in enumerate(source.episodes) if row['length'] > config.horizon]
        if not self.episodes:
            raise ValueError(f'No episodes longer than {config.horizon} frames')
        # Every frame t with a full horizon after it is equally likely.
        self.cumulative = np.cumsum([length - config.horizon for _, _, length in self.episodes])

    def segment(self, kind, number, episode, end, count):
        """Up to `count` observed frames, obs_stride apart, ending at frame `end`."""
        config, source = self.config, self.sources[number]
        states, tasks = source.frames(episode)
        frames = np.arange(end - (count - 1) * config.obs_stride, end + 1, config.obs_stride)
        frames = frames[frames >= 0]
        features = states[frames]
        if not config.absolute_state:
            features = relative(states[np.maximum(frames - config.obs_stride, 0)], features)
        changes = [i for i in range(len(frames)) if i == 0 or tasks[frames[i]] != tasks[frames[i - 1]]]
        texts = [(frames[i] - end, source.tasks[int(tasks[frames[i]])]) for i in changes]
        images = source.images(episode, frames.tolist(), config.image_size, config.patch) if self.with_images else None
        return make_segment(kind, features, frames - end, texts, images, source.cameras, config.max_text_bytes)

    def sample(self, rng, anchor=None):
        config = self.config
        position = rng.integers(self.cumulative[-1]) if anchor is None else anchor
        pick = int(np.searchsorted(self.cumulative, position, side='right'))
        number, episode, length = self.episodes[pick]
        start = int(self.cumulative[pick - 1]) if pick else 0
        t = int(rng.integers(length - config.horizon)) if anchor is None else anchor - start
        states, _ = self.sources[number].frames(episode)
        segments = [self.segment(CURRENT, number, episode, t, config.current_frames)]
        # Draw one spare, so that dropping the current episode still leaves `count` (no scan of the pool).
        pool = self.references[number]
        available = len(pool) - 1  # the pool holds the current episode too
        count = int(rng.integers(config.max_refs + 1)) if self.num_refs is None else self.num_refs
        if self.num_refs is not None and count > available:
            raise ValueError(f'{self.sources[number].root}: {count} references asked, {available} episodes to draw from')
        drawn = rng.choice(len(pool), size=min(count + 1, len(pool)), replace=False) if count else ()
        for reference, reference_length in [pool[i] for i in drawn if pool[i][0] != episode][:count]:
            # The window ends at the episode end when the episode fits, anywhere after a full window otherwise.
            first = min((config.ref_frames - 1) * config.obs_stride, reference_length - 1)
            end = int(rng.integers(first, reference_length))
            segments.append(self.segment(REFERENCE, number, reference, end, config.ref_frames))
        # Chunk-wise relative action: the measured states t+1..t+horizon against frame t.
        target = relative(states[t], states[t + 1:t + 1 + config.horizon])
        return {'segments': segments, 'target': torch.as_tensor(target)}

    def __getitem__(self, index):
        position, anchor = index
        return self.sample(np.random.default_rng([self.seed, position]), anchor=anchor)

    def __getitems__(self, indices):
        """Load a batch with threads: reading a sample mostly waits on file IO, not on the CPU."""
        if len(indices) == 1:
            return [self[indices[0]]]
        if self._pool_pid != os.getpid():  # each DataLoader worker builds its own pool
            self._pool, self._pool_pid = ThreadPoolExecutor(LOADER_THREADS), os.getpid()
        return list(self._pool.map(self.__getitem__, indices))

    def __getstate__(self):
        return {**self.__dict__, '_pool': None, '_pool_pid': None}


def sampled_quantiles(roots, config, samples=None):
    """Estimate raw per-feature bounds from training observations and future actions."""
    if samples is None:
        samples = Samples(roots, config, seed=config.seed, images=False)
    else:  # reuse loaded training samples, without decoding video
        samples = copy.copy(samples)
        samples.with_images = False
    rng = np.random.default_rng(config.seed)
    inputs, targets = [], []
    for _ in range(config.stats_samples):
        sample = samples.sample(rng)
        inputs += [segment['state'] for segment in sample['segments']]
        targets.append(sample['target'])
    result = {}
    for name, values in (('state', inputs), ('actions', targets)):
        lo, hi = np.quantile(torch.cat(values).numpy(), [0.01, 0.99], axis=0)
        result[name] = {'q01': lo, 'q99': hi}
    return result


def statistics(roots, config, samples=None):
    """P1/P99 affine parameters. Rotation coordinates retain their native scale.

    `samples` may pass already loaded training samples of `roots` (seed config.seed)."""
    if config.rebot_quantile_stats:
        bounds = json.loads(Path(config.rebot_quantile_stats).read_text())['norm_stats']
    else:
        bounds = sampled_quantiles(roots, config, samples)
    result = {}
    input_dim = STATE_DIM if config.absolute_state else FEATURE_DIM
    for key, group, dim in [('state', 'input', input_dim), ('actions', 'action', FEATURE_DIM)]:
        lo, hi = (np.asarray(bounds[key][q], dtype=np.float64) for q in ('q01', 'q99'))
        if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or np.any(hi < lo):
            raise ValueError(f'Invalid {key} quantile bounds')
        center, scale = (hi + lo) / 2, (hi - lo) / 2
        scale[scale < 1e-6] = 1  # Center near-constant coordinates without amplifying noise.
        # Each arm has xyz, rotation (9D absolute or 6D relative), and gripper.
        center.reshape(2, -1)[:, 3:-1] = 0
        scale.reshape(2, -1)[:, 3:-1] = 1
        result[f'{group}_center'] = center.tolist()
        result[f'{group}_scale'] = scale.tolist()
    return result
