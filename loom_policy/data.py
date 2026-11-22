"""Read export-contract datasets (LeRobot v3.0, synchronous 30 fps) and build training samples.

A sample is one current segment (observed frames ending at t), 0..max_refs reference
segments (windows of other episodes of the same dataset) and the action target
(frames t+1 .. t+horizon relative to t). A segment holds images, state features and
instruction text; every token carries its frame offset from the segment anchor,
which is the last frame of the segment.
"""
from collections import OrderedDict, defaultdict
import json
import math
from pathlib import Path
import zlib

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
    if patch is None:
        return size
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


def resize(image, size, patch=None):
    """Resize a decoded frame or a deployment RGB array with the same swscale operation."""
    if not isinstance(image, av.VideoFrame):
        image = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format='rgb24')
    height, width = rgb_size(image.height, image.width, size, patch)
    return image.to_ndarray(width=width, height=height, format='rgb24')


def read_frames(path, indices, size, patch=None):
    """Selected frames of a single camera as [N, H, W, 3] uint8."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
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


class Source:
    """One dataset root."""

    def __init__(self, root):
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
        self.cache = OrderedDict()

    def episode_id(self, episode):
        row = self.episodes[episode]
        return f"{self.root.parent.name}/{self.root.name}/{row.get('episode_id', row['episode_index'])}"

    def frames(self, episode):
        """(states [N, 26] float64, task_index [N]) of one episode; a few episodes stay cached."""
        if episode not in self.cache:
            row = self.episodes[episode]
            path = self.root / self.info['data_path'].format(
                chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
            table = pq.read_table(path, columns=[STATE_KEY, 'frame_index', 'task_index'],
                                  filters=[('episode_index', '=', row['episode_index'])])
            order = np.argsort(table['frame_index'].to_numpy())
            states = np.stack(table[STATE_KEY].to_numpy(zero_copy_only=False)).astype(np.float64)[order]
            if len(states) != row['length']:
                raise ValueError(f'{self.episode_id(episode)}: frame rows differ from episode metadata')
            self.cache[episode] = states, table['task_index'].to_numpy()[order]
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        return self.cache[episode]

    def images(self, episode, indices, size, patch=None):
        """RGB views in frame, camera order. Adaptive views can have different shapes."""
        row = self.episodes[episode]
        views = []
        for camera in self.cameras:
            key = f'observation.images.{CAMERAS[camera]}'
            path = self.root / self.info['video_path'].format(
                video_key=key, chunk_index=row[f'videos/{key}/chunk_index'],
                file_index=row[f'videos/{key}/file_index'])
            start = round(row[f'videos/{key}/from_timestamp'] * FPS)
            views.append(read_frames(path, [start + i for i in indices], size, patch))
        if patch is not None:
            return list(zip(*views, strict=True))
        return np.stack(views, axis=1)


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


class Samples(torch.utils.data.IterableDataset):
    """Endless random samples from the train or val episodes of the given datasets.

    `num_refs` fixes the number of reference segments (None draws 0..max_refs);
    `images=False` skips video decoding, for statistics.
    """

    def __init__(self, roots, config, split='train', seed=0, num_refs=None, images=True):
        self.config, self.split, self.seed, self.num_refs, self.with_images = config, split, seed, num_refs, images
        self.sources = [Source(root) for root in roots]
        # Current segments come from `split`; reference segments always from the train episodes of the
        # same dataset, so validation never reads a held-out episode as context.
        self.episodes, self.references = [], [[] for _ in self.sources]
        threshold = round(config.val_fraction * 10000)
        for number, source in enumerate(self.sources):
            for episode, row in enumerate(source.episodes):
                held_out = zlib.crc32(source.episode_id(episode).encode()) % 10000 < threshold
                if not held_out:
                    self.references[number].append((episode, row['length']))
                if row['length'] > config.horizon and held_out == (split == 'val'):
                    self.episodes.append((number, episode, row['length']))
        if not self.episodes:
            raise ValueError(f'No {split} episodes longer than {config.horizon} frames')
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
        patch = config.patch if config.adaptive_rgb else None
        images = source.images(episode, frames.tolist(), config.image_size, patch) if self.with_images else None
        return make_segment(kind, features, frames - end, texts, images, source.cameras, config.max_text_bytes)

    def sample(self, rng):
        config = self.config
        pick = int(np.searchsorted(self.cumulative, rng.integers(self.cumulative[-1]), side='right'))
        number, episode, length = self.episodes[pick]
        t = int(rng.integers(length - config.horizon))
        states, _ = self.sources[number].frames(episode)
        segments = [self.segment(CURRENT, number, episode, t, config.current_frames)]
        # Draw one spare, so that dropping the current episode still leaves `count` (no scan of the pool).
        pool = self.references[number]
        available = len(pool) - (self.split == 'train')
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

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([*np.atleast_1d(self.seed), worker.id if worker else 0])
        while True:
            yield self.sample(rng)


def statistics(roots, config):
    """Affine normalization parameters for both observation states and action chunks."""
    if config.rebot_quantile_stats:
        source = json.loads(Path(config.rebot_quantile_stats).read_text())['norm_stats']
        result = {}
        for key, group, dim in [('state', 'input', STATE_DIM), ('actions', 'action', FEATURE_DIM)]:
            lo, hi = (np.asarray(source[key][q], dtype=np.float64) for q in ('q01', 'q99'))
            if lo.shape != (dim,) or hi.shape != (dim,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
                raise ValueError(f'Invalid {key} quantile bounds in {config.rebot_quantile_stats}')
            # The ReBot exporter sets rotation bounds to [-1, 1], giving center=0 and scale=1.
            # These checkpoint buffers also feed prefix encoding and output denormalization.
            result[f'{group}_mean'] = ((hi + lo) / 2).tolist()
            result[f'{group}_std'] = ((hi - lo) / 2).tolist()
        return result
    samples = Samples(roots, config, seed=config.seed, images=False)
    rng = np.random.default_rng(config.seed)
    inputs, targets = [], []
    for _ in range(config.stats_samples):
        sample = samples.sample(rng)
        inputs += [segment['state'] for segment in sample['segments']]
        targets.append(sample['target'])
    result = {}
    for name, values in (('input', inputs), ('action', targets)):
        values = torch.cat(values).double()
        result[f'{name}_mean'] = values.mean(0).tolist()
        result[f'{name}_std'] = values.std(0).clamp_min(1e-3).tolist()
    return result
