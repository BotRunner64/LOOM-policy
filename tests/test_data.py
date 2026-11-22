import json

import numpy as np
import pytest

from conftest import identify
from loom_policy.data import CURRENT, REFERENCE, Samples, Source, collate, find_datasets, rgb_size, statistics
from loom_policy.geometry import relative


def test_find_datasets(dataset):
    assert find_datasets([dataset]) == find_datasets([dataset.parent]) == find_datasets([dataset.parents[1]]) == [dataset]
    with pytest.raises(ValueError):
        find_datasets([dataset / 'meta'])


def test_video_frames_match_rows(dataset):
    images = Source(dataset).images(3, [0, 7, 30, 59], (48, 64))
    assert images.shape == (4, 3, 48, 64, 3)
    for i, frame in enumerate([0, 7, 30, 59]):
        for camera in range(3):
            assert identify(images[i, camera]) == (frame, camera)


@pytest.mark.parametrize('native,expected', [
    ((480, 640), (192, 256)), ((640, 480), (256, 192)), ((720, 1280), (160, 288)),
    ((1024, 1024), (208, 208)), ((96, 160), (96, 160)), ((3, 2), (16, 16)),
    ((1, 49152), (16, 3072)), ((49152, 1), (3072, 16)),
])
def test_adaptive_grid(native, expected):
    assert rgb_size(*native, (192, 256), 16) == expected


def test_segment_facts(dataset, config):
    samples = Samples([dataset], config)
    source = samples.sources[0]
    states, _ = source.frames(0)
    segment = samples.segment(REFERENCE, 0, 0, 30, 5)
    assert segment['offsets'].tolist() == [-12, -9, -6, -3, 0]
    assert len(segment['images']) == 5
    assert [view.shape for view in segment['images'][1]] == [(3, 32, 48), (3, 48, 32), (3, 32, 32)]
    assert identify(segment['images'][1][2]) == (21, 2)
    np.testing.assert_allclose(segment['state'][2], relative(states[21], states[24]))
    # Episode 0 switches instruction at frame 20: one text per run, at the offset where the run starts.
    assert [(offset, bytes(data.tolist()).decode()) for offset, data in segment['texts']] == [
        (-12, 'pick the cup'), (-9, '把杯子放下')]
    # Frames before the episode start are dropped; the previous frame is clamped to frame 0.
    early = samples.segment(CURRENT, 0, 0, 4, 5)
    assert early['offsets'].tolist() == [-3, 0]
    np.testing.assert_allclose(early['state'][0], relative(states[0], states[1]))


def test_sample_and_collate(dataset, config):
    samples = Samples([dataset], config, num_refs=2)
    rng = np.random.default_rng(0)
    batch = collate([samples.sample(rng) for _ in range(3)])
    assert batch['target'].shape == (3, 8, 20)
    assert batch['seg_sample'].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert batch['seg_kind'].tolist() == [CURRENT, REFERENCE, REFERENCE] * 3
    assert len(batch['image_groups']) == 3
    for group in batch['image_groups']:
        assert len(group['pixels']) == len(batch['state'])
        assert group['seg'].tolist() == batch['st_seg'].tolist()
        assert group['off'].tolist() == batch['st_off'].tolist()
        assert all(identify(image)[1] == camera for image, camera in zip(group['pixels'], group['cam'].tolist()))


def test_long_instruction_is_an_error(dataset, config):
    config.max_text_bytes = 8
    with pytest.raises(ValueError, match='Instruction'):
        Samples([dataset], config).segment(CURRENT, 0, 1, 10, 2)


def test_split_and_statistics(dataset, config):
    config.val_fraction = 0.5
    train, val = (Samples([dataset], config, split=split) for split in ('train', 'val'))
    assert sorted(e for _, e, _ in train.episodes + val.episodes) == [0, 1, 2, 3, 4]
    assert train.episodes and val.episodes
    # References of both splits are train episodes, and never the episode of the current segment.
    assert val.references == train.references == [[(e, n) for _, e, n in train.episodes]]
    val.num_refs = len(train.episodes)
    sample = val.sample(np.random.default_rng(0))
    assert len(sample['segments']) == 1 + len(train.episodes)
    train.num_refs = len(train.episodes)
    with pytest.raises(ValueError, match='references asked'):
        train.sample(np.random.default_rng(0))
    stats = statistics([dataset], config)
    assert all(len(stats[key]) == 20 for key in ('input_mean', 'input_std', 'action_mean', 'action_std'))


def test_rebot_quantile_statistics(config, tmp_path):
    config.absolute_state = True
    config.rebot_quantile_stats = str(tmp_path / 'norm_stats.json')
    source = {}
    for key, dim in [('state', 26), ('actions', 20)]:
        lo, hi = np.full(dim, 2.), np.full(dim, 10.)
        for arm in (0, dim // 2):
            rotation = slice(arm + 3, arm + dim // 2 - 1)
            lo[rotation], hi[rotation] = -1, 1
        source[key] = {'q01': lo.tolist(), 'q99': hi.tolist()}
    (tmp_path / 'norm_stats.json').write_text(json.dumps({'norm_stats': source}))
    stats = statistics([], config)
    for group, dim in [('input', 26), ('action', 20)]:
        center, scale = (np.asarray(stats[f'{group}_{suffix}']) for suffix in ('mean', 'std'))
        assert center.shape == scale.shape == (dim,)
        np.testing.assert_allclose((np.array([2, 6, 10, 14]) - center[0]) / scale[0], [-1, 0, 1, 2])
        assert center[3] == 0 and scale[3] == 1
