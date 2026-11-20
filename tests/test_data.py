import numpy as np
import pytest

from conftest import identify
from loom_policy.data import CURRENT, REFERENCE, Samples, Source, collate, find_datasets, statistics
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


def test_segment_facts(dataset, config):
    samples = Samples([dataset], config)
    source = samples.sources[0]
    states, _ = source.frames(0)
    segment = samples.segment(REFERENCE, 0, 0, 30, 5)
    assert segment['offsets'].tolist() == [-12, -9, -6, -3, 0]
    assert segment['images'].shape == (5, 3, 3, 32, 48)
    assert identify(segment['images'][1, 2]) == (21, 2)
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
    assert len(batch['images']) == len(batch['img_seg']) == 3 * len(batch['state'])
    assert batch['img_off'][batch['img_seg'] == 0].max() == 0


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
