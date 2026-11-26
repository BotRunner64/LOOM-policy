import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from loom_policy import train
from loom_policy.config import Config
from loom_policy.eval import evaluate, load_model
from loom_policy.data import CURRENT, Samples, collate, statistics
from loom_policy.policy import Policy
from loom_policy.data import Source
from loom_policy.geometry import relative
from loom_policy.model import LoomPolicy


def test_gbs128_partitions():
    for cap in (1, 8, 16):
        for world in range(2, 65):
            groups = [train.micro_batches(128, world, rank, cap) for rank in range(world)]
            assert len({len(rank_groups) for rank_groups in groups}) == 1
            assert max(len(batch) for rank_groups in groups for batch in rank_groups) <= cap
            assert sorted(i for rank_groups in groups for batch in rank_groups for i in batch) == list(range(128))


def test_epoch_anchor_coverage_and_resume(dataset, config):
    samples = Samples([dataset], config, images=False)
    n = int(samples.cumulative[-1])
    layout = train.BatchLayout(8, n)
    def sequence(world, cap, start, stop):
        items = [item for rank in range(world)
                 for group in layout.index_batches(start, stop, world, rank, cap, config.seed) for item in group]
        return sorted(set(items))  # empty shards reuse a zero-weight placeholder

    steps = layout.steps_per_epoch
    ordered = sequence(1, 8, 0, 2 * steps)
    anchors = [a for _, a in ordered]
    assert sorted(anchors[:n]) == sorted(anchors[n:]) == list(range(n))
    assert anchors[:n] != anchors[n:]
    split = steps - 1
    assert sequence(3, 1, 0, split) + sequence(2, 2, split, 2 * steps) == ordered
    # Check both ends of every episode, including the last eligible anchor.
    start = 0
    for number, ep, length in samples.episodes:
        states, _ = samples.sources[number].frames(ep)
        for t in (0, length - config.horizon - 1):
            drawn = samples.sample(np.random.default_rng(0), anchor=start + t)
            np.testing.assert_array_equal(drawn['target'], relative(states[t], states[t+1:t+1+config.horizon]))
        start += length - config.horizon


def test_wsd_changed_world_resume(dataset, tmp_path, config):
    # Ten eligible anchors, hence global update sizes 3, 3, 3, 1 per epoch.
    config.global_batch_size = 3
    config.horizon, config.epochs = 51, 3
    config.warmup_epochs, config.decay_epochs = 0.6, 0.5
    config.log_every = 1
    config_path = tmp_path / 'wsd.json'
    config_path.write_text(json.dumps(config.to_dict()))
    common = ['--data', str(dataset), '--config', str(config_path)]
    python = [sys.executable, '-m']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    def job(run_dir, ranks, *extra):
        launch = python if ranks == 1 else [*python, 'torch.distributed.run', '--standalone',
                                            '--nproc-per-node', str(ranks), '-m']
        subprocess.run([*launch, 'loom_policy', 'train', *common, '--output', str(run_dir), *extra],
                       env=env, check=True, timeout=120)

    control, resumed = tmp_path / 'control', tmp_path / 'resumed'
    job(control, 1, '--wsd-end-epoch', '2')
    job(resumed, 2, '--micro-batch-size', '1', '--stop-at-epoch', '1')
    job(resumed, 1, '--wsd-end-epoch', '2')
    a, b = [torch.load(d / 'last.pt', weights_only=True) for d in (control, resumed)]
    assert a['consumed_samples'] == b['consumed_samples'] == 20
    assert a['step'] == b['step'] == 8
    for name, weight in a['model'].items():
        torch.testing.assert_close(b['model'][name], weight, atol=2e-5, rtol=1e-4)
    records = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
    assert b['epoch_size'] == 10 and b['config']['epochs'] == 2
    assert [r['global_batch_size'] for r in records] == [3, 3, 3, 1] * 2
    np.testing.assert_allclose([r['epoch'] for r in records], [.3, .6, .9, 1, 1.3, 1.6, 1.9, 2])
    np.testing.assert_allclose([r['lr'] for r in records[-3:]],
                               config.lr * (1 - np.cos(np.pi * np.array([.8, .2, 0]))) / 2)


def test_rebot_single_absolute_state(dataset, tmp_path):
    preset = Path(__file__).resolve().parents[1] / 'configs/rebot_observation.json'
    config = Config.load(preset, ['dim=32', 'trunk_dim=32', 'trunk_heads=2', 'trunk_layers=1',
                                 'num_latents=4', 'stats_samples=8'])
    samples = Samples([dataset], config)
    sample = samples.sample(np.random.default_rng(0))
    segment, = sample['segments']
    assert segment['kind'] == CURRENT and segment['state'].shape == (1, 26)
    assert len(segment['images']) == 1 and len(segment['images'][0]) == 3
    # Identify the anchor by its recorded absolute state and check all future actions against it.
    for _, episode, _ in samples.episodes:
        states, _ = samples.sources[0].frames(episode)
        matches = np.flatnonzero((states == segment['state'].numpy()).all(axis=1))
        if matches.size:
            t = matches.item()
            break
    else:
        raise AssertionError('absolute state does not match a recorded frame')
    np.testing.assert_allclose(sample['target'], relative(states[t], states[t+1:t+51]))

    model = LoomPolicy(config, statistics([dataset], config))
    assert model.input_scale.shape == (26,) and model.action_scale.shape == (20,)
    loss, actions = model(collate([sample]))
    loss.backward()
    assert torch.isfinite(loss) and actions.shape == (1, 50, 20)
    checkpoint = tmp_path / 'absolute.pt'
    torch.save({'model': model.state_dict(), 'config': config.to_dict()}, checkpoint)
    policy = Policy(checkpoint, 'cpu')
    images = samples.sources[0].images(episode, [t], config.image_size, config.patch)
    args = (states[t:t+1], images, ['top', 'left_wrist', 'right_wrist'], 'pick the cup')
    torch.testing.assert_close(policy.segment(CURRENT, *args)['state'], segment['state'])
    prefix = states[t+1:t+4]
    chunk = policy.act(*args, prefix_states=prefix)
    assert chunk.shape == (50, 26) and np.isfinite(chunk).all()
    np.testing.assert_array_equal(chunk[:3], prefix)


def test_train_resume_evaluate_and_act(dataset, tmp_path, config, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)  # a tiny model; CPU keeps it compiler-free
    config.trunk_dim, config.trunk_heads = config.dim, config.heads
    config.epochs = 1
    config.log_every = config.save_every = 2
    config.global_batch_size = 8
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(config.to_dict()))
    args = ['--data', str(dataset), '--output', str(tmp_path), '--config', str(path)]
    replace = os.replace

    def interrupt_after_checkpoint(src, dest):
        replace(src, dest)
        raise InterruptedError

    with monkeypatch.context() as patch:
        patch.setattr(train.os, 'replace', interrupt_after_checkpoint)
        with pytest.raises(InterruptedError):
            train.main(args)
    first = torch.load(tmp_path / 'last.pt', weights_only=True)
    assert first['step'] == 2 and first['consumed_samples'] == 16 < first['epoch_size']
    train.main(args)
    records = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert records[-1]['epoch'] == 1
    assert [r['step'] for r in records] == sorted({r['step'] for r in records})

    model = load_model(tmp_path / 'last.pt', torch.device('cpu'))
    assert 0 < model.action_scale.min() < 1  # statistics travel with the checkpoint
    metrics = evaluate(model, Samples([dataset], model.config, num_refs=1), 1, 2, torch.device('cpu'))
    assert set(metrics) == {'loss', 'translation_m', 'rotation_deg', 'gripper'}

    policy = Policy(tmp_path / 'last.pt', 'cpu')
    source = Source(dataset)
    states, _ = source.frames(1)
    images = source.images(1, [20, 23], (64, 64), patch=1)
    cameras = ['top', 'left_wrist', 'right_wrist']
    live = policy.segment(0, states[[20, 23]], images, cameras, 'pick the cup', previous=states[17])
    offline = Samples([dataset], model.config).segment(0, 0, 1, 23, 2)
    assert [[v.shape for v in frame] for frame in live['images']] == [
        [v.shape for v in frame] for frame in offline['images']]
    without = policy.act(states[[20, 23]], images, cameras, 'pick the cup', previous=states[17])
    assert without.shape == (8, 26)
    def act(prefix):
        return policy.act(states[[20, 23]], images, cameras, 'pick the cup', previous=states[17],
                          prefix_states=prefix)

    np.testing.assert_array_equal(act(states[:0]), without)
    prefix = states[24:28].copy()
    projected = []
    hook = policy.model.prefix_proj.register_forward_pre_hook(lambda module, args: projected.append(args[0]))
    chunk = act(prefix)
    hook.remove()
    expected = (torch.from_numpy(relative(states[23], prefix)) - model.action_center) / model.action_scale
    torch.testing.assert_close(projected[0][0, :4], expected)
    np.testing.assert_array_equal(chunk[:4], prefix)
    prefix[:, 0] += 0.01
    changed = act(prefix)
    np.testing.assert_array_equal(changed[:4], prefix)
    assert np.max(np.abs(changed[4:] - chunk[4:])) > 1e-6
    np.testing.assert_array_equal(act(states[24:32]), states[24:32])
    rotation = without[:, 3:12].reshape(8, 3, 3)
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.tile(np.eye(3), (8, 1, 1)), atol=1e-5)
    reference_states, _ = source.frames(2)
    frames = list(range(0, 15, 3))
    policy.set_references([{'states': reference_states[frames], 'images': source.images(2, frames, (64, 64), patch=1),
                            'cameras': cameras, 'instruction': 'pick the cup'}])
    assert np.abs(policy.act(states[[20, 23]], images, cameras, 'pick the cup', previous=states[17]) - without).max() > 0

    reference = {'states': reference_states[frames], 'images': source.images(2, frames, (64, 64), patch=1),
                 'cameras': cameras, 'instruction': 'pick the cup'}
    policy.set_references([reference, reference])
    assert [len(x) for x in policy.references] == [2, 2]
    assert policy.act(states[[20, 23]], images, cameras, 'pick the cup').shape == (8, 26)
