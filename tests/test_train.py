import json
from pathlib import Path

import numpy as np
import torch

from loom_policy import train
from loom_policy.config import Config
from loom_policy.eval import evaluate, load_model
from loom_policy.data import CURRENT, Samples, collate, statistics
from loom_policy.policy import Policy
from loom_policy.data import Source
from loom_policy.geometry import relative
from loom_policy.model import LoomPolicy


def test_rebot_single_absolute_state(dataset, tmp_path):
    preset = Path(__file__).resolve().parents[1] / 'configs/rebot_observation.json'
    config = Config.load(preset, ['dim=32', 'trunk_dim=32', 'trunk_heads=2', 'trunk_layers=1',
                                 'num_latents=4', 'val_fraction=0', 'stats_samples=8'])
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
    assert model.input_std.shape == (26,) and model.action_std.shape == (20,)
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


def run(dataset, directory, steps):
    overrides = ['image_size=[32,48]', 'obs_stride=3', 'ref_frames=5', 'horizon=8', 'dim=32', 'heads=2',
                 'num_latents=4', 'reader_blocks=1', 'reader_self_layers=1',
                 'trunk_dim=32', 'trunk_heads=2', 'trunk_layers=2', 'workers=0',
                 'batch_size=4', 'stats_samples=20', 'val_fraction=0.4', 'eval_batches=1', 'warmup=2',
                 f'steps={steps}', 'log_every=2', 'eval_every=4', 'save_every=4']
    train.main(['--data', str(dataset), '--output', str(directory), '--set', *overrides])


def test_train_resume_evaluate_and_act(dataset, tmp_path):
    run(dataset, tmp_path, 4)
    first = torch.load(tmp_path / 'last.pt', weights_only=True)
    assert first['step'] == 4
    # A finished run does nothing more; its config, not the command line, decides the length.
    run(dataset, tmp_path, 8)
    assert torch.load(tmp_path / 'last.pt', weights_only=True)['step'] == 4
    first['config']['steps'] = 8
    torch.save(first, tmp_path / 'last.pt')
    run(dataset, tmp_path, 8)
    assert torch.load(tmp_path / 'last.pt', weights_only=True)['step'] == 8
    records = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == [2, 4, 6, 8] and 'val_translation_m' in records[-1]

    model = load_model(tmp_path / 'last.pt', torch.device('cpu'))
    assert model.action_std.max() != 1  # statistics travel with the checkpoint
    metrics = evaluate(model, Samples([dataset], model.config, split='val', num_refs=1), 1, 2, torch.device('cpu'))
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
    expected = (torch.from_numpy(relative(states[23], prefix)) - model.action_mean) / model.action_std
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

    # The same raw views use the historical fixed-size processor for an old checkpoint.
    del first['config']['adaptive_rgb']
    del first['config']['share_reference_budget']
    del first['config']['trunk_dim']
    del first['config']['trunk_heads']
    torch.save(first, tmp_path / 'legacy.pt')
    legacy = Policy(tmp_path / 'legacy.pt', 'cpu')
    segment = legacy.segment(0, states[[20, 23]], images, cameras, 'pick the cup')
    assert all(view.shape == (3, 32, 48) for frame in segment['images'] for view in frame)
    reference = {'states': reference_states[frames], 'images': source.images(2, frames, (48, 64)),
                 'cameras': cameras, 'instruction': 'pick the cup'}
    legacy.set_references([reference, reference])
    assert legacy.references.shape == (2, 4, 32)
    policy.set_references([reference, reference])
    assert [len(x) for x in policy.references] == [2, 2]
    assert policy.act(states[[20, 23]], images, cameras, 'pick the cup').shape == (8, 26)
