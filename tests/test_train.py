import json

import numpy as np
import torch

from loom_policy import train
from loom_policy.eval import evaluate, load_model
from loom_policy.data import Samples
from loom_policy.policy import Policy
from loom_policy.data import Source


def run(dataset, directory, steps):
    overrides = ['image_size=[32,48]', 'obs_stride=3', 'ref_frames=5', 'horizon=8', 'dim=32', 'heads=2',
                 'num_latents=4', 'reader_blocks=1', 'reader_self_layers=1', 'trunk_layers=2', 'workers=0',
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
    images = source.images(1, [20, 23], (48, 64))
    cameras = ['top', 'left_wrist', 'right_wrist']
    without = policy.act(states[[20, 23]], images, cameras, 'pick the cup', previous=states[17])
    assert without.shape == (8, 26)
    rotation = without[:, 3:12].reshape(8, 3, 3)
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.tile(np.eye(3), (8, 1, 1)), atol=1e-5)
    reference_states, _ = source.frames(2)
    frames = list(range(0, 15, 3))
    policy.set_references([{'states': reference_states[frames], 'images': source.images(2, frames, (48, 64)),
                            'cameras': cameras, 'instruction': 'pick the cup'}])
    assert np.abs(policy.act(states[[20, 23]], images, cameras, 'pick the cup', previous=states[17]) - without).max() > 0
