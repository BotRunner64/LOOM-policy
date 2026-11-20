"""Offline open-loop evaluation: the same frozen model with different amounts of context."""
import argparse
import json

import numpy as np
import torch

from loom_policy.config import Config
from loom_policy.data import Samples, collate, find_datasets
from loom_policy.geometry import errors
from loom_policy.model import LoomPolicy


def load_model(checkpoint, device):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = LoomPolicy(Config(**state['config']))
    model.load_state_dict(state['model'])
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model, samples, batches, batch_size, device):
    """Mean loss and physical errors over `batches` batches; the sample sequence is fixed by the seed."""
    model.eval()
    rng = np.random.default_rng(samples.seed)
    totals = {}
    for _ in range(batches):
        batch = collate([samples.sample(rng) for _ in range(batch_size)])
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            loss, actions = model(batch)
        metrics = {'loss': loss.item(), **errors(actions.cpu().numpy(), batch['target'].cpu().numpy())}
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value / batches
    model.train()
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', nargs='+', required=True, help='dataset roots or folders of them')
    parser.add_argument('--refs', type=int, nargs='+', default=[0, 1], help='numbers of reference segments to compare')
    parser.add_argument('--split', default='val', choices=('train', 'val'))
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    roots = find_datasets(args.data)
    for refs in args.refs:
        samples = Samples(roots, model.config, split=args.split, seed=model.config.seed, num_refs=refs)
        print(json.dumps({'refs': refs, **evaluate(model, samples, args.batches, args.batch_size, device)}))


if __name__ == '__main__':
    main()
