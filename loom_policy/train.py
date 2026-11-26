"""Train the LOOM policy. Single process, or DDP under torchrun. `--output` resumes if it has a checkpoint."""
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from loom_policy.config import Config
from loom_policy.data import Samples, batch_to, collate, find_datasets, statistics
from loom_policy.eval import evaluate
from loom_policy.model import LoomPolicy


def micro_batches(global_batch, world, rank, cap):
    """Positions in one global update. All ranks participate in the same DDP rounds."""
    if min(global_batch, world, cap) < 1:
        raise ValueError('batch, world size and micro-batch cap must be positive')
    positions = range(rank, global_batch, world)
    rounds = math.ceil(global_batch / (world * cap))
    # Balance the local samples across rounds, including for uneven world sizes.
    return [positions[i * len(positions) // rounds:(i + 1) * len(positions) // rounds]
            for i in range(rounds)]


@dataclass(frozen=True)
class BatchLayout:
    batch: int
    epoch_size: int

    @property
    def steps_per_epoch(self):
        return (self.epoch_size + self.batch - 1) // self.batch

    def cursor(self, step):
        epoch, update = divmod(step, self.steps_per_epoch)
        return epoch * self.epoch_size + update * self.batch

    def index_batches(self, start_step, stop_step, world, rank, cap, seed):
        shuffled_epoch, order = None, None
        for step in range(start_step, stop_step):
            base = self.cursor(step)
            size = self.cursor(step + 1) - base
            epoch, offset = divmod(base, self.epoch_size)
            if epoch != shuffled_epoch:
                dtype = np.int32 if self.epoch_size <= 2**31 - 1 else np.int64
                order = np.arange(self.epoch_size, dtype=dtype)
                np.random.default_rng([seed, epoch]).shuffle(order)
                shuffled_epoch = epoch
            for group in micro_batches(size, world, rank, cap):
                # Ranks with an empty shard participate with one zero-weight example.
                positions = group or [0]
                yield [(base + i, int(order[offset + i])) for i in positions]


def epoch_data_id(samples):
    """Identify the ordered pools used to resolve current anchors and references on resume."""
    digest = hashlib.sha256()
    for source, episodes in zip(samples.sources, samples.references, strict=True):
        for episode, length in episodes:
            digest.update(json.dumps([source.episode_id(episode), length]).encode() + b'\n')
    return digest.hexdigest()


def lr_multiplier(config, step, layout):
    epoch = layout.cursor(step) / layout.epoch_size
    if epoch < config.warmup_epochs:
        return min(1., layout.cursor(step + 1) / layout.epoch_size / config.warmup_epochs)
    remaining = min(1., max(0., (config.epochs - epoch) / config.decay_epochs))
    return 0.5 * (1 - math.cos(math.pi * remaining))


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--data', nargs='+', required=True, help='dataset roots or folders of them')
    parser.add_argument('--output', required=True, type=Path, help='run directory; resumed if it has last.pt')
    parser.add_argument('--packed', type=Path, help='packed export folder (loom-data pack) to read videos from; '
                        'it must hold <source>/<layout> for every dataset')
    parser.add_argument('--config', help='JSON config file')
    parser.add_argument('--micro-batch-size', type=int, help='per-rank cap for this fixed-global-batch job')
    parser.add_argument('--wsd-end-epoch', type=int, help='change total epochs before WSD decay begins')
    parser.add_argument('--stop-at-epoch', type=int, help='checkpoint and exit after this many complete epochs')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = parser.parse_args(argv)
    if args.stop_at_epoch is not None and args.stop_at_epoch < 1:
        parser.error('--stop-at-epoch must be positive')

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    checkpoint = args.output / 'last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True) if checkpoint.exists() else None
    # The plan travels with the checkpoint. Job limits apply only to this invocation.
    config = Config(**state['config']) if state else Config.load(args.config, args.set)
    step = state['step'] if state else 0
    cap = config.batch_size if args.micro_batch_size is None else args.micro_batch_size
    global_batch = config.global_batch_size
    if world > global_batch:
        parser.error('world size exceeds the planned global batch size')
    groups = micro_batches(global_batch, world, rank, cap)
    roots = find_datasets(args.data)
    train_samples = Samples(roots, config, seed=config.seed, packed=args.packed)
    epoch_size = int(train_samples.cumulative[-1])
    data_id = epoch_data_id(train_samples)
    if state and (state['epoch_size'] != epoch_size or state['epoch_data_id'] != data_id):
        parser.error('the training anchor or reference pool has changed since the checkpoint')
    layout = BatchLayout(global_batch, epoch_size)
    consumed = layout.cursor(step)
    if args.wsd_end_epoch is not None and args.wsd_end_epoch != config.epochs:
        if consumed / epoch_size > min(args.wsd_end_epoch, config.epochs) - config.decay_epochs:
            parser.error('can only move a WSD endpoint before entering either cooldown')
        config = replace(config, epochs=args.wsd_end_epoch)
    if state and state['consumed_samples'] != consumed:
        raise ValueError('checkpoint sample cursor disagrees with the fixed global batch plan')
    total_steps = config.epochs * layout.steps_per_epoch
    stop = min(args.stop_at_epoch or config.epochs, config.epochs) * layout.steps_per_epoch
    if rank == 0:
        print(f'Plan: {config.epochs} epochs of {epoch_size} anchors, {total_steps} steps, WSD, global batch {global_batch}. '
              f'Job: step {step} to {max(step, stop)}, {world} ranks, '
              f'micro-batch cap {cap}, {len(groups)} accumulation rounds.', flush=True)
    if step >= stop:
        return

    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('nccl' if cuda else 'gloo')

    torch.manual_seed(config.seed)
    # DDP broadcasts rank zero's normalization buffers when wrapping the model.
    stats = statistics(roots, config, train_samples) if state is None and rank == 0 else None
    model = LoomPolicy(config, stats).to(device)
    if cuda and config.compile:
        # Trunk blocks have one fixed shape per micro-batch size; the reader's padded shapes vary, so it stays eager.
        for block in model.trunk:
            block.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, config.adam_beta2),
                                  weight_decay=config.weight_decay, fused=cuda)
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        del state
    # On resume the loaded optimizer state carries initial_lr, which LambdaLR needs to continue.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(config, s, layout), last_epoch=step - 1)
    parallel = DistributedDataParallel(model, device_ids=[device.index] if cuda else None,
                                       gradient_as_bucket_view=True) if world > 1 else model

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(config.to_dict(), indent=2) + '\n')
        print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, '
              f'{len(roots)} datasets, step {step}/{total_steps}', flush=True)
        validation = Samples(roots, config, split='val', seed=config.seed, packed=args.packed) if config.val_fraction else None

    batches = layout.index_batches(step, stop, world, rank, cap, config.seed)
    loader = iter(torch.utils.data.DataLoader(
        train_samples, batch_sampler=batches, collate_fn=collate, num_workers=config.workers,
        pin_memory=cuda, persistent_workers=config.workers > 0, prefetch_factor=2 if config.workers else None))

    model.train()
    started, window_start = time.time(), step
    window = torch.zeros((), device=device)
    while step < stop:
        size = layout.cursor(step + 1) - layout.cursor(step)
        groups = micro_batches(size, world, rank, cap)
        # The RTC masks of this global update must also be independent of its partition.
        rng = np.random.default_rng([config.seed, step, 1])
        prefix = rng.integers(min(config.max_prefix_len, config.horizon - 1) + 1, size=size)
        valid = size * config.horizon - int(prefix.sum())
        optimizer.zero_grad(set_to_none=True)
        for i, group in enumerate(groups):
            batch = batch_to(next(loader), device)
            lengths = prefix[list(group) or [0]]
            batch['prefix_lengths'] = torch.as_tensor(lengths, device=device)
            local_valid = len(group) * config.horizon - int(lengths.sum()) if group else 0
            weight = world * local_valid / valid  # DDP averages gradients across ranks
            sync = parallel.no_sync() if world > 1 and i + 1 < len(groups) else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                    loss, _ = parallel(batch)
                    loss = loss * weight
                loss.backward()
            window += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        record = {}
        if step % config.log_every == 0 or step == stop:
            count = step - window_start
            if world > 1:
                torch.distributed.all_reduce(window)
            record = {'loss': window.item() / (count * world), 'lr': scheduler.get_last_lr()[0],
                      'samples_per_s': (layout.cursor(step) - layout.cursor(window_start)) / (time.time() - started),
                      'global_batch_size': size, 'world_size': world, 'accum_steps': len(groups),
                      'consumed_samples': layout.cursor(step), 'epoch': layout.cursor(step) / epoch_size}
            if cuda:
                record['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
            started, window_start = time.time(), step
            window.zero_()
        if rank:
            continue
        if validation and (step % config.eval_every == 0 or step == total_steps):
            metrics = evaluate(model, validation, config.eval_batches, config.batch_size, device)
            record.update({f'val_{key}': value for key, value in metrics.items()})
        if record:
            record = {'step': step, **record}
            print(json.dumps(record), flush=True)
            with open(args.output / 'metrics.jsonl', 'a') as file:
                file.write(json.dumps(record) + '\n')
        if step % config.save_every == 0 or step == stop:
            temporary = checkpoint.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
                        'consumed_samples': layout.cursor(step), 'epoch_size': epoch_size, 'epoch_data_id': data_id,
                        'config': json.loads(json.dumps(config.to_dict()))}, temporary)
            os.replace(temporary, checkpoint)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
