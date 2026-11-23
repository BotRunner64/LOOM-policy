"""Train the LOOM policy. Single process, or DDP under torchrun. `--output` resumes if it has a checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from loom_policy.config import Config
from loom_policy.data import Samples, batch_to, collate, find_datasets, statistics
from loom_policy.eval import evaluate
from loom_policy.model import LoomPolicy


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--data', nargs='+', required=True, help='dataset roots or folders of them')
    parser.add_argument('--output', required=True, type=Path, help='run directory; resumed if it has last.pt')
    parser.add_argument('--config', help='JSON config file')
    limit = parser.add_mutually_exclusive_group()
    limit.add_argument('--job-steps', type=int, help='maximum additional optimizer steps for this job')
    limit.add_argument('--stop-at-step', type=int, help='checkpoint and exit early at this absolute step')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = parser.parse_args(argv)
    if args.job_steps is not None and args.job_steps < 1:
        parser.error('--job-steps must be positive')

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    checkpoint = args.output / 'last.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True) if checkpoint.exists() else None
    # The plan travels with the checkpoint. Job limits apply only to this invocation.
    config = Config.from_checkpoint(state['config']) if state else Config.load(args.config, args.set)
    step = state['step'] if state else 0
    stop = config.steps
    if args.job_steps is not None:
        stop = min(stop, step + args.job_steps)
    elif args.stop_at_step is not None:
        stop = min(stop, args.stop_at_step)
    if rank == 0:
        print(f'Plan: {config.steps} steps. Job: step {step} to {max(step, stop)}.', flush=True)
    if step >= stop:
        return

    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if cuda else 'cpu')
    if cuda:
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group('nccl' if cuda else 'gloo')

    roots = find_datasets(args.data)
    torch.manual_seed(config.seed)
    # DDP broadcasts rank zero's normalization buffers when wrapping the model.
    stats = statistics(roots, config) if state is None and rank == 0 else None
    model = LoomPolicy(config, stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, config.adam_beta2),
                                  weight_decay=config.weight_decay, fused=cuda)
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        del state
    schedule = lambda s: min((s + 1) / config.warmup, 0.5 * (1 + math.cos(math.pi * min(s / config.steps, 1))))
    # On resume the loaded optimizer state carries initial_lr, which LambdaLR needs to continue.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=step - 1)
    parallel = DistributedDataParallel(model, device_ids=[device.index] if cuda else None,
                                       gradient_as_bucket_view=True) if world > 1 else model

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'config.json').write_text(json.dumps(config.to_dict(), indent=2) + '\n')
        print(f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, '
              f'{len(roots)} datasets, step {step}/{config.steps}', flush=True)
        validation = Samples(roots, config, split='val', seed=config.seed) if config.val_fraction else None

    # The seed changes with the rank and the resume step, so no process replays another's samples.
    torch.manual_seed(config.seed + step * world + rank)
    samples = Samples(roots, config, seed=[config.seed, rank, step])
    loader = iter(torch.utils.data.DataLoader(
        samples, batch_size=config.batch_size, collate_fn=collate, num_workers=config.workers,
        pin_memory=cuda, persistent_workers=config.workers > 0, prefetch_factor=4 if config.workers else None))

    model.train()
    started, window, window_start = time.time(), 0., step
    while step < stop:
        batch = batch_to(next(loader), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            loss, _ = parallel(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        if rank:
            continue
        window += loss.item()
        record = {}
        if step % config.log_every == 0 or step == stop:
            count = step - window_start
            record = {'loss': window / count, 'lr': scheduler.get_last_lr()[0],
                      'samples_per_s': count * config.batch_size * world / (time.time() - started)}
            if cuda:
                record['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
            started, window, window_start = time.time(), 0., step
        if validation and (step % config.eval_every == 0 or step == config.steps):
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
                        'config': json.loads(json.dumps(config.to_dict()))}, temporary)
            os.replace(temporary, checkpoint)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
