"""Serve a single-frame LOOM checkpoint to a ReBot controller."""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import rebot_policy_link as link

from loom_policy.data import CAMERAS
from loom_policy.policy import Policy


FPS = 30
SIDES = ('left', 'right')


def prefix_states(prefix, stamp, length):
    streams = link.validate_actions(prefix)['state']
    times = link.future_timestamps(stamp, length, FPS)
    if set(streams) != set(link.REBOT_STATE) or any(s['timestamp_ns'] != times for s in streams.values()):
        raise ValueError('RTC prefix must match the declared streams, length and observation time grid')
    columns = []
    for side in SIDES:
        columns.append(np.asarray(streams[f'{side}_eef']['pose']))
        columns.append(np.asarray(streams[f'{side}_gripper']['openness'])[:, None])
    return np.concatenate(columns, axis=1)


class RebotPolicy:
    def __init__(self, policy, prefix_steps):
        if policy.config.current_frames != 1:
            raise ValueError('This server requires a single-frame checkpoint')
        if not 1 <= prefix_steps <= min(policy.config.max_prefix_len, policy.config.horizon - 1):
            raise ValueError('RTC prefix length exceeds the training range or is less than one')
        self.policy = policy
        self.prefix_steps = prefix_steps

    def __call__(self, observation):
        stamp = observation['timestamp_ns']
        tables = observation['state']
        state = []
        for side in SIDES:
            state.extend(tables[f'{side}_eef']['pose'][-1])
            state.append(tables[f'{side}_gripper']['openness'][-1])
        images = []
        for camera in CAMERAS:
            with Image.open(BytesIO(observation['rgb'][camera]['jpeg'][-1])) as image:
                images.append(np.asarray(image.convert('RGB')))
        prefix = None
        if 'action_prefix' in observation:
            prefix = prefix_states(observation['action_prefix'], stamp, self.prefix_steps)
        chunk = self.policy.act([state], [images], CAMERAS, observation['instruction'], prefix_states=prefix)
        if chunk.shape != (self.policy.config.horizon, 26) or not np.isfinite(chunk).all():
            raise ValueError('Expected a finite chunk of absolute ReBot states')
        times = link.future_timestamps(stamp, len(chunk), FPS)
        streams = {}
        for index, side in enumerate(SIDES):
            arm = chunk[:, index * 13:(index + 1) * 13]
            streams[f'{side}_eef'] = {'timestamp_ns': times, 'pose': arm[:, :12].tolist()}
            streams[f'{side}_gripper'] = {'timestamp_ns': times, 'openness': arm[:, 12].tolist()}
        return {'state': streams}

    def warmup(self):
        arm = [0., 0., 0., *np.eye(3).flatten(), 0.5]
        frames = [[np.zeros((480, 640, 3), dtype=np.uint8)] * len(CAMERAS)]
        args = ([arm + arm], frames, CAMERAS, 'warmup')
        actions = self.policy.act(*args)
        conditioned = self.policy.act(*args, prefix_states=actions[:self.prefix_steps])
        if not np.isfinite(conditioned).all():
            raise ValueError('Non-finite output during warmup')


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--rtc-prefix-steps', type=int, default=3)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    infer = RebotPolicy(Policy(checkpoint), args.rtc_prefix_steps)
    infer.warmup()
    metadata = link.metadata(rgb=CAMERAS, rtc_prefix_steps=args.rtc_prefix_steps,
                             checkpoint=str(checkpoint), policy='LOOM')
    link.serve(infer, metadata, args.host, args.port,
               ready=lambda _: print(f'LOOM ready on ws://{args.host}:{args.port}', flush=True))


if __name__ == '__main__':
    main()
