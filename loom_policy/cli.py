"""Train, evaluate, or serve a LOOM policy."""
import argparse
from importlib import import_module
import sys


COMMANDS = {
    'train': ('loom_policy.train', 'Train a policy'),
    'eval': ('loom_policy.eval', 'Evaluate a checkpoint offline'),
    'serve': ('loom_policy.serve_rebot', 'Serve a checkpoint to a ReBot controller'),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog='loom-policy', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, (_, summary) in COMMANDS.items():
        commands.add_parser(name, help=summary)
    # Each command owns its options and loads its dependencies only when selected.
    name = parser.parse_args(argv[:1]).command
    module = import_module(COMMANDS[name][0])
    return module.main(argv[1:], prog=f'loom-policy {name}')
