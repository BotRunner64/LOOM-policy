"""Index current observation times; choose episode splits before indexing."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import ExportDataset
from .events import ChunkConfig
from .model import ModelConfig


def build_manifest(dataset, output, episodes=None, stride=16, chunk=None,
                   max_age_ns=250_000_000, references=()):
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    chunk = chunk or ChunkConfig()
    root = ExportDataset(dataset)
    rows = []
    refs = [{"dataset": str(root.root), "episode_id": x} for x in references]
    for eid in episodes if episodes is not None else root.metadata:
        if eid in references:
            continue
        ep = root.episode(eid)
        offsets = chunk.offsets_ns
        timeline = np.unique(np.concatenate([t for t, _ in ep.streams.values()]))[::stride]
        for time in timeline:
            try:
                _, mask = ep.labels(int(time), offsets, max_age_ns)
            except ValueError:
                continue
            if mask.any():
                rows.append({"dataset": str(root.root), "episode_id": eid,
                             "time_ns": int(time), "references": refs})
    if not rows:
        raise ValueError("No valid samples; check episode selection, chunk timing and staleness limit")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True, help="Training config defining output chunk timing and staleness")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", nargs="+", help="Explicit target episode IDs; omitted means all")
    parser.add_argument("--references", nargs="*", default=[])
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(build_manifest(args.dataset, args.output, args.episodes, args.stride,
                         ModelConfig(**config.get("model", {})).chunk,
                         config.get("sampling", {}).get("max_age_ns", 250_000_000), args.references))


if __name__ == "__main__":
    main()
