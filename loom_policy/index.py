"""Build explicit query manifests; choose episode splits before indexing."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import ExportDataset


def build_manifest(dataset, output, episodes=None, stride=16, horizon=8, dt_ns=50_000_000,
                   max_age_ns=250_000_000, references=()):
    if min(stride, horizon, dt_ns) < 1:
        raise ValueError("stride, horizon, dt_ns must be positive")
    root = ExportDataset(dataset)
    rows = []
    refs = [{"dataset": str(root.root), "episode_id": x} for x in references]
    for eid in episodes if episodes is not None else root.metadata:
        if eid in references:
            continue
        ep = root.episode(eid)
        offsets = [dt_ns * (i + 1) for i in range(horizon)]
        timeline = np.unique(np.concatenate([t for t, _ in ep.streams.values()]))[::stride]
        for time in timeline:
            try:
                _, mask = ep.labels(int(time), offsets, max_age_ns)
            except ValueError:
                continue
            if mask.any():
                rows.append({"dataset": str(root.root), "episode_id": eid,
                             "query_time_ns": int(time), "offsets_ns": offsets, "references": refs})
    if not rows:
        raise ValueError("No valid queries; check episode selection, horizon and staleness limit")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", nargs="+", help="Explicit target episode IDs; omitted means all")
    parser.add_argument("--references", nargs="*", default=[])
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--dt-ns", type=int, default=50_000_000)
    parser.add_argument("--max-age-ns", type=int, default=250_000_000)
    args = parser.parse_args()
    print(build_manifest(args.dataset, args.output, args.episodes, args.stride, args.horizon,
                         args.dt_ns, args.max_age_ns, args.references))


if __name__ == "__main__":
    main()
