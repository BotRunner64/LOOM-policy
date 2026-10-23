"""Offline query or robot-adapter entry point. Does not send robot commands."""
import argparse
import json
from pathlib import Path
import torch
from .data import ManifestDataset, SamplingConfig
from .geometry import decode_actions
from .model import LoomPolicy, ModelConfig
from .train import load_checkpoint


def predict(checkpoint, manifest, index=0, device="cpu"):
    saved = load_checkpoint(checkpoint, device)
    data = ManifestDataset(manifest, SamplingConfig(**saved["sampling_config"]))
    # Inference never requests or loads future supervision.
    from .data import load_input
    row = data.rows[index]
    inputs, _, anchors = load_input(row, data.sampling)
    model = LoomPolicy(ModelConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.no_grad():
        actions = model([inputs])[0].cpu().numpy()
    return {"episode_id": row["episode_id"], "query_time_ns": row["query_time_ns"],
            "offsets_ns": row["offsets_ns"], "relative_actions": actions.tolist(),
            "absolute_targets": decode_actions(actions, anchors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.manifest, args.index, args.device)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
