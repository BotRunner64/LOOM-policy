"""Single-process baseline trainer with resumable optimizer/data/RNG state."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import torch
from .data import ManifestDataset, SamplingConfig
from .model import LoomPolicy, ModelConfig, masked_l1_loss


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 3:
        raise ValueError("Expected a LOOM direct-chunk checkpoint (format_version=3)")
    return checkpoint


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def manifest_sources(dataset):
    result = set()
    for row in dataset.rows:
        result.add((str(Path(row["dataset"]).resolve()), row["episode_id"]))
        for ref in row.get("references", []):
            result.add((str(Path(ref.get("dataset", row["dataset"])).resolve()), ref["episode_id"]))
    return result


def train(config, resume=None, steps_override=None):
    model_config = ModelConfig(**config.get("model", {}))
    sampling = SamplingConfig(**config.get("sampling", {}))
    settings = config.get("train", {})
    seed = settings.get("seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(settings.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(settings.get("cpu_threads", 2))
    data = ManifestDataset(config["manifest"], sampling, model_config.chunk)
    validation = ManifestDataset(config["validation_manifest"], sampling, model_config.chunk) if config.get("validation_manifest") else None
    if validation and manifest_sources(data) & manifest_sources(validation):
        raise ValueError("Train/validation source episodes overlap, including reference episodes")
    manifest_hash = hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()
    model = LoomPolicy(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get("lr", 3e-4),
                                  weight_decay=settings.get("weight_decay", 1e-2))
    epoch = cursor = step = 0
    if resume:
        saved = load_checkpoint(resume, device)
        if saved["model_config"] != asdict(model_config) or saved["sampling_config"] != asdict(sampling):
            raise ValueError("Resume requires unchanged model and sampling config")
        if saved["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume manifest changed")
        # Device, output, steps and checkpoint cadence may change on resume.
        for key in ("batch_size", "lr", "weight_decay", "seed"):
            if saved["train_settings"].get(key) != settings.get(key):
                raise ValueError(f"Resume training setting changed: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, step = saved["epoch"], saved["cursor"], saved["step"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        random.setstate(saved["python_rng"])
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    output = Path(config.get("output", "runs/baseline"))
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "last.pt").exists():
        raise ValueError("Output already has last.pt; use --resume or choose a new output directory")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    steps = steps_override if steps_override is not None else settings.get("steps", 1000)
    batch_size = settings.get("batch_size", 1)
    save_every = settings.get("save_every", 100)
    if steps < step or batch_size < 1 or save_every < 1 or settings.get("eval_every", save_every) < 1:
        raise ValueError("Invalid steps, batch_size or save_every")

    def checkpoint():
        save_checkpoint(output / "last.pt", {
            "format_version": 3, "step": step, "epoch": epoch, "cursor": cursor,
            "model_config": asdict(model_config), "sampling_config": asdict(sampling),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "train_settings": settings, "manifest_sha256": manifest_hash,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "semantics": "direct-chunk-l1/state26-step-body/action20-chunk-body-rot6d",
        })

    try:
        while step < steps:
            order = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            if cursor >= len(order):
                epoch += 1
                cursor = 0
                continue
            indices = order[cursor:cursor + batch_size]
            batch = [data[i] for i in indices]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = masked_l1_loss(model([sample["input"] for sample in batch]), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("grad_clip", 1.0), error_if_nonfinite=True)
            optimizer.step()
            cursor += len(indices)
            step += 1
            record = {"step": step, "epoch": epoch, "loss": float(loss.detach()), "grad_norm": float(grad_norm)}
            if validation and step % settings.get("eval_every", save_every) == 0:
                model.eval()
                with torch.no_grad():
                    # Evaluate the complete supplied validation manifest.
                    total_error, valid_fields = 0., 0
                    for i in range(len(validation)):
                        sample = validation[i]
                        count = int(sample["mask"].sum())
                        total_error += float(masked_l1_loss(model([sample["input"]]), [sample])) * count
                        valid_fields += count
                record["validation_loss"] = total_error / valid_fields
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if step % save_every == 0:
                checkpoint()
    except KeyboardInterrupt:
        checkpoint()
        raise
    checkpoint()
    return output / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, help="Total optimizer steps, including resumed steps")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("manifest", "validation_manifest", "output"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    print(train(config, args.resume, args.steps))


if __name__ == "__main__":
    main()
