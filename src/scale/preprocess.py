"""Native-rate MuQ feature extraction and feature validation."""
from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scale.io_utils import atomic_save_npy, read_jsonl


def records_for_shard(manifests: list[str], shard_index: int, num_shards: int) -> list[dict]:
    records = {}
    for manifest in manifests:
        for record in read_jsonl(manifest):
            records.setdefault(record["chunk_id"], record)
    ordered = sorted(records.values(), key=lambda item: item["chunk_id"])
    return [item for index, item in enumerate(ordered) if index % num_shards == shard_index]


def existing_ok(path: Path, dim: int, dtype: np.dtype) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return value.ndim == 3 and value.shape[0] == 1 and value.shape[-1] == dim and value.shape[1] > 0 and value.dtype == dtype
    except Exception:
        return False


def run_ssl(args: argparse.Namespace) -> None:
    import librosa
    from muq import MuQ

    device = torch.device(args.device)
    model = MuQ.from_pretrained(args.muq_checkpoint).to(device).eval()
    root = Path(args.output_dir)
    for name in ("muq_local", "muq_global"):
        (root / name).mkdir(parents=True, exist_ok=True)
    records = records_for_shard(args.manifest, args.shard_index, args.num_shards)
    if args.limit is not None:
        records = records[: args.limit]
    failed = 0
    with (root / f"errors.ssl.shard-{args.shard_index:03d}.jsonl").open("a", encoding="utf-8") as errors:
        for record in tqdm(records, desc="muq"):
            paths = {name: root / name / f"{record['chunk_id']}.npy" for name in ("muq_local", "muq_global")}
            if args.skip_existing and all(existing_ok(path, 1024, np.dtype("float32")) for path in paths.values()):
                continue
            try:
                waveform, _ = librosa.load(record["audio_path"], sr=24000, mono=True,
                                           offset=float(record["start_sec"]), duration=float(record["duration"]))
                audio = torch.from_numpy(waveform).float().to(device)[None]
                parts = [audio[:, start:start + 30 * 24000]
                         for start in range(0, audio.shape[-1], 30 * 24000)
                         if audio[:, start:start + 30 * 24000].shape[-1] >= 1025]
                if not parts:
                    raise ValueError("Audio chunk is too short for MuQ extraction")
                with torch.inference_mode():
                    global_value = model(audio, output_hidden_states=True)["hidden_states"][10]
                    local_value = torch.cat([model(part, output_hidden_states=True)["hidden_states"][10]
                                             for part in parts], dim=1)
                width = min(local_value.shape[1], global_value.shape[1])
                atomic_save_npy(paths["muq_local"], local_value[:, :width].detach().cpu().float().numpy())
                atomic_save_npy(paths["muq_global"], global_value[:, :width].detach().cpu().float().numpy())
            except Exception as exc:
                failed += 1
                errors.write(json.dumps({"chunk_id": record["chunk_id"], "error": repr(exc),
                                         "traceback": traceback.format_exc()}) + "\n")
                errors.flush()
                if args.fail_fast:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if failed:
        raise SystemExit(2)


def validate(args: argparse.Namespace) -> None:
    specs = {}
    for raw in args.feature:
        name, directory, dim, rate, dtype = raw.split("=", 4)
        specs[name] = (Path(directory), int(dim), float(rate), np.dtype(dtype))
    records = records_for_shard(args.manifest, 0, 1)
    errors = []
    summary = {name: {"ok": 0, "missing": 0, "invalid": 0} for name in specs}
    for record in tqdm(records, desc="validate"):
        for name, (directory, dim, rate, dtype) in specs.items():
            path = directory / f"{record['chunk_id']}.npy"
            if not path.is_file():
                summary[name]["missing"] += 1
                errors.append({"chunk_id": record["chunk_id"], "feature": name, "error": "missing"})
                continue
            try:
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != dim or value.dtype != dtype:
                    raise ValueError(f"shape={value.shape}, dtype={value.dtype}")
                expected = float(record["duration"]) * rate
                if abs(value.shape[1] - expected) > max(args.frame_tolerance, rate):
                    raise ValueError(f"frames={value.shape[1]}, expected~={expected:.1f}")
                summary[name]["ok"] += 1
            except Exception as exc:
                summary[name]["invalid"] += 1
                errors.append({"chunk_id": record["chunk_id"], "feature": name, "error": str(exc)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": len(records), "summary": summary, "errors": errors}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SCALE MuQ preprocessing")
    commands = root.add_subparsers(dest="command", required=True)
    ssl = commands.add_parser("ssl")
    ssl.add_argument("--manifest", action="append", required=True)
    ssl.add_argument("--output-dir", required=True)
    ssl.add_argument("--device", default="cuda:0")
    ssl.add_argument("--provider", choices=["muq"], default="muq")
    ssl.add_argument("--muq-checkpoint", default="OpenMuQ/MuQ-large-msd-iter")
    ssl.add_argument("--shard-index", type=int, default=0)
    ssl.add_argument("--num-shards", type=int, default=1)
    ssl.add_argument("--limit", type=int)
    ssl.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ssl.add_argument("--fail-fast", action="store_true")
    ssl.set_defaults(func=run_ssl)
    check = commands.add_parser("validate")
    check.add_argument("--manifest", action="append", required=True)
    check.add_argument("--feature", action="append", required=True)
    check.add_argument("--frame-tolerance", type=float, default=2.0)
    check.add_argument("--output", required=True)
    check.set_defaults(func=validate)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
