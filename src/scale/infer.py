from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scale.decoder import decode
from scale.io_utils import feature_array, read_jsonl
from scale.lyrics_chorus import ChorusLyricsTokenizer, pad_lyrics_batch
from scale.metrics import write_predictions
from scale.runtime import apply_overrides, load_config, load_model


def parse_directories(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, directory = value.split("=", 1)
        result[name] = Path(directory)
    return result


def feature_batch(
    record: dict, directories: dict, specs: dict, lyrics_dir: Path,
    lyrics_tokenizer: ChorusLyricsTokenizer | None, device: torch.device,
) -> dict:
    features = {}
    lengths = {}
    for name, spec in specs.items():
        array = feature_array(
            directories[name] / f"{record['chunk_id']}.npy", int(spec["dim"])
        )
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None].to(device)
        features[name] = tensor
        lengths[name] = torch.tensor([tensor.shape[1]], device=device)
    result = {
        "features": features,
        "feature_lengths": lengths,
        "dataset_ids": torch.tensor([int(record["dataset_id"])], device=device),
    }
    if lyrics_tokenizer is not None:
        lyrics_path = lyrics_dir / f"{record['song_id']}.json"
        try:
            lyrics = (
                lyrics_tokenizer.encode(
                    lyrics_path, chunk_start_sec=float(record["start_sec"])
                )
                if lyrics_path.is_file()
                else lyrics_tokenizer.empty()
            )
        except (OSError, TypeError, ValueError):
            lyrics = lyrics_tokenizer.empty()
        result.update({
            name: value.to(device)
            for name, value in pad_lyrics_batch(
                [lyrics], lyrics_tokenizer.pad_token_id
            ).items()
        })
    return result


@torch.inference_mode()
def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    apply_overrides(config, args.override)
    device = torch.device(args.device)
    model, _ = load_model(config, args.checkpoint, device, prefer_ema=not args.online_model)
    model.eval()
    lyrics_tokenizer = None
    if bool(config.model.get("use_lyrics", True)):
        lyrics_tokenizer = ChorusLyricsTokenizer(
            str(config.model.lyrics_tokenizer_path),
            int(config.model.lyrics_max_block_tokens),
            int(config.model.lyrics_max_line_tokens),
            int(config.model.lyrics_max_blocks),
            int(config.model.lyrics_max_lines_per_block),
            str(config.model.lyrics_line_token),
        )
    lyrics_dir = Path(args.lyrics_dir)
    directories = parse_directories(args.feature_dir)
    expected = set(config.model.feature_specs.keys())
    if set(directories) != expected:
        raise ValueError(f"Feature dirs must be exactly {sorted(expected)}, got {sorted(directories)}")
    records = read_jsonl(args.manifest)
    for record in records:
        record["dataset_id"] = args.dataset_id
    records.sort(key=lambda item: (item["song_id"], float(item["start_sec"])))
    frame_hz = float(config.decode.frame_hz)
    song_outputs = {}
    for record in tqdm(records, desc="infer"):
        batch = feature_batch(
            record, directories, config.model.feature_specs,
            lyrics_dir, lyrics_tokenizer, device,
        )
        output = model(batch)
        valid = int((~output["padding_mask"][0]).sum().item())
        song = song_outputs.setdefault(record["song_id"], [])
        song.append(
            (
                float(record["start_sec"]),
                output["boundary_logits"][0, :valid].cpu(),
                output["function_logits"][0, :valid].cpu(),
            )
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for song_id, chunks in song_outputs.items():
        total_frames = max(int(round(start * frame_hz)) + boundary.shape[0] for start, boundary, _ in chunks)
        boundary_sum = torch.zeros(total_frames)
        function_sum = torch.zeros(total_frames, int(config.model.num_classes))
        counts = torch.zeros(total_frames)
        for start, boundary, function in chunks:
            offset = int(round(start * frame_hz))
            end = offset + boundary.shape[0]
            boundary_sum[offset:end] += boundary
            function_sum[offset:end] += function
            counts[offset:end] += 1
        valid = counts > 0
        if not valid.all():
            raise RuntimeError(f"Feature chunks leave gaps for {song_id}")
        segments = decode(
            boundary_sum / counts,
            function_sum / counts[:, None],
            int(config.decode.get("label_mask_dataset_id", args.dataset_id)),
            frame_hz,
            float(config.decode.threshold),
            float(config.decode.min_segment_seconds),
        )
        write_predictions(
            [{"id": song_id, "msa": segments}], output_dir, args.txt_output_dir
        )
        if args.save_logits:
            np.savez_compressed(
                output_dir / f"{song_id}.logits.npz",
                boundary_logits=(boundary_sum / counts).numpy(),
                function_logits=(function_sum / counts[:, None]).numpy(),
                frame_hz=frame_hz,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-dir", action="append", required=True)
    parser.add_argument("--lyrics-dir", default="/__scale_lyrics_disabled__")
    parser.add_argument("--dataset-id", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--txt-output-dir",
        help="Optional ALMA-compatible MSA txt output directory (one <time> <label> row).",
    )
    parser.add_argument("--online-model", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    main(parser.parse_args())
