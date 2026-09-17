from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import soundfile as sf

from scale.io_utils import write_jsonl


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
CHUNK_RE = re.compile(r"^(?P<song>.+)_(?P<start>\d+)$")


def read_audio_scp(paths: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for scp_path in paths:
        with Path(scp_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(maxsplit=1)
                audio_path = fields[-1]
                audio_id = fields[0] if len(fields) == 2 else Path(audio_path).stem
                if audio_id in mapping and mapping[audio_id] != audio_path:
                    raise ValueError(f"Duplicate audio id {audio_id}: {mapping[audio_id]} / {audio_path}")
                mapping[audio_id] = audio_path
    return mapping


def scan_audio_roots(roots: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for root in roots:
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                mapping.setdefault(path.stem, str(path))
    return mapping


def load_split_ids(paths: list[str]) -> set[str] | None:
    if not paths:
        return None
    result: set[str] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            result.update(line.strip() for line in handle if line.strip())
    return result


def duration_seconds(path: str) -> float:
    try:
        info = sf.info(path)
        return float(info.frames / info.samplerate)
    except Exception:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return -1.0
        try:
            return float(result.stdout.strip())
        except ValueError:
            return -1.0


def chunk_audio(
    song_id: str,
    audio_path: str,
    dataset: str,
    chunk_duration: int,
) -> list[dict]:
    if not song_id or Path(song_id).name != song_id:
        raise ValueError(f"Unsafe audio id: {song_id!r}")
    total = duration_seconds(audio_path)
    if total <= 0:
        raise RuntimeError(f"Cannot determine duration: {audio_path}")
    records = []
    start = 0
    while start < total:
        duration = min(float(chunk_duration), total - start)
        records.append(
            {
                "chunk_id": f"{song_id}_{start}",
                "song_id": song_id,
                "dataset": dataset,
                "audio_path": str(Path(audio_path).resolve()),
                "start_sec": float(start),
                "duration": float(duration),
            }
        )
        start += chunk_duration
    return records


def build(args: argparse.Namespace) -> None:
    audio_map = read_audio_scp(args.audio_scp)
    audio_map.update(scan_audio_roots(args.audio_root))
    split_ids = load_split_ids(args.split_ids)
    records = []
    missing: list[str] = []
    skipped: list[str] = []
    reference_dir = Path(args.reference_dir)
    for feature_path in sorted(reference_dir.glob("*.npy")):
        chunk_id = feature_path.stem
        match = CHUNK_RE.match(chunk_id)
        if match is None:
            missing.append(f"unparseable_chunk:{chunk_id}")
            continue
        song_id = match.group("song")
        start_sec = int(match.group("start"))
        if split_ids is not None and song_id not in split_ids:
            continue
        audio_path = audio_map.get(song_id)
        if audio_path is None:
            missing.append(f"missing_audio:{song_id}")
            continue
        if args.require_audio_files and not Path(audio_path).is_file():
            missing.append(f"missing_audio_file:{song_id}:{audio_path}")
            continue
        total_duration = duration_seconds(audio_path)
        if args.require_audio_files and total_duration <= 0:
            missing.append(f"invalid_audio_file:{song_id}:{audio_path}")
            continue
        duration = args.chunk_duration
        if total_duration > 0:
            duration = max(0.0, min(float(args.chunk_duration), total_duration - start_sec))
        if duration <= args.min_duration:
            skipped.append(f"too_short:{chunk_id}:{duration:.3f}")
            continue
        records.append(
            {
                "chunk_id": chunk_id,
                "song_id": song_id,
                "dataset": args.dataset,
                "audio_path": audio_path,
                "start_sec": float(start_sec),
                "duration": float(duration),
            }
        )
    if missing:
        report = Path(args.output).with_suffix(".missing.txt")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(missing) + "\n", encoding="utf-8")
    if skipped:
        skipped_report = Path(args.output).with_suffix(".skipped.txt")
        skipped_report.parent.mkdir(parents=True, exist_ok=True)
        skipped_report.write_text("\n".join(skipped) + "\n", encoding="utf-8")
    if args.strict and missing:
        raise RuntimeError(f"Manifest has {len(missing)} errors; see {report}")
    write_jsonl(args.output, records)
    print(
        f"wrote {len(records)} records to {args.output}; "
        f"errors={len(missing)} skipped={len(skipped)}"
    )


def single(args: argparse.Namespace) -> None:
    audio_path = str(Path(args.audio).resolve())
    song_id = args.song_id or Path(audio_path).stem
    records = chunk_audio(song_id, audio_path, args.dataset, args.chunk_duration)
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} chunks to {args.output}")


def from_scp(args: argparse.Namespace) -> None:
    audio_map = read_audio_scp([args.input])
    records = []
    for song_id, raw_path in audio_map.items():
        audio_path = str(Path(raw_path).expanduser().resolve())
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Missing audio for {song_id}: {audio_path}")
        records.extend(
            chunk_audio(
                song_id,
                audio_path,
                args.dataset,
                args.chunk_duration,
            )
        )
    if not records:
        raise RuntimeError(f"No audio entries found in {args.input}")
    write_jsonl(args.output, records)
    print(
        f"wrote {len(records)} chunks for {len(audio_map)} songs to {args.output}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Build SCALE chunk manifests")
    sub = root.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--reference-dir", required=True)
    build_parser.add_argument("--audio-scp", action="append", default=[])
    build_parser.add_argument("--audio-root", action="append", default=[])
    build_parser.add_argument("--split-ids", action="append", default=[])
    build_parser.add_argument("--dataset", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--chunk-duration", type=int, default=420)
    build_parser.add_argument("--min-duration", type=float, default=0.1)
    build_parser.add_argument("--strict", action="store_true")
    build_parser.add_argument("--require-audio-files", action="store_true")
    build_parser.set_defaults(func=build)
    single_parser = sub.add_parser("single")
    single_parser.add_argument("--audio", required=True)
    single_parser.add_argument("--song-id")
    single_parser.add_argument("--dataset", default="SongFormBench-HX")
    single_parser.add_argument("--chunk-duration", type=int, default=420)
    single_parser.add_argument("--output", required=True)
    single_parser.set_defaults(func=single)
    scp_parser = sub.add_parser(
        "scp", help="Build inference chunks directly from a SCALE audio SCP"
    )
    scp_parser.add_argument("--input", required=True)
    scp_parser.add_argument("--dataset", default="SongFormBench-HX")
    scp_parser.add_argument("--chunk-duration", type=int, default=420)
    scp_parser.add_argument("--output", required=True)
    scp_parser.set_defaults(func=from_scp)
    return root


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
