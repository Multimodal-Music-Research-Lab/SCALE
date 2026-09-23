"""Unified audio, lyric extraction, prediction, and evaluation entry points."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from tqdm import tqdm

from scale.manifest import AUDIO_SUFFIXES


def repository() -> Path:
    return Path(os.environ.get("SCALE_REPO", Path(__file__).resolve().parents[2]))


def run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("[SCALE]", " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], check=True, cwd=cwd, env=env)


def run_lyrics_with_progress(
    command: list[str],
    *,
    cwd: Path,
    env: dict,
    output_dir: Path,
    song_ids: list[str],
    log_path: Path,
) -> None:
    """Run SoulX quietly while reporting completed lyric JSON files."""
    print("[SCALE]", " ".join(str(value) for value in command), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    process = None
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                [str(value) for value in command],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            with tqdm(total=len(song_ids), desc="lyrics") as progress:
                while process.poll() is None:
                    completed = sum(
                        (output_dir / f"{song_id}.json").is_file()
                        for song_id in song_ids
                    )
                    if completed > progress.n:
                        progress.update(completed - progress.n)
                    time.sleep(0.25)

                completed = sum(
                    (output_dir / f"{song_id}.json").is_file()
                    for song_id in song_ids
                )
                if completed > progress.n:
                    progress.update(completed - progress.n)
            return_code = process.returncode
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise

    if return_code != 0:
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            log_tail = log_text[-4000:]
        except OSError:
            log_text = ""
            log_tail = "<unable to read SoulX log>"
        saved_log_path = output_dir / "_soulxsinger_error.log"
        try:
            saved_log_path.write_text(log_text, encoding="utf-8")
        except OSError:
            saved_log_path = log_path
        print(
            f"[SCALE][lyrics][ERROR] SoulX exited with code {return_code}. "
            f"Log: {saved_log_path}",
            file=sys.stderr,
        )
        if log_tail.strip():
            print(log_tail.rstrip(), file=sys.stderr)
        raise subprocess.CalledProcessError(return_code, command)


def audio_entries(args: argparse.Namespace) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    if args.audio:
        entries.extend((Path(value).stem, Path(value)) for value in args.audio)
    elif args.audio_dir:
        directory = Path(args.audio_dir)
        entries.extend((path.stem, path) for path in sorted(directory.rglob("*"))
                       if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    else:
        for raw in Path(args.audio_list).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            full_path = Path(raw)
            if full_path.is_file():
                entries.append((full_path.stem, full_path))
                continue
            fields = raw.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Invalid audio-list entry: {raw}")
            entries.append((fields[0], Path(fields[1])))
    if not entries:
        raise ValueError("No input audio was found")
    seen: set[str] = set()
    for song_id, path in entries:
        if song_id in seen:
            raise ValueError(f"Duplicate song id: {song_id}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(song_id)
    return entries


def extract_lyrics(entries: list[tuple[str, Path]], directory: Path, args: argparse.Namespace,
                   workspace: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    missing = [(song_id, audio) for song_id, audio in entries
               if not (directory / f"{song_id}.json").is_file()]
    if not missing:
        print(f"[SCALE] Reusing {len(entries)} existing lyric files")
    else:
        links = workspace / "lyric_audio"
        links.mkdir()
        scp = workspace / "lyrics.scp"
        paths = []
        for song_id, audio in missing:
            target = links / f"{song_id}{audio.suffix}"
            target.symlink_to(audio.resolve())
            paths.append(str(target))
        scp.write_text("\n".join(paths) + "\n", encoding="utf-8")
        root = repository()
        env = dict(os.environ)
        vendor = root / "third_party" / "soulx_singer"
        env["PYTHONPATH"] = str(vendor) + os.pathsep + env.get("PYTHONPATH", "")
        command = [os.environ["SOULX_PY"], str(root / "tools" / "extract_lyrics.py"),
                   "--scp_path", str(scp), "--output_dir", str(directory),
                   "--language", args.language, "--device", args.device,
                   "--sat_model", os.environ["SAT_MODEL"],
                   "--sat_tokenizer", os.environ["SAT_TOKENIZER"]]
        run_lyrics_with_progress(
            command,
            cwd=vendor,
            env=env,
            output_dir=directory,
            song_ids=[song_id for song_id, _ in missing],
            log_path=workspace / "soulxsinger.log",
        )
    failed = [song_id for song_id, _ in entries if not (directory / f"{song_id}.json").is_file()]
    invalid = []
    for song_id, _ in entries:
        path = directory / f"{song_id}.json"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload.get("segments"), list):
                    invalid.append(song_id)
            except (OSError, ValueError, TypeError):
                invalid.append(song_id)
    if (failed or invalid) and not args.allow_missing_lyrics:
        raise RuntimeError(f"Lyric extraction failed or produced invalid files: {failed + invalid}")
    if failed or invalid:
        print(f"[SCALE] Audio-only fallback for: {failed + invalid}")


def infer(args: argparse.Namespace) -> None:
    entries = audio_entries(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = repository()
    with tempfile.TemporaryDirectory(prefix="scale-infer-") as temporary:
        workspace = Path(temporary)
        scp = workspace / "audio.scp"
        scp.write_text("\n".join(f"{song_id} {path.resolve()}" for song_id, path in entries) + "\n", encoding="utf-8")
        manifest = workspace / "manifest.jsonl"
        run([sys.executable, "-m", "scale.manifest", "scp", "--input", str(scp),
             "--dataset", "inference", "--output", str(manifest)])
        lyrics = Path(args.lyrics_dir).resolve() if args.lyrics_dir else output / "lyrics_cache"
        if args.use_lyrics == "yes":
            extract_lyrics(entries, lyrics, args, workspace)
        else:
            lyrics = workspace / "lyrics_disabled"
        ssl = workspace / "features"
        run([sys.executable, "-m", "scale.preprocess", "ssl", "--manifest", str(manifest),
             "--output-dir", str(ssl), "--device", args.device, "--provider", "muq",
             "--muq-checkpoint", os.environ.get("MUQ_CHECKPOINT", "OpenMuQ/MuQ-large-msd-iter"), "--fail-fast"])
        prediction = output / "predictions"
        txt = output / "est_txt"
        command = [sys.executable, "-m", "scale.infer", "--config", args.config,
                   "--checkpoint", args.checkpoint, "--manifest", str(manifest),
                   "--feature-dir", f"muq_local={ssl / 'muq_local'}",
                   "--feature-dir", f"muq_global={ssl / 'muq_global'}",
                   "--lyrics-dir", str(lyrics),
                   "--device", args.device, "--output-dir", str(prediction),
                   "--txt-output-dir", str(txt)]
        if args.save_logits:
            command.append("--save-logits")
        run(command)
    print(f"Predictions: {prediction}\nStructure labels: {txt}")


def evaluate(args: argparse.Namespace) -> None:
    annotations = Path(args.annotation_dir).resolve()
    estimates = Path(args.est_dir).resolve()
    if not annotations.is_dir() or not estimates.is_dir():
        raise FileNotFoundError(f"Annotation and estimate directories must exist: {annotations}, {estimates}")
    output = Path(args.metrics_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = "scale.evaluation.partial_annotations" if args.partial_labels else "scale.evaluation.full_annotations"
    command = [sys.executable, "-m", module, "--ann_dir", str(annotations),
               "--est_dir", str(estimates), "--output_dir", str(output)]
    if args.prechorus_policy:
        command += ["--prechorus2what", args.prechorus_policy]
    run(command)
    print(f"Metrics: {output}")


def add_inference(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", action="append")
    source.add_argument("--audio-list")
    source.add_argument("--audio-dir")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-lyrics", required=True, choices=["yes", "no"])
    parser.add_argument("--language", default="English", choices=["English", "Mandarin", "Cantonese"])
    parser.add_argument("--lyrics-dir")
    parser.add_argument("--allow-missing-lyrics", action="store_true")
    parser.add_argument("--config", default=str(repository() / "configs" / "train.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-logits", action="store_true")


def add_evaluation(parser: argparse.ArgumentParser, *, infer_mode: bool = False) -> None:
    parser.add_argument("--annotation-dir", required=True)
    if not infer_mode:
        parser.add_argument("--est-dir", required=True)
        parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--prechorus-policy", choices=["verse", "chorus"], default="verse")
    parser.add_argument("--partial-labels", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCALE inference and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    add_inference(commands.add_parser("infer"))
    add_evaluation(commands.add_parser("evaluate"))
    combined = commands.add_parser("infer-eval")
    add_inference(combined)
    add_evaluation(combined, infer_mode=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args)
        return
    infer(args)
    if args.command == "infer-eval":
        args.est_dir = str(Path(args.output_dir) / "est_txt")
        args.metrics_dir = str(Path(args.output_dir) / "metrics")
        evaluate(args)


if __name__ == "__main__":
    main()
