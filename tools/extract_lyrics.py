import argparse
import gc
import json
import os
import re
import shutil
import tempfile
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf
from wtpsplit_lite import SaT


def patch_torch_for_nemo_compat() -> None:
    """Provide small Torch API shims needed by NeMo 2.6 on Torch 2.2."""
    import sys
    import types

    import torch
    import torch.distributed.tensor.parallel as tensor_parallel

    if not hasattr(tensor_parallel, "SequenceParallel"):
        class SequenceParallel(tensor_parallel.ParallelStyle):
            def __init__(self, *args, **kwargs):
                pass

            def _apply(self, module, device_mesh):
                return module

        tensor_parallel.SequenceParallel = SequenceParallel

    if not hasattr(torch.nn, "attention"):
        attention = types.ModuleType("torch.nn.attention")
        attention.SDPBackend = torch.backends.cuda.SDPBackend
        attention.sdpa_kernel = torch.backends.cuda.sdp_kernel
        torch.nn.attention = attention
        sys.modules["torch.nn.attention"] = attention


patch_torch_for_nemo_compat()

from preprocess.tools import F0Extractor, VocalDetector, VocalSeparator
from preprocess.tools.lyric_transcription import LyricTranscriber


PREPROCESS_ROOT = os.environ.get("SOULX_MODEL_ROOT", "pretrained_models/SoulX-Singer-Preprocess")


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def format_timestamp(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_token(text: str) -> str:
    """
    Normalize tokens for matching lyric lines to timestamped words.
    Keep apostrophes in contractions and strip surrounding punctuation.
    """
    text = text.strip().lower()
    text = re.sub(r"^[^\w']+|[^\w']+$", "", text)
    return text


def tokenize_text(text: str) -> list[str]:
    return [normalize_token(tok) for tok in clean_spaces(text).split() if normalize_token(tok)]


def words_to_text(words: list[str], language: str) -> str:
    clean_words = [w for w in words if w and w != "<SP>"]
    if language == "English":
        return " ".join(clean_words)
    return "".join(clean_words)


def build_word_timestamps(words: list[str], durs: list[float], start_ms: int, language: str) -> list[dict]:
    cursor = start_ms
    results = []
    for word, dur in zip(words, durs):
        dur_ms = max(0, int(round(float(dur) * 1000)))
        end_ms = cursor + dur_ms
        if word and word != "<SP>":
            results.append(
                {
                    "start_ms": cursor,
                    "end_ms": end_ms,
                    "start": format_timestamp(cursor),
                    "end": format_timestamp(end_ms),
                    "text": word if language == "English" else word.strip(),
                }
            )
        cursor = end_ms
    return results


def split_segment_text_with_wtpsplit(text: str, sat_model: SaT) -> list[str]:
    text = clean_spaces(text)
    if not text:
        return []

    raw_lines = sat_model.split(
        text,
        stride=128,
        block_size=256,
        weighting="hat",
        treat_newline_as_space=True,
    )
    lines = [clean_spaces(x) for x in raw_lines if clean_spaces(x)]

    if not lines:
        lines = [text]

    return lines


def align_lines_to_words(lines: list[str], words: list[dict]) -> list[dict]:
    """
    Line segmentation preserves the original word order.
    Match each line to consecutive words and use their start and end times.
    """
    clean_words = [deepcopy(w) for w in words if w["text"].strip()]
    word_tokens = [normalize_token(w["text"]) for w in clean_words]

    results = []
    cursor = 0

    for line_idx, line in enumerate(lines):
        line_tokens = tokenize_text(line)
        if not line_tokens:
            continue

        # Start with a slice matching the token count.
        start_idx = cursor
        end_idx = min(cursor + len(line_tokens), len(clean_words))
        candidate_words = clean_words[start_idx:end_idx]
        candidate_tokens = [normalize_token(w["text"]) for w in candidate_words]

        # Extend the slice from the cursor to look for an exact match.
        if candidate_tokens != line_tokens:
            matched = False
            for j in range(start_idx + 1, len(clean_words) + 1):
                candidate_words = clean_words[start_idx:j]
                candidate_tokens = [normalize_token(w["text"]) for w in candidate_words]
                if candidate_tokens == line_tokens:
                    end_idx = j
                    matched = True
                    break

            # Fall back to the token-count slice when no exact match exists.
            if not matched:
                end_idx = min(start_idx + len(line_tokens), len(clean_words))
                candidate_words = clean_words[start_idx:end_idx]

        if not candidate_words:
            continue

        results.append(
            {
                "line_idx_in_segment": line_idx,
                "start_ms": candidate_words[0]["start_ms"],
                "end_ms": candidate_words[-1]["end_ms"],
                "start": format_timestamp(candidate_words[0]["start_ms"]),
                "end": format_timestamp(candidate_words[-1]["end_ms"]),
                "text": clean_spaces(" ".join(w["text"] for w in candidate_words)),
                "words": candidate_words,
            }
        )
        cursor = end_idx

    # Append any remaining words to the final line.
    if cursor < len(clean_words):
        leftover = clean_words[cursor:]
        if results:
            results[-1]["end_ms"] = leftover[-1]["end_ms"]
            results[-1]["end"] = format_timestamp(leftover[-1]["end_ms"])
            results[-1]["words"].extend(leftover)
            results[-1]["text"] = clean_spaces(" ".join(w["text"] for w in results[-1]["words"]))
        else:
            results.append(
                {
                    "line_idx_in_segment": 0,
                    "start_ms": leftover[0]["start_ms"],
                    "end_ms": leftover[-1]["end_ms"],
                    "start": format_timestamp(leftover[0]["start_ms"]),
                    "end": format_timestamp(leftover[-1]["end_ms"]),
                    "text": clean_spaces(" ".join(w["text"] for w in leftover)),
                    "words": leftover,
                }
            )

    return results


class LineLyricsBatchExtractor:
    def __init__(
        self,
        language: str,
        device: str,
        vocal_sep: bool,
        sat_model_name: str,
        sat_tokenizer_name_or_path: str | None = None,
        tmp_root: str | None = None,
    ):
        self.language = language
        self.device = device
        self.vocal_sep = vocal_sep
        self.tmp_root = Path(tmp_root) if tmp_root else None
        if self.tmp_root is not None:
            self.tmp_root.mkdir(parents=True, exist_ok=True)

        self.sat = SaT(
            sat_model_name,
            tokenizer_name_or_path=sat_tokenizer_name_or_path,
            hub_prefix=None,
        )

        self.separator = None
        if self.vocal_sep:
            self.separator = VocalSeparator(
                sep_model_path=f"{PREPROCESS_ROOT}/mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt",
                sep_config_path=f"{PREPROCESS_ROOT}/mel-band-roformer-karaoke/config_karaoke_becruily.yaml",
                der_model_path=f"{PREPROCESS_ROOT}/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
                der_config_path=f"{PREPROCESS_ROOT}/dereverb_mel_band_roformer/dereverb_mel_band_roformer_anvuew.yaml",
                device=self.device,
            )

        self.f0_extractor = F0Extractor(
            model_path=f"{PREPROCESS_ROOT}/rmvpe/rmvpe.pt",
            device=self.device,
        )

        self.transcriber = LyricTranscriber(
            zh_model_path=f"{PREPROCESS_ROOT}/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            en_model_path=f"{PREPROCESS_ROOT}/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
            device=self.device,
        )

    def prepare_vocal(self, audio_path: str, work_dir: Path) -> Path:
        vocal_path = work_dir / "vocal.wav"

        if self.vocal_sep:
            if self.separator is None:
                raise RuntimeError("VocalSeparator is not available in this environment.")
            separated = self.separator.process(audio_path)
            sf.write(vocal_path, separated.vocals_dereverbed.T, separated.sample_rate)
        else:
            vocal, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            sf.write(vocal_path, vocal, sample_rate)

        return vocal_path

    def process_one(self, audio_path: Path) -> dict:
        tmp_base_dir = str(self.tmp_root) if self.tmp_root is not None else None

        with tempfile.TemporaryDirectory(prefix=f"{audio_path.stem}_", dir=tmp_base_dir) as tmp_dir:
            work_dir = Path(tmp_dir)
            vocal_path = self.prepare_vocal(str(audio_path), work_dir)

            vocal_f0 = self.f0_extractor.process(
                str(vocal_path),
                f0_path=str(vocal_path).replace(".wav", "_f0.npy"),
            )

            detector = VocalDetector(cut_wavs_output_dir=str(work_dir / "cut_wavs"))
            raw_segments = detector.process(str(vocal_path), f0=vocal_f0)

            segments = []
            for seg in raw_segments:
                segment_f0_path = seg["wav_fn"].replace(".wav", "_f0.npy")
                self.f0_extractor.process(seg["wav_fn"], f0_path=segment_f0_path)

                words, durs = self.transcriber.process(seg["wav_fn"], self.language)
                text = words_to_text(words, self.language)
                if not text:
                    continue

                start_ms = int(seg["start_time_ms"])
                end_ms = int(seg["end_time_ms"])
                word_items = build_word_timestamps(words, durs, start_ms, self.language)

                lines = split_segment_text_with_wtpsplit(text, self.sat)
                line_items = align_lines_to_words(lines, word_items)

                segments.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "start": format_timestamp(start_ms),
                        "end": format_timestamp(end_ms),
                        "text": text,
                        "words": word_items,
                        "lines": line_items,
                    }
                )

            return {
                "audio_path": str(audio_path),
                "language": self.language,
                "vocal_sep": self.vocal_sep,
                "segments": segments,
            }

def read_audio_files_from_scp(scp_path: Path) -> list[Path]:
    audio_files = []
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            audio_files.append(Path(line))
    return audio_files

def get_audio_files(input_dir: Path, recursive: bool) -> list[Path]:
    audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in files if p.is_file() and p.suffix.lower() in audio_exts)


def build_output_path(
    audio_path: Path,
    input_dir: Path | None,
    output_dir: Path | None,
    from_scp: bool = False,
) -> Path:
    if output_dir is None:
        return audio_path.with_suffix(".json")

    if from_scp:
        return output_dir / f"{audio_path.stem}.json"

    rel = audio_path.relative_to(input_dir)
    return (output_dir / rel).with_suffix(".json")

def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "torch.cuda.outofmemoryerror" in msg
        or "out of memory" in msg and "cuda" in msg
    )


def clear_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    gc.collect()


def append_oom_log(log_path: Path, audio_path: Path, exc: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "audio_path": str(audio_path),
        "error_type": "cuda_oom",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch extract SoulX-Singer lyric segments, then split each segment into line-level lyrics with timestamps."
    )
    parser.add_argument("--input_dir", default=None, help="Directory containing audio files.")
    parser.add_argument("--scp_path", default=None, help="Path to .scp file listing audio paths, one per line.")
    parser.add_argument("--output_dir", default=None, help="Directory to save final json files. Default: save next to wav.")
    parser.add_argument(
        "--oom_log_path",
        default=None,
        help="Path to write OOM skip logs as jsonl. Default: <output_dir>/_oom_skip_log.jsonl",
    )
    parser.add_argument("--language", default="Mandarin", choices=["Mandarin", "Cantonese", "English"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vocal_sep", type=parse_bool, default=True)
    parser.add_argument("--recursive", type=parse_bool, default=False, help="Whether to search wav files recursively.")
    parser.add_argument(
        "--tmp_root",
        default="/tmp/soulxsinger_tmp",
        help="Writable directory for temporary intermediate files.",
    )
    parser.add_argument(
        "--sat_model",
        default=os.environ.get("SAT_MODEL", str(Path(__file__).resolve().parents[1] / "models/sat-3l-sm")),
        help="Path to wtpsplit-lite SaT model.",
    )
    parser.add_argument(
        "--sat_tokenizer",
        default=os.environ.get("SAT_TOKENIZER", str(Path(__file__).resolve().parents[1] / "models/xlm-roberta-base")),
        help="Path to tokenizer used by SaT.",
    )
    args = parser.parse_args()

    if args.input_dir is None and args.scp_path is None:
        raise ValueError("You must provide either --input_dir or --scp_path")

    if args.input_dir is not None and args.scp_path is not None:
        raise ValueError("Please provide only one of --input_dir or --scp_path")

    input_dir = Path(args.input_dir) if args.input_dir else None
    scp_path = Path(args.scp_path) if args.scp_path else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if input_dir is not None and not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if scp_path is not None and not scp_path.exists():
        raise FileNotFoundError(f"SCP file not found: {scp_path}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    oom_log_path = (
        Path(args.oom_log_path)
        if args.oom_log_path
        else (output_dir / "_oom_skip_log.jsonl" if output_dir is not None else Path.cwd() / "_oom_skip_log.jsonl")
    )

    if scp_path is not None:
        audio_files = read_audio_files_from_scp(scp_path)
        from_scp = True
    else:
        audio_files = get_audio_files(input_dir, args.recursive)
        from_scp = False

    if not audio_files:
        raise RuntimeError("No audio files found from the given input source")

    extractor = LineLyricsBatchExtractor(
        language=args.language,
        device=args.device,
        vocal_sep=args.vocal_sep,
        sat_model_name=args.sat_model,
        sat_tokenizer_name_or_path=args.sat_tokenizer,
        tmp_root=args.tmp_root,
    )

    for audio_path in audio_files:
        output_json = build_output_path(
            audio_path=audio_path,
            input_dir=input_dir,
            output_dir=output_dir,
            from_scp=from_scp,
        )

        if output_json.exists():
            print(f"[SKIP] {audio_path} -> {output_json} already exists")
            continue

        print(f"[PROCESS] {audio_path}")

        try:
            result = extractor.process_one(audio_path)

            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[DONE] {output_json}")

        except Exception as e:
            if is_cuda_oom_error(e):
                print(f"[OOM-SKIP] {audio_path}")
                print(f"[OOM-ERROR] {e}")
                append_oom_log(oom_log_path, audio_path, e)
                clear_cuda_memory()
                continue
            raise

        finally:
            clear_cuda_memory()


if __name__ == "__main__":
    main()