# SCALE

SCALE predicts section boundaries and labels from music audio. It uses separate audio pathways for boundary detection and section labeling, with timestamped lyrics providing additional chorus evidence. Explore example predictions on the [SCALE demo](https://multimodal-music-research-lab.github.io/scale-demo/). The static demo source is in [`docs/`](docs/).

## Quick start: inference with pretrained weights

After completing [installation](#installation) and [model setup](#model-setup), place the SCALE checkpoint at `checkpoints/best.ckpt` and run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer.sh \
  --audio /path/to/song.wav \
  --checkpoint checkpoints/best.ckpt \
  --output-dir outputs/example \
  --use-lyrics yes \
  --language English
```

This command extracts audio features and timestamped lyrics, then writes section predictions. Training data and retraining are not required for inference.

**Release status:** a verified pretrained checkpoint will be distributed through [GitHub Releases](https://github.com/Multimodal-Music-Research-Lab/SCALE/releases); no download is available yet. The command above requires a compatible SCALE checkpoint obtained separately. SheetSage-MSA is not yet publicly released; full training reproduction depends on its annotations, splits, and lyric-selection metadata becoming available. Neither weights nor datasets are bundled with the code.

## Installation

Use Linux with an NVIDIA GPU, a CUDA toolkit compatible with PyTorch, and `ffmpeg`/`ffprobe` available on `PATH`. The core environment uses Python 3.11 and PyTorch 2.4.0. Run all commands from the repository root.

```bash
conda create -n scale python=3.11 -y
conda activate scale
python -m pip install torch==2.4.0 torchaudio==2.4.0
python -m pip install packaging ninja wheel
python -m pip install --no-build-isolation -r requirements/scale.txt
python -m pip install -e .
```

`mamba-ssm` and `causal-conv1d` contain CUDA extensions. Their builds must match the installed PyTorch and CUDA versions.

Automatic lyric extraction uses a separate Python 3.10 environment:

```bash
conda create -n scale-lyrics python=3.10 -y
conda activate scale-lyrics
python -m pip install -r requirements/soulx.txt
export SOULX_PY="$(command -v python)"
conda activate scale
```

Keep `SOULX_PY` set to the lyric environment's Python when running SCALE commands. You can skip this environment when using existing timestamped lyric JSON files or `--use-lyrics no`.

## Model setup

Store the downloaded SCALE checkpoint in `checkpoints/best.ckpt`, or pass its location directly with `--checkpoint`. The loader accepts SCALE checkpoint dictionaries containing `model` and optionally `ema_model`; the filename extension does not change the format. A generic Lightning checkpoint is not interchangeable. EMA weights are used when available.

The core model also needs a local [Longformer-base](https://huggingface.co/allenai/longformer-base-4096) snapshot. It is loaded even when lyric extraction is disabled for a full-model checkpoint. Download it with the SCALE environment active:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/longformer-base-4096', local_dir='models/longformer-base-4096')"
```

[MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter) is downloaded automatically on its first use. For offline use, download it in advance and set `MUQ_CHECKPOINT` to the local snapshot directory.

For automatic lyric extraction, also download the [SoulX preprocessing models](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess), [SaT](https://huggingface.co/segment-any-text/sat-3l-sm), and its tokenizer:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Singer-Preprocess', local_dir='models/SoulX-Singer-Preprocess')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('segment-any-text/sat-3l-sm', local_dir='models/sat-3l-sm')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('FacebookAI/xlm-roberta-base', local_dir='models/xlm-roberta-base')"
```

The scripts use the active `python` by default. These environment variables let you keep dependencies and data outside the repository:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCALE_PY` | `python` | Core Python interpreter |
| `SOULX_PY` | `python` | Lyric extraction interpreter; set to the separate environment |
| `SCALE_LONGFORMER_PATH` | `models/longformer-base-4096` | Local Longformer snapshot |
| `SOULX_MODEL_ROOT` | `models/SoulX-Singer-Preprocess` | SoulX preprocessing weights |
| `SAT_MODEL` | `models/sat-3l-sm` | SaT model |
| `SAT_TOKENIZER` | `models/xlm-roberta-base` | SaT tokenizer |
| `MUQ_CHECKPOINT` | `OpenMuQ/MuQ-large-msd-iter` | MuQ model ID or local snapshot |
| `SCALE_DATA_ROOT` | `data/` | Training annotations, splits, and lyrics |
| `SCALE_FEATURE_ROOT` | `features/` | Generated training audio features |
| `SCALE_MANIFEST_ROOT` | `manifests/` | Generated training manifests |

Default directories are relative to the repository. Use absolute paths for overrides. `scripts/env.sh` derives the repository location automatically and respects standard Hugging Face cache settings.

## Inference

For multiple songs, replace `--audio` in the quick-start command with `--audio-dir /path/to/audio` or `--audio-list /path/to/audio.scp`. A list contains either one audio path per line or `song_id /absolute/path/to/audio.wav`. Song IDs must be unique.

To reuse previously extracted lyrics, add `--lyrics-dir /path/to/lyrics`. Each file must be named `<song_id>.json` in the format produced by `scripts/extract_lyrics.sh`. Missing files are extracted automatically. Supported extraction languages are `English`, `Mandarin`, and `Cantonese`.

For audio-only predictions, use `--use-lyrics no`. This skips lyric extraction and supplies empty lyric evidence to the full model. To allow audio-only fallback for extraction failures, add `--allow-missing-lyrics`.

Output files are:

```text
outputs/example/
  predictions/<song_id>.json   # Sections with start/end times in seconds
  est_txt/<song_id>.txt        # Boundary time and section label per line
  lyrics_cache/<song_id>.json  # Created when extracting lyrics without --lyrics-dir
```

MuQ features for this pipeline are temporary. Use the preprocessing commands below to retain features for training. Inference uses the shared HX and SheetSage-MSA output labels; no source embedding is used.

## Evaluation

Provide ground-truth files named `<song_id>.txt`, with one `time_in_seconds label` pair per line and a final `end` boundary. Audio, prediction, and annotation IDs must match.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_eval.sh \
  --audio-dir /path/to/test_audio \
  --annotation-dir /path/to/test_annotations \
  --checkpoint checkpoints/best.ckpt \
  --output-dir outputs/test \
  --use-lyrics yes \
  --language English \
  --prechorus-policy verse
```

To evaluate existing predictions:

```bash
bash scripts/evaluate.sh \
  --annotation-dir /path/to/test_annotations \
  --est-dir outputs/test/est_txt \
  --metrics-dir outputs/test/metrics \
  --prechorus-policy verse
```

Evaluation writes per-song and summary CSV files and a Markdown summary. Pre-chorus is mapped to verse by default; `--prechorus-policy chorus` selects the alternative. For supported partial-annotation files, add `--partial-labels`; unannotated regions must not be treated as silence.

## Training data

`configs/train.yaml` trains on HX and SheetSage-MSA and validates on HX. It uses the original HX timestamps, samples HX four times per epoch, and samples SheetSage-MSA once. HX training and validation IDs must be disjoint from the evaluation set.

SheetSage-MSA is derived from [SheetSage Hooktheory data](https://github.com/chrisdonahue/sheetsage-data). The upstream data alone is not a replacement for the processed SheetSage-MSA annotations. Until the dataset is released, the full training setup cannot be reproduced from this repository alone. The file layout and preprocessing steps below describe the inputs expected by the training code.

Prepare this layout under `data/`, or set `SCALE_DATA_ROOT` to your own directory:

```text
data/
  hx/
    audio.scp
    annotations.jsonl
    train.txt
    val.txt
    lyrics/<song_id>.json
  sheetsagemsa/
    audio.scp
    annotations.jsonl
    train.txt
    lyrics_song_ids.txt
    lyrics/<song_id>.json
```

- `audio.scp`: song IDs paired with local full-audio paths. Audio is obtained separately.
- HX `annotations.jsonl`: one JSON object per song, for example `{"id":"song1","labels":[[0.0,"intro"],[12.0,"verse"],[30.0,"end"]]}`. Use the actual song duration for `end`, original timestamps, and the HX labels used in the experiment.
- SheetSage-MSA `annotations.jsonl`: one annotated interval per object, for example `{"ori_audio_path":"song1.wav","segment_start":12.0,"segment_end":30.0,"label":["verse"]}`. The audio filename stem must match its song ID. Gaps are unlabeled.
- Split files: one song ID per line, without the feature chunk offset or file extension.
- `lyrics_song_ids.txt`: the reviewed SheetSage-MSA song IDs eligible for English lyric supervision. This metadata is part of the pending dataset release; do not substitute an automatically generated all-song list when reproducing the experiment.

### Preprocessing

Generate manifests from local audio lists before extracting MuQ features:

```bash
for dataset in hx sheetsagemsa; do
  bash scripts/build_manifests.sh \
    --input "data/$dataset/audio.scp" \
    --dataset "$dataset" \
    --output "manifests/$dataset.jsonl"

  CUDA_VISIBLE_DEVICES=0 bash scripts/extract_muq.sh \
    --manifest "manifests/$dataset.jsonl" \
    --output-dir "features/$dataset" \
    --device cuda:0 \
    --fail-fast
done
```

These examples use the default directories; substitute your chosen paths if you set the directory overrides. The manifest records local audio paths and chunks of up to 420 seconds. MuQ extraction produces `muq_local/<song_id>_<offset>.npy` and `muq_global/<song_id>_<offset>.npy`, using 30-second local windows and up to 420-second global context. Split membership is applied by the training dataset loader.

Extract training lyrics with the lyric environment configured:

```bash
bash scripts/extract_lyrics.sh \
  --input_dir /path/to/hx_audio \
  --output_dir data/hx/lyrics \
  --language English \
  --device cuda:0
```

Audio filename stems must match the annotation IDs. For SheetSage-MSA, use the reviewed English song selection and save its lyric files to `data/sheetsagemsa/lyrics`. Missing or ineligible lyric evidence is masked during training.

## Training

Once the datasets and features are prepared:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh \
  --seed 42 \
  --device cuda:0 \
  --output-dir outputs/train
```

The script reads `configs/train.yaml`. To use a configuration stored elsewhere, set `SCALE_CONFIG=/path/to/train.yaml`. Model, loss, and optimizer settings are defined in that file.

Repeating the command resumes from `last.pt`, or the latest numbered step checkpoint if `last.pt` is absent. Use a new output directory for a fresh run. To resume explicitly, add `--resume /path/to/last.pt`.

`best.pt` is selected by the validation mean of ACC, HR.5F, and HR3F. The trainer also saves `best_acc.pt`, `best_hr05f.pt`, `last.pt`, and periodic step checkpoints. Select checkpoints using validation results and use the same selection rule for comparisons.

## Repository contents

Only source code, dependency lists, documentation, and the training configuration belong in Git. Checkpoints, downloaded models, local manifests, annotations, splits, lyric JSON files, extracted features, logs, predictions, and metrics are local assets. Their default directories are ignored by `.gitignore` and are created when needed. Keep custom output directories outside the repository or add them to your local ignore rules.

The main entry points are in `scripts/`; model, data, and evaluation implementations are in `src/scale/`. `third_party/soulx_singer/` contains only the preprocessing components needed for lyric extraction.

## License and acknowledgments

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). SCALE builds on SongFormer, MuQ, Longformer, SoulX-Singer, and SaT. Dataset and pretrained model terms apply separately from the code license.
