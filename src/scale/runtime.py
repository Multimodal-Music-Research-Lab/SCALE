from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from scale.decoder import decode
from scale.metrics import evaluate_track, summarize_evaluation


def load_config(path: str | Path, _seen: set[Path] | None = None):
    """Load a config, optionally merging a relative ``_base_`` config."""
    path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance involving {path}")
    seen.add(path)
    config = OmegaConf.load(path)
    base = config.pop("_base_", None)
    if base is None:
        seen.remove(path)
        return config
    base_path = Path(str(base)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    merged = OmegaConf.merge(load_config(base_path, seen), config)
    seen.remove(path)
    return merged


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_overrides(config, overrides: list[str]):
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        key = re.sub(r"\.(\d+)(?=\.|$)", r"[\1]", key)
        value = OmegaConf.from_dotlist([f"value={raw_value}"]).value
        OmegaConf.update(config, key, value, merge=False)
    return config


def import_object(path: str):
    module_name, name = path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[name])
    return getattr(module, name)


def create_dataset(config, split: str):
    section = config[f"{split}_dataset"]
    dataset_class = import_object(str(section["class_path"]))
    return dataset_class(section["dataset_abstracts"], section["hparams"])


def create_loader(dataset, config, split: str):
    settings = config[f"{split}_dataloader"]
    workers = int(settings.get("num_workers", 0))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(settings.get("batch_size", 1)),
        shuffle=bool(settings.get("shuffle", split == "train")),
        num_workers=workers,
        pin_memory=bool(settings.get("pin_memory", True)),
        drop_last=bool(settings.get("drop_last", False)),
        persistent_workers=workers > 0 and bool(settings.get("persistent_workers", True)),
        prefetch_factor=int(settings.get("prefetch_factor", 2)) if workers > 0 else None,
        collate_fn=dataset.collate_fn,
    )


def load_model(config, checkpoint: str | None, device: torch.device, prefer_ema: bool = True):
    from scale.model import Model

    model = Model(config.model).to(device)
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        key = "ema_model" if prefer_ema and state.get("ema_model") is not None else "model"
        model.load_state_dict(state[key], strict=True)
    return model, state


def _chunk_identity(sample_id: str) -> tuple[str, float]:
    """Recover the ALMA-style song stem and chunk offset from `<song>_<sec>`."""
    song_id, separator, raw_offset = str(sample_id).rpartition("_")
    if not separator:
        return str(sample_id), 0.0
    try:
        return song_id, float(raw_offset)
    except ValueError:
        return str(sample_id), 0.0


def _stitch_annotations(chunks: list[dict]) -> list[tuple[float, str]]:
    result: list[tuple[float, str]] = []
    end_time = 0.0
    for chunk in sorted(chunks, key=lambda item: item["offset_sec"]):
        offset = float(chunk["offset_sec"])
        annotation = chunk["annotation"]
        for local_time, label in annotation[:-1]:
            local_time = float(local_time)
            absolute_time = offset + local_time
            if not result:
                result.append((absolute_time, label))
            elif absolute_time < result[-1][0] - 1e-6:
                raise ValueError("Overlapping annotation chunks are not monotonic")
            elif abs(absolute_time - result[-1][0]) <= 1e-6:
                result[-1] = (result[-1][0], label)
            elif local_time <= 1e-6 and label == result[-1][1]:
                # BaseSCALEDataset inserts a segment start at local time zero
                # when a long song is sliced.  Ignore that artificial seam only;
                # internal boundaries remain valid even when their adjacent
                # segments have identical functional labels.
                continue
            else:
                result.append((absolute_time, label))
        end_time = max(end_time, offset + float(annotation[-1][0]))
    if not result or end_time <= result[-1][0]:
        raise ValueError("Cannot construct a non-empty full-song annotation")
    result.append((end_time, "end"))
    return result


@torch.inference_mode()
def evaluate_loader_detailed(model, loader, config, device: torch.device) -> dict:
    from scale.dataset import move_batch

    model.eval()
    records = []
    predictions = []
    confusion_total = defaultdict(float)
    iou_intersections = defaultdict(float)
    iou_unions = defaultdict(float)
    evaluation_config = config.get("evaluation", {})
    prechorus2what = evaluation_config.get("prechorus2what", None)
    frame_hz = float(config.decode.frame_hz)
    song_chunks = defaultdict(list)
    for batch in loader:
        if batch is None:
            continue
        batch = move_batch(batch, device)
        output = model(batch)
        sample_ids = batch.get("chunk_ids", batch.get("data_ids", []))
        for index, annotation in enumerate(batch["msa_infos"]):
            sample_id = sample_ids[index] if index < len(sample_ids) else f"index-{index}"
            try:
                valid_length = int((~output["padding_mask"][index]).sum().item())
                song_id, offset_sec = _chunk_identity(sample_id)
                song_chunks[song_id].append(
                    {
                        "offset_sec": offset_sec,
                        "boundary_logits": output["boundary_logits"][
                            index, :valid_length
                        ].float().cpu(),
                        "function_logits": output["function_logits"][
                            index, :valid_length
                        ].float().cpu(),
                        "dataset_id": int(batch["dataset_ids"][index].item()),
                        "annotation": annotation,
                    }
                )
            except Exception:
                logger.exception("Skipping evaluation chunk {} after model failure", sample_id)

    for song_id, chunks in sorted(song_chunks.items()):
        try:
            dataset_ids = {chunk["dataset_id"] for chunk in chunks}
            if len(dataset_ids) != 1:
                raise ValueError(f"Mixed dataset IDs for {song_id}: {sorted(dataset_ids)}")
            total_frames = max(
                int(round(chunk["offset_sec"] * frame_hz))
                + chunk["boundary_logits"].shape[0]
                for chunk in chunks
            )
            num_classes = chunks[0]["function_logits"].shape[-1]
            boundary_sum = torch.zeros(total_frames)
            function_sum = torch.zeros(total_frames, num_classes)
            counts = torch.zeros(total_frames)
            for chunk in chunks:
                offset = int(round(chunk["offset_sec"] * frame_hz))
                end = offset + chunk["boundary_logits"].shape[0]
                boundary_sum[offset:end] += chunk["boundary_logits"]
                function_sum[offset:end] += chunk["function_logits"]
                counts[offset:end] += 1
            if not bool((counts > 0).all()):
                raise ValueError(f"Feature chunks leave inference gaps for {song_id}")
            estimate = decode(
                boundary_sum / counts,
                function_sum / counts[:, None],
                int(
                    config.decode.get(
                        "label_mask_dataset_id", next(iter(dataset_ids))
                    )
                ),
                frame_hz,
                float(config.decode.threshold),
                float(config.decode.min_segment_seconds),
            )
            annotation = _stitch_annotations(chunks)
            record, confusion, iou_rows = evaluate_track(
                annotation, estimate, prechorus2what=prechorus2what
            )
            records.append({"id": song_id, **record})
            predictions.append({"id": song_id, "msa": estimate})
            for key, value in confusion.items():
                confusion_total[key] += value
            for item in iou_rows:
                label = item["label"]
                iou_intersections[label] += item.get("intsec_dur", 0.0)
                iou_unions[label] += item.get("uni_dur", 0.0)
        except Exception:
            logger.exception("Skipping evaluation song {} after decode/metric failure", song_id)
    summary = summarize_evaluation(records, confusion_total, iou_intersections, iou_unions)
    return {
        "summary": summary,
        "records": records,
        "predictions": predictions,
        "confusion": dict(confusion_total),
    }


@torch.inference_mode()
def evaluate_loader(model, loader, config, device: torch.device) -> dict[str, float]:
    """Training-compatible three-metric view of the full ALMA-style evaluation."""
    summary = evaluate_loader_detailed(model, loader, config, device)["summary"]

    def finite_or_zero(value: float) -> float:
        value = float(value)
        return value if np.isfinite(value) else 0.0

    return {
        "HR0.5F": finite_or_zero(summary["HR.5F"]),
        "HR3F": finite_or_zero(summary["HR3F"]),
        "ACC": finite_or_zero(summary["acc"]),
    }


def cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    def ratio(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
