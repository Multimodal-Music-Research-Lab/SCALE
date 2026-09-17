from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scale.io_utils import feature_array
from scale.io_utils import read_jsonl
from scale.lyrics_chorus import ChorusLyricsTokenizer, pad_lyrics_batch
from scale.data.base_dataset import Dataset as BaseSCALEDataset


class Dataset(BaseSCALEDataset):
    """Legacy labels with independent, native-rate modality tensors."""

    def __init__(self, dataset_abstracts, hparams):
        self.feature_specs = {
            str(name): {"dim": int(spec["dim"]), "rate": float(spec["rate"])}
            for name, spec in hparams.feature_specs.items()
        }
        self.ssl_feature_names = list(hparams.ssl_feature_names)
        self.ssl_feature_dims = [self.feature_specs[name]["dim"] for name in self.ssl_feature_names]
        self.strict_features = bool(hparams.get("strict_features", True))
        self.strict_lyrics = bool(hparams.get("strict_lyrics", True))
        self.use_lyrics = bool(hparams.get("use_lyrics", True))
        self.lyrics_tokenizer = None
        if self.use_lyrics:
            self.lyrics_tokenizer = ChorusLyricsTokenizer(
                str(hparams.lyrics_tokenizer_path),
                int(hparams.lyrics_max_block_tokens),
                int(hparams.lyrics_max_line_tokens),
                int(hparams.lyrics_max_blocks),
                int(hparams.lyrics_max_lines_per_block),
                str(hparams.lyrics_line_token),
            )
        self.scale_dirs = {
            item["internal_tmp_id"]: {str(k): str(v) for k, v in item["scale_feature_dirs"].items()}
            for item in dataset_abstracts
        }
        self.lyrics_dirs = {
            item["internal_tmp_id"]: Path(str(item["lyrics_json_dir"]))
            for item in dataset_abstracts
        }
        self.lyrics_allowlists = {}
        for item in dataset_abstracts:
            internal_id = item["internal_tmp_id"]
            if not self.use_lyrics:
                self.lyrics_allowlists[internal_id] = None
                continue
            allowlist_path = item.get("lyrics_song_ids_path")
            if not allowlist_path:
                self.lyrics_allowlists[internal_id] = None
                continue
            path = Path(str(allowlist_path))
            if not path.is_file():
                raise FileNotFoundError(f"Missing lyrics song allowlist: {path}")
            self.lyrics_allowlists[internal_id] = {
                line.strip().split("\t", 1)[0]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
        self.lyrics_cache = {}
        self.manifest_records = {}
        self.allowed_chunks = {}
        for abstract in dataset_abstracts:
            internal_id = abstract["internal_tmp_id"]
            records = {
                record["chunk_id"]: record for record in read_jsonl(abstract["scale_manifest"])
            }
            self.manifest_records[internal_id] = records
            self.allowed_chunks[internal_id] = set(records)
        super().__init__(dataset_abstracts=dataset_abstracts, hparams=hparams)
        self.valid_data_ids = [
            spec for spec in self.valid_data_ids if spec[2] in self.allowed_chunks[spec[0]]
        ]
        if self.strict_features:
            before = len(self.valid_data_ids)
            self.valid_data_ids = [spec for spec in self.valid_data_ids if self._all_features_exist(spec)]
            removed = before - len(self.valid_data_ids)
            if removed:
                raise RuntimeError(
                    f"{removed}/{before} dataset entries lack SCALE features. "
                    "Run preprocess validate before training."
                )

    def _all_features_exist(self, sample_spec) -> bool:
        internal_id, _, chunk_id, _ = sample_spec
        directories = self.scale_dirs[internal_id]
        audio_ready = all(
            (Path(directory) / f"{chunk_id}.npy").exists() for directory in directories.values()
        )
        return audio_ready

    def __getitem__(self, index):
        sample_spec = self.valid_data_ids[index]
        item = super().__getitem__(index)
        if item is None:
            return None
        internal_id, _, chunk_id, _ = sample_spec
        ssl = item.pop("input_embedding")
        if ssl.shape[-1] != sum(self.ssl_feature_dims):
            raise ValueError(
                f"SSL dim mismatch for {chunk_id}: {ssl.shape[-1]} != {sum(self.ssl_feature_dims)}"
            )
        splits = np.split(ssl, np.cumsum(self.ssl_feature_dims)[:-1], axis=-1)
        features = {name: value for name, value in zip(self.ssl_feature_names, splits)}
        directories = self.scale_dirs[internal_id]
        for name, directory in directories.items():
            if name in features:
                continue
            features[name] = feature_array(
                Path(directory) / f"{chunk_id}.npy", self.feature_specs[name]["dim"]
            )
        record = self.manifest_records[internal_id][chunk_id]
        item["features"] = features
        item["chunk_id"] = chunk_id
        if self.use_lyrics:
            lyrics_path = self.lyrics_dirs[internal_id] / f"{record['song_id']}.json"
            allowlist = self.lyrics_allowlists[internal_id]
            lyrics_enabled = allowlist is None or record["song_id"] in allowlist
            cache_key = (str(lyrics_path), float(record["start_sec"]), lyrics_enabled)
            lyrics = self.lyrics_cache.get(cache_key)
            if lyrics is None:
                try:
                    lyrics = (
                        self.lyrics_tokenizer.encode(
                            lyrics_path, chunk_start_sec=float(record["start_sec"])
                        )
                        if lyrics_enabled and lyrics_path.is_file()
                        else self.lyrics_tokenizer.empty()
                    )
                except (OSError, TypeError, ValueError):
                    lyrics = self.lyrics_tokenizer.empty()
                self.lyrics_cache[cache_key] = lyrics
            if lyrics is None:
                lyrics = self.lyrics_tokenizer.empty()
            item["lyrics"] = lyrics
        return item

    @staticmethod
    def _pad_features(batch: list[dict], feature_specs: dict) -> tuple[dict, dict]:
        tensors = {}
        lengths = {}
        for name, spec in feature_specs.items():
            values = [item["features"][name] for item in batch]
            max_length = max(value.shape[0] for value in values)
            padded = np.zeros((len(values), max_length, spec["dim"]), dtype=np.float32)
            feature_lengths = np.zeros(len(values), dtype=np.int64)
            for index, value in enumerate(values):
                padded[index, : value.shape[0]] = value
                feature_lengths[index] = value.shape[0]
            tensors[name] = torch.from_numpy(padded)
            lengths[name] = torch.from_numpy(feature_lengths)
        return tensors, lengths

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        features, feature_lengths = self._pad_features(batch, self.feature_specs)
        max_sequence = max(item["mask"].shape[0] for item in batch)
        batch_size = len(batch)
        masks = np.ones((batch_size, max_sequence), dtype=bool)
        boundaries = np.zeros((batch_size, max_sequence), dtype=np.float32)
        wide_boundaries = np.zeros_like(boundaries)
        functions = np.zeros((batch_size, max_sequence, self.hparams.num_classes), dtype=np.float32)
        boundary_mask = np.zeros((batch_size, max_sequence), dtype=bool)
        function_mask = np.zeros((batch_size, max_sequence), dtype=bool)
        sequence_lengths = np.zeros(batch_size, dtype=np.int64)
        for index, item in enumerate(batch):
            length = min(max_sequence, item["mask"].shape[0])
            sequence_lengths[index] = length
            masks[index, :length] = item["mask"][:length]
            boundaries[index, :length] = item["true_boundary"][:length]
            wide_boundaries[index, :length] = item["widen_true_boundary"][:length]
            functions[index, :length] = item["true_function"][:length]
            boundary_mask[index, :length] = item.get(
                "boundary_mask", np.zeros(length, dtype=bool)
            )[:length]
            function_mask[index, :length] = item.get(
                "function_mask", np.zeros(length, dtype=bool)
            )[:length]
        result = {
            "data_ids": [item["data_id"] for item in batch],
            "chunk_ids": [item["chunk_id"] for item in batch],
            "features": features,
            "feature_lengths": feature_lengths,
            "masks": torch.from_numpy(masks),
            "true_boundaries": torch.from_numpy(boundaries),
            "widen_true_boundaries": torch.from_numpy(wide_boundaries),
            "true_functions": torch.from_numpy(functions),
            "boundary_mask": torch.from_numpy(boundary_mask),
            "function_mask": torch.from_numpy(function_mask),
            "dataset_ids": torch.tensor([item["dataset_id"] for item in batch], dtype=torch.long),
            "label_id_masks": torch.from_numpy(
                np.stack([item["label_id_mask"] for item in batch], axis=0)[:, None]
            ),
            "msa_infos": [item["msa_info"] for item in batch],
        }
        if not self.use_lyrics:
            return result
        lyrics_batch = pad_lyrics_batch(
            [item["lyrics"] for item in batch], self.lyrics_tokenizer.pad_token_id
        )
        result.update(lyrics_batch)
        positive_ratio = float(self.hparams.lyrics_chorus_positive_ratio)
        negative_ratio = float(self.hparams.lyrics_chorus_negative_ratio)
        chorus_id = int(self.hparams.lyrics_chorus_label_id)
        frame_rate = float(self.hparams.output_logits_frame_rates)
        hard_functions = functions.argmax(axis=-1)
        function_valid = ~function_mask
        starts = lyrics_batch["lyrics_block_starts"].numpy()
        ends = lyrics_batch["lyrics_block_ends"].numpy()
        interval_mask = lyrics_batch["lyrics_block_mask"].numpy()
        labels = np.full(starts.shape, -100, dtype=np.int64)
        weights = np.zeros(starts.shape, dtype=np.float32)
        for sample in range(batch_size):
            for interval in np.flatnonzero(interval_mask[sample]):
                left = max(0, int(np.floor(starts[sample, interval] * frame_rate)))
                right = min(
                    int(sequence_lengths[sample]),
                    int(np.ceil(ends[sample, interval] * frame_rate)),
                )
                if right <= left:
                    continue
                valid = function_valid[sample, left:right]
                assigned = hard_functions[sample, left:right][valid]
                if assigned.size == 0:
                    continue
                chorus_ratio = float((assigned == chorus_id).mean())
                if chorus_ratio >= positive_ratio:
                    labels[sample, interval] = 1
                    weights[sample, interval] = chorus_ratio
                elif 1.0 - chorus_ratio >= negative_ratio:
                    labels[sample, interval] = 0
                    weights[sample, interval] = 1.0 - chorus_ratio
        result["lyrics_chorus_labels"] = torch.from_numpy(labels)
        result["lyrics_chorus_weights"] = torch.from_numpy(weights)
        return result


def move_batch(batch: dict, device: torch.device) -> dict:
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move(child) for key, child in value.items()}
        return value

    return {key: move(value) for key, value in batch.items()}
