import random
import os
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import math
from argparse import Namespace
from scipy.ndimage import gaussian_filter1d
from omegaconf import ListConfig

from .label2id import (
    DATASET_ID_ALLOWED_LABEL_IDS,
    DATASET_LABEL_TO_DATASET_ID,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from .dataset_adapter import DatasetAdapter


class SheetSageMSAAdapter(DatasetAdapter):
    def __init__(self, **kwargs):
        (
            structure_jsonl_paths,
            hparams,
            internal_tmp_id,
            dataset_type,
            input_embedding_dir,
            split_ids_path,
        ) = (
            kwargs["structure_jsonl_paths"],
            kwargs["hparams"],
            kwargs["internal_tmp_id"],
            kwargs["dataset_type"],
            kwargs.get("input_embedding_dir", None),
            kwargs.get("split_ids_path", None),
        )

        self.frame_rates = hparams.frame_rates
        self.hparams = hparams
        self.label_to_id = LABEL_TO_ID
        self.dataset_id_to_dataset_id = DATASET_LABEL_TO_DATASET_ID
        self.id_to_label = ID_TO_LABEL
        self.internal_tmp_id = internal_tmp_id
        self.dataset_type = dataset_type
        self.EPS = 1e-6

        # dataset-specific label mask
        self.dataset_id2label_mask = {}
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            self.dataset_id2label_mask[key] = np.ones(self.hparams.num_classes, dtype=bool)
            self.dataset_id2label_mask[key][allowed_ids] = False

        assert isinstance(structure_jsonl_paths, (ListConfig, tuple, list))

        # load segments per audio id
        self.id2segments = defaultdict(list)
        data = self.load_jsonl(structure_jsonl_paths)

        # input embedding dirs (space-separated)
        self.input_embedding_dir = input_embedding_dir
        all_input_embedding_dirs = input_embedding_dir.split()

        # valid ids that exist in all embedding dirs
        valid_data_ids = self.get_ids_from_dir(all_input_embedding_dirs[0])
        for x in all_input_embedding_dirs:
            valid_data_ids = valid_data_ids.intersection(self.get_ids_from_dir(x))

        # read split ids (song stems)
        split_ids = []
        with open(split_ids_path) as f:
            for line in f:
                if not line.strip():
                    continue
                split_ids.append(line.strip())
        split_ids = set(split_ids)

        # filter valid ids by split membership:
        # embedding id is like "<song_stem>_0" or "<song_stem>_420"
        valid_data_ids = [x for x in valid_data_ids if "_".join(x.split("_")[:-1]) in split_ids]

        valid_data_ids = [
            (internal_tmp_id, dataset_type, x, "SheetSageMSAAdapter")
            for x in valid_data_ids
        ]
        self.valid_data_ids = valid_data_ids

        rng = random.Random(42)
        rng.shuffle(self.valid_data_ids)

        # group jsonl segments by song stem
        for item in data:
            self.id2segments[Path(item["ori_audio_path"]).stem].append(item)

    def get_ids_from_dir(self, dir_path: str):
        ids = os.listdir(dir_path)
        ids = [Path(x).stem for x in ids if x.endswith(".npy")]
        return set(ids)

    def time2frame(self, this_time: float):
        return int(this_time * self.frame_rates)

    def load_jsonl(self, paths):
        data = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
        return data

    def split_and_label(self, query_start, query_end, segments):
        """
        segments: List[dict], keys: segment_start, segment_end, label (List[str])
        """
        points = set([query_start, query_end])
        for seg in segments:
            if query_start <= seg["segment_start"] <= query_end:
                points.add(seg["segment_start"])
            if query_start <= seg["segment_end"] <= query_end:
                points.add(seg["segment_end"])
        sorted_points = sorted(points)

        result = []
        for i in range(len(sorted_points) - 1):
            part_start = sorted_points[i]
            part_end = sorted_points[i + 1]
            labels = []
            for seg in segments:
                if seg["segment_start"] <= part_start and seg["segment_end"] >= part_end:
                    labels.extend(seg["label"])
            if not labels:
                labels = ["NO_LABEL"]
            result.append(
                {"segment_start": part_start, "segment_end": part_end, "labels": labels}
            )

        for idx in range(len(result)):
            result[idx]["labels"] = list(set(result[idx]["labels"]))
        return result

    def merge_small_intervals(self, parts, min_duration=2.0):
        new_parts = []
        i = 0
        while i < len(parts):
            part = parts[i]
            duration = part["segment_end"] - part["segment_start"]
            if duration < min_duration:
                if len(new_parts) > 0 and (i + 1) < len(parts):
                    if random.choice([True, False]):
                        prev = new_parts[-1]
                        prev["segment_end"] = part["segment_end"]
                    else:
                        next_part = parts[i + 1]
                        next_part["segment_start"] = part["segment_start"]
                elif len(new_parts) > 0:
                    prev = new_parts[-1]
                    prev["segment_end"] = part["segment_end"]
                elif (i + 1) < len(parts):
                    next_part = parts[i + 1]
                    next_part["segment_start"] = part["segment_start"]
                i += 1
            else:
                new_parts.append(part)
                i += 1
        return new_parts

    def rounding_time(self, segments, num_decimals=3):
        for idx in range(len(segments)):
            segments[idx]["segment_start"] = round(segments[idx]["segment_start"], num_decimals)
            segments[idx]["segment_end"] = round(segments[idx]["segment_end"], num_decimals)
        return segments

    def get_ids(self):
        return list(self.valid_data_ids)

    def convert_label(self, label: str):
        mapping = {
            "intro": "intro",
            "verse": "verse",
            "chorus": "chorus",
            "bridge": "bridge",
            "inst": "inst",
            "outro": "outro",
            "pre-chorus": "pre-chorus",
            "prechorus": "pre-chorus",
            "silence": "silence",
            "NO_LABEL": "NO_LABEL",
        }
        if label not in mapping:
            raise ValueError(f"Unknown label: {label}")
        return mapping[label]

    def parts_to_label_and_times(self, parts, use_random_tag=True):
        local_times = []
        local_labels = []
        for part in parts:
            local_times.append(part["segment_start"])
            label = random.choice(part["labels"]) if use_random_tag else part["labels"]
            local_labels.append(self.label_to_id[self.convert_label(label)])
        return np.array(local_times), local_labels

    def get_parts(self, utt, query_start, query_end):
        key = "_".join(utt.split("_")[:-1])
        assert key in self.id2segments, f"{key} not in id2segments"
        segments = self.id2segments[key]
        segments = self.rounding_time(segments)
        parts = self.split_and_label(query_start, query_end, segments)
        parts = self.merge_small_intervals(parts, min_duration=2.0)
        parts = self.merge_small_intervals(parts, min_duration=2.0)
        return parts

    def widen_temporal_events(self, events, num_neighbors):
        def theoretical_gaussian_max(sigma):
            return 1 / (np.sqrt(2 * np.pi) * sigma)

        sigma = num_neighbors / 3.0
        smoothed = gaussian_filter1d(events.astype(float), sigma=sigma)
        smoothed /= theoretical_gaussian_max(sigma)
        smoothed = np.clip(smoothed, 0, 1)
        return smoothed

    def get_item_json(self, utt, start_time, end_time):
        embd_list = []
        embd_dirs = self.input_embedding_dir.split()
        for embd_dir in embd_dirs:
            if not Path(embd_dir).exists():
                raise FileNotFoundError(f"Embedding directory {embd_dir} does not exist")
            tmp = np.load(Path(embd_dir) / f"{utt}.npy").squeeze(axis=0)
            embd_list.append(tmp)

        if len(embd_list) > 1:
            shapes = [x.shape for x in embd_list]
            max_shape = max(shapes, key=lambda x: x[0])
            min_shape = min(shapes, key=lambda x: x[0])
            if abs(max_shape[0] - min_shape[0]) > 2:
                raise ValueError(f"Embedding shapes differ too much: {max_shape} vs {min_shape}")
            for i in range(len(embd_list)):
                embd_list[i] = embd_list[i][: min_shape[0], :]

        input_embedding = np.concatenate(embd_list, axis=-1)

        item = self.get_item_json_without_embedding(utt, start_time, end_time)
        if item is None:
            return None
        item["input_embedding"] = input_embedding
        return item

    def get_item_json_without_embedding(self, utt, start_time, end_time):
        SLICE_DUR = int(math.ceil(end_time - start_time))

        local_times, local_labels = self.parts_to_label_and_times(
            self.get_parts(utt, start_time, end_time)
        )

        assert np.all(local_times[:-1] < local_times[1:]), f"time must be sorted, but {utt} is {local_times}"

        local_times = local_times - start_time
        time_L = 0.0
        time_R = float(SLICE_DUR)

        keep_boundarys = (time_L + self.EPS < local_times) & (local_times < time_R - self.EPS)
        if keep_boundarys.sum() <= 0:
            return None

        mask = np.ones([int(SLICE_DUR * self.frame_rates)], dtype=bool)
        mask[self.time2frame(time_L): self.time2frame(time_R)] = False

        true_boundary = np.zeros([int(SLICE_DUR * self.frame_rates)], dtype=float)
        for idx in np.flatnonzero(keep_boundarys):
            true_boundary[self.time2frame(local_times[idx])] = 1

        true_function = np.zeros(
            [int(SLICE_DUR * self.frame_rates), self.hparams.num_classes],
            dtype=float,
        )
        true_function_list = []
        msa_info = []
        last_pos = self.time2frame(time_L)

        for idx in np.flatnonzero(keep_boundarys):
            true_function[last_pos: self.time2frame(local_times[idx]), local_labels[idx - 1]] = 1
            true_function_list.append(int(local_labels[idx - 1]))
            msa_info.append((float(max(local_times[idx - 1], time_L)), str(self.id_to_label[int(local_labels[idx - 1])])))
            last_pos = self.time2frame(local_times[idx])

        true_function[last_pos: self.time2frame(time_R), local_labels[int(np.flatnonzero(keep_boundarys)[-1])]] = 1
        true_function_list.append(int(local_labels[int(np.flatnonzero(keep_boundarys)[-1])]))
        msa_info.append((float(local_times[int(np.flatnonzero(keep_boundarys)[-1])]),
                         str(self.id_to_label[int(local_labels[int(np.flatnonzero(keep_boundarys)[-1])])])))
        msa_info.append((float(time_R), "end"))

        frame_len = int(SLICE_DUR * self.frame_rates)
        boundary_mask = np.zeros([frame_len], dtype=bool)
        function_mask = np.zeros([frame_len], dtype=bool)

        for i in range(len(msa_info) - 1):
            seg_start, seg_label = msa_info[i]
            seg_end, _ = msa_info[i + 1]
            start_frame = self.time2frame(seg_start)
            end_frame = self.time2frame(seg_end)

            is_no_label = (seg_label == "NO_LABEL")
            if is_no_label:
                function_mask[start_frame:end_frame] = True

                left_offset = self.time2frame(seg_start + 4)
                right_offset = self.time2frame(seg_end - 4)
                if i == 0:
                    if right_offset > 0:
                        boundary_mask[0:min(right_offset, frame_len)] = True
                elif i == len(msa_info) - 2:
                    if left_offset < frame_len:
                        boundary_mask[left_offset:frame_len] = True
                elif right_offset > left_offset:
                    boundary_mask[left_offset:right_offset] = True

        return {
            "data_id": self.internal_tmp_id + "_" + f"{utt}_{start_time}",
            "mask": mask,
            "true_boundary": true_boundary,
            "widen_true_boundary": self.widen_temporal_events(true_boundary, num_neighbors=self.hparams.num_neighbors),
            "true_function": true_function,
            "true_function_list": true_function_list,
            "msa_info": msa_info,
            "dataset_id": self.dataset_id_to_dataset_id[self.dataset_type],
            "label_id_mask": self.dataset_id2label_mask[self.dataset_id_to_dataset_id[self.dataset_type]],
            "boundary_mask": boundary_mask,
            "function_mask": function_mask,
        }
