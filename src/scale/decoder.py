from __future__ import annotations

from bisect import bisect_right

import numpy as np
import torch
import torch.nn.functional as F

from scale.data.label2id import DATASET_ID_ALLOWED_LABEL_IDS, ID_TO_LABEL


def _select_peaks(scores: torch.Tensor, threshold: float, min_distance: int) -> list[int]:
    if scores.numel() < 3:
        return []
    radius = max(1, min_distance // 2)
    maxima = F.max_pool1d(
        scores[None, None], kernel_size=2 * radius + 1, stride=1, padding=radius
    )[0, 0]
    candidates = torch.nonzero((scores >= maxima) & (scores >= threshold), as_tuple=False).flatten()
    order = candidates[torch.argsort(scores[candidates], descending=True)].tolist()
    selected: list[int] = []
    for index in order:
        if index == 0 or index == scores.numel() - 1:
            continue
        if all(abs(index - other) >= min_distance for other in selected):
            selected.append(index)
    return sorted(selected)


def decode(
    boundary_logits: torch.Tensor,
    function_logits: torch.Tensor,
    dataset_id: int,
    frame_hz: float,
    threshold: float,
    min_segment_seconds: float,
) -> list[tuple[float, str]]:
    if boundary_logits.ndim == 2:
        boundary_logits = boundary_logits[0]
    if function_logits.ndim == 3:
        function_logits = function_logits[0]
    boundary_scores = torch.sigmoid(boundary_logits.float())
    min_distance = max(1, int(round(min_segment_seconds * frame_hz)))
    boundaries = [0] + _select_peaks(boundary_scores, threshold, min_distance) + [len(boundary_scores)]
    allowed = DATASET_ID_ALLOWED_LABEL_IDS[int(dataset_id)]
    result: list[tuple[float, str]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        segment_logits = function_logits[start:end].mean(dim=0)
        label_id = allowed[int(segment_logits[allowed].argmax().item())]
        label = ID_TO_LABEL[label_id]
        result.append((float(start / frame_hz), label))
    if not result:
        result.append((0.0, ID_TO_LABEL[allowed[0]]))
    result.append((float(len(boundary_scores) / frame_hz), "end"))
    return result


def duration_weighted_accuracy(
    annotation: list[tuple[float, str]], estimate: list[tuple[float, str]], digits: int = 3
) -> float:
    scale = 10**digits
    ann_times = [int(round(time, digits) * scale) for time, _ in annotation]
    est_times = [int(round(time, digits) * scale) for time, _ in estimate]
    common_start = max(ann_times[0], est_times[0])
    common_end = min(ann_times[-1], est_times[-1])
    if common_end <= common_start:
        return 0.0
    points = sorted(
        {common_start, common_end}
        | {time for time in ann_times if common_start <= time <= common_end}
        | {time for time in est_times if common_start <= time <= common_end}
    )
    correct = total = 0
    for left, right in zip(points[:-1], points[1:]):
        duration = right - left
        ann_label = annotation[bisect_right(ann_times, left) - 1][1]
        est_label = estimate[bisect_right(est_times, left) - 1][1]
        total += duration
        correct += duration if ann_label == est_label else 0
    return float(correct / max(total, 1))


def structural_metrics(annotation, estimate) -> dict[str, float]:
    import mir_eval

    ann_times = np.asarray([time for time, _ in annotation], dtype=float)
    est_times = np.asarray([time for time, _ in estimate], dtype=float)
    ann_intervals = np.column_stack([ann_times[:-1], ann_times[1:]])
    est_intervals = np.column_stack([est_times[:-1], est_times[1:]])
    _, _, hr05_f = mir_eval.segment.detection(
        ann_intervals, est_intervals, window=0.5, trim=False
    )
    _, _, hr3_f = mir_eval.segment.detection(
        ann_intervals, est_intervals, window=3.0, trim=False
    )
    return {
        "HR0.5F": float(hr05_f),
        "HR3F": float(hr3_f),
        "ACC": duration_weighted_accuracy(annotation, estimate),
    }
