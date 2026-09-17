from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import mir_eval
import numpy as np
import pandas as pd

from scale.data.label2id import LABEL_TO_ID
from scale.postprocessing.calc_acc import cal_acc
from scale.postprocessing.calc_iou import cal_iou


EVAL_LABELS = (
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
    "pre-chorus",
)


def apply_prechorus_mapping(
    msa: list[tuple[float, str]], target: str | None
) -> list[tuple[float, str]]:
    if target is None:
        return list(msa)
    if target not in {"verse", "chorus"}:
        raise ValueError(f"prechorus2what must be verse, chorus, or null; got {target!r}")
    return [(time, target if label == "pre-chorus" else label) for time, label in msa]


def _intervals_and_labels(msa: list[tuple[float, str]]) -> tuple[np.ndarray, np.ndarray]:
    if len(msa) < 2 or msa[-1][1] != "end":
        raise ValueError("MSA must contain at least one segment and finish with an end marker")
    times = np.asarray([float(time) for time, _ in msa], dtype=np.float64)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError(f"MSA boundary times must be finite and strictly increasing: {times}")
    labels = [label for _, label in msa[:-1]]
    unknown = sorted(set(labels) - set(LABEL_TO_ID))
    if unknown:
        raise ValueError(f"Unknown MSA labels: {unknown}")
    return np.column_stack([times[:-1], times[1:]]), np.asarray(
        [LABEL_TO_ID[label] for label in labels]
    )


def duration_confusion(
    annotation: list[tuple[float, str]],
    estimate: list[tuple[float, str]],
    digits: int = 3,
) -> defaultdict[tuple[str, str], float]:
    scale = 10**digits
    ann_times = [int(round(time, digits) * scale) for time, _ in annotation]
    est_times = [int(round(time, digits) * scale) for time, _ in estimate]
    common_start = max(ann_times[0], est_times[0])
    common_end = min(ann_times[-1], est_times[-1])
    confusion: defaultdict[tuple[str, str], float] = defaultdict(float)
    if common_end <= common_start:
        return confusion

    points = sorted(
        {common_start, common_end}
        | {time for time in ann_times if common_start <= time <= common_end}
        | {time for time in est_times if common_start <= time <= common_end}
    )
    for left, right in zip(points[:-1], points[1:]):
        ann_label = annotation[bisect.bisect_right(ann_times, left) - 1][1]
        est_label = estimate[bisect.bisect_right(est_times, left) - 1][1]
        if ann_label != "end" and est_label != "end":
            confusion[(ann_label, est_label)] += (right - left) / scale
    return confusion


def classification_metrics_from_confusion(confusion: dict) -> dict[str, dict[str, float]]:
    labels = sorted(
        set(EVAL_LABELS)
        | {gold for gold, _ in confusion}
        | {predicted for _, predicted in confusion}
    )
    metrics = {}
    for label in labels:
        true_positive = float(confusion.get((label, label), 0.0))
        support = float(sum(value for (gold, _), value in confusion.items() if gold == label))
        predicted = float(
            sum(value for (_, pred), value in confusion.items() if pred == label)
        )
        precision = true_positive / predicted if predicted > 0 else np.nan
        recall = true_positive / support if support > 0 else np.nan
        if support > 0 and predicted > 0 and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        elif support > 0:
            f1 = 0.0
        else:
            f1 = np.nan
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_dur": support,
            "pred_dur": predicted,
        }
    return metrics


def flatten_classification_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{metric}_{label.replace('-', '_')}": value
        for label, values in metrics.items()
        for metric, value in values.items()
    }


def evaluate_track(
    annotation: list[tuple[float, str]],
    estimate: list[tuple[float, str]],
    prechorus2what: str | None = None,
) -> tuple[dict[str, float], defaultdict[tuple[str, str], float], list[dict]]:
    # Match ALMA/manual50: map both sides before every boundary, label, ACC and IoU metric.
    annotation = apply_prechorus_mapping(annotation, prechorus2what)
    estimate = apply_prechorus_mapping(estimate, prechorus2what)
    ann_intervals, ann_labels = _intervals_and_labels(annotation)
    est_intervals, est_labels = _intervals_and_labels(estimate)

    import scipy

    scipy.inf = np.inf
    from msaf.eval import compute_results

    result = compute_results(
        ann_inter=ann_intervals,
        est_inter=est_intervals,
        ann_labels=ann_labels,
        est_labels=est_labels,
        bins=11,
        est_file="scale",
        weight=0.58,
    )
    hr1_p, hr1_r, hr1_f = mir_eval.segment.detection(
        ann_intervals, est_intervals, window=1.0, trim=False
    )
    result["HitRate_1P"] = hr1_p
    result["HitRate_1R"] = hr1_r
    result["HitRate_1F"] = hr1_f
    result["acc"] = cal_acc(annotation, estimate, post_digit=3)
    result.pop("track_id", None)
    result.pop("ds_name", None)

    confusion = duration_confusion(annotation, estimate, digits=3)
    result.update(flatten_classification_metrics(classification_metrics_from_confusion(confusion)))
    iou_rows = cal_iou(annotation, estimate)
    for item in iou_rows:
        result[f"iou-{item['label']}"] = item["iou"]
    return result, confusion, iou_rows


def summarize_evaluation(
    records: list[dict],
    confusion: dict[tuple[str, str], float],
    iou_intersections: dict[str, float],
    iou_unions: dict[str, float],
) -> dict[str, float]:
    frame = pd.DataFrame(records)

    def mean(column: str) -> float:
        return float(frame[column].mean()) if len(frame) and column in frame else np.nan

    total_intersection = float(sum(iou_intersections.values()))
    total_union = float(sum(iou_unions.values()))
    summary = {
        "num_samples": len(frame),
        "HR.5F": mean("HitRate_0.5F"),
        "HR3F": mean("HitRate_3F"),
        "HR1F": mean("HitRate_1F"),
        "PWF": mean("PWF"),
        "Sf": mean("Sf"),
        "acc": mean("acc"),
        "iou": total_intersection / total_union if total_union > 0 else 0.0,
    }
    for label in EVAL_LABELS:
        union = float(iou_unions.get(label, 0.0))
        if union > 0:
            summary[f"iou_{label.replace('-', '_')}"] = float(
                iou_intersections.get(label, 0.0)
            ) / union
    summary.update(flatten_classification_metrics(classification_metrics_from_confusion(confusion)))
    return summary


def write_evaluation_outputs(
    output_dir: str | Path,
    records: list[dict],
    summary: dict[str, float],
    confusion: dict[tuple[str, str], float],
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_dir / "eval_infer.csv", index=False)
    summary_frame = pd.DataFrame([summary])
    summary_frame.to_csv(output_dir / "eval_infer_summary.csv", index=False)
    table = summary_frame.to_markdown()
    (output_dir / "eval_infer_summary.md").write_text(table + "\n", encoding="utf-8")

    labels = sorted(
        set(EVAL_LABELS)
        | {gold for gold, _ in confusion}
        | {predicted for _, predicted in confusion}
    )
    confusion_frame = pd.DataFrame(0.0, index=labels, columns=labels)
    for (gold, predicted), duration in confusion.items():
        confusion_frame.loc[gold, predicted] = duration
    confusion_frame.to_csv(output_dir / "duration_confusion.csv")
    row_sums = confusion_frame.sum(axis=1).replace(0, np.nan)
    confusion_frame.div(row_sums, axis=0).to_csv(
        output_dir / "duration_confusion_recall.csv"
    )
    return table


def msa_to_json_segments(msa: list[tuple[float, str]]) -> list[dict]:
    return [
        {"label": msa[index][1], "start": msa[index][0], "end": msa[index + 1][0]}
        for index in range(len(msa) - 1)
    ]


def msa_to_text(msa: list[tuple[float, str]]) -> str:
    return "\n".join(f"{float(time):.6f} {label}" for time, label in msa)


def write_predictions(
    predictions: Iterable[dict],
    json_dir: str | Path | None,
    text_dir: str | Path | None,
) -> None:
    json_dir = Path(json_dir) if json_dir else None
    text_dir = Path(text_dir) if text_dir else None
    if json_dir:
        json_dir.mkdir(parents=True, exist_ok=True)
    if text_dir:
        text_dir.mkdir(parents=True, exist_ok=True)
    for prediction in predictions:
        sample_id = str(prediction["id"])
        msa = prediction["msa"]
        if json_dir:
            (json_dir / f"{sample_id}.json").write_text(
                json.dumps(msa_to_json_segments(msa), ensure_ascii=False, indent=4),
                encoding="utf-8",
            )
        if text_dir:
            (text_dir / f"{sample_id}.txt").write_text(
                msa_to_text(msa), encoding="utf-8"
            )
