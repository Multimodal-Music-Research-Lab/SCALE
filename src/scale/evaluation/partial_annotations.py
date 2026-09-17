# -*- coding: utf-8 -*-
"""
Eval for partial annotations (SheetSageMSA partial labels) by evaluating ONLY within
the annotated time window per track.

Key idea (Scheme A):
- For each track, derive the eval window [ann_start, ann_end] from annotation MSA.
- Slice BOTH ann_msa and est_msa into that window.
- Then compute compute_results / mir_eval / acc / iou on the sliced MSAs.

Usage:
  python eval_partial_window.py \
    --ann_dir /path/to/ann_txt_dir \
    --est_dir /path/to/est_txt_dir \
    --output_dir ./eval_out \
    [--prechorus2what verse|chorus] \
    [--merge_continuous_segments] \
    [--trim_to_ann_window]   (default true; keep it) \
    [--ann_window_min_dur 0.5]

Notes:
- ann_dir and est_dir should both contain per-track .txt files in SCALE MSA txt format
  (e.g. lines like: "0.000000 intro" ... last line "219.096375 end")
"""

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import argparse
import os
import bisect
from collections import defaultdict
from pathlib import Path

import mir_eval
import pandas as pd
from loguru import logger
from tqdm import tqdm

from scale.data.custom_types import MsaInfo
from scale.data.label2id import LABEL_TO_ID
from scale.data.msa_info_utils import load_msa_info
from msaf.eval import compute_results
from scale.postprocessing.calc_acc import cal_acc
from scale.postprocessing.calc_iou import cal_iou


LEGAL_LABELS = {
    "end",
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
    "pre-chorus",
}


def to_inters_labels(msa_info: MsaInfo):
    # msa_info: [(t0, lab0), (t1, lab1), ..., (t_end, "end")]
    label_ids = np.array([LABEL_TO_ID[x[1]] for x in msa_info[:-1]])
    times = [x[0] for x in msa_info]
    inters = np.column_stack([np.array(times[:-1]), np.array(times[1:])])
    return inters, label_ids


def merge_continuous_segments(segments: MsaInfo) -> MsaInfo:
    """
    Merge continuous segments with the same label.

    Input:  [(start_time, label), ... , (end_time, 'end')]
    Output: same format
    """
    if not segments or len(segments) < 2:
        return segments

    merged = []
    current_start = segments[0][0]
    current_label = segments[0][1]

    for i in range(1, len(segments)):
        time, label = segments[i]

        if label == "end":
            if current_label != "end":
                merged.append((current_start, current_label))
            merged.append((time, "end"))
            break

        if label != current_label:
            merged.append((current_start, current_label))
            current_start = time
            current_label = label

    return merged


def slice_msa(msa: MsaInfo, start: float, end: float) -> MsaInfo:
    """
    Slice a full-song msa into [start, end] window.

    - Find active label at 'start' (using right-bisect).
    - Keep boundaries inside (start, end).
    - Force end with (end, 'end').
    """
    if not msa:
        return [(start, "end"), (end, "end")]
    assert msa[-1][1] == "end", "msa must end with 'end'"
    if end <= start:
        raise ValueError(f"Bad slice window: start={start}, end={end}")

    times = [t for t, _ in msa]

    # If start is beyond msa end, return trivial
    if start >= times[-1]:
        return [(start, "end"), (end, "end")]

    idx = bisect.bisect_right(times, start) - 1
    idx = max(idx, 0)
    active_label = msa[idx][1]

    # If the active label is "end" (rare), make it trivial
    if active_label == "end":
        return [(start, "end"), (end, "end")]

    sliced = [(start, active_label)]

    for t, lab in msa[idx + 1 :]:
        if t <= start:
            continue
        if t >= end:
            break
        sliced.append((t, lab))

    sliced.append((end, "end"))
    return merge_continuous_segments(sliced)


def normalize_labels(msa: MsaInfo):
    """
    Sanity-check labels and coerce unknown labels into raising errors early.
    """
    for t, lab in msa:
        if lab not in LEGAL_LABELS:
            raise ValueError(f"Illegal label {lab!r} at t={t}")
    return msa


def apply_prechorus_mapping(msa: MsaInfo, prechorus2what: str | None) -> MsaInfo:
    if prechorus2what is None:
        return msa
    if prechorus2what == "verse":
        return [(t, "verse") if l == "pre-chorus" else (t, l) for t, l in msa]
    if prechorus2what == "chorus":
        return [(t, "chorus") if l == "pre-chorus" else (t, l) for t, l in msa]
    raise ValueError(f"Unknown prechorus2what: {prechorus2what}")


def get_ann_window(ann_msa: MsaInfo, ann_window_min_dur: float = 0.5) -> tuple[float, float]:
    """
    Derive evaluation window from annotation MSA.

    For your SheetSageMSA partial labels, ann_msa typically represents just one segment and ends with 'end'.
    We will:
      start = ann_msa[0][0]
      end   = ann_msa[-1][0]
    """
    if not ann_msa or len(ann_msa) < 2:
        raise ValueError("Annotation MSA too short to derive a window.")
    if ann_msa[-1][1] != "end":
        raise ValueError("Annotation MSA must end with 'end' to derive a window.")

    start = float(ann_msa[0][0])
    end = float(ann_msa[-1][0])

    if end - start < ann_window_min_dur:
        raise ValueError(f"Annotation window too short: {end-start:.4f}s")
    return start, end


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_dir", type=str, required=True, help="Dir of GT .txt MSA files")
    p.add_argument("--est_dir", type=str, required=True, help="Dir of predicted .txt MSA files")
    p.add_argument("--output_dir", type=str, default="./eval_infer_results")

    p.add_argument("--prechorus2what", type=str, default=None, help="verse|chorus|None")
    p.add_argument("--merge_continuous_segments", action="store_true")
    p.add_argument("--trim_to_ann_window", action="store_true", default=True)
    p.add_argument("--ann_window_min_dur", type=float, default=0.5)

    args = p.parse_args()

    ann_dir = args.ann_dir
    est_dir = args.est_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ann_files = sorted([x for x in os.listdir(ann_dir) if x.endswith(".txt")])
    est_files = sorted([x for x in os.listdir(est_dir) if x.endswith(".txt")])

    common_files = sorted(list(set(ann_files) & set(est_files)))
    logger.info(f"Common number of files: {len(common_files)}")

    results_list = []
    ious = {}

    for fname in tqdm(common_files, desc="eval"):
        try:
            ann_path = os.path.join(ann_dir, fname)
            est_path = os.path.join(est_dir, fname)

            ann_msa = load_msa_info(ann_path)
            est_msa = load_msa_info(est_path)

            # label mapping (optional)
            ann_msa = apply_prechorus_mapping(ann_msa, args.prechorus2what)
            est_msa = apply_prechorus_mapping(est_msa, args.prechorus2what)

            if args.merge_continuous_segments:
                ann_msa = merge_continuous_segments(ann_msa)
                est_msa = merge_continuous_segments(est_msa)

            # sanity
            ann_msa = normalize_labels(ann_msa)
            est_msa = normalize_labels(est_msa)

            # --- Scheme A: evaluate only within annotation window ---
            if args.trim_to_ann_window:
                w_start, w_end = get_ann_window(ann_msa, ann_window_min_dur=args.ann_window_min_dur)
                ann_msa_eval = slice_msa(ann_msa, w_start, w_end)
                est_msa_eval = slice_msa(est_msa, w_start, w_end)
            else:
                ann_msa_eval = ann_msa
                est_msa_eval = est_msa

            ann_inter, ann_labels = to_inters_labels(ann_msa_eval)
            est_inter, est_labels = to_inters_labels(est_msa_eval)

            result = compute_results(
                ann_inter,
                est_inter,
                ann_labels,
                est_labels,
                bins=11,
                est_file="test.txt",
                weight=0.58,
            )

            acc = cal_acc(ann_msa_eval, est_msa_eval, post_digit=3)
            ious[fname] = cal_iou(ann_msa_eval, est_msa_eval)

            # HitRate_1* via mir_eval
            hr1_p, hr1_r, hr1_f = mir_eval.segment.detection(
                ann_inter, est_inter, window=1, trim=False
            )
            result["HitRate_1P"], result["HitRate_1R"], result["HitRate_1F"] = hr1_p, hr1_r, hr1_f

            result.update({"id": Path(fname).stem})
            result.update({"acc": acc})
            if args.trim_to_ann_window:
                result.update({"ann_win_start": w_start, "ann_win_end": w_end, "ann_win_dur": w_end - w_start})

            for v in ious[fname]:
                result.update({f"iou-{v['label']}": v["iou"]})

            # clean fields
            if "track_id" in result:
                del result["track_id"]
            if "ds_name" in result:
                del result["ds_name"]

            results_list.append(result)

        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            continue

    df = pd.DataFrame(results_list)
    df.to_csv(f"{output_dir}/eval_infer.csv", index=False)

    # overall IOU summary
    intsec_dur_total = defaultdict(float)
    uni_dur_total = defaultdict(float)

    for tid, value in ious.items():
        for item in value:
            label = item["label"]
            intsec_dur_total[label] += item.get("intsec_dur", 0)
            uni_dur_total[label] += item.get("uni_dur", 0)

    total_intsec = sum(intsec_dur_total.values())
    total_uni = sum(uni_dur_total.values())
    overall_iou = total_intsec / total_uni if total_uni > 0 else 0.0

    class_ious = {}
    for label in intsec_dur_total:
        intsec = intsec_dur_total[label]
        uni = uni_dur_total[label]
        class_ious[label] = intsec / uni if uni > 0 else 0.0

    summary = pd.DataFrame(
        [
            {
                "num_samples": len(df),
                "HR.5F": df["HitRate_0.5F"].mean() if len(df) else np.nan,
                "HR3F": df["HitRate_3F"].mean() if len(df) else np.nan,
                "HR1F": df["HitRate_1F"].mean() if len(df) else np.nan,
                "PWF": df["PWF"].mean() if len(df) else np.nan,
                "Sf": df["Sf"].mean() if len(df) else np.nan,
                "acc": df["acc"].mean() if len(df) else np.nan,
                "iou": overall_iou,
                **{f"iou_{k}": v for k, v in class_ious.items()},
            }
        ]
    )

    with open(f"{output_dir}/eval_infer_summary.md", "w", encoding="utf-8") as f:
        print(summary.to_markdown(index=False), file=f)

    summary.to_csv(f"{output_dir}/eval_infer_summary.csv", index=False)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
