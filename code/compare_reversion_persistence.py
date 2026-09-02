"""Compare accepted-label revisions without assuming identity numbers are stable.

Later rules allocate the next free integer, so removing an earlier fresh identity
can renumber every later fresh identity. This audit first matches baseline and
candidate identities by maximum whole-movie pixel overlap, then scores matched
lineages in the baseline identity windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import tifffile

from continuity_confidence import score_fixed_windows


def identity_windows(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for identity in sorted(set(map(int, np.unique(labels))) - {0}):
        present = np.flatnonzero(np.any(labels == identity, axis=(1, 2)))
        rows.append({
            "identity": identity,
            "first_frame_index": int(present[0]),
            "last_frame_index": int(present[-1]),
            "named_frames": int(len(present)),
        })
    return pd.DataFrame(rows)


def maximum_overlap_mapping(
        baseline: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    baseline_ids = sorted(set(map(int, np.unique(baseline))) - {0})
    candidate_ids = sorted(set(map(int, np.unique(candidate))) - {0})
    overlap = np.zeros((len(baseline_ids), len(candidate_ids)), np.int64)
    baseline_row = {identity: row for row, identity in enumerate(baseline_ids)}
    candidate_col = {
        identity: col for col, identity in enumerate(candidate_ids)}
    both = (baseline > 0) & (candidate > 0)
    pairs, counts = np.unique(
        np.stack((baseline[both], candidate[both]), axis=1),
        axis=0, return_counts=True)
    for (baseline_id, candidate_id), count in zip(pairs, counts):
        overlap[baseline_row[int(baseline_id)],
                candidate_col[int(candidate_id)]] = int(count)
    baseline_counts = {
        identity: int(np.count_nonzero(baseline == identity))
        for identity in baseline_ids}
    candidate_counts = {
        identity: int(np.count_nonzero(candidate == identity))
        for identity in candidate_ids}
    rows = []
    if overlap.size:
        baseline_rows, candidate_cols = linear_sum_assignment(-overlap)
        for baseline_index, candidate_index in zip(
                baseline_rows, candidate_cols):
            pixels = int(overlap[baseline_index, candidate_index])
            if pixels == 0:
                continue
            baseline_id = baseline_ids[int(baseline_index)]
            candidate_id = candidate_ids[int(candidate_index)]
            rows.append({
                "baseline_identity": baseline_id,
                "candidate_identity": candidate_id,
                "overlap_pixels": pixels,
                "baseline_overlap_fraction":
                    pixels / baseline_counts[baseline_id],
                "candidate_overlap_fraction":
                    pixels / candidate_counts[candidate_id],
            })
    return pd.DataFrame(rows, columns=[
        "baseline_identity", "candidate_identity", "overlap_pixels",
        "baseline_overlap_fraction", "candidate_overlap_fraction"])


def _score_summary(labels: np.ndarray, windows: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame, dict]:
    cells, gaps, pooled = score_fixed_windows(
        labels, windows[[
            "identity", "first_frame_index", "last_frame_index"]])
    pooled["arithmetic_mean_confidence"] = float(cells.confidence.mean())
    return cells, gaps, pooled


def compare(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"label shapes differ: {baseline.shape} != {candidate.shape}")
    mapping = maximum_overlap_mapping(baseline, candidate)
    baseline_windows = identity_windows(baseline)
    candidate_windows = identity_windows(candidate)
    mapped_baseline_ids = set(mapping.baseline_identity.astype(int))
    mapped_candidate_ids = set(mapping.candidate_identity.astype(int))
    removed = sorted(
        set(baseline_windows.identity.astype(int)) - mapped_baseline_ids)
    added = sorted(
        set(candidate_windows.identity.astype(int)) - mapped_candidate_ids)

    remapped = np.zeros_like(candidate)
    for row in mapping.itertuples(index=False):
        remapped[candidate == int(row.candidate_identity)] = \
            int(row.baseline_identity)
    matched_windows = baseline_windows[
        baseline_windows.identity.isin(mapped_baseline_ids)].copy()
    baseline_matched, baseline_matched_gaps, baseline_matched_summary = \
        _score_summary(baseline, matched_windows)
    candidate_matched, candidate_matched_gaps, candidate_matched_summary = \
        _score_summary(remapped, matched_windows)
    baseline_self, baseline_self_gaps, baseline_self_summary = \
        _score_summary(baseline, baseline_windows)
    candidate_self, candidate_self_gaps, candidate_self_summary = \
        _score_summary(candidate, candidate_windows)

    minimum_span = int(np.ceil(0.15 * len(candidate)))
    short_candidate = candidate_windows[
        (candidate_windows.last_frame_index
         - candidate_windows.first_frame_index + 1) < minimum_span].copy()
    candidate_to_baseline = dict(zip(
        mapping.candidate_identity.astype(int),
        mapping.baseline_identity.astype(int)))
    short_candidate["matched_baseline_identity"] = [
        candidate_to_baseline.get(int(identity))
        for identity in short_candidate.identity]

    metrics = {
        "frames": int(len(candidate)),
        "changed_pixels": int(np.count_nonzero(candidate != baseline)),
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != baseline, axis=(1, 2)))),
        "baseline_identity_count": int(len(baseline_windows)),
        "candidate_identity_count": int(len(candidate_windows)),
        "matched_identity_count": int(len(mapping)),
        "removed_baseline_identities": removed,
        "added_candidate_identities": added,
        "minimum_nonfragment_span_frames": minimum_span,
        "short_candidate_identities":
            short_candidate.identity.astype(int).tolist(),
        "matched_fixed_windows": {
            "baseline": baseline_matched_summary,
            "candidate": candidate_matched_summary,
            "arithmetic_mean_confidence_delta": float(
                candidate_matched_summary["arithmetic_mean_confidence"]
                - baseline_matched_summary["arithmetic_mean_confidence"]),
            "pooled_confidence_delta": float(
                candidate_matched_summary["pooled_confidence"]
                - baseline_matched_summary["pooled_confidence"]),
            "gap_frames_delta": int(
                candidate_matched_summary["gap_frames"]
                - baseline_matched_summary["gap_frames"]),
        },
        "self_windows": {
            "baseline": baseline_self_summary,
            "candidate": candidate_self_summary,
            "arithmetic_mean_confidence_delta": float(
                candidate_self_summary["arithmetic_mean_confidence"]
                - baseline_self_summary["arithmetic_mean_confidence"]),
            "pooled_confidence_delta": float(
                candidate_self_summary["pooled_confidence"]
                - baseline_self_summary["pooled_confidence"]),
            "gap_frames_delta": int(
                candidate_self_summary["gap_frames"]
                - baseline_self_summary["gap_frames"]),
        },
    }
    return {
        "mapping": mapping,
        "short_candidate": short_candidate,
        "baseline_matched": baseline_matched,
        "candidate_matched": candidate_matched,
        "baseline_matched_gaps": baseline_matched_gaps,
        "candidate_matched_gaps": candidate_matched_gaps,
        "baseline_self": baseline_self,
        "candidate_self": candidate_self,
        "baseline_self_gaps": baseline_self_gaps,
        "candidate_self_gaps": candidate_self_gaps,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        tifffile.imread(args.baseline), tifffile.imread(args.candidate))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()

