"""Score field-wide continuity candidates against frozen failure/control cases."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import continuity_confidence as continuity


def identity_present(labels: np.ndarray, identity: int, frame: int) -> bool:
    return bool(np.any(labels[int(frame)] == int(identity)))


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    result = continuity.run_score(upstream_dir, params, out)
    candidate = tifffile.imread(
        Path(params["candidate_labels_path"]) if upstream_dir is None
        else Path(upstream_dir) / "95_A3.tif")
    baseline = tifffile.imread(params["baseline_labels_path"])
    cases = pd.read_csv(params["review_cases_path"])
    failures = cases[cases["case_type"] == "failure"]
    controls = cases[cases["case_type"] == "control"]

    rows = []
    recovered_case_frames = 0
    for case in failures.itertuples(index=False):
        frames = range(int(case.start_frame_index), int(case.end_frame_index) + 1)
        baseline_present = sum(identity_present(
            baseline, int(case.identity), frame) for frame in frames)
        candidate_present = sum(identity_present(
            candidate, int(case.identity), frame) for frame in frames)
        recovered = candidate_present - baseline_present
        recovered_case_frames += recovered
        rows.append({
            "case_id": case.case_id, "case_type": "failure",
            "identity": int(case.identity), "frames": len(frames),
            "baseline_present_frames": baseline_present,
            "candidate_present_frames": candidate_present,
            "recovered_frames": recovered, "passed": recovered > 0,
        })

    exact_controls = 0
    for case in controls.itertuples(index=False):
        identity = int(case.identity)
        frames = [int(case.pre_frame_index), int(case.post_frame_index)]
        exact = all(np.array_equal(
            candidate[frame] == identity, baseline[frame] == identity)
            for frame in frames)
        exact_controls += int(exact)
        rows.append({
            "case_id": case.case_id, "case_type": "control",
            "identity": identity, "frames": len(frames),
            "baseline_present_frames": len(frames),
            "candidate_present_frames": len(frames),
            "recovered_frames": 0, "passed": exact,
        })

    comparisons = pd.read_csv(out.out / "identity_comparison.csv")
    failure_ids = set(failures["identity"].astype(int))
    failure_rows = comparisons[comparisons["identity"].isin(failure_ids)]
    metrics = result["summary"]
    metrics.update({
        "arithmetic_mean_confidence_baseline": float(
            comparisons["confidence_baseline"].mean()),
        "arithmetic_mean_confidence_candidate": float(
            comparisons["confidence_candidate"].mean()),
        "arithmetic_mean_confidence_delta": float(
            comparisons["confidence_delta"].mean()),
        "failure_cases": int(len(failures)),
        "failure_case_frames": int(sum(
            int(row.end_frame_index) - int(row.start_frame_index) + 1
            for row in failures.itertuples(index=False))),
        "failure_cases_improved": int(sum(
            bool(row["passed"]) for row in rows
            if row["case_type"] == "failure")),
        "failure_case_frames_recovered": int(recovered_case_frames),
        "failure_identity_mean_confidence_baseline": float(
            failure_rows["confidence_baseline"].mean()),
        "failure_identity_mean_confidence_candidate": float(
            failure_rows["confidence_candidate"].mean()),
        "controls_exact": int(exact_controls),
        "controls_total": int(len(controls)),
        "all_accepted_nonzero_pixels_exact": bool(np.array_equal(
            candidate[baseline > 0], baseline[baseline > 0])),
        "producer_received_review_cases": False,
        "producer_identity_targets": 0,
        "producer_frame_targets": 0,
        "producer_coordinate_targets": 0,
    })
    metrics["automatically_eligible"] = bool(
        metrics["arithmetic_mean_confidence_delta"] > 0
        and metrics["pooled_confidence_delta"] > 0
        and metrics["failure_case_frames_recovered"] > 0
        and metrics["controls_exact"] == metrics["controls_total"]
        and metrics["all_accepted_nonzero_pixels_exact"]
        and metrics["identity_set_exact"]
        and metrics["per_frame_existing_identity_preserved"]
        and metrics["assigned_unclaimed_disjoint"]
        and metrics["zero_signal_additions"] == 0)
    case_path = out.out / "case_scores.csv"
    pd.DataFrame(rows).to_csv(case_path, index=False)
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    result["outputs"]["case_scores"] = case_path
    result["summary"] = metrics
    return result
