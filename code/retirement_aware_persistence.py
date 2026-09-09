"""Fixed-window persistence after a mathematically proved alias retirement.

The ordinary guard correctly treats an arbitrary missing identity as a
regression.  This variant excludes an identity from both denominators only when
the complete-field alias producer proves and fully applies its retirement.
All surviving identities retain the original baseline windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from continuity_confidence import _windows_from_labels, score_fixed_windows


def _aggregate(table: pd.DataFrame) -> dict:
    window = int(table.window_frames.sum()) if len(table) else 0
    named = int(table.named_frames.sum()) if len(table) else 0
    adjacent = int(sum(max(int(value) - 1, 0)
                       for value in table.window_frames)) if len(table) else 0
    pairs = int(sum(round(float(row.adjacent_retention)
                          * max(int(row.window_frames) - 1, 0))
                    for row in table.itertuples(index=False))) if len(table) else 0
    coverage = named / window if window else 1.0
    retention = pairs / adjacent if adjacent else coverage
    return {
        "identities": int(len(table)), "window_frames": window,
        "named_frames": named, "gap_frames": window - named,
        "gap_runs": int(table.gap_runs.sum()) if len(table) else 0,
        "coverage": coverage, "adjacent_retention": retention,
        "pooled_confidence": 100.0 * float(np.sqrt(coverage * retention)),
        "perfect_identities": int(np.count_nonzero(
            table.confidence.to_numpy(float) == 100.0)) if len(table) else 0,
    }


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        raise ValueError("candidate output is required")
    baseline = tifffile.imread(params["baseline_labels_path"])
    candidate = tifffile.imread(upstream_dir / "95_A3.tif")
    audit = pd.read_csv(params["retirement_audit_path"])
    applications = pd.read_csv(params["retirement_applications_path"])
    baseline_ids = set(map(int, np.unique(baseline))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    retired = baseline_ids - candidate_ids
    new_ids = candidate_ids - baseline_ids
    proved = set(audit[audit.eligible.astype(bool)].alias_owner.astype(int))
    fully_applied = set(applications[
        applications.applied.astype(bool)].alias_owner.astype(int))
    proof_exact = bool(retired == proved == fully_applied and not new_ids)

    windows = _windows_from_labels(baseline)
    windows = windows[~windows.identity.astype(int).isin(retired)]
    before, before_gaps, _ = score_fixed_windows(baseline, windows)
    after, after_gaps, _ = score_fixed_windows(candidate, windows)
    before_summary, after_summary = _aggregate(before), _aggregate(after)
    compared = before.merge(
        after, on="identity", suffixes=("_baseline", "_candidate"))
    compared["confidence_delta"] = (
        compared.confidence_candidate - compared.confidence_baseline)
    metrics = {
        "targeting_mode": "producer_proof_only",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "retired_aliases": sorted(map(int, retired)),
        "producer_proved_aliases": sorted(map(int, proved)),
        "fully_applied_aliases": sorted(map(int, fully_applied)),
        "retirement_proof_exact": proof_exact,
        "surviving_baseline_identities": int(len(before)),
        "candidate_identities": int(len(candidate_ids)),
        "new_identity_count": int(len(new_ids)),
        "arithmetic_mean_confidence_baseline": float(before.confidence.mean()),
        "arithmetic_mean_confidence_candidate": float(after.confidence.mean()),
        "arithmetic_mean_confidence_delta": float(
            after.confidence.mean() - before.confidence.mean()),
        "pooled_confidence_baseline": before_summary["pooled_confidence"],
        "pooled_confidence_candidate": after_summary["pooled_confidence"],
        "pooled_confidence_delta": (
            after_summary["pooled_confidence"]
            - before_summary["pooled_confidence"]),
        "gap_frames_baseline": before_summary["gap_frames"],
        "gap_frames_candidate": after_summary["gap_frames"],
        "gap_frames_delta": (
            after_summary["gap_frames"] - before_summary["gap_frames"]),
        "gap_runs_baseline": before_summary["gap_runs"],
        "gap_runs_candidate": after_summary["gap_runs"],
        "perfect_identities_baseline": before_summary["perfect_identities"],
        "perfect_identities_candidate": after_summary["perfect_identities"],
    }
    metrics["all_persistence_gates_pass"] = bool(
        proof_exact
        and metrics["arithmetic_mean_confidence_delta"] >= 0
        and metrics["pooled_confidence_delta"] >= 0
        and metrics["gap_frames_delta"] <= 0
        and metrics["gap_runs_candidate"] <= metrics["gap_runs_baseline"]
        and metrics["perfect_identities_candidate"]
            >= metrics["perfect_identities_baseline"])
    before.to_csv(out.out / "baseline_surviving_identity_confidence.csv",
                  index=False)
    after.to_csv(out.out / "candidate_surviving_identity_confidence.csv",
                 index=False)
    compared.to_csv(out.out / "surviving_identity_confidence_comparison.csv",
                    index=False)
    before_gaps.to_csv(out.out / "baseline_surviving_gap_runs.csv", index=False)
    after_gaps.to_csv(out.out / "candidate_surviving_gap_runs.csv", index=False)
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "metrics": metrics_path,
        "comparison": out.out / "surviving_identity_confidence_comparison.csv",
        "baseline_gaps": out.out / "baseline_surviving_gap_runs.csv",
        "candidate_gaps": out.out / "candidate_surviving_gap_runs.csv",
    }, "summary": metrics}

