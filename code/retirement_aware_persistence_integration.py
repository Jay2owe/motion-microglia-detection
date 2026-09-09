"""Production audit for persistence after producer-proved alias retirement."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import retirement_aware_persistence as persistence
from continuity_confidence import _windows_from_labels, score_fixed_windows


STAGE_NAME = "102_retirement_aware_persistence"


def run(baseline: np.ndarray, candidate: np.ndarray,
        retirement_audit: pd.DataFrame,
        retirement_applications: pd.DataFrame,
        output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                   pd.DataFrame, dict]:
    """Compare fixed original windows after exact complete alias retirement."""
    baseline_ids = set(map(int, np.unique(baseline))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    retired = baseline_ids - candidate_ids
    new_ids = candidate_ids - baseline_ids
    proved = set(retirement_audit[
        retirement_audit.eligible.astype(bool)].alias_owner.astype(int)) \
        if len(retirement_audit) else set()
    fully_applied = set(retirement_applications[
        retirement_applications.applied.astype(bool)].alias_owner.astype(int)) \
        if len(retirement_applications) else set()
    proof_exact = bool(retired == proved == fully_applied and not new_ids)

    windows = _windows_from_labels(baseline)
    windows = windows[~windows.identity.astype(int).isin(retired)]
    before, before_gaps, _ = score_fixed_windows(baseline, windows)
    after, after_gaps, _ = score_fixed_windows(candidate, windows)
    before_summary = persistence._aggregate(before)
    after_summary = persistence._aggregate(after)
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
    if not metrics["all_persistence_gates_pass"]:
        raise AssertionError(
            "producer-proved alias retirement worsened surviving persistence")

    stage = Path(output_root) / STAGE_NAME / "out"
    stage.mkdir(parents=True, exist_ok=True)
    before.to_csv(
        stage / "baseline_surviving_identity_confidence.csv", index=False)
    after.to_csv(
        stage / "candidate_surviving_identity_confidence.csv", index=False)
    compared.to_csv(
        stage / "surviving_identity_confidence_comparison.csv", index=False)
    before_gaps.to_csv(stage / "baseline_surviving_gap_runs.csv", index=False)
    after_gaps.to_csv(stage / "candidate_surviving_gap_runs.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return before, after, compared, metrics
