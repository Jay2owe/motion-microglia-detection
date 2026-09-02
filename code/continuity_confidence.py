"""Approval-gated continuity candidate production and fixed-window scoring.

Candidate producers in this module are field-wide: they accept no review identity,
frame, coordinate, or region. Identity lists are consumed only by ``run_score`` after
the candidate TIFF has been frozen.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tifffile

from presence import (DEFAULTS, complete_presence_bidirectional,
                      resolve_minimum_persistent_area)
from identity_aliases import (consolidate_host_conditioned_aliases,
                              fill_alias_bridge_gaps)


def label_presence(labels: np.ndarray, identities: list[int]) -> np.ndarray:
    row_of = {int(identity): row for row, identity in enumerate(identities)}
    present = np.zeros((len(identities), len(labels)), bool)
    for t, frame in enumerate(labels):
        for value in np.unique(frame):
            identity = int(value)
            if identity in row_of:
                present[row_of[identity], t] = True
    return present


def _false_runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], ~np.asarray(values, bool), [False])).astype(
        np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts.astype(int), ends.astype(int)))


def score_fixed_windows(labels: np.ndarray, windows: pd.DataFrame
                        ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    identities = sorted(windows["identity"].astype(int).tolist())
    present = label_presence(labels, identities)
    row_of = {identity: row for row, identity in enumerate(identities)}
    cells: list[dict] = []
    gaps: list[dict] = []
    pooled_named = pooled_frames = pooled_pairs = pooled_adjacent = 0
    for window in windows.sort_values("identity").itertuples(index=False):
        identity = int(window.identity)
        first = int(window.first_frame_index)
        last = int(window.last_frame_index)
        values = present[row_of[identity], first:last + 1]
        frame_count = len(values)
        named = int(values.sum())
        adjacent = max(frame_count - 1, 0)
        pairs = int(np.count_nonzero(values[:-1] & values[1:]))
        coverage = named / frame_count
        retention = pairs / adjacent if adjacent else coverage
        runs = _false_runs(values)
        for gap_run, (start, end) in enumerate(runs, 1):
            gaps.append({
                "identity": identity,
                "gap_run": gap_run,
                "start_frame_index": first + start,
                "end_frame_index": first + end,
                "frames": end - start + 1,
            })
        cells.append({
            "identity": identity,
            "first_frame_index": first,
            "last_frame_index": last,
            "window_frames": frame_count,
            "named_frames": named,
            "gap_frames": frame_count - named,
            "gap_runs": len(runs),
            "coverage": coverage,
            "adjacent_retention": retention,
            "confidence": 100.0 * float(np.sqrt(coverage * retention)),
        })
        pooled_named += named
        pooled_frames += frame_count
        pooled_pairs += pairs
        pooled_adjacent += adjacent
    coverage = pooled_named / pooled_frames
    retention = pooled_pairs / pooled_adjacent
    table = pd.DataFrame(cells).sort_values("identity")
    gap_table = pd.DataFrame(gaps, columns=[
        "identity", "gap_run", "start_frame_index", "end_frame_index", "frames"])
    return table, gap_table, {
        "formula": "100 * sqrt(named/window * named-named-adjacent/adjacent)",
        "identities": len(identities),
        "window_frames": pooled_frames,
        "named_frames": pooled_named,
        "gap_frames": pooled_frames - pooled_named,
        "gap_runs": len(gap_table),
        "coverage": coverage,
        "adjacent_retention": retention,
        "pooled_confidence": 100.0 * float(np.sqrt(coverage * retention)),
        "perfect_identities": int(np.count_nonzero(table.confidence == 100.0)),
    }


def _aligned_evidence(params: dict, frame_count: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    offset = int(params.get("source_frame_offset", 2))
    raw_all = tifffile.imread(params["raw_path"])
    lag_all = tifffile.imread(params["lag_path"])
    raw = raw_all[offset:offset + frame_count]
    lag = lag_all[offset:offset + max(frame_count - 1, 0)]
    if len(raw) != frame_count or raw.shape[1:] != lag_all.shape[1:]:
        raise ValueError("aligned raw/lag evidence does not match labels")
    return raw, lag


def _write_candidate(out, labels: np.ndarray, unclaimed: np.ndarray,
                     audit: pd.DataFrame, changed: np.ndarray,
                     metrics: dict) -> dict:
    labels_path = out.out / "95_A3.tif"
    unclaimed_path = out.out / "95_A3_unclaimed_original_ids.tif"
    changed_path = out.out / "changed_pixels.tif"
    audit_path = out.out / "producer_audit.csv"
    metrics_path = out.out / "producer_metrics.json"
    tifffile.imwrite(labels_path, labels, compression="zlib")
    tifffile.imwrite(unclaimed_path, unclaimed, compression="zlib")
    tifffile.imwrite(changed_path, changed.astype(np.uint8), compression="zlib")
    audit.to_csv(audit_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {
        "outputs": {"labels": labels_path, "unclaimed": unclaimed_path,
                    "changed_pixels": changed_path, "audit": audit_path,
                    "metrics": metrics_path},
        "summary": metrics,
    }


def run_presence_candidate(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    if params.get("targeting_mode") != "field_wide_discovery":
        raise ValueError("candidate must declare field_wide_discovery")
    presence_params = dict(params["presence_params"])
    if presence_params.get("forced_identity_ids") \
            or presence_params.get("forced_intervals"):
        raise ValueError("field-wide producer cannot receive identity targets")
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw, lag = _aligned_evidence(params, len(labels))
    started = time.perf_counter()
    candidate, recovered, audit = complete_presence_bidirectional(
        labels, raw, lag, presence_params)
    elapsed = time.perf_counter() - started
    candidate_unclaimed = np.where(candidate == 0, unclaimed, 0).astype(
        unclaimed.dtype)
    changed = candidate != labels
    new_foreground = (labels == 0) & (candidate > 0)
    reassigned = (labels > 0) & (candidate > 0) & changed
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "forced_identity_ids": [],
        "forced_intervals": [],
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(changed.reshape(len(labels), -1).any(1))),
        "new_foreground_pixels": int(new_foreground.sum()),
        "reassigned_pixels": int(reassigned.sum()),
        "zero_signal_additions": int(np.count_nonzero(new_foreground & (raw == 0))),
        "recovered_pixels": int(recovered.sum()),
        "input_identities": int(len(np.unique(labels)) - 1),
        "output_identities": int(len(np.unique(candidate)) - 1),
        "assigned_unclaimed_disjoint": bool(not np.any(
            (candidate > 0) & (candidate_unclaimed > 0))),
        "elapsed_seconds": elapsed,
        "resolved_minimum_persistent_area_px": float(
            resolve_minimum_persistent_area(
                labels, {**DEFAULTS, **presence_params})),
        "presence_params": presence_params,
    }
    return _write_candidate(
        out, candidate, candidate_unclaimed, audit, changed, metrics)


def run_alias_candidate(upstream_dir: Path | None, params: dict, out) -> dict:
    """Discover and apply host-conditioned aliases across the complete field."""
    del upstream_dir
    if params.get("targeting_mode") != "field_wide_discovery":
        raise ValueError("candidate must declare field_wide_discovery")
    alias_params = dict(params["alias_params"])
    if alias_params.get("allowed_pairs"):
        raise ValueError("field-wide alias producer cannot receive an allow-list")
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw, lag = _aligned_evidence(params, len(labels))
    started = time.perf_counter()
    aliased, aliases, candidates = consolidate_host_conditioned_aliases(
        labels, raw, alias_params)
    bridged, bridges, inferred = fill_alias_bridge_gaps(
        aliased, raw, lag, aliases, alias_params)
    elapsed = time.perf_counter() - started
    candidate_unclaimed = np.where(bridged == 0, unclaimed, 0).astype(
        unclaimed.dtype)
    changed = bridged != labels
    aliases_path = out.out / "alias_decisions.csv"
    candidates_path = out.out / "alias_candidates.csv"
    bridges_path = out.out / "bridge_events.csv"
    aliases.to_csv(aliases_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    bridges.to_csv(bridges_path, index=False)
    audit = aliases.assign(event_kind="alias_rename") if len(aliases) else aliases
    metrics = {
        "targeting_mode": "field_wide_discovery",
        "allowed_pairs": [],
        "maximum_alias_gap_frames": int(alias_params["maximum_alias_gap_frames"]),
        "alias_decisions": int(len(aliases)),
        "alias_candidates": int(len(candidates)),
        "bridge_events": int(len(bridges)),
        "bridge_inferred_pixels": int(inferred.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(changed.reshape(len(labels), -1).any(1))),
        "new_foreground_pixels": int(np.count_nonzero((labels == 0) & (bridged > 0))),
        "zero_signal_additions": int(np.count_nonzero(
            (labels == 0) & (bridged > 0) & (raw == 0))),
        "input_identities": int(len(np.unique(labels)) - 1),
        "output_identities": int(len(np.unique(bridged)) - 1),
        "assigned_unclaimed_disjoint": bool(not np.any(
            (bridged > 0) & (candidate_unclaimed > 0))),
        "elapsed_seconds": elapsed,
    }
    result = _write_candidate(
        out, bridged, candidate_unclaimed, audit, changed, metrics)
    result["outputs"].update({
        "alias_decisions": aliases_path, "alias_candidates": candidates_path,
        "bridge_events": bridges_path})
    return result


def run_score(upstream_dir: Path | None, params: dict, out) -> dict:
    if upstream_dir is None:
        candidate_path = Path(params["candidate_labels_path"])
        candidate_unclaimed_path = Path(params["candidate_unclaimed_path"])
    else:
        candidate_path = upstream_dir / "95_A3.tif"
        candidate_unclaimed_path = upstream_dir / "95_A3_unclaimed_original_ids.tif"
    candidate = tifffile.imread(candidate_path)
    baseline = tifffile.imread(params["baseline_labels_path"])
    baseline_unclaimed = tifffile.imread(params["baseline_unclaimed_path"])
    candidate_unclaimed = tifffile.imread(candidate_unclaimed_path)
    raw, _ = _aligned_evidence(params, len(baseline))
    windows = pd.read_csv(params["windows_path"])
    groups = pd.read_csv(params["groups_path"])
    cells, gaps, metrics = score_fixed_windows(candidate, windows)
    baseline_cells, _, baseline_metrics = score_fixed_windows(baseline, windows)
    compared = cells.merge(
        baseline_cells[["identity", "confidence", "gap_frames", "gap_runs"]],
        on="identity", suffixes=("_candidate", "_baseline"))
    compared["confidence_delta"] = (
        compared.confidence_candidate - compared.confidence_baseline)
    group_name = str(params["group"])
    group_ids = set(groups.loc[groups.dominant_group == group_name,
                               "identity"].astype(int))
    group_rows = compared[compared.identity.isin(group_ids)].copy()
    changed = candidate != baseline
    new_foreground = (baseline == 0) & (candidate > 0)
    per_frame_identity_preserved = all(
        (set(map(int, np.unique(baseline[t]))) - {0}) <=
        (set(map(int, np.unique(candidate[t]))) - {0})
        for t in range(len(baseline)))
    input_ids = set(map(int, np.unique(baseline))) - {0}
    output_ids = set(map(int, np.unique(candidate))) - {0}
    metrics.update({
        "baseline_pooled_confidence": baseline_metrics["pooled_confidence"],
        "pooled_confidence_delta": (
            metrics["pooled_confidence"] - baseline_metrics["pooled_confidence"]),
        "baseline_gap_frames": baseline_metrics["gap_frames"],
        "gap_frames_delta": metrics["gap_frames"] - baseline_metrics["gap_frames"],
        "group": group_name,
        "group_identities": len(group_ids),
        "group_mean_confidence_baseline": float(
            group_rows.confidence_baseline.mean()),
        "group_mean_confidence_candidate": float(
            group_rows.confidence_candidate.mean()),
        "group_mean_confidence_delta": float(group_rows.confidence_delta.mean()),
        "group_gap_frames_baseline": int(group_rows.gap_frames_baseline.sum()),
        "group_gap_frames_candidate": int(group_rows.gap_frames_candidate.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(changed.reshape(len(baseline), -1).any(1))),
        "new_foreground_pixels": int(new_foreground.sum()),
        "zero_signal_additions": int(np.count_nonzero(new_foreground & (raw == 0))),
        "foreground_union_delta": int(np.count_nonzero(
            (candidate > 0) | (candidate_unclaimed > 0)) - np.count_nonzero(
            (baseline > 0) | (baseline_unclaimed > 0))),
        "assigned_unclaimed_disjoint": bool(not np.any(
            (candidate > 0) & (candidate_unclaimed > 0))),
        "identity_set_exact": input_ids == output_ids,
        "missing_identities": sorted(input_ids - output_ids),
        "extra_identities": sorted(output_ids - input_ids),
        "per_frame_existing_identity_preserved": per_frame_identity_preserved,
        "targeting_mode": "score_and_review_only",
    })
    cells_path = out.out / "identity_confidence.csv"
    gaps_path = out.out / "gap_runs.csv"
    compared_path = out.out / "identity_comparison.csv"
    group_path = out.qc / "group_comparison.csv"
    metrics_path = out.out / "metrics.json"
    cells.to_csv(cells_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    compared.to_csv(compared_path, index=False)
    group_rows.to_csv(group_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": {
        "identity_confidence": cells_path, "gap_runs": gaps_path,
        "identity_comparison": compared_path, "group_comparison": group_path,
        "metrics": metrics_path,
    }, "summary": metrics}


def _windows_from_labels(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for identity in sorted(int(value) for value in np.unique(labels) if value):
        present = np.asarray([np.any(frame == identity) for frame in labels])
        frames = np.flatnonzero(present)
        rows.append({"identity": identity,
                     "first_frame_index": int(frames[0]),
                     "last_frame_index": int(frames[-1])})
    return pd.DataFrame(rows)


def run_alias_score(upstream_dir: Path | None, params: dict, out) -> dict:
    """Report strict fixed-ID score and a separately audited mapped-lineage score."""
    result = run_score(upstream_dir, params, out)
    if upstream_dir is None:
        aliases_path = Path(params["alias_decisions_path"])
        candidate_path = Path(params["candidate_labels_path"])
    else:
        aliases_path = upstream_dir / "alias_decisions.csv"
        candidate_path = upstream_dir / "95_A3.tif"
    aliases = pd.read_csv(aliases_path)
    baseline = tifffile.imread(params["baseline_labels_path"])
    candidate = tifffile.imread(candidate_path)
    mapped = baseline.copy()
    for row in aliases.itertuples(index=False):
        source = int(row.source_identity)
        target = int(row.target_identity)
        start_t, end_t = int(row.source_start_t), int(row.source_end_t)
        interval = mapped[start_t:end_t + 1]
        interval[interval == source] = target
    mapped_windows = _windows_from_labels(mapped)
    mapped_cells, mapped_gaps, mapped_metrics = score_fixed_windows(
        candidate, mapped_windows)
    mapped_ids = set(map(int, np.unique(mapped))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    strict_metrics_path = out.out / "metrics.json"
    metrics = json.loads(strict_metrics_path.read_text(encoding="utf-8"))
    metrics.update({
        "mapped_lineage_formula": mapped_metrics["formula"],
        "mapped_lineage_pooled_confidence": mapped_metrics["pooled_confidence"],
        "mapped_lineage_gap_frames": mapped_metrics["gap_frames"],
        "mapped_lineage_gap_runs": mapped_metrics["gap_runs"],
        "mapped_lineage_identity_set_exact": mapped_ids == candidate_ids,
        "mapped_lineage_identities": len(mapped_ids),
        "alias_decisions_audited": int(len(aliases)),
    })
    strict_metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                                   encoding="utf-8")
    mapped_windows_path = out.out / "mapped_lineage_windows.csv"
    mapped_cells_path = out.out / "mapped_lineage_confidence.csv"
    mapped_gaps_path = out.out / "mapped_lineage_gaps.csv"
    mapped_windows.to_csv(mapped_windows_path, index=False)
    mapped_cells.to_csv(mapped_cells_path, index=False)
    mapped_gaps.to_csv(mapped_gaps_path, index=False)
    result["outputs"].update({
        "mapped_lineage_windows": mapped_windows_path,
        "mapped_lineage_confidence": mapped_cells_path,
        "mapped_lineage_gaps": mapped_gaps_path,
    })
    result["summary"] = metrics
    return result
