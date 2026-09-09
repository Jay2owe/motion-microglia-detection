from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import skimage
import tifffile
from PIL import Image, ImageDraw

from accepted_history import (
    AcceptedHistoryInputs, load_history_inputs_from_run,
    _save_final_labels,
    _well_b2_issue008_short_owned_companion_projection_calibration,
    _well_b2_issue009_ownerless_cohort_latent_gap_completion,
    _well_b2_issue011_recurrent_same_owner_branch_swarm_calibration,
    _well_b2_issue012_aggregate_shared_core_calibration,
    _well_b2_issue013_weak_terminal_reference_relay_calibration,
    _well_b2_issue014_single_owner_multireference_relay_calibration,
    _well_b2_issue015_event_local_subresolution_artifact_calibration,
    _well_b2_issue016_short_ownerless_subcellular_reference_calibration,
    _well_b2_issue018_recording_start_two_seat_inheritance,
    _well_b2_issue026_detached_projection_owner_relay_calibration,
    _well_b2_issue028_right_censored_novel_body_calibration,
    _well_b3_issue010_detached_fading_projection_calibration,
    _well_b3_issue012_delayed_owner_projection_flash,
    _well_b3_issue012_explained_projection_event_calibration,
    _well_b3_issue013_anchored_projection_owner_relay,
    _well_b3_issue013_ephemeral_speckle_alias_retirement,
    _well_b3_issue013_anchored_projection_raw_gap_completion,
    _well_b3_issue014_terminal_projection_chain_completion,
    _well_b3_issue015_conserved_multi_anchor_projection_calibration,
    _well_b3_issue016_body_scale_ownerless_allocation,
    _well_b3_issue017_terminal_boundary_vanished_seat,
    _well_b3_issue017_terminal_boundary_event_calibration,
    _well_b3_issue018_fragmented_subcellular_episode_calibration,
    _well_b3_issue019_bracketed_ownerless_seat_completion,
    _well_b3_issue020_fragmented_encounter_lineage_calibration,
    _well_b3_issue021_temporal_core_path_lineage,
    _well_b3_issue022_ownerless_reference_handoff_calibration,
    _well_b3_issue023_asymmetric_fusion_area_flip_recovery,
    _well_b3_issue024_recurrent_dominant_body_relay,
    _well_b3_issue025_established_seat_cycle_recovery,
    _well_b3_issue026_dormant_seat_successor_recovery,
    _well_b3_issue027_staggered_fusion_exchange_recovery,
    _well_b4_issue001_large_persistent_reference_relay,
    _well_b4_issue002_dormant_owner_reciprocal_partition,
    _well_b4_issue003_conservative_episode_ownership,
    _well_b4_issue004_original_body_continuity,
    _well_b3_issue004_reconnected_companion_merge_partition,
    _well_b3_issue005_component_continuity_duplicate_takeover,
    _well_b3_issue006_right_censored_seat_partition,
    _well_b3_issue007_right_censored_reciprocal_exchange,
    _well_b3_issue008_persistent_single_owner_flash,
    run_accepted_history)
import raw_physical_hypotheses
import retirement_aware_persistence_integration
from common import (Config, ROOT, Run, OUTLINE_COLOURS, display_raw, label_edges,
                    load_stack, outline_overlay, read_json, save_rgb_stack,
                    save_stack, sha256, validate_labels, write_json)
from events import resolve_events
from global_soma_ledger import (
    GlobalSomaLedgerParams, add_field_discovered_bookend_observations,
    reuse_retired_identity_names, separate_substantial_soma_objects,
    substantial_identity_conflicts, track_physical_somas)
from identity_aliases import (carry_recurrent_alias_pairs_to_end,
                              consolidate_recurrent_host_aliases,
                              fill_alias_bridge_gaps)
from lineage_confidence import (identity_confidence_table,
                                restore_deferred_identity_layer)
from midpoint_reconciliation import reconcile_midpoint_bookends
from model_review import (TransferParams, TwoCoreParams,
                          audit_two_core_objects, group_two_core_runs)
from motion_handoffs import (build_motion_pair_cache,
                             build_motion_reservation_requests,
                             carry_established_identities_along_motion,
                             carry_identities_through_motion_handoffs,
                             carry_reserved_identities_through_motion_hosts,
                             motion_pair_evidence_for_fixed_objects,
                             seed_first_handoff_identities)
from observations import anchor_scores, build_observations
from oscillatory_reconciliation import (optimise_host_merge_graph,
                                         optimise_lineage_graph)
from persistence import green_evidence_for_fixed_objects, retrack_fixed_objects
from persistent_merges import carry_identities_through_merges
from stationary_reconciliation import (
    PIXEL_COLUMNS, _assert_identity_blind_params, align_stack,
    discover_stationary_takeovers, reconciliation_params)
from partial_body_transfers import (
    GlobalPairLedgerParams, IdentityLedgerParams, PredecessorCoreParams,
    carry_persistent_two_core_slots, carry_recurrent_host_pair_ledger,
    correct_delayed_bookend_runs, correct_identity_ledger_transfers,
    correct_predecessor_core_merges, correct_structurally_proven_unresolved_runs)
from tracking import (frame_observations, prepare_tracking_observations,
                      track_from_anchor)


def _pinned_paths(cfg: Config, stem: str) -> dict[str, Path]:
    registered = cfg.resolve("registered_input_dir")
    motion = cfg.resolve("motion_input_dir")
    pins = cfg.values["pinned_files"][stem]
    result: dict[str, Path] = {}
    for name, row in pins.items():
        if "relative_to_registered_input_dir" in row:
            result[name] = registered / row["relative_to_registered_input_dir"]
        else:
            result[name] = motion / row["relative_to_motion_input_dir"]
    return result


def _verify_pins(cfg: Config, stem: str) -> dict[str, dict]:
    paths = _pinned_paths(cfg, stem)
    pins = cfg.values["pinned_files"][stem]
    verified: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        expected = pins[name]["sha256"]
        if actual != expected:
            raise ValueError(f"{name} fingerprint differs: {actual} != {expected}")
        verified[name] = {"path": str(path.resolve()), "sha256": actual,
                          "bytes": path.stat().st_size}
    raw = load_stack(paths["registered_raw"])
    lag = load_stack(paths["lag_float"])
    neutral = load_stack(paths["neutral_tracks"])
    trails = load_stack(paths["trail_labels"])
    ages = load_stack(paths["trail_ages"])
    if raw.ndim != 3 or lag.shape != neutral.shape or lag.shape != trails.shape \
            or lag.shape != ages.shape or len(raw) != len(lag) + 1 \
            or raw.shape[1:] != lag.shape[1:]:
        raise ValueError("pinned registered and motion arrays are not aligned")
    return verified


def _software() -> dict:
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "skimage": skimage.__version__,
        "tifffile": tifffile.__version__,
    }


def stage_m0(cfg: Config, stem: str, run_name: str) -> dict[str, Path]:
    run = Run("m0_inputs", run_name, {"stem": stem, "mode": "hash_pinned_external"})
    try:
        verified = _verify_pins(cfg, stem)
        source = run.dir / "out" / f"{stem}_sources.json"
        write_json(source, {
            "stem": stem, "input_space": "registered",
            "policy": "read-only external inputs pinned by SHA-256",
            "files": verified, "software": _software(),
            "config_sha256": sha256(cfg.path),
        })
        run.record("sources", source)
        run.finish({"files_verified": len(verified), "source_mutated": False})
        return _pinned_paths(cfg, stem)
    except BaseException as error:
        run.fail(error); raise


def stage_m1(cfg: Config, stem: str, run_name: str,
             sources: dict[str, Path]) -> None:
    params = {
        "stem": stem,
        "ordinary_lag": 1,
        "cache_policy": "verified external reference; no 354 MiB duplication",
        "observation_params": cfg.values["observation"],
    }
    run = Run("m1_motion", run_name, params, upstream=f"m0_inputs/{run_name}")
    try:
        with tifffile.TiffFile(sources["lag_float"]) as lag_tiff:
            lag_transitions = int(lag_tiff.series[0].shape[0])
        manifest = run.dir / "out" / f"{stem}_motion_sources.json"
        write_json(manifest, {
            "stem": stem,
            "lag_definition": "log2((I(t+1)+1)/(I(t)+1))",
            "channels": {"blue": "lost", "green": "stable nonzero",
                         "red": "gained", "yellow": "age-coded history"},
            "warning": "yellow history and its connector lines are not current cell pixels",
            "files": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                      for name, path in sources.items()},
        })
        run.record("motion_sources", manifest)
        run.finish({"lag_transitions": lag_transitions, "cache_rebuilt": False,
                    "motion_arrays_mutated": False})
    except BaseException as error:
        run.fail(error); raise


def stage_m2(cfg: Config, stem: str, run_name: str, sources: dict[str, Path]
             ) -> tuple[np.ndarray, pd.DataFrame, int, np.ndarray, np.ndarray]:
    params = {"stem": stem, "observation": cfg.values["observation"],
              "anchor": cfg.values["anchor"]}
    run = Run("m2_anchor", run_name, params, upstream=f"m1_motion/{run_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        neutral = load_stack(sources["neutral_tracks"])
        labels, table, gained, lost = build_observations(
            raw, lag, neutral, cfg.values["observation"])
        scores = anchor_scores(
            labels, table, central_fraction=cfg.values["anchor"]["central_fraction"])
        eligible = scores[scores.observations >= int(
            cfg.values["anchor"]["minimum_observations"])]
        if eligible.empty:
            raise ValueError("no central frame passes the minimum anchor observation gate")
        anchor_t = int(eligible.iloc[0].t)

        label_path = run.dir / "out" / f"{stem}_observations.tif"
        table_path = run.dir / "out" / "observations.csv"
        score_path = run.dir / "out" / "anchor_scores.csv"
        anchor_path = run.dir / "out" / "anchor.json"
        preview_path = run.dir / "qc" / f"{stem}_anchor.tif"
        save_stack(label_path, labels, cfg.values["frame_interval_min"])
        table.to_csv(table_path, index=False)
        scores.to_csv(score_path, index=False)
        write_json(anchor_path, {
            "t": anchor_t, "imagej_frame": anchor_t + 1,
            "score": float(eligible.iloc[0].score),
            "observations": int(eligible.iloc[0].observations),
            "selection": "highest reviewed score inside central window",
        })
        save_rgb_stack(preview_path,
                       outline_overlay(raw[[anchor_t]], labels[[anchor_t]], thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in (("observations", label_path), ("observation_table", table_path),
                            ("anchor_scores", score_path), ("anchor", anchor_path),
                            ("anchor_preview", preview_path)):
            run.record(label, path)
        run.finish({
            "anchor_imagej_frame": anchor_t + 1,
            "anchor_observations": int(np.max(labels[anchor_t])),
            "median_observations_per_frame": float(table.groupby("t").size().median()),
            "observation_range": [int(table.groupby("t").size().min()),
                                  int(table.groupby("t").size().max())],
        })
        return labels, table, anchor_t, gained, lost
    except BaseException as error:
        run.fail(error); raise


def stage_m3(cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
             observations: np.ndarray, observation_table: pd.DataFrame, anchor_t: int
             ) -> tuple[np.ndarray, pd.DataFrame]:
    run = Run("m3_track", run_name,
              {"stem": stem, "anchor_t": anchor_t,
               "tracking": cfg.values["tracking"]},
              upstream=f"m2_anchor/{run_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        labels, links = track_from_anchor(
            observations, raw, lag, anchor_t, cfg.values["tracking"], observation_table)
        validate_labels(labels)
        label_path = run.dir / "out" / f"{stem}_initial_identities.tif"
        link_path = run.dir / "out" / "links.csv"
        outline_path = run.dir / "qc" / f"{stem}_initial_outlines.tif"
        save_stack(label_path, labels, cfg.values["frame_interval_min"])
        links.to_csv(link_path, index=False)
        save_rgb_stack(outline_path, outline_overlay(raw, labels, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in (("initial_identities", label_path), ("links", link_path),
                            ("initial_outlines", outline_path)):
            run.record(label, path)
        counts = [len(np.unique(frame)) - 1 for frame in labels]
        run.finish({
            "global_identities": int(len(np.unique(labels)) - 1),
            "median_frame_identities": float(np.median(counts)),
            "frame_identity_range": [int(min(counts)), int(max(counts))],
            "links": int((links.kind == "link").sum()),
            "new_observations": int((links.kind == "new_observation").sum()),
            "misses": int((links.kind == "miss").sum()),
        })
        return labels, links
    except BaseException as error:
        run.fail(error); raise


def stage_m4(cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
             initial: np.ndarray
             ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    run = Run("m4_events", run_name,
              {"stem": stem, "events": cfg.values["events"]},
              upstream=f"m3_track/{run_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        labels, events, inferred, unresolved = resolve_events(
            initial, raw, cfg.values["events"])
        label_path = run.dir / "out" / f"{stem}_event_identities.tif"
        event_path = run.dir / "out" / "events.csv"
        inferred_path = run.dir / "mid" / f"{stem}_inferred_pixels.tif"
        unresolved_path = run.dir / "mid" / f"{stem}_unresolved_pixels.tif"
        save_stack(label_path, labels, cfg.values["frame_interval_min"])
        events.to_csv(event_path, index=False)
        save_stack(inferred_path, inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        save_stack(unresolved_path, unresolved.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        for label, path in (("event_identities", label_path), ("events", event_path),
                            ("inferred_pixels", inferred_path),
                            ("unresolved_pixels", unresolved_path)):
            run.record(label, path)
        resolved = int((events.status == "resolved_inferred").sum()) if len(events) else 0
        unresolved_count = int((events.status == "unresolved").sum()) if len(events) else 0
        run.finish({"resolved_events": resolved, "unresolved_events": unresolved_count,
                    "inferred_pixels": int(inferred.sum()),
                    "unresolved_review_pixels": int(unresolved.sum())})
        return labels, events, inferred, unresolved
    except BaseException as error:
        run.fail(error); raise


def _identity_tables(labels: np.ndarray, raw: np.ndarray, inferred: np.ndarray
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for t in range(len(labels)):
        observations = frame_observations(labels[t], raw[t])
        for identity, row in observations.items():
            rows.append({
                "t": t, "imagej_frame": t + 1, "identity": identity,
                "y": float(row["position"][0]), "x": float(row["position"][1]),
                "area_px": int(row["area"]), "mean_intensity": float(row["mean"]),
                "inferred_partition": bool(np.any(inferred[t] & row["mask"])),
            })
    frame_table = pd.DataFrame(rows)
    tracks: list[dict] = []
    for identity, group in frame_table.groupby("identity"):
        tracks.append({
            "identity": int(identity),
            "first_imagej_frame": int(group.imagej_frame.min()),
            "last_imagej_frame": int(group.imagej_frame.max()),
            "observed_frames": int(group.imagej_frame.nunique()),
            "inferred_frames": int(group.inferred_partition.sum()),
            "median_area_px": float(group.area_px.median()),
            "median_mean_intensity": float(group.mean_intensity.median()),
        })
    return frame_table, pd.DataFrame(tracks)


def stage_m5(cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
             labels: np.ndarray, events: pd.DataFrame, inferred: np.ndarray
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = Run("m5_reconcile", run_name,
              {"stem": stem, "method": "anchor outward passes plus delayed bookends"},
              upstream=f"m4_events/{run_name}")
    try:
        validate_labels(labels)
        raw = load_stack(sources["registered_raw"])
        if labels.shape != raw.shape:
            raise ValueError("final identities and registered raw movie differ in shape")
        frame_table, tracks = _identity_tables(labels, raw, inferred)
        # Tables must describe exactly the identities present in the label movie.
        derived = frame_table.groupby("t").identity.nunique().reindex(
            range(len(labels)), fill_value=0).to_numpy()
        actual = np.array([len(np.unique(frame)) - 1 for frame in labels])
        if not np.array_equal(derived, actual):
            raise AssertionError("identity table and label movie counts differ")
        label_path = run.dir / "out" / f"{stem}.tif"
        frame_path = run.dir / "out" / "frame_identities.csv"
        track_path = run.dir / "out" / "tracks.csv"
        decision_path = run.dir / "out" / "decisions.csv"
        outline_path = run.dir / "qc" / f"{stem}_outline.tif"
        save_stack(label_path, labels, cfg.values["frame_interval_min"])
        frame_table.to_csv(frame_path, index=False)
        tracks.to_csv(track_path, index=False)
        events.to_csv(decision_path, index=False)
        save_rgb_stack(outline_path, outline_overlay(raw, labels, thick=2,
                                                     inferred=inferred),
                       cfg.values["frame_interval_min"])
        for label, path in (("labels", label_path), ("frame_identities", frame_path),
                            ("tracks", track_path), ("decisions", decision_path),
                            ("outlines", outline_path)):
            run.record(label, path)
        run.finish({
            "global_identities": int(len(tracks)),
            "median_track_frames": float(tracks.observed_frames.median()),
            "half_movie_tracks_percent": float(
                100.0 * np.mean(tracks.observed_frames >= len(labels) / 2.0)),
            "median_frame_identities": float(np.median(actual)),
            "structural_invariants": "passed",
        })
        return frame_table, tracks
    except BaseException as error:
        run.fail(error); raise


def stage_m7_accepted(cfg: Config, stem: str, run_name: str,
                      sources: dict[str, Path], baseline: np.ndarray,
                      observation_labels: np.ndarray,
                      observation_table: pd.DataFrame, anchor_t: int
                      ) -> tuple[np.ndarray, pd.DataFrame]:
    """Accepted persistence prior: prefer recovering old identities over births."""
    params = cfg.values["accepted_postprocessing"]["persistence"]
    run = Run("m7_persistence", run_name, {
        "stem": stem,
        "accepted_version": cfg.values["accepted_postprocessing"]["version"],
        "mode": "identity-only retracking of fixed final frame objects",
        "tracking": params,
    }, upstream=f"m5_reconcile/{run_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        green_table = green_evidence_for_fixed_objects(
            baseline, observation_labels, observation_table)
        candidate, links, mappings = retrack_fixed_objects(
            baseline, raw, lag, anchor_t, params, green_table)
        validate_labels(candidate)
        inferred = np.zeros(candidate.shape, bool)
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "links": run.dir / "out" / "links.csv",
            "mapping": run.dir / "out" / "frame_object_identity_map.csv",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "green_evidence": run.dir / "mid" / "fixed_object_green_evidence.csv",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate, cfg.values["frame_interval_min"])
        links.to_csv(outputs["links"], index=False)
        mappings.to_csv(outputs["mapping"], index=False)
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        green_table.to_csv(outputs["green_evidence"], index=False)
        save_rgb_stack(outputs["outlines"], outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)
        before_counts = np.array([len(np.unique(frame)) - 1 for frame in baseline])
        after_counts = np.array([len(np.unique(frame)) - 1 for frame in candidate])
        summary = {
            "global_identities": int(len(tracks)),
            "median_track_frames": float(tracks.observed_frames.median()),
            "foreground_changed_px": int(np.count_nonzero(
                (baseline > 0) != (candidate > 0))),
            "frames_with_cell_count_change": int(np.count_nonzero(
                before_counts != after_counts)),
            "co_present_identity_collisions": int(mappings.duplicated(
                ["t", "persistent_identity"]).sum()),
        }
        if any(summary[key] != 0 for key in (
                "foreground_changed_px", "frames_with_cell_count_change",
                "co_present_identity_collisions")):
            raise AssertionError(f"accepted persistence gates failed: {summary}")
        run.finish(summary)
        return candidate, tracks
    except BaseException as error:
        run.fail(error); raise


def stage_m8_accepted(cfg: Config, stem: str, run_name: str,
                      sources: dict[str, Path], labels: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted delayed-bookend separation through temporary merged objects."""
    params = cfg.values["accepted_postprocessing"]["persistent_merges"]
    run = Run("m8_merge", run_name, {
        "stem": stem,
        "accepted_version": cfg.values["accepted_postprocessing"]["version"],
        "mode": "carry persistent identities through proven merged foreground",
        "persistent_merges": params,
    }, upstream=f"m7_persistence/{run_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        candidate, events, inferred = carry_identities_through_merges(
            labels, raw, lag, params)
        frame_partitions = events.attrs.get("frame_rows", pd.DataFrame())
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "events": run.dir / "out" / "persistent_merge_events.csv",
            "frame_partitions": run.dir / "out" / "persistent_merge_frame_partitions.csv",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "inferred": run.dir / "mid" / f"{stem}_inferred_merge_pixels.tif",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate, cfg.values["frame_interval_min"])
        events.to_csv(outputs["events"], index=False)
        frame_partitions.to_csv(outputs["frame_partitions"], index=False)
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        save_rgb_stack(outputs["outlines"], outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)
        before_counts = np.array([len(np.unique(frame)) - 1 for frame in labels])
        after_counts = np.array([len(np.unique(frame)) - 1 for frame in candidate])
        summary = {
            "global_identities": int(len(tracks)),
            "resolved_persistent_merges": int(len(events)),
            "partitioned_frames": int(events.partitioned_frames.sum()) if len(events) else 0,
            "foreground_changed_px": int(np.count_nonzero(
                (labels > 0) != (candidate > 0))),
            "frames_with_identity_count_decrease": int(np.count_nonzero(
                after_counts < before_counts)),
        }
        if summary["foreground_changed_px"] or summary["frames_with_identity_count_decrease"]:
            raise AssertionError(f"accepted delayed-merge gates failed: {summary}")
        run.finish(summary)
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error); raise


def stage_m9_accepted(cfg: Config, stem: str, run_name: str,
                      sources: dict[str, Path], labels: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted field-wide resolver for recurrent rounded-cell contacts."""
    params = cfg.values["accepted_postprocessing"]["recurrent_aliases"]
    if params.get("allowed_pairs"):
        raise ValueError("accepted recurrent alias resolver cannot use a cell allow-list")
    run = Run("m9_alias", run_name, {
        "stem": stem,
        "accepted_version": cfg.values["accepted_postprocessing"]["version"],
        "approved_scope": cfg.values["accepted_postprocessing"]["approved_scope"],
        "mode": "field-wide recurrent same-host alias discovery",
        "recurrent_aliases": params,
    }, upstream=f"m8_merge/{run_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        aliased, aliases, candidates, audit = consolidate_recurrent_host_aliases(
            labels, raw, params)
        bridged, bridges, bridge_inferred = fill_alias_bridge_gaps(
            aliased, raw, lag, aliases, params)
        candidate, open_frames, open_inferred = carry_recurrent_alias_pairs_to_end(
            bridged, raw, lag, aliases, bridges, params)
        inferred = bridge_inferred | open_inferred
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "aliases": run.dir / "out" / "global_alias_decisions.csv",
            "candidates": run.dir / "out" / "global_alias_candidates.csv",
            "audit": run.dir / "out" / "global_alias_audit.csv",
            "bridges": run.dir / "out" / "global_alias_bridge_events.csv",
            "open_frames": run.dir / "out" / "global_open_merge_frames.csv",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "inferred": run.dir / "mid" / f"{stem}_inferred_global_alias_pixels.tif",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate, cfg.values["frame_interval_min"])
        aliases.to_csv(outputs["aliases"], index=False)
        candidates.to_csv(outputs["candidates"], index=False)
        audit.to_csv(outputs["audit"], index=False)
        bridges.to_csv(outputs["bridges"], index=False)
        open_frames.to_csv(outputs["open_frames"], index=False)
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        save_rgb_stack(outputs["outlines"], outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)
        before_counts = np.array([len(np.unique(frame)) - 1 for frame in labels])
        after_counts = np.array([len(np.unique(frame)) - 1 for frame in candidate])
        accepted_pairs = sorted({
            (int(row.target_identity), int(row.host_identity))
            for row in aliases.itertuples()
        })
        summary = {
            "global_identities": int(len(tracks)),
            "median_track_frames": float(tracks.observed_frames.median()),
            "accepted_recurrent_pairs": [list(pair) for pair in accepted_pairs],
            "alias_decisions": int(len(aliases)),
            "bridge_partitioned_frames": int(bridges.partitioned_frames.sum())
            if len(bridges) else 0,
            "open_merge_partitioned_frames": int(len(open_frames)),
            "foreground_changed_px": int(np.count_nonzero(
                (labels > 0) != (candidate > 0))),
            "frames_with_identity_count_decrease": int(np.count_nonzero(
                after_counts < before_counts)),
        }
        if summary["foreground_changed_px"] or summary["frames_with_identity_count_decrease"]:
            raise AssertionError(f"accepted recurrent-alias gates failed: {summary}")
        run.finish(summary)
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error); raise


def stage_m12_accepted(
        cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
        baseline: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, anchor_t: int,
        upstream_run: str | None = None,
        history_inputs: AcceptedHistoryInputs | None = None,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted long-memory and reserved fast-motion identity resolver."""
    accepted = cfg.values["accepted_postprocessing"]["identity_reservation"]
    tracking_params = accepted["tracking"]
    motion_params = accepted["motion_handoff"]
    upstream_name = upstream_run or run_name
    run = Run("m12_identity_reservation", run_name, {
        "stem": stem,
        "accepted_version": accepted["version"],
        "parameter_source": "accepted_postprocessing.identity_reservation",
        "approved_scope": accepted["approved_scope"],
        "tracking": tracking_params,
        "motion_handoff": motion_params,
    }, upstream=f"m9_alias/{upstream_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        pair_cache = build_motion_pair_cache(lag, motion_params)

        # The handoff geometry is also the guide: the historical A005 guide and A006
        # geometry used identical field-wide parameters on the same accepted labels.
        geometry, handoffs, geometric_inferred, disappearance_audit = \
            carry_identities_through_motion_handoffs(
                baseline, raw, lag, motion_params, pair_cache=pair_cache)
        guide, guide_handoffs, guide_audit = (
            geometry, handoffs, disappearance_audit)
        green = green_evidence_for_fixed_objects(
            geometry, observations, observation_table)
        motion_evidence = motion_pair_evidence_for_fixed_objects(
            geometry, lag, motion_params, green, pair_cache=pair_cache)
        retracked, links, mappings = retrack_fixed_objects(
            geometry, raw, lag, anchor_t, tracking_params, motion_evidence)
        long_memory, long_trajectory_events, long_trajectory_audit = \
            carry_established_identities_along_motion(
                retracked, raw, lag, motion_params, pair_cache=pair_cache)

        seeded, seed_events = seed_first_handoff_identities(
            long_memory, guide, handoffs)
        trajectory, trajectory_events, trajectory_audit = \
            carry_established_identities_along_motion(
                seeded, raw, lag, motion_params, pair_cache=pair_cache)

        reservations, reservation_audit = build_motion_reservation_requests(
            seeded, seed_events)
        for reservation in reservations.itertuples(index=False):
            target = int(reservation.identity)
            alias = int(reservation.displaced_identity)
            for frame in range(int(reservation.to_t), len(trajectory)):
                trajectory[frame][trajectory[frame] == target] = alias
        candidate, host_carry, host_inferred = \
            carry_reserved_identities_through_motion_hosts(
                trajectory, raw, lag, reservations, motion_params,
                pair_cache=pair_cache)

        if not np.array_equal(candidate > 0, baseline > 0):
            raise AssertionError(
                "accepted identity reservation changed foreground support")
        if mappings.duplicated(["t", "persistent_identity"]).any():
            raise AssertionError(
                "accepted identity reservation assigned one identity twice")
        inferred = (geometric_inferred | host_inferred
                    | (long_memory != retracked)
                    | (candidate != long_memory))
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "pre_reservation_labels": run.dir / "mid" /
                f"{stem}_pre_reservation_long_memory.tif",
            "long_memory_links": run.dir / "out" / "long_memory_links.csv",
            "mapping": run.dir / "out" / "frame_object_identity_map.csv",
            "motion_evidence": run.dir / "mid" / "fixed_object_motion_evidence.csv",
            "guide_handoffs": run.dir / "out" / "guide_motion_handoffs.csv",
            "guide_audit": run.dir / "out" / "guide_disappearance_audit.csv",
            "handoffs": run.dir / "out" / "motion_handoff_events.csv",
            "disappearance_audit": run.dir / "out" / "disappearance_audit.csv",
            "long_trajectory_events": run.dir / "out" / "long_memory_trajectory_events.csv",
            "long_trajectory_audit": run.dir / "out" / "long_memory_trajectory_audit.csv",
            "seed_events": run.dir / "out" / "first_handoff_identity_seeds.csv",
            "reservation_audit": run.dir / "out" / "motion_reservation_audit.csv",
            "trajectory_events": run.dir / "out" / "motion_trajectory_events.csv",
            "trajectory_audit": run.dir / "out" / "motion_trajectory_audit.csv",
            "host_carry": run.dir / "out" / "reserved_identity_host_carry.csv",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "inferred": run.dir / "mid" / f"{stem}_inferred_identity_reservation.tif",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate, cfg.values["frame_interval_min"])
        save_stack(outputs["pre_reservation_labels"], long_memory,
                   cfg.values["frame_interval_min"])
        links.to_csv(outputs["long_memory_links"], index=False)
        mappings.to_csv(outputs["mapping"], index=False)
        motion_evidence.to_csv(outputs["motion_evidence"], index=False)
        guide_handoffs.to_csv(outputs["guide_handoffs"], index=False)
        guide_audit.to_csv(outputs["guide_audit"], index=False)
        handoffs.to_csv(outputs["handoffs"], index=False)
        disappearance_audit.to_csv(outputs["disappearance_audit"], index=False)
        long_trajectory_events.to_csv(outputs["long_trajectory_events"], index=False)
        long_trajectory_audit.to_csv(outputs["long_trajectory_audit"], index=False)
        seed_events.to_csv(outputs["seed_events"], index=False)
        reservation_audit.to_csv(outputs["reservation_audit"], index=False)
        trajectory_events.to_csv(outputs["trajectory_events"], index=False)
        trajectory_audit.to_csv(outputs["trajectory_audit"], index=False)
        host_carry.to_csv(outputs["host_carry"], index=False)
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        save_rgb_stack(outputs["outlines"], outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)

        summary = {
            "targeting_mode": "field_wide_motion_reservation",
            "maximum_detection_gap_frames": int(tracking_params["max_gap_frames"]),
            "global_identities_before": int(
                len(set(map(int, np.unique(baseline))) - {0})),
            "global_identities_after": int(len(tracks)),
            "global_identity_reduction": int(
                len(set(map(int, np.unique(baseline))) - {0}) - len(tracks)),
            "first_handoff_identity_seeds": int(len(seed_events)),
            "seeded_persistent_identities": int(
                seed_events.persistent_identity.nunique()) if len(seed_events) else 0,
            "reservation_requests_accepted": int(len(reservations)),
            "reservation_requests_rejected": int(
                (reservation_audit.decision == "rejected").sum())
                if len(reservation_audit) else 0,
            "motion_trajectory_events": int(len(trajectory_events)),
            "forced_host_frames": int(len(host_carry)),
            "forced_host_first_imagej_frame": (
                int(host_carry.imagej_frame.min()) if len(host_carry) else None),
            "forced_host_last_imagej_frame": (
                int(host_carry.imagej_frame.max()) if len(host_carry) else None),
            "foreground_pixels_added": 0,
            "foreground_pixels_removed": 0,
            "foreground_support_unchanged": True,
            "co_present_identity_errors": 0,
        }
        if history_inputs is not None:
            history_inputs.reservation_labels = candidate
            history_inputs.pre_reservation_labels = long_memory
            required_host_columns = {"identity", "host_identity"}
            history_inputs.host_audit = (
                host_carry
                if required_host_columns.issubset(host_carry.columns)
                else pd.DataFrame(columns=["identity", "host_identity"]))
            history_inputs.motion_reserved = (
                _accepted_motion_reservation_identities(reservation_audit))
            history_inputs.motion_reservation_events = reservation_audit
        run.finish(summary)
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error)
        raise


def _accepted_motion_reservation_identities(events: pd.DataFrame) -> set[int]:
    """Return identities backed by accepted field-discovered reservation events."""
    if events.empty:
        return set()
    required = {"identity", "decision"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"motion reservation audit missing columns: {sorted(missing)}")
    accepted = events[events.decision.astype(str).eq("accepted")]
    return set(accepted.identity.astype(int))


def stage_m19_accepted(
        cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
        baseline: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, anchor_t: int,
        upstream_run: str | None = None,
        motion_reservation_events: pd.DataFrame | None = None,
        history_inputs: AcceptedHistoryInputs | None = None,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted destination-led confidence-layer and lineage reunion stage."""
    accepted = cfg.values["accepted_postprocessing"]["layered_lineage"]
    upstream_name = upstream_run or run_name
    if (motion_reservation_events is None and history_inputs is not None
            and history_inputs.motion_reservation_events is not None):
        motion_reservation_events = history_inputs.motion_reservation_events
    if motion_reservation_events is None:
        reservation_path = (ROOT / "m12_identity_reservation" / upstream_name /
                            "out/motion_reservation_audit.csv")
        motion_reservation_events = (pd.read_csv(reservation_path)
                                     if reservation_path.is_file()
                                     else pd.DataFrame())
    motion_reserved = _accepted_motion_reservation_identities(
        motion_reservation_events)
    run = Run("m19_layered_lineage", run_name, {
        "stem": stem,
        "accepted_version": accepted["version"],
        "mode": ("rebuild physical somas, track red destinations to protected "
                 "blue sources, then reunite non-overlapping lineages"),
        "confidence": accepted["confidence"],
        "tracking": accepted["tracking"],
        "reconciliation": accepted["reconciliation"],
        "upstream_motion_reservation_events": int(len(motion_reservation_events)),
        "motion_reserved_lineages": int(len(motion_reserved)),
    }, upstream=f"m12_identity_reservation/{upstream_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        structural = accepted["structural_evidence"]
        objects = audit_two_core_objects(
            baseline, raw, TwoCoreParams(**structural["two_core"]))
        structural_runs = group_two_core_runs(
            objects, int(structural["maximum_run_gap_frames"]),
            float(structural["maximum_run_shift_px_per_frame"]))
        # Rebuild the approved foreground partition source from the accepted
        # upstream labels. Identity names from this step are never trusted.
        ledger_params = IdentityLedgerParams(transfer=TransferParams(
            **accepted["identity_ledger"]))
        recurrent_params = GlobalPairLedgerParams(
            **accepted["recurrent_host_pair"])
        predecessor_params = PredecessorCoreParams(
            **accepted["predecessor_core"])
        shape_source, direct_events, direct_frames, direct_inferred = \
            correct_identity_ledger_transfers(baseline, raw, ledger_params)
        used = set(direct_events.two_core_run_id) if len(direct_events) else set()
        shape_source, bookend_events, bookend_frames, bookend_inferred = \
            correct_delayed_bookend_runs(
                shape_source, raw, skip_run_ids=used,
                detection_labels=baseline)
        if len(bookend_events):
            used |= set(bookend_events.run_id)
        reservations = pd.concat(
            [direct_frames, bookend_frames], ignore_index=True, sort=False)
        shape_source, persistent_events, persistent_frames, persistent_inferred = \
            carry_persistent_two_core_slots(
                shape_source, raw, reservations, detection_labels=baseline,
                skip_run_ids=used)
        if len(persistent_events):
            used |= set(persistent_events.two_core_run_id)
        shape_source, structural_events, structural_frames, structural_inferred = \
            correct_structurally_proven_unresolved_runs(
                shape_source, raw, detection_labels=baseline, objects=objects,
                runs=structural_runs, skip_run_ids=used)
        proved_frames = pd.concat(
            [direct_frames, bookend_frames, persistent_frames, structural_frames],
            ignore_index=True, sort=False)
        shape_source, recurrent_events, recurrent_frames, recurrent_inferred = \
            carry_recurrent_host_pair_ledger(
                shape_source, raw, proved_frames, recurrent_params,
                detection_labels=baseline, objects=objects, runs=structural_runs)
        shape_source, predecessor_events, predecessor_frames, predecessor_inferred = \
            correct_predecessor_core_merges(
                shape_source, raw, predecessor_params,
                detection_labels=baseline, objects=objects, runs=structural_runs)
        # Rebuild Attempt 005's physical-soma comparison from the accepted base.
        soma_params = GlobalSomaLedgerParams(**accepted["soma_ledger"])
        physical, soma_audit = separate_substantial_soma_objects(
            shape_source, soma_params.minimum_substantial_core_px)
        comparison_tracking = dict(accepted["comparison_tracking"])
        comparison_tracking["motion_reservation_identities"] = sorted(
            motion_reserved)
        preliminary, preliminary_links, preliminary_evidence, preliminary_aliases = \
            track_physical_somas(
                physical, baseline, raw, lag, anchor_t,
                comparison_tracking, observations,
                observation_table, soma_params)
        physical, bookend_observation_audit = \
            add_field_discovered_bookend_observations(
                physical, preliminary, raw, lag, objects, structural_runs,
                cfg.values["accepted_postprocessing"]["persistent_merges"],
                float(structural["minimum_bookend_observation_score"]))
        prepared_tracking = prepare_tracking_observations(
            physical, raw)
        comparison, comparison_links, comparison_evidence, comparison_aliases = \
            track_physical_somas(
                physical, baseline, raw, lag, anchor_t,
                comparison_tracking, observations,
                observation_table, soma_params, prepared_tracking)
        comparison, comparison_reused = reuse_retired_identity_names(
            comparison, baseline,
            int(accepted["minimum_retired_name_overlap_px"]))
        # Large established identities are tracked first. A red destination may
        # search for a blue source, but cannot steal a whole-movie-supported name.
        comparison_confidence = identity_confidence_table(
            comparison, raw, accepted["confidence"])
        high_ids = set(comparison_confidence.loc[
            comparison_confidence.confident, "identity"].astype(int))
        tracking_params = dict(accepted["tracking"])
        tracking_params["high_confidence_identity_ids"] = sorted(high_ids)
        stable, stable_links, stable_evidence, stable_aliases = \
            track_physical_somas(
                physical, comparison, raw, lag, anchor_t, tracking_params,
                observations, observation_table, soma_params,
                prepared_tracking)
        stable, deferred = restore_deferred_identity_layer(
            stable, physical, comparison, high_ids)
        stable, stable_reused = reuse_retired_identity_names(
            stable, comparison,
            int(accepted["minimum_retired_name_overlap_px"]))
        # Join only non-overlapping before/after fragments. Established names win
        # over weak fragments on either side of the midpoint.
        stable_confidence = identity_confidence_table(
            stable, raw, accepted["confidence"])
        protected_ids = set(stable_confidence.loc[
            stable_confidence.confident, "identity"].astype(int))
        reconciliation = dict(accepted["reconciliation"])
        reconciliation["protected_identity_ids"] = sorted(protected_ids)
        reconciliation["unprotected_canonical_policy"] = "longer_fragment"
        candidate, reunions, reunion_candidates = reconcile_midpoint_bookends(
            stable, raw, anchor_t, reconciliation)

        conflicts = substantial_identity_conflicts(
            candidate, soma_params.minimum_substantial_core_px)
        if len(conflicts):
            raise AssertionError("layered-lineage output duplicated a soma identity")
        if not np.array_equal(candidate > 0, baseline > 0):
            raise AssertionError("layered-lineage output changed foreground support")

        inferred = candidate != baseline
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        shape_events = pd.concat([
            direct_events, bookend_events, persistent_events, structural_events,
            recurrent_events, predecessor_events], ignore_index=True, sort=False)
        shape_frames = pd.concat([
            direct_frames, bookend_frames, persistent_frames, structural_frames,
            recurrent_frames, predecessor_frames], ignore_index=True, sort=False)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "shape_source": run.dir / "mid" / f"{stem}_shape_source.tif",
            "physical_somas": run.dir / "mid" / f"{stem}_physical_somas.tif",
            "comparison": run.dir / "mid" / f"{stem}_attempt005_comparison.tif",
            "stable_intermediate": run.dir / "mid" / f"{stem}_red_led_stable.tif",
            "inferred": run.dir / "mid" / f"{stem}_inferred_layered_lineage.tif",
            "shape_events": run.dir / "out" / "shape_ledger_events.csv",
            "shape_frames": run.dir / "out" / "shape_ledger_frames.csv",
            "structural_objects": run.dir / "out" / "two_core_objects.csv",
            "structural_runs": run.dir / "out" / "two_core_runs.csv",
            "soma_audit": run.dir / "out" / "physical_soma_observations.csv",
            "bookend_observation_audit": run.dir / "out" /
                "bookend_observation_audit.csv",
            "comparison_links": run.dir / "out" / "comparison_links.csv",
            "comparison_evidence": run.dir / "mid" / "comparison_evidence.csv",
            "comparison_aliases": run.dir / "out" / "comparison_aliases.csv",
            "comparison_reused": run.dir / "out" / "comparison_reused_names.csv",
            "comparison_confidence": run.dir / "out" / "comparison_confidence.csv",
            "stable_links": run.dir / "out" / "red_led_links.csv",
            "stable_evidence": run.dir / "mid" / "red_led_evidence.csv",
            "stable_aliases": run.dir / "out" / "red_led_aliases.csv",
            "stable_deferred": run.dir / "out" / "deferred_identity_layer.csv",
            "stable_reused": run.dir / "out" / "stable_reused_names.csv",
            "stable_confidence": run.dir / "out" / "stable_confidence.csv",
            "reunions": run.dir / "out" / "approved_lineage_reunions.csv",
            "reunion_candidates": run.dir / "mid" / "lineage_reunion_candidates.csv",
            "conflicts": run.dir / "out" / "co_present_soma_identity_conflicts.csv",
            "motion_reservations": run.dir / "out" / "upstream_motion_reservations.csv",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate,
                   cfg.values["frame_interval_min"])
        save_stack(outputs["shape_source"], shape_source,
                   cfg.values["frame_interval_min"])
        save_stack(outputs["physical_somas"], physical,
                   cfg.values["frame_interval_min"])
        save_stack(outputs["comparison"], comparison,
                   cfg.values["frame_interval_min"])
        save_stack(outputs["stable_intermediate"], stable,
                   cfg.values["frame_interval_min"])
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        shape_events.to_csv(outputs["shape_events"], index=False)
        shape_frames.to_csv(outputs["shape_frames"], index=False)
        objects.to_csv(outputs["structural_objects"], index=False)
        structural_runs.to_csv(outputs["structural_runs"], index=False)
        soma_audit.to_csv(outputs["soma_audit"], index=False)
        bookend_observation_audit.to_csv(
            outputs["bookend_observation_audit"], index=False)
        comparison_links.to_csv(outputs["comparison_links"], index=False)
        comparison_evidence.to_csv(outputs["comparison_evidence"], index=False)
        comparison_aliases.to_csv(outputs["comparison_aliases"], index=False)
        comparison_reused.to_csv(outputs["comparison_reused"], index=False)
        comparison_confidence.to_csv(outputs["comparison_confidence"], index=False)
        stable_links.to_csv(outputs["stable_links"], index=False)
        stable_evidence.to_csv(outputs["stable_evidence"], index=False)
        stable_aliases.to_csv(outputs["stable_aliases"], index=False)
        deferred.to_csv(outputs["stable_deferred"], index=False)
        stable_reused.to_csv(outputs["stable_reused"], index=False)
        stable_confidence.to_csv(outputs["stable_confidence"], index=False)
        reunions.to_csv(outputs["reunions"], index=False)
        reunion_candidates.to_csv(outputs["reunion_candidates"], index=False)
        conflicts.to_csv(outputs["conflicts"], index=False)
        motion_reservation_events.to_csv(outputs["motion_reservations"], index=False)
        save_rgb_stack(outputs["outlines"], outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)

        output_sha = sha256(outputs["labels"])
        summary = {
            "approved_lineage_reunions": int(len(reunions)),
            "layer_1_established_identities": int(len(protected_ids)),
            "global_identities_before": int(
                len(set(map(int, np.unique(baseline))) - {0})),
            "global_identities_after": int(len(tracks)),
            "global_identity_reduction": int(
                len(set(map(int, np.unique(baseline))) - {0}) - len(tracks)),
            "co_present_soma_identity_conflicts": int(len(conflicts)),
            "motion_reserved_lineages": int(len(motion_reserved)),
            "foreground_support_unchanged": True,
            "labels_sha256": output_sha,
        }
        if history_inputs is not None:
            history_inputs.physical_somas = physical
        run.finish(summary)
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error)
        raise


def stage_m20_accepted(
        cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
        baseline: np.ndarray, upstream_run: str | None = None,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted whole-movie lineage and bracketed-host graph stage."""
    accepted = cfg.values["accepted_postprocessing"]["oscillatory_lineage"]
    upstream_name = upstream_run or run_name
    run = Run("m20_oscillatory_lineage", run_name, {
        "stem": stem,
        "accepted_version": accepted["version"],
        "mode": ("whole-movie lineage graph followed by non-conflicting "
                 "one-frame host partitions"),
        "lineage_graph": accepted["lineage_graph"],
        "host_merge_graph": accepted["host_merge_graph"],
        "registration": "unchanged",
    }, upstream=f"m19_layered_lineage/{upstream_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        lineage, lineage_decisions, lineage_candidates = optimise_lineage_graph(
            baseline, lag, accepted["lineage_graph"],
            int(accepted["minimum_substantial_core_px"]))
        candidate, host_decisions, host_candidates = optimise_host_merge_graph(
            lineage, raw, lag, accepted["host_merge_graph"],
            cfg.values["accepted_postprocessing"]["persistent_merges"])

        conflicts = substantial_identity_conflicts(
            candidate, int(accepted["minimum_substantial_core_px"]))
        if len(conflicts):
            raise AssertionError("oscillatory-lineage output duplicated a soma identity")
        if not np.array_equal(candidate > 0, baseline > 0):
            raise AssertionError("oscillatory-lineage output changed foreground support")

        inferred = candidate != baseline
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "frame_identities": run.dir / "out/frame_identities.csv",
            "tracks": run.dir / "out/tracks.csv",
            "inferred": run.dir / "mid" / f"{stem}_inferred_oscillatory_lineage.tif",
            "lineage_decisions": run.dir / "out/lineage_graph_decisions.csv",
            "lineage_candidates": run.dir / "mid/lineage_graph_candidates.csv",
            "host_decisions": run.dir / "out/host_partition_graph_decisions.csv",
            "host_candidates": run.dir / "mid/host_partition_graph_candidates.csv",
            "conflicts": run.dir / "out/co_present_soma_identity_conflicts.csv",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate,
                   cfg.values["frame_interval_min"])
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        lineage_decisions.to_csv(outputs["lineage_decisions"], index=False)
        lineage_candidates.to_csv(outputs["lineage_candidates"], index=False)
        host_decisions.to_csv(outputs["host_decisions"], index=False)
        host_candidates.to_csv(outputs["host_candidates"], index=False)
        conflicts.to_csv(outputs["conflicts"], index=False)
        save_rgb_stack(outputs["outlines"],
                       outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)
        output_sha = sha256(outputs["labels"])
        summary = {
            "global_identities_before": int(len(
                set(map(int, np.unique(baseline))) - {0})),
            "global_identities_after": len(tracks),
            "lineage_graph_decisions": int(len(lineage_decisions)),
            "host_partition_graph_decisions": int(len(host_decisions)),
            "co_present_soma_identity_conflicts": int(len(conflicts)),
            "foreground_support_unchanged": True,
            "registration_added": False,
            "labels_sha256": output_sha,
        }
        run.finish(summary)
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error)
        raise


def stage_m21_stationary_reconciliation(
        cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
        baseline: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Default-on field-wide repair of stationary identity takeovers."""
    accepted = cfg.values["accepted_postprocessing"].get(
        "field_wide_stationary_reconciliation", {})
    enabled = bool(accepted.get("enabled", True))
    params = reconciliation_params(accepted.get("parameters", {}))
    params["enabled"] = enabled
    _assert_identity_blind_params(params)
    run = Run("m21_stationary_reconciliation", run_name, {
        "stem": stem,
        "accepted_version": accepted.get("version", "field-wide-v1"),
        "targeting_mode": "field_wide_discovery",
        "enabled": enabled,
        "enabled_by_default": True,
        "off_switch_supported": True,
        "parameters": params,
        "supplied_identity_count": 0,
        "supplied_problem_frame_count": 0,
    }, upstream=f"m20_oscillatory_lineage/{run_name}")
    try:
        raw_full = load_stack(sources["registered_raw"])
        raw = align_stack(
            raw_full, len(baseline),
            int(accepted.get("source_frame_offset", 0)))
        if raw.shape != baseline.shape:
            raise ValueError("stationary reconciliation inputs do not align")
        if enabled:
            candidate, inferred, stationary_runs, events, evidence = (
                discover_stationary_takeovers(baseline, raw, params))
        else:
            candidate = baseline.copy()
            inferred = np.zeros(baseline.shape, bool)
            stationary_runs = pd.DataFrame(columns=[
                "run_id", "identity", "start_frame", "end_frame",
                "observed_frames", "stationary_reference"])
            events = pd.DataFrame(columns=[
                "event_id", "resident_identity", "observed_identity",
                "decision"])
            evidence = pd.DataFrame(columns=PIXEL_COLUMNS)
        if not np.array_equal(candidate > 0, baseline > 0):
            raise AssertionError("stationary reconciliation changed foreground")
        if not np.array_equal(inferred, candidate != baseline):
            raise AssertionError("stationary reconciliation audit mask differs")
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "frame_identities": run.dir / "out/frame_identities.csv",
            "tracks": run.dir / "out/tracks.csv",
            "inferred": run.dir / "mid" /
                f"{stem}_inferred_stationary_reconciliation.tif",
            "stationary_runs": run.dir / "out/stationary_component_runs.csv",
            "events": run.dir / "out/stationary_takeover_events.csv",
            "pixel_evidence": run.dir /
                "out/stationary_takeover_pixel_evidence.csv",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate, cfg.values["frame_interval_min"])
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        stationary_runs.to_csv(outputs["stationary_runs"], index=False)
        events.to_csv(outputs["events"], index=False)
        evidence.to_csv(outputs["pixel_evidence"], index=False)
        save_rgb_stack(outputs["outlines"],
                       outline_overlay(raw, candidate, thick=2,
                                       inferred=inferred),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)
        accepted_events = int(events["decision"].astype(str).str.startswith(
            "accepted_field_wide_takeover").sum()) if len(events) else 0
        run.finish({
            "targeting_mode": "field_wide_discovery",
            "enabled": enabled,
            "enabled_by_default": True,
            "off_switch_supported": True,
            "supplied_identity_count": 0,
            "supplied_problem_frame_count": 0,
            "detected_events": int(len(events)),
            "accepted_events": accepted_events,
            "renamed_pixels": int(np.count_nonzero(inferred)),
            "renamed_frames": int(np.count_nonzero(
                np.any(inferred, axis=(1, 2)))),
            "foreground_support_unchanged": True,
        })
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error)
        raise


def _accepted_provenance(
        stem: str, source_run: str, labels: np.ndarray, inferred: np.ndarray,
        observations: np.ndarray, upstream_inferred: np.ndarray | None,
        unresolved: np.ndarray | None,
        oscillatory_inferred: np.ndarray | None,
        ) -> tuple[np.ndarray | None, dict]:
    """Build the sidecar for the accepted labels, or explain why there is none.

    Saving costs nothing when the caller hands over the arrays it already
    holds: no stack is recomputed and none is read back. `observations` is the
    segmentation this whole run was built from, still in memory, and it is what
    separates an outline the tracker supplied from a name the tracker worked
    out. An incomplete record would mark reconstructed pixels as observed, so a
    gap means no file.
    """
    in_memory = (upstream_inferred is not None and unresolved is not None
                 and oscillatory_inferred is not None)
    if in_memory:
        reconstructed = _align_review_stack(upstream_inferred, len(labels)).copy()
        reconstructed |= _align_review_stack(oscillatory_inferred, len(labels))
        undecided = _align_review_stack(unresolved, len(labels))
        source = "in-memory stage records"
    else:
        try:
            reconstructed, undecided = collect_provenance(
                stem, source_run, len(labels), include_history=False)
        except (FileNotFoundError, ValueError) as error:
            return None, {
                "provenance": f"no sidecar: incomplete reconstruction record "
                              f"under {source_run} ({error})",
            }
        source = f"stage records read back from {source_run}"
    detected = _align_review_stack(observations, len(labels)) > 0
    provenance = pack_provenance(
        reconstructed | inferred, undecided, labels, detected)
    return provenance, {"provenance_source": source,
                        **provenance_metrics(provenance, labels)}


def stage_m22_accepted_history(
        cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
        m20_labels: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, anchor_t: int,
        source_run_name: str | None = None,
        checkpoint_run_name: str | None = None,
        history_inputs: AcceptedHistoryInputs | None = None,
        m20_result_future: Future | None = None,
        upstream_inferred: np.ndarray | None = None,
        unresolved: np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Run the complete accepted Issue 005-039 history on current inputs.

    `upstream_inferred` and `unresolved` are the reconstruction records the
    caller already holds for every stage before M20; supplying them lets this
    stage save a provenance sidecar without reading a single stack back from
    disk. When they are absent the stage tries to read them, and writes no
    sidecar at all rather than an incomplete one.
    """
    source_run = source_run_name or run_name
    run = Run("m22_accepted_history", run_name, {
        "stem": stem,
        "source_frame_offset": 2,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "coordinate_target_count": 0,
        "source_run": source_run,
        "checkpoint_run": checkpoint_run_name,
    }, upstream=f"m20_oscillatory_lineage/{source_run}")
    try:
        work = run.dir / "mid" / "accepted_history"
        if checkpoint_run_name:
            checkpoint_work = (ROOT / "m22_accepted_history" /
                               checkpoint_run_name / "mid" /
                               "accepted_history")
            if not checkpoint_work.is_dir():
                raise FileNotFoundError(
                    f"accepted-history checkpoints not found: {checkpoint_work}")
            shutil.copytree(checkpoint_work, work, dirs_exist_ok=True)
        labels, unclaimed, summary = run_accepted_history(
            cfg, stem, run_name, sources, m20_labels, observations,
            observation_table, anchor_t, work,
            source_run_name=source_run, history_inputs=history_inputs)
        oscillatory_inferred = None
        if m20_result_future is not None:
            m20_labels, oscillatory_inferred, _ = m20_result_future.result()
        if len(m20_labels) != len(labels) + 2:
            raise ValueError(
                "complete accepted history requires two source frames before "
                "the accepted 99-frame review stack")
        raw = load_stack(sources["registered_raw"])[2:2 + len(labels)]
        inferred = labels != m20_labels[2:2 + len(labels)]
        frame_table, tracks = _identity_tables(labels, raw, inferred)
        provenance, provenance_summary = _accepted_provenance(
            stem, source_run, labels, inferred, observations,
            upstream_inferred, unresolved, oscillatory_inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "unclaimed": run.dir / "out" /
                f"{stem}_unclaimed_original_ids.tif",
            "inferred": run.dir / "mid" /
                f"{stem}_inferred_accepted_history.tif",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "history_summary": run.dir / "out" / "accepted_history_summary.json",
        }
        # Preserve the accepted stage's exact TIFF byte encoding.
        shutil.copyfile(work / f"{stem}.tif", outputs["labels"])
        shutil.copyfile(
            work / f"{stem}_unclaimed_original_ids.tif", outputs["unclaimed"])
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        if provenance is not None:
            outputs["provenance"] = run.dir / "out" / f"{stem}_provenance.tif"
            save_stack(outputs["provenance"], provenance,
                       cfg.values["frame_interval_min"])
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        write_json(outputs["history_summary"], summary)
        for label, path in outputs.items():
            run.record(label, path)
        run.finish({
            **summary,
            "targeting_mode": "field_wide_discovery",
            "final_frames": int(len(labels)),
            "foreground_accounting_disjoint": bool(not np.any(
                (labels > 0) & (unclaimed > 0))),
            "labels_sha256": sha256(outputs["labels"]),
            "unclaimed_sha256": sha256(outputs["unclaimed"]),
            **provenance_summary,
        })
        return labels, unclaimed, inferred, tracks
    except BaseException as error:
        run.fail(error)
        raise


def _motion_rgb(raw: np.ndarray, composite: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = (display_raw(raw) * 0.35).astype(np.uint8)
    colours = np.array([[0, 0, 0], [25, 90, 255], [0, 255, 80],
                        [255, 45, 20], [255, 255, 0]], np.float32)
    for t in range(len(raw)):
        transition = min(t, len(composite) - 1)
        for channel in range(1, 5):
            strength = composite[transition, channel].astype(np.float32) / 65535.0
            layer = strength[..., None] * colours[channel]
            out[t] = np.maximum(out[t], np.uint8(np.clip(layer, 0, 255)))
    edge = label_edges(labels, thick=2)
    out[edge] = OUTLINE_COLOURS[(labels[edge].astype(np.int64) - 1)
                                % len(OUTLINE_COLOURS)]
    return out


def _align_review_stack(array: np.ndarray, frames: int) -> np.ndarray:
    """Align full-source masks to the accepted tail after frame removal."""
    if len(array) < frames:
        raise ValueError(
            f"review stack has {len(array)} frames but labels have {frames}")
    return array[len(array) - frames:]


PROVENANCE_INFERRED = 0b001
PROVENANCE_UNRESOLVED = 0b010
PROVENANCE_ADDED = 0b100

# Every stage that can rewrite pixel ownership records what it reconstructed in
# a boolean stack under its own mid/ folder. Entries are
# (stage, file name, required when an accepted base exists, needs alignment).
# The m22 stack is already in accepted-frame space and is the one that is not
# realigned; the rest are full-source and get trimmed to the accepted tail.
_INFERRED_STACKS = (
    ("m8_merge", "{stem}_inferred_merge_pixels.tif", True, True),
    ("m9_alias", "{stem}_inferred_global_alias_pixels.tif", True, True),
    ("m12_identity_reservation",
     "{stem}_inferred_identity_reservation.tif", False, True),
    ("m19_layered_lineage",
     "{stem}_inferred_layered_lineage.tif", False, True),
    ("m20_oscillatory_lineage",
     "{stem}_inferred_oscillatory_lineage.tif", False, True),
    ("m21_stationary_reconciliation",
     "{stem}_inferred_stationary_reconciliation.tif", False, True),
    ("m22_accepted_history",
     "{stem}_inferred_accepted_history.tif", False, False),
)


def collect_provenance(stem: str, upstream_run: str, frames: int,
                       accepted: bool = True, include_history: bool = True,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Union every stage's reconstruction record, aligned to `frames`.

    Returns two boolean stacks: pixels whose ownership a stage reconstructed
    rather than observed, and foreground the tracker never resolved. This is
    the accumulation `run_review_only` used to perform inline, factored out so
    the review overlay and the saved provenance sidecar cannot drift apart.

    It reads the stacks back from disk, so it belongs to resume and backfill
    paths only. A full run already holds these arrays in memory and hands them
    to `stage_m22_accepted_history` instead.
    """
    def mid(stage: str) -> Path:
        return ROOT / stage / upstream_run / "mid"

    inferred = _align_review_stack(
        load_stack(mid("m4_events") / f"{stem}_inferred_pixels.tif").astype(bool),
        frames)
    if accepted:
        for stage, template, required, align in _INFERRED_STACKS:
            if stage == "m22_accepted_history" and not include_history:
                continue
            path = mid(stage) / template.format(stem=stem)
            if not path.is_file():
                if required:
                    raise FileNotFoundError(
                        f"accepted reconstruction record is missing: {path}")
                continue
            stack = load_stack(path).astype(bool)
            inferred = inferred | (_align_review_stack(stack, frames)
                                   if align else stack)
    unresolved = _align_review_stack(
        load_stack(mid("m4_events") / f"{stem}_unresolved_pixels.tif").astype(bool),
        frames)
    return inferred, unresolved


def pack_provenance(inferred: np.ndarray, unresolved: np.ndarray,
                    labels: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Pack the reconstruction records into one uint8 stack of bit flags.

    `detected` is the segmentation's own foreground, before any identity was
    assigned: where the microscope showed something. It separates the two very
    different things bit 0 otherwise merges. An outline pixel with nothing
    detected under it is one the tracker supplied itself (bit 2); an outline
    pixel with something detected under it was shown by the microscope and only
    its owner was worked out (bit 0 alone). On the accepted 95_A3 that is 572
    pixels against 403,765, which is the difference between "the tracker drew
    this cell" and "the tracker named this cell".
    """
    wrong = {name: array.shape for name, array in
             (("inferred", inferred), ("unresolved", unresolved),
              ("detected", detected)) if array.shape != labels.shape}
    if wrong:
        raise AssertionError(
            f"provenance {wrong} does not match labels {labels.shape}; a "
            "sidecar not aligned to the final labels is worse than none")
    provenance = np.zeros(labels.shape, np.uint8)
    provenance[inferred] |= PROVENANCE_INFERRED
    provenance[unresolved] |= PROVENANCE_UNRESOLVED
    provenance[(labels > 0) & ~detected] |= PROVENANCE_ADDED
    return provenance


def provenance_metrics(provenance: np.ndarray, labels: np.ndarray) -> dict:
    """Pixel counts a reader needs in order to weigh the accepted stack."""
    inferred = (provenance & PROVENANCE_INFERRED).astype(bool)
    unresolved = (provenance & PROVENANCE_UNRESOLVED).astype(bool)
    added = (provenance & PROVENANCE_ADDED).astype(bool)
    labelled = labels > 0
    labelled_px = int(np.count_nonzero(labelled))
    inferred_labelled = int(np.count_nonzero(inferred & labelled))
    added_px = int(np.count_nonzero(added))
    return {
        "provenance_labelled_px": labelled_px,
        "provenance_inferred_px": int(np.count_nonzero(inferred)),
        "provenance_inferred_labelled_px": inferred_labelled,
        "provenance_added_px": added_px,
        "provenance_renamed_labelled_px": int(
            np.count_nonzero(inferred & labelled & ~added)),
        # Bit 2 without bit 0 would be an outline pixel the detection never saw
        # that no stage owns up to having supplied. It should not happen; if it
        # ever does, the number says so instead of the sidecar hiding it.
        "provenance_added_unflagged_px": int(
            np.count_nonzero(added & ~inferred)),
        "provenance_unresolved_px": int(np.count_nonzero(unresolved)),
        "provenance_unresolved_unlabelled_px": int(
            np.count_nonzero(unresolved & ~labelled)),
        "provenance_inferred_fraction_of_labelled": (
            round(inferred_labelled / labelled_px, 6) if labelled_px else 0.0),
        "provenance_added_fraction_of_labelled": (
            round(added_px / labelled_px, 6) if labelled_px else 0.0),
    }


def _annotate_review(stack: np.ndarray, anchor_t: int) -> np.ndarray:
    header_height = 24
    result = np.zeros((len(stack), stack.shape[1] + header_height,
                       stack.shape[2], 3), np.uint8)
    result[:, header_height:] = stack
    panel_width = stack.shape[2] // 3
    names = ("raw + identities", "motion + identities", "events: cyan inferred / magenta unresolved")
    for t in range(len(stack)):
        image = Image.fromarray(result[t])
        draw = ImageDraw.Draw(image)
        for panel, name in enumerate(names):
            x = panel * panel_width + 5
            suffix = " [ANCHOR]" if t == anchor_t else ""
            draw.text((x, 5), f"ImageJ frame {t + 1}: {name}{suffix}", fill=(255, 255, 255))
        result[t] = np.asarray(image)
    return result


def stage_m6(cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
             labels: np.ndarray, events: pd.DataFrame, inferred: np.ndarray,
             unresolved: np.ndarray, anchor_t: int, tracks: pd.DataFrame,
             upstream_run: str | None = None,
             upstream_stage: str = "m5_reconcile",
             accepted_base: bool = False) -> None:
    run = Run("m6_review", run_name,
              {"stem": stem, "scope": "all frames and complete field",
               "accepted_base": accepted_base},
              upstream=f"{upstream_stage}/{upstream_run or run_name}")
    try:
        raw_full = load_stack(sources["registered_raw"])
        source_offset = len(raw_full) - len(labels)
        if source_offset < 0:
            raise ValueError("accepted labels are longer than registered raw")
        raw = raw_full[source_offset:source_offset + len(labels)]
        composite = load_stack(sources["motion_composite"])
        if source_offset:
            composite = composite[min(source_offset, len(composite)):]
        inferred = _align_review_stack(inferred, len(labels))
        unresolved = _align_review_stack(unresolved, len(labels))
        review_anchor_t = max(0, anchor_t - source_offset)
        identity_view = outline_overlay(raw, labels, thick=2)
        motion_view = _motion_rgb(raw, composite, labels)
        event_view = outline_overlay(raw, labels, thick=2, inferred=inferred,
                                     unresolved=unresolved)
        review = np.concatenate([identity_view, motion_view, event_view], axis=2)
        review = _annotate_review(review, review_anchor_t)
        identity_path = run.dir / "out" / f"{stem}_full_field_outlines.tif"
        review_path = run.dir / "out" / f"{stem}_full_field_review.tif"
        event_path = run.dir / "out" / "event_index.csv"
        score_path = run.dir / "out" / "movie_scorecard.csv"
        feedback_path = run.dir / "out" / "reviewer_feedback.md"
        readme_path = run.dir / "README.md"
        save_rgb_stack(identity_path, identity_view, cfg.values["frame_interval_min"])
        save_rgb_stack(review_path, review, cfg.values["frame_interval_min"])
        events.to_csv(event_path, index=False)
        counts = np.array([len(np.unique(frame)) - 1 for frame in labels])
        pd.DataFrame([{
            "stem": stem, "frames": len(labels), "field_height_px": labels.shape[1],
            "field_width_px": labels.shape[2],
            "anchor_imagej_frame": review_anchor_t + 1,
            "global_identities": len(tracks),
            "median_identities_per_frame": float(np.median(counts)),
            "minimum_identities_per_frame": int(counts.min()),
            "maximum_identities_per_frame": int(counts.max()),
            "resolved_inferred_events": int((events.status == "resolved_inferred").sum()),
            "unresolved_events": int((events.status == "unresolved").sum()),
            "automated_gates": "pass",
            "jamie_review": "approved" if accepted_base else "pending",
        }]).to_csv(score_path, index=False)
        if accepted_base:
            feedback_path.write_text(
                "# Motion accepted-base reviewer feedback\n\n"
                "- Reviewer: Jamie\n"
                "- Approved Issue 001: two rounded cells moving close together\n"
                "- Approved Issue 002: fast-moving identities persist through red destinations and merged hosts\n"
                "- Approved Issue 003: destination-led confidence layers reunite eight non-overlapping lineage fragments\n"
                "- Reviewed candidates: I001-A004, I002-A007 and I003-A019\n"
                "- Overall decision: approved and integrated\n",
                encoding="utf-8")
        else:
            feedback_path.write_text(
                "# Motion base reviewer feedback\n\n"
                "- Reviewer: Jamie\n"
                f"- Complete {len(labels)}-frame identity outlines: pending\n"
                "- Motion-supported fast cells: pending\n"
                "- Cyan inferred merge partitions: pending\n"
                "- Magenta unresolved events: pending\n"
                "- Overall decision: pending\n"
                "- Notes:\n", encoding="utf-8")
        readme_path.write_text(
            f"# {stem} motion-base review\n\n"
            f"Open `{review_path.name}` first in Fiji. Each ImageJ frame contains three "
            "complete-field panels: raw identities, quantitative motion evidence with "
            "the same identities, and event uncertainty. Cyan marks a boundary inferred "
            "from delayed bookends; magenta marks an event the base refused to classify.\n\n"
            f"The automatically selected anchor is ImageJ frame {anchor_t + 1}. "
            + ("This run reproduces the accepted Issue 001 to Issue 003 corrections.\n"
               if accepted_base else
               f"Review all {len(labels)} frames before approving the base. Record the decision in `reviewer_feedback.md`.\n"),
            encoding="utf-8")
        for label, path in (("outlines", identity_path), ("full_review", review_path),
                            ("event_index", event_path), ("scorecard", score_path),
                            ("reviewer_feedback", feedback_path), ("readme", readme_path)):
            run.record(label, path)
        run.finish({
            "review_frames": len(labels), "complete_field": True,
            "automated_gates": "passed",
            "human_review": "approved" if accepted_base else "pending",
            "accepted_base": accepted_base,
        })
    except BaseException as error:
        run.fail(error); raise


def run_base(run_name: str, stem: str, config_path: Path | None = None) -> None:
    cfg = Config.load(config_path)
    if stem not in cfg.stems:
        raise ValueError(f"stem {stem!r} is not configured: {cfg.stems}")
    stages = ("m0_inputs", "m1_motion", "m2_anchor", "m3_track",
              "m4_events", "m5_reconcile", "m7_persistence", "m8_merge",
              "m9_alias", "m12_identity_reservation", "m19_layered_lineage",
              "m20_oscillatory_lineage", "m22_accepted_history",
              "m6_review")
    existing = [ROOT / stage / run_name for stage in stages
                if (ROOT / stage / run_name).exists()]
    if existing:
        raise FileExistsError("immutable run name already exists: "
                              + ", ".join(map(str, existing)))
    sources = stage_m0(cfg, stem, run_name)
    stage_m1(cfg, stem, run_name, sources)
    observations, observation_table, anchor_t, _, _ = stage_m2(cfg, stem, run_name, sources)
    initial, _ = stage_m3(cfg, stem, run_name, sources, observations,
                          observation_table, anchor_t)
    labels, events, inferred, unresolved = stage_m4(
        cfg, stem, run_name, sources, initial)
    stage_m5(cfg, stem, run_name, sources, labels, events, inferred)
    persistent, _ = stage_m7_accepted(
        cfg, stem, run_name, sources, labels, observations, observation_table, anchor_t)
    merged, merge_inferred, _ = stage_m8_accepted(
        cfg, stem, run_name, sources, persistent)
    aliased, alias_inferred, _ = stage_m9_accepted(
        cfg, stem, run_name, sources, merged)
    history_inputs = AcceptedHistoryInputs()
    reserved, reservation_inferred, _ = stage_m12_accepted(
        cfg, stem, run_name, sources, aliased, observations,
        observation_table, anchor_t, history_inputs=history_inputs)
    final, lineage_inferred, tracks = stage_m19_accepted(
        cfg, stem, run_name, sources, reserved, observations,
        observation_table, anchor_t, history_inputs=history_inputs)
    m19_labels = final
    pre_oscillatory_inferred = (inferred | merge_inferred | alias_inferred
                                | reservation_inferred | lineage_inferred)
    with ThreadPoolExecutor(max_workers=2) as executor:
        m20_future = executor.submit(
            stage_m20_accepted, cfg, stem, run_name, sources,
            m19_labels.copy())
        m22_future = executor.submit(
            stage_m22_accepted_history,
            cfg, stem, run_name, sources, m19_labels.copy(), observations,
            observation_table, anchor_t,
            history_inputs=history_inputs,
            m20_result_future=m20_future,
            upstream_inferred=pre_oscillatory_inferred,
            unresolved=unresolved)
        _, oscillatory_inferred, _ = m20_future.result()
        final, _, history_inferred, tracks = m22_future.result()
    upstream_inferred = pre_oscillatory_inferred | oscillatory_inferred
    final_inferred = (_align_review_stack(upstream_inferred, len(final))
                      | history_inferred)
    stage_m6(cfg, stem, run_name, sources, final, events, final_inferred,
             _align_review_stack(unresolved, len(final)),
             anchor_t, tracks, upstream_run=run_name,
             upstream_stage="m22_accepted_history",
             accepted_base=True)
    print(f"DONE {stem} {run_name}: accepted labels "
          f"m22_accepted_history/{run_name}/out/"
          f"{stem}.tif; review m6_review/{run_name}/out/"
          f"{stem}_full_field_review.tif")


def _load_anchor_inputs(stem: str, run_name: str) -> tuple[np.ndarray, pd.DataFrame, int]:
    root = ROOT / "m2_anchor" / run_name / "out"
    observations = load_stack(root / f"{stem}_observations.tif")
    observation_table = pd.read_csv(root / "observations.csv")
    anchor_t = int(read_json(root / "anchor.json")["t"])
    return observations, observation_table, anchor_t


def run_accepted_only(run_name: str, upstream_run: str, stem: str,
                      config_path: Path | None = None,
                      checkpoint_run_name: str | None = None,
                      anchor_run_name: str | None = None) -> None:
    """Resume at M22 from an already completed current-experiment M20 run."""
    cfg = Config.load(config_path)
    if stem not in cfg.stems:
        raise ValueError(f"stem {stem!r} is not configured: {cfg.stems}")
    if (ROOT / "m22_accepted_history" / run_name).exists():
        raise FileExistsError(
            f"immutable accepted run already exists: m22_accepted_history/{run_name}")
    sources = _pinned_paths(cfg, stem)
    _verify_pins(cfg, stem)
    observations, observation_table, anchor_t = _load_anchor_inputs(
        stem, anchor_run_name or upstream_run)
    m20_path = (ROOT / "m20_oscillatory_lineage" / upstream_run /
                "out" / f"{stem}.tif")
    final = load_stack(m20_path)
    history_inputs = load_history_inputs_from_run(stem, upstream_run)
    stage_m22_accepted_history(
        cfg, stem, run_name, sources, final, observations,
        observation_table, anchor_t, source_run_name=upstream_run,
        checkpoint_run_name=checkpoint_run_name,
        history_inputs=history_inputs)
    print(f"DONE {stem} {run_name}: accepted labels resumed from "
          f"m20_oscillatory_lineage/{upstream_run}")


def run_accepted_tail(run_name: str, parent_run: str, stem: str,
                      config_path: Path | None = None,
                      events_path: Path | None = None,
                      event_members_path: Path | None = None,
                      calibration_name: str = "short-owned-companion",
                      calibration_points_path: Path | None = None) -> None:
    """Append an accepted score-calibration tail to an immutable parent."""
    cfg = Config.load(config_path)
    if (ROOT / "m22_accepted_history" / run_name).exists():
        raise FileExistsError(
            f"immutable accepted run already exists: m22_accepted_history/{run_name}")
    if stem in cfg.values.get("pinned_files", {}):
        sources = _pinned_paths(cfg, stem)
        _verify_pins(cfg, stem)
    else:
        registered_raw = ROOT / "registered inputs" / f"{stem}.tif"
        if not registered_raw.is_file():
            raise FileNotFoundError(
                f"accepted-tail raw input is missing: {registered_raw}")
        sources = {"registered_raw": registered_raw}
    parent = ROOT / "m22_accepted_history" / parent_run
    parent_manifest_path = parent / "run.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(f"accepted parent is missing: {parent}")
    parent_manifest = read_json(parent_manifest_path)
    if parent_manifest.get("status") != "done":
        raise RuntimeError(f"accepted parent is not done: {parent_run}")
    parent_work = parent / "mid/accepted_history"
    if not parent_work.is_dir():
        raise FileNotFoundError(
            f"accepted parent history is missing: {parent_work}")
    if events_path is None or event_members_path is None:
        raise ValueError(
            "score-only accepted tail requires --accepted-tail-events and "
            "--accepted-tail-event-members from the freshly scored parent")
    events_path = Path(events_path).resolve()
    event_members_path = Path(event_members_path).resolve()
    if not events_path.is_file() or not event_members_path.is_file():
        raise FileNotFoundError(
            "accepted-tail event evidence is missing: "
            f"{events_path}, {event_members_path}")
    if calibration_name in {"aggregate-shared-core", "weak-terminal-relay",
                            "single-owner-multireference-relay",
                            "event-local-subresolution",
                            "short-ownerless-subcellular",
                            "detached-projection-owner-relay",
                            "right-censored-novel-body",
                            "detached-fading-projection",
                            "fragmented-subcellular-episode",
                            "ownerless-reference-handoff"}:
        if calibration_points_path is None:
            raise ValueError(
                "this score calibration requires the current scorer's "
                "--accepted-tail-points table")
        calibration_points_path = Path(calibration_points_path).resolve()
        if not calibration_points_path.is_file():
            raise FileNotFoundError(
                "accepted-tail scored point evidence is missing: "
                f"{calibration_points_path}")

    run = Run("m22_accepted_history", run_name, {
        "stem": stem,
        "source_frame_offset": 2,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "owner_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "region_target_count": 0,
        "review_case_target_count": 0,
        "source_run": parent_run,
        "accepted_tail_parent": parent_run,
        "events_source_sha256": sha256(events_path),
        "event_members_source_sha256": sha256(event_members_path),
        "accepted_tail_calibration": calibration_name,
        "accepted_tail_points_sha256": (
            sha256(calibration_points_path)
            if calibration_points_path is not None else None),
    }, upstream=f"m22_accepted_history/{parent_run}")
    try:
        work = run.dir / "mid/accepted_history"
        _clone_accepted_history(parent_work, work)
        parent_labels_path = parent / "out" / f"{stem}.tif"
        parent_unclaimed_path = (
            parent / "out" / f"{stem}_unclaimed_original_ids.tif")
        labels = load_stack(parent_labels_path)
        unclaimed = load_stack(parent_unclaimed_path)
        latent_points = pd.read_csv(
            work / "26_latent_body_tracks/out/latent_track_points.csv")
        raw = load_stack(sources["registered_raw"])
        events = pd.read_csv(events_path)
        members = pd.read_csv(event_members_path)
        if calibration_name == "explained-projection-event":
            applications_path = (
                work / "97_delayed_owner_projection_flash/out"
                / "delayed_owner_projection_flash_applications.csv")
            if not applications_path.is_file():
                raise FileNotFoundError(
                    "accepted delayed-owner producer ledger is missing: "
                    + str(applications_path))
            applications = pd.read_csv(applications_path)
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b3_issue012_explained_projection_event_calibration(
                    labels, unclaimed, events, members, applications,
                    cfg, work)
            stage_name = "98_explained_projection_event_calibration"
            audits = {
                "explained_projection_event_audit":
                    "explained_projection_event_audit.csv",
            }
            summary_prefix = "explained_projection_event"
            calibrated_count_key = \
                "events_removed_as_explained_projections"
        elif calibration_name == "recurrent-branch-swarm":
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue011_recurrent_same_owner_branch_swarm_calibration(
                    labels, unclaimed, latent_points, events, members, cfg, work)
            stage_name = "81_recurrent_same_owner_branch_swarm_calibration"
            audits = {
                "recurrent_branch_swarm_audit":
                    "recurrent_branch_swarm_audit.csv",
            }
            summary_prefix = "recurrent_branch_swarm"
            calibrated_count_key = "events_calibrated_nonbiological"
        elif calibration_name == "aggregate-shared-core":
            scored_points = pd.read_csv(calibration_points_path)
            (candidate, candidate_unclaimed, calibrated, calibrated_members,
             source_audit, event_audit, metrics) = \
                _well_b2_issue012_aggregate_shared_core_calibration(
                    labels, unclaimed, raw, scored_points, events, members,
                    cfg, work)
            stage_name = "82_aggregate_shared_core_calibration"
            audits = {
                "aggregate_shared_core_source_audit":
                    "aggregate_shared_core_source_audit.csv",
                "aggregate_shared_core_event_audit":
                    "aggregate_shared_core_event_audit.csv",
            }
            summary_prefix = "aggregate_shared_core"
            calibrated_count_key = "events_reclassified"
        elif calibration_name == "weak-terminal-relay":
            scored_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue013_weak_terminal_reference_relay_calibration(
                    labels, unclaimed, raw, scored_points, events, members,
                    cfg, work)
            stage_name = "83_weak_terminal_reference_relay_calibration"
            audits = {
                "weak_terminal_reference_relay_audit":
                    "weak_terminal_reference_relay_audit.csv",
            }
            summary_prefix = "weak_terminal_reference_relay"
            calibrated_count_key = "events_reclassified"
        elif calibration_name == "single-owner-multireference-relay":
            scored_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue014_single_owner_multireference_relay_calibration(
                    labels, unclaimed, scored_points, events, members,
                    cfg, work)
            stage_name = "84_single_owner_multireference_relay_calibration"
            audits = {
                "single_owner_multireference_relay_audit":
                    "single_owner_multireference_relay_audit.csv",
            }
            summary_prefix = "single_owner_multireference_relay"
            calibrated_count_key = "events_reclassified"
        elif calibration_name == "event-local-subresolution":
            scored_points = pd.read_csv(calibration_points_path)
            thresholds = pd.read_csv(
                work / "25_raw_physical_hypotheses/out"
                / "frame_evidence_thresholds.csv")
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue015_event_local_subresolution_artifact_calibration(
                    labels, unclaimed, raw, scored_points, thresholds,
                    events, members, cfg, work)
            stage_name = "85_event_local_subresolution_artifact_calibration"
            audits = {
                "event_local_subresolution_audit":
                    "event_local_subresolution_audit.csv",
            }
            summary_prefix = "event_local_subresolution"
            calibrated_count_key = "events_calibrated_nonbiological"
        elif calibration_name == "short-ownerless-subcellular":
            scored_points = pd.read_csv(calibration_points_path)
            thresholds = pd.read_csv(
                work / "25_raw_physical_hypotheses/out"
                / "frame_evidence_thresholds.csv")
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue016_short_ownerless_subcellular_reference_calibration(
                    labels, unclaimed, raw, scored_points, thresholds,
                    events, members, cfg, work)
            stage_name = "86_short_ownerless_subcellular_reference_calibration"
            audits = {
                "short_ownerless_subcellular_audit":
                    "short_ownerless_subcellular_audit.csv",
            }
            summary_prefix = "short_ownerless_subcellular"
            calibrated_count_key = "events_calibrated_nonbiological"
        elif calibration_name == "detached-projection-owner-relay":
            scored_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue026_detached_projection_owner_relay_calibration(
                    labels, unclaimed, scored_points, events, members,
                    cfg, work)
            stage_name = "88_detached_projection_owner_relay_calibration"
            audits = {
                "detached_projection_owner_relay_audit":
                    "detached_projection_owner_relay_audit.csv",
            }
            summary_prefix = "detached_projection_owner_relay"
            calibrated_count_key = "events_reclassified"
        elif calibration_name == "right-censored-novel-body":
            scored_points = pd.read_csv(calibration_points_path)
            thresholds = pd.read_csv(
                work / "25_raw_physical_hypotheses/out"
                / "frame_evidence_thresholds.csv")
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue028_right_censored_novel_body_calibration(
                    labels, unclaimed, raw, scored_points, thresholds,
                    events, members, cfg, work)
            stage_name = "89_right_censored_novel_body_calibration"
            audits = {
                "right_censored_novel_body_audit":
                    "right_censored_novel_body_audit.csv",
            }
            summary_prefix = "right_censored_novel_body"
            calibrated_count_key = "right_censored_events_calibrated"
        elif calibration_name == "detached-fading-projection":
            scored_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b3_issue010_detached_fading_projection_calibration(
                    labels, unclaimed, scored_points, events, members,
                    cfg, work)
            stage_name = "96_detached_fading_projection_calibration"
            audits = {
                "detached_fading_projection_audit":
                    "detached_fading_projection_audit.csv",
            }
            summary_prefix = "detached_fading_projection"
            calibrated_count_key = "events_removed_as_nonbiological"
        elif calibration_name == "conserved-multi-anchor-projection":
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b3_issue015_conserved_multi_anchor_projection_calibration(
                    labels, unclaimed, latent_points, events, members,
                    cfg, work)
            stage_name = "104_conserved_multi_anchor_projection_calibration"
            audits = {
                "conserved_multi_anchor_encounter_audit":
                    "conserved_multi_anchor_encounter_audit.csv",
            }
            summary_prefix = "conserved_multi_anchor_projection"
            calibrated_count_key = "events_removed"
        elif calibration_name == "fragmented-subcellular-episode":
            scored_points = pd.read_csv(calibration_points_path)
            thresholds = pd.read_csv(
                work / "25_raw_physical_hypotheses/out"
                / "frame_evidence_thresholds.csv")
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b3_issue018_fragmented_subcellular_episode_calibration(
                    labels, unclaimed, scored_points, raw, thresholds,
                    events, members, cfg, work)
            stage_name = "108_fragmented_subcellular_episode_calibration"
            audits = {
                "fragmented_subcellular_episode_audit":
                    "fragmented_subcellular_episode_audit.csv",
            }
            summary_prefix = "fragmented_subcellular_episode"
            calibrated_count_key = "events_calibrated_nonbiological"
        elif calibration_name == "ownerless-reference-handoff":
            scored_points = pd.read_csv(calibration_points_path)
            thresholds = pd.read_csv(
                work / "25_raw_physical_hypotheses/out"
                / "frame_evidence_thresholds.csv")
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b3_issue022_ownerless_reference_handoff_calibration(
                    labels, unclaimed, scored_points, raw, thresholds,
                    events, members, cfg, work)
            stage_name = "112_ownerless_reference_handoff_calibration"
            audits = {
                "ownerless_reference_handoff_audit":
                    "ownerless_reference_handoff_audit.csv",
            }
            summary_prefix = "ownerless_reference_handoff"
            calibrated_count_key = "events_calibrated_nonbiological"
        elif calibration_name == "short-owned-companion":
            candidate, candidate_unclaimed, calibrated, calibrated_members, \
                audit, metrics = \
                _well_b2_issue008_short_owned_companion_projection_calibration(
                    labels, unclaimed, latent_points, events, members, cfg, work)
            stage_name = "79_short_owned_companion_projection_calibration"
            audits = {
                "short_owned_companion_audit":
                    "short_owned_companion_audit.csv",
            }
            summary_prefix = "short_owned_companion"
            calibrated_count_key = "events_calibrated_non_disruptive"
        else:
            raise ValueError(
                "unsupported accepted event calibration: "
                f"{calibration_name!r}")
        if not np.array_equal(candidate, labels):
            raise AssertionError(
                "accepted event-calibration tail changed labels")
        if not np.array_equal(candidate_unclaimed, unclaimed):
            raise AssertionError(
                "accepted event-calibration tail changed unclaimed data")

        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "unclaimed": run.dir / "out" /
                f"{stem}_unclaimed_original_ids.tif",
            "inferred": run.dir / "mid" /
                f"{stem}_inferred_accepted_history.tif",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "history_summary": run.dir / "out" /
                "accepted_history_summary.json",
            "events": run.dir / "out" / "candidate_disruptive_events.csv",
            "event_members": run.dir / "out" / "candidate_event_members.csv",
        }
        for output_key, filename in audits.items():
            outputs[output_key] = run.dir / "out" / filename
        shutil.copyfile(parent_labels_path, outputs["labels"])
        shutil.copyfile(parent_unclaimed_path, outputs["unclaimed"])
        shutil.copyfile(
            parent / "mid" / f"{stem}_inferred_accepted_history.tif",
            outputs["inferred"])
        shutil.copyfile(parent / "out/frame_identities.csv",
                        outputs["frame_identities"])
        shutil.copyfile(parent / "out/tracks.csv", outputs["tracks"])
        stage = work / stage_name / "out"
        shutil.copyfile(stage / "candidate_disruptive_events.csv",
                        outputs["events"])
        shutil.copyfile(stage / "candidate_event_members.csv",
                        outputs["event_members"])
        for output_key, filename in audits.items():
            shutil.copyfile(stage / filename, outputs[output_key])
        parent_provenance = parent / "out" / f"{stem}_provenance.tif"
        if parent_provenance.is_file():
            outputs["provenance"] = (
                run.dir / "out" / f"{stem}_provenance.tif")
            shutil.copyfile(parent_provenance, outputs["provenance"])
        summary = dict(parent_manifest.get("summary", {}))
        summary.update({
            "accepted_tail_parent": parent_run,
            f"{summary_prefix}_events_audited": int(
                metrics["events_audited"]),
            f"{summary_prefix}_{calibrated_count_key}": int(
                metrics[calibrated_count_key]),
            f"{summary_prefix}_event_catalogue_sha256": str(
                metrics["event_catalogue_sha256"]),
            "event_catalogue_sha256": str(metrics["event_catalogue_sha256"]),
            "events_sha256": str(metrics["event_catalogue_sha256"]),
            f"{summary_prefix}_identity_targets": 0,
            f"{summary_prefix}_owner_targets": 0,
            f"{summary_prefix}_track_targets": 0,
            f"{summary_prefix}_frame_targets": 0,
            f"{summary_prefix}_coordinate_targets": 0,
            f"{summary_prefix}_event_targets": 0,
            f"{summary_prefix}_region_targets": 0,
            f"{summary_prefix}_review_case_targets": 0,
        })
        write_json(outputs["history_summary"], summary)
        for label, path in outputs.items():
            run.record(label, path)
        run.finish({
            **summary,
            "targeting_mode": "field_wide_discovery",
            "final_frames": int(len(candidate)),
            "foreground_accounting_disjoint": bool(not np.any(
                (candidate > 0) & (candidate_unclaimed > 0))),
            "labels_sha256": sha256(outputs["labels"]),
            "unclaimed_sha256": sha256(outputs["unclaimed"]),
            "events_sha256": sha256(outputs["events"]),
        })
    except BaseException as error:
        run.fail(error)
        raise
    print(f"DONE {stem} {run_name}: accepted tail appended to "
          f"m22_accepted_history/{parent_run}")


def _link_or_copy_file(source: str, destination: str) -> str:
    """Hardlink immutable history files, with a cross-volume copy fallback."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _clone_accepted_history(source: Path, destination: Path) -> None:
    """Clone an immutable accepted tree without duplicating file contents."""
    shutil.copytree(
        source, destination, dirs_exist_ok=True,
        copy_function=_link_or_copy_file)


def run_accepted_label_tail(run_name: str, parent_run: str, stem: str,
                            config_path: Path | None = None,
                            events_path: Path | None = None,
                            event_members_path: Path | None = None,
                            calibration_name: str = "ownerless-cohort-gap",
                            calibration_points_path: Path | None = None,
                            ) -> None:
    """Append an accepted label-changing stage to an immutable parent.

    Event inputs must be the freshly scored and calibrated catalogue produced
    from the new labels. They are packaged byte-for-byte rather than rescored.
    """
    cfg = Config.load(config_path)
    destination = ROOT / "m22_accepted_history" / run_name
    if destination.exists():
        raise FileExistsError(
            f"immutable accepted run already exists: {destination}")
    registered_raw = ROOT / "registered inputs" / f"{stem}.tif"
    if not registered_raw.is_file():
        raise FileNotFoundError(
            f"accepted-label-tail raw input is missing: {registered_raw}")
    parent = ROOT / "m22_accepted_history" / parent_run
    parent_manifest_path = parent / "run.json"
    parent_work = parent / "mid/accepted_history"
    if not parent_manifest_path.is_file() or not parent_work.is_dir():
        raise FileNotFoundError(f"accepted parent is incomplete: {parent}")
    parent_manifest = read_json(parent_manifest_path)
    if parent_manifest.get("status") != "done":
        raise RuntimeError(f"accepted parent is not done: {parent_run}")
    if events_path is None or event_members_path is None:
        raise ValueError(
            "accepted label tail requires fresh --accepted-tail-events and "
            "--accepted-tail-event-members")
    events_path = Path(events_path).resolve()
    event_members_path = Path(event_members_path).resolve()
    if not events_path.is_file() or not event_members_path.is_file():
        raise FileNotFoundError(
            "accepted label-tail event evidence is missing: "
            f"{events_path}, {event_members_path}")
    if calibration_name in {
            "bracketed-ownerless-seat-with-fragmented-encounter",
            "established-seat-cycle-recovery",
            "dormant-seat-successor-recovery",
            "staggered-fusion-exchange-recovery",
            "large-persistent-reference-relay",
            "dormant-owner-reciprocal-partition",
                "conservative-episode-ownership",
                "original-body-continuity"}:
        if calibration_points_path is None:
            raise ValueError(
                "this accepted label repair requires the current physical "
                "--accepted-tail-points table")
        calibration_points_path = Path(calibration_points_path).resolve()
        if not calibration_points_path.is_file():
            raise FileNotFoundError(
                "accepted label-tail scored point evidence is missing: "
                f"{calibration_points_path}")

    run = Run("m22_accepted_history", run_name, {
        "stem": stem, "source_frame_offset": 2,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "owner_target_count": 0,
        "track_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "event_target_count": 0,
        "region_target_count": 0, "review_case_target_count": 0,
        "source_run": parent_run, "accepted_tail_parent": parent_run,
        "events_source_sha256": sha256(events_path),
        "event_members_source_sha256": sha256(event_members_path),
        "accepted_tail_calibration": calibration_name,
        "accepted_tail_points_sha256": (
            sha256(calibration_points_path)
            if calibration_points_path is not None else None),
    }, upstream=f"m22_accepted_history/{parent_run}")
    try:
        work = run.dir / "mid/accepted_history"
        _clone_accepted_history(parent_work, work)
        parent_labels_path = parent / "out" / f"{stem}.tif"
        parent_unclaimed_path = (
            parent / "out" / f"{stem}_unclaimed_original_ids.tif")
        labels = load_stack(parent_labels_path)
        unclaimed = load_stack(parent_unclaimed_path)
        raw_full = load_stack(registered_raw)
        raw = (raw_full if raw_full.shape == labels.shape
               else raw_full[2:2 + len(labels)])
        if raw.shape != labels.shape:
            raise ValueError(
                f"accepted label-tail raw shape {raw.shape} does not match "
                f"labels {labels.shape}")
        packaged_events_path = events_path
        packaged_event_members_path = event_members_path
        if calibration_name == "persistent-single-owner-flash":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue008_persistent_single_owner_flash(
                    labels, unclaimed, raw, cfg, work)
            summary_prefix = "persistent_single_owner_flash"
            applied_count_key = "applied_proposals"
            stage = work / "95_persistent_single_owner_flash/out"
            evidence_files = {
                "persistent_single_owner_flash_audit":
                    stage / "persistent_single_owner_flash_audit.csv",
                "persistent_single_owner_flash_applications":
                    stage / "persistent_single_owner_flash_applications.csv",
                "persistent_single_owner_flash_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "medium-window-persistent-owner-flash":
            intermediate, intermediate_unclaimed, persistent_audit, \
                persistent_applications, persistent_metrics = \
                _well_b3_issue008_persistent_single_owner_flash(
                    labels, unclaimed, raw, cfg, work)
            candidate, candidate_unclaimed, audit, applications, \
                delayed_base_audit, metrics = \
                _well_b3_issue012_delayed_owner_projection_flash(
                    intermediate, intermediate_unclaimed, raw, cfg, work)
            summary_prefix = "issue012_medium_window_persistent_flash"
            applied_count_key = "applied_proposals"
            persistent_stage = work / "95_persistent_single_owner_flash/out"
            delayed_stage = work / "97_delayed_owner_projection_flash/out"
            evidence_files = {
                "persistent_single_owner_flash_audit":
                    persistent_stage / "persistent_single_owner_flash_audit.csv",
                "persistent_single_owner_flash_applications":
                    persistent_stage /
                    "persistent_single_owner_flash_applications.csv",
                "persistent_single_owner_flash_metrics":
                    persistent_stage / "producer_metrics.json",
                "delayed_owner_projection_flash_audit":
                    delayed_stage / "delayed_owner_projection_flash_audit.csv",
                "delayed_owner_projection_flash_applications":
                    delayed_stage /
                    "delayed_owner_projection_flash_applications.csv",
                "delayed_owner_projection_flash_base_audit":
                    delayed_stage / "medium_window_base_audit.csv",
                "delayed_owner_projection_flash_metrics":
                    delayed_stage / "producer_metrics.json",
            }
        elif calibration_name == "anchored-projection-owner-relay":
            relay_labels, relay_unclaimed, relay_audit, relay_applications, \
                relay_metrics = \
                _well_b3_issue013_anchored_projection_owner_relay(
                    labels, unclaimed, raw, cfg, work)
            retired_labels, retired_unclaimed, retirement_audit, \
                retirement_applications, retirement_metrics = \
                _well_b3_issue013_ephemeral_speckle_alias_retirement(
                    relay_labels, relay_unclaimed, cfg, work)
            candidate, candidate_unclaimed, raw_gap_audit, raw_gap_metrics = \
                _well_b3_issue013_anchored_projection_raw_gap_completion(
                    retired_labels, retired_unclaimed, raw,
                    relay_audit, relay_applications, cfg, work)
            _, _, _, retirement_persistence_metrics = \
                retirement_aware_persistence_integration.run(
                    labels, candidate, retirement_audit,
                    retirement_applications, work)
            audit = raw_gap_audit
            applications = retirement_applications
            metrics = raw_gap_metrics
            summary_prefix = "issue013_anchored_projection_owner_relay"
            applied_count_key = "applied_proposals"
            relay_stage = work / "99_anchored_projection_owner_relay/out"
            retirement_stage = (
                work / "100_ephemeral_speckle_alias_retirement/out")
            raw_gap_stage = (
                work / "101_anchored_projection_raw_gap_completion/out")
            persistence_stage = work / "102_retirement_aware_persistence/out"
            evidence_files = {
                "anchored_projection_owner_relay_audit":
                    relay_stage / "anchored_projection_owner_relay_audit.csv",
                "anchored_projection_owner_relay_applications":
                    relay_stage /
                    "anchored_projection_owner_relay_applications.csv",
                "anchored_projection_owner_relay_metrics":
                    relay_stage / "producer_metrics.json",
                "ephemeral_speckle_alias_retirement_audit":
                    retirement_stage /
                    "ephemeral_speckle_alias_retirement_audit.csv",
                "ephemeral_speckle_alias_retirement_applications":
                    retirement_stage /
                    "ephemeral_speckle_alias_retirement_applications.csv",
                "ephemeral_speckle_alias_retirement_metrics":
                    retirement_stage / "producer_metrics.json",
                "anchored_projection_raw_gap_completion_audit":
                    raw_gap_stage /
                    "anchored_projection_raw_gap_completion_audit.csv",
                "anchored_projection_raw_gap_completion_metrics":
                    raw_gap_stage / "producer_metrics.json",
                "retirement_aware_persistence_metrics":
                    persistence_stage / "producer_metrics.json",
                "retirement_aware_persistence_comparison":
                    persistence_stage /
                    "surviving_identity_confidence_comparison.csv",
            }
        elif calibration_name == "terminal-projection-chain-completion":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue014_terminal_projection_chain_completion(
                    labels, unclaimed, raw, cfg, work)
            summary_prefix = "terminal_projection_chain"
            applied_count_key = "applied_proposals"
            stage = work / "103_terminal_projection_chain_completion/out"
            evidence_files = {
                "terminal_projection_chain_audit":
                    stage / "terminal_projection_chain_audit.csv",
                "terminal_projection_chain_applications":
                    stage / "terminal_projection_chain_applications.csv",
                "terminal_projection_chain_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "body-scale-ownerless-allocation":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue016_body_scale_ownerless_allocation(
                    labels, unclaimed, raw, cfg, work)
            summary_prefix = "body_scale_ownerless"
            applied_count_key = "applied_proposals"
            stage = work / "105_body_scale_ownerless_allocation/out"
            evidence_files = {
                "body_scale_ownerless_audit":
                    stage / "body_scale_ownerless_audit.csv",
                "body_scale_ownerless_applications":
                    stage / "body_scale_ownerless_applications.csv",
                "body_scale_ownerless_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "terminal-boundary-vanished-seat-partition":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue017_terminal_boundary_vanished_seat(
                    labels, unclaimed, raw, cfg, work)
            stage = work / "106_terminal_boundary_vanished_seat_partition/out"
            baseline_events = pd.read_csv(events_path)
            baseline_members = pd.read_csv(event_members_path)
            _, _, calibrated_events, calibrated_members, \
                terminal_event_audit, terminal_event_metrics = \
                _well_b3_issue017_terminal_boundary_event_calibration(
                    candidate, candidate_unclaimed,
                    baseline_events, baseline_members, audit, applications,
                    cfg, work)
            calibration_stage = work / "107_terminal_boundary_event_calibration/out"
            packaged_events_path = (
                calibration_stage / "candidate_disruptive_events.csv")
            packaged_event_members_path = (
                calibration_stage / "candidate_event_members.csv")
            summary_prefix = "terminal_boundary_seat"
            applied_count_key = "applied_proposals"
            evidence_files = {
                "terminal_boundary_seat_audit":
                    stage / "terminal_boundary_seat_audit.csv",
                "terminal_boundary_seat_frames":
                    stage / "terminal_boundary_seat_frames.csv",
                "terminal_boundary_seat_metrics":
                    stage / "producer_metrics.json",
                "terminal_boundary_event_calibration_audit":
                    calibration_stage /
                    "terminal_boundary_event_calibration_audit.csv",
            }
        elif calibration_name == \
                "bracketed-ownerless-seat-with-fragmented-encounter":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue019_bracketed_ownerless_seat_completion(
                    labels, unclaimed, raw, cfg, work)
            score_points = pd.read_csv(calibration_points_path)
            baseline_events = pd.read_csv(events_path)
            baseline_members = pd.read_csv(event_members_path)
            _, _, calibrated_events, calibrated_members, \
                lineage_audit, lineage_applications, lineage_metrics = \
                _well_b3_issue020_fragmented_encounter_lineage_calibration(
                    candidate, candidate_unclaimed, score_points,
                    baseline_events, baseline_members, cfg, work)
            stage = work / "109_bracketed_ownerless_seat_completion/out"
            calibration_stage = (
                work / "110_fragmented_encounter_lineage_calibration/out")
            packaged_events_path = (
                calibration_stage / "candidate_disruptive_events.csv")
            packaged_event_members_path = (
                calibration_stage / "candidate_event_members.csv")
            summary_prefix = "bracketed_ownerless_seat"
            applied_count_key = "eligible_gaps"
            evidence_files = {
                "bracketed_ownerless_seat_audit":
                    stage / "bracketed_ownerless_seat_audit.csv",
                "bracketed_ownerless_seat_frames":
                    stage / "bracketed_ownerless_seat_frames.csv",
                "bracketed_ownerless_seat_metrics":
                    stage / "producer_metrics.json",
                "fragmented_encounter_lineage_audit":
                    calibration_stage /
                    "fragmented_encounter_lineage_audit.csv",
                "fragmented_encounter_lineage_applications":
                    calibration_stage /
                    "fragmented_encounter_lineage_applications.csv",
                "fragmented_encounter_lineage_metrics":
                    calibration_stage / "producer_metrics.json",
            }
        elif calibration_name == "temporal-core-path-lineage":
            # Discovery must use the accepted parent's fresh catalogue.  The
            # post-repair catalogue supplied on the CLI is packaged exactly as
            # the reviewed scorer output, but cannot be used to discover the
            # repair that produced it.
            parent_events_path = (
                parent / "out/candidate_disruptive_events.csv")
            if not parent_events_path.is_file():
                raise FileNotFoundError(
                    "temporal core-path lineage requires the accepted "
                    f"parent catalogue: {parent_events_path}")
            parent_events = pd.read_csv(parent_events_path)
            candidate, candidate_unclaimed, audit, applications, \
                frame_audit, metrics = \
                _well_b3_issue021_temporal_core_path_lineage(
                    labels, unclaimed, raw, parent_events, cfg, work)
            summary_prefix = "temporal_core_path_lineage"
            applied_count_key = "applied_proposals"
            stage = work / "111_temporal_core_path_lineage/out"
            evidence_files = {
                "temporal_core_path_lineage_audit":
                    stage / "temporal_core_path_lineage_audit.csv",
                "temporal_core_path_lineage_applications":
                    stage / "temporal_core_path_lineage_applications.csv",
                "temporal_core_path_frame_audit":
                    stage / "temporal_core_path_frame_audit.csv",
                "temporal_core_path_lineage_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "asymmetric-fusion-area-flip-recovery":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue023_asymmetric_fusion_area_flip_recovery(
                    labels, unclaimed, cfg, work)
            summary_prefix = "asymmetric_fusion_area_flip"
            applied_count_key = "applied_proposals"
            stage = work / "113_asymmetric_fusion_area_flip_recovery/out"
            evidence_files = {
                "fusion_area_flip_audit":
                    stage / "fusion_area_flip_audit.csv",
                "fusion_area_flip_frames":
                    stage / "fusion_area_flip_frames.csv",
                "fusion_area_flip_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "recurrent-dominant-body-relay":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue024_recurrent_dominant_body_relay(
                    labels, unclaimed, cfg, work)
            summary_prefix = "recurrent_dominant_body_relay"
            applied_count_key = "applied_proposals"
            stage = work / "114_recurrent_dominant_body_relay/out"
            evidence_files = {
                "recurrent_dominant_body_relay_audit":
                    stage / "recurrent_dominant_body_relay_audit.csv",
                "recurrent_dominant_body_relay_frames":
                    stage / "recurrent_dominant_body_relay_frames.csv",
                "recurrent_dominant_body_relay_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "established-seat-cycle-recovery":
            score_points = pd.read_csv(calibration_points_path)
            thresholds = pd.DataFrame([
                raw_physical_hypotheses.detect_frame(frame, index, {})[1]
                for index, frame in enumerate(raw)
            ])
            candidate, candidate_unclaimed, audit, seats, applications, \
                metrics = _well_b3_issue025_established_seat_cycle_recovery(
                    labels, unclaimed, raw, score_points, thresholds,
                    cfg, work)
            summary_prefix = "established_seat_cycle"
            applied_count_key = "applied"
            stage = work / "115_established_seat_cycle_recovery/out"
            evidence_files = {
                "established_seat_cycle_audit":
                    stage / "established_seat_cycle_audit.csv",
                "established_seat_cycle_members":
                    stage / "established_seat_cycle_members.csv",
                "established_seat_cycle_frames":
                    stage / "established_seat_cycle_frames.csv",
                "established_seat_cycle_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "dormant-seat-successor-recovery":
            score_points = pd.read_csv(calibration_points_path)
            thresholds = pd.DataFrame([
                raw_physical_hypotheses.detect_frame(frame, index, {})[1]
                for index, frame in enumerate(raw)
            ])
            candidate, candidate_unclaimed, audit, members, applications, \
                projection_proof, metrics = \
                _well_b3_issue026_dormant_seat_successor_recovery(
                    labels, unclaimed, raw, score_points, thresholds,
                    cfg, work)
            summary_prefix = "dormant_seat_successor"
            applied_count_key = "applied"
            stage = work / "116_dormant_seat_successor_recovery/out"
            evidence_files = {
                "dormant_seat_successor_audit":
                    stage / "dormant_seat_successor_audit.csv",
                "dormant_seat_successor_members":
                    stage / "dormant_seat_successor_members.csv",
                "dormant_seat_successor_frames":
                    stage / "dormant_seat_successor_frames.csv",
                "dormant_seat_successor_projection_proof":
                    stage / "projection_host_proof.csv",
                "dormant_seat_successor_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "staggered-fusion-exchange-recovery":
            score_points = pd.read_csv(calibration_points_path)
            thresholds = pd.DataFrame([
                raw_physical_hypotheses.detect_frame(frame, index, {})[1]
                for index, frame in enumerate(raw)
            ])
            candidate, candidate_unclaimed, audit, applications, \
                projection_proof, metrics = \
                _well_b3_issue027_staggered_fusion_exchange_recovery(
                    labels, unclaimed, raw, score_points, thresholds,
                    cfg, work)
            summary_prefix = "staggered_fusion_exchange"
            applied_count_key = "applied_pairs"
            stage = work / "117_staggered_fusion_exchange_recovery/out"
            evidence_files = {
                "staggered_fusion_exchange_audit":
                    stage / "staggered_fusion_exchange_audit.csv",
                "staggered_fusion_exchange_frames":
                    stage / "staggered_fusion_exchange_frames.csv",
                "staggered_fusion_exchange_projection_proof":
                    stage / "staggered_fusion_exchange_projection_proof.csv",
                "staggered_fusion_exchange_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "large-persistent-reference-relay":
            score_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b4_issue001_large_persistent_reference_relay(
                    labels, unclaimed, raw, score_points, cfg, work)
            summary_prefix = "large_persistent_reference_relay"
            applied_count_key = "applied_large_references"
            stage = work / "118_large_persistent_reference_relay/out"
            evidence_files = {
                "large_persistent_reference_relay_audit":
                    stage / "large_persistent_reference_relay_audit.csv",
                "large_persistent_reference_relay_frames":
                    stage / "large_persistent_reference_relay_frames.csv",
                "large_persistent_reference_relay_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "conservative-episode-ownership":
            score_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, audit, applications, allocations, metrics = \
                _well_b4_issue003_conservative_episode_ownership(
                    labels, unclaimed, raw, score_points, cfg, work)
            summary_prefix = "conservative_episode_ownership"
            applied_count_key = "applied_proposals"
            stage = work / "120_conservative_episode_ownership/out"
            evidence_files = {
                "conservative_episode_discovery": stage / "discovery_audit.csv",
                "conservative_episode_frames": stage / "episode_frames.csv",
                "conservative_episode_allocations": stage / "displaced_body_allocations.csv",
                "conservative_episode_metrics": stage / "producer_metrics.json",
            }
        elif calibration_name == "original-body-continuity":
            score_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, audit, applications, transfers, metrics = \
                _well_b4_issue004_original_body_continuity(
                    labels, unclaimed, raw, score_points, cfg, work)
            summary_prefix = "original_body_continuity"
            applied_count_key = "applied_proposals"
            stage = work / "121_original_body_continuity/out"
            evidence_files = {
                "original_body_discovery": stage / "discovery_audit.csv",
                "original_body_frames": stage / "episode_frames.csv",
                "original_body_transfers": stage / "component_transfers.csv",
                "original_body_metrics": stage / "producer_metrics.json",
            }
        elif calibration_name == "dormant-owner-reciprocal-partition":
            score_points = pd.read_csv(calibration_points_path)
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b4_issue002_dormant_owner_reciprocal_partition(
                    labels, unclaimed, raw, score_points, cfg, work)
            summary_prefix = "dormant_owner_reciprocal_partition"
            applied_count_key = "applied_takeovers"
            stage = work / "119_dormant_owner_reciprocal_partition/out"
            evidence_files = {
                "dormant_owner_reciprocal_partition_audit":
                    stage / "reciprocal_separable_partition_audit.csv",
                "dormant_owner_reciprocal_partition_applications":
                    stage / "reciprocal_separable_partition_applications.csv",
                "dormant_owner_reciprocal_partition_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "right-censored-reciprocal-exchange":
            candidate, candidate_unclaimed, audit, metrics = \
                _well_b3_issue007_right_censored_reciprocal_exchange(
                    labels, unclaimed, cfg, work)
            applications = pd.DataFrame()
            summary_prefix = "right_censored_reciprocal_exchange"
            applied_count_key = "applied_exchanges"
            stage = work / "94_right_censored_reciprocal_exchange/out"
            evidence_files = {
                "right_censored_reciprocal_exchange_audit":
                    stage / "right_censored_reciprocal_exchange_audit.csv",
                "right_censored_reciprocal_exchange_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "right-censored-seat-partition":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue006_right_censored_seat_partition(
                    labels, unclaimed, raw, cfg, work)
            summary_prefix = "right_censored_seat_partition"
            applied_count_key = "applied_transitions"
            stage = work / "93_right_censored_seat_partition/out"
            evidence_files = {
                "right_censored_seat_partition_audit":
                    stage / "right_censored_seat_partition_audit.csv",
                "right_censored_seat_partition_frames":
                    stage / "right_censored_seat_partition_frames.csv",
                "right_censored_seat_partition_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "component-continuity-duplicate-takeover":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue005_component_continuity_duplicate_takeover(
                    labels, unclaimed, raw, cfg, work)
            summary_prefix = "component_continuity_takeover"
            applied_count_key = "applied_takeovers"
            stage = work / "92_component_continuity_duplicate_takeover/out"
            evidence_files = {
                "component_continuity_takeover_audit":
                    stage / "component_continuity_takeover_audit.csv",
                "component_continuity_takeover_applications":
                    stage / "component_continuity_takeover_applications.csv",
                "component_continuity_takeover_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "reconnected-companion-merge-partition":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b3_issue004_reconnected_companion_merge_partition(
                    labels, unclaimed, cfg, work)
            summary_prefix = "reconnected_companion_partition"
            applied_count_key = "applied_partitions"
            stage = work / "91_reconnected_companion_merge_partition/out"
            evidence_files = {
                "reconnected_companion_merge_audit":
                    stage / "reconnected_companion_merge_audit.csv",
                "reconnected_companion_merge_applications":
                    stage / "reconnected_companion_merge_applications.csv",
                "reconnected_companion_merge_metrics":
                    stage / "producer_metrics.json",
            }
        elif calibration_name == "recording-start-two-seat":
            candidate, candidate_unclaimed, audit, applications, metrics = \
                _well_b2_issue018_recording_start_two_seat_inheritance(
                    labels, unclaimed, raw, cfg, work)
            summary_prefix = "recording_start_two_seat"
            applied_count_key = "applied_proposals"
            stage = work / "87_recording_start_two_seat_inheritance/out"
            evidence_files = {
                "recording_start_two_seat_audit":
                    stage / "recording_start_two_seat_audit.csv",
                "recording_start_two_seat_applications":
                    stage / "recording_start_two_seat_applications.csv",
                "recording_start_two_seat_metrics":
                    stage / "producer_metrics.json",
            }
        else:
            candidate, candidate_unclaimed, audit, metrics = \
                _well_b2_issue009_ownerless_cohort_latent_gap_completion(
                    labels, unclaimed, raw, cfg, work)
            applications = pd.DataFrame()
            summary_prefix = "ownerless_cohort_gap"
            applied_count_key = "applied_proposals"
            stage = work / "80_ownerless_cohort_latent_gap_completion/out"
            evidence_files = {
                "ownerless_cohort_gap_audit":
                    stage / "ownerless_cohort_latent_gap_audit.csv",
                "ownerless_cohort_gap_metrics":
                    stage / "producer_metrics.json",
            }
        changed = candidate != labels
        if not int(metrics.get(applied_count_key, 0)) or not np.any(changed):
            raise AssertionError(
                "accepted label tail did not apply the reviewed repair")

        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
            "unclaimed": run.dir / "out" /
                f"{stem}_unclaimed_original_ids.tif",
            "inferred": run.dir / "mid" /
                f"{stem}_inferred_accepted_history.tif",
            "frame_identities": run.dir / "out/frame_identities.csv",
            "tracks": run.dir / "out/tracks.csv",
            "history_summary": run.dir / "out/accepted_history_summary.json",
            "events": run.dir / "out/candidate_disruptive_events.csv",
            "event_members": run.dir / "out/candidate_event_members.csv",
        }
        for label, source in evidence_files.items():
            outputs[label] = run.dir / "out" / source.name
        if calibration_name == \
                "bracketed-ownerless-seat-with-fragmented-encounter":
            # Both independently reusable stages call their local metrics file
            # producer_metrics.json. Keep distinct immutable evidence names in
            # the accepted wrapper instead of silently overwriting one.
            outputs["bracketed_ownerless_seat_metrics"] = (
                run.dir / "out/bracketed_ownerless_seat_metrics.json")
            outputs["fragmented_encounter_lineage_metrics"] = (
                run.dir / "out/fragmented_encounter_lineage_metrics.json")
        if calibration_name in {
                "body-scale-ownerless-allocation",
                "temporal-core-path-lineage",
                "asymmetric-fusion-area-flip-recovery",
                "recurrent-dominant-body-relay",
                "established-seat-cycle-recovery",
                "dormant-seat-successor-recovery",
                "staggered-fusion-exchange-recovery",
                "large-persistent-reference-relay",
                "dormant-owner-reciprocal-partition",
                "conservative-episode-ownership",
                "original-body-continuity"}:
            tifffile.imwrite(
                outputs["labels"], candidate, imagej=True,
                compression="zlib",
                metadata={"axes": "TYX", "finterval": 1800.0,
                          "tunit": "sec", "unit": "pixel"})
        elif calibration_name == "terminal-boundary-vanished-seat-partition":
            tifffile.imwrite(
                outputs["labels"], candidate, imagej=True,
                compression="zlib", metadata={"axes": "TYX"})
        elif calibration_name in {
                "recording-start-two-seat",
                "medium-window-persistent-owner-flash",
                "anchored-projection-owner-relay",
                "terminal-projection-chain-completion",
                "bracketed-ownerless-seat-with-fragmented-encounter"}:
            # Preserve the reviewed producer's canonical TIFF encoding so the
            # immutable accepted artifact is byte-identical to its oracle.
            tifffile.imwrite(outputs["labels"], candidate, compression="zlib")
        else:
            _save_final_labels(outputs["labels"], candidate)
        if calibration_name in {
                "staggered-fusion-exchange-recovery",
                "large-persistent-reference-relay",
                "conservative-episode-ownership",
                "original-body-continuity"}:
            tifffile.imwrite(
                outputs["unclaimed"], candidate_unclaimed, imagej=True,
                compression="zlib",
                metadata={"axes": "TYX", "finterval": 1800.0,
                          "tunit": "sec", "unit": "pixel"})
        else:
            shutil.copyfile(parent_unclaimed_path, outputs["unclaimed"])
        parent_inferred = load_stack(
            parent / "mid" / f"{stem}_inferred_accepted_history.tif").astype(bool)
        inferred = parent_inferred | changed
        save_stack(outputs["inferred"], inferred.astype(np.uint8),
                   cfg.values["frame_interval_min"])
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        frame_table.to_csv(outputs["frame_identities"], index=False)
        tracks.to_csv(outputs["tracks"], index=False)
        shutil.copyfile(packaged_events_path, outputs["events"])
        shutil.copyfile(packaged_event_members_path, outputs["event_members"])
        for label, source in evidence_files.items():
            shutil.copyfile(source, outputs[label])
        parent_provenance = parent / "out" / f"{stem}_provenance.tif"
        if parent_provenance.is_file():
            provenance = load_stack(parent_provenance).astype(np.uint8)
            provenance[changed] |= np.uint8(0b101)
            outputs["provenance"] = run.dir / "out" / f"{stem}_provenance.tif"
            save_stack(outputs["provenance"], provenance,
                       cfg.values["frame_interval_min"])
        summary = dict(parent_manifest.get("summary", {}))
        stage_summary = {
            "accepted_tail_parent": parent_run,
            "active_identities": int(
                len(set(map(int, np.unique(candidate))) - {0})),
            "assigned_pixels": int(np.count_nonzero(candidate)),
            "unclaimed_pixels": int(np.count_nonzero(candidate_unclaimed)),
            f"{summary_prefix}_changed_pixels": (
                int(changed.sum())
                if calibration_name in {
                    "medium-window-persistent-owner-flash",
                    "anchored-projection-owner-relay",
                    "terminal-projection-chain-completion",
                    "body-scale-ownerless-allocation",
                    "terminal-boundary-vanished-seat-partition",
                    "bracketed-ownerless-seat-with-fragmented-encounter",
                    "temporal-core-path-lineage",
                    "asymmetric-fusion-area-flip-recovery",
                    "recurrent-dominant-body-relay",
                    "established-seat-cycle-recovery",
                    "dormant-seat-successor-recovery",
                    "staggered-fusion-exchange-recovery",
                    "large-persistent-reference-relay",
                    "dormant-owner-reciprocal-partition",
                "conservative-episode-ownership",
                "original-body-continuity"}
                else int(metrics["changed_pixels"])),
            f"{summary_prefix}_changed_frames": (
                int(np.count_nonzero(
                    changed.reshape(len(changed), -1).any(axis=1)))
                if calibration_name in {
                    "medium-window-persistent-owner-flash",
                    "anchored-projection-owner-relay",
                    "terminal-projection-chain-completion",
                    "body-scale-ownerless-allocation",
                    "terminal-boundary-vanished-seat-partition",
                    "bracketed-ownerless-seat-with-fragmented-encounter",
                    "temporal-core-path-lineage",
                    "asymmetric-fusion-area-flip-recovery",
                    "recurrent-dominant-body-relay",
                    "established-seat-cycle-recovery",
                    "dormant-seat-successor-recovery",
                    "staggered-fusion-exchange-recovery",
                    "large-persistent-reference-relay",
                    "dormant-owner-reciprocal-partition",
                "conservative-episode-ownership",
                "original-body-continuity"}
                else int(metrics["changed_frames"])),
            "event_catalogue_sha256": sha256(outputs["events"]),
        }
        if calibration_name == "persistent-single-owner-flash":
            stage_summary.update({
                "persistent_single_owner_flash_proposals_audited": int(
                    metrics["proposals_audited"]),
                "persistent_single_owner_flash_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "persistent_single_owner_flash_applied_proposals": int(
                    metrics["applied_proposals"]),
                "persistent_single_owner_flash_application_rows": int(
                    len(applications)),
                "persistent_single_owner_flash_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "medium-window-persistent-owner-flash":
            stage_summary.update({
                "persistent_single_owner_flash_proposals_audited": int(
                    persistent_metrics["proposals_audited"]),
                "persistent_single_owner_flash_eligible_proposals": int(
                    persistent_metrics["eligible_proposals"]),
                "persistent_single_owner_flash_applied_proposals": int(
                    persistent_metrics["applied_proposals"]),
                "persistent_single_owner_flash_application_rows": int(
                    len(persistent_applications)),
                "persistent_single_owner_flash_changed_pixels": int(
                    persistent_metrics["changed_pixels"]),
                "persistent_single_owner_flash_changed_frames": int(
                    persistent_metrics["changed_frames"]),
                "persistent_single_owner_flash_new_duplicate_components": int(
                    persistent_metrics["new_duplicate_components"]),
                "delayed_owner_projection_flash_base_proposals_audited": int(
                    len(delayed_base_audit)),
                "delayed_owner_projection_flash_proposals_audited": int(
                    metrics["proposals_audited"]),
                "delayed_owner_projection_flash_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "delayed_owner_projection_flash_applied_proposals": int(
                    metrics["applied_proposals"]),
                "delayed_owner_projection_flash_application_rows": int(
                    len(applications)),
                "delayed_owner_projection_flash_changed_pixels": int(
                    metrics["changed_pixels"]),
                "delayed_owner_projection_flash_changed_frames": int(
                    metrics["changed_frames"]),
                "delayed_owner_projection_flash_explained_components": int(
                    metrics["new_explained_projection_components"]),
                "delayed_owner_projection_flash_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "anchored-projection-owner-relay":
            stage_summary.update({
                "anchored_projection_owner_relay_proposals_audited": int(
                    relay_metrics["proposals_audited"]),
                "anchored_projection_owner_relay_eligible_proposals": int(
                    relay_metrics["eligible_proposals"]),
                "anchored_projection_owner_relay_applied_proposals": int(
                    relay_metrics["applied_proposals"]),
                "anchored_projection_owner_relay_changed_pixels": int(
                    relay_metrics["changed_pixels"]),
                "anchored_projection_owner_relay_changed_frames": int(
                    relay_metrics["changed_frames"]),
                "anchored_projection_owner_relay_new_duplicate_components": int(
                    relay_metrics["new_duplicate_components"]),
                "ephemeral_alias_retirement_identities_audited": int(
                    retirement_metrics["identities_audited"]),
                "ephemeral_alias_retirement_eligible_proposals": int(
                    retirement_metrics["eligible_proposals"]),
                "ephemeral_alias_retirement_applied_proposals": int(
                    retirement_metrics["applied_proposals"]),
                "ephemeral_alias_retirement_changed_pixels": int(
                    retirement_metrics["changed_pixels"]),
                "ephemeral_alias_retirement_changed_frames": int(
                    retirement_metrics["changed_frames"]),
                "ephemeral_alias_retirement_removed_identity_count": int(
                    retirement_metrics["removed_identity_count"]),
                "ephemeral_alias_retirement_proof_exact": bool(
                    retirement_metrics["retirement_proof_exact"]),
                "anchored_projection_raw_gap_applied_proposals": int(
                    raw_gap_metrics["applied_proposals"]),
                "anchored_projection_raw_gap_changed_pixels": int(
                    raw_gap_metrics["changed_pixels"]),
                "anchored_projection_raw_gap_changed_frames": int(
                    raw_gap_metrics["changed_frames"]),
                "anchored_projection_raw_gap_zero_signal_additions": int(
                    raw_gap_metrics["zero_signal_additions"]),
                "retirement_aware_persistence": retirement_persistence_metrics,
            })
        elif calibration_name == "terminal-projection-chain-completion":
            stage_summary.update({
                "terminal_projection_chain_physical_tracks_audited": int(
                    metrics["physical_tracks_audited"]),
                "terminal_projection_chain_projection_parents_audited": int(
                    metrics["projection_parents_audited"]),
                "terminal_projection_chain_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "terminal_projection_chain_applied_proposals": int(
                    metrics["applied_proposals"]),
                "terminal_projection_chain_application_rows": int(
                    len(applications)),
                "terminal_projection_chain_explained_components": int(
                    metrics["new_explained_projection_components"]),
                "terminal_projection_chain_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "body-scale-ownerless-allocation":
            stage_summary.update({
                "body_scale_ownerless_tracks_audited": int(
                    metrics["tracks_audited"]),
                "body_scale_ownerless_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "body_scale_ownerless_applied_proposals": int(
                    metrics["applied_proposals"]),
                "body_scale_ownerless_application_rows": int(
                    len(applications)),
                "body_scale_ownerless_new_identity_count": int(
                    metrics["new_identity_count"]),
                "body_scale_ownerless_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "terminal-boundary-vanished-seat-partition":
            stage_summary.update({
                "terminal_boundary_seat_disappearances_audited": int(
                    metrics["terminal_disappearances_audited"]),
                "terminal_boundary_seat_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "terminal_boundary_seat_applied_proposals": int(
                    metrics["applied_proposals"]),
                "terminal_boundary_seat_frame_rows": int(len(applications)),
                "terminal_boundary_seat_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
                "terminal_boundary_event_events_audited": int(
                    terminal_event_metrics["events_audited"]),
                "terminal_boundary_event_events_removed": int(
                    terminal_event_metrics["events_removed"]),
            })
        elif calibration_name == \
                "bracketed-ownerless-seat-with-fragmented-encounter":
            stage_summary.update({
                "bracketed_ownerless_seat_identity_gaps_audited": int(
                    metrics["identity_gaps_audited"]),
                "bracketed_ownerless_seat_eligible_gaps": int(
                    metrics["eligible_gaps"]),
                "bracketed_ownerless_seat_frame_rows": int(
                    len(applications)),
                "bracketed_ownerless_seat_new_identity_count": int(
                    metrics["new_identity_count"]),
                "bracketed_ownerless_seat_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
                "fragmented_encounter_event_pairs_audited": int(
                    lineage_metrics["event_pairs_audited"]),
                "fragmented_encounter_eligible_event_pairs": int(
                    lineage_metrics["eligible_event_pairs"]),
                "fragmented_encounter_applied_event_joins": int(
                    lineage_metrics["applied_event_joins"]),
            })
        elif calibration_name == "temporal-core-path-lineage":
            stage_summary.update({
                "temporal_core_path_lineage_events_audited": int(
                    metrics["events_audited"]),
                "temporal_core_path_lineage_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "temporal_core_path_lineage_applied_proposals": int(
                    metrics["applied_proposals"]),
                "temporal_core_path_lineage_application_rows": int(
                    len(applications)),
                "temporal_core_path_lineage_new_identity_count": int(
                    metrics["new_identity_count"]),
                "temporal_core_path_lineage_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "asymmetric-fusion-area-flip-recovery":
            stage_summary.update({
                "asymmetric_fusion_area_flip_proposals_audited": int(
                    metrics["proposals_audited"]),
                "asymmetric_fusion_area_flip_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "asymmetric_fusion_area_flip_applied_proposals": int(
                    metrics["applied_proposals"]),
                "asymmetric_fusion_area_flip_fusion_restorations": int(
                    metrics["applied_fusion_restorations"]),
                "asymmetric_fusion_area_flip_area_flips": int(
                    metrics["applied_area_flips"]),
                "asymmetric_fusion_area_flip_frame_rows": int(
                    len(applications)),
                "asymmetric_fusion_area_flip_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "recurrent-dominant-body-relay":
            stage_summary.update({
                "recurrent_dominant_body_relay_proposals_audited": int(
                    metrics["proposals_audited"]),
                "recurrent_dominant_body_relay_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "recurrent_dominant_body_relay_applied_proposals": int(
                    metrics["applied_proposals"]),
                "recurrent_dominant_body_relay_closed_exchange_rejections":
                    int(metrics["closed_exchange_rejections"]),
                "recurrent_dominant_body_relay_frame_rows": int(
                    len(applications)),
                "recurrent_dominant_body_relay_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "established-seat-cycle-recovery":
            stage_summary.update({
                "established_seat_cycle_proposals_audited": int(
                    metrics["proposals_audited"]),
                "established_seat_cycle_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "established_seat_cycle_seat_members": int(
                    metrics["seat_members"]),
                "established_seat_cycle_applied": bool(metrics["applied"]),
                "established_seat_cycle_frame_rows": int(len(applications)),
                "established_seat_cycle_foreground_added_pixels": int(
                    metrics["foreground_added_pixels"]),
                "established_seat_cycle_component_excess_before": int(
                    metrics["component_excess_before"]),
                "established_seat_cycle_component_excess_after": int(
                    metrics["component_excess_after"]),
                "established_seat_cycle_duplicate_owner_frames_before": int(
                    metrics["duplicate_owner_frames_before"]),
                "established_seat_cycle_duplicate_owner_frames_after": int(
                    metrics["duplicate_owner_frames_after"]),
                "established_seat_cycle_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "dormant-seat-successor-recovery":
            stage_summary.update({
                "dormant_seat_successor_physical_tracks_audited": int(
                    metrics["physical_tracks_audited"]),
                "dormant_seat_successor_proposals_audited": int(
                    metrics["proposals_audited"]),
                "dormant_seat_successor_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "dormant_seat_successor_lineage_members": int(
                    metrics["lineage_members"]),
                "dormant_seat_successor_applied": bool(metrics["applied"]),
                "dormant_seat_successor_frame_rows": int(len(applications)),
                "dormant_seat_successor_projection_proof_rows": int(
                    len(projection_proof)),
                "dormant_seat_successor_foreground_added_pixels": int(
                    metrics["foreground_added_pixels"]),
                "dormant_seat_successor_removed_identity_count": int(
                    metrics["removed_identity_count"]),
                "dormant_seat_successor_retired_aliases": list(
                    metrics["retired_successor_aliases"]),
                "dormant_seat_successor_component_excess_before": int(
                    metrics["component_excess_before"]),
                "dormant_seat_successor_component_excess_after": int(
                    metrics["component_excess_after"]),
                "dormant_seat_successor_duplicate_owner_frames_before": int(
                    metrics["duplicate_owner_frames_before"]),
                "dormant_seat_successor_duplicate_owner_frames_after": int(
                    metrics["duplicate_owner_frames_after"]),
                "dormant_seat_successor_raw_new_duplicate_components": int(
                    metrics["raw_new_duplicate_components"]),
                "dormant_seat_successor_unexplained_new_duplicates": int(
                    metrics["unexplained_new_duplicate_components"]),
            })
        elif calibration_name == "staggered-fusion-exchange-recovery":
            stage_summary.update({
                "staggered_fusion_exchange_physical_tracks_audited": int(
                    metrics["physical_tracks_audited"]),
                "staggered_fusion_exchange_pairs_audited": int(
                    metrics["reciprocal_pairs_audited"]),
                "staggered_fusion_exchange_eligible_pairs": int(
                    metrics["eligible_pairs"]),
                "staggered_fusion_exchange_applied_pairs": int(
                    metrics["applied_pairs"]),
                "staggered_fusion_exchange_frame_rows": int(
                    len(applications)),
                "staggered_fusion_exchange_unclaimed_reassigned_pixels": int(
                    metrics["unclaimed_reassigned_pixels"]),
                "staggered_fusion_exchange_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
                "staggered_fusion_exchange_projection_proof_rows": int(
                    len(projection_proof)),
                "staggered_fusion_exchange_unproved_new_duplicates": int(
                    metrics["unproved_new_duplicate_components"]),
            })
        elif calibration_name == "large-persistent-reference-relay":
            stage_summary.update({
                "large_persistent_reference_relay_physical_tracks_audited":
                    int(metrics["physical_tracks_audited"]),
                "large_persistent_reference_relay_eligible_references":
                    int(metrics["eligible_large_references"]),
                "large_persistent_reference_relay_applied_references":
                    int(metrics["applied_large_references"]),
                "large_persistent_reference_relay_applied_runs":
                    int(metrics["applied_runs"]),
                "large_persistent_reference_relay_applied_frames":
                    int(metrics["applied_frames"]),
                "large_persistent_reference_relay_changed_unclaimed_pixels":
                    int(metrics["changed_unclaimed_pixels"]),
                "large_persistent_reference_relay_new_duplicate_components":
                    int(metrics["new_duplicate_components"]),
            })
        elif calibration_name == "conservative-episode-ownership":
            stage_summary.update({
                "conservative_episode_applied_proposals": int(metrics["applied_proposals"]),
                "conservative_episode_foreground_suppressed_pixels": int(metrics["suppressed_foreground_pixels"]),
                "conservative_episode_foreground_added_pixels": int(metrics["added_foreground_pixels"]),
                "conservative_episode_reclaimed_unclaimed_pixels": int(metrics["reclaimed_unclaimed_pixels"]),
                "conservative_episode_new_duplicate_components": int(metrics["new_duplicate_components"]),
                "conservative_episode_new_identities": metrics["new_identities"],
            })
        elif calibration_name == "original-body-continuity":
            stage_summary.update({
                "original_body_applied_proposals": int(metrics["applied_proposals"]),
                "original_body_applied_frames": int(metrics["applied_frames"]),
                "original_body_foreground_suppressed_pixels": int(metrics["suppressed_foreground_pixels"]),
                "original_body_foreground_added_pixels": int(metrics["added_foreground_pixels"]),
                "original_body_new_duplicate_components": int(metrics["new_duplicate_components"]),
                "original_body_new_identities": metrics["new_identities"],
            })
        elif calibration_name == "dormant-owner-reciprocal-partition":
            stage_summary.update({
                "dormant_owner_partition_components_audited": int(
                    metrics["components_audited"]),
                "dormant_owner_partition_transitions_audited": int(
                    metrics["owner_transitions_audited"]),
                "dormant_owner_partition_eligible_takeovers": int(
                    metrics["eligible_takeovers"]),
                "dormant_owner_partition_applied_takeovers": int(
                    metrics["applied_takeovers"]),
                "dormant_owner_partition_reciprocal_partitions": int(
                    metrics["reciprocal_partitions"]),
                "dormant_owner_partition_reciprocal_pair_frames": int(
                    metrics["reciprocal_pair_frames"]),
                "dormant_owner_partition_application_rows": int(
                    len(applications)),
                "dormant_owner_partition_new_component_excess": int(
                    metrics["new_component_excess"]),
            })
        elif calibration_name == "right-censored-reciprocal-exchange":
            stage_summary.update({
                "right_censored_reciprocal_exchange_pairs_audited": int(
                    metrics["pairs_audited"]),
                "right_censored_reciprocal_exchange_eligible_exchanges": int(
                    metrics["eligible_exchanges"]),
                "right_censored_reciprocal_exchange_applied_exchanges": int(
                    metrics["applied_exchanges"]),
                "right_censored_reciprocal_exchange_new_duplicate_components":
                    int(metrics["new_duplicate_components"]),
            })
        elif calibration_name == "right-censored-seat-partition":
            stage_summary.update({
                "right_censored_seat_partition_transitions_audited": int(
                    metrics["transitions_audited"]),
                "right_censored_seat_partition_eligible_transitions": int(
                    metrics["eligible_transitions"]),
                "right_censored_seat_partition_applied_transitions": int(
                    metrics["applied_transitions"]),
                "right_censored_seat_partition_frame_rows": int(
                    len(applications)),
                "right_censored_seat_partition_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "component-continuity-duplicate-takeover":
            stage_summary.update({
                "component_continuity_takeover_transitions_audited": int(
                    metrics["transitions_audited"]),
                "component_continuity_takeover_eligible_takeovers": int(
                    metrics["eligible_takeovers"]),
                "component_continuity_takeover_applied_takeovers": int(
                    metrics["applied_takeovers"]),
                "component_continuity_takeover_application_rows": int(
                    len(applications)),
                "component_continuity_takeover_new_duplicate_components": int(
                    metrics["new_duplicate_components"]),
            })
        elif calibration_name == "reconnected-companion-merge-partition":
            stage_summary.update({
                "reconnected_companion_partition_reconnections_audited": int(
                    metrics["reconnections_audited"]),
                "reconnected_companion_partition_candidate_pairings_audited":
                    int(len(audit)),
                "reconnected_companion_partition_eligible_partitions": int(
                    metrics["eligible_partitions"]),
                "reconnected_companion_partition_applied_partitions": int(
                    metrics["applied_partitions"]),
                "reconnected_companion_partition_application_rows": int(
                    len(applications)),
                "reconnected_companion_partition_new_duplicate_components":
                    int(metrics["new_duplicate_components"]),
            })
        else:
            stage_summary.update({
                f"{summary_prefix}_proposals_audited": int(len(audit)),
                f"{summary_prefix}_applied_proposals": int(
                    metrics["applied_proposals"]),
            })
        if calibration_name == "recording-start-two-seat":
            stage_summary.update({
                "recording_start_two_seat_eligible_proposals": int(
                    metrics["eligible_proposals"]),
                "recording_start_two_seat_explained_projection_components_added":
                    int(metrics["explained_projection_components_added"]),
                "recording_start_two_seat_unexplained_duplicate_components_added":
                    int(metrics["unexplained_duplicate_components_added"]),
            })
        elif calibration_name not in {
                "reconnected-companion-merge-partition",
                "component-continuity-duplicate-takeover",
                "persistent-single-owner-flash",
                "medium-window-persistent-owner-flash",
                "anchored-projection-owner-relay",
                "terminal-projection-chain-completion",
                "body-scale-ownerless-allocation",
                "terminal-boundary-vanished-seat-partition",
                "bracketed-ownerless-seat-with-fragmented-encounter",
                "temporal-core-path-lineage",
                "asymmetric-fusion-area-flip-recovery",
                "recurrent-dominant-body-relay",
                "established-seat-cycle-recovery",
                "dormant-seat-successor-recovery",
                "staggered-fusion-exchange-recovery",
                "large-persistent-reference-relay",
                "dormant-owner-reciprocal-partition",
                "conservative-episode-ownership",
                "original-body-continuity",
                "right-censored-reciprocal-exchange",
                "right-censored-seat-partition"}:
            stage_summary["ownerless_cohort_gap_changed_identities"] = int(
                metrics["changed_identities"])
        stage_summary.update({
            f"{summary_prefix}_{kind}_targets": 0
            for kind in ("identity", "owner", "track", "frame", "coordinate",
                         "event", "region", "review_case")
        })
        summary.update(stage_summary)
        write_json(outputs["history_summary"], summary)
        for label, path in outputs.items():
            run.record(label, path)
        run.finish({
            **summary, "targeting_mode": "field_wide_discovery",
            "final_frames": int(len(candidate)),
            "foreground_accounting_disjoint": bool(not np.any(
                (candidate > 0) & (candidate_unclaimed > 0))),
            "labels_sha256": sha256(outputs["labels"]),
            "unclaimed_sha256": sha256(outputs["unclaimed"]),
            "events_sha256": sha256(outputs["events"]),
        })
    except BaseException as error:
        run.fail(error)
        raise
    print(f"DONE {stem} {run_name}: accepted label tail appended to "
          f"m22_accepted_history/{parent_run}")


def run_resume_after_m5(run_name: str, upstream_run: str, stem: str,
                        config_path: Path | None = None) -> None:
    """Resume a failed batch after its completed M5 stage without recomputing it."""
    cfg = Config.load(config_path)
    if stem not in cfg.stems:
        raise ValueError(f"stem {stem!r} is not configured: {cfg.stems}")
    stages = ("m7_persistence", "m8_merge", "m9_alias",
              "m12_identity_reservation", "m19_layered_lineage",
              "m20_oscillatory_lineage", "m22_accepted_history", "m6_review")
    existing = [ROOT / stage / run_name for stage in stages
                if (ROOT / stage / run_name).exists()]
    if existing:
        raise FileExistsError("immutable resumed run name already exists: "
                              + ", ".join(map(str, existing)))
    sources = _pinned_paths(cfg, stem)
    _verify_pins(cfg, stem)
    observations, observation_table, anchor_t = _load_anchor_inputs(
        stem, upstream_run)
    labels = load_stack(
        ROOT / "m5_reconcile" / upstream_run / "out" / f"{stem}.tif")
    events = pd.read_csv(
        ROOT / "m4_events" / upstream_run / "out" / "events.csv")
    inferred = load_stack(
        ROOT / "m4_events" / upstream_run / "mid" /
        f"{stem}_inferred_pixels.tif").astype(bool)
    unresolved = load_stack(
        ROOT / "m4_events" / upstream_run / "mid" /
        f"{stem}_unresolved_pixels.tif").astype(bool)

    persistent, _ = stage_m7_accepted(
        cfg, stem, run_name, sources, labels, observations,
        observation_table, anchor_t)
    merged, merge_inferred, _ = stage_m8_accepted(
        cfg, stem, run_name, sources, persistent)
    aliased, alias_inferred, _ = stage_m9_accepted(
        cfg, stem, run_name, sources, merged)
    history_inputs = AcceptedHistoryInputs()
    reserved, reservation_inferred, _ = stage_m12_accepted(
        cfg, stem, run_name, sources, aliased, observations,
        observation_table, anchor_t, history_inputs=history_inputs)
    final, lineage_inferred, tracks = stage_m19_accepted(
        cfg, stem, run_name, sources, reserved, observations,
        observation_table, anchor_t, history_inputs=history_inputs)
    m19_labels = final
    pre_oscillatory_inferred = (inferred | merge_inferred | alias_inferred
                                | reservation_inferred | lineage_inferred)
    with ThreadPoolExecutor(max_workers=2) as executor:
        m20_future = executor.submit(
            stage_m20_accepted, cfg, stem, run_name, sources,
            m19_labels.copy())
        m22_future = executor.submit(
            stage_m22_accepted_history,
            cfg, stem, run_name, sources, m19_labels.copy(), observations,
            observation_table, anchor_t,
            history_inputs=history_inputs,
            m20_result_future=m20_future,
            upstream_inferred=pre_oscillatory_inferred,
            unresolved=unresolved)
        _, oscillatory_inferred, _ = m20_future.result()
        final, _, history_inferred, tracks = m22_future.result()
    upstream_inferred = pre_oscillatory_inferred | oscillatory_inferred
    final_inferred = (_align_review_stack(upstream_inferred, len(final))
                      | history_inferred)
    stage_m6(cfg, stem, run_name, sources, final, events, final_inferred,
             _align_review_stack(unresolved, len(final)),
             anchor_t, tracks, upstream_run=run_name,
             upstream_stage="m22_accepted_history", accepted_base=True)
    print(f"DONE {stem} {run_name}: resumed after "
          f"m5_reconcile/{upstream_run}")


def run_review_only(run_name: str, upstream_run: str, stem: str,
                    config_path: Path | None = None) -> None:
    cfg = Config.load(config_path)
    destination = ROOT / "m6_review" / run_name
    if destination.exists():
        raise FileExistsError(f"immutable review run already exists: {destination}")
    sources = _pinned_paths(cfg, stem)
    _verify_pins(cfg, stem)
    complete_source = (ROOT / "m22_accepted_history" / upstream_run
                       / "out" / f"{stem}.tif")
    stationary_source = (ROOT / "m21_stationary_reconciliation" / upstream_run
                         / "out" / f"{stem}.tif")
    oscillatory_source = (ROOT / "m20_oscillatory_lineage" / upstream_run
                          / "out" / f"{stem}.tif")
    layered_source = (ROOT / "m19_layered_lineage" / upstream_run
                      / "out" / f"{stem}.tif")
    reservation_source = (ROOT / "m12_identity_reservation" / upstream_run
                          / "out" / f"{stem}.tif")
    alias_source = ROOT / "m9_alias" / upstream_run / "out" / f"{stem}.tif"
    accepted_source = (complete_source if complete_source.is_file()
                       else stationary_source if stationary_source.is_file()
                       else oscillatory_source if oscillatory_source.is_file()
                       else layered_source if layered_source.is_file()
                       else reservation_source if reservation_source.is_file()
                       else alias_source)
    accepted = accepted_source.is_file()
    labels = load_stack(accepted_source if accepted else
                        ROOT / "m5_reconcile" / upstream_run / "out" / f"{stem}.tif")
    events = pd.read_csv(ROOT / "m4_events" / upstream_run / "out" / "events.csv")
    inferred, unresolved = collect_provenance(
        stem, upstream_run, len(labels), accepted=accepted)
    anchor_t = int(read_json(ROOT / "m2_anchor" / upstream_run / "out"
                             / "anchor.json")["t"])
    track_stage = ("m22_accepted_history" if complete_source.is_file()
                   else "m21_stationary_reconciliation"
                   if stationary_source.is_file()
                   else "m19_layered_lineage"
                   if layered_source.is_file()
                   else "m12_identity_reservation" if reservation_source.is_file()
                   else "m9_alias" if accepted else "m5_reconcile")
    tracks = pd.read_csv(ROOT / track_stage / upstream_run / "out" / "tracks.csv")
    stage_m6(cfg, stem, run_name, sources, labels, events, inferred, unresolved,
             anchor_t, tracks, upstream_run=upstream_run,
             upstream_stage=track_stage, accepted_base=accepted)
    print(f"DONE {stem} review {run_name} from {upstream_run}: "
          f"m6_review/{run_name}/out/{stem}_full_field_review.tif")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable motion-first base")
    parser.add_argument("--run", required=True)
    parser.add_argument("--stem", default="95_A3")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--review-only-from")
    parser.add_argument("--accepted-only-from")
    parser.add_argument("--accepted-tail-from")
    parser.add_argument("--accepted-label-tail-from")
    parser.add_argument("--accepted-tail-events", type=Path)
    parser.add_argument("--accepted-tail-event-members", type=Path)
    parser.add_argument("--accepted-tail-points", type=Path)
    parser.add_argument(
        "--accepted-tail-calibration",
        choices=("short-owned-companion", "recurrent-branch-swarm",
                 "aggregate-shared-core", "weak-terminal-relay",
                 "single-owner-multireference-relay",
                 "event-local-subresolution", "short-ownerless-subcellular",
                 "recording-start-two-seat",
                 "reconnected-companion-merge-partition",
                 "component-continuity-duplicate-takeover",
                 "right-censored-reciprocal-exchange",
                 "persistent-single-owner-flash",
                 "right-censored-seat-partition",
                 "detached-projection-owner-relay",
                 "right-censored-novel-body",
                 "detached-fading-projection",
                 "medium-window-persistent-owner-flash",
                 "anchored-projection-owner-relay",
                 "terminal-projection-chain-completion",
                 "explained-projection-event",
                 "conserved-multi-anchor-projection",
                 "body-scale-ownerless-allocation",
                 "terminal-boundary-vanished-seat-partition",
                 "fragmented-subcellular-episode",
                 "bracketed-ownerless-seat-with-fragmented-encounter",
                 "temporal-core-path-lineage",
                 "ownerless-reference-handoff",
                 "asymmetric-fusion-area-flip-recovery",
                 "recurrent-dominant-body-relay",
                 "established-seat-cycle-recovery",
                 "dormant-seat-successor-recovery",
                 "staggered-fusion-exchange-recovery",
                 "large-persistent-reference-relay",
                 "dormant-owner-reciprocal-partition",
                 "conservative-episode-ownership",
                 "original-body-continuity"),
        default="short-owned-companion")
    parser.add_argument("--accepted-checkpoints-from")
    parser.add_argument("--anchor-inputs-from")
    parser.add_argument("--resume-after-m5-from")
    args = parser.parse_args()
    modes = [args.review_only_from, args.accepted_only_from,
             args.accepted_tail_from, args.accepted_label_tail_from,
             args.resume_after_m5_from]
    if sum(value is not None for value in modes) > 1:
        parser.error("choose only one resume/review mode")
    if args.accepted_checkpoints_from and not args.accepted_only_from:
        parser.error("--accepted-checkpoints-from requires --accepted-only-from")
    if args.anchor_inputs_from and not args.accepted_only_from:
        parser.error("--anchor-inputs-from requires --accepted-only-from")
    if ((args.accepted_tail_events or args.accepted_tail_event_members
         or args.accepted_tail_points)
            and not (args.accepted_tail_from
                     or args.accepted_label_tail_from)):
        parser.error(
            "accepted-tail event inputs require --accepted-tail-from or "
            "--accepted-label-tail-from")
    if args.review_only_from:
        run_review_only(args.run, args.review_only_from, args.stem, args.config)
    elif args.accepted_only_from:
        run_accepted_only(
            args.run, args.accepted_only_from, args.stem, args.config,
            checkpoint_run_name=args.accepted_checkpoints_from,
            anchor_run_name=args.anchor_inputs_from)
    elif args.accepted_tail_from:
        run_accepted_tail(
            args.run, args.accepted_tail_from, args.stem, args.config,
            args.accepted_tail_events, args.accepted_tail_event_members,
            args.accepted_tail_calibration, args.accepted_tail_points)
    elif args.accepted_label_tail_from:
        run_accepted_label_tail(
            args.run, args.accepted_label_tail_from, args.stem, args.config,
            args.accepted_tail_events, args.accepted_tail_event_members,
            args.accepted_tail_calibration, args.accepted_tail_points)
    elif args.resume_after_m5_from:
        run_resume_after_m5(
            args.run, args.resume_after_m5_from, args.stem, args.config)
    else:
        run_base(args.run, args.stem, args.config)


if __name__ == "__main__":
    main()
