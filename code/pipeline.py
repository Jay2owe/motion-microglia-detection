from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import skimage
import tifffile
from PIL import Image, ImageDraw

from common import (Config, ROOT, Run, OUTLINE_COLOURS, display_raw, label_edges,
                    load_stack, outline_overlay, read_json, save_rgb_stack,
                    save_stack, sha256, validate_labels, write_json)
from events import resolve_events
from global_soma_ledger import (
    GlobalSomaLedgerParams, add_three_cell_bookend_observations,
    reuse_retired_identity_names, separate_substantial_soma_objects,
    substantial_identity_conflicts, track_physical_somas)
from identity_aliases import (carry_recurrent_alias_pairs_to_end,
                              consolidate_recurrent_host_aliases,
                              fill_alias_bridge_gaps)
from lineage_confidence import (identity_confidence_table,
                                restore_deferred_identity_layer)
from midpoint_reconciliation import reconcile_midpoint_bookends
from model_review import TransferParams, score_review_gold_standard
from motion_handoffs import (build_motion_pair_cache,
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
from partial_body_transfers import (
    GlobalPairLedgerParams, IdentityLedgerParams, PredecessorCoreParams,
    carry_persistent_two_core_slots, carry_recurrent_host_pair_ledger,
    correct_delayed_bookend_runs, correct_identity_ledger_transfers,
    correct_predecessor_core_merges, correct_structurally_proven_unresolved_runs)
from tracking import frame_observations, track_from_anchor


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
        run.finish({"lag_transitions": 100, "cache_rebuilt": False,
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
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted long-memory and reserved fast-motion identity resolver."""
    accepted = cfg.values["accepted_postprocessing"]["identity_reservation"]
    tracking_params = accepted["tracking"]
    motion_params = accepted["motion_handoff"]
    upstream_name = upstream_run or run_name
    run = Run("m12_identity_reservation", run_name, {
        "stem": stem,
        "accepted_version": accepted["version"],
        "approved_candidate": accepted["approved_run"],
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
        persistent_ids = set(seed_events.persistent_identity.astype(int))
        if len(persistent_ids) != 1:
            raise AssertionError(
                "accepted identity reservation did not resolve one supplied identity")
        supplied_identity = persistent_ids.pop()
        trajectory, trajectory_events, trajectory_audit = \
            carry_established_identities_along_motion(
                seeded, raw, lag, motion_params, pair_cache=pair_cache)

        canonical_reservations: list[dict] = []
        for seed in seed_events.itertuples():
            target = int(seed.persistent_identity)
            alias = int(seed.displaced_identity)
            start_t = int(seed.t) + 1
            while start_t < len(seeded) and np.any(seeded[start_t] == alias):
                start_t += 1
            if start_t >= len(seeded):
                raise AssertionError("accepted corrected motion alias never ended")
            for frame in range(start_t, len(trajectory)):
                trajectory[frame][trajectory[frame] == target] = alias
            canonical_reservations.append({
                "kind": "motion_alias_ended_target_reserved",
                "identity": target,
                "to_t": start_t,
            })
        reservations = pd.DataFrame(canonical_reservations)
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
        missing = [t + 1 for t in range(54, len(candidate))
                   if not np.any(candidate[t] == supplied_identity)]
        if missing:
            raise AssertionError(
                f"accepted supplied identity missing on ImageJ frames {missing}")

        inferred = (geometric_inferred | host_inferred
                    | (long_memory != retracked)
                    | (candidate != long_memory))
        frame_table, tracks = _identity_tables(candidate, raw, inferred)
        outputs = {
            "labels": run.dir / "out" / f"{stem}.tif",
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
            "trajectory_events": run.dir / "out" / "motion_trajectory_events.csv",
            "trajectory_audit": run.dir / "out" / "motion_trajectory_audit.csv",
            "host_carry": run.dir / "out" / "reserved_identity_host_carry.csv",
            "frame_identities": run.dir / "out" / "frame_identities.csv",
            "tracks": run.dir / "out" / "tracks.csv",
            "inferred": run.dir / "mid" / f"{stem}_inferred_identity_reservation.tif",
            "outlines": run.dir / "qc" / f"{stem}_outline.tif",
        }
        save_stack(outputs["labels"], candidate, cfg.values["frame_interval_min"])
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
            "supplied_persistent_identity": int(supplied_identity),
            "supplied_identity_continuous_imagej_frames_55_101": True,
            "maximum_detection_gap_frames": int(tracking_params["max_gap_frames"]),
            "global_identities_before": int(
                len(set(map(int, np.unique(baseline))) - {0})),
            "global_identities_after": int(len(tracks)),
            "global_identity_reduction": int(
                len(set(map(int, np.unique(baseline))) - {0}) - len(tracks)),
            "first_handoff_identity_seeds": int(len(seed_events)),
            "motion_trajectory_events": int(len(trajectory_events)),
            "forced_host_frames": int(len(host_carry)),
            "forced_host_first_imagej_frame": int(host_carry.imagej_frame.min()),
            "forced_host_last_imagej_frame": int(host_carry.imagej_frame.max()),
            "foreground_pixels_added": 0,
            "foreground_pixels_removed": 0,
            "foreground_support_unchanged": True,
            "co_present_identity_errors": 0,
            "human_review": "approved",
            "accepted_into_base": True,
        }
        run.finish(summary)
        return candidate, inferred, tracks
    except BaseException as error:
        run.fail(error)
        raise


def stage_m19_accepted(
        cfg: Config, stem: str, run_name: str, sources: dict[str, Path],
        baseline: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, anchor_t: int,
        upstream_run: str | None = None,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Accepted destination-led confidence-layer and lineage reunion stage."""
    accepted = cfg.values["accepted_postprocessing"]["layered_lineage"]
    upstream_name = upstream_run or run_name
    run = Run("m19_layered_lineage", run_name, {
        "stem": stem,
        "accepted_version": accepted["version"],
        "approved_candidate": accepted["approved_candidate"],
        "mode": ("rebuild physical somas, track red destinations to protected "
                 "blue sources, then reunite non-overlapping lineages"),
        "confidence": accepted["confidence"],
        "tracking": accepted["tracking"],
        "reconciliation": accepted["reconciliation"],
    }, upstream=f"m12_identity_reservation/{upstream_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        gold = pd.read_csv(ROOT / accepted["review_benchmark"])
        objects = pd.read_csv(ROOT / accepted["review_objects"])
        review_runs = pd.read_csv(ROOT / accepted["review_runs"])
        reviewer = pd.read_csv(ROOT / accepted["approved_review_labels"])
        if len(reviewer) != 8 or set(
                reviewer.verdict.astype(str).str.lower()) != {"approved"}:
            raise AssertionError(
                "accepted layered-lineage stage lacks eight approved verdicts")

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
                runs=review_runs, skip_run_ids=used)
        proved_frames = pd.concat(
            [direct_frames, bookend_frames, persistent_frames, structural_frames],
            ignore_index=True, sort=False)
        shape_source, recurrent_events, recurrent_frames, recurrent_inferred = \
            carry_recurrent_host_pair_ledger(
                shape_source, raw, proved_frames, recurrent_params,
                detection_labels=baseline, objects=objects, runs=review_runs)
        shape_source, predecessor_events, predecessor_frames, predecessor_inferred = \
            correct_predecessor_core_merges(
                shape_source, raw, predecessor_params,
                detection_labels=baseline, objects=objects, runs=review_runs)
        expected_shape = load_stack(ROOT / accepted["expected_shape_source"])
        if not np.array_equal(shape_source, expected_shape):
            raise AssertionError(
                "accepted foreground partition source was not reproduced exactly")

        # Rebuild Attempt 005's physical-soma comparison from the accepted base.
        soma_params = GlobalSomaLedgerParams(**accepted["soma_ledger"])
        physical, soma_audit = separate_substantial_soma_objects(
            shape_source, soma_params.minimum_substantial_core_px)
        preliminary, preliminary_links, preliminary_evidence, preliminary_aliases = \
            track_physical_somas(
                physical, baseline, raw, lag, anchor_t,
                accepted["comparison_tracking"], observations,
                observation_table, soma_params)
        physical, three_cell_partitions = add_three_cell_bookend_observations(
            physical, preliminary, raw, lag, gold,
            cfg.values["accepted_postprocessing"]["persistent_merges"])
        comparison, comparison_links, comparison_evidence, comparison_aliases = \
            track_physical_somas(
                physical, baseline, raw, lag, anchor_t,
                accepted["comparison_tracking"], observations,
                observation_table, soma_params)
        comparison, comparison_reused = reuse_retired_identity_names(
            comparison, baseline,
            int(accepted["minimum_retired_name_overlap_px"]))
        expected_comparison = load_stack(ROOT / accepted["expected_comparison"])
        if not np.array_equal(comparison, expected_comparison):
            raise AssertionError(
                "accepted Attempt 005 comparison was not reproduced exactly")

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
                observations, observation_table, soma_params)
        stable, deferred = restore_deferred_identity_layer(
            stable, physical, comparison, high_ids)
        stable, stable_reused = reuse_retired_identity_names(
            stable, comparison,
            int(accepted["minimum_retired_name_overlap_px"]))
        expected_stable = load_stack(
            ROOT / accepted["expected_stable_intermediate"])
        if not np.array_equal(stable, expected_stable):
            raise AssertionError(
                "accepted destination-led intermediate was not reproduced exactly")

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

        reviewed_path = ROOT / accepted["approved_candidate"]
        if sha256(reviewed_path) != accepted["approved_candidate_sha256"]:
            raise AssertionError("approved Attempt 019 fingerprint changed")
        reviewed = load_stack(reviewed_path)
        if not np.array_equal(candidate, reviewed):
            raise AssertionError(
                "accepted layered-lineage rebuild differs from reviewed Attempt 019")

        score = score_review_gold_standard(candidate, gold)
        under = score[score.truth == "undersegmentation"]
        two_ranks = set(gold.loc[gold.true_cell_count == 2, "review_rank"])
        two = under[under.review_rank.isin(two_ranks)]
        singles = score[score.truth == "single_cell"]
        uncertain = score[score.truth == "uncertain"]
        conflicts = substantial_identity_conflicts(
            candidate, soma_params.minimum_substantial_core_px)
        if int(two.cores_separated.sum()) != 44:
            raise AssertionError("accepted Issue 003 lost a reviewed two-cell case")
        if int(under.cores_separated.sum()) != 45:
            raise AssertionError("accepted Issue 003 lost an under-segmentation")
        if int(singles.gate_passed.sum()) != 15:
            raise AssertionError("accepted Issue 003 split a genuine single cell")
        if int(uncertain.cores_separated.sum()) != 0:
            raise AssertionError("accepted Issue 003 changed an uncertain case")
        if len(conflicts):
            raise AssertionError("accepted Issue 003 duplicated a soma identity")
        if not np.array_equal(candidate > 0, baseline > 0):
            raise AssertionError("accepted Issue 003 changed foreground support")
        if len(set(map(int, np.unique(candidate))) - {0}) != 112:
            raise AssertionError("accepted Issue 003 identity count is not 112")
        if not np.all(np.any(candidate[54:101] == 58, axis=(1, 2))):
            raise AssertionError("accepted Issue 002 identity 58 was interrupted")

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
            "soma_audit": run.dir / "out" / "physical_soma_observations.csv",
            "three_cell_partitions": run.dir / "out" / "three_cell_partition.csv",
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
            "reviewer_verdicts": run.dir / "out" / "approved_reviewer_verdicts.csv",
            "gold_score": run.dir / "out" / "review_gold_standard_score.csv",
            "conflicts": run.dir / "out" / "co_present_soma_identity_conflicts.csv",
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
        soma_audit.to_csv(outputs["soma_audit"], index=False)
        three_cell_partitions.to_csv(outputs["three_cell_partitions"], index=False)
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
        reviewer.to_csv(outputs["reviewer_verdicts"], index=False)
        score.to_csv(outputs["gold_score"], index=False)
        conflicts.to_csv(outputs["conflicts"], index=False)
        save_rgb_stack(outputs["outlines"], outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)

        output_sha = sha256(outputs["labels"])
        if output_sha != accepted["approved_candidate_sha256"]:
            raise AssertionError(
                "accepted output file is not byte-identical to reviewed Attempt 019")
        summary = {
            "human_review": "approved",
            "accepted_into_base": True,
            "approved_lineage_reunions": int(len(reunions)),
            "layer_1_established_identities": int(len(protected_ids)),
            "global_identities_before": int(
                len(set(map(int, np.unique(baseline))) - {0})),
            "global_identities_after": int(len(tracks)),
            "global_identity_reduction": int(
                len(set(map(int, np.unique(baseline))) - {0}) - len(tracks)),
            "confirmed_two_cell_cases": "44/44",
            "reviewed_single_cells": "15/15",
            "uncertain_cases_changed": 0,
            "co_present_soma_identity_conflicts": 0,
            "foreground_support_unchanged": True,
            "array_exact_to_attempt_019": True,
            "file_sha_exact_to_attempt_019": True,
            "labels_sha256": output_sha,
        }
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
        "approved_candidate": accepted["approved_candidate"],
        "mode": ("whole-movie lineage graph followed by non-conflicting "
                 "one-frame host partitions"),
        "lineage_graph": accepted["lineage_graph"],
        "host_merge_graph": accepted["host_merge_graph"],
        "registration": "unchanged",
    }, upstream=f"m19_layered_lineage/{upstream_name}")
    try:
        raw = load_stack(sources["registered_raw"])
        lag = load_stack(sources["lag_float"])
        approval_path = ROOT / accepted["approval_record"]
        approval = approval_path.read_text(encoding="utf-8")
        if "I004-A006" not in approval or "accept Approach 3" not in approval:
            raise AssertionError("Issue 004 approval record is incomplete")
        reviewer = pd.read_csv(ROOT / accepted["approved_review_labels"])
        approved_row = reviewer[
            reviewer.variant.astype(str).eq("3")
            & reviewer.verdict.astype(str).str.lower().eq("approved")]
        if len(approved_row) != 1:
            raise AssertionError("Approach 3 lacks one explicit approved verdict")

        lineage, lineage_decisions, lineage_candidates = optimise_lineage_graph(
            baseline, lag, accepted["lineage_graph"],
            int(accepted["minimum_substantial_core_px"]))
        candidate, host_decisions, host_candidates = optimise_host_merge_graph(
            lineage, raw, lag, accepted["host_merge_graph"],
            cfg.values["accepted_postprocessing"]["persistent_merges"])

        reviewed_path = ROOT / accepted["approved_candidate"]
        if sha256(reviewed_path) != accepted["approved_candidate_sha256"]:
            raise AssertionError("approved Issue 004 candidate fingerprint changed")
        reviewed = load_stack(reviewed_path)
        if not np.array_equal(candidate, reviewed):
            raise AssertionError(
                "accepted Issue 004 rebuild differs from approved Approach 3")

        gold = pd.read_csv(ROOT / accepted["review_benchmark"])
        score = score_review_gold_standard(candidate, gold)
        under = score[score.truth == "undersegmentation"]
        two_ranks = set(gold.loc[gold.true_cell_count == 2, "review_rank"])
        two = under[under.review_rank.isin(two_ranks)]
        singles = score[score.truth == "single_cell"]
        uncertain = score[score.truth == "uncertain"]
        conflicts = substantial_identity_conflicts(
            candidate, int(accepted["minimum_substantial_core_px"]))
        if int(two.cores_separated.sum()) != 44:
            raise AssertionError("accepted Issue 004 lost a reviewed two-cell case")
        if int(under.cores_separated.sum()) != 45:
            raise AssertionError("accepted Issue 004 lost an under-segmentation")
        if int(singles.gate_passed.sum()) != 15:
            raise AssertionError("accepted Issue 004 split a genuine single cell")
        if int(uncertain.cores_separated.sum()) != 0:
            raise AssertionError("accepted Issue 004 changed an uncertain case")
        if len(conflicts):
            raise AssertionError("accepted Issue 004 duplicated a soma identity")
        if not np.array_equal(candidate > 0, baseline > 0):
            raise AssertionError("accepted Issue 004 changed foreground support")
        identities = set(map(int, np.unique(candidate))) - {0}
        if len(identities) != 111:
            raise AssertionError("accepted Issue 004 identity count is not 111")
        if np.any(candidate == 149):
            raise AssertionError("accepted Issue 004 retained relay identity 149")
        if not np.all(np.any(candidate == 55, axis=(1, 2))):
            raise AssertionError("accepted Issue 004 identity 55 is not continuous")
        if len(lineage_decisions) != 1 or len(host_decisions) != 7:
            raise AssertionError("accepted Issue 004 decision count changed")

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
            "reviewer_verdict": run.dir / "out/approved_reviewer_verdict.csv",
            "gold_score": run.dir / "out/review_gold_standard_score.csv",
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
        approved_row.to_csv(outputs["reviewer_verdict"], index=False)
        score.to_csv(outputs["gold_score"], index=False)
        conflicts.to_csv(outputs["conflicts"], index=False)
        save_rgb_stack(outputs["outlines"],
                       outline_overlay(raw, candidate, thick=2),
                       cfg.values["frame_interval_min"])
        for label, path in outputs.items():
            run.record(label, path)
        output_sha = sha256(outputs["labels"])
        if output_sha != accepted["approved_candidate_sha256"]:
            raise AssertionError(
                "accepted Issue 004 output is not byte-identical to Approach 3")
        summary = {
            "human_review": "approved",
            "accepted_into_base": True,
            "global_identities_before": int(len(
                set(map(int, np.unique(baseline))) - {0})),
            "global_identities_after": len(tracks),
            "lineage_graph_decisions": int(len(lineage_decisions)),
            "host_partition_graph_decisions": int(len(host_decisions)),
            "target_identity_55_frames": int(np.count_nonzero(
                np.any(candidate == 55, axis=(1, 2)))),
            "target_identity_149_removed": True,
            "confirmed_two_cell_cases": "44/44",
            "reviewed_single_cells": "15/15",
            "uncertain_cases_changed": 0,
            "co_present_soma_identity_conflicts": 0,
            "foreground_support_unchanged": True,
            "registration_added": False,
            "array_exact_to_attempt_006": True,
            "file_sha_exact_to_attempt_006": True,
            "labels_sha256": output_sha,
        }
        run.finish(summary)
        return candidate, inferred, tracks
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
        raw = load_stack(sources["registered_raw"])
        composite = load_stack(sources["motion_composite"])
        identity_view = outline_overlay(raw, labels, thick=2)
        motion_view = _motion_rgb(raw, composite, labels)
        event_view = outline_overlay(raw, labels, thick=2, inferred=inferred,
                                     unresolved=unresolved)
        review = np.concatenate([identity_view, motion_view, event_view], axis=2)
        review = _annotate_review(review, anchor_t)
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
            "field_width_px": labels.shape[2], "anchor_imagej_frame": anchor_t + 1,
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
                "- Complete 101-frame identity outlines: pending\n"
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
               "Review all 101 frames before approving the base. Record the decision in `reviewer_feedback.md`.\n"),
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
              "m20_oscillatory_lineage",
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
    reserved, reservation_inferred, _ = stage_m12_accepted(
        cfg, stem, run_name, sources, aliased, observations,
        observation_table, anchor_t)
    final, lineage_inferred, tracks = stage_m19_accepted(
        cfg, stem, run_name, sources, reserved, observations,
        observation_table, anchor_t)
    final, oscillatory_inferred, tracks = stage_m20_accepted(
        cfg, stem, run_name, sources, final)
    final_inferred = (inferred | merge_inferred | alias_inferred
                      | reservation_inferred | lineage_inferred
                      | oscillatory_inferred)
    stage_m6(cfg, stem, run_name, sources, final, events, final_inferred, unresolved,
             anchor_t, tracks, upstream_run=run_name,
             upstream_stage="m20_oscillatory_lineage",
             accepted_base=True)
    print(f"DONE {stem} {run_name}: accepted labels "
          f"m20_oscillatory_lineage/{run_name}/out/"
          f"{stem}.tif; review m6_review/{run_name}/out/"
          f"{stem}_full_field_review.tif")


def run_review_only(run_name: str, upstream_run: str, stem: str,
                    config_path: Path | None = None) -> None:
    cfg = Config.load(config_path)
    destination = ROOT / "m6_review" / run_name
    if destination.exists():
        raise FileExistsError(f"immutable review run already exists: {destination}")
    sources = _pinned_paths(cfg, stem)
    _verify_pins(cfg, stem)
    oscillatory_source = (ROOT / "m20_oscillatory_lineage" / upstream_run
                          / "out" / f"{stem}.tif")
    layered_source = (ROOT / "m19_layered_lineage" / upstream_run
                      / "out" / f"{stem}.tif")
    reservation_source = (ROOT / "m12_identity_reservation" / upstream_run
                          / "out" / f"{stem}.tif")
    alias_source = ROOT / "m9_alias" / upstream_run / "out" / f"{stem}.tif"
    accepted_source = (oscillatory_source if oscillatory_source.is_file()
                       else layered_source if layered_source.is_file()
                       else reservation_source if reservation_source.is_file()
                       else alias_source)
    accepted = accepted_source.is_file()
    labels = load_stack(accepted_source if accepted else
                        ROOT / "m5_reconcile" / upstream_run / "out" / f"{stem}.tif")
    events = pd.read_csv(ROOT / "m4_events" / upstream_run / "out" / "events.csv")
    inferred = load_stack(ROOT / "m4_events" / upstream_run / "mid"
                          / f"{stem}_inferred_pixels.tif").astype(bool)
    if accepted:
        inferred |= load_stack(ROOT / "m8_merge" / upstream_run / "mid"
                               / f"{stem}_inferred_merge_pixels.tif").astype(bool)
        inferred |= load_stack(ROOT / "m9_alias" / upstream_run / "mid"
                               / f"{stem}_inferred_global_alias_pixels.tif").astype(bool)
        reservation_inferred = (ROOT / "m12_identity_reservation" / upstream_run
                                / "mid" / f"{stem}_inferred_identity_reservation.tif")
        if reservation_inferred.is_file():
            inferred |= load_stack(reservation_inferred).astype(bool)
        lineage_inferred = (ROOT / "m19_layered_lineage" / upstream_run
                            / "mid" / f"{stem}_inferred_layered_lineage.tif")
        if lineage_inferred.is_file():
            inferred |= load_stack(lineage_inferred).astype(bool)
        oscillatory_inferred = (ROOT / "m20_oscillatory_lineage" / upstream_run
                                / "mid" / f"{stem}_inferred_oscillatory_lineage.tif")
        if oscillatory_inferred.is_file():
            inferred |= load_stack(oscillatory_inferred).astype(bool)
    unresolved = load_stack(ROOT / "m4_events" / upstream_run / "mid"
                            / f"{stem}_unresolved_pixels.tif").astype(bool)
    anchor_t = int(read_json(ROOT / "m2_anchor" / upstream_run / "out"
                             / "anchor.json")["t"])
    track_stage = ("m19_layered_lineage"
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
    args = parser.parse_args()
    if args.review_only_from:
        run_review_only(args.run, args.review_only_from, args.stem, args.config)
    else:
        run_base(args.run, args.stem, args.config)


if __name__ == "__main__":
    main()
