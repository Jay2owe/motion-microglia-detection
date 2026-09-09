"""Run the complete accepted, identity-blind postprocessing history.

This module promotes the non-revoked accepted biological operations through B2
stage 80 into one production path. B2 stage 87, B3 stages 90-95, 97 and
99-106, 109 and post-score label stages 111 and 113-121, and score-only stages 78,
79, 81-86, 88, 89, 96, 98, 104, 107, 108, 110 and 112 are
exposed here for downstream fresh-catalogue calibration. Historical TIFFs are
regression oracles only; production receives current-run arrays and evidence
tables.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

from common import Config, ROOT, load_stack, save_stack
from global_soma_ledger import (
    GlobalSomaLedgerParams, reuse_retired_identity_names,
    track_physical_somas)
from lineage_confidence import (
    identity_confidence_table, multilayer_identity_confidence_table,
    restore_deferred_identity_layer)
from local_lineage_reservations import reserve_local_lineages
import latent_body_tracks
import merge_residency
from midpoint_reconciliation import reconcile_midpoint_bookends
from oscillatory_reconciliation import (
    optimise_host_merge_graph, optimise_lineage_graph)
from partial_body_transfers import detect_identity_ledger_transfers
from physical_lineage_aliases import reconcile_physical_lineages
import owner_consensus
import ownerless_body_recovery
import ownerless_cohort_recovery
import persistent_body_identity
from presence import (DEFAULTS as PRESENCE_DEFAULTS,
                      complete_presence_bidirectional,
                      resolve_minimum_persistent_area)
from reciprocal_lineages import reconcile_reciprocal_lineages
import raw_physical_hypotheses
import recording_start_split
import resident_takeover
import separable_merge_recovery
import bracketed_encounter_recovery
import recurrent_exclusive_owner_relay
import isolated_owner_blip_recovery
import atomic_two_seat_recovery
import recording_start_duplicate_release
import late_owner_backfill
import bracketed_handoff_recovery
import post_split_backfill
import concurrent_duplicate_invasion
import transiently_misowned_body_recovery
import isolated_unowned_lifetime
import boundary_owner_blip
import foreign_owner_terminal_convergence
import terminal_companion_assimilation
import cross_reference_owner_relay
import recording_boundary_owner_cycle
import duplicate_soma_exclusivity
import complete_duplicate_lineage
import recurrent_two_body_fusion
import exclusive_pair_seat_recovery
import reciprocal_two_seat_exchange
import bounded_owner_excursion
import recurrent_isolated_alias
import distinct_history_core_preservation
import branched_motion_lineage_integration
import complete_flip_recovery_integration
import projection_tolerant_seat_integration
import retroactive_successor_integration
import terminal_two_core_split_integration
import bracketed_established_owner_invasion_integration
import terminal_projection_diversion_integration
import delayed_projection_reclaim_integration
import boundary_owner_excursion_integration
import terminal_two_seat_assimilation_integration
import bracketed_mixed_owner_flash_integration
import dominant_seat_projection_diversion_integration
import subresolution_point_artifact_calibration_integration
import short_owned_companion_projection_calibration_integration
import ownerless_cohort_latent_gap_completion_integration
import recurrent_same_owner_branch_swarm_calibration_integration
import aggregate_shared_core_calibration_integration
import weak_terminal_reference_relay_calibration_integration
import single_owner_multireference_relay_calibration_integration
import event_local_subresolution_artifact_calibration_integration
import short_ownerless_subcellular_reference_calibration_integration
import recording_start_two_seat_inheritance_integration
import detached_projection_owner_relay_calibration_integration
import right_censored_novel_body_calibration_integration
import detached_fading_projection_calibration_integration
import delayed_owner_projection_flash_integration
import explained_projection_event_calibration_integration
import gap_tolerant_reciprocal_seat_exchange_integration
import reconnected_companion_merge_partition_integration
import component_continuity_duplicate_takeover_integration
import right_censored_seat_partition_integration
import right_censored_reciprocal_exchange_integration
import persistent_single_owner_flash_integration
import anchored_projection_owner_relay_integration
import ephemeral_speckle_alias_retirement_integration
import anchored_projection_raw_gap_completion_integration
import retirement_aware_persistence_integration
import terminal_projection_chain_completion_integration
import conserved_multi_anchor_projection_calibration_integration
import body_scale_ownerless_allocation_integration
import terminal_boundary_vanished_seat_integration
import terminal_boundary_event_calibration_integration
import fragmented_subcellular_episode_calibration_integration
import bracketed_ownerless_seat_completion_integration
import fragmented_encounter_lineage_calibration_integration
import temporal_core_path_lineage_integration
import ownerless_reference_handoff_calibration_integration
import asymmetric_fusion_area_flip_recovery_integration
import recurrent_dominant_body_relay_integration
import established_seat_cycle_recovery_integration
import dormant_seat_successor_recovery_integration
import staggered_fusion_exchange_recovery_integration
import large_persistent_reference_relay_integration
import dormant_owner_reciprocal_partition_integration
import conservative_episode_ownership_integration
import original_body_continuity_integration
from stationary_gap_recovery import recover_stationary_gaps
from stationary_reconciliation import (
    _assert_identity_blind_params, discover_stationary_takeovers,
    reconciliation_params)
from territory_memory import reconcile_territory_memory
from territory_reconciliation import (
    optimise_territory_ownership_graph,
    reconcile_same_territory_replacements)
from unclaimed_body_identity import assign_persistent_unclaimed_bodies
from tracking import prepare_tracking_observations


PRODUCTION_OPS = Path(__file__).resolve().with_name("accepted_history_ops")
PARAMETER_PATH = Path(__file__).resolve().with_name(
    "accepted_history_parameters.json")
ACCEPTED_PARAMETERS = json.loads(PARAMETER_PATH.read_text(encoding="utf-8"))
I006 = PRODUCTION_OPS / "issue006"
I007 = PRODUCTION_OPS / "issue007"
I008 = PRODUCTION_OPS / "issue008"
I009 = PRODUCTION_OPS / "issue009"
I010 = PRODUCTION_OPS / "issue010"
I011 = PRODUCTION_OPS / "issue011"
I012 = PRODUCTION_OPS / "issue012"
I013 = PRODUCTION_OPS / "issue013"
I015 = PRODUCTION_OPS / "issue015"
I017 = PRODUCTION_OPS / "issue017"


@dataclass
class StageDirs:
    root: Path

    def __post_init__(self) -> None:
        self.out = self.root / "out"
        self.mid = self.root / "mid"
        self.qc = self.root / "qc"
        for path in (self.out, self.mid, self.qc):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class AcceptedHistoryInputs:
    """Current-run evidence consumed by the accepted-history calculation."""
    physical_somas: np.ndarray | None = None
    reservation_labels: np.ndarray | None = None
    pre_reservation_labels: np.ndarray | None = None
    host_audit: pd.DataFrame | None = None
    motion_reserved: set[int] | None = None
    motion_reservation_events: pd.DataFrame | None = None

    def required(self) -> tuple[
            np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, set[int]]:
        values = {
            "physical_somas": self.physical_somas,
            "reservation_labels": self.reservation_labels,
            "pre_reservation_labels": self.pre_reservation_labels,
            "host_audit": self.host_audit,
            "motion_reserved": self.motion_reserved,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                "accepted-history inputs are incomplete: " + ", ".join(missing))
        return (
            self.physical_somas, self.reservation_labels,
            self.pre_reservation_labels, self.host_audit,
            self.motion_reserved,
        )


def _params(issue: str, name: str) -> dict:
    """Return a fresh production-owned copy of one accepted parameter set."""
    return deepcopy(ACCEPTED_PARAMETERS[issue][name])


def _general(round_name: str) -> dict:
    return deepcopy(ACCEPTED_PARAMETERS["issue023"][round_name])


def read_csv_or_empty(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Read an evidence table, accepting a zero-byte file as no evidence."""
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(required_columns))
    missing = set(required_columns) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return table


def _issue_module(code_dir: Path, name: str):
    """Load one production-local generic algorithm without import collisions."""
    local_names = {path.stem for path in code_dir.glob("*.py")}
    for local_name in local_names:
        sys.modules.pop(local_name, None)
    sys.path.insert(0, str(code_dir))
    try:
        return importlib.import_module(name)
    finally:
        try:
            sys.path.remove(str(code_dir))
        except ValueError:
            pass


def _save(path: Path, array: np.ndarray, interval: float) -> Path:
    save_stack(path, array, interval)
    return path


def _save_final_labels(path: Path, array: np.ndarray) -> Path:
    """Use the accepted Issue 032 review-candidate TIFF encoding exactly."""
    tifffile.imwrite(
        path, np.asarray(array), imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})
    return path


def _save_final_plain_labels(path: Path, array: np.ndarray) -> Path:
    """Use the accepted Issue 045 A002 TIFF encoding exactly."""
    tifffile.imwrite(path, np.asarray(array), compression="zlib")
    return path


def _save_final_unclaimed(path: Path, array: np.ndarray) -> Path:
    """Use the accepted R10 unclaimed-ledger TIFF encoding exactly."""
    tifffile.imwrite(path, np.asarray(array), imagej=True, compression="zlib",
                     metadata={"axes": "TYX", "unit": "pixel"})
    return path


def _stage(root: Path, number: int, name: str) -> StageDirs:
    return StageDirs(root / f"{number:02d}_{name}")


def _cropped_frame_components(
        frame: np.ndarray, frame_number: int, minimum: int,
        component_type,
        ) -> list:
    """Reproduce Issue 009 components without per-identity full-frame labels."""
    result = []
    width = int(frame.shape[1])
    for identity, slices in enumerate(ndi.find_objects(frame), start=1):
        if slices is None:
            continue
        y0, x0 = int(slices[0].start), int(slices[1].start)
        local = frame[slices] == identity
        parts, count = ndi.label(
            local, structure=np.ones((3, 3), np.uint8))
        areas = np.bincount(parts.ravel())
        for number in range(1, count + 1):
            if int(areas[number]) < minimum:
                continue
            yy, xx = np.nonzero(parts == number)
            yy = yy + y0
            xx = xx + x0
            pixels = yy.astype(np.int64) * width + xx
            result.append(component_type(
                frame_number, identity, number, pixels,
                int(areas[number]), float(xx.mean()), float(yy.mean())))
    return result


def _install_issue009_component_extractor() -> None:
    """Replace only the loaded production copy of Issue 009 geometry."""
    geometry = sys.modules.get("m1_lineage_hierarchy")
    if geometry is None:
        raise RuntimeError("Issue 009 geometry module was not loaded")
    component_type = geometry.Component
    geometry.frame_components = lambda frame, frame_number, minimum: (
        _cropped_frame_components(
            frame, frame_number, minimum, component_type))


def _field_motion_reservations(run_name: str) -> set[int]:
    audit = ROOT / "m12_identity_reservation" / run_name / \
        "out/motion_reservation_audit.csv"
    if audit.is_file():
        table = read_csv_or_empty(audit, ("decision", "identity"))
        if len(table):
            return set(table.loc[
                table.decision.astype(str).eq("accepted"),
                "identity"].astype(int))
    seeds = ROOT / "m12_identity_reservation" / run_name / \
        "out/first_handoff_identity_seeds.csv"
    if seeds.is_file():
        table = read_csv_or_empty(seeds, ("persistent_identity",))
        if len(table):
            return set(table.persistent_identity.astype(int))
    return set()


def load_history_inputs_from_run(
        stem: str, run_name: str) -> AcceptedHistoryInputs:
    """Load immutable audit artifacts only for an explicit resume operation."""
    physical = load_stack(
        ROOT / "m19_layered_lineage" / run_name / "mid" /
        f"{stem}_physical_somas.tif")
    reservation = load_stack(
        ROOT / "m12_identity_reservation" / run_name / "out" / f"{stem}.tif")
    pre_path = (ROOT / "m12_identity_reservation" / run_name / "mid" /
                f"{stem}_pre_reservation_long_memory.tif")
    if not pre_path.is_file():
        raise FileNotFoundError(
            "explicit accepted-history resume requires current-run "
            f"pre-reservation evidence: {pre_path}")
    host_audit = read_csv_or_empty(
        ROOT / "m12_identity_reservation" / run_name /
        "out/reserved_identity_host_carry.csv",
        ("identity", "host_identity"))
    return AcceptedHistoryInputs(
        physical_somas=physical,
        reservation_labels=reservation,
        pre_reservation_labels=load_stack(pre_path),
        host_audit=host_audit,
        motion_reserved=_field_motion_reservations(run_name),
    )


def align_transition_stack(
        transitions: np.ndarray, label_frames: int,
        source_frame_offset: int) -> np.ndarray:
    """Align full-source transitions to a trimmed label movie."""
    expected = max(int(label_frames) - 1, 0)
    if len(transitions) == expected:
        return transitions
    start = int(source_frame_offset)
    aligned = transitions[start:start + expected]
    if len(aligned) != expected:
        raise ValueError(
            f"cannot align {len(transitions)} transitions to "
            f"{label_frames} label frames at offset {start}")
    return aligned


def _issue005(
        config: Config, physical: np.ndarray, reservation: np.ndarray,
        raw: np.ndarray, lag: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, anchor_t: int,
        motion_reserved: set[int]) -> np.ndarray:
    accepted = config.values["accepted_postprocessing"]
    layered = accepted["layered_lineage"]
    oscillatory = accepted["oscillatory_lineage"]
    comparison_params = _params("issue005", "comparison")
    stable_params = _params("issue005", "stable")
    final_params = _params("issue005", "final")
    soma_params = GlobalSomaLedgerParams(**layered["soma_ledger"])
    prepared_tracking = prepare_tracking_observations(
        physical, raw)

    comparison_settings = dict(layered["comparison_tracking"])
    comparison_settings.update(comparison_params["occupancy_tracking"])
    comparison_settings["motion_reservation_identities"] = sorted(
        motion_reserved)
    comparison, _, _, _ = track_physical_somas(
        physical, reservation, raw, lag, anchor_t, comparison_settings,
        observations, observation_table, soma_params, prepared_tracking)
    comparison, _ = reuse_retired_identity_names(
        comparison, reservation,
        int(layered["minimum_retired_name_overlap_px"]))
    confidence_layers = multilayer_identity_confidence_table(
        comparison, raw, comparison_params["classification"])
    layer_map = {int(row.identity): str(row.confidence_layer)
                 for row in confidence_layers.itertuples()}

    stable_settings = dict(layered["tracking"])
    stable_settings.update(stable_params["occupancy_tracking"])
    stable_settings.update(stable_params["reversal_aware_motion"])
    stable_settings["high_confidence_identity_ids"] = sorted(
        identity for identity, layer in layer_map.items()
        if layer != "small_or_dim")
    stable_settings.update({
        "confidence_layer_by_identity": layer_map,
        "confidence_layer_order": stable_params["order"],
        "confidence_default_layer": "small_or_dim",
        "small_confidence_layer": "small_or_dim",
    })
    stable, _, _, _ = track_physical_somas(
        physical, comparison, raw, lag, anchor_t, stable_settings,
        observations, observation_table, soma_params, prepared_tracking)
    large_ids = {identity for identity, layer in layer_map.items()
                 if layer != "small_or_dim"}
    stable, _ = restore_deferred_identity_layer(
        stable, physical, comparison, large_ids)
    stable, _ = reuse_retired_identity_names(
        stable, comparison, int(layered["minimum_retired_name_overlap_px"]))

    confidence = identity_confidence_table(
        stable, raw, layered["confidence"])
    protected = set(confidence.loc[confidence.confident, "identity"].astype(int))
    reconcile = dict(layered["reconciliation"])
    reconcile["protected_identity_ids"] = sorted(protected)
    reconcile["unprotected_canonical_policy"] = "longer_fragment"
    candidate, _, _ = reconcile_midpoint_bookends(
        stable, raw, anchor_t, reconcile)
    candidate, _, _ = optimise_lineage_graph(
        candidate, lag, oscillatory["lineage_graph"],
        int(final_params["minimum_substantial_core_px"]))
    candidate, _, _ = optimise_host_merge_graph(
        candidate, raw, lag, oscillatory["host_merge_graph"],
        accepted["persistent_merges"])
    candidate, _, _ = optimise_territory_ownership_graph(
        candidate, anchor_t, final_params["territory"])
    candidate, _, _ = reconcile_same_territory_replacements(
        candidate, anchor_t, final_params["territory"])
    return candidate


def _issue006(labels: np.ndarray, raw: np.ndarray,
              lag: np.ndarray) -> np.ndarray:
    code = I006
    detect = _issue_module(code, "s1_detect_contacts")
    apply = _issue_module(code, "s2_apply_tenure")
    detection_params = _params("issue006", "detection")
    params = _params("issue006", "tenure")
    events = detect.detect_contacts(labels, detection_params["detection"])
    candidate, _ = apply.apply_tenure(
        labels, raw, lag, events, params["tenure"], enabled=True)
    return candidate


def _issue007(labels: np.ndarray, pre: np.ndarray, lag: np.ndarray,
              host_audit: pd.DataFrame) -> np.ndarray:
    code = I007
    detector = _issue_module(code, "m1_detect_hypotheses")
    reconciler = _issue_module(code, "m2_reconcile")
    params = _params("issue007", "reconcile")
    events, candidates = detector.detect(
        labels, pre, lag, host_audit, params["detection"])
    candidate, _, _ = reconciler.reconcile(
        labels, pre, lag, host_audit, events, candidates,
        set(params["methods"]), params["detection"], params["graph"])
    return candidate


def _issue008(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
              stem: str, root: Path) -> np.ndarray:
    code = I008
    trim = _stage(root, 7, "trimmed_evidence")
    _save(trim.out / f"{stem}.tif", labels, 30.0)
    _save(trim.out / f"{stem}_registered_raw.tif", raw, 30.0)
    _save(trim.out / f"{stem}_lag01_float.tif", lag, 30.0)

    quarantine = _stage(root, 8, "global_quarantine")
    module = _issue_module(code, "m7_global_quarantine")
    params = _params("issue008", "global_quarantine")
    params.update({"stem": stem, "source_frame_offset": 2})
    module.run(trim.out, params, quarantine)

    bridge = _stage(root, 9, "general_merge_bridge")
    module = _issue_module(code, "m8_general_merge_bridge")
    params = _params("issue008", "general_merge_bridge")
    params.update({"stem": stem, "source_frame_offset": 2,
                   "trim_dir": str(trim.out)})
    module.run(quarantine.out, params, bridge)

    duplicate = _issue_module(code, "m9_duplicate_cleanup")
    params = _params("issue008", "duplicate_cleanup")
    candidate, _ = duplicate.clean_duplicates(
        load_stack(bridge.out / f"{stem}.tif"), params["duplicates"], 2)

    quarantine2 = _issue_module(code, "m13_tenure_quarantine")
    params = _params("issue008", "tenure_quarantine")
    candidate, _, _ = quarantine2.quarantine_labels(
        candidate, lag, params, {"B"})

    persistent = _issue_module(code, "m16_persistent_quarantine")
    params = _params("issue008", "persistent_quarantine")
    candidate, _, _ = persistent.persistent_quarantine(
        candidate, lag, params["persistent"], params["activation"])

    params = _params("issue008", "conservative_duplicate_cleanup")
    candidate, _ = duplicate.clean_duplicates(
        candidate, params["duplicates"], 2)
    return candidate


def _issue009(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
              stem: str, root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    code = I009
    inputs = _stage(root, 10, "hierarchy_inputs")
    label_path = _save(inputs.out / f"{stem}.tif", labels, 30.0)
    trim_dir = root / "07_trimmed_evidence/out"

    ordered = _stage(root, 11, "ordered_quarantine")
    module = _issue_module(code, "m2_ordered_quarantine")
    _install_issue009_component_extractor()
    params = _params("issue009", "ordered_quarantine")
    params.update({"stem": stem, "baseline_labels": str(label_path),
                   "trim_dir": str(trim_dir), "source_frame_offset": 2})
    module.run(None, params, ordered)
    ordered_labels = load_stack(ordered.out / f"{stem}.tif")

    authority_module = _issue_module(code, "m5_movie_authority")
    _install_issue009_component_extractor()
    accepted_params = _params("issue009", "hierarchy_reconcile")
    authority, _ = authority_module.movie_authority(
        ordered_labels, accepted_params["authority"])
    authority_dir = _stage(root, 12, "movie_authority")
    authority.to_csv(authority_dir.out / "movie_wide_authority.csv", index=False)

    reconciled = _stage(root, 13, "hierarchy_reconcile")
    module = _issue_module(code, "m6_hierarchy_reconcile")
    _install_issue009_component_extractor()
    accepted_params.update({
        "stem": stem, "parent_labels": str(ordered.out / f"{stem}.tif"),
        "trim_dir": str(trim_dir), "source_frame_offset": 2})
    module.run(authority_dir.out, accepted_params, reconciled)
    return load_stack(reconciled.out / f"{stem}.tif"), authority


def _issue010(labels: np.ndarray, authority: pd.DataFrame,
              lag: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = I010
    profiler = _issue_module(code, "m1_profile_persistence")
    dropper = _issue_module(code, "m2_drop_short_lived")
    reclaimer = _issue_module(code, "m3_reclaim_gaps")
    profile_params = _params("issue010", "persistence_profile")
    profiles, _ = profiler.profile(labels, authority, profile_params)
    drop_params = _params("issue010", "drop_short_lived")
    dropped = dropper.choose_dropped(
        profiles, int(drop_params["maximum_short_frames"]),
        bool(drop_params["preserve_edge_censored"]))
    ownership, unclaimed, foreground = dropper.drop(labels, dropped)
    reclaim_params = _params("issue010", "reclaim_gaps")
    candidate, candidate_unclaimed, _ = reclaimer.reclaim(
        labels, ownership, unclaimed, profiles, lag,
        reclaim_params)
    return candidate, candidate_unclaimed, foreground


def _issue011(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
              stem: str, root: Path) -> tuple[np.ndarray, np.ndarray]:
    code = I011
    profile = _stage(root, 15, "merge_split_profile")
    profiler = _issue_module(code, "m1_profile_merge_splits")
    profile_params = _params("issue011", "merge_split_profile")
    features = profiler.extract_features(labels, raw, 2)
    settings = {**profile_params["profile"], "source_frame_offset": 2}
    profiles, soft_events = profiler.bracketed_soft_gaps(
        features, len(labels), settings)
    contact_events = profiler.contact_separation_events(
        labels, features, profiles, settings)
    events = pd.concat([soft_events, contact_events], ignore_index=True)
    features.to_csv(profile.out / "frame_features.csv", index=False)
    profiles.to_csv(profile.out / "identity_profiles.csv", index=False)
    events.to_csv(profile.out / "merge_split_events.csv", index=False)

    reconciler = _issue_module(code, "m2_reconcile_merge_splits")
    params = _params("issue011", "merge_split_reconcile")
    methods = set(params["methods"])
    persistent = set(profiles.loc[
        profiles.persistent.astype(bool), "identity"].astype(int))
    candidates: list[dict] = []
    for _, event in events.iterrows():
        companions = [int(value) for value in
                      str(event.candidate_identities).split("|")
                      if value and value != "nan"]
        for companion in companions:
            if companion in persistent:
                item = reconciler.pair_swap_candidate(
                    labels, features, event, companion, methods,
                    params["profile"], params["reconcile"])
            else:
                item = reconciler.relay_candidate(
                    features, profiles, event, companion, methods,
                    params["profile"], params["reconcile"])
            if item is not None:
                item["methods"] = "+".join(sorted(methods))
                candidates.append(item)
    candidate_table = pd.DataFrame(candidates)
    actions = reconciler.choose_actions(
        candidate_table, float(params["reconcile"]["minimum_improvement"]))
    candidate, candidate_unclaimed = reconciler.apply_actions(
        labels, unclaimed, actions)
    reconcile = _stage(root, 16, "merge_split_reconcile")
    candidate_table.to_csv(reconcile.out / "candidate_actions.csv", index=False)
    actions.to_csv(reconcile.out / "merge_split_actions.csv", index=False)
    return candidate, candidate_unclaimed


def _issue012(labels: np.ndarray, unclaimed: np.ndarray,
              protected_parent: np.ndarray, raw: np.ndarray,
              lag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    code = I012
    ownership = _issue_module(code, "m1_reconcile_domain_ownership")
    params = _params("issue012", "domain_ownership")
    candidate, candidate_unclaimed, actions, _ = ownership.reconcile(
        labels, unclaimed, lag, set(params["methods"]), params["ownership"], 2,
        labels != protected_parent)

    detector = _issue_module(code, "m8_detect_dormant_releases")
    params8 = _params("issue012", "dormant_detection")
    events, candidates = detector.detect(
        candidate, raw, lag, actions, 2, params8["detection"])
    reconciler = _issue_module(code, "m9_reconcile_dormant_releases")
    params9 = _params("issue012", "dormant_reconcile")
    # The accepted reconciler ranks the full candidate table internally.
    selected = reconciler.select_events(
        events, candidates, params9["reconcile"])
    candidate, candidate_unclaimed, _ = reconciler.reconcile(
        candidate, candidate_unclaimed, raw, selected, 2,
        params9["reconcile"])
    return candidate, candidate_unclaimed


def _seat_identity(labels: np.ndarray, unclaimed: np.ndarray,
                   raw: np.ndarray, stem: str,
                   root: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    code = I013
    inputs = _stage(root, 17, "seat_inputs")
    label_path = _save(inputs.out / f"{stem}.tif", labels, 30.0)
    unclaimed_path = _save(
        inputs.out / f"{stem}_unclaimed_original_ids.tif", unclaimed, 30.0)
    raw_path = _save(inputs.out / f"{stem}_registered_raw.tif", raw, 30.0)
    output = _stage(root, 18, "seat_identity")
    module = _issue_module(code, "m1_seat_identity")
    params = _params("issue013", "seat_identity")
    params.update({"stem": stem, "parent_labels": str(label_path),
                   "parent_unclaimed": str(unclaimed_path),
                   "registered_raw": str(raw_path),
                   "source_frame_offset": 2})
    module.run(None, params, output)
    return (load_stack(output.out / f"{stem}.tif"),
            load_stack(output.out / f"{stem}_unclaimed_original_ids.tif"),
            pd.read_csv(output.out / "residency_registry.csv"))


def _general_blob_residency(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        parent_registry: pd.DataFrame) -> tuple[
            np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Discover residency events from first established donor collapse."""
    module = _issue_module(I015, "m5_blob_residency")
    ledger, _, _ = detect_identity_ledger_transfers(labels, raw)
    selected = select_blob_residency_events(ledger)
    settings = _params("issue015", "blob_residency")
    result = labels.copy()
    result_unclaimed = unclaimed.copy()
    ledger_rows: list[dict] = []
    action_rows: list[dict] = []
    manual_rows: list[dict] = []
    foreground = (labels > 0) | (unclaimed > 0)
    for row in selected.itertuples(index=False):
        event = {
            "case_id": str(row.event_id),
            "donor_identity": int(row.donor_identity),
            "host_identity": int(row.host_identity),
            "source_frame": int(row.imagej_frame),
            "residency_activation_source_frame": int(row.imagej_frame) + 2,
        }
        eligible, reason = blob_residency_precondition(
            labels, foreground, int(row.donor_identity),
            int(row.imagej_frame), 2)
        if not eligible:
            manual_rows.append({
                "event_id": str(row.event_id),
                "donor_identity": int(row.donor_identity),
                "host_identity": int(row.host_identity),
                "source_imagej_frame": int(row.imagej_frame),
                "reason": reason,
                "disposition": "manual_review_no_automatic_change",
            })
            continue
        candidate = result.copy()
        candidate_unclaimed = result_unclaimed.copy()
        try:
            rows, actions = module.process_event(
                candidate, candidate_unclaimed, labels, foreground,
                event, 2, settings)
        except ValueError as error:
            manual_rows.append({
                "event_id": str(row.event_id),
                "donor_identity": int(row.donor_identity),
                "host_identity": int(row.host_identity),
                "source_imagej_frame": int(row.imagej_frame),
                "reason": str(error),
                "disposition": "manual_review_no_automatic_change",
            })
            continue
        result, result_unclaimed = candidate, candidate_unclaimed
        ledger_rows.extend(rows)
        action_rows.extend(actions)
    ledger_table = pd.DataFrame(ledger_rows)
    registry = module.merge_residency_registry(parent_registry, ledger_table)
    manual_table = pd.DataFrame(manual_rows, columns=[
        "event_id", "donor_identity", "host_identity",
        "source_imagej_frame", "reason", "disposition"])
    return result, result_unclaimed, registry, ledger_table, manual_table


def blob_residency_precondition(
        labels: np.ndarray, foreground: np.ndarray, donor_identity: int,
        source_imagej_frame: int, source_frame_offset: int = 2,
        ) -> tuple[bool, str]:
    """Require a field-derived co-resident predecessor blob before correction."""
    start = int(source_imagej_frame) - int(source_frame_offset) - 1
    prior = start - 1
    if prior < 0 or prior >= len(labels):
        return False, "no_predecessor_frame"
    components, count = ndi.label(
        foreground[prior], structure=np.ones((3, 3), np.uint8))
    values, counts = np.unique(
        components[labels[prior] == int(donor_identity)], return_counts=True)
    choices = [(int(n), int(value)) for value, n in zip(values, counts)
               if int(value) > 0]
    if not choices:
        return False, "no_predecessor_foreground_blob"
    component = max(choices)[1]
    if component > count:
        return False, "no_predecessor_foreground_blob"
    members = set(map(int, np.unique(labels[prior][components == component])))
    members.discard(0)
    members.discard(int(donor_identity))
    if not members:
        return False, "no_co_resident_predecessor_label"
    return True, "eligible"


def select_blob_residency_events(ledger: pd.DataFrame) -> pd.DataFrame:
    """Select first established donor collapses without identity allowlists."""
    if ledger.empty:
        return ledger.copy()
    selected = ledger[
        (ledger.donor_core_ancestry.astype(float) >= 0.80)
        & (ledger.host_core_ancestry.astype(float) >= 0.80)
        & (ledger.direct_transfer_fraction.astype(float) >= 0.25)
        & (ledger.consecutive_prior_presence_frames.astype(int) >= 7)
        & ~ledger.donor_still_present.astype(bool)
    ].sort_values(["imagej_frame", "event_id"])
    # Only the first established collapse of a donor can establish residency.
    return selected.drop_duplicates("donor_identity", keep="first")


def derive_long_merge_protected_identities(
        evidence: pd.DataFrame, maximum_bookend_frames: int) -> set[int]:
    """Protect identities participating in field-observed long merge gaps."""
    long_merge = evidence[
        (evidence.mechanism.astype(str) == "same_host_merge_hiding")
        & (evidence.missing_frames.astype(int)
           > int(maximum_bookend_frames))]
    protected = set(long_merge.identity.astype(int))
    for row in long_merge.itertuples(index=False):
        protected.update(
            value for value in (int(row.next_owner), int(row.prior_owner))
            if value > 0)
    return protected


def _continuity_repair(
        baseline: np.ndarray, baseline_unclaimed: np.ndarray,
        raw_full: np.ndarray, lag_full: np.ndarray,
        registry: pd.DataFrame, protected_alias_ids: set[int],
        root: Path) -> tuple[np.ndarray, np.ndarray]:
    code = I017
    census_module = _issue_module(code, "m1_census")
    evidence_module = _issue_module(code, "m2_classify_evidence")
    repair = _issue_module(code, "m3_repair")
    census_dir = _stage(root, 19, "continuity_census")
    measured = census_module.measure(
        baseline, frame_offset=2, minutes_per_frame=30.0,
        compatibility_ids=protected_alias_ids)
    census_module._write_csv(census_dir.out / "gap_runs.csv", measured["gaps"])
    census_module._write_csv(
        census_dir.out / "termination_audit.csv", measured["terminations"])

    evidence_inputs = _stage(root, 20, "continuity_evidence_inputs")
    labels_path = _save(evidence_inputs.out / "labels.tif", baseline, 30.0)
    unclaimed_path = _save(
        evidence_inputs.out / "unclaimed.tif", baseline_unclaimed, 30.0)
    raw_path = _save(evidence_inputs.out / "raw.tif", raw_full, 30.0)
    lag_path = _save(evidence_inputs.out / "lag.tif", lag_full, 30.0)
    registry_path = evidence_inputs.out / "residency_registry.csv"
    registry.to_csv(registry_path, index=False)
    evidence_dir = _stage(root, 21, "continuity_evidence")
    params = _params("issue017", "continuity_evidence")
    params.update({
        "labels_path": str(labels_path), "unclaimed_path": str(unclaimed_path),
        "residency_registry_path": str(registry_path),
        "raw_path": str(raw_path), "lag_path": str(lag_path),
        "source_frame_offset": 2, "compatibility_ids": [],
    })
    evidence_module.run(census_dir.out, params, evidence_dir)
    evidence = read_csv_or_empty(
        evidence_dir.out / "gap_evidence.csv",
        ("mechanism", "identity", "missing_frames", "next_owner",
         "prior_owner", "gap_start_source_frame", "gap_end_source_frame",
         "post_source_frame"))
    terminations = read_csv_or_empty(
        evidence_dir.out / "termination_evidence.csv",
        ("mechanism", "identity", "last_source_frame"))
    raw = raw_full[2:2 + len(baseline)]
    lag = lag_full[2:2 + len(baseline) - 1]

    labels, _, _ = repair._bracketed_recovery(
        baseline, raw, lag, evidence, 2, protected_alias_ids)
    labels = repair._restore_locked(labels, baseline, protected_alias_ids)
    labels, _ = repair._account_foreground(
        labels, baseline, baseline_unclaimed)

    merge_params = _params(
        "issue017", "continuity_repair")["merge_params"]
    protected_merge_ids = derive_long_merge_protected_identities(
        evidence, int(merge_params["maximum_bookend_frames"]))
    labels, _ = repair._merge_continuation(
        labels, raw, lag, evidence, 2, merge_params, protected_alias_ids,
        protected_merge_ids)
    labels = repair._restore_locked(labels, baseline, protected_alias_ids)
    labels, _ = repair._account_foreground(
        labels, baseline, baseline_unclaimed)

    before_ending = labels.copy()
    labels, _, _ = repair._ending_search(
        labels, raw, lag, terminations, 2, protected_alias_ids)
    labels = repair._restore_locked(labels, baseline, protected_alias_ids)
    labels, _ = repair._account_foreground(
        labels, baseline, baseline_unclaimed)
    labels, _, _ = repair._truncate_to_contiguous_end_path(
        before_ending, labels, terminations, 2, protected_alias_ids)
    labels = repair._restore_locked(labels, baseline, protected_alias_ids)
    return repair._account_foreground(
        labels, baseline, baseline_unclaimed)


def _issue023_tail(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        lag_trim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = reconcile_territory_memory(
        labels, unclaimed, raw,
        _general("R06_generalise_territory_memory"))
    labels, unclaimed = result.labels, result.unclaimed
    result = assign_persistent_unclaimed_bodies(
        labels, unclaimed, raw,
        _general("R07_generalise_unclaimed_body_identity"))
    labels, unclaimed = result.labels, result.unclaimed
    result = reserve_local_lineages(
        labels, unclaimed, lag_trim,
        _general("R08_generalise_local_lineage_reservation"))
    labels, unclaimed = result.labels, result.unclaimed
    result = reconcile_reciprocal_lineages(
        labels, unclaimed, lag_trim,
        _general("R09_generalise_reciprocal_lineages"))
    labels, unclaimed = result.labels, result.unclaimed
    result = recover_stationary_gaps(
        labels, unclaimed, raw,
        _general("R10_generalise_stationary_claims"))
    return result.labels, result.unclaimed


def _issue027_short_blank_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        lag: np.ndarray, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Recover short raw-positive internal gaps without supplied cell targets."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_short_blank_recovery", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    forbidden = {
        "forced_identity_ids": params.get("forced_identity_ids", []),
        "forced_intervals": params.get("forced_intervals", []),
    }
    supplied = {name: value for name, value in forbidden.items() if value}
    if supplied:
        raise ValueError(
            "field-wide short-blank recovery cannot receive targets: "
            + ", ".join(sorted(supplied)))
    params["forced_identity_ids"] = []
    params["forced_intervals"] = []

    if enabled:
        candidate, recovered, audit = complete_presence_bidirectional(
            labels, raw, lag, params)
    else:
        candidate = labels.copy()
        recovered = np.zeros(labels.shape, bool)
        audit = pd.DataFrame()
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError(
            "short-blank recovery changed pre-existing assigned pixels")
    if np.any(recovered & (raw == 0)):
        raise AssertionError("short-blank recovery added zero-signal pixels")
    candidate_unclaimed = np.where(candidate == 0, unclaimed, 0).astype(
        unclaimed.dtype)
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError(
            "short-blank recovery overlaps assigned and unclaimed ledgers")

    stage = output_root / "22_short_blank_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit_path = stage / "producer_audit.csv"
    audit.to_csv(audit_path, index=False)
    outcomes = ({str(name): int(count)
                 for name, count in audit["outcome"].value_counts().items()}
                if len(audit) and "outcome" in audit else {})
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "forced_identity_ids": [],
        "forced_intervals": [],
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "recovered_pixels": int(np.count_nonzero(recovered)),
        "zero_signal_additions": int(np.count_nonzero(recovered & (raw == 0))),
        "audit_rows": int(len(audit)),
        "outcomes": outcomes,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue029_small_cell_blank_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        lag: np.ndarray, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Extend Issue 027 only to its field-derived small-cell area band."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_small_cell_blank_recovery", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    forbidden = {
        "forced_identity_ids": params.get("forced_identity_ids", []),
        "forced_intervals": params.get("forced_intervals", []),
    }
    supplied = {name: value for name, value in forbidden.items() if value}
    if supplied:
        raise ValueError(
            "field-wide small-cell recovery cannot receive targets: "
            + ", ".join(sorted(supplied)))
    params["forced_identity_ids"] = []
    params["forced_intervals"] = []
    if bool(params.get("carve_hosts", False)):
        raise ValueError("accepted Issue 029 small-cell recovery cannot carve hosts")

    if enabled:
        candidate, recovered, audit = complete_presence_bidirectional(
            labels, raw, lag, params)
    else:
        candidate = labels.copy()
        recovered = np.zeros(labels.shape, bool)
        audit = pd.DataFrame()
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError(
            "small-cell recovery changed pre-existing assigned pixels")
    if np.any(recovered & (raw == 0)):
        raise AssertionError("small-cell recovery added zero-signal pixels")
    candidate_unclaimed = np.where(candidate == 0, unclaimed, 0).astype(
        unclaimed.dtype)
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError(
            "small-cell recovery overlaps assigned and unclaimed ledgers")

    stage = output_root / "23_small_cell_blank_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage / "producer_audit.csv", index=False)
    outcomes = ({str(name): int(count)
                 for name, count in audit["outcome"].value_counts().items()}
                if len(audit) and "outcome" in audit else {})
    resolved_params = {**PRESENCE_DEFAULTS, **params}
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "forced_identity_ids": [],
        "forced_intervals": [],
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "recovered_pixels": int(np.count_nonzero(recovered)),
        "zero_signal_additions": int(np.count_nonzero(recovered & (raw == 0))),
        "resolved_minimum_persistent_area_px": float(
            resolve_minimum_persistent_area(labels, resolved_params)),
        "audit_rows": int(len(audit)),
        "outcomes": outcomes,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue030_moderate_gap_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        lag: np.ndarray, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Extend raw-supported recovery only to the accepted duration band."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_moderate_gap_recovery", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    forbidden = {
        "forced_identity_ids": params.get("forced_identity_ids", []),
        "forced_intervals": params.get("forced_intervals", []),
    }
    supplied = {name: value for name, value in forbidden.items() if value}
    if supplied:
        raise ValueError(
            "field-wide moderate-gap recovery cannot receive targets: "
            + ", ".join(sorted(supplied)))
    params["forced_identity_ids"] = []
    params["forced_intervals"] = []
    if bool(params.get("carve_hosts", False)):
        raise ValueError("accepted Issue 030 moderate-gap recovery cannot carve hosts")

    if enabled:
        candidate, recovered, audit = complete_presence_bidirectional(
            labels, raw, lag, params)
    else:
        candidate = labels.copy()
        recovered = np.zeros(labels.shape, bool)
        audit = pd.DataFrame()
    if np.any((labels > 0) & (candidate != labels)):
        raise AssertionError(
            "moderate-gap recovery changed pre-existing assigned pixels")
    if np.any(recovered & (raw == 0)):
        raise AssertionError("moderate-gap recovery added zero-signal pixels")
    candidate_unclaimed = np.where(candidate == 0, unclaimed, 0).astype(
        unclaimed.dtype)
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError(
            "moderate-gap recovery overlaps assigned and unclaimed ledgers")

    stage = output_root / "24_moderate_gap_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    audit.to_csv(stage / "producer_audit.csv", index=False)
    outcomes = ({str(name): int(count)
                 for name, count in audit["outcome"].value_counts().items()}
                if len(audit) and "outcome" in audit else {})
    resolved_params = {**PRESENCE_DEFAULTS, **params}
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "forced_identity_ids": [],
        "forced_intervals": [],
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "recovered_pixels": int(np.count_nonzero(recovered)),
        "zero_signal_additions": int(np.count_nonzero(recovered & (raw == 0))),
        "resolved_minimum_persistent_area_px": float(
            resolve_minimum_persistent_area(labels, resolved_params)),
        "audit_rows": int(len(audit)),
        "outcomes": outcomes,
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue032_owner_consensus(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        motion_full: np.ndarray, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Correct owner substitutions from target-free raw physical tracks."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_owner_consensus", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted owner consensus must be field-wide")
    raw_params = deepcopy(configured.get("raw_detection", {}))
    latent_params = {**raw_params,
                     **deepcopy(configured.get("latent_tracking", {}))}
    consensus_params = deepcopy(configured.get("owner_consensus", {}))
    forbidden_names = {
        "labels_path", "review_cases_path", "event_ids", "identity_ids",
        "frame_ids", "coordinates", "forced_identity_ids",
        "forced_intervals", "physical_track_points_path",
    }
    supplied = sorted(
        name for values in (raw_params, latent_params, consensus_params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide owner consensus received forbidden targets: " +
            ", ".join(sorted(set(supplied))))

    hypotheses_stage = output_root / "25_raw_physical_hypotheses" / "out"
    tracks_stage = output_root / "26_latent_body_tracks" / "out"
    owner_stage = output_root / "27_owner_consensus" / "out"
    for stage in (hypotheses_stage, tracks_stage, owner_stage):
        stage.mkdir(parents=True, exist_ok=True)

    if not enabled:
        audit = pd.DataFrame(columns=["proposal_id", "outcome",
                                      "changed_pixels"])
        audit.to_csv(owner_stage / "producer_audit.csv", index=False)
        metrics = {
            "version": configured.get("version", "unversioned"),
            "enabled": False, "targeting_mode": "field_wide_discovery",
            "identity_target_count": 0, "frame_target_count": 0,
            "coordinate_target_count": 0, "event_target_count": 0,
            "changed_pixels": 0, "changed_frames": 0,
            "active_identities": int(
                len(set(map(int, np.unique(labels))) - {0})),
            "new_duplicate_components": 0,
        }
        (owner_stage / "producer_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return labels.copy(), unclaimed.copy(), audit, metrics

    hypothesis_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for frame_index, frame in enumerate(raw):
        rows, threshold = raw_physical_hypotheses.detect_frame(
            frame, frame_index, raw_params)
        hypothesis_rows.extend(rows)
        threshold_rows.append(threshold)
    hypotheses = pd.DataFrame(hypothesis_rows)
    thresholds = pd.DataFrame(threshold_rows)
    hypotheses_path = hypotheses_stage / "raw_physical_hypotheses.csv"
    thresholds_path = hypotheses_stage / "frame_evidence_thresholds.csv"
    hypotheses.to_csv(hypotheses_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    raw_metrics = {
        "targeting_mode": "field_wide_discovery", "frames": int(len(raw)),
        "hypotheses": int(len(hypotheses)),
        "strong_hypotheses": int(hypotheses["strong"].sum()),
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "parameters": raw_params,
    }
    (hypotheses_stage / "producer_metrics.json").write_text(
        json.dumps(raw_metrics, indent=2) + "\n", encoding="utf-8")

    # The accepted detector stages are separately runnable. Preserve their
    # CSV boundary so downstream floating-point inputs reproduce byte-exactly.
    hypotheses = pd.read_csv(hypotheses_path)
    thresholds = pd.read_csv(thresholds_path)
    motion_offset = int(configured.get("motion_frame_offset", 2))
    motion = motion_full[motion_offset:motion_offset + len(raw)]
    tracks = latent_body_tracks.link_observations(
        hypotheses, raw, motion, latent_params)
    points, track_summaries = latent_body_tracks.expand_track_points(
        tracks, raw, thresholds, latent_params)
    encounters, encounter_frames = latent_body_tracks.encounter_analysis(
        points, raw,
        float(latent_params.get("maximum_separable_valley_ratio", 0.65)))
    reconnections = latent_body_tracks.reconnection_hypotheses(
        points, raw, latent_params)
    points_path = tracks_stage / "latent_track_points.csv"
    points.to_csv(points_path, index=False)
    track_summaries.to_csv(tracks_stage / "physical_tracks.csv", index=False)
    encounters.to_csv(tracks_stage / "encounter_episodes.csv", index=False)
    encounter_frames.to_csv(
        tracks_stage / "encounter_frames.csv", index=False)
    reconnections.to_csv(
        tracks_stage / "reconnection_hypotheses.csv", index=False)
    track_metrics = {
        "targeting_mode": "field_wide_discovery",
        "physical_tracks": int(len(track_summaries)),
        "track_points": int(len(points)),
        "observed_points": int((points["state"] == "observed").sum()),
        "latent_visible_points": int(
            (points["state"] == "latent_visible").sum()),
        "dormant_points": int((points["state"] == "dormant").sum()),
        "encounter_episodes": int(len(encounters)),
        "reconnection_hypotheses": int(len(reconnections)),
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "parameters": latent_params,
    }
    (tracks_stage / "producer_metrics.json").write_text(
        json.dumps(track_metrics, indent=2) + "\n", encoding="utf-8")

    # The accepted issue workflow passed this table through CSV. Re-reading it
    # preserves the reviewed floating-point representation exactly.
    production_points = pd.read_csv(points_path)
    attached = owner_consensus.attach_owners(production_points, labels)
    proposals = owner_consensus.discover_proposals(
        attached, consensus_params)
    candidate, audit = owner_consensus.apply_dependency_ordered_proposals(
        labels, attached, proposals, consensus_params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("owner consensus changed foreground segmentation")
    if not np.array_equal(unclaimed, unclaimed.copy()):
        raise AssertionError("owner consensus changed the unclaimed ledger")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError("owner consensus changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"owner consensus created {duplicates} duplicate components")
    attached.to_csv(owner_stage / "physical_owner_points.csv", index=False)
    audit.to_csv(owner_stage / "producer_audit.csv", index=False)
    applied = audit[audit["outcome"] == "applied"]
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": True, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "event_target_count": 0,
        "raw_hypotheses": int(len(hypotheses)),
        "physical_tracks": int(len(track_summaries)),
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "applied_orphan_absorptions": int(
            applied["proposal_id"].astype(str).str.startswith("O").sum()),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": 0,
        "unclaimed_changed_pixels": 0,
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "raw_detection": raw_params,
        "latent_tracking": latent_params,
        "owner_consensus": consensus_params,
    }
    (owner_stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, metrics


def _issue033_separable_merge_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Restore owners after field-discovered, raw-separable encounters."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_separable_merge_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted separable-merge recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    forbidden_names = {
        "labels_path", "review_cases_path", "event_ids", "event_targets",
        "identity_ids", "identity_targets", "frame_ids", "frame_targets",
        "coordinates", "coordinate_targets", "forced_identity_ids",
        "forced_intervals", "physical_track_points_path",
        "encounter_frames_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide separable-merge recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "28_separable_merge_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "accepted separable-merge recovery requires current-run "
                f"physical tracks: {points_path}")
        # Preserve the accepted separately-runnable CSV boundary exactly.
        points = pd.read_csv(points_path)
        candidate, audit = \
            separable_merge_recovery.recover_post_encounter_owners(
                labels, raw, points, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "proposal_type", "track_id", "applied", "changed_pixels",
            "reason"])

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "separable-merge recovery changed foreground segmentation")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "separable-merge recovery changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"separable-merge recovery created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "event_target_count": 0,
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": 0,
        "unclaimed_changed_pixels": 0,
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, metrics


def _issue034_recording_start_split(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Restore field-discovered recording-start residents after later splits."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_recording_start_split", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted recording-start split must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    forbidden_names = {
        "labels_path", "raw_path", "review_cases_path", "case_ids",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
        "forced_identity_ids", "forced_intervals",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide recording-start split received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "29_recording_start_split" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, audit = \
            recording_start_split.recover_recording_start_residents(
                labels, raw, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "newcomer_identity", "incumbent_identity", "applied",
            "changed_pixels", "changed_frames", "reason"])

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "recording-start split changed foreground segmentation")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "recording-start split changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"recording-start split created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "event_target_count": 0,
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": 0,
        "unclaimed_changed_pixels": 0,
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, metrics


def _issue035_merge_residency(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Restore stable residents displaced during field-discovered merges."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_merge_residency", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted merge residency must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    forbidden_names = {
        "labels_path", "raw_path", "physical_track_points_path",
        "review_cases_path", "case_ids", "track_ids", "track_targets",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
        "forced_identity_ids", "forced_intervals",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide merge residency received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "30_merge_residency" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "merge residency requires current-run physical track points: "
                f"{points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, candidate_unclaimed, audit = \
            merge_residency.recover_merge_residents(
                labels, unclaimed, raw, physical_points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=[
            "resident_track", "resident_owner", "host_owner",
            "first_frame", "last_frame", "applied", "changed_pixels",
            "released_pixels", "changed_frames", "reason"])

    baseline_foreground = (labels > 0) | (unclaimed > 0)
    candidate_foreground = ((candidate > 0) |
                            (candidate_unclaimed > 0))
    foreground_changes = int(np.count_nonzero(
        baseline_foreground != candidate_foreground))
    if foreground_changes:
        raise AssertionError(
            "merge residency changed combined foreground segmentation")
    overlap = int(np.count_nonzero(
        (candidate > 0) & (candidate_unclaimed > 0)))
    if overlap:
        raise AssertionError(
            f"merge residency left {overlap} assigned/unclaimed overlaps")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "merge residency changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"merge residency created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "frame_target_count": 0,
        "coordinate_target_count": 0, "event_target_count": 0,
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "changed_frames": int(np.count_nonzero(np.any(
            (candidate != labels) | (candidate_unclaimed != unclaimed),
            axis=(1, 2)))),
        "combined_foreground_changed_pixels": foreground_changes,
        "assigned_unclaimed_overlap_pixels": overlap,
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue036_persistent_body_identity(
        labels: np.ndarray, unclaimed: np.ndarray, config: Config,
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Allocate identities to persistent field-discovered released bodies."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_persistent_body_identity", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted persistent body identity must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    forbidden_names = {
        "labels_path", "raw_path", "physical_track_points_path",
        "review_cases_path", "case_ids", "track_ids", "track_targets",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
        "forced_identity_ids", "forced_intervals",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide persistent body identity received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "31_persistent_body_identity" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "persistent body identity requires current-run physical tracks: "
                f"{points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, candidate_unclaimed, audit = \
            persistent_body_identity.allocate_persistent_released_lineages(
                labels, unclaimed, physical_points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = pd.DataFrame(columns=[
            "physical_track", "seed_owner", "seed_first_frame",
            "seed_last_frame", "applied", "new_identity",
            "changed_label_pixels", "changed_unclaimed_pixels",
            "changed_frames", "reason"])

    baseline_foreground = (labels > 0) | (unclaimed > 0)
    candidate_foreground = ((candidate > 0) |
                            (candidate_unclaimed > 0))
    foreground_changes = int(np.count_nonzero(
        baseline_foreground != candidate_foreground))
    if foreground_changes:
        raise AssertionError(
            "persistent body identity changed combined foreground")
    overlap = int(np.count_nonzero(
        (candidate > 0) & (candidate_unclaimed > 0)))
    if overlap:
        raise AssertionError(
            f"persistent body identity left {overlap} ledger overlaps")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    if len(candidate_identities) != len(baseline_identities) + len(applied):
        raise AssertionError(
            "persistent body identity did not add one identity per transaction")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"persistent body identity created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0,
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "changed_frames": int(np.count_nonzero(np.any(
            (candidate != labels) | (candidate_unclaimed != unclaimed),
            axis=(1, 2)))),
        "combined_foreground_changed_pixels": foreground_changes,
        "assigned_unclaimed_overlap_pixels": overlap,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue037_resident_takeover(
        labels: np.ndarray, unclaimed: np.ndarray, config: Config,
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Restore separated established residents invaded by one owner."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_resident_takeover", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted resident takeover must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    forbidden_names = {
        "labels_path", "raw_path", "physical_track_points_path",
        "review_cases_path", "case_ids", "track_ids", "track_targets",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
        "forced_identity_ids", "forced_intervals",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide resident takeover received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "32_resident_takeover" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "resident takeover requires current-run physical tracks: "
                f"{points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, audit, attached = resident_takeover.recover(
            labels, physical_points, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "proposal_id", "outcome", "changed_pixels", "changed_frames"])
        attached = pd.DataFrame()
    candidate_unclaimed = unclaimed.copy()

    foreground_changes = int(np.count_nonzero(
        (candidate > 0) != (labels > 0)))
    if foreground_changes:
        raise AssertionError("resident takeover changed foreground")
    unclaimed_changes = int(np.count_nonzero(
        candidate_unclaimed != unclaimed))
    if unclaimed_changes:
        raise AssertionError("resident takeover changed unclaimed ledger")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError("resident takeover changed active identities")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"resident takeover created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    applied = (audit[audit["outcome"] == "applied"]
               if len(audit) and "outcome" in audit else audit.iloc[0:0])
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0,
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int((
            audit.get("discovery_status", pd.Series(dtype=str)) ==
            "eligible").sum()),
        "applied_proposals": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": unclaimed_changes,
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": foreground_changes,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue038_ownerless_body_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Allocate connected raw cores to persistent never-owned bodies."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_ownerless_body_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted ownerless-body recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "frame_evidence_thresholds_path",
        "review_cases_path", "case_ids", "track_ids", "track_targets",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
        "forced_identity_ids", "forced_intervals",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide ownerless-body recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "33_ownerless_body_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        thresholds_path = (
            output_root / "25_raw_physical_hypotheses" / "out" /
            "frame_evidence_thresholds.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "ownerless-body recovery requires current-run physical tracks: "
                f"{points_path}")
        if not thresholds_path.is_file():
            raise FileNotFoundError(
                "ownerless-body recovery requires current-run thresholds: "
                f"{thresholds_path}")
        physical_points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, audit, attached = ownerless_body_recovery.recover(
            labels, unclaimed, raw, physical_points, thresholds, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "proposal_id", "outcome", "changed_pixels", "changed_frames"])
        attached = pd.DataFrame()
    candidate_unclaimed = unclaimed.copy()

    existing_changes = int(np.count_nonzero(
        (labels > 0) & (candidate != labels)))
    if existing_changes:
        raise AssertionError(
            "ownerless-body recovery changed an existing label pixel")
    additions = (candidate > 0) & (labels == 0)
    overlap = int(np.count_nonzero(additions & (unclaimed > 0)))
    if overlap:
        raise AssertionError(
            f"ownerless-body recovery overlapped {overlap} unclaimed pixels")
    zero_signal = int(np.count_nonzero(additions & (raw == 0)))
    if zero_signal:
        raise AssertionError(
            f"ownerless-body recovery added {zero_signal} zero-signal pixels")
    unclaimed_changes = int(np.count_nonzero(
        candidate_unclaimed != unclaimed))
    if unclaimed_changes:
        raise AssertionError("ownerless-body recovery changed unclaimed ledger")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    applied = (audit[audit["outcome"] == "applied"]
               if len(audit) and "outcome" in audit else audit.iloc[0:0])
    if len(candidate_identities) != len(baseline_identities) + len(applied):
        raise AssertionError(
            "ownerless-body recovery did not add one identity per transaction")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"ownerless-body recovery created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0,
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int((
            audit.get("discovery_status", pd.Series(dtype=str)) ==
            "eligible").sum()),
        "applied_proposals": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": unclaimed_changes,
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_added_pixels": int(np.count_nonzero(additions)),
        "existing_label_pixels_changed": existing_changes,
        "unclaimed_overlap_pixels": overlap,
        "zero_signal_additions": zero_signal,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue039_ownerless_cohort_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Allocate one identity to each medium-lived never-owned body cohort."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_ownerless_cohort_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted ownerless-cohort recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "frame_evidence_thresholds_path",
        "review_cases_path", "case_ids", "track_ids", "track_targets",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
        "forced_identity_ids", "forced_intervals",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide ownerless-cohort recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "34_ownerless_cohort_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        thresholds_path = (
            output_root / "25_raw_physical_hypotheses" / "out" /
            "frame_evidence_thresholds.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "ownerless-cohort recovery requires current-run physical "
                f"tracks: {points_path}")
        if not thresholds_path.is_file():
            raise FileNotFoundError(
                "ownerless-cohort recovery requires current-run thresholds: "
                f"{thresholds_path}")
        physical_points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, audit, edges, applications, attached = \
            ownerless_cohort_recovery.recover(
                labels, unclaimed, raw, physical_points, thresholds, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "proposal_id", "discovery_role", "discovery_status"])
        edges = pd.DataFrame(columns=["edge_id", "edge_status"])
        applications = pd.DataFrame(columns=[
            "seat_id", "outcome", "changed_pixels", "changed_frames"])
        attached = pd.DataFrame()
    candidate_unclaimed = unclaimed.copy()

    existing_changes = int(np.count_nonzero(
        (labels > 0) & (candidate != labels)))
    if existing_changes:
        raise AssertionError(
            "ownerless-cohort recovery changed an existing label pixel")
    additions = (candidate > 0) & (labels == 0)
    overlap = int(np.count_nonzero(additions & (unclaimed > 0)))
    if overlap:
        raise AssertionError(
            f"ownerless-cohort recovery overlapped {overlap} unclaimed pixels")
    zero_signal = int(np.count_nonzero(additions & (raw == 0)))
    if zero_signal:
        raise AssertionError(
            f"ownerless-cohort recovery added {zero_signal} zero-signal pixels")
    unclaimed_changes = int(np.count_nonzero(
        candidate_unclaimed != unclaimed))
    if unclaimed_changes:
        raise AssertionError("ownerless-cohort recovery changed unclaimed ledger")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    applied = (applications[applications["outcome"] == "applied"]
               if len(applications) and "outcome" in applications
               else applications.iloc[0:0])
    if len(candidate_identities) != len(baseline_identities) + len(applied):
        raise AssertionError(
            "ownerless-cohort recovery did not add one identity per seat")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"ownerless-cohort recovery created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    edges.to_csv(stage / "fragment_edges.csv", index=False)
    applications.to_csv(stage / "seat_application_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0,
        "tracks_audited": int(len(audit)),
        "eligible_seed_tracks": int((
            audit.get("discovery_role", pd.Series(dtype=str)) == "seed").sum()),
        "eligible_follower_tracks": int(((
            audit.get("discovery_role", pd.Series(dtype=str)) ==
            "follower_candidate") & (
            audit.get("discovery_status", pd.Series(dtype=str)) ==
            "eligible")).sum()),
        "reciprocal_fragment_edges": int((
            edges.get("edge_status", pd.Series(dtype=str)) ==
            "reciprocal").sum()),
        "physical_seats": int(len(applications)),
        "applied_seats": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": unclaimed_changes,
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_added_pixels": int(np.count_nonzero(additions)),
        "existing_label_pixels_changed": existing_changes,
        "unclaimed_overlap_pixels": overlap,
        "zero_signal_additions": zero_signal,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue041_bracketed_encounter_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Restore identity seats across complete bracketed encounters."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_bracketed_encounter_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted bracketed-encounter recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "review_cases_path", "case_ids",
        "track_ids", "track_targets", "event_ids", "event_targets",
        "identity_ids", "identity_targets", "frame_ids", "frame_targets",
        "coordinates", "coordinate_targets", "forced_identity_ids",
        "forced_intervals", "review_regions",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide bracketed recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "35_bracketed_encounter_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "bracketed-encounter recovery requires current-run physical "
                f"tracks: {points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, candidate_unclaimed, audit = \
            bracketed_encounter_recovery.recover(
                labels, unclaimed, raw, physical_points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        audit = pd.DataFrame(columns=[
            "track_id", "owner_before", "owner_middle", "owner_after",
            "first_frame", "last_frame", "applied", "reason"])

    baseline_union = (labels > 0) | (unclaimed > 0)
    candidate_union = (candidate > 0) | (candidate_unclaimed > 0)
    foreground_changes = int(np.count_nonzero(candidate_union != baseline_union))
    if foreground_changes:
        raise AssertionError(
            "bracketed-encounter recovery changed foreground-ledger union")
    preexisting_unclaimed_changes = int(np.count_nonzero(
        (unclaimed > 0) & (candidate_unclaimed != unclaimed)))
    if preexisting_unclaimed_changes:
        raise AssertionError(
            "bracketed-encounter recovery changed existing unclaimed pixels")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "bracketed-encounter recovery changed active identities")
    duplicates = bracketed_encounter_recovery._new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"bracketed-encounter recovery created {duplicates} duplicates")

    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    audit.to_csv(stage / "bracketed_excursion_audit.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "tracks_audited": int(physical_points.track_id.nunique()),
        "bracketed_excursions_audited": int(len(audit)),
        "applied_excursions": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": foreground_changes,
        "label_foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "preexisting_unclaimed_changed_pixels": preexisting_unclaimed_changes,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue042_recurrent_exclusive_owner_relay(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Continue field-discovered dedicated seats across recurrent relays."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_recurrent_exclusive_owner_relay", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted recurrent exclusive-owner relay must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "review_cases_path", "case_ids",
        "track_ids", "track_targets", "event_ids", "event_targets",
        "identity_ids", "identity_targets", "frame_ids", "frame_targets",
        "coordinates", "coordinate_targets", "forced_identity_ids",
        "forced_intervals", "review_regions",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide recurrent exclusive-owner relay received forbidden "
            "targets: " + ", ".join(sorted(set(supplied))))

    stage = output_root / "36_recurrent_exclusive_owner_relay" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "recurrent exclusive-owner relay requires current-run "
                f"physical tracks: {points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, candidate_unclaimed, track_audit, frame_audit = \
            recurrent_exclusive_owner_relay.recover(
                labels, unclaimed, raw, physical_points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        track_audit = pd.DataFrame(columns=[
            "track_id", "eligible", "applied", "reason"])
        frame_audit = pd.DataFrame(columns=[
            "frame", "track_id", "dedicated_owner", "applied", "reason"])

    baseline_union = (labels > 0) | (unclaimed > 0)
    candidate_union = (candidate > 0) | (candidate_unclaimed > 0)
    foreground_changes = int(np.count_nonzero(candidate_union != baseline_union))
    if foreground_changes:
        raise AssertionError(
            "recurrent exclusive-owner relay changed foreground-ledger union")
    preexisting_unclaimed_changes = int(np.count_nonzero(
        (unclaimed > 0) & (candidate_unclaimed != unclaimed)))
    if preexisting_unclaimed_changes:
        raise AssertionError(
            "recurrent exclusive-owner relay changed existing unclaimed pixels")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "recurrent exclusive-owner relay changed active identities")
    duplicates = recurrent_exclusive_owner_relay._new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"recurrent exclusive-owner relay created {duplicates} duplicates")

    eligible = (track_audit[track_audit["eligible"].astype(bool)]
                if len(track_audit) and "eligible" in track_audit
                else track_audit.iloc[0:0])
    applied = (eligible[eligible["applied"].astype(bool)]
               if len(eligible) and "applied" in eligible
               else eligible.iloc[0:0])
    track_audit.to_csv(stage / "exclusive_owner_relay_audit.csv", index=False)
    frame_audit.to_csv(stage / "exclusive_owner_relay_frames.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "tracks_audited": int(physical_points.track_id.nunique()),
        "eligible_tracks": int(len(eligible)),
        "applied_tracks": int(len(applied)),
        "applied_runs": int(applied.get(
            "applied_runs", pd.Series(dtype=float)).sum()),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": foreground_changes,
        "label_foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "preexisting_unclaimed_changed_pixels": preexisting_unclaimed_changes,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, track_audit, frame_audit,
            metrics)


def _issue043_isolated_owner_blip_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore stable owners through field-discovered isolated owner blips."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_isolated_owner_blip_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted isolated owner-blip recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in isolated_owner_blip_recovery.FORBIDDEN_TARGETS and value)
    if supplied:
        raise ValueError(
            "field-wide isolated owner-blip recovery received forbidden "
            "targets: " + ", ".join(sorted(set(supplied))))

    stage = output_root / "37_isolated_owner_blip_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "isolated owner-blip recovery requires current-run physical "
                f"tracks: {points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, candidate_unclaimed, event_audit, frame_audit = \
            isolated_owner_blip_recovery.recover(
                labels, unclaimed, raw, physical_points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        event_audit = pd.DataFrame(columns=[
            "track_id", "eligible", "applied", "reason"])
        frame_audit = pd.DataFrame(columns=[
            "frame", "track_id", "stable_owner", "applied", "reason"])

    baseline_union = (labels > 0) | (unclaimed > 0)
    candidate_union = (candidate > 0) | (candidate_unclaimed > 0)
    foreground_changes = int(np.count_nonzero(candidate_union != baseline_union))
    if foreground_changes:
        raise AssertionError(
            "isolated owner-blip recovery changed foreground-ledger union")
    preexisting_unclaimed_changes = int(np.count_nonzero(
        (unclaimed > 0) & (candidate_unclaimed != unclaimed)))
    if preexisting_unclaimed_changes:
        raise AssertionError(
            "isolated owner-blip recovery changed existing unclaimed pixels")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "isolated owner-blip recovery changed active identities")
    duplicates = recurrent_exclusive_owner_relay._new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"isolated owner-blip recovery created {duplicates} duplicates")

    eligible = (event_audit[event_audit["eligible"].astype(bool)]
                if len(event_audit) and "eligible" in event_audit
                else event_audit.iloc[0:0])
    applied = (eligible[eligible["applied"].astype(bool)]
               if len(eligible) and "applied" in eligible
               else eligible.iloc[0:0])
    event_audit.to_csv(stage / "isolated_owner_blip_audit.csv", index=False)
    frame_audit.to_csv(stage / "isolated_owner_blip_frames.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "review_case_targets_received": False,
        "tracks_audited": int(physical_points.track_id.nunique()),
        "run_triples_audited": int(len(event_audit)),
        "eligible_blips": int(len(eligible)),
        "structurally_applicable_blips": int(len(applied)),
        "applied_blips": int(len(applied)),
        "changed_label_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_unclaimed_pixels": int(np.count_nonzero(
            candidate_unclaimed != unclaimed)),
        "changed_frames": int(np.count_nonzero(np.any(
            candidate != labels, axis=(1, 2)))),
        "foreground_changed_pixels": foreground_changes,
        "label_foreground_changed_pixels": int(np.count_nonzero(
            (candidate > 0) != (labels > 0))),
        "preexisting_unclaimed_changed_pixels": preexisting_unclaimed_changes,
        "baseline_active_identities": int(len(baseline_identities)),
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, event_audit, frame_audit,
            metrics)


def _issue045_atomic_two_seat_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Restore a resident and allocate its separate claimant atomically."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_atomic_two_seat_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted atomic two-seat recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = atomic_two_seat_recovery.FORBIDDEN_TARGETS | {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "frame_evidence_thresholds_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide atomic two-seat recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "38_atomic_two_seat_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        thresholds_path = (
            output_root / "25_raw_physical_hypotheses" / "out" /
            "frame_evidence_thresholds.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "atomic two-seat recovery requires current-run physical "
                f"tracks: {points_path}")
        if not thresholds_path.is_file():
            raise FileNotFoundError(
                "atomic two-seat recovery requires current-run thresholds: "
                f"{thresholds_path}")
        physical_points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, cohorts, applications, frame_audit = \
            atomic_two_seat_recovery.recover(
                labels, unclaimed, raw, physical_points, thresholds, params)
    else:
        candidate = labels.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        cohorts = pd.DataFrame(columns=[
            "cohort_id", "discovery_status", "discovery_reason"])
        applications = pd.DataFrame(columns=[
            "cohort_id", "resident_track", "claimant_track", "resident_owner",
            "assigned_claimant_identity", "outcome", "reason",
            "changed_pixels", "changed_frames"])
        frame_audit = pd.DataFrame(columns=[
            "cohort_id", "frame", "seat", "operation",
            "changed_pixels", "new_identity"])

    metrics = atomic_two_seat_recovery.summarize(
        labels, candidate, unclaimed, raw, physical_points, cohorts,
        applications, params)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    if metrics["preexisting_unclaimed_changed_pixels"]:
        raise AssertionError(
            "atomic two-seat recovery changed existing unclaimed pixels")
    if metrics["new_same_frame_identity_components"]:
        raise AssertionError(
            "atomic two-seat recovery created duplicate identity components")
    cohorts.to_csv(stage / "two_seat_cohorts.csv", index=False)
    applications.to_csv(stage / "two_seat_applications.csv", index=False)
    frame_audit.to_csv(stage / "two_seat_frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed.copy(), cohorts, applications, frame_audit,
            metrics)


def _issue046_recording_start_duplicate_release(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Backfill complete early two-seat owner episodes atomically."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_recording_start_duplicate_release", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted recording-start duplicate release must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    recording_start_duplicate_release.assert_target_free(params)
    forbidden_config = sorted(
        key for key, value in configured.items()
        if value and any(
            token in key.lower()
            for token in
            recording_start_duplicate_release.FORBIDDEN_TARGET_TOKENS))
    if forbidden_config:
        raise ValueError(
            "field-wide recording-start duplicate release received forbidden "
            "targets: " + ", ".join(forbidden_config))

    stage = output_root / "39_recording_start_duplicate_release" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        candidate, audit = \
            recording_start_duplicate_release.recover_duplicate_owner_release(
                labels, raw, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "newcomer_identity", "birth_frame", "incumbent_identity",
            "eligible", "applied", "changed_pixels", "changed_frames",
            "reason"])

    metrics = recording_start_duplicate_release.summarize(
        labels, candidate, unclaimed, raw, audit, params)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    if metrics["foreground_changed_pixels"]:
        raise AssertionError(
            "recording-start duplicate release changed foreground")
    if metrics["preexisting_unclaimed_changed_pixels"] or \
            metrics["unclaimed_overlap_pixels"]:
        raise AssertionError(
            "recording-start duplicate release changed/overlapped unclaimed")
    if metrics["zero_signal_additions"]:
        raise AssertionError(
            "recording-start duplicate release added zero-signal pixels")
    if not metrics["old_identity_set_preserved"]:
        raise AssertionError(
            "recording-start duplicate release changed active identities")
    if metrics["new_same_frame_identity_components"]:
        raise AssertionError(
            "recording-start duplicate release created duplicate components")
    audit.to_csv(stage / "duplicate_release_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, metrics


def _issue048_late_owner_backfill(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Backfill a late stable owner through its complete physical lineage."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_late_owner_backfill", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted late-owner backfill must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    late_owner_backfill.assert_target_free(configured)
    late_owner_backfill.assert_target_free(params)
    forbidden_names = late_owner_backfill.FORBIDDEN_TARGETS | {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "frame_evidence_thresholds_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide late-owner backfill received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "40_late_owner_backfill" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        thresholds_path = (
            output_root / "25_raw_physical_hypotheses" / "out" /
            "frame_evidence_thresholds.csv")
        if not points_path.is_file() or not thresholds_path.is_file():
            raise FileNotFoundError(
                "late-owner backfill requires current-run physical tracks "
                f"and thresholds: {points_path}, {thresholds_path}")
        physical_points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, proposals, applications, frame_audit = \
            late_owner_backfill.recover(
                labels, unclaimed, raw, physical_points, thresholds, params)
    else:
        candidate = labels.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        proposals = pd.DataFrame(columns=[
            "proposal_id", "track_id", "discovery_status",
            "discovery_reason"])
        applications = pd.DataFrame(columns=[
            "proposal_id", "track_id", "stable_owner", "outcome", "reason",
            "changed_pixels", "changed_frames"])
        frame_audit = pd.DataFrame(columns=[
            "proposal_id", "frame", "stable_owner", "changed_pixels"])

    metrics = late_owner_backfill.summarize(
        labels, candidate, unclaimed, raw, physical_points, proposals,
        applications, "candidate" if enabled else "disabled")
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "parameters": params,
    })
    if metrics["removed_foreground_pixels"]:
        raise AssertionError("late-owner backfill removed foreground")
    if metrics["preexisting_unclaimed_changed_pixels"] or \
            metrics["unclaimed_overlap_additions"]:
        raise AssertionError(
            "late-owner backfill changed/overlapped existing unclaimed pixels")
    if metrics["zero_signal_additions"]:
        raise AssertionError("late-owner backfill added zero-signal pixels")
    if not metrics["old_identity_set_preserved"]:
        raise AssertionError("late-owner backfill changed active identities")
    if metrics["donor_named_frame_losses"]:
        raise AssertionError("late-owner backfill erased a donor frame")
    if metrics["new_same_frame_identity_components"]:
        raise AssertionError(
            "late-owner backfill created duplicate identity components")

    proposals.to_csv(stage / "late_owner_proposals.csv", index=False)
    applications.to_csv(stage / "late_owner_applications.csv", index=False)
    frame_audit.to_csv(stage / "late_owner_frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed.copy(), proposals, applications,
            frame_audit, metrics)


def _issue051_bracketed_handoff_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Repair a terminal owner bracket split across a physical handoff."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_bracketed_handoff_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted bracketed-handoff recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    bracketed_handoff_recovery.assert_target_free(configured)
    bracketed_handoff_recovery.assert_target_free(params)
    forbidden_names = bracketed_handoff_recovery.FORBIDDEN_TARGETS | {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "frame_evidence_thresholds_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide bracketed-handoff recovery received forbidden "
            "targets: " + ", ".join(sorted(set(supplied))))

    stage = output_root / "41_bracketed_handoff_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        thresholds_path = (
            output_root / "25_raw_physical_hypotheses" / "out" /
            "frame_evidence_thresholds.csv")
        if not points_path.is_file() or not thresholds_path.is_file():
            raise FileNotFoundError(
                "bracketed-handoff recovery requires current-run physical "
                f"tracks and thresholds: {points_path}, {thresholds_path}")
        physical_points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, proposals, applications, frame_audit = \
            bracketed_handoff_recovery.recover(
                labels, unclaimed, raw, physical_points, thresholds, params)
    else:
        candidate = labels.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        proposals = pd.DataFrame(columns=[
            "proposal_id", "successor_track", "discovery_status",
            "discovery_reason"])
        applications = pd.DataFrame(columns=[
            "proposal_id", "outcome", "reason", "changed_pixels",
            "changed_frames"])
        frame_audit = pd.DataFrame(columns=[
            "proposal_id", "frame", "changed_pixels"])

    metrics = bracketed_handoff_recovery.summarize(
        labels, candidate, unclaimed, raw, physical_points, proposals,
        applications, "candidate" if enabled else "disabled")
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "parameters": params,
    })
    if metrics["removed_foreground_pixels"]:
        raise AssertionError("bracketed-handoff recovery removed foreground")
    if metrics["preexisting_unclaimed_changed_pixels"] or \
            metrics["unclaimed_overlap_additions"]:
        raise AssertionError(
            "bracketed-handoff recovery changed/overlapped unclaimed pixels")
    if metrics["zero_signal_additions"]:
        raise AssertionError(
            "bracketed-handoff recovery added zero-signal pixels")
    if not metrics["old_identity_set_preserved"]:
        raise AssertionError(
            "bracketed-handoff recovery changed active identities")
    if metrics["donor_named_frame_losses"]:
        raise AssertionError(
            "bracketed-handoff recovery erased a named owner frame")
    if metrics["new_same_frame_identity_components"]:
        raise AssertionError(
            "bracketed-handoff recovery created duplicate components")

    proposals.to_csv(stage / "handoff_proposals.csv", index=False)
    applications.to_csv(stage / "handoff_applications.csv", index=False)
    frame_audit.to_csv(stage / "handoff_frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed.copy(), proposals, applications,
            frame_audit, metrics)


def _issue055_post_split_backfill(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Backfill a late owner only after a field-discovered physical split."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_post_split_backfill", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("accepted post-split backfill must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    post_split_backfill.assert_target_free(configured)
    post_split_backfill.assert_target_free(params)
    forbidden_names = post_split_backfill.FORBIDDEN_TARGETS | {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide post-split backfill received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "42_post_split_backfill" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        if not points_path.is_file():
            raise FileNotFoundError(
                "post-split backfill requires current-run physical tracks: "
                f"{points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, proposals, applications, frame_audit = \
            post_split_backfill.recover(
                labels, unclaimed, raw, physical_points, params)
    else:
        candidate = labels.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        proposals = pd.DataFrame(columns=[
            "proposal_id", "target_track", "discovery_status",
            "discovery_reason"])
        applications = pd.DataFrame(columns=[
            "proposal_id", "target_track", "outcome", "reason",
            "changed_pixels", "changed_frames"])
        frame_audit = pd.DataFrame(columns=[
            "proposal_id", "frame", "changed_pixels"])

    metrics = post_split_backfill.summarize(
        labels, candidate, unclaimed, raw, physical_points, proposals,
        applications, "candidate" if enabled else "disabled")
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "parameters": params,
    })
    if metrics["removed_foreground_pixels"]:
        raise AssertionError("post-split backfill removed foreground")
    if metrics["preexisting_unclaimed_changed_pixels"] or \
            metrics["unclaimed_overlap_additions"]:
        raise AssertionError(
            "post-split backfill changed/overlapped unclaimed pixels")
    if metrics["zero_signal_additions"]:
        raise AssertionError("post-split backfill added zero-signal pixels")
    if not metrics["old_identity_set_preserved"]:
        raise AssertionError("post-split backfill changed active identities")
    if metrics["donor_named_frame_losses"]:
        raise AssertionError("post-split backfill erased a named owner frame")
    if metrics["new_same_frame_identity_components"]:
        raise AssertionError(
            "post-split backfill created duplicate components")

    proposals.to_csv(stage / "post_split_proposals.csv", index=False)
    applications.to_csv(stage / "post_split_applications.csv", index=False)
    frame_audit.to_csv(stage / "post_split_frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed.copy(), proposals, applications,
            frame_audit, metrics)


def _issue056_concurrent_duplicate_invasion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Restore residents displaced by a concurrent duplicate owner."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_concurrent_duplicate_invasion", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted concurrent-owner recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    concurrent_duplicate_invasion.assert_target_free(configured)
    concurrent_duplicate_invasion.assert_target_free(params)
    forbidden_names = concurrent_duplicate_invasion.FORBIDDEN_TARGETS | {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path", "frame_evidence_thresholds_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide concurrent-owner recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "43_concurrent_duplicate_invasion" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        points_path = (output_root / "26_latent_body_tracks" / "out" /
                       "latent_track_points.csv")
        thresholds_path = (output_root / "25_raw_physical_hypotheses" /
                           "out" / "frame_evidence_thresholds.csv")
        if not points_path.is_file() or not thresholds_path.is_file():
            raise FileNotFoundError(
                "concurrent-owner recovery requires current-run physical "
                f"tracks and thresholds: {points_path}; {thresholds_path}")
        physical_points = pd.read_csv(points_path)
        thresholds = pd.read_csv(thresholds_path)
        candidate, proposals, applications, frame_audit, _ = \
            concurrent_duplicate_invasion.recover(
                labels, unclaimed, raw, physical_points, thresholds, params)
    else:
        candidate = labels.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        proposals = pd.DataFrame(columns=[
            "proposal_id", "resident_track", "discovery_status",
            "discovery_reason"])
        applications = pd.DataFrame(columns=[
            "proposal_id", "resident_track", "outcome", "reason",
            "changed_pixels", "changed_frames"])
        frame_audit = pd.DataFrame(columns=[
            "proposal_id", "frame", "changed_pixels"])

    metrics = concurrent_duplicate_invasion.summarize(
        labels, candidate, unclaimed, raw, physical_points, proposals,
        applications, "candidate" if enabled else "disabled")
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "parameters": params,
    })
    if metrics["removed_foreground_pixels"]:
        raise AssertionError(
            "concurrent-owner recovery removed foreground")
    if metrics["preexisting_unclaimed_changed_pixels"] or \
            metrics["unclaimed_overlap_additions"]:
        raise AssertionError(
            "concurrent-owner recovery changed/overlapped unclaimed pixels")
    if metrics["zero_signal_additions"]:
        raise AssertionError(
            "concurrent-owner recovery added zero-signal pixels")
    if not metrics["old_identity_set_preserved"]:
        raise AssertionError(
            "concurrent-owner recovery changed active identities")
    if metrics["donor_named_frame_losses"]:
        raise AssertionError(
            "concurrent-owner recovery erased a named owner frame")
    if metrics["new_same_frame_identity_components"]:
        raise AssertionError(
            "concurrent-owner recovery created duplicate components")

    proposals.to_csv(stage / "concurrent_invasion_proposals.csv", index=False)
    applications.to_csv(
        stage / "concurrent_invasion_applications.csv", index=False)
    frame_audit.to_csv(
        stage / "concurrent_invasion_frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed.copy(), proposals, applications,
            frame_audit, metrics)


def _issue063_post_split_excursion_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Restore short owner interruptions after durable two-body splits."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_post_split_excursion_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted post-split excursion recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = bracketed_encounter_recovery.FORBIDDEN_TARGETS | {
        "labels_path", "unclaimed_path", "raw_path",
        "physical_track_points_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide post-split excursion recovery received forbidden "
            "targets: " + ", ".join(sorted(set(supplied))))

    stage = output_root / "44_post_split_excursion_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if enabled:
        if not points_path.is_file():
            raise FileNotFoundError(
                "post-split excursion recovery requires current-run physical "
                f"tracks: {points_path}")
        physical_points = pd.read_csv(points_path)
        candidate, candidate_unclaimed, audit = \
            bracketed_encounter_recovery.recover(
                labels, unclaimed, raw, physical_points, params)
    else:
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        physical_points = pd.DataFrame(columns=["track_id"])
        audit = pd.DataFrame(columns=[
            "track_id", "owner_before", "owner_middle", "owner_after",
            "first_frame", "last_frame", "applied", "reason"])

    label_changes = candidate != labels
    unclaimed_changes = candidate_unclaimed != unclaimed
    baseline_union = (labels > 0) | (unclaimed > 0)
    candidate_union = (candidate > 0) | (candidate_unclaimed > 0)
    foreground_changes = int(np.count_nonzero(
        baseline_union != candidate_union))
    preexisting_unclaimed_changes = int(np.count_nonzero(
        (unclaimed > 0) & unclaimed_changes))
    unclaimed_overlap = int(np.count_nonzero(
        (candidate > 0) & (candidate_unclaimed > 0)))
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    duplicates = bracketed_encounter_recovery._new_duplicate_components(
        labels, candidate)
    if foreground_changes:
        raise AssertionError(
            "post-split excursion recovery changed foreground-ledger union")
    if preexisting_unclaimed_changes or unclaimed_overlap:
        raise AssertionError(
            "post-split excursion recovery changed/overlapped unclaimed pixels")
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "post-split excursion recovery changed active identities")
    if duplicates:
        raise AssertionError(
            "post-split excursion recovery created duplicate components")

    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    audit.to_csv(stage / "post_split_excursion_audit.csv", index=False)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0,
        "track_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
        "review_case_targets_received": False,
        "tracks_audited": int(physical_points.track_id.nunique()),
        "excursions_audited": int(len(audit)),
        "eligible_proposals": int(len(applied)),
        "applied_proposals": int(len(applied)),
        "rejected_application_proposals": 0,
        "changed_pixels": int(np.count_nonzero(label_changes)),
        "changed_frames": int(np.count_nonzero(np.any(
            label_changes, axis=(1, 2)))),
        "relabelled_foreground_pixels": int(np.count_nonzero(
            label_changes & (labels > 0) & (candidate > 0))),
        "raw_supported_additions": int(np.count_nonzero(
            label_changes & (labels == 0) & (candidate > 0))),
        "removed_foreground_pixels": int(np.count_nonzero(
            label_changes & (labels > 0) & (candidate == 0))),
        "foreground_ledger_changes": foreground_changes,
        "preexisting_unclaimed_changed_pixels":
            preexisting_unclaimed_changes,
        "unclaimed_overlap_additions": unclaimed_overlap,
        "old_identity_set_preserved":
            candidate_identities == baseline_identities,
        "new_same_frame_identity_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, candidate_unclaimed, audit, metrics


def _issue066_transient_misownership_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Name isolated ownerless tails after a field-discovered dormant fork."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_transient_misownership_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted transient-misownership recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    transiently_misowned_body_recovery.assert_target_free(params)

    stage = output_root / "46_transient_misownership_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    thresholds_path = (output_root / "25_raw_physical_hypotheses" / "out" /
                       "frame_evidence_thresholds.csv")
    missing = [str(path) for path in (points_path, thresholds_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "accepted transient-misownership recovery requires current-run "
            "physical evidence: " + ", ".join(missing))
    points = pd.read_csv(points_path)
    thresholds = pd.read_csv(thresholds_path)

    if enabled:
        candidate, candidate_unclaimed, proposals, audit, attached = \
            transiently_misowned_body_recovery.recover(
                labels, unclaimed, raw, points, thresholds, params)
    else:
        proposals, attached = transiently_misowned_body_recovery.discover(
            labels, unclaimed, points, params)
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        audit = proposals.copy()
        audit["assigned_identity"] = 0
        audit["application_status"] = "disabled"
        audit["application_reason"] = "disabled"
        for column in (
                "changed_pixels", "changed_frames",
                "relabelled_owner_pixels", "claimed_unclaimed_pixels",
                "added_background_pixels", "skipped_core_frames"):
            audit[column] = 0

    metrics = transiently_misowned_body_recovery.summarize(
        labels, unclaimed, candidate, candidate_unclaimed,
        proposals, audit)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    proposals.to_csv(stage / "track_proposals.csv", index=False)
    audit.to_csv(stage / "application_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, proposals, audit,
            attached, metrics)


def _issue070_isolated_unowned_lifetime_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Allocate names to field-discovered complete isolated lifetimes."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_isolated_unowned_lifetime_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted isolated-lifetime recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    isolated_unowned_lifetime.assert_target_free(params)

    stage = output_root / "47_isolated_unowned_lifetime_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    thresholds_path = (output_root / "25_raw_physical_hypotheses" / "out" /
                       "frame_evidence_thresholds.csv")
    missing = [str(path) for path in (points_path, thresholds_path)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "accepted isolated-lifetime recovery requires current-run "
            "physical evidence: " + ", ".join(missing))
    points = pd.read_csv(points_path)
    thresholds = pd.read_csv(thresholds_path)

    if enabled:
        candidate, candidate_unclaimed, proposals, applications, \
            frame_audit, attached = isolated_unowned_lifetime.recover(
                labels, unclaimed, raw, points, thresholds, params)
    else:
        proposals, attached = isolated_unowned_lifetime.discover(
            labels, unclaimed, points, params)
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        applications = pd.DataFrame(
            columns=isolated_unowned_lifetime.APPLICATION_COLUMNS)
        frame_audit = pd.DataFrame(
            columns=isolated_unowned_lifetime.FRAME_COLUMNS)

    metrics = isolated_unowned_lifetime.summarize(
        labels, unclaimed, candidate, candidate_unclaimed,
        proposals, applications)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    proposals.to_csv(stage / "proposals.csv", index=False)
    applications.to_csv(stage / "unowned_lifetime_audit.csv", index=False)
    frame_audit.to_csv(
        stage / "unowned_lifetime_frame_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, proposals, applications,
            frame_audit, attached, metrics)


def _issue071_boundary_owner_blip_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Restore field-discovered owner blips censored by a movie boundary."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_boundary_owner_blip_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted boundary owner-blip recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    boundary_owner_blip.assert_target_free(params)

    stage = output_root / "48_boundary_owner_blip_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "accepted boundary owner-blip recovery requires current-run "
            f"physical evidence: {points_path}")
    points = pd.read_csv(points_path)

    if enabled:
        candidate, candidate_unclaimed, proposals, applications, \
            frame_audit = boundary_owner_blip.recover(
                labels, unclaimed, raw, points, params)
        mode = "boundary_censored"
    else:
        scored = separable_merge_recovery.attach_owners(points, labels)
        proposals = boundary_owner_blip.discover(
            scored, len(labels), params)
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        applications = pd.DataFrame(
            columns=boundary_owner_blip.APPLICATION_COLUMNS)
        frame_audit = pd.DataFrame(
            columns=boundary_owner_blip.FRAME_COLUMNS)
        mode = "disabled"

    metrics = boundary_owner_blip.summarize(
        labels, unclaimed, candidate, candidate_unclaimed,
        proposals, applications, mode)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    proposals.to_csv(stage / "proposals.csv", index=False)
    applications.to_csv(stage / "boundary_blip_audit.csv", index=False)
    frame_audit.to_csv(stage / "boundary_blip_frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, proposals, applications,
            frame_audit, metrics)


def _issue078_foreign_owner_terminal_convergence(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Preserve every eligible pair of owner seats through convergence."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_foreign_owner_terminal_convergence", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    foreign_owner_terminal_convergence.assert_target_free(params)

    stage = output_root / "50_foreign_owner_terminal_convergence" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "accepted foreign-owner convergence recovery requires "
            f"current-run physical evidence: {points_path}")
    points = pd.read_csv(points_path)

    if enabled:
        candidate, proposals, applications, frame_audit, attached = \
            foreign_owner_terminal_convergence.recover(
                labels, unclaimed, raw, points, params)
        mode = "candidate"
    else:
        proposals, attached = foreign_owner_terminal_convergence.discover(
            labels, raw, points, params)
        candidate = labels.copy()
        applications = pd.DataFrame(columns=[
            "proposal_id", "left_track", "right_track", "outcome",
            "reason", "changed_pixels", "changed_frames"])
        frame_audit = pd.DataFrame()
        mode = "disabled"

    candidate_unclaimed = unclaimed.copy()
    metrics = foreign_owner_terminal_convergence.summarize(
        labels, candidate, unclaimed, raw, points, proposals,
        applications, mode)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    proposals.to_csv(stage / "proposals.csv", index=False)
    applications.to_csv(
        stage / "terminal_convergence_audit.csv", index=False)
    frame_audit.to_csv(stage / "frame_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, proposals, applications,
            frame_audit, attached, metrics)


def _issue080_terminal_companion_assimilation(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Preserve a stable companion owner at an eligible terminal assimilation."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_terminal_companion_assimilation", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    terminal_companion_assimilation.assert_target_free(params)

    stage = output_root / "51_terminal_companion_assimilation" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "accepted terminal companion assimilation requires current-run "
            f"physical evidence: {points_path}")
    points = pd.read_csv(points_path)

    if enabled:
        candidate, proposals, applications, frame_audit, attached = \
            terminal_companion_assimilation.recover(
                labels, raw, points, params)
        mode = "candidate"
    else:
        proposals, attached = terminal_companion_assimilation.discover(
            labels, raw, points, params)
        candidate = labels.copy()
        applications = pd.DataFrame(columns=[
            "proposal_id", "physical_track", "outcome", "reason",
            "changed_pixels", "changed_frames"])
        frame_audit = pd.DataFrame()
        mode = "disabled"

    candidate_unclaimed = unclaimed.copy()
    metrics = terminal_companion_assimilation.summarize(
        labels, candidate, unclaimed, raw, points, proposals,
        applications, mode)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
    })
    proposals.to_csv(stage / "proposals.csv", index=False)
    applications.to_csv(
        stage / "terminal_assimilation_audit.csv", index=False)
    frame_audit.to_csv(stage / "frame_audit.csv", index=False)
    attached.to_csv(stage / "physical_owner_points.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, proposals, applications,
            frame_audit, attached, metrics)


def _issue089_cross_reference_owner_relay(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Preserve stationary and moving owners through eligible merge relays."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_cross_reference_owner_relay", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    cross_reference_owner_relay.assert_target_free(params)

    stage = output_root / "52_cross_reference_owner_relay" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    thresholds_path = (output_root / "25_raw_physical_hypotheses" / "out" /
                       "frame_evidence_thresholds.csv")
    if not points_path.is_file() or not thresholds_path.is_file():
        raise FileNotFoundError(
            "accepted cross-reference owner relay requires current-run "
            f"physical evidence: {points_path}, {thresholds_path}")
    points = pd.read_csv(points_path)
    thresholds = pd.read_csv(thresholds_path)
    discovered_proposals, attached = cross_reference_owner_relay.discover(
        labels, points, params)
    discovery_path = stage / "reference_handoff_audit.csv"
    points_audit_path = stage / "physical_owner_points.csv"
    discovered_proposals.to_csv(discovery_path, index=False)
    attached.to_csv(points_audit_path, index=False)

    if enabled:
        # The accepted tuning path separates discovery and application with
        # an immutable CSV audit.  Re-read that generic boundary so production
        # preserves the accepted catalogue bytes as well as its arrays.
        precomputed = (pd.read_csv(discovery_path),
                       pd.read_csv(points_audit_path))
        candidate, candidate_unclaimed, proposals, applications, frame_audit = \
            cross_reference_owner_relay.recover(
                labels, unclaimed, raw, points, params, thresholds,
                precomputed=precomputed)
        mode = "candidate"
    else:
        proposals = discovered_proposals
        candidate = labels.copy()
        candidate_unclaimed = unclaimed.copy()
        applications = pd.DataFrame()
        frame_audit = pd.DataFrame()
        mode = "disabled"

    metrics = cross_reference_owner_relay.summarize(
        labels, unclaimed, candidate, candidate_unclaimed,
        proposals, applications, raw)
    metrics.update({
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": mode,
    })
    proposals.to_csv(stage / "reference_relay_proposals.csv", index=False)
    applications.to_csv(
        stage / "reference_relay_application_audit.csv", index=False)
    frame_audit.to_csv(stage / "frame_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, proposals, applications,
            frame_audit, attached, metrics)


def _issue091_recording_boundary_owner_cycle(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Repair every uniquely supported recording-boundary owner cycle."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_recording_boundary_owner_cycle", {})
    enabled = bool(configured.get("enabled", False))
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    recording_boundary_owner_cycle.assert_target_free(params)

    stage = output_root / "53_recording_boundary_owner_cycle" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if not points_path.is_file():
        raise FileNotFoundError(
            "accepted recording-boundary owner-cycle repair requires "
            f"current-run physical evidence: {points_path}")
    points = pd.read_csv(points_path)
    if enabled:
        (candidate, candidate_unclaimed, edges, cycles, members, anchors,
         applications, details) = recording_boundary_owner_cycle.recover(
             labels, unclaimed, raw, points, params)
        mode = "candidate"
    else:
        edges, cycles, members = recording_boundary_owner_cycle.discover_cycles(
            labels, points, params)
        candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
        anchors = applications = pd.DataFrame()
        details = {
            "metrics": {
                "audited_edges": int(len(edges)),
                "audited_cycles": int(len(cycles)),
                "eligible_cycles": int(cycles.discovery_status.eq(
                    "eligible").sum()) if len(cycles) else 0,
                "applied_cycles": 0,
                "changed_pixels": 0,
                "changed_frames": 0,
                "foreground_changed_pixels": 0,
                "unclaimed_changed_pixels": 0,
                "active_identity_set_preserved": True,
            },
            "anchor_frame_audit": pd.DataFrame(),
            "duplicate_release_audit": pd.DataFrame(),
        }
        mode = "disabled"
    metrics = {
        **details["metrics"],
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": mode,
        "targeting_mode": "field_wide_discovery",
        "identity_targets": 0,
        "track_targets": 0,
        "frame_targets": 0,
        "coordinate_targets": 0,
        "event_targets": 0,
        "region_targets": 0,
        "review_case_targets": 0,
    }
    edges.to_csv(stage / "boundary_transition_edges.csv", index=False)
    cycles.to_csv(stage / "boundary_owner_cycles.csv", index=False)
    members.to_csv(stage / "boundary_cycle_members.csv", index=False)
    anchors.to_csv(stage / "boundary_anchor_audit.csv", index=False)
    applications.to_csv(
        stage / "boundary_cycle_application_audit.csv", index=False)
    details["anchor_frame_audit"].to_csv(
        stage / "anchor_frame_audit.csv", index=False)
    details["duplicate_release_audit"].to_csv(
        stage / "duplicate_release_audit.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, edges, cycles, members, anchors,
            applications, metrics)


def _issue092_complete_duplicate_lineage(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Allocate identities only to sustained independent duplicate somata."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_complete_duplicate_lineage", {})
    enabled = bool(configured.get("enabled", False))
    duplicate_params = deepcopy(configured.get("duplicate_parameters", {}))
    lineage_params = deepcopy(configured.get("lineage_parameters", {}))
    duplicate_params["targeting_mode"] = "field_wide_discovery"
    lineage_params["targeting_mode"] = "field_wide_discovery"
    duplicate_soma_exclusivity.assert_target_free(duplicate_params)
    complete_duplicate_lineage.assert_target_free(lineage_params)

    stage = output_root / "54_complete_duplicate_lineage" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    thresholds_path = (output_root / "25_raw_physical_hypotheses" / "out" /
                       "frame_evidence_thresholds.csv")
    if not points_path.is_file() or not thresholds_path.is_file():
        raise FileNotFoundError(
            "accepted complete-duplicate-lineage allocation requires "
            f"current-run physical evidence: {points_path}, {thresholds_path}")
    points = pd.read_csv(points_path)
    thresholds = pd.read_csv(thresholds_path)

    # The capacity-one projection supplies only a field-wide discovery audit.
    # Its masks are deliberately discarded so projection-like branches remain
    # unchanged unless the complete-lineage rule independently accepts them.
    (_, _, duplicate_before, duplicate_after,
     duplicate_applications, _) = duplicate_soma_exclusivity.recover(
         labels, unclaimed, raw, points, duplicate_params)
    if enabled:
        (candidate, candidate_unclaimed, proposals, applications,
         frame_audit, _) = complete_duplicate_lineage.recover(
             labels, unclaimed, raw, points, thresholds,
             duplicate_applications, lineage_params)
        mode = "candidate"
    else:
        candidate, candidate_unclaimed = labels.copy(), unclaimed.copy()
        proposals = applications = frame_audit = pd.DataFrame()
        mode = "disabled"

    changed = candidate != labels
    ids_before = set(map(int, np.unique(labels))) - {0}
    ids_after = set(map(int, np.unique(candidate))) - {0}
    union_before = (labels > 0) | (unclaimed > 0)
    union_after = (candidate > 0) | (candidate_unclaimed > 0)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "mode": mode,
        "targeting_mode": "field_wide_discovery",
        "duplicate_rows_audited": int(len(duplicate_before)),
        "duplicate_projection_applications": int(
            len(duplicate_applications)),
        "proposals": int(len(proposals)),
        "eligible_proposals": int(proposals.discovery_status.eq(
            "eligible").sum()) if len(proposals) else 0,
        "applied_proposals": int(applications.outcome.eq(
            "applied").sum()) if len(applications) else 0,
        "new_identities": sorted(ids_after - ids_before),
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_frames": int(np.count_nonzero(np.any(
            changed, axis=(1, 2)))),
        "raw_supported_additions": int(np.count_nonzero(
            union_after & ~union_before)),
        "foreground_removed_pixels": int(np.count_nonzero(
            union_before & ~union_after)),
        "foreground_ledger_union_preserved_or_extended": bool(not np.any(
            union_before & ~union_after)),
        "projection_conservative_default": True,
        "rejected_proposals_applied": 0,
        "identity_targets": 0,
        "track_targets": 0,
        "frame_targets": 0,
        "coordinate_targets": 0,
        "event_targets": 0,
        "region_targets": 0,
        "review_case_targets": 0,
    }
    duplicate_before.to_csv(
        stage / "duplicate_soma_before.csv", index=False)
    duplicate_after.to_csv(stage / "duplicate_soma_after.csv", index=False)
    duplicate_applications.to_csv(
        stage / "duplicate_exclusivity_application_audit.csv", index=False)
    proposals.to_csv(stage / "complete_lineage_proposals.csv", index=False)
    applications.to_csv(
        stage / "complete_lineage_applications.csv", index=False)
    frame_audit.to_csv(stage / "complete_lineage_frames.csv", index=False)
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, candidate_unclaimed, duplicate_before,
            duplicate_applications, proposals, applications, frame_audit,
            metrics)


def _well_issue001_isolated_pair_encounter_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Recover isolated separable pairs using only current-run evidence."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_isolated_pair_encounter_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted isolated-pair recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    forbidden_names = {
        "labels_path", "raw_path", "review_cases_path", "case_ids",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "track_ids", "track_targets", "frame_ids", "frame_targets",
        "coordinates", "coordinate_targets", "regions", "region_targets",
        "forced_identity_ids", "forced_intervals",
        "physical_track_points_path", "encounter_frames_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide isolated-pair recovery received forbidden targets: "
            + ", ".join(sorted(set(supplied))))

    stage = output_root / "55_isolated_pair_encounter_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    if enabled:
        evidence = output_root / "26_latent_body_tracks" / "out"
        points_path = evidence / "latent_track_points.csv"
        encounters_path = evidence / "encounter_frames.csv"
        missing = [path for path in (points_path, encounters_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "accepted isolated-pair recovery requires current-run "
                "physical evidence: " + ", ".join(map(str, missing)))
        # Preserve the accepted separately-runnable CSV boundary exactly.
        points = pd.read_csv(points_path)
        encounters = pd.read_csv(encounters_path)
        candidate, audit = \
            separable_merge_recovery.recover_isolated_pair_encounters(
                labels, raw, points, encounters, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=[
            "encounter_id", "frame", "track_a", "track_b", "applied",
            "changed_pixels", "reason"])

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "isolated-pair recovery changed foreground segmentation")
    baseline_identities = set(map(int, np.unique(labels))) - {0}
    candidate_identities = set(map(int, np.unique(candidate))) - {0}
    if candidate_identities != baseline_identities:
        raise AssertionError(
            "isolated-pair recovery changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            f"isolated-pair recovery created {duplicates} duplicate components")
    audit.to_csv(stage / "producer_audit.csv", index=False)
    applied = (audit[audit["applied"].astype(bool)]
               if len(audit) and "applied" in audit else audit.iloc[0:0])
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled, "targeting_mode": "field_wide_discovery",
        "identity_target_count": 0, "track_target_count": 0,
        "frame_target_count": 0, "coordinate_target_count": 0,
        "event_target_count": 0, "region_target_count": 0,
        "review_case_target_count": 0,
        "proposals_audited": int(len(audit)),
        "applied_proposals": int(len(applied)),
        "applied_track_pairs": int(
            applied[["track_a", "track_b"]].drop_duplicates().shape[0]
            if len(applied) else 0),
        "changed_pixels": int(np.count_nonzero(candidate != labels)),
        "changed_frames": int(np.count_nonzero(
            np.any(candidate != labels, axis=(1, 2)))),
        "absorbed_coreless_pixels": int(
            audit.get("absorbed_coreless_pixels", pd.Series(dtype=float))
            .fillna(0).sum()),
        "reverted_duplicate_proposals": int(
            audit.reason.eq("new_duplicate_component").sum()),
        "reverted_crowded_proposals": int(
            audit.reason.eq("crowded_three_body_context").sum()),
        "foreground_changed_pixels": 0,
        "unclaimed_changed_pixels": 0,
        "active_identities": int(len(candidate_identities)),
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, metrics


def _well_issue004_recurrent_two_body_fusion_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recover recurrent two-body fusions from current-run field evidence."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_recurrent_two_body_fusion_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted recurrent-fusion recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    forbidden_names = {
        "labels_path", "raw_path", "review_cases_path", "case_ids",
        "event_ids", "event_targets", "identity_ids", "identity_targets",
        "track_ids", "track_targets", "frame_ids", "frame_targets",
        "coordinates", "coordinate_targets", "regions", "region_targets",
        "forced_identity_ids", "forced_intervals",
        "physical_track_points_path", "accepted_application_audit_path",
        "accepted_conflicted_lineage_audit_path",
    }
    supplied = sorted(
        name for values in (configured, params)
        for name, value in values.items()
        if name in forbidden_names and value)
    if supplied:
        raise ValueError(
            "field-wide recurrent-fusion recovery received forbidden "
            "targets: " + ", ".join(sorted(set(supplied))))
    recurrent_two_body_fusion.assert_target_free(params)

    stage = output_root / "60_recurrent_two_body_fusion" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if enabled and not points_path.is_file():
        raise FileNotFoundError(
            "accepted recurrent-fusion recovery requires current-run "
            f"physical evidence: {points_path}")

    if enabled:
        points = pd.read_csv(points_path)
        pair_audit, absorption_events, scored = \
            recurrent_two_body_fusion.discover(labels, points, params)
        candidate, verdicts, applications = \
            recurrent_two_body_fusion.apply(
                labels, raw, pair_audit, scored, params)
    else:
        candidate = labels.copy()
        pair_audit = pd.DataFrame()
        absorption_events = pd.DataFrame()
        verdicts = pd.DataFrame()
        applications = pd.DataFrame()

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "recurrent-fusion recovery changed foreground segmentation")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "recurrent-fusion recovery overlaps assigned and unclaimed pixels")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError(
            "recurrent-fusion recovery changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "recurrent-fusion recovery created "
            f"{duplicates} duplicate components")

    pair_audit.to_csv(stage / "recurrent_fusion_pair_audit.csv", index=False)
    absorption_events.to_csv(
        stage / "absorption_event_audit.csv", index=False)
    verdicts.to_csv(stage / "recurrent_fusion_verdicts.csv", index=False)
    applications.to_csv(
        stage / "recurrent_fusion_frame_applications.csv", index=False)
    changed = candidate != labels
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "pairs_audited": int(len(pair_audit)),
        "eligible_pairs": int(pair_audit.eligible.astype(bool).sum())
            if len(pair_audit) else 0,
        "applied_pairs": int(verdicts.outcome.eq("applied").sum())
            if len(verdicts) else 0,
        "absorption_events_audited": int(len(absorption_events)),
        "application_rows": int(len(applications)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return (candidate, unclaimed.copy(), pair_audit, absorption_events,
            verdicts, applications, metrics)


def _well_a2_issue001_exclusive_pair_seat_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Recover exclusive pair seats from current-run encounter evidence."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_exclusive_pair_seat_recovery", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted exclusive pair-seat recovery must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    exclusive_pair_seat_recovery.assert_target_free(params)

    stage = output_root / "61_exclusive_pair_seat_recovery" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    evidence = output_root / "26_latent_body_tracks" / "out"
    points_path = evidence / "latent_track_points.csv"
    encounters_path = evidence / "encounter_frames.csv"
    if enabled:
        missing = [path for path in (points_path, encounters_path)
                   if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "accepted exclusive pair-seat recovery requires current-run "
                "physical evidence: " + ", ".join(map(str, missing)))
        points = pd.read_csv(points_path)
        encounters = pd.read_csv(encounters_path)
        pair_audit, scored = exclusive_pair_seat_recovery.discover(
            labels, points, encounters, params)
        candidate, applications = exclusive_pair_seat_recovery.apply(
            labels, raw, pair_audit, scored, encounters, params)
    else:
        candidate = labels.copy()
        pair_audit = pd.DataFrame()
        applications = pd.DataFrame()

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "exclusive pair-seat recovery changed foreground segmentation")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "exclusive pair-seat recovery overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError(
            "exclusive pair-seat recovery changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "exclusive pair-seat recovery created "
            f"{duplicates} duplicate components")

    pair_audit.to_csv(stage / "exclusive_pair_seat_audit.csv", index=False)
    applications.to_csv(
        stage / "exclusive_pair_seat_applications.csv", index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "pairs_audited": int(len(pair_audit)),
        "eligible_pairs": int(pair_audit.eligible.astype(bool).sum())
            if len(pair_audit) else 0,
        "applied_pairs": int(applied[
            ["track_a", "track_b"]].drop_duplicates().shape[0])
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), pair_audit, applications, metrics


def _well_a2_issue002_reciprocal_two_seat_exchange(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair isolated one-frame reciprocal exchanges from field evidence."""
    del raw
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_reciprocal_two_seat_exchange", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted reciprocal two-seat exchange must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    reciprocal_two_seat_exchange.assert_target_free(params)

    stage = output_root / "62_reciprocal_two_seat_exchange" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if enabled and not points_path.is_file():
        raise FileNotFoundError(
            "accepted reciprocal two-seat exchange requires current-run "
            f"physical evidence: {points_path}")

    if enabled:
        points = pd.read_csv(points_path)
        audit, internal = reciprocal_two_seat_exchange.discover(
            labels, points, params)
        candidate, applications = reciprocal_two_seat_exchange.apply(
            labels, internal)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=reciprocal_two_seat_exchange.AUDIT_COLUMNS)
        applications = pd.DataFrame(columns=[
            "proposal_id", "frame", "track_a", "track_b", "owner_a",
            "owner_b", "applied", "changed_pixels", "reason"])

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "reciprocal two-seat exchange changed foreground segmentation")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "reciprocal two-seat exchange overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError(
            "reciprocal two-seat exchange changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "reciprocal two-seat exchange created "
            f"{duplicates} duplicate components")

    audit.to_csv(
        stage / "reciprocal_two_seat_exchange_audit.csv", index=False)
    applications.to_csv(
        stage / "reciprocal_two_seat_exchange_applications.csv", index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "reciprocal_transitions_audited": int(len(audit)),
        "eligible_exchanges": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_exchanges": int(len(applied)),
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, applications, metrics


def _well_a2_issue003_bounded_owner_excursion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair short duplicate-owner excursions from field evidence."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_bounded_owner_excursion", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted bounded owner excursion must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    bounded_owner_excursion.assert_target_free(params)

    stage = output_root / "63_bounded_owner_excursion" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if enabled and not points_path.is_file():
        raise FileNotFoundError(
            "accepted bounded owner excursion requires current-run "
            f"physical evidence: {points_path}")

    if enabled:
        points = pd.read_csv(points_path)
        audit, internal = bounded_owner_excursion.discover(
            labels, points, params)
        candidate, applications = bounded_owner_excursion.apply(
            labels, raw, points, internal, params)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=bounded_owner_excursion.AUDIT_COLUMNS)
        applications = pd.DataFrame(
            columns=bounded_owner_excursion.APPLICATION_COLUMNS)

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "bounded owner excursion changed foreground segmentation")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "bounded owner excursion overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError(
            "bounded owner excursion changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "bounded owner excursion created "
            f"{duplicates} duplicate components")

    audit.to_csv(stage / "bounded_owner_excursion_audit.csv", index=False)
    applications.to_csv(
        stage / "bounded_owner_excursion_applications.csv", index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "run_triples_audited": int(len(audit)),
        "eligible_excursions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_excursions": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, applications, metrics


def _well_a2_issue004_recurrent_isolated_alias(
        labels: np.ndarray, unclaimed: np.ndarray, config: Config,
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair recurrent isolated transient aliases from field evidence."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_recurrent_isolated_alias", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted recurrent isolated alias must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    recurrent_isolated_alias.assert_target_free(params)

    stage = output_root / "64_recurrent_isolated_alias" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    if enabled and not points_path.is_file():
        raise FileNotFoundError(
            "accepted recurrent isolated alias requires current-run "
            f"physical evidence: {points_path}")

    if enabled:
        points = pd.read_csv(points_path)
        audit, internal = recurrent_isolated_alias.discover(
            labels, points, params)
        candidate, applications = recurrent_isolated_alias.apply(
            labels, points, internal)
    else:
        candidate = labels.copy()
        audit = pd.DataFrame(columns=recurrent_isolated_alias.AUDIT_COLUMNS)
        applications = pd.DataFrame(
            columns=recurrent_isolated_alias.APPLICATION_COLUMNS)

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "recurrent isolated alias changed foreground segmentation")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "recurrent isolated alias overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError(
            "recurrent isolated alias changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "recurrent isolated alias created "
            f"{duplicates} duplicate components")

    audit.to_csv(stage / "recurrent_isolated_alias_audit.csv", index=False)
    applications.to_csv(
        stage / "recurrent_isolated_alias_applications.csv", index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "recurrent_clusters_audited": int(len(audit)),
        "eligible_clusters": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_clusters": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "input_identity_count": int(len(before_ids)),
        "output_identity_count": int(len(after_ids)),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "label_component_ledger_exact": True,
        "new_duplicate_components": int(duplicates),
        "parameters": params,
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), audit, applications, metrics


def _latest_current_audit(output_root: Path,
                          relative_paths: tuple[str, ...]) -> Path | None:
    """Return the newest applicable audit produced inside this run only."""
    for relative in relative_paths:
        path = output_root / relative
        if path.is_file():
            return path
    return None


def _well_a5_issue002_distinct_history_core_preservation(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Preserve an established core in a safe terminal-owner diversion."""
    configured = config.values["accepted_postprocessing"].get(
        "field_wide_distinct_history_core_preservation", {})
    enabled = bool(configured.get("enabled", False))
    if configured.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "accepted distinct-history core preservation must be field-wide")
    params = deepcopy(configured.get("parameters", {}))
    params["targeting_mode"] = "field_wide_discovery"
    distinct_history_core_preservation.assert_target_free(configured)
    distinct_history_core_preservation.assert_target_free(params)

    stage = output_root / "65_distinct_history_core_preservation" / "out"
    stage.mkdir(parents=True, exist_ok=True)
    points_path = (output_root / "26_latent_body_tracks" / "out" /
                   "latent_track_points.csv")
    application_audit = _latest_current_audit(output_root, (
        "64_recurrent_isolated_alias/out/application_audit.csv",
        "63_bounded_owner_excursion/out/application_audit.csv",
        "62_reciprocal_two_seat_exchange/out/application_audit.csv",
        "61_exclusive_pair_seat_recovery/out/application_audit.csv",
        "60_recurrent_two_body_fusion/out/application_audit.csv",
        "46_transient_misownership_recovery/out/application_audit.csv",
    ))
    conflicted_audit = _latest_current_audit(output_root, (
        "64_recurrent_isolated_alias/out/conflicted_lineage_audit.csv",
        "63_bounded_owner_excursion/out/conflicted_lineage_audit.csv",
        "62_reciprocal_two_seat_exchange/out/conflicted_lineage_audit.csv",
        "61_exclusive_pair_seat_recovery/out/conflicted_lineage_audit.csv",
        "60_recurrent_two_body_fusion/out/conflicted_lineage_audit.csv",
        "59_lineage_capacity_recovery/out/conflicted_lineage_audit.csv",
        "58_terminal_multi_owner_reservation/out/conflicted_lineage_audit.csv",
        "57_area_continuity_recovery/out/conflicted_lineage_audit.csv",
        "56_established_lineage_reservation/out/conflicted_lineage_audit.csv",
        "52_conflicted_dim_lineage_isolation/out/conflicted_lineage_audit.csv",
    ))
    if enabled and not points_path.is_file():
        raise FileNotFoundError(
            "accepted distinct-history core preservation requires current-run "
            f"physical evidence: {points_path}")
    if enabled and application_audit is None:
        raise FileNotFoundError(
            "accepted distinct-history core preservation requires a "
            "current-run application audit")

    if enabled:
        params["application_audit_path"] = str(application_audit)
        if conflicted_audit is not None:
            params["conflicted_lineage_audit_path"] = str(conflicted_audit)
        points = pd.read_csv(points_path)
        candidate, proposals, frames = \
            distinct_history_core_preservation.discover_and_apply(
                labels, raw, points, params)
    else:
        candidate = labels.copy()
        proposals = pd.DataFrame()
        frames = pd.DataFrame()

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "distinct-history core preservation changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError(
            "distinct-history core preservation overlaps assigned and unclaimed")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if after_ids != before_ids:
        raise AssertionError(
            "distinct-history core preservation changed the active identity set")
    duplicates = owner_consensus.count_new_duplicate_components(
        labels, candidate)
    if duplicates:
        raise AssertionError(
            "distinct-history core preservation created "
            f"{duplicates} duplicate components")

    proposals.to_csv(stage / "distinct_history_proposals.csv", index=False)
    frames.to_csv(stage / "distinct_history_frames.csv", index=False)
    if application_audit is not None:
        shutil.copyfile(application_audit, stage / "application_audit.csv")
    if conflicted_audit is not None:
        shutil.copyfile(
            conflicted_audit, stage / "conflicted_lineage_audit.csv")
    else:
        (stage / "conflicted_lineage_audit.csv").write_text(
            "proposal_id,physical_track,assigned_identity,outcome,reason,"
            "changed_pixels,changed_frames\n", encoding="utf-8")
    changed = candidate != labels
    applied = proposals[proposals.applied.fillna(False).astype(bool)] \
        if len(proposals) else proposals
    protected_tracks, protected_identities = \
        distinct_history_core_preservation.protected_assignments(params)
    metrics = {
        "version": configured.get("version", "unversioned"),
        "enabled": enabled,
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "review_cases")},
        "transitions_audited": int(len(proposals)),
        "eligible_transitions": int(
            proposals.eligible.fillna(False).astype(bool).sum())
            if len(proposals) else 0,
        "applied_transitions": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_changed_pixels": 0,
        "unclaimed_changed_pixels": 0,
        "identity_set_exact": True,
        "new_duplicate_components": int(duplicates),
        "accepted_protected_tracks": int(len(protected_tracks)),
        "accepted_protected_identities": int(len(protected_identities)),
        "globally_conflicted_proposals": int(
            proposals.global_atomic_conflict.fillna(False).astype(bool).sum())
            if len(proposals) else 0,
        "parameters": {key: value for key, value in params.items()
                       if not str(key).endswith("_path")},
    }
    (stage / "producer_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return candidate, unclaimed.copy(), proposals, frames, metrics


def _well_a5_issue003_branched_motion_lineage_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, config: Config,
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Recover brief owner excursions on proved physical lineages."""
    return branched_motion_lineage_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_a5_issue003_complete_flip_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, config: Config,
        output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recover complete successor flips on proved physical lineages."""
    return complete_flip_recovery_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_a5_issue005_projection_tolerant_seat_preservation(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Preserve proved pre-contact seats while retaining projections."""
    return projection_tolerant_seat_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_a5_issue006_retroactive_successor_separation(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Backfill movie-novel successors on proved physical lineages."""
    return retroactive_successor_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_a5_issue004_terminal_two_core_split_inheritance(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Preserve two established seats through resolved and terminal splits."""
    return terminal_two_core_split_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b1_issue001_bracketed_established_owner_invasion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore bookending owners when an invader retains a proved seat."""
    return bracketed_established_owner_invasion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue001_terminal_projection_diversion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair terminal three-seat diversions from current-run evidence."""
    return terminal_projection_diversion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue002_delayed_projection_reclaim(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Separate asymmetric overlaps before a delayed owner reclaim."""
    return delayed_projection_reclaim_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue003_boundary_owner_excursion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore field-wide boundary-anchored owner excursions."""
    return boundary_owner_excursion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue004_terminal_two_seat_assimilation(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Preserve two raw-supported seats through terminal assimilation."""
    return terminal_two_seat_assimilation_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue005_bracketed_mixed_owner_flash(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore brief, fully bracketed mixed-owner excursions atomically."""
    return bracketed_mixed_owner_flash_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue006_dominant_seat_projection_diversion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Restore dominant seats while explicitly accounting for projections."""
    return dominant_seat_projection_diversion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue007_subresolution_point_artifact_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Remove only fully artifact-explained events from a fresh catalogue."""
    return subresolution_point_artifact_calibration_integration.run(
        labels, unclaimed, raw, points, thresholds, events, members,
        config.values, output_root)


def _well_b2_issue008_short_owned_companion_projection_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Suppress uniquely proved short same-owner projection events."""
    return short_owned_companion_projection_calibration_integration.run(
        labels, unclaimed, points, events, members, config.values, output_root)


def _well_b2_issue009_ownerless_cohort_latent_gap_completion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Complete only fully supported gaps in field-discovered cohorts."""
    return ownerless_cohort_latent_gap_completion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue011_recurrent_same_owner_branch_swarm_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Suppress only fully proved recurrent one-owner branch-swarm events."""
    return recurrent_same_owner_branch_swarm_calibration_integration.run(
        labels, unclaimed, points, events, members, config.values, output_root)


def _well_b2_issue012_aggregate_shared_core_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reclassify only completely evidenced aggregate shared-core events."""
    return aggregate_shared_core_calibration_integration.run(
        labels, unclaimed, raw, points, events, members,
        config.values, output_root)


def _well_b2_issue013_weak_terminal_reference_relay_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Reclassify only fully proved weak terminal reference relays."""
    return weak_terminal_reference_relay_calibration_integration.run(
        labels, unclaimed, raw, points, events, members,
        config.values, output_root)


def _well_b2_issue014_single_owner_multireference_relay_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Reclassify only fully proved single-owner multireference relays."""
    return single_owner_multireference_relay_calibration_integration.run(
        labels, unclaimed, points, events, members,
        config.values, output_root)


def _well_b2_issue015_event_local_subresolution_artifact_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove only fully proved event-local subresolution artifacts."""
    return event_local_subresolution_artifact_calibration_integration.run(
        labels, unclaimed, raw, points, thresholds, events, members,
        config.values, output_root)


def _well_b2_issue016_short_ownerless_subcellular_reference_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove only short ownerless references with subcellular raw cores."""
    return short_ownerless_subcellular_reference_calibration_integration.run(
        labels, unclaimed, raw, points, thresholds, events, members,
        config.values, output_root)


def _well_b2_issue018_recording_start_two_seat_inheritance(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore a short branch's owner across a separable boundary encounter."""
    return recording_start_two_seat_inheritance_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b2_issue026_detached_projection_owner_relay_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Exclude only fully evidenced detached projection relay constituents."""
    return detached_projection_owner_relay_calibration_integration.run(
        labels, unclaimed, points, events, members,
        config.values, output_root)


def _well_b2_issue028_right_censored_novel_body_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Exclude fully evidenced novel bodies censored by the recording end."""
    return right_censored_novel_body_calibration_integration.run(
        labels, unclaimed, raw, points, thresholds, events, members,
        config.values, output_root)


def _well_b3_issue010_detached_fading_projection_calibration(
        labels: np.ndarray, unclaimed: np.ndarray,
        points: pd.DataFrame, events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Exclude only fully evidenced detached fading projections."""
    return detached_fading_projection_calibration_integration.run(
        labels, unclaimed, points, events, members,
        config.values, output_root)


def _well_b3_issue012_explained_projection_event_calibration(
        labels: np.ndarray, unclaimed: np.ndarray,
        events: pd.DataFrame, members: pd.DataFrame,
        applications: pd.DataFrame, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Exclude events already explained by an accepted producer ledger."""
    return explained_projection_event_calibration_integration.run(
        labels, unclaimed, events, members, applications,
        config.values, output_root)


def _well_b3_issue002_gap_tolerant_reciprocal_seat_exchange(
        labels: np.ndarray, unclaimed: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore only fully proved reciprocal owner exchanges on two seats."""
    return gap_tolerant_reciprocal_seat_exchange_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_b3_issue004_reconnected_companion_merge_partition(
        labels: np.ndarray, unclaimed: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Partition only fully proved two-seat reconnection merge gaps."""
    return reconnected_companion_merge_partition_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_b3_issue005_component_continuity_duplicate_takeover(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore only directly continuous resident bodies after takeover."""
    return component_continuity_duplicate_takeover_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue006_right_censored_seat_partition(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Retain established seats through a right-censored terminal collapse."""
    return right_censored_seat_partition_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue007_right_censored_reciprocal_exchange(
        labels: np.ndarray, unclaimed: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Repair decisive right-censored reciprocal owner exchanges."""
    return right_censored_reciprocal_exchange_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_b3_issue008_persistent_single_owner_flash(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair persistent single-owner flashes with physical evidence."""
    return persistent_single_owner_flash_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue012_delayed_owner_projection_flash(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Repair proved delayed-owner projection flashes field-wide."""
    return delayed_owner_projection_flash_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue013_anchored_projection_owner_relay(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Restore projection ownership from unique durable-anchor history."""
    return anchored_projection_owner_relay_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue013_ephemeral_speckle_alias_retirement(
        labels: np.ndarray, unclaimed: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Retire only aliases whose complete field is producer-proved."""
    return ephemeral_speckle_alias_retirement_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_b3_issue013_anchored_projection_raw_gap_completion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        relay_audit: pd.DataFrame, relay_applications: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Add only unassigned positive-signal cores in a proved relay."""
    return anchored_projection_raw_gap_completion_integration.run(
        labels, unclaimed, raw, relay_audit, relay_applications,
        config.values, output_root)


def _well_b3_issue014_terminal_projection_chain_completion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Complete only fully proved terminal projection-chain raw cores."""
    return terminal_projection_chain_completion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue015_conserved_multi_anchor_projection_calibration(
        labels: np.ndarray, unclaimed: np.ndarray,
        points: pd.DataFrame, events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove only fully proved conserved-anchor encounter score alerts."""
    return conserved_multi_anchor_projection_calibration_integration.run(
        labels, unclaimed, points, events, members,
        config.values, output_root)


def _well_b3_issue016_body_scale_ownerless_allocation(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Allocate only durable, isolated and body-scale ownerless tracks."""
    return body_scale_ownerless_allocation_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue017_terminal_boundary_vanished_seat(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Partition only complete-field, right-censored vanished seats."""
    return terminal_boundary_vanished_seat_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue017_terminal_boundary_event_calibration(
        labels: np.ndarray, unclaimed: np.ndarray,
        events: pd.DataFrame, members: pd.DataFrame,
        applications: pd.DataFrame, frames: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove alerts resolved by a complete terminal-seat producer ledger."""
    return terminal_boundary_event_calibration_integration.run(
        labels, unclaimed, events, members, applications, frames,
        config.values, output_root)


def _well_b3_issue018_fragmented_subcellular_episode_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        raw: np.ndarray, thresholds: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove only field-proved fragmented subcellular-reference alerts."""
    return fragmented_subcellular_episode_calibration_integration.run(
        labels, unclaimed, points, raw, thresholds, events, members,
        config.values, output_root)


def _well_b3_issue019_bracketed_ownerless_seat_completion(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Complete only uniquely bracketed raw-visible ownerless seats."""
    return bracketed_ownerless_seat_completion_integration.run(
        labels, unclaimed, raw, config.values, output_root)


def _well_b3_issue020_fragmented_encounter_lineage_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join only score fragments sharing one durable physical encounter."""
    return fragmented_encounter_lineage_calibration_integration.run(
        labels, unclaimed, points, events, members,
        config.values, output_root)


def _well_b3_issue021_temporal_core_path_lineage(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        events: pd.DataFrame, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Allocate complete left-censored lineages after fresh scoring."""
    return temporal_core_path_lineage_integration.run(
        labels, unclaimed, raw, events, config.values, output_root)


def _well_b3_issue022_ownerless_reference_handoff_calibration(
        labels: np.ndarray, unclaimed: np.ndarray, points: pd.DataFrame,
        raw: np.ndarray, thresholds: pd.DataFrame,
        events: pd.DataFrame, members: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Remove only field-proved fragile ownerless handoff alerts."""
    return ownerless_reference_handoff_calibration_integration.run(
        labels, unclaimed, points, raw, thresholds, events, members,
        config.values, output_root)


def _well_b3_issue023_asymmetric_fusion_area_flip_recovery(
        labels: np.ndarray, unclaimed: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair only field-proved fusion loss and reciprocal area flips."""
    return asymmetric_fusion_area_flip_recovery_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_b3_issue024_recurrent_dominant_body_relay(
        labels: np.ndarray, unclaimed: np.ndarray,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair only field-proved closed recurrent three-owner relays."""
    return recurrent_dominant_body_relay_integration.run(
        labels, unclaimed, config.values, output_root)


def _well_b3_issue025_established_seat_cycle_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Repair only field-proved directed multi-owner seat cycles."""
    return established_seat_cycle_recovery_integration.run(
        labels, unclaimed, raw, points, thresholds,
        config.values, output_root)


def _well_b3_issue026_dormant_seat_successor_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Consolidate only uniquely proved movie-new successor lineages."""
    return dormant_seat_successor_recovery_integration.run(
        labels, unclaimed, raw, points, thresholds,
        config.values, output_root)


def _well_b3_issue027_staggered_fusion_exchange_recovery(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, thresholds: pd.DataFrame,
        config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame, dict[str, Any]]:
    """Repair only field-proved staggered reciprocal owner exchanges."""
    return staggered_fusion_exchange_recovery_integration.run(
        labels, unclaimed, raw, points, thresholds,
        config.values, output_root)


def _well_b4_issue001_large_persistent_reference_relay(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Preserve field-proved large stable references through owner relays."""
    return large_persistent_reference_relay_integration.run(
        labels, unclaimed, raw, points, config.values, output_root)


def _well_b4_issue002_dormant_owner_reciprocal_partition(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, config: Config, output_root: Path,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   dict[str, Any]]:
    """Repair field-proved dormant takeovers as reciprocal partitions."""
    return dormant_owner_reciprocal_partition_integration.run(
        labels, unclaimed, raw, points, config.values, output_root)


def _well_b4_issue003_conservative_episode_ownership(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, config: Config, output_root: Path):
    """Reconcile complete ownership episodes without suppressing detections."""
    return conservative_episode_ownership_integration.run(
        labels, unclaimed, raw, points, config.values, output_root)


def _well_b4_issue004_original_body_continuity(
        labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
        points: pd.DataFrame, config: Config, output_root: Path):
    """Preserve a durable initial body using independent observed raw cores."""
    return original_body_continuity_integration.run(
        labels, unclaimed, raw, points, config.values, output_root)


def run_accepted_history(
        config: Config, stem: str, run_name: str, sources: dict[str, Path],
        m20_labels: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, anchor_t: int,
        output_root: Path,
        source_run_name: str | None = None,
        history_inputs: AcceptedHistoryInputs | None = None,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the complete accepted labels from current-run inputs only."""
    output_root.mkdir(parents=True, exist_ok=True)
    source_run = source_run_name or run_name
    interval = float(config.values["frame_interval_min"])
    raw_full = load_stack(sources["registered_raw"])
    lag_full = load_stack(sources["lag_float"])
    neutral_full = load_stack(sources["neutral_tracks"])
    motion_full = load_stack(sources["motion_composite"])
    if history_inputs is None:
        raise ValueError(
            "accepted-history calculation requires explicit current-run inputs")
    physical, reservation, pre_reservation, host_audit, motion_reserved = (
        history_inputs.required())

    issue005_checkpoint = output_root / "01_issue005.tif"
    if issue005_checkpoint.is_file():
        # A failed immutable stage may resume its own freshly generated work.
        labels = load_stack(issue005_checkpoint)
    else:
        labels = _issue005(
            config, physical, reservation, raw_full, lag_full, observations,
            observation_table, anchor_t, motion_reserved)
        _save(issue005_checkpoint, labels, interval)
    issue006_checkpoint = output_root / "02_issue006.tif"
    if issue006_checkpoint.is_file():
        labels = load_stack(issue006_checkpoint)
    else:
        labels = _issue006(labels, raw_full, lag_full)
        _save(issue006_checkpoint, labels, interval)
    issue007_checkpoint = output_root / "03_issue007.tif"
    if issue007_checkpoint.is_file():
        labels = load_stack(issue007_checkpoint)
    else:
        labels = _issue007(labels, pre_reservation, lag_full, host_audit)
        _save(issue007_checkpoint, labels, interval)

    labels = labels[2:].copy()
    raw = raw_full[2:2 + len(labels)]
    lag = align_transition_stack(lag_full, len(labels), 2)
    issue008_checkpoint = output_root / "04_issue008.tif"
    if issue008_checkpoint.is_file():
        labels = load_stack(issue008_checkpoint)
    else:
        labels = _issue008(labels, raw, lag, stem, output_root)
        _save(issue008_checkpoint, labels, interval)
    issue009_checkpoint = output_root / "05_issue009.tif"
    authority_checkpoint = (
        output_root / "12_movie_authority/out/movie_wide_authority.csv")
    if issue009_checkpoint.is_file() and authority_checkpoint.is_file():
        labels = load_stack(issue009_checkpoint)
        authority = pd.read_csv(authority_checkpoint)
    else:
        labels, authority = _issue009(labels, raw, lag, stem, output_root)
        _save(issue009_checkpoint, labels, interval)
    issue009 = labels.copy()
    issue010_checkpoint = output_root / "06_issue010.tif"
    issue010_unclaimed = output_root / "06_issue010_unclaimed.tif"
    foreground = issue009 > 0
    if issue010_checkpoint.is_file() and issue010_unclaimed.is_file():
        labels = load_stack(issue010_checkpoint)
        unclaimed = load_stack(issue010_unclaimed)
    else:
        labels, unclaimed, foreground = _issue010(labels, authority, lag)
        _save(issue010_checkpoint, labels, interval)
        _save(issue010_unclaimed, unclaimed, interval)
    issue010 = labels.copy()
    issue011_checkpoint = output_root / "07_issue011.tif"
    issue011_unclaimed = output_root / "07_issue011_unclaimed.tif"
    if issue011_checkpoint.is_file() and issue011_unclaimed.is_file():
        labels = load_stack(issue011_checkpoint)
        unclaimed = load_stack(issue011_unclaimed)
    else:
        labels, unclaimed = _issue011(labels, unclaimed, raw, stem, output_root)
        _save(issue011_checkpoint, labels, interval)
        _save(issue011_unclaimed, unclaimed, interval)
    issue012_checkpoint = output_root / "08_issue012.tif"
    issue012_unclaimed = output_root / "08_issue012_unclaimed.tif"
    if issue012_checkpoint.is_file() and issue012_unclaimed.is_file():
        labels = load_stack(issue012_checkpoint)
        unclaimed = load_stack(issue012_unclaimed)
    else:
        labels, unclaimed = _issue012(
            labels, unclaimed, issue010, raw, lag)
        _save(issue012_checkpoint, labels, interval)
        _save(issue012_unclaimed, unclaimed, interval)

    issue017_checkpoint = output_root / "11_issue017_rebase.tif"
    issue017_unclaimed = output_root / "11_issue017_rebase_unclaimed.tif"
    if issue017_checkpoint.is_file() and issue017_unclaimed.is_file():
        labels = load_stack(issue017_checkpoint)
        unclaimed = load_stack(issue017_unclaimed)
    else:
        labels, unclaimed, registry = _seat_identity(
            labels, unclaimed, raw, stem, output_root)
        _save(output_root / "09_seat_identity.tif", labels, interval)
        _save(output_root / "09_seat_identity_unclaimed.tif", unclaimed, interval)
        labels, unclaimed, registry, _, manual_review = _general_blob_residency(
            labels, unclaimed, raw, registry)
        manual_review.to_csv(
            output_root / "18_seat_identity/out/"
            "blob_residency_manual_review.csv", index=False)
        _save(output_root / "10_blob_residency.tif", labels, interval)
        _save(output_root / "10_blob_residency_unclaimed.tif", unclaimed, interval)
        issue015_labels = labels.copy()
        issue015_unclaimed = unclaimed.copy()

        alias = reconcile_physical_lineages(
            labels, unclaimed, observations, observation_table,
            neutral_full, lag,
            _general("R05_generalise_full_lineage_alias"))
        alias_ids: set[int] = set()
        if len(alias.alias_audit):
            accepted_aliases = alias.alias_audit[
                alias.alias_audit.accepted.astype(bool)]
            alias_ids.update(accepted_aliases.alias_identity.astype(int))
            alias_ids.update(accepted_aliases.canonical_identity.astype(int))
        continuity_labels, continuity_unclaimed = _continuity_repair(
            issue015_labels, issue015_unclaimed, raw_full, lag_full,
            registry, alias_ids, output_root)
        alias_delta = alias.labels != issue015_labels
        continuity_delta = continuity_labels != issue015_labels
        alias_unclaimed_delta = alias.unclaimed != issue015_unclaimed
        continuity_unclaimed_delta = (
            continuity_unclaimed != issue015_unclaimed)
        if np.any(alias_delta & continuity_delta &
                  (alias.labels != continuity_labels)):
            raise AssertionError(
                "field-derived alias and continuity repairs conflict")
        if np.any(alias_unclaimed_delta & continuity_unclaimed_delta &
                  (alias.unclaimed != continuity_unclaimed)):
            raise AssertionError(
                "field-derived alias and continuity unclaimed ledgers conflict")
        labels = alias.labels.copy()
        labels[continuity_delta] = continuity_labels[continuity_delta]
        unclaimed = alias.unclaimed.copy()
        unclaimed[continuity_unclaimed_delta] = continuity_unclaimed[
            continuity_unclaimed_delta]
        _save(issue017_checkpoint, labels, interval)
        _save(issue017_unclaimed, unclaimed, interval)

    labels, unclaimed = _issue023_tail(labels, unclaimed, raw, lag)
    _save(output_root / "12_issue023_generalized.tif", labels, interval)
    _save(output_root / "12_issue023_generalized_unclaimed.tif",
          unclaimed, interval)
    stationary_cfg = config.values["accepted_postprocessing"].get(
        "field_wide_stationary_reconciliation", {})
    stationary_params = reconciliation_params(
        stationary_cfg.get("parameters", {}))
    stationary_params["enabled"] = bool(stationary_cfg.get("enabled", True))
    _assert_identity_blind_params(stationary_params)
    if stationary_params["enabled"]:
        labels, _, _, events, _ = discover_stationary_takeovers(
            labels, raw, stationary_params)
    else:
        events = pd.DataFrame()
    labels, unclaimed, short_blank_audit, short_blank_metrics = \
        _issue027_short_blank_recovery(
            labels, unclaimed, raw, lag, config, output_root)
    _save(output_root / "13_issue027_short_blank_recovery.tif",
          labels, interval)
    _save(output_root / "13_issue027_short_blank_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, small_cell_audit, small_cell_metrics = \
        _issue029_small_cell_blank_recovery(
            labels, unclaimed, raw, lag, config, output_root)
    _save(output_root / "14_issue029_small_cell_blank_recovery.tif",
          labels, interval)
    _save(output_root / "14_issue029_small_cell_blank_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, moderate_gap_audit, moderate_gap_metrics = \
        _issue030_moderate_gap_recovery(
            labels, unclaimed, raw, lag, config, output_root)
    _save(output_root / "15_issue030_moderate_gap_recovery.tif",
          labels, interval)
    _save(output_root / "15_issue030_moderate_gap_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, owner_audit, owner_metrics = \
        _issue032_owner_consensus(
            labels, unclaimed, raw, motion_full, config, output_root)
    _save(output_root / "16_issue032_owner_consensus.tif", labels, interval)
    _save(output_root / "16_issue032_owner_consensus_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, merge_audit, merge_metrics = \
        _issue033_separable_merge_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "17_issue033_separable_merge_recovery.tif",
          labels, interval)
    _save(output_root / "17_issue033_separable_merge_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, start_split_audit, start_split_metrics = \
        _issue034_recording_start_split(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "18_issue034_recording_start_split.tif",
          labels, interval)
    _save(output_root / "18_issue034_recording_start_split_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, residency_audit, residency_metrics = \
        _issue035_merge_residency(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "19_issue035_merge_residency.tif",
          labels, interval)
    _save(output_root / "19_issue035_merge_residency_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, body_identity_audit, body_identity_metrics = \
        _issue036_persistent_body_identity(
            labels, unclaimed, config, output_root)
    _save(output_root / "20_issue036_persistent_body_identity.tif",
          labels, interval)
    _save(output_root / "20_issue036_persistent_body_identity_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, takeover_audit, takeover_metrics = \
        _issue037_resident_takeover(
            labels, unclaimed, config, output_root)
    _save(output_root / "21_issue037_resident_takeover.tif",
          labels, interval)
    _save(output_root / "21_issue037_resident_takeover_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, ownerless_audit, ownerless_metrics = \
        _issue038_ownerless_body_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "22_issue038_ownerless_body_recovery.tif",
          labels, interval)
    _save(output_root / "22_issue038_ownerless_body_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, ownerless_cohort_audit, ownerless_cohort_metrics = \
        _issue039_ownerless_cohort_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "23_issue039_ownerless_cohort_recovery.tif",
          labels, interval)
    _save(output_root / "23_issue039_ownerless_cohort_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, bracketed_audit, bracketed_metrics = \
        _issue041_bracketed_encounter_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "24_issue041_bracketed_encounter_recovery.tif",
          labels, interval)
    _save(output_root / "24_issue041_bracketed_encounter_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, relay_audit, relay_frame_audit, relay_metrics = \
        _issue042_recurrent_exclusive_owner_relay(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "25_issue042_recurrent_exclusive_owner_relay.tif",
          labels, interval)
    _save(output_root /
          "25_issue042_recurrent_exclusive_owner_relay_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, blip_audit, blip_frame_audit, blip_metrics = \
        _issue043_isolated_owner_blip_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "26_issue043_isolated_owner_blip_recovery.tif",
          labels, interval)
    _save(output_root /
          "26_issue043_isolated_owner_blip_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, two_seat_cohorts, two_seat_applications, \
        two_seat_frame_audit, two_seat_metrics = \
        _issue045_atomic_two_seat_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "27_issue045_atomic_two_seat_recovery.tif",
          labels, interval)
    _save(output_root / "27_issue045_atomic_two_seat_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, duplicate_release_audit, duplicate_release_metrics = \
        _issue046_recording_start_duplicate_release(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "28_issue046_recording_start_duplicate_release.tif",
          labels, interval)
    _save(output_root /
          "28_issue046_recording_start_duplicate_release_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, late_owner_proposals, late_owner_applications, \
        late_owner_frame_audit, late_owner_metrics = \
        _issue048_late_owner_backfill(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "29_issue048_late_owner_backfill.tif",
          labels, interval)
    _save(output_root / "29_issue048_late_owner_backfill_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, handoff_proposals, handoff_applications, \
        handoff_frame_audit, handoff_metrics = \
        _issue051_bracketed_handoff_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "30_issue051_bracketed_handoff_recovery.tif",
          labels, interval)
    _save(output_root /
          "30_issue051_bracketed_handoff_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, post_split_proposals, post_split_applications, \
        post_split_frame_audit, post_split_metrics = \
        _issue055_post_split_backfill(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "31_issue055_post_split_backfill.tif",
          labels, interval)
    _save(output_root /
          "31_issue055_post_split_backfill_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, concurrent_invasion_proposals, \
        concurrent_invasion_applications, concurrent_invasion_frame_audit, \
        concurrent_invasion_metrics = \
        _issue056_concurrent_duplicate_invasion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "32_issue056_concurrent_duplicate_invasion.tif",
          labels, interval)
    _save(output_root /
          "32_issue056_concurrent_duplicate_invasion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, post_split_excursion_audit, \
        post_split_excursion_metrics = \
        _issue063_post_split_excursion_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root / "33_issue063_post_split_excursion_recovery.tif",
          labels, interval)
    _save(output_root /
          "33_issue063_post_split_excursion_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, transient_misownership_proposals, \
        transient_misownership_audit, transient_misownership_points, \
        transient_misownership_metrics = \
        _issue066_transient_misownership_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "35_issue066_transient_misownership_recovery.tif",
          labels, interval)
    _save(output_root /
          "35_issue066_transient_misownership_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, isolated_lifetime_proposals, \
        isolated_lifetime_applications, isolated_lifetime_frame_audit, \
        isolated_lifetime_points, isolated_lifetime_metrics = \
        _issue070_isolated_unowned_lifetime_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "36_issue070_isolated_unowned_lifetime_recovery.tif",
          labels, interval)
    _save(output_root /
          "36_issue070_isolated_unowned_lifetime_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, boundary_blip_proposals, \
        boundary_blip_applications, boundary_blip_frame_audit, \
        boundary_blip_metrics = _issue071_boundary_owner_blip_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "37_issue071_boundary_owner_blip_recovery.tif",
          labels, interval)
    _save(output_root /
          "37_issue071_boundary_owner_blip_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, terminal_convergence_proposals, \
        terminal_convergence_applications, terminal_convergence_frame_audit, \
        terminal_convergence_points, terminal_convergence_metrics = \
        _issue078_foreign_owner_terminal_convergence(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "39_issue078_foreign_owner_terminal_convergence.tif",
          labels, interval)
    _save(output_root /
          "39_issue078_foreign_owner_terminal_convergence_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, companion_assimilation_proposals, \
        companion_assimilation_applications, companion_assimilation_frame_audit, \
        companion_assimilation_points, companion_assimilation_metrics = \
        _issue080_terminal_companion_assimilation(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "40_issue080_terminal_companion_assimilation.tif",
          labels, interval)
    _save(output_root /
          "40_issue080_terminal_companion_assimilation_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, cross_reference_proposals, \
        cross_reference_applications, cross_reference_frame_audit, \
        cross_reference_points, cross_reference_metrics = \
        _issue089_cross_reference_owner_relay(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "41_issue089_cross_reference_owner_relay.tif",
          labels, interval)
    _save(output_root /
          "41_issue089_cross_reference_owner_relay_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, boundary_cycle_edges, boundary_cycles, \
        boundary_cycle_members, boundary_cycle_anchors, \
        boundary_cycle_applications, boundary_cycle_metrics = \
        _issue091_recording_boundary_owner_cycle(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "42_issue091_recording_boundary_owner_cycle.tif",
          labels, interval)
    _save(output_root /
          "42_issue091_recording_boundary_owner_cycle_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, duplicate_soma_before, \
        duplicate_projection_applications, duplicate_lineage_proposals, \
        duplicate_lineage_applications, duplicate_lineage_frame_audit, \
        duplicate_lineage_metrics = _issue092_complete_duplicate_lineage(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "43_issue092_complete_duplicate_lineage.tif",
          labels, interval)
    _save(output_root /
          "43_issue092_complete_duplicate_lineage_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, isolated_pair_audit, isolated_pair_metrics = \
        _well_issue001_isolated_pair_encounter_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "44_well_issue001_isolated_pair_encounter_recovery.tif",
          labels, interval)
    _save(output_root /
          "44_well_issue001_isolated_pair_encounter_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, recurrent_fusion_pairs, \
        recurrent_fusion_absorptions, recurrent_fusion_verdicts, \
        recurrent_fusion_applications, recurrent_fusion_metrics = \
        _well_issue004_recurrent_two_body_fusion_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "45_well_issue004_recurrent_two_body_fusion_recovery.tif",
          labels, interval)
    _save(output_root /
          "45_well_issue004_recurrent_two_body_fusion_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, exclusive_pair_seats, \
        exclusive_pair_seat_applications, exclusive_pair_seat_metrics = \
        _well_a2_issue001_exclusive_pair_seat_recovery(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "46_well_a2_issue001_exclusive_pair_seat_recovery.tif",
          labels, interval)
    _save(output_root /
          "46_well_a2_issue001_exclusive_pair_seat_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, reciprocal_exchange_audit, \
        reciprocal_exchange_applications, reciprocal_exchange_metrics = \
        _well_a2_issue002_reciprocal_two_seat_exchange(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "47_well_a2_issue002_reciprocal_two_seat_exchange.tif",
          labels, interval)
    _save(output_root /
          "47_well_a2_issue002_reciprocal_two_seat_exchange_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, bounded_excursion_audit, \
        bounded_excursion_applications, bounded_excursion_metrics = \
        _well_a2_issue003_bounded_owner_excursion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "48_well_a2_issue003_bounded_owner_excursion.tif",
          labels, interval)
    _save(output_root /
          "48_well_a2_issue003_bounded_owner_excursion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, recurrent_alias_audit, \
        recurrent_alias_applications, recurrent_alias_metrics = \
        _well_a2_issue004_recurrent_isolated_alias(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "49_well_a2_issue004_recurrent_isolated_alias.tif",
          labels, interval)
    _save(output_root /
          "49_well_a2_issue004_recurrent_isolated_alias_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, distinct_history_proposals, \
        distinct_history_frames, distinct_history_metrics = \
        _well_a5_issue002_distinct_history_core_preservation(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "50_well_a5_issue002_distinct_history_core_preservation.tif",
          labels, interval)
    _save(output_root /
          "50_well_a5_issue002_distinct_history_core_preservation_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, branched_lineage_audit, \
        branched_lineage_applications, branched_lineage_metrics = \
        _well_a5_issue003_branched_motion_lineage_recovery(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "51_well_a5_issue003_branched_motion_lineage_recovery.tif",
          labels, interval)
    _save(output_root /
          "51_well_a5_issue003_branched_motion_lineage_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, complete_flip_audit, \
        complete_flip_applications, complete_flip_duplicate_proof, \
        complete_flip_metrics = \
        _well_a5_issue003_complete_flip_recovery(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "52_well_a5_issue003_complete_flip_recovery.tif",
          labels, interval)
    _save(output_root /
          "52_well_a5_issue003_complete_flip_recovery_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, projection_seat_audit, projection_seat_frames, \
        projection_seat_projections, projection_seat_duplicate_proof, \
        projection_seat_metrics = \
        _well_a5_issue005_projection_tolerant_seat_preservation(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "53_well_a5_issue005_projection_tolerant_seat_preservation.tif",
          labels, interval)
    _save(output_root /
          "53_well_a5_issue005_projection_tolerant_seat_preservation_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, retroactive_successor_audit, \
        retroactive_successor_frames, \
        retroactive_successor_duplicate_proof, \
        retroactive_successor_metrics = \
        _well_a5_issue006_retroactive_successor_separation(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "54_well_a5_issue006_retroactive_successor_separation.tif",
          labels, interval)
    _save(output_root /
          "54_well_a5_issue006_retroactive_successor_separation_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, terminal_split_audit, terminal_split_frames, \
        terminal_split_metrics = \
        _well_a5_issue004_terminal_two_core_split_inheritance(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "55_well_a5_issue004_terminal_two_core_split_inheritance.tif",
          labels, interval)
    _save(output_root /
          "55_well_a5_issue004_terminal_two_core_split_inheritance_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, bracketed_invasion_audit, \
        bracketed_invasion_applications, bracketed_invasion_metrics = \
        _well_b1_issue001_bracketed_established_owner_invasion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "56_well_b1_issue001_bracketed_established_owner_invasion.tif",
          labels, interval)
    _save(output_root /
          "56_well_b1_issue001_bracketed_established_owner_invasion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, terminal_diversion_proposals, \
        terminal_diversion_applications, terminal_diversion_metrics = \
        _well_b2_issue001_terminal_projection_diversion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "57_well_b2_issue001_terminal_projection_diversion.tif",
          labels, interval)
    _save(output_root /
          "57_well_b2_issue001_terminal_projection_diversion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, delayed_reclaim_proposals, \
        delayed_reclaim_applications, delayed_reclaim_metrics = \
        _well_b2_issue002_delayed_projection_reclaim(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "58_well_b2_issue002_delayed_projection_reclaim.tif",
          labels, interval)
    _save(output_root /
          "58_well_b2_issue002_delayed_projection_reclaim_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, boundary_excursion_audit, \
        boundary_excursion_applications, boundary_excursion_metrics = \
        _well_b2_issue003_boundary_owner_excursion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "59_well_b2_issue003_boundary_owner_excursion.tif",
          labels, interval)
    _save(output_root /
          "59_well_b2_issue003_boundary_owner_excursion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, terminal_assimilation_audit, \
        terminal_assimilation_applications, terminal_assimilation_metrics = \
        _well_b2_issue004_terminal_two_seat_assimilation(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "60_well_b2_issue004_terminal_two_seat_assimilation.tif",
          labels, interval)
    _save(output_root /
          "60_well_b2_issue004_terminal_two_seat_assimilation_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, mixed_flash_audit, mixed_flash_applications, \
        mixed_flash_metrics = _well_b2_issue005_bracketed_mixed_owner_flash(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "61_well_b2_issue005_bracketed_mixed_owner_flash.tif",
          labels, interval)
    _save(output_root /
          "61_well_b2_issue005_bracketed_mixed_owner_flash_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, dominant_seat_audit, dominant_seat_applications, \
        dominant_seat_base_audit, dominant_seat_metrics = \
        _well_b2_issue006_dominant_seat_projection_diversion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "62_well_b2_issue006_dominant_seat_projection_diversion.tif",
          labels, interval)
    _save(output_root /
          "62_well_b2_issue006_dominant_seat_projection_diversion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, ownerless_cohort_gap_audit, \
        ownerless_cohort_gap_metrics = \
        _well_b2_issue009_ownerless_cohort_latent_gap_completion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "63_well_b2_issue009_ownerless_cohort_latent_gap_completion.tif",
          labels, interval)
    _save(output_root /
          "63_well_b2_issue009_ownerless_cohort_latent_gap_completion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, recording_start_two_seat_audit, \
        recording_start_two_seat_applications, recording_start_two_seat_metrics = \
        _well_b2_issue018_recording_start_two_seat_inheritance(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "87_well_b2_issue018_recording_start_two_seat_inheritance.tif",
          labels, interval)
    _save(output_root /
          "87_well_b2_issue018_recording_start_two_seat_inheritance_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, gap_exchange_audit, gap_exchange_applications, \
        gap_exchange_metrics = \
        _well_b3_issue002_gap_tolerant_reciprocal_seat_exchange(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "90_well_b3_issue002_gap_tolerant_reciprocal_seat_exchange.tif",
          labels, interval)
    _save(output_root /
          "90_well_b3_issue002_gap_tolerant_reciprocal_seat_exchange_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, reconnected_partition_audit, \
        reconnected_partition_applications, reconnected_partition_metrics = \
        _well_b3_issue004_reconnected_companion_merge_partition(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "91_well_b3_issue004_reconnected_companion_merge_partition.tif",
          labels, interval)
    _save(output_root /
          "91_well_b3_issue004_reconnected_companion_merge_partition_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, continuity_takeover_audit, \
        continuity_takeover_applications, continuity_takeover_metrics = \
        _well_b3_issue005_component_continuity_duplicate_takeover(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "92_well_b3_issue005_component_continuity_duplicate_takeover.tif",
          labels, interval)
    _save(output_root /
          "92_well_b3_issue005_component_continuity_duplicate_takeover_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, right_censored_partition_audit, \
        right_censored_partition_frames, right_censored_partition_metrics = \
        _well_b3_issue006_right_censored_seat_partition(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "93_well_b3_issue006_right_censored_seat_partition.tif",
          labels, interval)
    _save(output_root /
          "93_well_b3_issue006_right_censored_seat_partition_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, right_censored_exchange_audit, \
        right_censored_exchange_metrics = \
        _well_b3_issue007_right_censored_reciprocal_exchange(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "94_well_b3_issue007_right_censored_reciprocal_exchange.tif",
          labels, interval)
    _save(output_root /
          "94_well_b3_issue007_right_censored_reciprocal_exchange_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, persistent_flash_audit, \
        persistent_flash_applications, persistent_flash_metrics = \
        _well_b3_issue008_persistent_single_owner_flash(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "95_well_b3_issue008_persistent_single_owner_flash.tif",
          labels, interval)
    _save(output_root /
          "95_well_b3_issue008_persistent_single_owner_flash_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, delayed_projection_flash_audit, \
        delayed_projection_flash_applications, \
        delayed_projection_flash_base_audit, \
        delayed_projection_flash_metrics = \
        _well_b3_issue012_delayed_owner_projection_flash(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "97_well_b3_issue012_delayed_owner_projection_flash.tif",
          labels, interval)
    _save(output_root /
          "97_well_b3_issue012_delayed_owner_projection_flash_unclaimed.tif",
          unclaimed, interval)
    issue013_parent_labels = labels.copy()
    labels, unclaimed, anchored_relay_audit, anchored_relay_applications, \
        anchored_relay_metrics = \
        _well_b3_issue013_anchored_projection_owner_relay(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "99_well_b3_issue013_anchored_projection_owner_relay.tif",
          labels, interval)
    _save(output_root /
          "99_well_b3_issue013_anchored_projection_owner_relay_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, alias_retirement_audit, \
        alias_retirement_applications, alias_retirement_metrics = \
        _well_b3_issue013_ephemeral_speckle_alias_retirement(
            labels, unclaimed, config, output_root)
    _save(output_root /
          "100_well_b3_issue013_ephemeral_speckle_alias_retirement.tif",
          labels, interval)
    _save(output_root /
          "100_well_b3_issue013_ephemeral_speckle_alias_retirement_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, anchored_raw_gap_audit, anchored_raw_gap_metrics = \
        _well_b3_issue013_anchored_projection_raw_gap_completion(
            labels, unclaimed, raw, anchored_relay_audit,
            anchored_relay_applications, config, output_root)
    _save(output_root /
          "101_well_b3_issue013_anchored_projection_raw_gap_completion.tif",
          labels, interval)
    _save(output_root /
          "101_well_b3_issue013_anchored_projection_raw_gap_completion_unclaimed.tif",
          unclaimed, interval)
    _, _, _, retirement_persistence_metrics = \
        retirement_aware_persistence_integration.run(
            issue013_parent_labels, labels, alias_retirement_audit,
            alias_retirement_applications, output_root)
    labels, unclaimed, terminal_projection_chain_audit, \
        terminal_projection_chain_applications, \
        terminal_projection_chain_metrics = \
        _well_b3_issue014_terminal_projection_chain_completion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "103_well_b3_issue014_terminal_projection_chain_completion.tif",
          labels, interval)
    _save(output_root /
          "103_well_b3_issue014_terminal_projection_chain_completion_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, body_scale_ownerless_audit, \
        body_scale_ownerless_applications, body_scale_ownerless_metrics = \
        _well_b3_issue016_body_scale_ownerless_allocation(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "105_well_b3_issue016_body_scale_ownerless_allocation.tif",
          labels, interval)
    _save(output_root /
          "105_well_b3_issue016_body_scale_ownerless_allocation_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, terminal_boundary_seat_audit, \
        terminal_boundary_seat_frames, terminal_boundary_seat_metrics = \
        _well_b3_issue017_terminal_boundary_vanished_seat(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "106_well_b3_issue017_terminal_boundary_vanished_seat.tif",
          labels, interval)
    _save(output_root /
          "106_well_b3_issue017_terminal_boundary_vanished_seat_unclaimed.tif",
          unclaimed, interval)
    labels, unclaimed, bracketed_ownerless_seat_audit, \
        bracketed_ownerless_seat_frames, bracketed_ownerless_seat_metrics = \
        _well_b3_issue019_bracketed_ownerless_seat_completion(
            labels, unclaimed, raw, config, output_root)
    _save(output_root /
          "109_well_b3_issue019_bracketed_ownerless_seat_completion.tif",
          labels, interval)
    _save(output_root /
          "109_well_b3_issue019_bracketed_ownerless_seat_completion_unclaimed.tif",
          unclaimed, interval)
    accepted_foreground = foreground > 0
    final_foreground = (labels > 0) | (unclaimed > 0)
    if np.any(accepted_foreground & ~final_foreground):
        raise AssertionError("complete accepted history removed foreground")
    if np.any((labels > 0) & (unclaimed > 0)):
        raise AssertionError("complete accepted history overlaps assigned and unclaimed pixels")
    issue045_enabled = bool(config.values["accepted_postprocessing"].get(
        "field_wide_atomic_two_seat_recovery", {}).get("enabled", False))
    issue066_enabled = bool(config.values["accepted_postprocessing"].get(
        "field_wide_transient_misownership_recovery", {}).get(
            "enabled", False))
    distinct_history_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_distinct_history_core_preservation", {}).get(
                "enabled", False))
    branched_lineage_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_branched_motion_lineage_recovery", {}).get(
                "enabled", False))
    complete_flip_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_complete_flip_recovery", {}).get(
                "enabled", False))
    projection_seat_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_projection_tolerant_seat_preservation", {}).get(
                "enabled", False))
    retroactive_successor_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_retroactive_successor_separation", {}).get(
                "enabled", False))
    terminal_split_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_terminal_two_core_split_inheritance", {}).get(
                "enabled", False))
    bracketed_invasion_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_bracketed_established_owner_invasion", {}).get(
                "enabled", False))
    terminal_diversion_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_terminal_projection_diversion", {}).get(
                "enabled", False))
    delayed_reclaim_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_delayed_projection_reclaim", {}).get(
                "enabled", False))
    boundary_excursion_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_boundary_owner_excursion", {}).get(
                "enabled", False))
    terminal_assimilation_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_terminal_two_seat_assimilation", {}).get(
                "enabled", False))
    mixed_flash_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_bracketed_mixed_owner_flash", {}).get(
                "enabled", False))
    dominant_seat_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_dominant_seat_projection_diversion", {}).get(
                "enabled", False))
    ownerless_cohort_gap_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_ownerless_cohort_latent_gap_completion", {}).get(
                "enabled", False))
    gap_exchange_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_gap_tolerant_reciprocal_seat_exchange", {}).get(
                "enabled", False))
    reconnected_partition_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_reconnected_companion_merge_partition", {}).get(
                "enabled", False))
    bracketed_ownerless_seat_enabled = bool(
        config.values["accepted_postprocessing"].get(
            "field_wide_bracketed_ownerless_seat_completion", {}).get(
                "enabled", False))
    if (bracketed_ownerless_seat_enabled
            and int(bracketed_ownerless_seat_metrics.get(
                "changed_pixels", 0)) > 0):
        # B3 Issue 019 A005 uses canonical plain zlib TIFF encoding.
        _save_final_plain_labels(output_root / f"{stem}.tif", labels)
    elif (reconnected_partition_enabled
            and int(reconnected_partition_metrics.get(
                "changed_pixels", 0)) > 0):
        # B3 Issue 004 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (gap_exchange_enabled
            and int(gap_exchange_metrics.get("changed_pixels", 0)) > 0):
        # B3 Issue 002 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (ownerless_cohort_gap_enabled
            and int(ownerless_cohort_gap_metrics.get(
                "changed_pixels", 0)) > 0):
        # B2 Issue 009 R01 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (dominant_seat_enabled
            and int(dominant_seat_metrics.get("changed_pixels", 0)) > 0):
        # B2 Issue 006 R01 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (mixed_flash_enabled
            and int(mixed_flash_metrics.get("changed_pixels", 0)) > 0):
        # B2 Issue 005 R01 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (terminal_assimilation_enabled
            and int(terminal_assimilation_metrics.get(
                "changed_pixels", 0)) > 0):
        # B2 Issue 004 R01 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (boundary_excursion_enabled
            and int(boundary_excursion_metrics.get(
                "changed_pixels", 0)) > 0):
        # B2 Issue 003 R01 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (delayed_reclaim_enabled
            and int(delayed_reclaim_metrics.get(
                "changed_pixels", 0)) > 0):
        # B2 Issue 002 R01 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (terminal_diversion_enabled
            and int(terminal_diversion_metrics.get(
                "changed_pixels", 0)) > 0):
        # B2 Issue 001 R03 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (bracketed_invasion_enabled
            and int(bracketed_invasion_metrics.get(
                "changed_pixels", 0)) > 0):
        # B1 Issue 001 R02 uses canonical ImageJ zlib label encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (terminal_split_enabled
            and int(terminal_split_metrics.get("changed_pixels", 0)) > 0):
        # Issue 004 R05's reviewed candidate uses plain zlib TIFF encoding.
        _save_final_plain_labels(output_root / f"{stem}.tif", labels)
    elif (retroactive_successor_enabled
            and int(retroactive_successor_metrics.get(
                "changed_pixels", 0)) > 0):
        # Issue 006 R01's reviewed candidate uses ImageJ zlib encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (projection_seat_enabled
            and int(projection_seat_metrics.get("changed_pixels", 0)) > 0):
        # Issue 005 R02's reviewed candidate uses ImageJ zlib encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (complete_flip_enabled
            and int(complete_flip_metrics.get("changed_pixels", 0)) > 0):
        # Issue 003 R04's reviewed candidate uses ImageJ zlib encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (branched_lineage_enabled
            and int(branched_lineage_metrics.get("changed_pixels", 0)) > 0):
        # Issue 003's reviewed candidate uses ImageJ zlib encoding.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif (distinct_history_enabled
            and int(distinct_history_metrics.get("changed_pixels", 0)) > 0):
        # A changed distinct-history result uses its canonical compressed encoding.
        _save_final_plain_labels(output_root / f"{stem}.tif", labels)
    elif issue066_enabled:
        # Issue 066 uses the canonical ImageJ-encoded review TIFF.
        # Preserve those exact accepted bytes after the label-changing stage.
        _save_final_labels(output_root / f"{stem}.tif", labels)
    elif issue045_enabled:
        _save_final_plain_labels(output_root / f"{stem}.tif", labels)
    else:
        _save_final_labels(output_root / f"{stem}.tif", labels)
    final_unclaimed = output_root / f"{stem}_unclaimed_original_ids.tif"
    temporal_unclaimed = any(bool(config.values["accepted_postprocessing"].get(
        name, {}).get("enabled", False)) for name in (
            "field_wide_bracketed_encounter_recovery",
            "field_wide_recurrent_exclusive_owner_relay",
            "field_wide_isolated_owner_blip_recovery",
            "field_wide_transient_misownership_recovery"))
    if temporal_unclaimed:
        # Issues 041-045 make the unclaimed ledger a time-varying accepted
        # result; preserve the reviewed candidate's temporal ImageJ metadata.
        _save_final_labels(final_unclaimed, unclaimed)
    else:
        _save_final_unclaimed(final_unclaimed, unclaimed)
    summary = {
        "source_frames_removed": [1, 2],
        "identity_target_count": 0,
        "coordinate_target_count": 0,
        "motion_reservation_events": len(motion_reserved),
        "stationary_events": int(len(events)),
        "short_blank_recovery_audit_rows": int(len(short_blank_audit)),
        "short_blank_recovered_pixels": int(
            short_blank_metrics["recovered_pixels"]),
        "short_blank_identity_targets": 0,
        "short_blank_interval_targets": 0,
        "small_cell_recovery_audit_rows": int(len(small_cell_audit)),
        "small_cell_recovered_pixels": int(
            small_cell_metrics["recovered_pixels"]),
        "small_cell_identity_targets": 0,
        "small_cell_interval_targets": 0,
        "moderate_gap_recovery_audit_rows": int(len(moderate_gap_audit)),
        "moderate_gap_recovered_pixels": int(
            moderate_gap_metrics["recovered_pixels"]),
        "moderate_gap_identity_targets": 0,
        "moderate_gap_interval_targets": 0,
        "owner_consensus_audit_rows": int(len(owner_audit)),
        "owner_consensus_applied_proposals": int(
            owner_metrics["applied_proposals"]),
        "owner_consensus_changed_pixels": int(
            owner_metrics["changed_pixels"]),
        "owner_consensus_identity_targets": 0,
        "owner_consensus_frame_targets": 0,
        "owner_consensus_coordinate_targets": 0,
        "owner_consensus_event_targets": 0,
        "separable_merge_audit_rows": int(len(merge_audit)),
        "separable_merge_applied_proposals": int(
            merge_metrics["applied_proposals"]),
        "separable_merge_changed_pixels": int(
            merge_metrics["changed_pixels"]),
        "separable_merge_identity_targets": 0,
        "separable_merge_frame_targets": 0,
        "separable_merge_coordinate_targets": 0,
        "separable_merge_event_targets": 0,
        "recording_start_split_audit_rows": int(len(start_split_audit)),
        "recording_start_split_applied_proposals": int(
            start_split_metrics["applied_proposals"]),
        "recording_start_split_changed_pixels": int(
            start_split_metrics["changed_pixels"]),
        "recording_start_split_changed_frames": int(
            start_split_metrics["changed_frames"]),
        "recording_start_split_identity_targets": 0,
        "recording_start_split_frame_targets": 0,
        "recording_start_split_coordinate_targets": 0,
        "recording_start_split_event_targets": 0,
        "merge_residency_audit_rows": int(len(residency_audit)),
        "merge_residency_applied_proposals": int(
            residency_metrics["applied_proposals"]),
        "merge_residency_changed_label_pixels": int(
            residency_metrics["changed_label_pixels"]),
        "merge_residency_changed_unclaimed_pixels": int(
            residency_metrics["changed_unclaimed_pixels"]),
        "merge_residency_changed_frames": int(
            residency_metrics["changed_frames"]),
        "merge_residency_identity_targets": 0,
        "merge_residency_frame_targets": 0,
        "merge_residency_coordinate_targets": 0,
        "merge_residency_event_targets": 0,
        "persistent_body_identity_audit_rows": int(len(body_identity_audit)),
        "persistent_body_identity_applied_proposals": int(
            body_identity_metrics["applied_proposals"]),
        "persistent_body_identity_changed_label_pixels": int(
            body_identity_metrics["changed_label_pixels"]),
        "persistent_body_identity_changed_unclaimed_pixels": int(
            body_identity_metrics["changed_unclaimed_pixels"]),
        "persistent_body_identity_changed_frames": int(
            body_identity_metrics["changed_frames"]),
        "persistent_body_identity_identity_targets": 0,
        "persistent_body_identity_track_targets": 0,
        "persistent_body_identity_frame_targets": 0,
        "persistent_body_identity_coordinate_targets": 0,
        "persistent_body_identity_event_targets": 0,
        "resident_takeover_audit_rows": int(len(takeover_audit)),
        "resident_takeover_eligible_proposals": int(
            takeover_metrics["eligible_proposals"]),
        "resident_takeover_applied_proposals": int(
            takeover_metrics["applied_proposals"]),
        "resident_takeover_changed_label_pixels": int(
            takeover_metrics["changed_label_pixels"]),
        "resident_takeover_changed_unclaimed_pixels": int(
            takeover_metrics["changed_unclaimed_pixels"]),
        "resident_takeover_changed_frames": int(
            takeover_metrics["changed_frames"]),
        "resident_takeover_identity_targets": 0,
        "resident_takeover_track_targets": 0,
        "resident_takeover_frame_targets": 0,
        "resident_takeover_coordinate_targets": 0,
        "resident_takeover_event_targets": 0,
        "ownerless_body_recovery_audit_rows": int(len(ownerless_audit)),
        "ownerless_body_recovery_eligible_proposals": int(
            ownerless_metrics["eligible_proposals"]),
        "ownerless_body_recovery_applied_proposals": int(
            ownerless_metrics["applied_proposals"]),
        "ownerless_body_recovery_changed_label_pixels": int(
            ownerless_metrics["changed_label_pixels"]),
        "ownerless_body_recovery_changed_unclaimed_pixels": int(
            ownerless_metrics["changed_unclaimed_pixels"]),
        "ownerless_body_recovery_changed_frames": int(
            ownerless_metrics["changed_frames"]),
        "ownerless_body_recovery_foreground_added_pixels": int(
            ownerless_metrics["foreground_added_pixels"]),
        "ownerless_body_recovery_identity_targets": 0,
        "ownerless_body_recovery_track_targets": 0,
        "ownerless_body_recovery_frame_targets": 0,
        "ownerless_body_recovery_coordinate_targets": 0,
        "ownerless_body_recovery_event_targets": 0,
        "ownerless_cohort_recovery_audit_rows": int(
            len(ownerless_cohort_audit)),
        "ownerless_cohort_recovery_eligible_seed_tracks": int(
            ownerless_cohort_metrics["eligible_seed_tracks"]),
        "ownerless_cohort_recovery_eligible_follower_tracks": int(
            ownerless_cohort_metrics["eligible_follower_tracks"]),
        "ownerless_cohort_recovery_reciprocal_fragment_edges": int(
            ownerless_cohort_metrics["reciprocal_fragment_edges"]),
        "ownerless_cohort_recovery_physical_seats": int(
            ownerless_cohort_metrics["physical_seats"]),
        "ownerless_cohort_recovery_applied_seats": int(
            ownerless_cohort_metrics["applied_seats"]),
        "ownerless_cohort_recovery_changed_label_pixels": int(
            ownerless_cohort_metrics["changed_label_pixels"]),
        "ownerless_cohort_recovery_changed_unclaimed_pixels": int(
            ownerless_cohort_metrics["changed_unclaimed_pixels"]),
        "ownerless_cohort_recovery_changed_frames": int(
            ownerless_cohort_metrics["changed_frames"]),
        "ownerless_cohort_recovery_foreground_added_pixels": int(
            ownerless_cohort_metrics["foreground_added_pixels"]),
        "ownerless_cohort_recovery_identity_targets": 0,
        "ownerless_cohort_recovery_track_targets": 0,
        "ownerless_cohort_recovery_frame_targets": 0,
        "ownerless_cohort_recovery_coordinate_targets": 0,
        "ownerless_cohort_recovery_event_targets": 0,
        "bracketed_encounter_recovery_audit_rows": int(
            len(bracketed_audit)),
        "bracketed_encounter_recovery_applied_excursions": int(
            bracketed_metrics["applied_excursions"]),
        "bracketed_encounter_recovery_changed_label_pixels": int(
            bracketed_metrics["changed_label_pixels"]),
        "bracketed_encounter_recovery_changed_unclaimed_pixels": int(
            bracketed_metrics["changed_unclaimed_pixels"]),
        "bracketed_encounter_recovery_changed_frames": int(
            bracketed_metrics["changed_frames"]),
        "bracketed_encounter_recovery_foreground_changed_pixels": int(
            bracketed_metrics["foreground_changed_pixels"]),
        "bracketed_encounter_recovery_preexisting_unclaimed_changed_pixels": int(
            bracketed_metrics["preexisting_unclaimed_changed_pixels"]),
        "bracketed_encounter_recovery_new_duplicate_components": int(
            bracketed_metrics["new_duplicate_components"]),
        "bracketed_encounter_recovery_identity_targets": 0,
        "bracketed_encounter_recovery_track_targets": 0,
        "bracketed_encounter_recovery_frame_targets": 0,
        "bracketed_encounter_recovery_coordinate_targets": 0,
        "bracketed_encounter_recovery_event_targets": 0,
        "recurrent_exclusive_owner_relay_audit_rows": int(
            len(relay_audit)),
        "recurrent_exclusive_owner_relay_frame_audit_rows": int(
            len(relay_frame_audit)),
        "recurrent_exclusive_owner_relay_eligible_tracks": int(
            relay_metrics["eligible_tracks"]),
        "recurrent_exclusive_owner_relay_applied_tracks": int(
            relay_metrics["applied_tracks"]),
        "recurrent_exclusive_owner_relay_applied_runs": int(
            relay_metrics["applied_runs"]),
        "recurrent_exclusive_owner_relay_changed_label_pixels": int(
            relay_metrics["changed_label_pixels"]),
        "recurrent_exclusive_owner_relay_changed_unclaimed_pixels": int(
            relay_metrics["changed_unclaimed_pixels"]),
        "recurrent_exclusive_owner_relay_changed_frames": int(
            relay_metrics["changed_frames"]),
        "recurrent_exclusive_owner_relay_foreground_changed_pixels": int(
            relay_metrics["foreground_changed_pixels"]),
        "recurrent_exclusive_owner_relay_preexisting_unclaimed_changed_pixels": int(
            relay_metrics["preexisting_unclaimed_changed_pixels"]),
        "recurrent_exclusive_owner_relay_new_duplicate_components": int(
            relay_metrics["new_duplicate_components"]),
        "recurrent_exclusive_owner_relay_identity_targets": 0,
        "recurrent_exclusive_owner_relay_track_targets": 0,
        "recurrent_exclusive_owner_relay_frame_targets": 0,
        "recurrent_exclusive_owner_relay_coordinate_targets": 0,
        "recurrent_exclusive_owner_relay_event_targets": 0,
        "isolated_owner_blip_recovery_audit_rows": int(len(blip_audit)),
        "isolated_owner_blip_recovery_frame_audit_rows": int(
            len(blip_frame_audit)),
        "isolated_owner_blip_recovery_run_triples_audited": int(
            blip_metrics["run_triples_audited"]),
        "isolated_owner_blip_recovery_eligible_blips": int(
            blip_metrics["eligible_blips"]),
        "isolated_owner_blip_recovery_structurally_applicable_blips": int(
            blip_metrics["structurally_applicable_blips"]),
        "isolated_owner_blip_recovery_applied_blips": int(
            blip_metrics["applied_blips"]),
        "isolated_owner_blip_recovery_changed_label_pixels": int(
            blip_metrics["changed_label_pixels"]),
        "isolated_owner_blip_recovery_changed_unclaimed_pixels": int(
            blip_metrics["changed_unclaimed_pixels"]),
        "isolated_owner_blip_recovery_changed_frames": int(
            blip_metrics["changed_frames"]),
        "isolated_owner_blip_recovery_foreground_changed_pixels": int(
            blip_metrics["foreground_changed_pixels"]),
        "isolated_owner_blip_recovery_preexisting_unclaimed_changed_pixels": int(
            blip_metrics["preexisting_unclaimed_changed_pixels"]),
        "isolated_owner_blip_recovery_new_duplicate_components": int(
            blip_metrics["new_duplicate_components"]),
        "isolated_owner_blip_recovery_identity_targets": 0,
        "isolated_owner_blip_recovery_track_targets": 0,
        "isolated_owner_blip_recovery_frame_targets": 0,
        "isolated_owner_blip_recovery_coordinate_targets": 0,
        "isolated_owner_blip_recovery_event_targets": 0,
        "atomic_two_seat_recovery_cohorts_audited": int(
            two_seat_metrics["cohorts_audited"]),
        "atomic_two_seat_recovery_eligible_cohorts": int(
            two_seat_metrics["eligible_cohorts"]),
        "atomic_two_seat_recovery_applied_cohorts": int(
            two_seat_metrics["applied_cohorts"]),
        "atomic_two_seat_recovery_rejected_application_cohorts": int(
            two_seat_metrics["rejected_application_cohorts"]),
        "atomic_two_seat_recovery_application_rows": int(
            len(two_seat_applications)),
        "atomic_two_seat_recovery_frame_audit_rows": int(
            len(two_seat_frame_audit)),
        "atomic_two_seat_recovery_changed_pixels": int(
            two_seat_metrics["changed_pixels"]),
        "atomic_two_seat_recovery_changed_frames": int(
            two_seat_metrics["changed_frames"]),
        "atomic_two_seat_recovery_raw_supported_additions": int(
            two_seat_metrics["raw_supported_additions"]),
        "atomic_two_seat_recovery_relabelled_foreground_pixels": int(
            two_seat_metrics["relabelled_foreground_pixels"]),
        "atomic_two_seat_recovery_removed_foreground_pixels": int(
            two_seat_metrics["removed_foreground_pixels"]),
        "atomic_two_seat_recovery_zero_signal_additions": int(
            two_seat_metrics["zero_signal_additions"]),
        "atomic_two_seat_recovery_preexisting_unclaimed_changed_pixels": int(
            two_seat_metrics["preexisting_unclaimed_changed_pixels"]),
        "atomic_two_seat_recovery_unclaimed_overlap_additions": int(
            two_seat_metrics["unclaimed_overlap_additions"]),
        "atomic_two_seat_recovery_old_identity_set_preserved": bool(
            two_seat_metrics["old_identity_set_preserved"]),
        "atomic_two_seat_recovery_new_identities": list(
            two_seat_metrics["new_identities"]),
        "atomic_two_seat_recovery_new_duplicate_components": int(
            two_seat_metrics["new_same_frame_identity_components"]),
        "atomic_two_seat_recovery_identity_targets": 0,
        "atomic_two_seat_recovery_track_targets": 0,
        "atomic_two_seat_recovery_frame_targets": 0,
        "atomic_two_seat_recovery_coordinate_targets": 0,
        "atomic_two_seat_recovery_event_targets": 0,
        "recording_start_duplicate_release_audit_rows": int(
            len(duplicate_release_audit)),
        "recording_start_duplicate_release_audited_identities": int(
            duplicate_release_metrics["audited_identities"]),
        "recording_start_duplicate_release_eligible_cohorts": int(
            duplicate_release_metrics["eligible_cohorts"]),
        "recording_start_duplicate_release_applied_cohorts": int(
            duplicate_release_metrics["applied_cohorts"]),
        "recording_start_duplicate_release_rejected_application_cohorts": int(
            duplicate_release_metrics["rejected_application_cohorts"]),
        "recording_start_duplicate_release_changed_pixels": int(
            duplicate_release_metrics["changed_pixels"]),
        "recording_start_duplicate_release_changed_frames": int(
            duplicate_release_metrics["changed_frames"]),
        "recording_start_duplicate_release_foreground_changed_pixels": int(
            duplicate_release_metrics["foreground_changed_pixels"]),
        "recording_start_duplicate_release_unclaimed_overlap_pixels": int(
            duplicate_release_metrics["unclaimed_overlap_pixels"]),
        "recording_start_duplicate_release_zero_signal_additions": int(
            duplicate_release_metrics["zero_signal_additions"]),
        "recording_start_duplicate_release_old_identity_set_preserved": bool(
            duplicate_release_metrics["old_identity_set_preserved"]),
        "recording_start_duplicate_release_new_duplicate_components": int(
            duplicate_release_metrics["new_same_frame_identity_components"]),
        "recording_start_duplicate_release_identity_targets": 0,
        "recording_start_duplicate_release_track_targets": 0,
        "recording_start_duplicate_release_frame_targets": 0,
        "recording_start_duplicate_release_coordinate_targets": 0,
        "recording_start_duplicate_release_event_targets": 0,
        "late_owner_backfill_tracks_audited": int(
            late_owner_metrics["tracks_audited"]),
        "late_owner_backfill_proposals_audited": int(
            late_owner_metrics["proposals_audited"]),
        "late_owner_backfill_eligible_lineages": int(
            late_owner_metrics["eligible_lineages"]),
        "late_owner_backfill_applied_lineages": int(
            late_owner_metrics["applied_lineages"]),
        "late_owner_backfill_rejected_application_lineages": int(
            late_owner_metrics["rejected_application_lineages"]),
        "late_owner_backfill_changed_pixels": int(
            late_owner_metrics["changed_pixels"]),
        "late_owner_backfill_changed_frames": int(
            late_owner_metrics["changed_frames"]),
        "late_owner_backfill_raw_supported_additions": int(
            late_owner_metrics["raw_supported_additions"]),
        "late_owner_backfill_relabelled_foreground_pixels": int(
            late_owner_metrics["relabelled_foreground_pixels"]),
        "late_owner_backfill_removed_foreground_pixels": int(
            late_owner_metrics["removed_foreground_pixels"]),
        "late_owner_backfill_zero_signal_additions": int(
            late_owner_metrics["zero_signal_additions"]),
        "late_owner_backfill_preexisting_unclaimed_changed_pixels": int(
            late_owner_metrics["preexisting_unclaimed_changed_pixels"]),
        "late_owner_backfill_unclaimed_overlap_additions": int(
            late_owner_metrics["unclaimed_overlap_additions"]),
        "late_owner_backfill_old_identity_set_preserved": bool(
            late_owner_metrics["old_identity_set_preserved"]),
        "late_owner_backfill_donor_named_frame_losses": int(
            late_owner_metrics["donor_named_frame_losses"]),
        "late_owner_backfill_new_duplicate_components": int(
            late_owner_metrics["new_same_frame_identity_components"]),
        "late_owner_backfill_identity_targets": 0,
        "late_owner_backfill_track_targets": 0,
        "late_owner_backfill_frame_targets": 0,
        "late_owner_backfill_coordinate_targets": 0,
        "late_owner_backfill_event_targets": 0,
        "bracketed_handoff_recovery_tracks_audited": int(
            handoff_metrics["tracks_audited"]),
        "bracketed_handoff_recovery_proposals_audited": int(
            handoff_metrics["proposals_audited"]),
        "bracketed_handoff_recovery_eligible_handoffs": int(
            handoff_metrics["eligible_handoffs"]),
        "bracketed_handoff_recovery_applied_handoffs": int(
            handoff_metrics["applied_handoffs"]),
        "bracketed_handoff_recovery_rejected_application_handoffs": int(
            handoff_metrics["rejected_application_handoffs"]),
        "bracketed_handoff_recovery_application_rows": int(
            len(handoff_applications)),
        "bracketed_handoff_recovery_frame_audit_rows": int(
            len(handoff_frame_audit)),
        "bracketed_handoff_recovery_changed_pixels": int(
            handoff_metrics["changed_pixels"]),
        "bracketed_handoff_recovery_changed_frames": int(
            handoff_metrics["changed_frames"]),
        "bracketed_handoff_recovery_raw_supported_additions": int(
            handoff_metrics["raw_supported_additions"]),
        "bracketed_handoff_recovery_relabelled_foreground_pixels": int(
            handoff_metrics["relabelled_foreground_pixels"]),
        "bracketed_handoff_recovery_removed_foreground_pixels": int(
            handoff_metrics["removed_foreground_pixels"]),
        "bracketed_handoff_recovery_zero_signal_additions": int(
            handoff_metrics["zero_signal_additions"]),
        "bracketed_handoff_recovery_preexisting_unclaimed_changed_pixels": int(
            handoff_metrics["preexisting_unclaimed_changed_pixels"]),
        "bracketed_handoff_recovery_unclaimed_overlap_additions": int(
            handoff_metrics["unclaimed_overlap_additions"]),
        "bracketed_handoff_recovery_old_identity_set_preserved": bool(
            handoff_metrics["old_identity_set_preserved"]),
        "bracketed_handoff_recovery_donor_named_frame_losses": int(
            handoff_metrics["donor_named_frame_losses"]),
        "bracketed_handoff_recovery_new_duplicate_components": int(
            handoff_metrics["new_same_frame_identity_components"]),
        "bracketed_handoff_recovery_identity_targets": 0,
        "bracketed_handoff_recovery_track_targets": 0,
        "bracketed_handoff_recovery_frame_targets": 0,
        "bracketed_handoff_recovery_coordinate_targets": 0,
        "bracketed_handoff_recovery_event_targets": 0,
        "post_split_backfill_tracks_audited": int(
            post_split_metrics["tracks_audited"]),
        "post_split_backfill_proposals_audited": int(
            post_split_metrics["proposals_audited"]),
        "post_split_backfill_eligible_proposals": int(
            post_split_metrics["eligible_proposals"]),
        "post_split_backfill_applied_proposals": int(
            post_split_metrics["applied_proposals"]),
        "post_split_backfill_rejected_application_proposals": int(
            post_split_metrics["rejected_application_proposals"]),
        "post_split_backfill_application_rows": int(
            len(post_split_applications)),
        "post_split_backfill_frame_audit_rows": int(
            len(post_split_frame_audit)),
        "post_split_backfill_changed_pixels": int(
            post_split_metrics["changed_pixels"]),
        "post_split_backfill_changed_frames": int(
            post_split_metrics["changed_frames"]),
        "post_split_backfill_raw_supported_additions": int(
            post_split_metrics["raw_supported_additions"]),
        "post_split_backfill_relabelled_foreground_pixels": int(
            post_split_metrics["relabelled_foreground_pixels"]),
        "post_split_backfill_removed_foreground_pixels": int(
            post_split_metrics["removed_foreground_pixels"]),
        "post_split_backfill_zero_signal_additions": int(
            post_split_metrics["zero_signal_additions"]),
        "post_split_backfill_preexisting_unclaimed_changed_pixels": int(
            post_split_metrics["preexisting_unclaimed_changed_pixels"]),
        "post_split_backfill_unclaimed_overlap_additions": int(
            post_split_metrics["unclaimed_overlap_additions"]),
        "post_split_backfill_old_identity_set_preserved": bool(
            post_split_metrics["old_identity_set_preserved"]),
        "post_split_backfill_donor_named_frame_losses": int(
            post_split_metrics["donor_named_frame_losses"]),
        "post_split_backfill_new_duplicate_components": int(
            post_split_metrics["new_same_frame_identity_components"]),
        "post_split_backfill_identity_targets": 0,
        "post_split_backfill_track_targets": 0,
        "post_split_backfill_frame_targets": 0,
        "post_split_backfill_coordinate_targets": 0,
        "post_split_backfill_event_targets": 0,
        "concurrent_duplicate_invasion_tracks_audited": int(
            concurrent_invasion_metrics["tracks_audited"]),
        "concurrent_duplicate_invasion_proposals_audited": int(
            concurrent_invasion_metrics["proposals_audited"]),
        "concurrent_duplicate_invasion_eligible_proposals": int(
            concurrent_invasion_metrics["eligible_proposals"]),
        "concurrent_duplicate_invasion_applied_proposals": int(
            concurrent_invasion_metrics["applied_proposals"]),
        "concurrent_duplicate_invasion_rejected_application_proposals": int(
            concurrent_invasion_metrics["rejected_application_proposals"]),
        "concurrent_duplicate_invasion_application_rows": int(
            len(concurrent_invasion_applications)),
        "concurrent_duplicate_invasion_frame_audit_rows": int(
            len(concurrent_invasion_frame_audit)),
        "concurrent_duplicate_invasion_changed_pixels": int(
            concurrent_invasion_metrics["changed_pixels"]),
        "concurrent_duplicate_invasion_changed_frames": int(
            concurrent_invasion_metrics["changed_frames"]),
        "concurrent_duplicate_invasion_raw_supported_additions": int(
            concurrent_invasion_metrics["raw_supported_additions"]),
        "concurrent_duplicate_invasion_relabelled_foreground_pixels": int(
            concurrent_invasion_metrics["relabelled_foreground_pixels"]),
        "concurrent_duplicate_invasion_removed_foreground_pixels": int(
            concurrent_invasion_metrics["removed_foreground_pixels"]),
        "concurrent_duplicate_invasion_zero_signal_additions": int(
            concurrent_invasion_metrics["zero_signal_additions"]),
        "concurrent_duplicate_invasion_preexisting_unclaimed_changed_pixels": int(
            concurrent_invasion_metrics[
                "preexisting_unclaimed_changed_pixels"]),
        "concurrent_duplicate_invasion_unclaimed_overlap_additions": int(
            concurrent_invasion_metrics["unclaimed_overlap_additions"]),
        "concurrent_duplicate_invasion_old_identity_set_preserved": bool(
            concurrent_invasion_metrics["old_identity_set_preserved"]),
        "concurrent_duplicate_invasion_donor_named_frame_losses": int(
            concurrent_invasion_metrics["donor_named_frame_losses"]),
        "concurrent_duplicate_invasion_new_duplicate_components": int(
            concurrent_invasion_metrics[
                "new_same_frame_identity_components"]),
        "concurrent_duplicate_invasion_identity_targets": 0,
        "concurrent_duplicate_invasion_track_targets": 0,
        "concurrent_duplicate_invasion_frame_targets": 0,
        "concurrent_duplicate_invasion_coordinate_targets": 0,
        "concurrent_duplicate_invasion_event_targets": 0,
        "post_split_excursion_recovery_audit_rows": int(
            len(post_split_excursion_audit)),
        "post_split_excursion_recovery_tracks_audited": int(
            post_split_excursion_metrics["tracks_audited"]),
        "post_split_excursion_recovery_eligible_proposals": int(
            post_split_excursion_metrics["eligible_proposals"]),
        "post_split_excursion_recovery_applied_proposals": int(
            post_split_excursion_metrics["applied_proposals"]),
        "post_split_excursion_recovery_changed_pixels": int(
            post_split_excursion_metrics["changed_pixels"]),
        "post_split_excursion_recovery_changed_frames": int(
            post_split_excursion_metrics["changed_frames"]),
        "post_split_excursion_recovery_relabelled_foreground_pixels": int(
            post_split_excursion_metrics[
                "relabelled_foreground_pixels"]),
        "post_split_excursion_recovery_raw_supported_additions": int(
            post_split_excursion_metrics["raw_supported_additions"]),
        "post_split_excursion_recovery_removed_foreground_pixels": int(
            post_split_excursion_metrics["removed_foreground_pixels"]),
        "post_split_excursion_recovery_foreground_ledger_changes": int(
            post_split_excursion_metrics["foreground_ledger_changes"]),
        "post_split_excursion_recovery_unclaimed_overlap_additions": int(
            post_split_excursion_metrics["unclaimed_overlap_additions"]),
        "post_split_excursion_recovery_old_identity_set_preserved": bool(
            post_split_excursion_metrics["old_identity_set_preserved"]),
        "post_split_excursion_recovery_new_duplicate_components": int(
            post_split_excursion_metrics[
                "new_same_frame_identity_components"]),
        "post_split_excursion_recovery_identity_targets": 0,
        "post_split_excursion_recovery_track_targets": 0,
        "post_split_excursion_recovery_frame_targets": 0,
        "post_split_excursion_recovery_coordinate_targets": 0,
        "post_split_excursion_recovery_event_targets": 0,
        "transient_misownership_recovery_tracks_audited": int(
            transient_misownership_metrics["audited_tracks"]),
        "transient_misownership_recovery_eligible_proposals": int(
            transient_misownership_metrics["eligible_proposals"]),
        "transient_misownership_recovery_applied_proposals": int(
            transient_misownership_metrics["applied_proposals"]),
        "transient_misownership_recovery_proposal_rows": int(
            len(transient_misownership_proposals)),
        "transient_misownership_recovery_application_rows": int(
            len(transient_misownership_audit)),
        "transient_misownership_recovery_physical_point_rows": int(
            len(transient_misownership_points)),
        "transient_misownership_recovery_changed_pixels": int(
            transient_misownership_metrics["changed_pixels"]),
        "transient_misownership_recovery_changed_frames": int(
            transient_misownership_metrics["changed_frames"]),
        "transient_misownership_recovery_claimed_unclaimed_pixels": int(
            transient_misownership_metrics["claimed_unclaimed_pixels"]),
        "transient_misownership_recovery_added_background_pixels": int(
            transient_misownership_metrics["added_background_pixels"]),
        "transient_misownership_recovery_foreground_added_pixels": int(
            transient_misownership_metrics["foreground_added_pixels"]),
        "transient_misownership_recovery_foreground_removed_pixels": int(
            transient_misownership_metrics["foreground_removed_pixels"]),
        "transient_misownership_recovery_old_identity_set_preserved": bool(
            transient_misownership_metrics["old_identity_set_preserved"]),
        "transient_misownership_recovery_new_identities": int(
            transient_misownership_metrics["new_identities"]),
        "transient_misownership_recovery_identity_targets": 0,
        "transient_misownership_recovery_track_targets": 0,
        "transient_misownership_recovery_frame_targets": 0,
        "transient_misownership_recovery_coordinate_targets": 0,
        "transient_misownership_recovery_event_targets": 0,
        "isolated_unowned_lifetime_tracks_audited": int(
            isolated_lifetime_metrics["proposals_audited"]),
        "isolated_unowned_lifetime_eligible_proposals": int(
            isolated_lifetime_metrics["eligible_proposals"]),
        "isolated_unowned_lifetime_applied_proposals": int(
            isolated_lifetime_metrics["applied_proposals"]),
        "isolated_unowned_lifetime_proposal_rows": int(
            len(isolated_lifetime_proposals)),
        "isolated_unowned_lifetime_application_rows": int(
            len(isolated_lifetime_applications)),
        "isolated_unowned_lifetime_frame_audit_rows": int(
            len(isolated_lifetime_frame_audit)),
        "isolated_unowned_lifetime_physical_point_rows": int(
            len(isolated_lifetime_points)),
        "isolated_unowned_lifetime_changed_pixels": int(
            isolated_lifetime_metrics["changed_pixels"]),
        "isolated_unowned_lifetime_changed_frames": int(
            isolated_lifetime_metrics["changed_frames"]),
        "isolated_unowned_lifetime_claimed_unclaimed_pixels": int(
            isolated_lifetime_metrics["claimed_unclaimed_pixels"]),
        "isolated_unowned_lifetime_raw_supported_additions": int(
            isolated_lifetime_metrics["raw_supported_additions"]),
        "isolated_unowned_lifetime_relabelled_foreground_pixels": int(
            isolated_lifetime_metrics["relabelled_foreground_pixels"]),
        "isolated_unowned_lifetime_foreground_removed_pixels": int(
            isolated_lifetime_metrics["foreground_removed_pixels"]),
        "isolated_unowned_lifetime_old_identity_set_preserved": bool(
            isolated_lifetime_metrics["old_identity_set_preserved"]),
        "isolated_unowned_lifetime_new_identities": int(
            isolated_lifetime_metrics["new_identity_count"]),
        "isolated_unowned_lifetime_new_duplicate_components": int(
            isolated_lifetime_metrics[
                "new_same_frame_identity_components"]),
        "isolated_unowned_lifetime_identity_targets": 0,
        "isolated_unowned_lifetime_track_targets": 0,
        "isolated_unowned_lifetime_frame_targets": 0,
        "isolated_unowned_lifetime_coordinate_targets": 0,
        "isolated_unowned_lifetime_event_targets": 0,
        "boundary_owner_blip_proposals_audited": int(
            boundary_blip_metrics["proposals_audited"]),
        "boundary_owner_blip_eligible_proposals": int(
            boundary_blip_metrics["eligible_proposals"]),
        "boundary_owner_blip_applied_proposals": int(
            boundary_blip_metrics["applied_proposals"]),
        "boundary_owner_blip_proposal_rows": int(
            len(boundary_blip_proposals)),
        "boundary_owner_blip_application_rows": int(
            len(boundary_blip_applications)),
        "boundary_owner_blip_frame_audit_rows": int(
            len(boundary_blip_frame_audit)),
        "boundary_owner_blip_changed_pixels": int(
            boundary_blip_metrics["changed_pixels"]),
        "boundary_owner_blip_changed_frames": int(
            boundary_blip_metrics["changed_frames"]),
        "boundary_owner_blip_changed_unclaimed_pixels": int(
            boundary_blip_metrics["changed_unclaimed_pixels"]),
        "boundary_owner_blip_foreground_changed_pixels": int(
            boundary_blip_metrics["foreground_changed_pixels"]),
        "boundary_owner_blip_foreground_ledger_union_changed_pixels": int(
            boundary_blip_metrics[
                "foreground_ledger_union_changed_pixels"]),
        "boundary_owner_blip_old_identity_set_preserved": bool(
            boundary_blip_metrics["old_identity_set_preserved"]),
        "boundary_owner_blip_donor_named_frame_losses": int(
            boundary_blip_metrics["donor_named_frame_losses"]),
        "boundary_owner_blip_new_duplicate_components": int(
            boundary_blip_metrics["new_same_frame_identity_components"]),
        "boundary_owner_blip_identity_targets": 0,
        "boundary_owner_blip_track_targets": 0,
        "boundary_owner_blip_frame_targets": 0,
        "boundary_owner_blip_coordinate_targets": 0,
        "boundary_owner_blip_event_targets": 0,
        "foreign_owner_terminal_convergence_proposals_audited": int(
            terminal_convergence_metrics["pair_proposals_audited"]),
        "foreign_owner_terminal_convergence_eligible_proposals": int(
            terminal_convergence_metrics["eligible_terminal_convergences"]),
        "foreign_owner_terminal_convergence_applied_proposals": int(
            terminal_convergence_metrics["applied_terminal_convergences"]),
        "foreign_owner_terminal_convergence_proposal_rows": int(
            len(terminal_convergence_proposals)),
        "foreign_owner_terminal_convergence_application_rows": int(
            len(terminal_convergence_applications)),
        "foreign_owner_terminal_convergence_frame_audit_rows": int(
            len(terminal_convergence_frame_audit)),
        "foreign_owner_terminal_convergence_physical_point_rows": int(
            len(terminal_convergence_points)),
        "foreign_owner_terminal_convergence_changed_pixels": int(
            terminal_convergence_metrics["changed_pixels"]),
        "foreign_owner_terminal_convergence_changed_frames": int(
            terminal_convergence_metrics["changed_frames"]),
        "foreign_owner_terminal_convergence_foreground_changed_pixels": int(
            terminal_convergence_metrics["foreground_changed_pixels"]),
        "foreign_owner_terminal_convergence_unclaimed_ledger_changed_pixels": int(
            terminal_convergence_metrics["unclaimed_ledger_changed_pixels"]),
        "foreign_owner_terminal_convergence_zero_signal_additions": int(
            terminal_convergence_metrics["zero_signal_additions"]),
        "foreign_owner_terminal_convergence_old_identity_set_preserved": bool(
            terminal_convergence_metrics["old_identity_set_preserved"]),
        "foreign_owner_terminal_convergence_donor_named_frame_losses": int(
            terminal_convergence_metrics["donor_named_frame_losses"]),
        "foreign_owner_terminal_convergence_new_duplicate_components": int(
            terminal_convergence_metrics[
                "new_same_frame_identity_components"]),
        "foreign_owner_terminal_convergence_identity_targets": 0,
        "foreign_owner_terminal_convergence_track_targets": 0,
        "foreign_owner_terminal_convergence_frame_targets": 0,
        "foreign_owner_terminal_convergence_coordinate_targets": 0,
        "foreign_owner_terminal_convergence_event_targets": 0,
        "terminal_companion_assimilation_tracks_audited": int(
            companion_assimilation_metrics["tracks_audited"]),
        "terminal_companion_assimilation_proposals_audited": int(
            companion_assimilation_metrics[
                "terminal_assimilation_proposals"]),
        "terminal_companion_assimilation_applied_proposals": int(
            companion_assimilation_metrics[
                "applied_terminal_assimilations"]),
        "terminal_companion_assimilation_proposal_rows": int(
            len(companion_assimilation_proposals)),
        "terminal_companion_assimilation_application_rows": int(
            len(companion_assimilation_applications)),
        "terminal_companion_assimilation_frame_audit_rows": int(
            len(companion_assimilation_frame_audit)),
        "terminal_companion_assimilation_physical_point_rows": int(
            len(companion_assimilation_points)),
        "terminal_companion_assimilation_changed_pixels": int(
            companion_assimilation_metrics["changed_pixels"]),
        "terminal_companion_assimilation_changed_frames": int(
            companion_assimilation_metrics["changed_frames"]),
        "terminal_companion_assimilation_foreground_changed_pixels": int(
            companion_assimilation_metrics["foreground_changed_pixels"]),
        "terminal_companion_assimilation_unclaimed_ledger_changed_pixels": int(
            companion_assimilation_metrics[
                "unclaimed_ledger_changed_pixels"]),
        "terminal_companion_assimilation_zero_signal_additions": int(
            companion_assimilation_metrics["zero_signal_additions"]),
        "terminal_companion_assimilation_old_identity_set_preserved": bool(
            companion_assimilation_metrics["old_identity_set_preserved"]),
        "terminal_companion_assimilation_donor_named_frame_losses": int(
            companion_assimilation_metrics["donor_named_frame_losses"]),
        "terminal_companion_assimilation_new_duplicate_components": int(
            companion_assimilation_metrics[
                "new_same_frame_identity_components"]),
        "terminal_companion_assimilation_identity_targets": 0,
        "terminal_companion_assimilation_track_targets": 0,
        "terminal_companion_assimilation_frame_targets": 0,
        "terminal_companion_assimilation_coordinate_targets": 0,
        "terminal_companion_assimilation_event_targets": 0,
        "cross_reference_owner_relay_pairs_audited": int(
            cross_reference_metrics["audited_reference_pairs"]),
        "cross_reference_owner_relay_eligible_relays": int(
            cross_reference_metrics["eligible_reference_relays"]),
        "cross_reference_owner_relay_applied_relays": int(
            cross_reference_metrics["applied_reference_relays"]),
        "cross_reference_owner_relay_proposal_rows": int(
            len(cross_reference_proposals)),
        "cross_reference_owner_relay_application_rows": int(
            len(cross_reference_applications)),
        "cross_reference_owner_relay_frame_audit_rows": int(
            len(cross_reference_frame_audit)),
        "cross_reference_owner_relay_physical_point_rows": int(
            len(cross_reference_points)),
        "cross_reference_owner_relay_changed_label_pixels": int(
            cross_reference_metrics["changed_label_pixels"]),
        "cross_reference_owner_relay_changed_frames": int(
            cross_reference_metrics["changed_frames"]),
        "cross_reference_owner_relay_changed_unclaimed_pixels": int(
            cross_reference_metrics["changed_unclaimed_pixels"]),
        "cross_reference_owner_relay_foreground_ledger_changed_pixels": int(
            cross_reference_metrics["foreground_ledger_changed_pixels"]),
        "cross_reference_owner_relay_foreground_removed_pixels": int(
            cross_reference_metrics["foreground_removed_pixels"]),
        "cross_reference_owner_relay_raw_supported_additions": int(
            cross_reference_metrics["raw_supported_foreground_additions"]),
        "cross_reference_owner_relay_zero_signal_additions": int(
            cross_reference_metrics["zero_signal_additions"]),
        "cross_reference_owner_relay_existing_unclaimed_changed_pixels": int(
            cross_reference_metrics["existing_unclaimed_changed_pixels"]),
        "cross_reference_owner_relay_active_identity_set_preserved": bool(
            cross_reference_metrics["active_identity_set_preserved"]),
        "cross_reference_owner_relay_new_duplicate_components": int(
            cross_reference_metrics["new_duplicate_components"]),
        "cross_reference_owner_relay_identity_targets": 0,
        "cross_reference_owner_relay_track_targets": 0,
        "cross_reference_owner_relay_frame_targets": 0,
        "cross_reference_owner_relay_coordinate_targets": 0,
        "cross_reference_owner_relay_event_targets": 0,
        "cross_reference_owner_relay_region_targets": 0,
        "cross_reference_owner_relay_review_case_targets": 0,
        "recording_boundary_cycle_edges_audited": int(
            len(boundary_cycle_edges)),
        "recording_boundary_cycles_audited": int(len(boundary_cycles)),
        "recording_boundary_cycle_members": int(
            len(boundary_cycle_members)),
        "recording_boundary_cycle_anchor_rows": int(
            len(boundary_cycle_anchors)),
        "recording_boundary_cycle_application_rows": int(
            len(boundary_cycle_applications)),
        "recording_boundary_cycle_applied_cycles": int(
            boundary_cycle_metrics["applied_cycles"]),
        "recording_boundary_cycle_changed_pixels": int(
            boundary_cycle_metrics["changed_pixels"]),
        "recording_boundary_cycle_changed_frames": int(
            boundary_cycle_metrics["changed_frames"]),
        "recording_boundary_cycle_identity_targets": 0,
        "recording_boundary_cycle_track_targets": 0,
        "recording_boundary_cycle_frame_targets": 0,
        "recording_boundary_cycle_coordinate_targets": 0,
        "recording_boundary_cycle_event_targets": 0,
        "recording_boundary_cycle_region_targets": 0,
        "recording_boundary_cycle_review_case_targets": 0,
        "duplicate_soma_rows_audited": int(len(duplicate_soma_before)),
        "duplicate_soma_projection_application_rows": int(
            len(duplicate_projection_applications)),
        "complete_duplicate_lineage_proposal_rows": int(
            len(duplicate_lineage_proposals)),
        "complete_duplicate_lineage_application_rows": int(
            len(duplicate_lineage_applications)),
        "complete_duplicate_lineage_frame_audit_rows": int(
            len(duplicate_lineage_frame_audit)),
        "complete_duplicate_lineage_applied_proposals": int(
            duplicate_lineage_metrics["applied_proposals"]),
        "complete_duplicate_lineage_changed_pixels": int(
            duplicate_lineage_metrics["changed_pixels"]),
        "complete_duplicate_lineage_changed_frames": int(
            duplicate_lineage_metrics["changed_frames"]),
        "complete_duplicate_lineage_new_identities": list(
            duplicate_lineage_metrics["new_identities"]),
        "complete_duplicate_lineage_identity_targets": 0,
        "complete_duplicate_lineage_track_targets": 0,
        "complete_duplicate_lineage_frame_targets": 0,
        "complete_duplicate_lineage_coordinate_targets": 0,
        "complete_duplicate_lineage_event_targets": 0,
        "complete_duplicate_lineage_region_targets": 0,
        "complete_duplicate_lineage_review_case_targets": 0,
        "isolated_pair_recovery_audit_rows": int(len(isolated_pair_audit)),
        "isolated_pair_recovery_applied_proposals": int(
            isolated_pair_metrics["applied_proposals"]),
        "isolated_pair_recovery_applied_track_pairs": int(
            isolated_pair_metrics["applied_track_pairs"]),
        "isolated_pair_recovery_changed_pixels": int(
            isolated_pair_metrics["changed_pixels"]),
        "isolated_pair_recovery_changed_frames": int(
            isolated_pair_metrics["changed_frames"]),
        "isolated_pair_recovery_absorbed_coreless_pixels": int(
            isolated_pair_metrics["absorbed_coreless_pixels"]),
        "isolated_pair_recovery_reverted_duplicate_proposals": int(
            isolated_pair_metrics["reverted_duplicate_proposals"]),
        "isolated_pair_recovery_reverted_crowded_proposals": int(
            isolated_pair_metrics["reverted_crowded_proposals"]),
        "isolated_pair_recovery_identity_targets": 0,
        "isolated_pair_recovery_track_targets": 0,
        "isolated_pair_recovery_frame_targets": 0,
        "isolated_pair_recovery_coordinate_targets": 0,
        "isolated_pair_recovery_event_targets": 0,
        "isolated_pair_recovery_region_targets": 0,
        "isolated_pair_recovery_review_case_targets": 0,
        "recurrent_fusion_pairs_audited": int(len(recurrent_fusion_pairs)),
        "recurrent_fusion_eligible_pairs": int(
            recurrent_fusion_metrics["eligible_pairs"]),
        "recurrent_fusion_applied_pairs": int(
            recurrent_fusion_metrics["applied_pairs"]),
        "recurrent_fusion_absorption_events_audited": int(
            len(recurrent_fusion_absorptions)),
        "recurrent_fusion_verdict_rows": int(
            len(recurrent_fusion_verdicts)),
        "recurrent_fusion_application_rows": int(
            len(recurrent_fusion_applications)),
        "recurrent_fusion_changed_pixels": int(
            recurrent_fusion_metrics["changed_pixels"]),
        "recurrent_fusion_changed_frames": int(
            recurrent_fusion_metrics["changed_frames"]),
        "recurrent_fusion_new_duplicate_components": int(
            recurrent_fusion_metrics["new_duplicate_components"]),
        "recurrent_fusion_identity_targets": 0,
        "recurrent_fusion_track_targets": 0,
        "recurrent_fusion_frame_targets": 0,
        "recurrent_fusion_coordinate_targets": 0,
        "recurrent_fusion_event_targets": 0,
        "recurrent_fusion_region_targets": 0,
        "recurrent_fusion_review_case_targets": 0,
        "exclusive_pair_seat_pairs_audited": int(
            len(exclusive_pair_seats)),
        "exclusive_pair_seat_eligible_pairs": int(
            exclusive_pair_seat_metrics["eligible_pairs"]),
        "exclusive_pair_seat_applied_pairs": int(
            exclusive_pair_seat_metrics["applied_pairs"]),
        "exclusive_pair_seat_application_rows": int(
            len(exclusive_pair_seat_applications)),
        "exclusive_pair_seat_applied_frames": int(
            exclusive_pair_seat_metrics["applied_frames"]),
        "exclusive_pair_seat_changed_pixels": int(
            exclusive_pair_seat_metrics["changed_pixels"]),
        "exclusive_pair_seat_changed_frames": int(
            exclusive_pair_seat_metrics["changed_frames"]),
        "exclusive_pair_seat_new_duplicate_components": int(
            exclusive_pair_seat_metrics["new_duplicate_components"]),
        "exclusive_pair_seat_identity_targets": 0,
        "exclusive_pair_seat_track_targets": 0,
        "exclusive_pair_seat_frame_targets": 0,
        "exclusive_pair_seat_coordinate_targets": 0,
        "exclusive_pair_seat_event_targets": 0,
        "exclusive_pair_seat_region_targets": 0,
        "exclusive_pair_seat_review_case_targets": 0,
        "reciprocal_two_seat_exchange_transitions_audited": int(
            len(reciprocal_exchange_audit)),
        "reciprocal_two_seat_exchange_eligible_exchanges": int(
            reciprocal_exchange_metrics["eligible_exchanges"]),
        "reciprocal_two_seat_exchange_applied_exchanges": int(
            reciprocal_exchange_metrics["applied_exchanges"]),
        "reciprocal_two_seat_exchange_application_rows": int(
            len(reciprocal_exchange_applications)),
        "reciprocal_two_seat_exchange_applied_frames": int(
            reciprocal_exchange_metrics["applied_frames"]),
        "reciprocal_two_seat_exchange_changed_pixels": int(
            reciprocal_exchange_metrics["changed_pixels"]),
        "reciprocal_two_seat_exchange_changed_frames": int(
            reciprocal_exchange_metrics["changed_frames"]),
        "reciprocal_two_seat_exchange_new_duplicate_components": int(
            reciprocal_exchange_metrics["new_duplicate_components"]),
        "reciprocal_two_seat_exchange_identity_targets": 0,
        "reciprocal_two_seat_exchange_track_targets": 0,
        "reciprocal_two_seat_exchange_frame_targets": 0,
        "reciprocal_two_seat_exchange_coordinate_targets": 0,
        "reciprocal_two_seat_exchange_event_targets": 0,
        "reciprocal_two_seat_exchange_region_targets": 0,
        "reciprocal_two_seat_exchange_review_case_targets": 0,
        "bounded_owner_excursion_run_triples_audited": int(
            len(bounded_excursion_audit)),
        "bounded_owner_excursion_eligible_excursions": int(
            bounded_excursion_metrics["eligible_excursions"]),
        "bounded_owner_excursion_applied_excursions": int(
            bounded_excursion_metrics["applied_excursions"]),
        "bounded_owner_excursion_application_rows": int(
            len(bounded_excursion_applications)),
        "bounded_owner_excursion_applied_frames": int(
            bounded_excursion_metrics["applied_frames"]),
        "bounded_owner_excursion_changed_pixels": int(
            bounded_excursion_metrics["changed_pixels"]),
        "bounded_owner_excursion_changed_frames": int(
            bounded_excursion_metrics["changed_frames"]),
        "bounded_owner_excursion_new_duplicate_components": int(
            bounded_excursion_metrics["new_duplicate_components"]),
        "bounded_owner_excursion_identity_targets": 0,
        "bounded_owner_excursion_track_targets": 0,
        "bounded_owner_excursion_frame_targets": 0,
        "bounded_owner_excursion_coordinate_targets": 0,
        "bounded_owner_excursion_event_targets": 0,
        "bounded_owner_excursion_region_targets": 0,
        "bounded_owner_excursion_review_case_targets": 0,
        "recurrent_isolated_alias_clusters_audited": int(
            len(recurrent_alias_audit)),
        "recurrent_isolated_alias_eligible_clusters": int(
            recurrent_alias_metrics["eligible_clusters"]),
        "recurrent_isolated_alias_applied_clusters": int(
            recurrent_alias_metrics["applied_clusters"]),
        "recurrent_isolated_alias_application_rows": int(
            len(recurrent_alias_applications)),
        "recurrent_isolated_alias_applied_frames": int(
            recurrent_alias_metrics["applied_frames"]),
        "recurrent_isolated_alias_changed_pixels": int(
            recurrent_alias_metrics["changed_pixels"]),
        "recurrent_isolated_alias_changed_frames": int(
            recurrent_alias_metrics["changed_frames"]),
        "recurrent_isolated_alias_new_duplicate_components": int(
            recurrent_alias_metrics["new_duplicate_components"]),
        "recurrent_isolated_alias_identity_targets": 0,
        "recurrent_isolated_alias_track_targets": 0,
        "recurrent_isolated_alias_frame_targets": 0,
        "recurrent_isolated_alias_coordinate_targets": 0,
        "recurrent_isolated_alias_event_targets": 0,
        "recurrent_isolated_alias_region_targets": 0,
        "recurrent_isolated_alias_review_case_targets": 0,
        "distinct_history_transitions_audited": int(
            len(distinct_history_proposals)),
        "distinct_history_eligible_transitions": int(
            distinct_history_metrics["eligible_transitions"]),
        "distinct_history_applied_transitions": int(
            distinct_history_metrics["applied_transitions"]),
        "distinct_history_frame_rows": int(len(distinct_history_frames)),
        "distinct_history_changed_pixels": int(
            distinct_history_metrics["changed_pixels"]),
        "distinct_history_changed_frames": int(
            distinct_history_metrics["changed_frames"]),
        "distinct_history_new_duplicate_components": int(
            distinct_history_metrics["new_duplicate_components"]),
        "distinct_history_identity_targets": 0,
        "distinct_history_owner_targets": 0,
        "distinct_history_track_targets": 0,
        "distinct_history_frame_targets": 0,
        "distinct_history_coordinate_targets": 0,
        "distinct_history_event_targets": 0,
        "distinct_history_region_targets": 0,
        "distinct_history_review_case_targets": 0,
        "branched_lineage_proposals_audited": int(
            len(branched_lineage_audit)),
        "branched_lineage_application_rows": int(
            len(branched_lineage_applications)),
        "branched_lineage_applied_atomic_groups": int(
            branched_lineage_metrics["applied_atomic_groups"]),
        "branched_lineage_changed_pixels": int(
            branched_lineage_metrics["changed_pixels"]),
        "branched_lineage_changed_frames": int(
            branched_lineage_metrics["changed_frames"]),
        "branched_lineage_new_duplicate_components": int(
            branched_lineage_metrics["new_duplicate_components"]),
        "branched_lineage_identity_targets": 0,
        "branched_lineage_owner_targets": 0,
        "branched_lineage_track_targets": 0,
        "branched_lineage_frame_targets": 0,
        "branched_lineage_coordinate_targets": 0,
        "branched_lineage_event_targets": 0,
        "branched_lineage_region_targets": 0,
        "branched_lineage_review_case_targets": 0,
        "complete_flip_proposals_audited": int(len(complete_flip_audit)),
        "complete_flip_application_rows": int(
            len(complete_flip_applications)),
        "complete_flip_duplicate_proof_rows": int(
            len(complete_flip_duplicate_proof)),
        "complete_flip_applied_atomic_groups": int(
            complete_flip_metrics["applied_atomic_groups"]),
        "complete_flip_changed_pixels": int(
            complete_flip_metrics["changed_pixels"]),
        "complete_flip_changed_frames": int(
            complete_flip_metrics["changed_frames"]),
        "complete_flip_new_duplicate_components": int(
            complete_flip_metrics["new_duplicate_components"]),
        "complete_flip_lineage_proven_new_duplicate_components": int(
            complete_flip_metrics[
                "lineage_proven_new_duplicate_components"]),
        "complete_flip_identity_targets": 0,
        "complete_flip_owner_targets": 0,
        "complete_flip_track_targets": 0,
        "complete_flip_frame_targets": 0,
        "complete_flip_coordinate_targets": 0,
        "complete_flip_event_targets": 0,
        "complete_flip_region_targets": 0,
        "complete_flip_review_case_targets": 0,
        "projection_seat_transitions_audited": int(
            len(projection_seat_audit)),
        "projection_seat_frame_rows": int(len(projection_seat_frames)),
        "projection_seat_projection_rows": int(
            len(projection_seat_projections)),
        "projection_seat_duplicate_proof_rows": int(
            len(projection_seat_duplicate_proof)),
        "projection_seat_eligible_transitions": int(
            projection_seat_metrics["eligible_transitions"]),
        "projection_seat_applied_transitions": int(
            projection_seat_metrics["applied_transitions"]),
        "projection_seat_changed_pixels": int(
            projection_seat_metrics["changed_pixels"]),
        "projection_seat_changed_frames": int(
            projection_seat_metrics["changed_frames"]),
        "projection_seat_new_duplicate_components": int(
            projection_seat_metrics["new_duplicate_components"]),
        "projection_seat_lineage_proven_new_duplicate_components": int(
            projection_seat_metrics[
                "lineage_proven_new_duplicate_components"]),
        "projection_seat_identity_targets": 0,
        "projection_seat_owner_targets": 0,
        "projection_seat_track_targets": 0,
        "projection_seat_frame_targets": 0,
        "projection_seat_coordinate_targets": 0,
        "projection_seat_event_targets": 0,
        "projection_seat_region_targets": 0,
        "projection_seat_review_case_targets": 0,
        "retroactive_successor_transitions_audited": int(
            len(retroactive_successor_audit)),
        "retroactive_successor_frame_rows": int(
            len(retroactive_successor_frames)),
        "retroactive_successor_duplicate_proof_rows": int(
            len(retroactive_successor_duplicate_proof)),
        "retroactive_successor_eligible_transitions": int(
            retroactive_successor_metrics["eligible_transitions"]),
        "retroactive_successor_applied_transitions": int(
            retroactive_successor_metrics["applied_transitions"]),
        "retroactive_successor_changed_pixels": int(
            retroactive_successor_metrics["changed_pixels"]),
        "retroactive_successor_changed_frames": int(
            retroactive_successor_metrics["changed_frames"]),
        "retroactive_successor_new_duplicate_components": int(
            retroactive_successor_metrics["new_duplicate_components"]),
        "retroactive_successor_identity_targets": 0,
        "retroactive_successor_owner_targets": 0,
        "retroactive_successor_track_targets": 0,
        "retroactive_successor_frame_targets": 0,
        "retroactive_successor_coordinate_targets": 0,
        "retroactive_successor_event_targets": 0,
        "retroactive_successor_region_targets": 0,
        "retroactive_successor_review_case_targets": 0,
        "terminal_split_proposals_audited": int(len(terminal_split_audit)),
        "terminal_split_frame_rows": int(len(terminal_split_frames)),
        "terminal_split_eligible_proposals": int(
            terminal_split_metrics["eligible_proposals"]),
        "terminal_split_applied_proposals": int(
            terminal_split_metrics["applied_proposals"]),
        "terminal_split_terminal_partitions": int(
            terminal_split_metrics["terminal_partitions"]),
        "terminal_split_changed_pixels": int(
            terminal_split_metrics["changed_pixels"]),
        "terminal_split_changed_frames": int(
            terminal_split_metrics["changed_frames"]),
        "terminal_split_new_duplicate_components": int(
            terminal_split_metrics["new_duplicate_components"]),
        "terminal_split_identity_targets": 0,
        "terminal_split_owner_targets": 0,
        "terminal_split_track_targets": 0,
        "terminal_split_frame_targets": 0,
        "terminal_split_coordinate_targets": 0,
        "terminal_split_event_targets": 0,
        "terminal_split_region_targets": 0,
        "terminal_split_review_case_targets": 0,
        "bracketed_invasion_run_triples_audited": int(
            len(bracketed_invasion_audit)),
        "bracketed_invasion_application_rows": int(
            len(bracketed_invasion_applications)),
        "bracketed_invasion_eligible_invasions": int(
            bracketed_invasion_metrics["eligible_invasions"]),
        "bracketed_invasion_applied_invasions": int(
            bracketed_invasion_metrics["applied_invasions"]),
        "bracketed_invasion_applied_frames": int(
            bracketed_invasion_metrics["applied_frames"]),
        "bracketed_invasion_changed_pixels": int(
            bracketed_invasion_metrics["changed_pixels"]),
        "bracketed_invasion_changed_frames": int(
            bracketed_invasion_metrics["changed_frames"]),
        "bracketed_invasion_new_duplicate_components": int(
            bracketed_invasion_metrics["new_duplicate_components"]),
        "bracketed_invasion_identity_targets": 0,
        "bracketed_invasion_owner_targets": 0,
        "bracketed_invasion_track_targets": 0,
        "bracketed_invasion_frame_targets": 0,
        "bracketed_invasion_coordinate_targets": 0,
        "bracketed_invasion_event_targets": 0,
        "bracketed_invasion_region_targets": 0,
        "bracketed_invasion_review_case_targets": 0,
        "terminal_diversion_proposals_audited": int(
            len(terminal_diversion_proposals)),
        "terminal_diversion_application_rows": int(
            len(terminal_diversion_applications)),
        "terminal_diversion_eligible_proposals": int(
            terminal_diversion_metrics["eligible_proposals"]),
        "terminal_diversion_applied_proposals": int(
            terminal_diversion_metrics["applied_proposals"]),
        "terminal_diversion_changed_pixels": int(
            terminal_diversion_metrics["changed_pixels"]),
        "terminal_diversion_changed_frames": int(
            terminal_diversion_metrics["changed_frames"]),
        "terminal_diversion_new_identity_count": int(
            terminal_diversion_metrics["new_identity_count"]),
        "terminal_diversion_new_duplicate_components": int(
            terminal_diversion_metrics["new_duplicate_components"]),
        "terminal_diversion_explained_projection_components": int(
            terminal_diversion_metrics[
                "new_explained_projection_components"]),
        "terminal_diversion_identity_targets": 0,
        "terminal_diversion_owner_targets": 0,
        "terminal_diversion_track_targets": 0,
        "terminal_diversion_frame_targets": 0,
        "terminal_diversion_coordinate_targets": 0,
        "terminal_diversion_event_targets": 0,
        "terminal_diversion_region_targets": 0,
        "terminal_diversion_review_case_targets": 0,
        "delayed_reclaim_proposals_audited": int(
            len(delayed_reclaim_proposals)),
        "delayed_reclaim_application_rows": int(
            len(delayed_reclaim_applications)),
        "delayed_reclaim_eligible_proposals": int(
            delayed_reclaim_metrics["eligible_proposals"]),
        "delayed_reclaim_applied_proposals": int(
            delayed_reclaim_metrics["applied_proposals"]),
        "delayed_reclaim_changed_pixels": int(
            delayed_reclaim_metrics["changed_pixels"]),
        "delayed_reclaim_changed_frames": int(
            delayed_reclaim_metrics["changed_frames"]),
        "delayed_reclaim_new_identity_count": int(
            delayed_reclaim_metrics["new_identity_count"]),
        "delayed_reclaim_new_duplicate_components": int(
            delayed_reclaim_metrics["new_duplicate_components"]),
        "delayed_reclaim_explained_projection_components": int(
            delayed_reclaim_metrics[
                "new_explained_projection_components"]),
        "delayed_reclaim_identity_targets": 0,
        "delayed_reclaim_owner_targets": 0,
        "delayed_reclaim_track_targets": 0,
        "delayed_reclaim_frame_targets": 0,
        "delayed_reclaim_coordinate_targets": 0,
        "delayed_reclaim_event_targets": 0,
        "delayed_reclaim_region_targets": 0,
        "delayed_reclaim_review_case_targets": 0,
        "boundary_excursion_run_triples_audited": int(
            len(boundary_excursion_audit)),
        "boundary_excursion_application_rows": int(
            len(boundary_excursion_applications)),
        "boundary_excursion_eligible_excursions": int(
            boundary_excursion_metrics["eligible_excursions"]),
        "boundary_excursion_applied_excursions": int(
            boundary_excursion_metrics["applied_excursions"]),
        "boundary_excursion_changed_pixels": int(
            boundary_excursion_metrics["changed_pixels"]),
        "boundary_excursion_changed_frames": int(
            boundary_excursion_metrics["changed_frames"]),
        "boundary_excursion_new_identity_count": int(
            boundary_excursion_metrics["new_identity_count"]),
        "boundary_excursion_new_duplicate_components": int(
            boundary_excursion_metrics["new_duplicate_components"]),
        "boundary_excursion_explained_projection_components": int(
            boundary_excursion_metrics[
                "new_explained_projection_components"]),
        "boundary_excursion_identity_targets": 0,
        "boundary_excursion_owner_targets": 0,
        "boundary_excursion_track_targets": 0,
        "boundary_excursion_frame_targets": 0,
        "boundary_excursion_coordinate_targets": 0,
        "boundary_excursion_event_targets": 0,
        "boundary_excursion_region_targets": 0,
        "boundary_excursion_review_case_targets": 0,
        "terminal_assimilation_proposals_audited": int(
            len(terminal_assimilation_audit)),
        "terminal_assimilation_application_rows": int(
            len(terminal_assimilation_applications)),
        "terminal_assimilation_eligible_proposals": int(
            terminal_assimilation_metrics["eligible_proposals"]),
        "terminal_assimilation_applied_proposals": int(
            terminal_assimilation_metrics["applied_proposals"]),
        "terminal_assimilation_changed_pixels": int(
            terminal_assimilation_metrics["changed_pixels"]),
        "terminal_assimilation_changed_frames": int(
            terminal_assimilation_metrics["changed_frames"]),
        "terminal_assimilation_new_identity_count": int(
            terminal_assimilation_metrics["new_identity_count"]),
        "terminal_assimilation_new_duplicate_components": int(
            terminal_assimilation_metrics["new_duplicate_components"]),
        "terminal_assimilation_explained_projection_components": int(
            terminal_assimilation_metrics[
                "new_explained_projection_components"]),
        "terminal_assimilation_identity_targets": 0,
        "terminal_assimilation_owner_targets": 0,
        "terminal_assimilation_track_targets": 0,
        "terminal_assimilation_frame_targets": 0,
        "terminal_assimilation_coordinate_targets": 0,
        "terminal_assimilation_event_targets": 0,
        "terminal_assimilation_region_targets": 0,
        "terminal_assimilation_review_case_targets": 0,
        "mixed_owner_flash_proposals_audited": int(len(mixed_flash_audit)),
        "mixed_owner_flash_application_rows": int(
            len(mixed_flash_applications)),
        "mixed_owner_flash_eligible_proposals": int(
            mixed_flash_metrics["eligible_proposals"]),
        "mixed_owner_flash_applied_proposals": int(
            mixed_flash_metrics["applied_proposals"]),
        "mixed_owner_flash_changed_pixels": int(
            mixed_flash_metrics["changed_pixels"]),
        "mixed_owner_flash_changed_frames": int(
            mixed_flash_metrics["changed_frames"]),
        "mixed_owner_flash_new_identity_count": int(
            mixed_flash_metrics["new_identity_count"]),
        "mixed_owner_flash_new_duplicate_components": int(
            mixed_flash_metrics["new_duplicate_components"]),
        "mixed_owner_flash_identity_targets": 0,
        "mixed_owner_flash_owner_targets": 0,
        "mixed_owner_flash_track_targets": 0,
        "mixed_owner_flash_frame_targets": 0,
        "mixed_owner_flash_coordinate_targets": 0,
        "mixed_owner_flash_event_targets": 0,
        "mixed_owner_flash_region_targets": 0,
        "mixed_owner_flash_review_case_targets": 0,
        "dominant_seat_diversion_base_intervals_audited": int(
            len(dominant_seat_base_audit)),
        "dominant_seat_diversion_proposals_audited": int(
            len(dominant_seat_audit)),
        "dominant_seat_diversion_application_rows": int(
            len(dominant_seat_applications)),
        "dominant_seat_diversion_eligible_proposals": int(
            dominant_seat_metrics["eligible_proposals"]),
        "dominant_seat_diversion_applied_proposals": int(
            dominant_seat_metrics["applied_proposals"]),
        "dominant_seat_diversion_changed_pixels": int(
            dominant_seat_metrics["changed_pixels"]),
        "dominant_seat_diversion_changed_frames": int(
            dominant_seat_metrics["changed_frames"]),
        "dominant_seat_diversion_explained_projection_components": int(
            dominant_seat_metrics["new_explained_projection_components"]),
        "dominant_seat_diversion_unexplained_duplicate_components": int(
            dominant_seat_metrics["new_duplicate_components"]),
        "dominant_seat_diversion_identity_targets": 0,
        "dominant_seat_diversion_owner_targets": 0,
        "dominant_seat_diversion_track_targets": 0,
        "dominant_seat_diversion_frame_targets": 0,
        "dominant_seat_diversion_coordinate_targets": 0,
        "dominant_seat_diversion_event_targets": 0,
        "dominant_seat_diversion_region_targets": 0,
        "dominant_seat_diversion_review_case_targets": 0,
        "ownerless_cohort_gap_proposals_audited": int(
            len(ownerless_cohort_gap_audit)),
        "ownerless_cohort_gap_applied_proposals": int(
            ownerless_cohort_gap_metrics["applied_proposals"]),
        "ownerless_cohort_gap_changed_pixels": int(
            ownerless_cohort_gap_metrics["changed_pixels"]),
        "ownerless_cohort_gap_changed_frames": int(
            ownerless_cohort_gap_metrics["changed_frames"]),
        "ownerless_cohort_gap_changed_identities": int(
            ownerless_cohort_gap_metrics["changed_identities"]),
        "ownerless_cohort_gap_preexisting_assigned_changed_pixels": int(
            ownerless_cohort_gap_metrics[
                "preexisting_assigned_changed_pixels"]),
        "ownerless_cohort_gap_unclaimed_overlap_pixels": int(
            ownerless_cohort_gap_metrics["unclaimed_overlap_pixels"]),
        "ownerless_cohort_gap_zero_signal_additions": int(
            ownerless_cohort_gap_metrics["zero_signal_additions"]),
        "ownerless_cohort_gap_new_duplicate_components": int(
            ownerless_cohort_gap_metrics["new_duplicate_components"]),
        "ownerless_cohort_gap_identity_targets": 0,
        "ownerless_cohort_gap_owner_targets": 0,
        "ownerless_cohort_gap_track_targets": 0,
        "ownerless_cohort_gap_frame_targets": 0,
        "ownerless_cohort_gap_coordinate_targets": 0,
        "ownerless_cohort_gap_event_targets": 0,
        "ownerless_cohort_gap_region_targets": 0,
        "ownerless_cohort_gap_review_case_targets": 0,
        "recording_start_two_seat_encounters_audited": int(
            len(recording_start_two_seat_audit)),
        "recording_start_two_seat_eligible_proposals": int(
            recording_start_two_seat_metrics["eligible_proposals"]),
        "recording_start_two_seat_applied_proposals": int(
            recording_start_two_seat_metrics["applied_proposals"]),
        "recording_start_two_seat_changed_pixels": int(
            recording_start_two_seat_metrics["changed_pixels"]),
        "recording_start_two_seat_changed_frames": int(
            recording_start_two_seat_metrics["changed_frames"]),
        "recording_start_two_seat_explained_projection_components_added": int(
            recording_start_two_seat_metrics[
                "explained_projection_components_added"]),
        "recording_start_two_seat_unexplained_duplicate_components_added": int(
            recording_start_two_seat_metrics[
                "unexplained_duplicate_components_added"]),
        "recording_start_two_seat_identity_targets": 0,
        "recording_start_two_seat_owner_targets": 0,
        "recording_start_two_seat_track_targets": 0,
        "recording_start_two_seat_frame_targets": 0,
        "recording_start_two_seat_coordinate_targets": 0,
        "recording_start_two_seat_event_targets": 0,
        "recording_start_two_seat_region_targets": 0,
        "recording_start_two_seat_review_case_targets": 0,
        "gap_tolerant_reciprocal_exchange_pairs_audited": int(
            len(gap_exchange_audit)),
        "gap_tolerant_reciprocal_exchange_eligible_exchanges": int(
            gap_exchange_metrics["eligible_exchanges"]),
        "gap_tolerant_reciprocal_exchange_applied_exchanges": int(
            gap_exchange_metrics["applied_exchanges"]),
        "gap_tolerant_reciprocal_exchange_application_rows": int(
            len(gap_exchange_applications)),
        "gap_tolerant_reciprocal_exchange_changed_pixels": int(
            gap_exchange_metrics["changed_pixels"]),
        "gap_tolerant_reciprocal_exchange_changed_frames": int(
            gap_exchange_metrics["changed_frames"]),
        "gap_tolerant_reciprocal_exchange_new_duplicate_components": int(
            gap_exchange_metrics["new_duplicate_components"]),
        "gap_tolerant_reciprocal_exchange_identity_targets": 0,
        "gap_tolerant_reciprocal_exchange_owner_targets": 0,
        "gap_tolerant_reciprocal_exchange_track_targets": 0,
        "gap_tolerant_reciprocal_exchange_frame_targets": 0,
        "gap_tolerant_reciprocal_exchange_coordinate_targets": 0,
        "gap_tolerant_reciprocal_exchange_event_targets": 0,
        "gap_tolerant_reciprocal_exchange_region_targets": 0,
        "gap_tolerant_reciprocal_exchange_review_case_targets": 0,
        "reconnected_companion_partition_reconnections_audited": int(
            reconnected_partition_metrics["reconnections_audited"]),
        "reconnected_companion_partition_candidate_pairings_audited": int(
            len(reconnected_partition_audit)),
        "reconnected_companion_partition_eligible_partitions": int(
            reconnected_partition_metrics["eligible_partitions"]),
        "reconnected_companion_partition_applied_partitions": int(
            reconnected_partition_metrics["applied_partitions"]),
        "reconnected_companion_partition_application_rows": int(
            len(reconnected_partition_applications)),
        "reconnected_companion_partition_changed_pixels": int(
            reconnected_partition_metrics["changed_pixels"]),
        "reconnected_companion_partition_changed_frames": int(
            reconnected_partition_metrics["changed_frames"]),
        "reconnected_companion_partition_new_duplicate_components": int(
            reconnected_partition_metrics["new_duplicate_components"]),
        "reconnected_companion_partition_identity_targets": 0,
        "reconnected_companion_partition_owner_targets": 0,
        "reconnected_companion_partition_track_targets": 0,
        "reconnected_companion_partition_frame_targets": 0,
        "reconnected_companion_partition_coordinate_targets": 0,
        "reconnected_companion_partition_event_targets": 0,
        "reconnected_companion_partition_region_targets": 0,
        "reconnected_companion_partition_review_case_targets": 0,
        "component_continuity_takeover_transitions_audited": int(
            continuity_takeover_metrics["transitions_audited"]),
        "component_continuity_takeover_eligible_takeovers": int(
            continuity_takeover_metrics["eligible_takeovers"]),
        "component_continuity_takeover_applied_takeovers": int(
            continuity_takeover_metrics["applied_takeovers"]),
        "component_continuity_takeover_application_rows": int(
            len(continuity_takeover_applications)),
        "component_continuity_takeover_changed_pixels": int(
            continuity_takeover_metrics["changed_pixels"]),
        "component_continuity_takeover_changed_frames": int(
            continuity_takeover_metrics["changed_frames"]),
        "component_continuity_takeover_new_duplicate_components": int(
            continuity_takeover_metrics["new_duplicate_components"]),
        "component_continuity_takeover_identity_targets": 0,
        "component_continuity_takeover_owner_targets": 0,
        "component_continuity_takeover_track_targets": 0,
        "component_continuity_takeover_frame_targets": 0,
        "component_continuity_takeover_coordinate_targets": 0,
        "component_continuity_takeover_event_targets": 0,
        "component_continuity_takeover_region_targets": 0,
        "component_continuity_takeover_review_case_targets": 0,
        "right_censored_seat_partition_transitions_audited": int(
            right_censored_partition_metrics["transitions_audited"]),
        "right_censored_seat_partition_eligible_transitions": int(
            right_censored_partition_metrics["eligible_transitions"]),
        "right_censored_seat_partition_applied_transitions": int(
            right_censored_partition_metrics["applied_transitions"]),
        "right_censored_seat_partition_frame_rows": int(
            len(right_censored_partition_frames)),
        "right_censored_seat_partition_changed_pixels": int(
            right_censored_partition_metrics["changed_pixels"]),
        "right_censored_seat_partition_changed_frames": int(
            right_censored_partition_metrics["changed_frames"]),
        "right_censored_seat_partition_new_duplicate_components": int(
            right_censored_partition_metrics["new_duplicate_components"]),
        "right_censored_seat_partition_identity_targets": 0,
        "right_censored_seat_partition_owner_targets": 0,
        "right_censored_seat_partition_track_targets": 0,
        "right_censored_seat_partition_frame_targets": 0,
        "right_censored_seat_partition_coordinate_targets": 0,
        "right_censored_seat_partition_event_targets": 0,
        "right_censored_seat_partition_region_targets": 0,
        "right_censored_seat_partition_review_case_targets": 0,
        "right_censored_reciprocal_exchange_pairs_audited": int(
            right_censored_exchange_metrics["pairs_audited"]),
        "right_censored_reciprocal_exchange_eligible_exchanges": int(
            right_censored_exchange_metrics["eligible_exchanges"]),
        "right_censored_reciprocal_exchange_applied_exchanges": int(
            right_censored_exchange_metrics["applied_exchanges"]),
        "right_censored_reciprocal_exchange_changed_pixels": int(
            right_censored_exchange_metrics["changed_pixels"]),
        "right_censored_reciprocal_exchange_changed_frames": int(
            right_censored_exchange_metrics["changed_frames"]),
        "right_censored_reciprocal_exchange_new_duplicate_components": int(
            right_censored_exchange_metrics["new_duplicate_components"]),
        "right_censored_reciprocal_exchange_identity_targets": 0,
        "right_censored_reciprocal_exchange_owner_targets": 0,
        "right_censored_reciprocal_exchange_track_targets": 0,
        "right_censored_reciprocal_exchange_frame_targets": 0,
        "right_censored_reciprocal_exchange_coordinate_targets": 0,
        "right_censored_reciprocal_exchange_event_targets": 0,
        "right_censored_reciprocal_exchange_region_targets": 0,
        "right_censored_reciprocal_exchange_review_case_targets": 0,
        "persistent_single_owner_flash_proposals_audited": int(
            persistent_flash_metrics["proposals_audited"]),
        "persistent_single_owner_flash_eligible_proposals": int(
            persistent_flash_metrics["eligible_proposals"]),
        "persistent_single_owner_flash_applied_proposals": int(
            persistent_flash_metrics["applied_proposals"]),
        "persistent_single_owner_flash_application_rows": int(
            len(persistent_flash_applications)),
        "persistent_single_owner_flash_changed_pixels": int(
            persistent_flash_metrics["changed_pixels"]),
        "persistent_single_owner_flash_changed_frames": int(
            persistent_flash_metrics["changed_frames"]),
        "persistent_single_owner_flash_new_duplicate_components": int(
            persistent_flash_metrics["new_duplicate_components"]),
        "persistent_single_owner_flash_identity_targets": 0,
        "persistent_single_owner_flash_owner_targets": 0,
        "persistent_single_owner_flash_track_targets": 0,
        "persistent_single_owner_flash_frame_targets": 0,
        "persistent_single_owner_flash_coordinate_targets": 0,
        "persistent_single_owner_flash_event_targets": 0,
        "persistent_single_owner_flash_region_targets": 0,
        "persistent_single_owner_flash_review_case_targets": 0,
        "delayed_owner_projection_flash_base_proposals_audited": int(
            len(delayed_projection_flash_base_audit)),
        "delayed_owner_projection_flash_proposals_audited": int(
            delayed_projection_flash_metrics["proposals_audited"]),
        "delayed_owner_projection_flash_eligible_proposals": int(
            delayed_projection_flash_metrics["eligible_proposals"]),
        "delayed_owner_projection_flash_applied_proposals": int(
            delayed_projection_flash_metrics["applied_proposals"]),
        "delayed_owner_projection_flash_application_rows": int(
            len(delayed_projection_flash_applications)),
        "delayed_owner_projection_flash_changed_pixels": int(
            delayed_projection_flash_metrics["changed_pixels"]),
        "delayed_owner_projection_flash_changed_frames": int(
            delayed_projection_flash_metrics["changed_frames"]),
        "delayed_owner_projection_flash_explained_components": int(
            delayed_projection_flash_metrics[
                "new_explained_projection_components"]),
        "delayed_owner_projection_flash_new_duplicate_components": int(
            delayed_projection_flash_metrics["new_duplicate_components"]),
        "delayed_owner_projection_flash_identity_targets": 0,
        "delayed_owner_projection_flash_owner_targets": 0,
        "delayed_owner_projection_flash_track_targets": 0,
        "delayed_owner_projection_flash_frame_targets": 0,
        "delayed_owner_projection_flash_coordinate_targets": 0,
        "delayed_owner_projection_flash_event_targets": 0,
        "delayed_owner_projection_flash_region_targets": 0,
        "delayed_owner_projection_flash_review_case_targets": 0,
        "anchored_projection_owner_relay_proposals_audited": int(
            anchored_relay_metrics["proposals_audited"]),
        "anchored_projection_owner_relay_eligible_proposals": int(
            anchored_relay_metrics["eligible_proposals"]),
        "anchored_projection_owner_relay_applied_proposals": int(
            anchored_relay_metrics["applied_proposals"]),
        "anchored_projection_owner_relay_application_rows": int(
            len(anchored_relay_applications)),
        "anchored_projection_owner_relay_changed_pixels": int(
            anchored_relay_metrics["changed_pixels"]),
        "anchored_projection_owner_relay_changed_frames": int(
            anchored_relay_metrics["changed_frames"]),
        "anchored_projection_owner_relay_new_duplicate_components": int(
            anchored_relay_metrics["new_duplicate_components"]),
        "anchored_projection_owner_relay_target_counts": dict(
            anchored_relay_metrics["target_counts"]),
        "ephemeral_alias_retirement_identities_audited": int(
            alias_retirement_metrics["identities_audited"]),
        "ephemeral_alias_retirement_eligible_proposals": int(
            alias_retirement_metrics["eligible_proposals"]),
        "ephemeral_alias_retirement_applied_proposals": int(
            alias_retirement_metrics["applied_proposals"]),
        "ephemeral_alias_retirement_changed_pixels": int(
            alias_retirement_metrics["changed_pixels"]),
        "ephemeral_alias_retirement_changed_frames": int(
            alias_retirement_metrics["changed_frames"]),
        "ephemeral_alias_retirement_removed_identity_count": int(
            alias_retirement_metrics["removed_identity_count"]),
        "ephemeral_alias_retirement_proof_exact": bool(
            alias_retirement_metrics["retirement_proof_exact"]),
        "ephemeral_alias_retirement_target_counts": dict(
            alias_retirement_metrics["target_counts"]),
        "anchored_projection_raw_gap_proved_relays": int(
            anchored_raw_gap_metrics["proved_relays_consumed"]),
        "anchored_projection_raw_gap_applied_proposals": int(
            anchored_raw_gap_metrics["applied_proposals"]),
        "anchored_projection_raw_gap_changed_pixels": int(
            anchored_raw_gap_metrics["changed_pixels"]),
        "anchored_projection_raw_gap_changed_frames": int(
            anchored_raw_gap_metrics["changed_frames"]),
        "anchored_projection_raw_gap_zero_signal_additions": int(
            anchored_raw_gap_metrics["zero_signal_additions"]),
        "anchored_projection_raw_gap_target_counts": dict(
            anchored_raw_gap_metrics["target_counts"]),
        "retirement_aware_persistence": retirement_persistence_metrics,
        "terminal_projection_chain_physical_tracks_audited": int(
            terminal_projection_chain_metrics["physical_tracks_audited"]),
        "terminal_projection_chain_projection_parents_audited": int(
            terminal_projection_chain_metrics["projection_parents_audited"]),
        "terminal_projection_chain_eligible_proposals": int(
            terminal_projection_chain_metrics["eligible_proposals"]),
        "terminal_projection_chain_applied_proposals": int(
            terminal_projection_chain_metrics["applied_proposals"]),
        "terminal_projection_chain_application_rows": int(
            len(terminal_projection_chain_applications)),
        "terminal_projection_chain_changed_pixels": int(
            terminal_projection_chain_metrics["changed_pixels"]),
        "terminal_projection_chain_changed_frames": int(
            terminal_projection_chain_metrics["changed_frames"]),
        "terminal_projection_chain_explained_components": int(
            terminal_projection_chain_metrics[
                "new_explained_projection_components"]),
        "terminal_projection_chain_new_duplicate_components": int(
            terminal_projection_chain_metrics["new_duplicate_components"]),
        "terminal_projection_chain_target_counts": dict(
            terminal_projection_chain_metrics["target_counts"]),
        "body_scale_ownerless_tracks_audited": int(
            body_scale_ownerless_metrics["tracks_audited"]),
        "body_scale_ownerless_eligible_proposals": int(
            body_scale_ownerless_metrics["eligible_proposals"]),
        "body_scale_ownerless_applied_proposals": int(
            body_scale_ownerless_metrics["applied_proposals"]),
        "body_scale_ownerless_application_rows": int(
            len(body_scale_ownerless_applications)),
        "body_scale_ownerless_changed_pixels": int(
            body_scale_ownerless_metrics["changed_pixels"]),
        "body_scale_ownerless_changed_frames": int(
            body_scale_ownerless_metrics["changed_frames"]),
        "body_scale_ownerless_new_identity_count": int(
            body_scale_ownerless_metrics["new_identity_count"]),
        "body_scale_ownerless_new_duplicate_components": int(
            body_scale_ownerless_metrics["new_duplicate_components"]),
        "body_scale_ownerless_target_counts": dict(
            body_scale_ownerless_metrics["target_counts"]),
        "terminal_boundary_seat_disappearances_audited": int(
            terminal_boundary_seat_metrics["terminal_disappearances_audited"]),
        "terminal_boundary_seat_eligible_proposals": int(
            terminal_boundary_seat_metrics["eligible_proposals"]),
        "terminal_boundary_seat_applied_proposals": int(
            terminal_boundary_seat_metrics["applied_proposals"]),
        "terminal_boundary_seat_frame_rows": int(
            len(terminal_boundary_seat_frames)),
        "terminal_boundary_seat_changed_pixels": int(
            terminal_boundary_seat_metrics["changed_pixels"]),
        "terminal_boundary_seat_changed_frames": int(
            terminal_boundary_seat_metrics["changed_frames"]),
        "terminal_boundary_seat_new_duplicate_components": int(
            terminal_boundary_seat_metrics["new_duplicate_components"]),
        "terminal_boundary_seat_target_counts": dict(
            terminal_boundary_seat_metrics["target_counts"]),
        "bracketed_ownerless_seat_identity_gaps_audited": int(
            bracketed_ownerless_seat_metrics["identity_gaps_audited"]),
        "bracketed_ownerless_seat_eligible_gaps": int(
            bracketed_ownerless_seat_metrics["eligible_gaps"]),
        "bracketed_ownerless_seat_changed_pixels": int(
            bracketed_ownerless_seat_metrics["changed_pixels"]),
        "bracketed_ownerless_seat_changed_frames": int(
            bracketed_ownerless_seat_metrics["changed_frames"]),
        "bracketed_ownerless_seat_new_identity_count": int(
            bracketed_ownerless_seat_metrics["new_identity_count"]),
        "bracketed_ownerless_seat_new_duplicate_components": int(
            bracketed_ownerless_seat_metrics["new_duplicate_components"]),
        "bracketed_ownerless_seat_target_counts": dict(
            bracketed_ownerless_seat_metrics["target_counts"]),
        "bracketed_ownerless_seat_frame_rows": int(
            len(bracketed_ownerless_seat_frames)),
        "active_identities": int(len(set(map(int, np.unique(labels))) - {0})),
        "assigned_pixels": int(np.count_nonzero(labels)),
        "unclaimed_pixels": int(np.count_nonzero(unclaimed)),
        "field_recovered_foreground_pixels": int(np.count_nonzero(
            final_foreground & ~accepted_foreground)),
        "foreground_pixels_removed": 0,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return labels, unclaimed, summary
