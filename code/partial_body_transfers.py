from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from model_review import (TransferParams, TwoCoreParams,
                          audit_identity_body_transfers,
                          audit_two_core_objects, eroded_cores,
                          group_two_core_runs, identity_components,
                          link_transfers_to_two_core_hosts)


CONNECTIVITY = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class PartialBodyTransferParams:
    """Evidence gates for a soma transferred while its old name still exists."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    transfer: TransferParams = field(default_factory=TransferParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    minimum_partition_px: int = 12


@dataclass(frozen=True)
class IdentityLedgerParams:
    """Evidence needed before a vanished identity is declared hidden in a host."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    transfer: TransferParams = field(default_factory=TransferParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    minimum_partition_px: int = 12
    minimum_prior_core_px: int = 12
    minimum_established_history_frames: int = 7
    strong_two_core_score: float = 0.30
    strong_prior_core_fraction: float = 0.40
    strong_direct_transfer_fraction: float = 0.80
    minimum_donor_core_ancestry: float = 0.55
    minimum_host_core_ancestry: float = 0.40


@dataclass(frozen=True)
class DelayedBookendParams:
    """Proof that two core slots resolve into two identities at another time."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    maximum_bookend_frames: int = 16
    bookend_search_radius_px: float = 10.0
    minimum_bookend_identity_history_frames: int = 3
    minimum_new_bookend_future_frames: int = 3
    minimum_new_bookend_two_core_score: float = 0.60
    minimum_bookend_core_px: int = 12
    minimum_partition_px: int = 12
    strong_two_core_score: float = 0.60
    moderate_two_core_score: float = 0.34
    moderate_core_separation_px: float = 16.0
    distinct_core_intensity_ratio: float = 0.90


@dataclass(frozen=True)
class PersistentSlotParams:
    """Carry a proved donor/host core pairing into later two-core runs."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    maximum_reservation_gap_frames: int = 48
    maximum_core_shift_px: float = 24.0
    core_shift_per_frame_px: float = 0.35
    minimum_partition_px: int = 12
    maximum_area_log2_change: float = 1.0
    minimum_core_fraction_ratio: float = 0.55
    maximum_core_fraction_ratio: float = 1.8


@dataclass(frozen=True)
class BidirectionalLedgerParams:
    """Carry a proved physical two-soma pairing over the whole movie."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    maximum_identity_gap_frames: int = 101
    maximum_core_shift_px: float = 16.0
    core_shift_per_frame_px: float = 0.25
    minimum_partition_px: int = 12
    minimum_confirmed_pair_events: int = 1
    minimum_confirmed_pair_frames: int = 2
    minimum_strong_core_score: float = 0.30
    minimum_strong_core_separation_px: float = 15.0


@dataclass(frozen=True)
class DualBookendParams:
    """Resolve a merged object from two physical identities at either bookend."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    maximum_bookend_frames: int = 16
    bookend_search_radius_px: float = 12.0
    minimum_identity_support_frames: int = 3
    minimum_bookend_core_px: int = 8
    minimum_partition_px: int = 12
    minimum_two_core_score: float = 0.30
    minimum_core_separation_px: float = 15.0
    maximum_existing_core_px: int = 18


@dataclass(frozen=True)
class GlobalPairLedgerParams:
    """Carry recurrent, proved identity pairs through distant re-merges."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    maximum_identity_gap_frames: int = 101
    maximum_core_shift_px: float = 18.0
    core_shift_per_frame_px: float = 0.35
    minimum_partition_px: int = 12
    minimum_two_core_score: float = 0.40
    moderate_two_core_score: float = 0.30
    moderate_core_separation_px: float = 15.0
    maximum_existing_core_px: int = 18
    minimum_proof_runs: int = 2
    minimum_proof_frames: int = 3


@dataclass(frozen=True)
class PredecessorCoreParams:
    """Recover names when two distinct prior somas become one outline."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    minimum_partition_px: int = 12
    minimum_core_ancestry_fraction: float = 0.60
    minimum_identity_history_frames: int = 7
    maximum_existing_core_px: int = 18


@dataclass(frozen=True)
class ShapeProvenTwoCoreParams:
    """Split exceptionally clear two-soma objects without inventing a cell name."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    minimum_two_core_score: float = 0.80
    minimum_core_intensity_ratio: float = 0.80
    minimum_core_separation_px: float = 12.0
    maximum_reference_distance_px: float = 36.0
    minimum_reference_history_frames: int = 3
    maximum_reference_core_px: int = 18
    minimum_partition_px: int = 12


@dataclass(frozen=True)
class RecurrentCoreSlotParams:
    """Use a recurring core slot's established alias in clear two-soma runs."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    maximum_reference_distance_px: float = 4.0
    minimum_reference_history_frames: int = 7
    minimum_reference_core_px: int = 18
    maximum_existing_core_px: int = 18
    minimum_two_core_score: float = 0.45
    minimum_partition_px: int = 12


@dataclass(frozen=True)
class StructuralFallbackParams:
    """Conservative structural proof for unresolved, clear two-soma objects."""

    two_core: TwoCoreParams = field(default_factory=TwoCoreParams)
    maximum_run_gap_frames: int = 2
    maximum_run_shift_px_per_frame: float = 18.0
    minimum_partition_px: int = 12
    strong_two_core_score: float = 0.60
    strong_core_separation_px: float = 15.0
    strong_core_intensity_ratio: float = 0.50
    repeated_pair_score: float = 0.45
    repeated_pair_intensity_ratio: float = 0.89
    repeated_pair_minimum_frames: int = 3
    maximum_reference_distance_px: float = 18.0
    minimum_reference_history_frames: int = 7
    maximum_existing_core_px: int = 18


def _matching_run(event: pd.Series, runs: pd.DataFrame) -> pd.Series | None:
    matches = runs[
        (runs.identity == int(event.host_identity))
        & (runs.first_imagej_frame <= int(event.imagej_frame) + 1)
        & (runs.last_imagej_frame >= int(event.imagej_frame) - 1)
    ].copy()
    if matches.empty:
        return None
    frame = int(event.imagej_frame)
    matches["temporal_distance"] = np.maximum(
        matches.first_imagej_frame - frame,
        np.maximum(frame - matches.last_imagej_frame, 0))
    return matches.sort_values(
        ["temporal_distance", "maximum_two_core_score"],
        ascending=[True, False]).iloc[0]


def detect_partial_body_transfer_intervals(
        labels: np.ndarray, raw: np.ndarray,
        params: PartialBodyTransferParams = PartialBodyTransferParams(),
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect body transfers that a disappearance-only rule cannot see."""
    objects = audit_two_core_objects(labels, raw, params.two_core)
    transfers = audit_identity_body_transfers(labels, params.transfer)
    transfers = link_transfers_to_two_core_hosts(transfers, objects)
    runs = group_two_core_runs(
        objects, params.maximum_run_gap_frames,
        params.maximum_run_shift_px_per_frame)

    rows: list[dict] = []
    partial = transfers[
        transfers.donor_still_present & transfers.host_has_two_cores]
    for event_index, event in partial.iterrows():
        run = _matching_run(event, runs)
        if run is None:
            continue
        rows.append({
            **event.to_dict(),
            "source_event_index": int(event_index),
            "two_core_run_id": str(run.run_id),
            "run_first_imagej_frame": int(run.first_imagej_frame),
            "run_last_imagej_frame": int(run.last_imagej_frame),
            "run_representative_imagej_frame": int(
                run.representative_imagej_frame),
            "run_maximum_two_core_score": float(run.maximum_two_core_score),
        })
    intervals = pd.DataFrame(rows)
    if len(intervals):
        intervals = intervals.sort_values(
            ["imagej_frame", "body_transfer_score"],
            ascending=[True, False]).reset_index(drop=True)
        intervals["partial_transfer_id"] = [
            f"PT{number:04d}" for number in range(1, len(intervals) + 1)]
    return intervals, objects, transfers


def _consecutive_prior_presence(labels: np.ndarray, t: int,
                                identity: int) -> int:
    frames = 0
    for frame in range(t - 1, -1, -1):
        if not np.any(labels[frame] == identity):
            break
        frames += 1
    return frames


def _core_ancestry(labels: np.ndarray, objects: pd.DataFrame,
                   event: pd.Series) -> dict | None:
    """Measure whether separate host cores came from donor and host bodies."""
    t = int(event.t)
    choices = objects[
        (objects.imagej_frame == int(event.imagej_frame))
        & (objects.identity == int(event.host_identity))].copy()
    if choices.empty:
        return None
    choices["transfer_distance"] = np.hypot(
        choices.centre_y - float(event.transfer_y),
        choices.centre_x - float(event.transfer_x))
    obj = choices.sort_values(
        ["transfer_distance", "two_core_score"],
        ascending=[True, False]).iloc[0]
    host_mask = _component_mask(
        labels[t], int(event.host_identity), int(obj.component))
    if host_mask is None:
        return None
    cores = eroded_cores(host_mask, 1)
    if len(cores) < 2:
        return None
    core_masks = (cores[0][1], cores[1][1])
    previous_donor = labels[t - 1] == int(event.donor_identity)
    previous_host = labels[t - 1] == int(event.host_identity)
    direct_transfer = previous_donor & (
        labels[t] == int(event.host_identity))
    donor_overlap = [float(np.mean(direct_transfer[core]))
                     for core in core_masks]
    host_overlap = [float(np.mean(previous_host[core]))
                    for core in core_masks]
    assignment_costs = [
        donor_overlap[0] + host_overlap[1],
        donor_overlap[1] + host_overlap[0],
    ]
    donor_core_index = int(np.argmax(assignment_costs))
    host_core_index = 1 - donor_core_index
    prior_cores = eroded_cores(previous_donor, 1)
    prior_core_px = int(prior_cores[0][0]) if prior_cores else 0
    prior_area = max(int(previous_donor.sum()), 1)
    return {
        "host_object_id": str(obj.object_id),
        "host_component": int(obj.component),
        "host_two_core_score": float(obj.two_core_score),
        "prior_donor_core_px": prior_core_px,
        "prior_donor_core_fraction": float(prior_core_px / prior_area),
        "donor_core_ancestry": float(donor_overlap[donor_core_index]),
        "host_core_ancestry": float(host_overlap[host_core_index]),
        "opposite_donor_core_ancestry": float(
            donor_overlap[host_core_index]),
        "opposite_host_core_ancestry": float(
            host_overlap[donor_core_index]),
    }


def detect_identity_ledger_transfers(
        labels: np.ndarray, raw: np.ndarray,
        params: IdentityLedgerParams = IdentityLedgerParams(),
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find both partial and complete identity losses into two-core hosts.

    A complete disappearance is not treated as death. It is eligible only when the
    preceding body has a real eroded core, the two present host cores retain distinct
    donor/host ancestry, and either identity history or unusually strong structural
    evidence proves that the vanished name represented a cell.
    """
    objects = audit_two_core_objects(labels, raw, params.two_core)
    transfers = audit_identity_body_transfers(labels, params.transfer)
    transfers = link_transfers_to_two_core_hosts(transfers, objects)
    runs = group_two_core_runs(
        objects, params.maximum_run_gap_frames,
        params.maximum_run_shift_px_per_frame)
    rows: list[dict] = []
    for event_index, event in transfers[
            transfers.host_has_two_cores].iterrows():
        ancestry = _core_ancestry(labels, objects, event)
        if ancestry is None:
            continue
        history = _consecutive_prior_presence(
            labels, int(event.t), int(event.donor_identity))
        established = (
            history >= params.minimum_established_history_frames)
        strong_structure = (
            float(ancestry["host_two_core_score"])
            >= params.strong_two_core_score)
        strong_body = (
            float(ancestry["prior_donor_core_fraction"])
            >= params.strong_prior_core_fraction
            and float(event.direct_transfer_fraction)
            >= params.strong_direct_transfer_fraction)
        donor_ancestry_passed = (
            int(ancestry["prior_donor_core_px"])
            >= params.minimum_prior_core_px
            and float(ancestry["donor_core_ancestry"])
            >= params.minimum_donor_core_ancestry)
        ancestry_passed = (
            donor_ancestry_passed
            and float(ancestry["host_core_ancestry"])
            >= params.minimum_host_core_ancestry)
        eligible = bool(
            (bool(event.donor_still_present) and donor_ancestry_passed)
            or (not bool(event.donor_still_present)
                and ancestry_passed
                and (established or strong_structure or strong_body)))
        run = _matching_run(event, runs)
        if run is None:
            continue
        rows.append({
            **event.to_dict(), **ancestry,
            "source_event_index": int(event_index),
            "consecutive_prior_presence_frames": int(history),
            "established_history_gate": bool(established),
            "strong_two_core_gate": bool(strong_structure),
            "strong_prior_body_gate": bool(strong_body),
            "core_ancestry_gate": bool(ancestry_passed),
            "ledger_eligible": eligible,
            "ledger_evidence_score": float(
                2.0 * bool(event.donor_still_present)
                + np.log1p(history) / 4.0
                + float(ancestry["donor_core_ancestry"])
                + float(ancestry["host_core_ancestry"])
                + float(ancestry["host_two_core_score"])),
            "two_core_run_id": str(run.run_id),
            "run_first_imagej_frame": int(run.first_imagej_frame),
            "run_last_imagej_frame": int(run.last_imagej_frame),
            "run_representative_imagej_frame": int(
                run.representative_imagej_frame),
        })
    ledger = pd.DataFrame(rows)
    if len(ledger):
        ledger = ledger.sort_values(
            ["ledger_eligible", "ledger_evidence_score"],
            ascending=[False, False]).reset_index(drop=True)
    return ledger, objects, transfers


def correct_identity_ledger_transfers(
        labels: np.ndarray, raw: np.ndarray,
        params: IdentityLedgerParams = IdentityLedgerParams(),
        detection_labels: np.ndarray | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Reserve established identities through both partial and complete losses.

    Each two-core run represents physical cell slots, not a sequence of identity
    births. If multiple aliases later flow into the same host run, only the strongest
    established donor becomes the canonical second identity. This prevents the repair
    itself from interpreting every one-frame name as another cell.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    ledger, objects, _ = detect_identity_ledger_transfers(source, raw, params)
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    selected_rows: list[dict] = []
    frame_rows: list[dict] = []
    if ledger.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred

    eligible = ledger[ledger.ledger_eligible].copy()
    if eligible.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred
    eligible = eligible.sort_values(
        ["two_core_run_id", "ledger_evidence_score",
         "consecutive_prior_presence_frames", "imagej_frame"],
        ascending=[True, False, False, True])

    for run_id, alternatives in eligible.groupby("two_core_run_id"):
        event = alternatives.iloc[0]
        event_t = int(event.t)
        donor = int(event.donor_identity)
        host_identity = int(event.host_identity)
        previous_donor = source[event_t - 1] == donor
        previous_host = source[event_t - 1] == host_identity
        if not np.any(previous_donor) or not np.any(previous_host):
            continue
        donor_reference = _centroid(previous_donor)
        host_reference = _centroid(previous_host)
        transfer = previous_donor & (source[event_t] == host_identity)

        first = int(event.run_first_imagej_frame)
        last = int(event.run_last_imagej_frame)
        event_frame = int(event.imagej_frame)
        ordered_frames = [event_frame]
        ordered_frames += list(range(event_frame - 1, first - 1, -1))
        ordered_frames += list(range(event_frame + 1, last + 1))
        references: dict[int, tuple[np.ndarray, np.ndarray]] = {
            event_frame: (donor_reference, host_reference)}
        partitioned = 0
        for imagej_frame in ordered_frames:
            if imagej_frame < event_frame:
                neighbour = imagej_frame + 1
            elif imagej_frame > event_frame:
                neighbour = imagej_frame - 1
            else:
                neighbour = imagej_frame
            if neighbour != imagej_frame and neighbour in references:
                donor_reference, host_reference = references[neighbour]

            obj = _object_for_frame(
                objects, host_identity, imagej_frame,
                (donor_reference + host_reference) / 2.0)
            if obj is None:
                continue
            original_host = _component_mask(
                source[imagej_frame - 1], host_identity, int(obj.component))
            if original_host is None:
                continue
            direct = transfer if imagej_frame == event_frame else None
            split = _split_two_core_host(
                original_host, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[original_host] = 0
            frame[donor_part] = donor
            frame[host_part] = host_identity
            if not np.array_equal(frame > 0, source[imagej_frame - 1] > 0):
                raise AssertionError(
                    "identity-ledger correction changed foreground support")
            inferred[imagej_frame - 1][original_host] = True
            references[imagej_frame] = (
                _centroid(donor_part), _centroid(host_part))
            partitioned += 1
            frame_rows.append({
                "two_core_run_id": str(run_id),
                "event_imagej_frame": event_frame,
                "imagej_frame": imagej_frame,
                "canonical_donor_identity": donor,
                "host_identity": host_identity,
                "partition_direction": (
                    "event" if imagej_frame == event_frame
                    else "backward" if imagej_frame < event_frame
                    else "forward"),
                **measures,
            })
        if partitioned:
            alias_events = alternatives[
                alternatives.donor_identity != donor]
            selected_rows.append({
                **event.to_dict(),
                "canonical_donor_identity": donor,
                "alternative_ledger_events": int(len(alternatives) - 1),
                "alternative_donor_identities": ";".join(map(
                    str, sorted(set(map(
                        int, alias_events.donor_identity.tolist()))))),
                "partitioned_frames": int(partitioned),
            })

    if not np.array_equal(candidate > 0, source > 0):
        raise AssertionError(
            "identity-ledger correction changed movie foreground support")
    original_ids = set(map(int, np.unique(source))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    if not candidate_ids.issubset(original_ids):
        raise AssertionError("identity-ledger correction created an identity")
    return (candidate, pd.DataFrame(selected_rows),
            pd.DataFrame(frame_rows), inferred)


def _nearest_identity(frame: np.ndarray, point: np.ndarray,
                      radius: float) -> tuple[int, float]:
    y, x = map(float, point)
    y0, y1 = max(0, int(np.floor(y - radius))), min(
        frame.shape[0], int(np.ceil(y + radius)) + 1)
    x0, x1 = max(0, int(np.floor(x - radius))), min(
        frame.shape[1], int(np.ceil(x + radius)) + 1)
    values = frame[y0:y1, x0:x1]
    pixels = np.column_stack(np.nonzero(values > 0))
    if not len(pixels):
        return 0, float("inf")
    absolute = pixels + np.array([y0, x0])
    distances = np.linalg.norm(absolute - np.asarray(point)[None, :], axis=1)
    index = int(np.argmin(distances))
    if float(distances[index]) > radius:
        return 0, float("inf")
    location = tuple(pixels[index])
    return int(values[location]), float(distances[index])


def _identity_has_body_core(frame: np.ndarray, identity: int,
                            minimum_core_px: int) -> bool:
    masks = identity_components(frame, identity)
    return any(
        bool(eroded_cores(mask, 1))
        and int(eroded_cores(mask, 1)[0][0]) >= minimum_core_px
        for _, mask in masks)


def _consecutive_future_presence(labels: np.ndarray, frame_index: int,
                                 identity: int) -> int:
    frames = 0
    for index in range(frame_index, len(labels)):
        if not np.any(labels[index] == identity):
            break
        frames += 1
    return frames


def _bookend_at_frame(labels: np.ndarray, frame_index: int,
                      host_identity: int, core_a: np.ndarray,
                      core_b: np.ndarray, params: DelayedBookendParams,
                      ) -> dict | None:
    first, distance_first = _nearest_identity(
        labels[frame_index], core_a, params.bookend_search_radius_px)
    second, distance_second = _nearest_identity(
        labels[frame_index], core_b, params.bookend_search_radius_px)
    if (first <= 0 or second <= 0 or first == second
            or host_identity not in (first, second)):
        return None
    other = second if first == host_identity else first
    if not _identity_has_body_core(
            labels[frame_index], other,
            params.minimum_bookend_core_px):
        return None
    history = _consecutive_prior_presence(
        labels, frame_index + 1, other)
    future = _consecutive_future_presence(labels, frame_index, other)
    return {
        "bookend_t": int(frame_index),
        "bookend_imagej_frame": int(frame_index + 1),
        "bookend_identity": int(other),
        "bookend_identity_history_frames": int(history),
        "bookend_identity_future_frames": int(future),
        "bookend_identity_support_frames": int(max(history, future)),
        "first_core_identity": int(first),
        "second_core_identity": int(second),
        "first_core_distance_px": float(distance_first),
        "second_core_distance_px": float(distance_second),
        "bookend_distance_sum_px": float(
            distance_first + distance_second),
    }


def _run_object(objects: pd.DataFrame, run: pd.Series,
                imagej_frame: int) -> pd.Series | None:
    rows = objects[
        (objects.identity == int(run.identity))
        & (objects.imagej_frame == imagej_frame)].copy()
    if rows.empty:
        return None
    rows["run_distance"] = np.hypot(
        rows.centre_y - float(run.centre_y),
        rows.centre_x - float(run.centre_x))
    return rows.sort_values(
        ["run_distance", "two_core_score"],
        ascending=[True, False]).iloc[0]


def detect_delayed_bookend_runs(
        labels: np.ndarray, raw: np.ndarray,
        params: DelayedBookendParams = DelayedBookendParams(),
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find two-core runs whose two slots resolve into identities at a bookend."""
    objects = audit_two_core_objects(labels, raw, params.two_core)
    runs = group_two_core_runs(
        objects, params.maximum_run_gap_frames,
        params.maximum_run_shift_px_per_frame)
    rows: list[dict] = []
    for run_index, run in runs.iterrows():
        first_obj = _run_object(
            objects, run, int(run.first_imagej_frame))
        last_obj = _run_object(
            objects, run, int(run.last_imagej_frame))
        if first_obj is None or last_obj is None:
            continue
        directions = (
            ("backward", first_obj,
             range(int(run.first_imagej_frame) - 2,
                   max(-1, int(run.first_imagej_frame) - 2
                       - params.maximum_bookend_frames), -1)),
            ("forward", last_obj,
             range(int(run.last_imagej_frame),
                   min(len(labels), int(run.last_imagej_frame)
                       + params.maximum_bookend_frames))),
        )
        found: list[dict] = []
        for direction, obj, frames in directions:
            core_a = np.array([obj.first_core_y, obj.first_core_x], float)
            core_b = np.array([obj.second_core_y, obj.second_core_x], float)
            for frame_index in frames:
                result = _bookend_at_frame(
                    labels, frame_index, int(run.identity),
                    core_a, core_b, params)
                if result is not None:
                    found.append({"bookend_direction": direction, **result})
                    break
        if not found:
            continue
        established = [row for row in found
                       if row["bookend_identity_support_frames"]
                       >= params.minimum_bookend_identity_history_frames]
        choices = established or found
        choices.sort(key=lambda row: (
            -row["bookend_identity_support_frames"],
            row["bookend_distance_sum_px"],
            0 if row["bookend_direction"] == "backward" else 1))
        selected = choices[0]
        morphology_gate = bool(
            float(run.maximum_two_core_score)
            >= params.strong_two_core_score
            or (float(run.maximum_two_core_score)
                >= params.moderate_two_core_score
                and float(run.maximum_core_separation_px)
                >= params.moderate_core_separation_px)
            or (float(run.maximum_two_core_score)
                >= params.moderate_two_core_score
                and float(run.minimum_core_intensity_ratio)
                >= params.distinct_core_intensity_ratio))
        historical_bookend = bool(
            selected["bookend_identity_history_frames"]
            >= params.minimum_bookend_identity_history_frames)
        persistent_new_bookend = bool(
            selected["bookend_identity_future_frames"]
            >= params.minimum_new_bookend_future_frames
            and float(run.maximum_two_core_score)
            >= params.minimum_new_bookend_two_core_score)
        rows.append({
            **run.to_dict(),
            "source_run_index": int(run_index),
            **selected,
            "opposite_bookend_identity": int(
                next((row["bookend_identity"] for row in found
                      if row is not selected), 0)),
            "bookends_found": int(len(found)),
            "bookend_morphology_gate": morphology_gate,
            "historical_bookend_gate": historical_bookend,
            "persistent_new_bookend_gate": persistent_new_bookend,
            "bookend_eligible": bool(
                (historical_bookend or persistent_new_bookend)
                and morphology_gate),
        })
    return pd.DataFrame(rows), objects


def correct_delayed_bookend_runs(
        labels: np.ndarray, raw: np.ndarray,
        params: DelayedBookendParams = DelayedBookendParams(),
        skip_run_ids: set[str] | None = None,
        detection_labels: np.ndarray | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Carry a proved second identity through its two-core merged run."""
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    detected, objects = detect_delayed_bookend_runs(source, raw, params)
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    skipped = skip_run_ids or set()
    for event in detected[detected.bookend_eligible].itertuples():
        if str(event.run_id) in skipped:
            continue
        donor = int(event.bookend_identity)
        host_identity = int(event.identity)
        bookend_t = int(event.bookend_t)
        donor_bookend = source[bookend_t] == donor
        host_bookend = source[bookend_t] == host_identity
        if not np.any(donor_bookend) or not np.any(host_bookend):
            continue
        donor_reference = _centroid(donor_bookend)
        host_reference = _centroid(host_bookend)
        partitioned = 0
        for imagej_frame in range(
                int(event.first_imagej_frame),
                int(event.last_imagej_frame) + 1):
            if np.any(source[imagej_frame - 1] == donor):
                # Never duplicate one identity into an existing separate body.
                continue
            obj = _run_object(objects, pd.Series(event._asdict()), imagej_frame)
            if obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], host_identity, int(obj.component))
            if host_mask is None:
                continue
            core_positions = (
                np.array([obj.first_core_y, obj.first_core_x], float),
                np.array([obj.second_core_y, obj.second_core_x], float))
            # The bookend's donor is assigned to the closer physical core.
            donor_core = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in core_positions]))
            direct = np.zeros(host_mask.shape, bool)
            direct[tuple(map(int, np.round(
                core_positions[donor_core])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host_identity
            if not np.array_equal(frame > 0, source[imagej_frame - 1] > 0):
                raise AssertionError(
                    "delayed-bookend correction changed foreground support")
            inferred[imagej_frame - 1][host_mask] = True
            donor_reference, host_reference = (
                _centroid(donor_part), _centroid(host_part))
            partitioned += 1
            frame_rows.append({
                "two_core_run_id": str(event.run_id),
                "imagej_frame": imagej_frame,
                "bookend_imagej_frame": int(event.bookend_imagej_frame),
                "bookend_direction": str(event.bookend_direction),
                "bookend_identity": donor,
                "host_identity": host_identity,
                **measures,
            })
        if partitioned:
            event_rows.append({
                **event._asdict(),
                "partitioned_frames": int(partitioned),
            })
    if not np.array_equal(candidate > 0, source > 0):
        raise AssertionError(
            "delayed-bookend correction changed movie foreground support")
    return candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows), inferred


def carry_persistent_two_core_slots(
        labels: np.ndarray, raw: np.ndarray, reservations: pd.DataFrame,
        params: PersistentSlotParams = PersistentSlotParams(),
        detection_labels: np.ndarray | None = None,
        skip_run_ids: set[str] | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Extend proved donor/host identity slots across later two-core runs.

    ``reservations`` contains actual partition frames from stronger direct-transfer or
    bookend corrections. Their donor seed positions are physical slot anchors. A new
    run is eligible only when it has the same host and a core close to one of those
    anchors within the long identity-memory window.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if reservations.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred
    required = {
        "imagej_frame", "host_identity", "donor_seed_y", "donor_seed_x",
        "donor_partition_px", "first_core_px", "second_core_px"}
    if not required.issubset(reservations.columns):
        missing = sorted(required - set(reservations.columns))
        raise ValueError(f"slot reservations missing columns: {missing}")
    objects = audit_two_core_objects(source, raw, params.two_core)
    runs = group_two_core_runs(
        objects, params.maximum_run_gap_frames,
        params.maximum_run_shift_px_per_frame)
    skipped = skip_run_ids or set()
    anchors: list[dict] = []
    for row in reservations.itertuples():
        donor_value = getattr(row, "canonical_donor_identity", np.nan)
        if pd.isna(donor_value):
            donor_value = getattr(row, "bookend_identity", np.nan)
        if pd.isna(donor_value):
            continue
        donor_core_px = getattr(row, "donor_core_px", np.nan)
        if pd.isna(donor_core_px):
            donor_core_px = min(
                int(row.first_core_px), int(row.second_core_px))
        anchors.append({
            "imagej_frame": int(row.imagej_frame),
            "host_identity": int(row.host_identity),
            "donor_identity": int(donor_value),
            "donor_y": float(row.donor_seed_y),
            "donor_x": float(row.donor_seed_x),
            "donor_area_px": int(row.donor_partition_px),
            "donor_core_px": int(donor_core_px),
            "donor_core_fraction": float(
                int(donor_core_px)
                / max(int(row.donor_partition_px), 1)),
            "source": "proved_partition",
        })
    event_rows: list[dict] = []
    partition_rows: list[dict] = []

    ordered_runs = runs.sort_values(
        ["first_imagej_frame", "maximum_two_core_score"],
        ascending=[True, False])
    for run in ordered_runs.itertuples():
        if str(run.run_id) in skipped:
            continue
        representative = int(run.representative_imagej_frame)
        obj = _run_object(objects, pd.Series(run._asdict()), representative)
        if obj is None:
            continue
        core_positions = (
            np.array([obj.first_core_y, obj.first_core_x], float),
            np.array([obj.second_core_y, obj.second_core_x], float))
        choices: list[dict] = []
        for anchor in anchors:
            if int(anchor["host_identity"]) != int(run.identity):
                continue
            if representative < int(anchor["imagej_frame"]):
                gap = int(anchor["imagej_frame"]) - int(run.last_imagej_frame)
            else:
                gap = int(run.first_imagej_frame) - int(anchor["imagej_frame"])
            gap = max(gap, 0)
            if gap > params.maximum_reservation_gap_frames:
                continue
            point = np.array([anchor["donor_y"], anchor["donor_x"]], float)
            distances = [float(np.linalg.norm(core - point))
                         for core in core_positions]
            core_index = int(np.argmin(distances))
            maximum_shift = (
                params.maximum_core_shift_px
                + params.core_shift_per_frame_px * gap)
            if distances[core_index] > maximum_shift:
                continue
            candidate_core_px = int(
                obj.first_core_px if core_index == 0
                else obj.second_core_px)
            # Compare soma core with soma core. The previous implementation
            # compared this core with the entire donor outline, which made a
            # persistent cell appear to shrink whenever its processes were large.
            expected_area = max(float(anchor["donor_core_px"]), 1.0)
            area_change = abs(float(np.log2(
                max(candidate_core_px, 1) / expected_area)))
            expected_fraction = max(
                float(anchor["donor_core_fraction"]), 1e-6)
            measured_fraction = candidate_core_px / max(float(obj.area_px), 1.0)
            fraction_ratio = measured_fraction / expected_fraction
            if (area_change > params.maximum_area_log2_change
                    or fraction_ratio < params.minimum_core_fraction_ratio
                    or fraction_ratio > params.maximum_core_fraction_ratio):
                continue
            choices.append({
                **anchor, "gap_frames": gap,
                "core_distance_px": distances[core_index],
                "core_index": core_index,
                "area_log2_change": area_change,
                "core_fraction_ratio": fraction_ratio,
                "cost": (gap / max(params.maximum_reservation_gap_frames, 1)
                         + distances[core_index] / max(maximum_shift, 1.0)
                         + area_change),
            })
        if not choices:
            continue
        choices.sort(key=lambda row: row["cost"])
        selected = choices[0]
        donor = int(selected["donor_identity"])
        host_identity = int(run.identity)
        donor_reference = core_positions[int(selected["core_index"])]
        host_reference = core_positions[1 - int(selected["core_index"])]
        partitioned = 0
        local_rows: list[dict] = []
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            if np.any(source[imagej_frame - 1] == donor):
                continue
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], host_identity,
                int(frame_obj.component))
            if host_mask is None:
                continue
            positions = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            donor_index = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in positions]))
            direct = np.zeros(host_mask.shape, bool)
            direct[tuple(map(int, np.round(positions[donor_index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host_identity
            if not np.array_equal(frame > 0, source[imagej_frame - 1] > 0):
                raise AssertionError(
                    "persistent-slot correction changed foreground support")
            inferred[imagej_frame - 1][host_mask] = True
            donor_reference, host_reference = (
                _centroid(donor_part), _centroid(host_part))
            partitioned += 1
            local_rows.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "host_identity": host_identity,
                "canonical_donor_identity": donor,
                "reservation_gap_frames": int(selected["gap_frames"]),
                "reservation_core_distance_px": float(
                    selected["core_distance_px"]),
                "reservation_area_log2_change": float(
                    selected["area_log2_change"]),
                "reservation_core_fraction_ratio": float(
                    selected["core_fraction_ratio"]),
                **measures,
            })
        if not partitioned:
            continue
        partition_rows.extend(local_rows)
        event_rows.append({
            "two_core_run_id": str(run.run_id),
            "first_imagej_frame": int(run.first_imagej_frame),
            "last_imagej_frame": int(run.last_imagej_frame),
            "host_identity": host_identity,
            "canonical_donor_identity": donor,
            "reservation_source_imagej_frame": int(
                selected["imagej_frame"]),
            "reservation_gap_frames": int(selected["gap_frames"]),
            "reservation_core_distance_px": float(
                selected["core_distance_px"]),
            "reservation_area_log2_change": float(
                selected["area_log2_change"]),
            "reservation_core_fraction_ratio": float(
                selected["core_fraction_ratio"]),
            "partitioned_frames": int(partitioned),
        })
        for local in local_rows:
            anchors.append({
                "imagej_frame": int(local["imagej_frame"]),
                "host_identity": host_identity,
                "donor_identity": donor,
                "donor_y": float(local["donor_seed_y"]),
                "donor_x": float(local["donor_seed_x"]),
                "donor_area_px": int(local["donor_partition_px"]),
                "donor_core_px": int(local["donor_core_px"]),
                "donor_core_fraction": float(
                    int(local["donor_core_px"])
                    / max(int(local["donor_partition_px"]), 1)),
                "source": "propagated_partition",
            })
    if not np.array_equal(candidate > 0, source > 0):
        raise AssertionError(
            "persistent-slot correction changed movie foreground support")
    return (candidate, pd.DataFrame(event_rows),
            pd.DataFrame(partition_rows), inferred)


def carry_bidirectional_identity_ledger(
        labels: np.ndarray, raw: np.ndarray, reservations: pd.DataFrame,
        params: BidirectionalLedgerParams = BidirectionalLedgerParams(),
        detection_labels: np.ndarray | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Propagate proved cell pairs backward and forward through two-core hosts.

    A reservation is proof that two identity names belong to two physical soma
    slots. All reservations for the same host/donor pair are treated as a single
    time-spanning ledger entry. This is deliberately bidirectional: a later clean
    separation can repair an earlier missing frame, and a cell name is never copied
    into a host when that name already occupies another body in the same frame.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if reservations.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred

    required = {
        "imagej_frame", "host_identity", "donor_seed_y", "donor_seed_x"}
    if not required.issubset(reservations.columns):
        missing = sorted(required - set(reservations.columns))
        raise ValueError(f"identity ledger missing columns: {missing}")

    anchor_rows: list[dict] = []
    for row in reservations.itertuples():
        donor = getattr(row, "canonical_donor_identity", np.nan)
        if pd.isna(donor):
            donor = getattr(row, "bookend_identity", np.nan)
        if pd.isna(donor):
            continue
        anchor_rows.append({
            "imagej_frame": int(row.imagej_frame),
            "host_identity": int(row.host_identity),
            "donor_identity": int(donor),
            "donor_y": float(row.donor_seed_y),
            "donor_x": float(row.donor_seed_x),
        })
    anchors = pd.DataFrame(anchor_rows)
    if anchors.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred

    pair_support = anchors.groupby(
        ["host_identity", "donor_identity"], as_index=False).agg(
            confirmed_pair_frames=("imagej_frame", "nunique"),
            first_proof_imagej_frame=("imagej_frame", "min"),
            last_proof_imagej_frame=("imagej_frame", "max"))

    objects = audit_two_core_objects(source, raw, params.two_core)
    runs = group_two_core_runs(
        objects, params.maximum_run_gap_frames,
        params.maximum_run_shift_px_per_frame)
    event_rows: list[dict] = []
    frame_rows: list[dict] = []

    for run in runs.sort_values(
            ["first_imagej_frame", "maximum_two_core_score"],
            ascending=[True, False]).itertuples():
        host = int(run.identity)
        host_pairs = pair_support[pair_support.host_identity == host]
        if host_pairs.empty:
            continue
        representative = int(run.representative_imagej_frame)
        obj = _run_object(objects, pd.Series(run._asdict()), representative)
        if obj is None:
            continue
        cores = (
            np.array([obj.first_core_y, obj.first_core_x], float),
            np.array([obj.second_core_y, obj.second_core_x], float))
        choices: list[dict] = []
        for pair in host_pairs.itertuples():
            donor = int(pair.donor_identity)
            pair_anchors = anchors[
                (anchors.host_identity == host)
                & (anchors.donor_identity == donor)]
            support_events = int(len(pair_anchors))
            support_frames = int(pair.confirmed_pair_frames)
            # Repeated corrected frames in one long merge are not independent
            # proof. Require either more than one separated/transfer event, or
            # a long corrected interval plus unusually clear two-soma structure.
            confirmed = bool(
                support_events >= params.minimum_confirmed_pair_events
                and support_frames >= params.minimum_confirmed_pair_frames)
            strong_object = bool(
                float(run.maximum_two_core_score)
                >= params.minimum_strong_core_score
                or float(run.maximum_core_separation_px)
                >= params.minimum_strong_core_separation_px)
            if not (confirmed and strong_object):
                continue
            for anchor in pair_anchors.itertuples():
                gap = min(
                    abs(representative - int(anchor.imagej_frame)),
                    abs(int(run.first_imagej_frame) - int(anchor.imagej_frame)),
                    abs(int(run.last_imagej_frame) - int(anchor.imagej_frame)))
                if gap > params.maximum_identity_gap_frames:
                    continue
                point = np.array([anchor.donor_y, anchor.donor_x], float)
                distances = [float(np.linalg.norm(core - point))
                             for core in cores]
                donor_index = int(np.argmin(distances))
                maximum_shift = (
                    params.maximum_core_shift_px
                    + params.core_shift_per_frame_px * gap)
                if distances[donor_index] > maximum_shift:
                    continue
                choices.append({
                    "donor_identity": donor,
                    "donor_index": donor_index,
                    "proof_imagej_frame": int(anchor.imagej_frame),
                    "gap_frames": int(gap),
                    "distance_px": float(distances[donor_index]),
                    "support_events": support_events,
                    "support_frames": support_frames,
                    "cost": (distances[donor_index]
                             / max(maximum_shift, 1.0)
                             + gap / max(params.maximum_identity_gap_frames, 1)
                             - 0.05 * min(support_frames, 10)),
                })
        if not choices:
            continue
        choices.sort(key=lambda choice: (
            choice["cost"], -choice["support_frames"],
            choice["donor_identity"]))
        selected = choices[0]
        donor = int(selected["donor_identity"])
        donor_reference = cores[int(selected["donor_index"])]
        host_reference = cores[1 - int(selected["donor_index"])]
        local: list[dict] = []
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            if np.any(source[imagej_frame - 1] == donor):
                continue
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], host, int(frame_obj.component))
            if host_mask is None:
                continue
            positions = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            donor_index = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in positions]))
            direct = np.zeros(host_mask.shape, bool)
            direct[tuple(map(int, np.round(positions[donor_index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host
            inferred[imagej_frame - 1][host_mask] = True
            donor_reference, host_reference = (
                _centroid(donor_part), _centroid(host_part))
            local.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "host_identity": host,
                "canonical_donor_identity": donor,
                "proof_imagej_frame": int(selected["proof_imagej_frame"]),
                "proof_gap_frames": int(selected["gap_frames"]),
                "proof_distance_px": float(selected["distance_px"]),
                "confirmed_pair_frames": int(selected["support_frames"]),
                **measures,
            })
        if not local:
            continue
        frame_rows.extend(local)
        event_rows.append({
            "two_core_run_id": str(run.run_id),
            "first_imagej_frame": int(run.first_imagej_frame),
            "last_imagej_frame": int(run.last_imagej_frame),
            "host_identity": host,
            "canonical_donor_identity": donor,
            "proof_imagej_frame": int(selected["proof_imagej_frame"]),
            "proof_gap_frames": int(selected["gap_frames"]),
            "proof_distance_px": float(selected["distance_px"]),
            "confirmed_pair_frames": int(selected["support_frames"]),
            "partitioned_frames": int(len(local)),
        })

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "bidirectional identity ledger changed foreground support")
    return (candidate, pd.DataFrame(event_rows),
            pd.DataFrame(frame_rows), inferred)


def _core_near_identity(frame: np.ndarray, identity: int, point: np.ndarray,
                        radius: float, minimum_core_px: int,
                        cache: dict | None = None,
                        frame_index: int = -1) -> dict | None:
    """Describe the nearest real body core for one identity."""
    choices: list[dict] = []
    key = (int(frame_index), int(identity), int(minimum_core_px))
    cached = cache.get(key) if cache is not None else None
    if cached is None:
        cached = []
        for _, mask in identity_components(frame, identity):
            for area, core in eroded_cores(mask, 1):
                if int(area) < minimum_core_px:
                    continue
                centre = _centroid(core)
                cached.append((int(area), centre))
        if cache is not None:
            cache[key] = cached
    for area, centre in cached:
            distance = float(np.linalg.norm(centre - point))
            if distance <= radius:
                choices.append({
                    "identity": int(identity), "area_px": int(area),
                    "y": float(centre[0]), "x": float(centre[1]),
                    "distance_px": distance,
                })
    return min(choices, key=lambda row: row["distance_px"]) if choices else None


def _all_core_choices(frame: np.ndarray, point: np.ndarray, radius: float,
                      minimum_core_px: int, cache: dict | None = None,
                      frame_index: int = -1) -> list[dict]:
    choices: list[dict] = []
    y, x = map(float, point)
    rows = slice(max(0, int(np.floor(y - radius))),
                 min(frame.shape[0], int(np.ceil(y + radius)) + 1))
    cols = slice(max(0, int(np.floor(x - radius))),
                 min(frame.shape[1], int(np.ceil(x + radius)) + 1))
    for identity_value in np.unique(frame[rows, cols]):
        identity = int(identity_value)
        if identity <= 0:
            continue
        choice = _core_near_identity(
            frame, identity, point, radius, minimum_core_px,
            cache, frame_index)
        if choice is not None:
            choices.append(choice)
    return choices


def _two_identity_bookend(
        labels: np.ndarray, host_identity: int, frame_index: int,
        core_a: np.ndarray, core_b: np.ndarray, params: DualBookendParams,
        cache: dict | None = None,
        ) -> dict | None:
    first = _all_core_choices(
        labels[frame_index], core_a, params.bookend_search_radius_px,
        params.minimum_bookend_core_px, cache, frame_index)
    second = _all_core_choices(
        labels[frame_index], core_b, params.bookend_search_radius_px,
        params.minimum_bookend_core_px, cache, frame_index)
    pairs: list[dict] = []
    for a in first:
        for b in second:
            if int(a["identity"]) == int(b["identity"]):
                continue
            identities = (int(a["identity"]), int(b["identity"]))
            support_a = max(
                _consecutive_prior_presence(
                    labels, frame_index + 1, identities[0]),
                _consecutive_future_presence(
                    labels, frame_index, identities[0]))
            support_b = max(
                _consecutive_prior_presence(
                    labels, frame_index + 1, identities[1]),
                _consecutive_future_presence(
                    labels, frame_index, identities[1]))
            if min(support_a, support_b) < params.minimum_identity_support_frames:
                continue
            # Prefer the existing host name if it belongs to a real core at the
            # bookend, otherwise the longer-lived of the two names becomes host.
            if host_identity in identities:
                host_bookend = host_identity
                donor = identities[1] if identities[0] == host_identity else identities[0]
            elif support_a > support_b or (
                    support_a == support_b and identities[0] < identities[1]):
                host_bookend, donor = identities[0], identities[1]
            else:
                host_bookend, donor = identities[1], identities[0]
            pairs.append({
                "bookend_t": int(frame_index),
                "bookend_imagej_frame": int(frame_index + 1),
                "first_bookend_identity": identities[0],
                "second_bookend_identity": identities[1],
                "host_bookend_identity": int(host_bookend),
                "donor_identity": int(donor),
                "first_support_frames": int(support_a),
                "second_support_frames": int(support_b),
                "distance_sum_px": float(
                    a["distance_px"] + b["distance_px"]),
                "first_core_y": float(a["y"]),
                "first_core_x": float(a["x"]),
                "second_core_y": float(b["y"]),
                "second_core_x": float(b["x"]),
            })
    if not pairs:
        return None
    pairs.sort(key=lambda row: (
        row["distance_sum_px"],
        -min(row["first_support_frames"], row["second_support_frames"]),
        row["donor_identity"]))
    return pairs[0]


def correct_dual_bookend_runs(
        labels: np.ndarray, raw: np.ndarray,
        params: DualBookendParams = DualBookendParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Resolve a two-core host even when neither later identity keeps its name.

    A merged label may replace both physical identities. Searching for "host plus one
    neighbour" then fails. This pass instead asks which two established, real soma
    cores occupy the two physical slots at either time bookend and reuses those names.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)
    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    core_cache: dict = {}
    for run in runs.itertuples():
        morphology = bool(
            float(run.maximum_two_core_score) >= params.minimum_two_core_score
            or float(run.maximum_core_separation_px)
            >= params.minimum_core_separation_px)
        if not morphology:
            continue
        first_obj = _run_object(
            objects, pd.Series(run._asdict()), int(run.first_imagej_frame))
        last_obj = _run_object(
            objects, pd.Series(run._asdict()), int(run.last_imagej_frame))
        if first_obj is None or last_obj is None:
            continue
        found: list[dict] = []
        directions = (
            ("backward", first_obj,
             range(int(run.first_imagej_frame) - 2,
                   max(-1, int(run.first_imagej_frame) - 2
                       - params.maximum_bookend_frames), -1)),
            ("forward", last_obj,
             range(int(run.last_imagej_frame),
                   min(len(source), int(run.last_imagej_frame)
                       + params.maximum_bookend_frames))),
        )
        for direction, obj, frames in directions:
            core_a = np.array([obj.first_core_y, obj.first_core_x], float)
            core_b = np.array([obj.second_core_y, obj.second_core_x], float)
            for frame_index in frames:
                row = _two_identity_bookend(
                    source, int(run.identity), frame_index,
                    core_a, core_b, params, core_cache)
                if row is not None:
                    found.append({"bookend_direction": direction, **row})
                    break
        if not found:
            continue
        found.sort(key=lambda row: (
            row["distance_sum_px"],
            -min(row["first_support_frames"], row["second_support_frames"]),
            0 if row["bookend_direction"] == "backward" else 1))
        selected = found[0]
        identity_a = int(selected["first_bookend_identity"])
        identity_b = int(selected["second_bookend_identity"])
        ref_a = np.array([
            selected["first_core_y"], selected["first_core_x"]], float)
        ref_b = np.array([
            selected["second_core_y"], selected["second_core_x"]], float)
        local: list[dict] = []
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], int(run.identity),
                int(frame_obj.component))
            if host_mask is None:
                continue
            existing: list[tuple[int, int]] = []
            for identity in (identity_a, identity_b):
                cores = eroded_cores(source[imagej_frame - 1] == identity, 1)
                existing.append((identity, int(cores[0][0]) if cores else 0))
            # A name attached only to a thin process does not account for a soma.
            if any(area > params.maximum_existing_core_px
                   for identity, area in existing
                   if identity != int(run.identity)):
                continue
            positions = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            direct = np.zeros(host_mask.shape, bool)
            first_index = int(np.argmin([
                np.linalg.norm(position - ref_a) for position in positions]))
            direct[tuple(map(int, np.round(positions[first_index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], ref_a, ref_b,
                params.minimum_partition_px, direct)
            if split is None:
                continue
            part_a, part_b, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[part_a] = identity_a
            frame[part_b] = identity_b
            inferred[imagej_frame - 1][host_mask] = True
            ref_a, ref_b = _centroid(part_a), _centroid(part_b)
            local.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "original_host_identity": int(run.identity),
                "first_identity": identity_a,
                "second_identity": identity_b,
                "bookend_imagej_frame": int(selected["bookend_imagej_frame"]),
                "bookend_direction": str(selected["bookend_direction"]),
                **measures,
            })
        if local:
            frame_rows.extend(local)
            event_rows.append({
                "two_core_run_id": str(run.run_id),
                "first_imagej_frame": int(run.first_imagej_frame),
                "last_imagej_frame": int(run.last_imagej_frame),
                "original_host_identity": int(run.identity),
                "first_identity": identity_a,
                "second_identity": identity_b,
                "bookend_imagej_frame": int(selected["bookend_imagej_frame"]),
                "bookend_direction": str(selected["bookend_direction"]),
                "partitioned_frames": int(len(local)),
            })
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("dual-bookend correction changed foreground support")
    return candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows), inferred


def carry_global_pair_ledger(
        labels: np.ndarray, raw: np.ndarray, pair_frames: pd.DataFrame,
        params: GlobalPairLedgerParams = GlobalPairLedgerParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Carry every proved unordered identity pair through distant two-core runs."""
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if pair_frames.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred

    anchors: list[dict] = []
    for row in pair_frames.itertuples():
        first = getattr(row, "first_identity", np.nan)
        second = getattr(row, "second_identity", np.nan)
        if pd.isna(first) or pd.isna(second):
            donor = getattr(row, "canonical_donor_identity", np.nan)
            if pd.isna(donor):
                donor = getattr(row, "bookend_identity", np.nan)
            host = getattr(row, "host_identity", np.nan)
            if pd.isna(donor) or pd.isna(host):
                continue
            first, second = int(donor), int(host)
            first_y = float(row.donor_seed_y)
            first_x = float(row.donor_seed_x)
            second_y = float(row.host_seed_y)
            second_x = float(row.host_seed_x)
        else:
            first, second = int(first), int(second)
            first_y = float(row.donor_seed_y)
            first_x = float(row.donor_seed_x)
            second_y = float(row.host_seed_y)
            second_x = float(row.host_seed_x)
        anchors.append({
            "imagej_frame": int(row.imagej_frame),
            "first_identity": int(first), "second_identity": int(second),
            "first_y": first_y, "first_x": first_x,
            "second_y": second_y, "second_x": second_x,
        })
    if not anchors:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred

    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)
    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    for run in runs.itertuples():
        morphology = bool(
            float(run.maximum_two_core_score) >= params.minimum_two_core_score
            or (float(run.maximum_two_core_score)
                >= params.moderate_two_core_score
                and float(run.maximum_core_separation_px)
                >= params.moderate_core_separation_px))
        if not morphology:
            continue
        representative = int(run.representative_imagej_frame)
        obj = _run_object(objects, pd.Series(run._asdict()), representative)
        if obj is None:
            continue
        cores = (
            np.array([obj.first_core_y, obj.first_core_x], float),
            np.array([obj.second_core_y, obj.second_core_x], float))
        choices: list[dict] = []
        for anchor in anchors:
            gap = abs(representative - int(anchor["imagej_frame"]))
            if gap > params.maximum_identity_gap_frames:
                continue
            refs = (
                np.array([anchor["first_y"], anchor["first_x"]], float),
                np.array([anchor["second_y"], anchor["second_x"]], float))
            costs = (
                float(np.linalg.norm(cores[0] - refs[0])
                      + np.linalg.norm(cores[1] - refs[1])),
                float(np.linalg.norm(cores[0] - refs[1])
                      + np.linalg.norm(cores[1] - refs[0])),
            )
            swap = int(np.argmin(costs))
            maximum = 2.0 * (
                params.maximum_core_shift_px
                + params.core_shift_per_frame_px * gap)
            if costs[swap] > maximum:
                continue
            identities = (
                (anchor["first_identity"], anchor["second_identity"])
                if swap == 0 else
                (anchor["second_identity"], anchor["first_identity"]))
            choices.append({
                **anchor, "gap_frames": int(gap), "cost": costs[swap],
                "core_identities": identities,
            })
        if not choices:
            continue
        choices.sort(key=lambda row: (
            row["cost"], row["gap_frames"], row["first_identity"],
            row["second_identity"]))
        selected = choices[0]
        identity_a, identity_b = map(int, selected["core_identities"])
        local: list[dict] = []
        ref_a, ref_b = cores
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], int(run.identity),
                int(frame_obj.component))
            if host_mask is None:
                continue
            blocked = False
            for identity in (identity_a, identity_b):
                if identity == int(run.identity):
                    continue
                existing = eroded_cores(
                    source[imagej_frame - 1] == identity, 1)
                if existing and int(existing[0][0]) > params.maximum_existing_core_px:
                    blocked = True
                    break
            if blocked:
                continue
            positions = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            first_index = int(np.argmin([
                np.linalg.norm(position - ref_a) for position in positions]))
            direct = np.zeros(host_mask.shape, bool)
            direct[tuple(map(int, np.round(positions[first_index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], ref_a, ref_b,
                params.minimum_partition_px, direct)
            if split is None:
                continue
            part_a, part_b, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[part_a] = identity_a
            frame[part_b] = identity_b
            inferred[imagej_frame - 1][host_mask] = True
            ref_a, ref_b = _centroid(part_a), _centroid(part_b)
            local.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "original_host_identity": int(run.identity),
                "first_identity": identity_a,
                "second_identity": identity_b,
                "proof_imagej_frame": int(selected["imagej_frame"]),
                "proof_gap_frames": int(selected["gap_frames"]),
                **measures,
            })
        if local:
            frame_rows.extend(local)
            event_rows.append({
                "two_core_run_id": str(run.run_id),
                "first_imagej_frame": int(run.first_imagej_frame),
                "last_imagej_frame": int(run.last_imagej_frame),
                "original_host_identity": int(run.identity),
                "first_identity": identity_a,
                "second_identity": identity_b,
                "proof_imagej_frame": int(selected["imagej_frame"]),
                "proof_gap_frames": int(selected["gap_frames"]),
                "partitioned_frames": int(len(local)),
            })
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("global pair ledger changed foreground support")
    return candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows), inferred


def carry_recurrent_host_pair_ledger(
        labels: np.ndarray, raw: np.ndarray, pair_frames: pd.DataFrame,
        params: GlobalPairLedgerParams = GlobalPairLedgerParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Carry only a repeatedly proved host/partner pair into unresolved re-merges.

    Unlike :func:`carry_global_pair_ledger`, this pass does not treat an unordered
    pair as permission to split any nearby object.  The same host identity and the
    same partner must have been resolved in multiple earlier runs, and only an
    unresolved two-core object carrying that host name can inherit the proof.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    required = {
        "canonical_donor_identity", "host_identity", "two_core_run_id",
        "imagej_frame", "donor_seed_y", "donor_seed_x",
        "host_seed_y", "host_seed_x",
    }
    if pair_frames.empty or not required.issubset(pair_frames.columns):
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred
    proofs = pair_frames.dropna(subset=[
        "canonical_donor_identity", "host_identity", "two_core_run_id",
        "imagej_frame", "donor_seed_y", "donor_seed_x",
        "host_seed_y", "host_seed_x"]).copy()
    if proofs.empty:
        return candidate, pd.DataFrame(), pd.DataFrame(), inferred
    proofs["canonical_donor_identity"] = \
        proofs.canonical_donor_identity.astype(int)
    proofs["host_identity"] = proofs.host_identity.astype(int)

    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)
    event_tables: list[pd.DataFrame] = []
    frame_tables: list[pd.DataFrame] = []
    for (host, donor), group in proofs.groupby(
            ["host_identity", "canonical_donor_identity"]):
        if (group.two_core_run_id.nunique() < params.minimum_proof_runs
                or group.imagej_frame.nunique() < params.minimum_proof_frames):
            continue
        target_rows: list[pd.Series] = []
        for _, run in runs[runs.identity == int(host)].iterrows():
            obj = _run_object(
                objects, run, int(run.representative_imagej_frame))
            if obj is None:
                continue
            t = int(run.representative_imagej_frame) - 1
            first = int(candidate[
                t, int(round(obj.first_core_y)), int(round(obj.first_core_x))])
            second = int(candidate[
                t, int(round(obj.second_core_y)), int(round(obj.second_core_x))])
            if first == int(host) and second == int(host):
                target_rows.append(run)
        if not target_rows:
            continue
        target_runs = pd.DataFrame(target_rows)
        fixed, events, frames, changed = carry_global_pair_ledger(
            candidate, raw, group, params, detection_labels=source,
            objects=objects, runs=target_runs)
        candidate = fixed
        inferred |= changed
        if not events.empty:
            events = events.copy()
            events["recurrent_host_identity"] = int(host)
            events["recurrent_partner_identity"] = int(donor)
            events["proof_run_count"] = int(group.two_core_run_id.nunique())
            event_tables.append(events)
        if not frames.empty:
            frame_tables.append(frames)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("recurrent host-pair ledger changed foreground support")
    return (
        candidate,
        pd.concat(event_tables, ignore_index=True) if event_tables
        else pd.DataFrame(),
        pd.concat(frame_tables, ignore_index=True) if frame_tables
        else pd.DataFrame(),
        inferred,
    )


def correct_predecessor_core_merges(
        labels: np.ndarray, raw: np.ndarray,
        params: PredecessorCoreParams = PredecessorCoreParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Restore two established names from their exact pre-merge core pixels."""
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)
    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    for _, run in runs.iterrows():
        imagej_frame = int(run.representative_imagej_frame)
        if imagej_frame <= 1:
            continue
        obj = _run_object(objects, run, imagej_frame)
        if obj is None:
            continue
        t = imagej_frame - 1
        core_positions = (
            (int(round(obj.first_core_y)), int(round(obj.first_core_x))),
            (int(round(obj.second_core_y)), int(round(obj.second_core_x))),
        )
        if any(candidate[t, y, x] != int(run.identity)
               for y, x in core_positions):
            continue
        host_mask = _component_mask(
            source[t], int(run.identity), int(obj.component))
        if host_mask is None:
            continue
        cores = eroded_cores(host_mask, params.two_core.erosion_iterations)
        if len(cores) < 2:
            continue
        assignments: list[tuple[int, float, np.ndarray]] = []
        for _, core in cores[:2]:
            values, counts = np.unique(source[t - 1][core], return_counts=True)
            choices = [(int(count), int(value))
                       for value, count in zip(values, counts) if value > 0]
            if not choices:
                assignments = []
                break
            count, identity = max(choices)
            fraction = float(count / max(int(core.sum()), 1))
            assignments.append((identity, fraction, _centroid(core)))
        if len(assignments) != 2:
            continue
        first_identity, first_fraction, first_reference = assignments[0]
        second_identity, second_fraction, second_reference = assignments[1]
        if (first_identity == second_identity
                or int(run.identity) in (first_identity, second_identity)
                or min(first_fraction, second_fraction)
                < params.minimum_core_ancestry_fraction):
            continue
        histories = (
            _consecutive_prior_presence(source, t, first_identity),
            _consecutive_prior_presence(source, t, second_identity),
        )
        if min(histories) < params.minimum_identity_history_frames:
            continue
        blocked = False
        for identity in (first_identity, second_identity):
            existing = eroded_cores(candidate[t] == identity, 1)
            if existing and int(existing[0][0]) > params.maximum_existing_core_px:
                blocked = True
                break
        if blocked:
            continue
        split = _split_two_core_host(
            host_mask, raw[t], first_reference, second_reference,
            params.minimum_partition_px, cores[0][1])
        if split is None:
            continue
        first_part, second_part, measures = split
        frame = candidate[t]
        frame[host_mask] = 0
        frame[first_part] = first_identity
        frame[second_part] = second_identity
        inferred[t][host_mask] = True
        event_rows.append({
            "two_core_run_id": str(run.run_id),
            "imagej_frame": imagej_frame,
            "original_host_identity": int(run.identity),
            "first_identity": first_identity,
            "second_identity": second_identity,
            "first_core_ancestry_fraction": first_fraction,
            "second_core_ancestry_fraction": second_fraction,
            "first_identity_history_frames": histories[0],
            "second_identity_history_frames": histories[1],
        })
        frame_rows.append({
            "two_core_run_id": str(run.run_id),
            "imagej_frame": imagej_frame,
            "original_host_identity": int(run.identity),
            "first_identity": first_identity,
            "second_identity": second_identity,
            **measures,
        })
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("predecessor-core correction changed foreground support")
    return (candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows),
            inferred)


def correct_shape_proven_two_core_runs(
        labels: np.ndarray, raw: np.ndarray,
        params: ShapeProvenTwoCoreParams = ShapeProvenTwoCoreParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Split only the clearest two-soma objects using an existing nearby name.

    This is the conservative fallback for a physical pair that has remained merged for
    the whole available bookend window. The shape must contain two similarly bright,
    well-separated cores. The second name must already be established elsewhere in the
    movie near the second core; this pass cannot allocate a fresh identity number.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)

    identity_frames = _identity_frame_indices(source)
    identity_history = {
        identity: len(frames) for identity, frames in identity_frames.items()}
    core_observations: list[dict] = []
    for frame_index, frame in enumerate(source):
        for identity in identity_frames:
            if frame_index not in identity_frames[identity]:
                continue
            for _, mask in identity_components(frame, identity):
                for area, core in eroded_cores(mask, 1):
                    if int(area) <= params.maximum_reference_core_px:
                        continue
                    centre = _centroid(core)
                    core_observations.append({
                        "identity": int(identity),
                        "imagej_frame": int(frame_index + 1),
                        "y": float(centre[0]), "x": float(centre[1]),
                        "history_frames": int(identity_history[identity]),
                    })
    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    for run in runs.itertuples():
        if float(run.maximum_two_core_score) < params.minimum_two_core_score:
            continue
        representative = int(run.representative_imagej_frame)
        obj = _run_object(objects, pd.Series(run._asdict()), representative)
        if obj is None:
            continue
        if (float(obj.core_intensity_ratio)
                < params.minimum_core_intensity_ratio
                or float(obj.core_separation_px)
                < params.minimum_core_separation_px):
            continue
        host = int(run.identity)
        positions = (
            np.array([obj.first_core_y, obj.first_core_x], float),
            np.array([obj.second_core_y, obj.second_core_x], float))
        references: list[dict] = []
        for identity, history in identity_history.items():
            if identity == host or history < params.minimum_reference_history_frames:
                continue
            best: dict | None = None
            for core in core_observations:
                if int(core["identity"]) != identity:
                    continue
                centre = np.array([core["y"], core["x"]], float)
                distances = [float(np.linalg.norm(centre - p))
                             for p in positions]
                index = int(np.argmin(distances))
                row = {
                    **core, "core_index": index,
                    "distance_px": distances[index],
                }
                if best is None or row["distance_px"] < best["distance_px"]:
                    best = row
            if (best is not None and best["distance_px"]
                    <= params.maximum_reference_distance_px):
                references.append(best)
        if not references:
            continue
        references.sort(key=lambda row: (
            row["distance_px"], -row["history_frames"], row["identity"]))
        reference = references[0]
        donor = int(reference["identity"])
        donor_index = int(reference["core_index"])
        donor_reference = positions[donor_index]
        host_reference = positions[1 - donor_index]
        local: list[dict] = []
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            existing = eroded_cores(source[imagej_frame - 1] == donor, 1)
            if existing and int(existing[0][0]) > params.maximum_reference_core_px:
                continue
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], host, int(frame_obj.component))
            if host_mask is None:
                continue
            frame_positions = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            direct = np.zeros(host_mask.shape, bool)
            index = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in frame_positions]))
            direct[tuple(map(int, np.round(frame_positions[index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host
            inferred[imagej_frame - 1][host_mask] = True
            donor_reference, host_reference = (
                _centroid(donor_part), _centroid(host_part))
            local.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "host_identity": host,
                "canonical_donor_identity": donor,
                "reference_imagej_frame": int(reference["imagej_frame"]),
                "reference_distance_px": float(reference["distance_px"]),
                **measures,
            })
        if local:
            frame_rows.extend(local)
            event_rows.append({
                "two_core_run_id": str(run.run_id),
                "first_imagej_frame": int(run.first_imagej_frame),
                "last_imagej_frame": int(run.last_imagej_frame),
                "host_identity": host,
                "canonical_donor_identity": donor,
                "reference_imagej_frame": int(reference["imagej_frame"]),
                "reference_distance_px": float(reference["distance_px"]),
                "partitioned_frames": int(len(local)),
            })
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("shape-proven two-core correction changed foreground")
    return candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows), inferred


def _identity_frame_indices(labels: np.ndarray) -> dict[int, list[int]]:
    present: dict[int, list[int]] = {}
    for frame_index, frame in enumerate(labels):
        for identity_value in np.unique(frame):
            identity = int(identity_value)
            if identity > 0:
                present.setdefault(identity, []).append(frame_index)
    return present


def correct_recurrent_core_slots(
        labels: np.ndarray, raw: np.ndarray,
        params: RecurrentCoreSlotParams = RecurrentCoreSlotParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Split a host when one of its physical cores has a recurrent established alias.

    The accepted tracker often names both somas as the host in the failure frame but
    gives the second physical slot its own name elsewhere. The closest established
    non-host core trajectory identifies that second cell. A simultaneous substantial
    core under that name blocks the split, preventing duplicate cells.
    """
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)

    identity_frames = _identity_frame_indices(source)
    references: list[dict] = []
    for identity, frames in identity_frames.items():
        if len(frames) < params.minimum_reference_history_frames:
            continue
        for frame_index in frames:
            for _, mask in identity_components(source[frame_index], identity):
                for area, core in eroded_cores(mask, 1):
                    if int(area) < params.minimum_reference_core_px:
                        continue
                    centre = _centroid(core)
                    references.append({
                        "identity": int(identity),
                        "imagej_frame": int(frame_index + 1),
                        "history_frames": int(len(frames)),
                        "core_area_px": int(area),
                        "y": float(centre[0]), "x": float(centre[1]),
                    })

    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    for run in runs.itertuples():
        if float(run.maximum_two_core_score) < params.minimum_two_core_score:
            continue
        representative = int(run.representative_imagej_frame)
        obj = _run_object(objects, pd.Series(run._asdict()), representative)
        if obj is None:
            continue
        host = int(run.identity)
        positions = (
            np.array([obj.first_core_y, obj.first_core_x], float),
            np.array([obj.second_core_y, obj.second_core_x], float))
        choices: list[dict] = []
        for reference in references:
            identity = int(reference["identity"])
            if identity == host:
                continue
            point = np.array([reference["y"], reference["x"]], float)
            distances = [float(np.linalg.norm(point - position))
                         for position in positions]
            index = int(np.argmin(distances))
            if distances[index] > params.maximum_reference_distance_px:
                continue
            other_distance = min(
                float(np.linalg.norm(
                    np.array([row["y"], row["x"]], float)
                    - positions[1 - index]))
                for row in references
                if int(row["identity"]) == identity)
            # The same identity must describe one slot much better than the other.
            if other_distance - distances[index] < 2.0:
                continue
            choices.append({
                **reference, "core_index": index,
                "distance_px": distances[index],
                "other_core_distance_px": other_distance,
            })
        if not choices:
            continue
        choices.sort(key=lambda row: (
            row["distance_px"], -row["history_frames"],
            -row["core_area_px"], row["identity"]))
        selected = choices[0]
        donor = int(selected["identity"])
        donor_reference = positions[int(selected["core_index"])]
        host_reference = positions[1 - int(selected["core_index"])]
        local: list[dict] = []
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            existing = eroded_cores(source[imagej_frame - 1] == donor, 1)
            if existing and int(existing[0][0]) > params.maximum_existing_core_px:
                continue
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], host, int(frame_obj.component))
            if host_mask is None:
                continue
            frame_positions = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            donor_index = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in frame_positions]))
            direct = np.zeros(host_mask.shape, bool)
            direct[tuple(map(int, np.round(frame_positions[donor_index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host
            inferred[imagej_frame - 1][host_mask] = True
            donor_reference, host_reference = (
                _centroid(donor_part), _centroid(host_part))
            local.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "host_identity": host,
                "canonical_donor_identity": donor,
                "reference_imagej_frame": int(selected["imagej_frame"]),
                "reference_distance_px": float(selected["distance_px"]),
                **measures,
            })
        if local:
            frame_rows.extend(local)
            event_rows.append({
                "two_core_run_id": str(run.run_id),
                "first_imagej_frame": int(run.first_imagej_frame),
                "last_imagej_frame": int(run.last_imagej_frame),
                "host_identity": host,
                "canonical_donor_identity": donor,
                "reference_imagej_frame": int(selected["imagej_frame"]),
                "reference_distance_px": float(selected["distance_px"]),
                "partitioned_frames": int(len(local)),
            })
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("recurrent core-slot correction changed foreground")
    return candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows), inferred


def correct_structurally_proven_unresolved_runs(
        labels: np.ndarray, raw: np.ndarray,
        params: StructuralFallbackParams = StructuralFallbackParams(),
        detection_labels: np.ndarray | None = None,
        objects: pd.DataFrame | None = None,
        runs: pd.DataFrame | None = None,
        skip_run_ids: set[str] | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Resolve only strong structural pairs left after evidence-led corrections."""
    source = labels if detection_labels is None else detection_labels
    if source.shape != labels.shape:
        raise ValueError("detection and candidate labels must have the same shape")
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    if objects is None:
        objects = audit_two_core_objects(source, raw, params.two_core)
    if runs is None:
        runs = group_two_core_runs(
            objects, params.maximum_run_gap_frames,
            params.maximum_run_shift_px_per_frame)
    skipped = skip_run_ids or set()
    identity_frames = _identity_frame_indices(source)
    references: list[dict] = []
    for identity, frames in identity_frames.items():
        if len(frames) < params.minimum_reference_history_frames:
            continue
        for frame_index in frames:
            for _, mask in identity_components(source[frame_index], identity):
                cores = eroded_cores(mask, 1)
                if not cores or int(cores[0][0]) <= params.maximum_existing_core_px:
                    continue
                centre = _centroid(cores[0][1])
                references.append({
                    "identity": int(identity),
                    "imagej_frame": int(frame_index + 1),
                    "history_frames": int(len(frames)),
                    "y": float(centre[0]), "x": float(centre[1]),
                })
    event_rows: list[dict] = []
    frame_rows: list[dict] = []
    for run in runs.itertuples():
        if str(run.run_id) in skipped:
            continue
        representative = int(run.representative_imagej_frame)
        obj = _run_object(objects, pd.Series(run._asdict()), representative)
        if obj is None:
            continue
        strong_score = float(run.maximum_two_core_score) >= params.strong_two_core_score
        strong_separation = bool(
            float(obj.core_separation_px)
            >= params.strong_core_separation_px
            and float(obj.core_intensity_ratio)
            >= params.strong_core_intensity_ratio)
        repeated_pair = bool(
            int(run.flagged_frames) >= params.repeated_pair_minimum_frames
            and float(run.maximum_two_core_score) >= params.repeated_pair_score
            and float(obj.core_intensity_ratio)
            >= params.repeated_pair_intensity_ratio)
        if not (strong_score or strong_separation or repeated_pair):
            continue
        host = int(run.identity)
        positions = (
            np.array([obj.first_core_y, obj.first_core_x], float),
            np.array([obj.second_core_y, obj.second_core_x], float))
        choices: list[dict] = []
        for reference in references:
            identity = int(reference["identity"])
            if identity == host:
                continue
            point = np.array([reference["y"], reference["x"]], float)
            distances = [float(np.linalg.norm(point - position))
                         for position in positions]
            index = int(np.argmin(distances))
            if distances[index] <= params.maximum_reference_distance_px:
                choices.append({
                    **reference, "core_index": index,
                    "distance_px": distances[index],
                })
        if not choices:
            continue
        choices.sort(key=lambda row: (
            row["distance_px"], -row["history_frames"], row["identity"]))
        selected = choices[0]
        donor = int(selected["identity"])
        donor_reference = positions[int(selected["core_index"])]
        host_reference = positions[1 - int(selected["core_index"])]
        local: list[dict] = []
        for imagej_frame in range(
                int(run.first_imagej_frame), int(run.last_imagej_frame) + 1):
            existing = eroded_cores(source[imagej_frame - 1] == donor, 1)
            if existing and int(existing[0][0]) > params.maximum_existing_core_px:
                continue
            frame_obj = _run_object(
                objects, pd.Series(run._asdict()), imagej_frame)
            if frame_obj is None:
                continue
            host_mask = _component_mask(
                source[imagej_frame - 1], host, int(frame_obj.component))
            if host_mask is None:
                continue
            positions_here = (
                np.array([frame_obj.first_core_y, frame_obj.first_core_x]),
                np.array([frame_obj.second_core_y, frame_obj.second_core_x]))
            donor_index = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in positions_here]))
            direct = np.zeros(host_mask.shape, bool)
            direct[tuple(map(int, np.round(positions_here[donor_index])))] = True
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host
            inferred[imagej_frame - 1][host_mask] = True
            donor_reference, host_reference = (
                _centroid(donor_part), _centroid(host_part))
            local.append({
                "two_core_run_id": str(run.run_id),
                "imagej_frame": imagej_frame,
                "host_identity": host,
                "canonical_donor_identity": donor,
                "reference_imagej_frame": int(selected["imagej_frame"]),
                "reference_distance_px": float(selected["distance_px"]),
                "structural_gate": (
                    "strong_score" if strong_score else
                    "strong_separation" if strong_separation else
                    "repeated_pair"),
                **measures,
            })
        if local:
            frame_rows.extend(local)
            event_rows.append({
                "two_core_run_id": str(run.run_id),
                "first_imagej_frame": int(run.first_imagej_frame),
                "last_imagej_frame": int(run.last_imagej_frame),
                "host_identity": host,
                "canonical_donor_identity": donor,
                "reference_imagej_frame": int(selected["imagej_frame"]),
                "reference_distance_px": float(selected["distance_px"]),
                "structural_gate": local[0]["structural_gate"],
                "partitioned_frames": int(len(local)),
            })
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("structural fallback changed foreground support")
    return candidate, pd.DataFrame(event_rows), pd.DataFrame(frame_rows), inferred


def _component_mask(frame: np.ndarray, identity: int, component: int
                    ) -> np.ndarray | None:
    for number, mask in identity_components(frame, identity):
        if number == component:
            return mask
    return None


def _object_for_frame(objects: pd.DataFrame, identity: int, imagej_frame: int,
                      reference: np.ndarray) -> pd.Series | None:
    choices = objects[
        (objects.identity == identity)
        & (objects.imagej_frame == imagej_frame)].copy()
    if choices.empty:
        return None
    choices["reference_distance"] = np.hypot(
        choices.centre_y - float(reference[0]),
        choices.centre_x - float(reference[1]))
    return choices.sort_values(
        ["reference_distance", "two_core_score"],
        ascending=[True, False]).iloc[0]


def _centroid(mask: np.ndarray) -> np.ndarray:
    return np.mean(np.column_stack(np.nonzero(mask)), axis=0)


def _split_two_core_host(host: np.ndarray, raw: np.ndarray,
                         donor_reference: np.ndarray,
                         host_reference: np.ndarray,
                         minimum_partition_px: int,
                         direct_transfer: np.ndarray | None = None,
                         ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    cores = eroded_cores(host, 1)
    if len(cores) < 2:
        return None
    first_area, first = cores[0]
    second_area, second = cores[1]
    core_masks = (first, second)
    core_positions = (_centroid(first), _centroid(second))

    if direct_transfer is not None and np.any(direct_transfer & host):
        overlaps = [int(np.count_nonzero(core & direct_transfer))
                    for core in core_masks]
        if overlaps[0] != overlaps[1]:
            donor_index = int(np.argmax(overlaps))
        else:
            donor_index = int(np.argmin([
                np.linalg.norm(position - donor_reference)
                for position in core_positions]))
    else:
        costs = []
        for donor_index in (0, 1):
            host_index = 1 - donor_index
            costs.append(
                np.linalg.norm(core_positions[donor_index] - donor_reference)
                + np.linalg.norm(core_positions[host_index] - host_reference))
        donor_index = int(np.argmin(costs))
    host_index = 1 - donor_index

    markers = np.zeros(host.shape, np.uint8)
    markers[core_masks[donor_index]] = 1
    markers[core_masks[host_index]] = 2
    smooth = ndi.gaussian_filter(raw.astype(np.float32), sigma=1.3)
    divided = watershed(-smooth, markers=markers, mask=host,
                        connectivity=CONNECTIVITY)
    donor_part, host_part = divided == 1, divided == 2
    if (int(donor_part.sum()) < minimum_partition_px
            or int(host_part.sum()) < minimum_partition_px):
        return None
    if not np.array_equal(donor_part | host_part, host):
        return None
    if (ndi.label(donor_part, structure=CONNECTIVITY)[1] != 1
            or ndi.label(host_part, structure=CONNECTIVITY)[1] != 1):
        return None
    return donor_part, host_part, {
        "first_core_px": int(first_area),
        "second_core_px": int(second_area),
        "donor_core_px": int((first_area, second_area)[donor_index]),
        "host_core_px": int((first_area, second_area)[host_index]),
        "donor_seed_y": float(core_positions[donor_index][0]),
        "donor_seed_x": float(core_positions[donor_index][1]),
        "host_seed_y": float(core_positions[host_index][0]),
        "host_seed_x": float(core_positions[host_index][1]),
        "donor_partition_px": int(donor_part.sum()),
        "host_partition_px": int(host_part.sum()),
    }


def correct_partial_body_transfers(
        labels: np.ndarray, raw: np.ndarray,
        params: PartialBodyTransferParams = PartialBodyTransferParams(),
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return two-core soma pixels to an existing donor identity.

    No identity is created and no foreground pixel is added or removed. The triggering
    frame is partitioned first from exact transferred pixels; the same two identities
    are then carried backward and forward over every two-core frame in that run.
    """
    intervals, objects, transfers = detect_partial_body_transfer_intervals(
        labels, raw, params)
    candidate = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    rows: list[dict] = []

    for event in intervals.itertuples():
        event_t = int(event.t)
        donor = int(event.donor_identity)
        host_identity = int(event.host_identity)
        previous_donor = labels[event_t - 1] == donor
        previous_host = labels[event_t - 1] == host_identity
        if not np.any(previous_donor) or not np.any(previous_host):
            continue
        donor_reference = _centroid(previous_donor)
        host_reference = _centroid(previous_host)
        transfer = ((labels[event_t - 1] == donor)
                    & (labels[event_t] == host_identity))

        frames = list(range(int(event.run_first_imagej_frame),
                            int(event.run_last_imagej_frame) + 1))
        ordered_frames = [int(event.imagej_frame)]
        ordered_frames += list(range(int(event.imagej_frame) - 1,
                                     int(event.run_first_imagej_frame) - 1, -1))
        ordered_frames += list(range(int(event.imagej_frame) + 1,
                                     int(event.run_last_imagej_frame) + 1))
        ordered_frames = [frame for frame in ordered_frames if frame in frames]

        references: dict[int, tuple[np.ndarray, np.ndarray]] = {
            int(event.imagej_frame): (donor_reference, host_reference)}
        for imagej_frame in ordered_frames:
            if imagej_frame < int(event.imagej_frame):
                neighbour = imagej_frame + 1
            elif imagej_frame > int(event.imagej_frame):
                neighbour = imagej_frame - 1
            else:
                neighbour = imagej_frame
            if neighbour != imagej_frame and neighbour in references:
                donor_reference, host_reference = references[neighbour]

            obj = _object_for_frame(
                objects, host_identity, imagej_frame,
                (donor_reference + host_reference) / 2.0)
            if obj is None:
                continue
            host_mask = _component_mask(
                labels[imagej_frame - 1], host_identity, int(obj.component))
            if host_mask is None:
                continue
            direct = transfer if imagej_frame == int(event.imagej_frame) else None
            split = _split_two_core_host(
                host_mask, raw[imagej_frame - 1], donor_reference,
                host_reference, params.minimum_partition_px, direct)
            if split is None:
                continue
            donor_part, host_part, measures = split
            frame = candidate[imagej_frame - 1]
            frame[host_mask] = 0
            frame[donor_part] = donor
            frame[host_part] = host_identity
            if not np.array_equal(frame > 0, labels[imagej_frame - 1] > 0):
                raise AssertionError(
                    "partial body-transfer correction changed foreground support")
            inferred[imagej_frame - 1][host_mask] = True
            references[imagej_frame] = (
                _centroid(donor_part), _centroid(host_part))
            rows.append({
                "partial_transfer_id": str(event.partial_transfer_id),
                "two_core_run_id": str(event.two_core_run_id),
                "event_imagej_frame": int(event.imagej_frame),
                "imagej_frame": imagej_frame,
                "donor_identity": donor,
                "host_identity": host_identity,
                "partition_direction": (
                    "event" if imagej_frame == int(event.imagej_frame)
                    else "backward" if imagej_frame < int(event.imagej_frame)
                    else "forward"),
                **measures,
            })

    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError(
            "partial body-transfer correction changed movie foreground support")
    original_ids = set(map(int, np.unique(labels))) - {0}
    candidate_ids = set(map(int, np.unique(candidate))) - {0}
    if not candidate_ids.issubset(original_ids):
        raise AssertionError("partial body-transfer correction created an identity")
    return candidate, intervals, pd.DataFrame(rows), inferred


def params_as_dict(params: PartialBodyTransferParams) -> dict:
    return asdict(params)
