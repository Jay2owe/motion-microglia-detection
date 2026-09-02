from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from model_review import eroded_cores, identity_components
from persistence import green_evidence_for_fixed_objects
from persistent_merges import _split_interval
from tracking import frame_observations, track_from_anchor


CONNECTIVITY = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class GlobalSomaLedgerParams:
    anchor_bookend_frames: int = 16
    anchor_alias_radius_px: float = 18.0
    minimum_substantial_core_px: int = 12
    preferred_continuity_window_frames: int = 3
    preferred_continuity_step_px: float = 24.0
    minimum_preferred_support_frames: int = 2


def separate_substantial_soma_objects(
        labels: np.ndarray, minimum_core_px: int = 12,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Make each substantial disconnected soma a distinct frame observation.

    Small detached processes keep the nearest substantial soma.  The operation
    changes only local object numbers; it preserves every foreground pixel.
    """
    result = np.zeros_like(labels)
    rows: list[dict] = []
    for t, frame in enumerate(labels):
        local_object = 1
        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            components, count = ndi.label(
                frame == identity, structure=CONNECTIVITY)
            records: list[dict] = []
            for component in range(1, count + 1):
                mask = components == component
                cores = eroded_cores(mask, 1)
                records.append({
                    "mask": mask,
                    "area": int(mask.sum()),
                    "core": int(cores[0][0]) if cores else 0,
                    "centre": np.mean(
                        np.column_stack(np.nonzero(mask)), axis=0),
                })
            bodies = [row for row in records
                      if row["core"] >= minimum_core_px]
            if len(bodies) <= 1:
                mask = frame == identity
                result[t][mask] = local_object
                rows.append({
                    "t": t, "imagej_frame": t + 1,
                    "local_object": local_object,
                    "source_identity": identity,
                    "substantial_bodies": len(bodies),
                    "source_components": count,
                    "area_px": int(mask.sum()),
                })
                local_object += 1
                continue

            assigned = [row["mask"].copy() for row in bodies]
            distances = [ndi.distance_transform_edt(~mask)
                         for mask in assigned]
            body_ids = {id(row) for row in bodies}
            for record in records:
                if id(record) in body_ids:
                    continue
                y, x = np.round(record["centre"]).astype(int)
                nearest = int(np.argmin(
                    [distance[y, x] for distance in distances]))
                assigned[nearest] |= record["mask"]
            for mask in assigned:
                result[t][mask] = local_object
                rows.append({
                    "t": t, "imagej_frame": t + 1,
                    "local_object": local_object,
                    "source_identity": identity,
                    "substantial_bodies": len(bodies),
                    "source_components": count,
                    "area_px": int(mask.sum()),
                })
                local_object += 1
        if not np.array_equal(result[t] > 0, frame > 0):
            raise AssertionError(
                f"frame {t + 1}: soma observations changed foreground support")
    return result, pd.DataFrame(rows)


def _substantial_identity_references(
        labels: np.ndarray, first: int, last: int,
        minimum_core_px: int,
        ) -> list[dict]:
    rows: list[dict] = []
    for t in range(max(0, first), min(len(labels), last + 1)):
        for identity in sorted(set(map(int, np.unique(labels[t]))) - {0}):
            for _, mask in identity_components(labels[t], identity):
                cores = eroded_cores(mask, 1)
                if not cores or int(cores[0][0]) < minimum_core_px:
                    continue
                centre = np.mean(
                    np.column_stack(np.nonzero(cores[0][1])), axis=0)
                rows.append({
                    "t": t, "identity": identity,
                    "y": float(centre[0]), "x": float(centre[1]),
                })
    return rows


def resolve_anchor_identity_map(
        physical_objects: np.ndarray, accepted_labels: np.ndarray,
        anchor_t: int,
        params: GlobalSomaLedgerParams = GlobalSomaLedgerParams(),
        ) -> tuple[dict[int, int], pd.DataFrame]:
    """Give each midpoint soma one established name using nearby bookends."""
    local_ids = sorted(set(map(int, np.unique(
        physical_objects[anchor_t]))) - {0})
    mapping: dict[int, int] = {}
    centres: dict[int, np.ndarray] = {}
    for local in local_ids:
        mask = physical_objects[anchor_t] == local
        centres[local] = np.mean(
            np.column_stack(np.nonzero(mask)), axis=0)
        values, counts = np.unique(accepted_labels[anchor_t][mask],
                                   return_counts=True)
        choices = [(int(count), int(value))
                   for value, count in zip(values, counts) if value > 0]
        if not choices:
            raise ValueError(f"anchor object {local} has no accepted identity")
        mapping[local] = max(choices)[1]

    grouped: dict[int, list[int]] = {}
    for local, identity in mapping.items():
        grouped.setdefault(identity, []).append(local)
    duplicated = {identity: locals_here for identity, locals_here in grouped.items()
                  if len(locals_here) > 1}
    used_by_unique = {identity for identity, locals_here in grouped.items()
                      if len(locals_here) == 1}
    references = _substantial_identity_references(
        accepted_labels,
        anchor_t - params.anchor_bookend_frames,
        anchor_t + params.anchor_bookend_frames,
        params.minimum_substantial_core_px)
    audit: list[dict] = []
    for original, duplicate_locals in sorted(duplicated.items()):
        candidate_ids = sorted({int(row["identity"]) for row in references
                                if int(row["identity"]) not in used_by_unique})
        score = np.full(
            (len(duplicate_locals), len(candidate_ids)), -1e6, float)
        support: dict[tuple[int, int], tuple[int, float]] = {}
        for row_index, local in enumerate(duplicate_locals):
            centre = centres[local]
            for column, identity in enumerate(candidate_ids):
                distances = [
                    float(np.linalg.norm(centre - np.array([row["y"], row["x"]])))
                    for row in references if int(row["identity"]) == identity]
                near = [distance for distance in distances
                        if distance <= params.anchor_alias_radius_px]
                if not near:
                    continue
                support[(local, identity)] = (len(near), float(np.median(near)))
                score[row_index, column] = len(near) - 0.05 * np.median(near)
        rows, columns = linear_sum_assignment(-score)
        if len(rows) != len(duplicate_locals):
            raise AssertionError(f"duplicate anchor identity {original} unresolved")
        for row_index, column in zip(rows, columns):
            if score[row_index, column] <= -1e5:
                raise AssertionError(
                    f"duplicate anchor identity {original} lacks a bookend alias")
            local = duplicate_locals[int(row_index)]
            identity = candidate_ids[int(column)]
            mapping[local] = identity
            frames, median_distance = support[(local, identity)]
            audit.append({
                "duplicated_identity": original,
                "local_object": local,
                "resolved_identity": identity,
                "bookend_support_frames": frames,
                "median_bookend_distance_px": median_distance,
            })
        used_by_unique.update(mapping[local] for local in duplicate_locals)
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError("anchor identity map is not one-to-one")
    return mapping, pd.DataFrame(audit)


def physical_soma_tracking_evidence(
        physical_objects: np.ndarray, accepted_labels: np.ndarray,
        observation_labels: np.ndarray, observation_table: pd.DataFrame,
        continuity_window_frames: int = 3,
        continuity_step_px: float = 24.0,
        minimum_support_frames: int = 2,
        ) -> pd.DataFrame:
    green = green_evidence_for_fixed_objects(
        physical_objects, observation_labels, observation_table)
    accepted_centres: dict[tuple[int, int], np.ndarray] = {}
    for t, frame in enumerate(accepted_labels):
        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            accepted_centres[t, identity] = np.mean(
                np.column_stack(np.nonzero(frame == identity)), axis=0)
    rows: list[dict] = []
    for t, frame in enumerate(physical_objects):
        for local in sorted(set(map(int, np.unique(frame))) - {0}):
            mask = frame == local
            values, counts = np.unique(accepted_labels[t][mask],
                                       return_counts=True)
            choices = [(int(count), int(value))
                       for value, count in zip(values, counts) if value > 0]
            preferred = max(choices)[1] if choices else 0
            centre = np.mean(np.column_stack(np.nonzero(mask)), axis=0)
            before_support = 0
            after_support = 0
            if preferred > 0:
                for delta in range(1, int(continuity_window_frames) + 1):
                    for sign in (-1, 1):
                        other_t = t + sign * delta
                        other = accepted_centres.get((other_t, preferred))
                        if (other is not None
                                and float(np.linalg.norm(centre - other))
                                <= float(continuity_step_px) * delta):
                            if sign < 0:
                                before_support += 1
                            else:
                                after_support += 1
            support_frames = before_support + after_support
            rows.append({
                "t": t, "local_id": local,
                "preferred_identity": preferred,
                "preferred_support_before_frames": before_support,
                "preferred_support_after_frames": after_support,
                "preferred_identity_protected": bool(
                    preferred > 0
                    and support_frames >= int(minimum_support_frames)),
            })
    return green.merge(pd.DataFrame(rows), on=["t", "local_id"], how="left")


def track_physical_somas(
        physical_objects: np.ndarray, accepted_labels: np.ndarray,
        raw: np.ndarray, lag: np.ndarray, anchor_t: int,
        tracking_params: dict, observation_labels: np.ndarray,
        observation_table: pd.DataFrame,
        ledger_params: GlobalSomaLedgerParams = GlobalSomaLedgerParams(),
        prepared_observations: list[dict[int, dict]] | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        anchor_future = executor.submit(
            resolve_anchor_identity_map,
            physical_objects, accepted_labels, anchor_t, ledger_params)
        evidence_future = executor.submit(
            physical_soma_tracking_evidence,
            physical_objects, accepted_labels,
            observation_labels, observation_table,
            ledger_params.preferred_continuity_window_frames,
            ledger_params.preferred_continuity_step_px,
            ledger_params.minimum_preferred_support_frames)
        anchor_map, anchor_audit = anchor_future.result()
        evidence = evidence_future.result()
    candidate, links = track_from_anchor(
        physical_objects, raw, lag, anchor_t, tracking_params,
        evidence, anchor_map, prepared_observations)
    if not np.array_equal(candidate > 0, physical_objects > 0):
        raise AssertionError("global soma tracking changed foreground support")
    return candidate, links, evidence, anchor_audit


def add_field_discovered_bookend_observations(
        physical_objects: np.ndarray, tracked_labels: np.ndarray,
        raw: np.ndarray, lag: np.ndarray, structural_objects: pd.DataFrame,
        structural_runs: pd.DataFrame, merge_params: dict,
        minimum_structural_score: float = 0.8,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Add observations for structural pairs with temporary-merge bookends.

    Full-field two-core runs nominate interactions, not cells. A run is actionable when
    its two detected cores have distinct tracked names at the representative bookend,
    then become one name, then separate again within the general bookend horizon. Every
    run is accepted or refused with a reason.
    """
    if physical_objects.shape != tracked_labels.shape:
        raise ValueError("physical objects and tracked labels differ in shape")
    result = physical_objects.copy()
    audit_rows: list[dict] = []
    accepted_keys: set[tuple[int, int, int, int]] = set()
    event_number = 0
    for run in structural_runs.itertuples(index=False):
        base = {
            "structural_run_id": str(run.run_id),
            "representative_imagej_frame": int(
                run.representative_imagej_frame),
            "maximum_two_core_score": float(run.maximum_two_core_score),
        }
        if float(run.maximum_two_core_score) < float(minimum_structural_score):
            audit_rows.append({
                **base, "decision": "rejected_structural_score",
                "reason": "full-field two-core evidence is below the general gate",
            })
            continue
        source_rows = structural_objects[
            structural_objects.object_id.astype(str).eq(
                str(run.representative_object_id))]
        if len(source_rows) != 1:
            audit_rows.append({
                **base, "decision": "rejected_missing_representative_object",
                "reason": "structural run does not map to exactly one detected object",
            })
            continue
        obj = source_rows.iloc[0]
        representative = int(run.representative_imagej_frame) - 1
        identity_a = int(tracked_labels[
            representative, int(round(obj.first_core_y)),
            int(round(obj.first_core_x))])
        identity_b = int(tracked_labels[
            representative, int(round(obj.second_core_y)),
            int(round(obj.second_core_x))])
        base.update({"identity_a": identity_a, "identity_b": identity_b})
        if identity_a <= 0 or identity_b <= 0 or identity_a == identity_b:
            audit_rows.append({
                **base, "decision": "rejected_no_separated_identity_bookend",
                "reason": "the detected cores do not carry two distinct tracked names",
            })
            continue
        horizon = int(merge_params["maximum_bookend_frames"])
        stop = min(len(result), representative + horizon + 1)
        start_t = next((
            t for t in range(representative + 1, stop)
            if not np.any(tracked_labels[t] == identity_a)
            and np.any(tracked_labels[t] == identity_b)), None)
        if start_t is None:
            audit_rows.append({
                **base, "decision": "rejected_no_temporary_merge",
                "reason": "both tracked names remain separate through the horizon",
            })
            continue
        future_stop = min(len(result), start_t + horizon + 1)
        future_t = next((
            t for t in range(start_t + 1, future_stop)
            if np.any(tracked_labels[t] == identity_a)
            and np.any(tracked_labels[t] == identity_b)), None)
        if future_t is None:
            audit_rows.append({
                **base, "start_t": start_t,
                "decision": "rejected_no_separated_return_bookend",
                "reason": "the vanished name does not return beside its host",
            })
            continue
        previous = frame_observations(
            tracked_labels[start_t - 1], raw[start_t - 1])
        current = frame_observations(tracked_labels[start_t], raw[start_t])
        history_start = max(
            0, start_t - int(merge_params["minimum_pre_event_frames"]))
        stable_history = all(
            np.any(tracked_labels[t] == identity_a)
            for t in range(history_start, start_t))
        if (not stable_history or identity_a not in previous
                or identity_b not in previous or identity_b not in current):
            audit_rows.append({
                **base, "start_t": start_t, "bookend_t": future_t,
                "decision": "rejected_unstable_pre_merge_identity",
                "reason": "the disappearing identity lacks stable pre-merge evidence",
            })
            continue
        expanded = ndi.binary_dilation(current[identity_b]["mask"], iterations=2)
        absorbed = float(np.mean(expanded[previous[identity_a]["mask"]]))
        host_pixels = np.column_stack(np.nonzero(current[identity_b]["mask"]))
        distance = float(np.min(np.linalg.norm(
            host_pixels - previous[identity_a]["position"][None, :], axis=1)))
        host_growth = current[identity_b]["area"] - previous[identity_b]["area"]
        growth_fraction = host_growth / max(previous[identity_a]["area"], 1)
        base.update({
            "host_distance_px": distance,
            "absorbed_fraction": absorbed,
            "host_growth_fraction": growth_fraction,
        })
        if not (
                distance <= float(merge_params["merge_reach_px"])
                and absorbed >= float(merge_params["minimum_absorbed_fraction"])
                and float(merge_params["minimum_host_growth_fraction"])
                <= growth_fraction
                <= float(merge_params["maximum_host_growth_fraction"])):
            audit_rows.append({
                **base, "start_t": start_t, "bookend_t": future_t,
                "decision": "rejected_merge_geometry",
                "reason": "absorption, distance, or host growth fails the general gate",
            })
            continue
        key = (start_t, future_t, min(identity_a, identity_b),
               max(identity_a, identity_b))
        if key in accepted_keys:
            audit_rows.append({
                **base, "start_t": start_t, "bookend_t": future_t,
                "decision": "rejected_duplicate_interaction",
                "reason": "an earlier structural run already supplied this interaction",
            })
            continue
        partition = _split_interval(
            tracked_labels, raw, lag, start_t, future_t,
            identity_a, identity_b, merge_params)
        if partition is None:
            audit_rows.append({
                **base, "start_t": start_t, "bookend_t": future_t,
                "decision": "rejected_partition_evidence",
                "reason": "persistent-merge partition evidence was insufficient",
            })
            continue
        partitioned, frame_rows = partition
        checks: list[tuple[int, np.ndarray, set[int]]] = []
        for frame_row in frame_rows:
            t = int(frame_row["t"])
            pair_mask = ((partitioned[t] == identity_a)
                         | (partitioned[t] == identity_b))
            source_objects = set(map(
                int, np.unique(result[t][pair_mask]))) - {0}
            checks.append((t, pair_mask, source_objects))
        applicable = bool(checks) and all(
            len(source_objects) == 1
            for _, _, source_objects in checks)
        if not applicable:
            audit_rows.append({
                **base, "start_t": start_t, "bookend_t": future_t,
                "decision": "rejected_already_separate_or_unmapped",
                "reason": ("the partition does not map to one physical object "
                           "in every interval frame"),
                "physical_object_count": min(
                    (len(row[2]) for row in checks), default=0),
            })
            continue
        event_number += 1
        event_id = f"SM{event_number:04d}"
        accepted_keys.add(key)
        for (t, pair_mask, source_objects), frame_row in zip(checks, frame_rows):
            retained = next(iter(source_objects))
            new_object = int(result[t].max()) + 1
            result[t][pair_mask] = 0
            result[t][partitioned[t] == identity_b] = retained
            result[t][partitioned[t] == identity_a] = new_object
            audit_rows.append({
                **base, "event_id": event_id,
                "start_t": start_t, "bookend_t": future_t,
                "decision": "accepted_single_physical_object",
                "reason": ("two structural cores have separated identity bookends "
                           "and share one physical object during the interval"),
                "physical_object_count": 1,
                **frame_row,
            })
    if not np.array_equal(result > 0, physical_objects > 0):
        raise AssertionError("field-discovered observation split changed foreground")
    return result, pd.DataFrame(audit_rows)


def substantial_identity_conflicts(
        labels: np.ndarray, minimum_core_px: int = 12) -> pd.DataFrame:
    rows: list[dict] = []
    for t, frame in enumerate(labels):
        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            bodies = 0
            for _, mask in identity_components(frame, identity):
                cores = eroded_cores(mask, 1)
                bodies += bool(cores and int(cores[0][0]) >= minimum_core_px)
            if bodies > 1:
                rows.append({
                    "imagej_frame": t + 1,
                    "identity": identity,
                    "substantial_bodies": bodies,
                })
    return pd.DataFrame(rows)


def reuse_retired_identity_names(
        candidate: np.ndarray, accepted_labels: np.ndarray,
        minimum_overlap_px: int = 1,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Replace temporary tracker names with unused accepted identity names.

    The assignment maximizes whole-movie pixel overlap and is one-to-one.  Existing
    candidate names never change, so approved reserved identities remain untouched.
    """
    candidate_ids = sorted(set(map(int, np.unique(candidate))) - {0})
    accepted_ids = sorted(set(map(int, np.unique(accepted_labels))) - {0})
    novel = sorted(set(candidate_ids) - set(accepted_ids))
    retired = sorted(set(accepted_ids) - set(candidate_ids))
    if not novel:
        return candidate.copy(), pd.DataFrame(columns=[
            "temporary_identity", "reused_identity", "overlap_px"])
    # Temporary names are valid ledger names. Reuse as many retired accepted names as
    # the evidence supports, but do not fail merely because a stricter retrack exposes
    # more physical tracks than the fragmented accepted result had unused names.
    if not retired:
        return candidate.copy(), pd.DataFrame(columns=[
            "temporary_identity", "reused_identity", "overlap_px"])
    overlap = np.zeros((len(novel), len(retired)), np.int64)
    novel_index = {identity: index for index, identity in enumerate(novel)}
    retired_index = {identity: index for index, identity in enumerate(retired)}
    for t in range(len(candidate)):
        pairs = np.column_stack((candidate[t].ravel(), accepted_labels[t].ravel()))
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        values, counts = np.unique(pairs, axis=0, return_counts=True)
        for (temporary, accepted), count in zip(values, counts):
            temporary = int(temporary)
            accepted = int(accepted)
            if temporary in novel_index and accepted in retired_index:
                overlap[novel_index[temporary], retired_index[accepted]] += int(count)
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {
        novel[int(row)]: retired[int(column)]
        for row, column in zip(rows, columns)
        if int(overlap[int(row), int(column)]) >= minimum_overlap_px}
    result = candidate.copy()
    for temporary, reused in mapping.items():
        result[candidate == temporary] = reused
    final_ids = set(map(int, np.unique(result))) - {0}
    if len(final_ids) != len(candidate_ids):
        raise AssertionError("retired-name reuse changed the identity count")
    audit = pd.DataFrame([
        {
            "temporary_identity": temporary,
            "reused_identity": reused,
            "overlap_px": int(overlap[
                novel_index[temporary], retired_index[reused]]),
        }
        for temporary, reused in sorted(mapping.items())
    ])
    return result, audit


def restore_reserved_identity_names(
        candidate: np.ndarray, guide: np.ndarray, reserved_identity_ids: list[int],
        minimum_guide_overlap: float = 0.5,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Put approved reserved names back on the physical objects in their guide paths.

    This runs after global assignment so a single reservation cannot trigger a chain of
    displaced observations and new names across the field. If the reserved name already
    occupies another object in that frame, the two names are swapped, preserving both
    physical objects and exclusive ownership.
    """
    if candidate.shape != guide.shape:
        raise ValueError("candidate and reserved-identity guide differ in shape")
    fixed = candidate.copy()
    rows: list[dict] = []
    for identity in sorted(map(int, reserved_identity_ids)):
        for t in range(len(fixed)):
            guide_mask = guide[t] == identity
            if not guide_mask.any():
                continue
            values, counts = np.unique(fixed[t][guide_mask], return_counts=True)
            choices = sorted(
                ((int(count), int(value)) for value, count in zip(values, counts)
                 if int(value) > 0), reverse=True)
            if not choices:
                continue
            overlap_px, owner = choices[0]
            coverage = overlap_px / max(int(guide_mask.sum()), 1)
            if coverage < float(minimum_guide_overlap) or owner == identity:
                continue
            owner_mask = fixed[t] == owner
            reserved_mask = fixed[t] == identity
            fixed[t][reserved_mask] = owner
            fixed[t][owner_mask] = identity
            rows.append({
                "identity": identity,
                "imagej_frame": t + 1,
                "displaced_identity": owner,
                "guide_overlap_px": overlap_px,
                "guide_coverage": float(coverage),
                "reserved_pixels": int(owner_mask.sum()),
                "displaced_pixels": int(reserved_mask.sum()),
            })
    if not np.array_equal(fixed > 0, candidate > 0):
        raise AssertionError("reserved identity restoration changed foreground support")
    return fixed, pd.DataFrame(rows)
