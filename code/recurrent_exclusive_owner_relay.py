"""Continue dedicated identity seats across recurrent shared-owner relays.

Discovery is field-wide.  The producer accepts mathematical thresholds and complete
physical-track evidence only; review identities, tracks, frames, events and regions
are forbidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed


import separable_merge_recovery as accepted_merge
import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
FORBIDDEN_TARGETS = {
    "review_cases_path", "case_ids", "event_ids", "event_targets",
    "identity_ids", "identity_targets", "track_ids", "track_targets",
    "frame_ids", "frame_targets", "coordinates", "coordinate_targets",
    "forced_identity_ids", "forced_intervals", "review_regions",
}


def _component_excess(frame: np.ndarray, identity: int) -> int:
    return component_accounting.component_excess(frame, identity)


def _new_duplicate_components(baseline: np.ndarray,
                              candidate: np.ndarray) -> int:
    return component_accounting.new_duplicate_components(
        baseline, candidate)


def _inside(mask: np.ndarray, row) -> bool:
    y = int(np.clip(round(float(row.y)), 0, mask.shape[0] - 1))
    x = int(np.clip(round(float(row.x)), 0, mask.shape[1] - 1))
    return bool(mask[y, x])


def _positive_runs(group: pd.DataFrame) -> list[dict]:
    runs: list[dict] = []
    for row in group.sort_values("frame").itertuples(index=False):
        if not bool(row.physically_visible):
            continue
        owner = int(row.accepted_owner)
        if owner <= 0:
            continue
        frame = int(row.frame)
        if (runs and runs[-1]["owner"] == owner
                and frame == runs[-1]["end"] + 1):
            runs[-1]["end"] = frame
            runs[-1]["frames"].append(frame)
        else:
            runs.append({"owner": owner, "start": frame, "end": frame,
                         "frames": [frame]})
    return runs


def discover_relays(scored: pd.DataFrame, frame_count: int,
                    params: dict) -> tuple[list[dict], pd.DataFrame]:
    """Return track-level relay hypotheses using owner multiplicity, not IDs."""
    visible = scored[scored.physically_visible.astype(bool)].copy()
    owner_tracks: dict[int, set[int]] = {}
    for row in visible.itertuples(index=False):
        owner = int(row.accepted_owner)
        if owner > 0:
            owner_tracks.setdefault(owner, set()).add(int(row.track_id))

    minimum_visible = max(
        int(params.get("minimum_visible_frames", 20)),
        int(np.ceil(frame_count * float(
            params.get("minimum_visible_fraction", 0.33)))))
    minimum_transitions = int(params.get("minimum_owner_transitions", 4))
    minimum_support = int(params.get("minimum_dedicated_support_frames", 8))
    minimum_fraction = float(params.get("minimum_dedicated_support_fraction", 0.2))
    minimum_runs = int(params.get("minimum_dedicated_runs", 2))
    maximum_gap = int(params.get("maximum_internal_track_gap_frames", 1))
    minimum_shared_tracks = int(params.get("minimum_competing_owner_tracks", 2))
    hypotheses: list[dict] = []
    audits: list[dict] = []

    for track_id, group in visible.groupby("track_id"):
        track_id = int(track_id)
        ordered = group.sort_values("frame")
        frames = ordered.frame.astype(int).to_numpy()
        runs = _positive_runs(ordered)
        transitions = max(0, len(runs) - 1)
        maximum_internal_gap = int(np.max(np.diff(frames)) - 1) if len(frames) > 1 else 0
        base = {
            "track_id": track_id,
            "visible_frames": int(len(ordered)),
            "owner_transitions": transitions,
            "maximum_internal_gap_frames": maximum_internal_gap,
            "eligible": False,
            "applied": False,
            "reason": "",
        }
        if len(ordered) < minimum_visible:
            audits.append({**base, "reason": "insufficient_visible_history"})
            continue
        if maximum_internal_gap > maximum_gap:
            audits.append({**base, "reason": "physical_history_not_continuous"})
            continue
        if transitions < minimum_transitions:
            audits.append({**base, "reason": "insufficient_owner_transitions"})
            continue

        counts = ordered[ordered.accepted_owner.astype(int) > 0].groupby(
            "accepted_owner").size().sort_values(ascending=False)
        candidates: list[dict] = []
        for owner_value, support_value in counts.items():
            owner = int(owner_value)
            support = int(support_value)
            owner_runs = [run for run in runs if int(run["owner"]) == owner]
            if len(owner_tracks.get(owner, set())) != 1:
                continue
            if support < max(minimum_support,
                             int(np.ceil(len(ordered) * minimum_fraction))):
                continue
            if len(owner_runs) < minimum_runs:
                continue
            competing = [
                int(value) for value in counts.index
                if int(value) != owner
                and len(owner_tracks.get(int(value), set())) >= minimum_shared_tracks
            ]
            if not competing:
                continue
            candidates.append({
                "dedicated_owner": owner,
                "dedicated_support_frames": support,
                "dedicated_support_fraction": float(support / len(ordered)),
                "dedicated_runs": len(owner_runs),
                "first_dedicated_frame": int(owner_runs[0]["start"]),
                "competing_owners": competing,
            })
        if not candidates:
            audits.append({**base, "reason": "no_unique_repeated_owner"})
            continue
        candidates.sort(key=lambda item: (
            -item["dedicated_support_frames"],
            item["first_dedicated_frame"], item["dedicated_owner"]))
        selected = candidates[0]
        hypothesis = {**base, **selected, "eligible": True,
                      "reason": "eligible"}
        hypotheses.append(hypothesis)
        audits.append(hypothesis.copy())
    return hypotheses, pd.DataFrame(audits)


def _stable_core_rows(by_frame: dict[int, pd.DataFrame], frame: int,
                      component: np.ndarray, owner: int, excluded_track: int,
                      canonical: dict[int, dict]) -> list:
    result = []
    for row in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
        track_id = int(row.track_id)
        if track_id == excluded_track or not _inside(component, row):
            continue
        stable = canonical.get(track_id)
        if stable is not None and int(stable["owner"]) == owner:
            result.append(row)
    return result


def _prepare_frame(candidate: np.ndarray, candidate_unclaimed: np.ndarray,
                   baseline_labels: np.ndarray, baseline_unclaimed: np.ndarray,
                   raw: np.ndarray, by_frame: dict[int, pd.DataFrame],
                   canonical: dict[int, dict], row, dedicated_owner: int,
                   unclaimed_id: int, params: dict,
                   ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    frame = int(row.frame)
    track_id = int(row.track_id)
    host_owner = int(row.accepted_owner)
    detail = {"frame": frame, "host_owner": host_owner,
              "changed_pixels": 0, "orphan_pixels": 0,
              "recovered_pixels": 0, "valley_ratio": np.nan,
              "reason": ""}
    if host_owner <= 0 or host_owner == dedicated_owner:
        return None, None, {**detail, "reason": "not_positive_wrong_owner"}

    target_mask = candidate[frame] == dedicated_owner
    if np.any(target_mask):
        target_components, target_count = ndi.label(target_mask, STRUCTURE)
        for component_id in range(1, target_count + 1):
            component = target_components == component_id
            for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
                if int(other.track_id) != track_id and _inside(component, other):
                    return None, None, {
                        **detail, "reason": "dedicated_seat_has_other_raw_core"}

    host_components, _ = ndi.label(candidate[frame] == host_owner, STRUCTURE)
    component_id = accepted_merge._component_at(
        host_components, float(row.x), float(row.y))
    if component_id <= 0:
        return None, None, {**detail, "reason": "resident_outside_host_component"}
    shared = host_components == component_id
    companions = _stable_core_rows(
        by_frame, frame, shared, host_owner, track_id, canonical)
    all_other_cores = [
        other for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False)
        if int(other.track_id) != track_id and _inside(shared, other)]
    sigma = float(params.get("watershed_sigma_px", 1.0))
    maximum_distance = float(params.get("maximum_companion_distance_px", 30.0))
    maximum_valley = float(params.get("maximum_separable_valley_ratio", 0.9))

    if companions:
        companions = sorted(companions, key=lambda other: (
            float(np.hypot(float(row.x) - float(other.x),
                           float(row.y) - float(other.y))),
            int(other.track_id)))
        nearest = companions[0]
        distance = float(np.hypot(float(row.x) - float(nearest.x),
                                  float(row.y) - float(nearest.y)))
        if distance < 4.0 or distance > maximum_distance:
            return None, None, {**detail, "reason": "companion_distance_out_of_range"}
        valley = accepted_merge._valley_ratio(
            raw[frame], (float(row.x), float(row.y)),
            (float(nearest.x), float(nearest.y)))
        if valley > maximum_valley:
            return None, None, {**detail, "valley_ratio": valley,
                                "reason": "raw_cores_not_separable"}
        resident_marker = accepted_merge._marker_pixel(
            shared, float(row.x), float(row.y))
        if resident_marker is None:
            return None, None, {**detail, "reason": "resident_marker_unavailable"}
        markers = np.zeros(candidate.shape[1:], np.int16)
        markers[resident_marker] = 1
        for companion in companions:
            marker = accepted_merge._marker_pixel(
                shared, float(companion.x), float(companion.y))
            if marker is not None and marker != resident_marker:
                markers[marker] = 2
        if not np.any(markers == 2):
            return None, None, {**detail, "reason": "host_markers_unavailable"}
        elevation = -ndi.gaussian_filter(raw[frame].astype(np.float32), sigma)
        recovered = watershed(elevation, markers=markers, mask=shared) == 1
    else:
        if all_other_cores:
            return None, None, {**detail, "reason": "shared_component_has_unstable_core"}
        retained = False
        for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
            if int(other.track_id) == track_id:
                continue
            stable = canonical.get(int(other.track_id))
            if stable is None or int(stable["owner"]) != host_owner:
                continue
            other_component = accepted_merge._component_at(
                host_components, float(other.x), float(other.y))
            if other_component > 0 and other_component != component_id:
                retained = True
                break
        if not retained:
            return None, None, {**detail, "reason": "host_has_no_retained_raw_core"}
        recovered = shared.copy()
        valley = np.nan

    recovered_area = int(np.count_nonzero(recovered))
    orphan_area = int(np.count_nonzero(target_mask))
    maximum_orphan_ratio = float(
        params.get("maximum_orphan_to_recovered_area_ratio", 6.0))
    if orphan_area > maximum_orphan_ratio * max(recovered_area, 1):
        return None, None, {**detail, "valley_ratio": valley,
                            "reason": "orphan_component_too_large"}
    if recovered_area < int(params.get("minimum_changed_pixels", 5)):
        return None, None, {**detail, "valley_ratio": valley,
                            "reason": "recovered_core_too_small"}

    trial = candidate[frame].copy()
    unclaimed_trial = candidate_unclaimed[frame].copy()
    trial[target_mask] = 0
    unclaimed_trial[target_mask] = unclaimed_id
    trial[recovered] = dedicated_owner

    host_after, host_count = ndi.label(trial == host_owner, STRUCTURE)
    retained_components: set[int] = set()
    for other in by_frame.get(frame, pd.DataFrame()).itertuples(index=False):
        stable = canonical.get(int(other.track_id))
        if stable is None or int(stable["owner"]) != host_owner:
            continue
        retained_id = accepted_merge._component_at(
            host_after, float(other.x), float(other.y))
        if retained_id > 0:
            retained_components.add(retained_id)
    coreless_host = np.zeros_like(recovered)
    for retained_id in range(1, host_count + 1):
        if retained_id not in retained_components:
            coreless_host |= host_after == retained_id
    if np.any(coreless_host):
        trial[coreless_host] = dedicated_owner

    if not np.any(trial == dedicated_owner) or not np.any(trial == host_owner):
        return None, None, {**detail, "valley_ratio": valley,
                            "reason": "identity_extinguished"}
    if (_component_excess(trial, dedicated_owner)
            > _component_excess(baseline_labels[frame], dedicated_owner)):
        return None, None, {**detail, "valley_ratio": valley,
                            "reason": "new_dedicated_owner_duplicate"}
    if (_component_excess(trial, host_owner)
            > _component_excess(baseline_labels[frame], host_owner)):
        return None, None, {**detail, "valley_ratio": valley,
                            "reason": "new_host_owner_duplicate"}
    if not np.array_equal(
            (trial > 0) | (unclaimed_trial > 0),
            (baseline_labels[frame] > 0) | (baseline_unclaimed[frame] > 0)):
        return None, None, {**detail, "valley_ratio": valley,
                            "reason": "foreground_ledger_union_changed"}
    changed = int(np.count_nonzero(trial != candidate[frame]))
    return trial, unclaimed_trial, {
        **detail, "changed_pixels": changed,
        "orphan_pixels": orphan_area, "recovered_pixels": recovered_area,
        "valley_ratio": valley, "reason": "prepared"}


def recover(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict,
            ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    supplied = sorted(name for name in FORBIDDEN_TARGETS
                      if params.get(name) not in (None, "", [], {}))
    if supplied:
        raise ValueError(
            "field-wide exclusive-owner relay received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("exclusive-owner relay must use field-wide discovery")
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")

    scored = accepted_merge.attach_owners(points, labels)
    visible = scored[scored.physically_visible.astype(bool)].copy()
    canonical = accepted_merge.canonical_owners(
        scored,
        int(params.get("minimum_canonical_support_frames", 8)),
        float(params.get("minimum_canonical_purity", 0.8)),
        bool(params.get("canonical_exclude_encounter_frames", False)))
    by_frame = {int(frame): group for frame, group in visible.groupby("frame")}
    hypotheses, track_audit = discover_relays(scored, len(labels), params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    occupied = np.zeros_like(labels, bool)
    next_unclaimed = int(np.max(unclaimed)) + 1
    frame_audits: list[dict] = []

    for hypothesis in hypotheses:
        track_id = int(hypothesis["track_id"])
        dedicated_owner = int(hypothesis["dedicated_owner"])
        first_frame = int(hypothesis["first_dedicated_frame"])
        track_rows = visible[
            (visible.track_id.astype(int) == track_id)
            & (visible.frame.astype(int) >= first_frame)
            & (visible.accepted_owner.astype(int) > 0)
            & (visible.accepted_owner.astype(int) != dedicated_owner)
        ].sort_values("frame")
        run_groups: list[list] = []
        for row in track_rows.itertuples(index=False):
            if (run_groups
                    and int(row.frame) == int(run_groups[-1][-1].frame) + 1
                    and int(row.accepted_owner) == int(run_groups[-1][-1].accepted_owner)):
                run_groups[-1].append(row)
            else:
                run_groups.append([row])
        applied_frames = 0
        applied_runs = 0
        changed_pixels = 0
        for run_index, run_rows in enumerate(run_groups):
            prepared = []
            refusal = ""
            for row in run_rows:
                trial, unclaimed_trial, detail = _prepare_frame(
                    candidate, candidate_unclaimed, labels, unclaimed, raw,
                    by_frame, canonical, row, dedicated_owner, next_unclaimed,
                    params)
                detail.update({"track_id": track_id,
                               "dedicated_owner": dedicated_owner,
                               "run_index": run_index, "applied": False})
                if trial is None or unclaimed_trial is None:
                    refusal = str(detail["reason"])
                    frame_audits.append(detail)
                    break
                change = trial != candidate[int(row.frame)]
                if np.any(occupied[int(row.frame)] & change):
                    refusal = "overlapping_proposal"
                    frame_audits.append({**detail, "reason": refusal})
                    break
                prepared.append((row, trial, unclaimed_trial, detail))
            if refusal:
                for row, _, _, detail in prepared:
                    frame_audits.append({**detail, "reason": "run_atomic_refusal"})
                continue
            for row, trial, unclaimed_trial, detail in prepared:
                frame = int(row.frame)
                change = trial != candidate[frame]
                candidate[frame] = trial
                candidate_unclaimed[frame] = unclaimed_trial
                occupied[frame] |= change
                applied_frames += 1
                changed_pixels += int(np.count_nonzero(change))
                frame_audits.append({**detail, "applied": True,
                                     "reason": "applied"})
            if prepared:
                applied_runs += 1
        index = track_audit.index[
            track_audit.track_id.astype(int) == track_id]
        if len(index):
            track_audit.loc[index, "applied"] = applied_frames > 0
            track_audit.loc[index, "applied_runs"] = applied_runs
            track_audit.loc[index, "applied_frames"] = applied_frames
            track_audit.loc[index, "changed_pixels"] = changed_pixels
            track_audit.loc[index, "assigned_unclaimed_id"] = next_unclaimed
            track_audit.loc[index, "reason"] = (
                "applied" if applied_frames else "no_complete_relay_run_passed")
        if applied_frames:
            next_unclaimed += 1

    if not np.array_equal(
            (candidate > 0) | (candidate_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("exclusive-owner relay changed foreground-ledger union")
    if np.any((unclaimed > 0) & (candidate_unclaimed != unclaimed)):
        raise AssertionError("exclusive-owner relay changed existing unclaimed pixels")
    if set(map(int, np.unique(candidate))) != set(map(int, np.unique(labels))):
        raise AssertionError("exclusive-owner relay changed active identities")
    duplicates = _new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError(
            f"exclusive-owner relay created {duplicates} duplicate components")
    return (candidate, candidate_unclaimed, track_audit,
            pd.DataFrame(frame_audits))
