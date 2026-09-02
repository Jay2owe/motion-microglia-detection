"""Recover two-cell residency from stable splits near a recording boundary."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed


STRUCTURE = np.ones((3, 3), np.uint8)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    return float(xx.mean()), float(yy.mean())


def _radius(area: float) -> float:
    return float(np.sqrt(max(float(area), 1.0) / np.pi))


def _component_at(components: np.ndarray, x: float, y: float,
                  search_radius: int) -> int:
    height, width = components.shape
    cx, cy = int(round(x)), int(round(y))
    if 0 <= cy < height and 0 <= cx < width and components[cy, cx] > 0:
        return int(components[cy, cx])
    y0, y1 = max(0, cy - search_radius), min(height, cy + search_radius + 1)
    x0, x1 = max(0, cx - search_radius), min(width, cx + search_radius + 1)
    yy, xx = np.nonzero(components[y0:y1, x0:x1] > 0)
    if not len(xx):
        return 0
    distance = (xx + x0 - x) ** 2 + (yy + y0 - y) ** 2
    best = int(np.argmin(distance))
    return int(components[yy[best] + y0, xx[best] + x0])


def _marker(mask: np.ndarray, x: float, y: float) -> tuple[int, int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    index = int(np.argmin((xx - x) ** 2 + (yy - y) ** 2))
    return int(yy[index]), int(xx[index])


def _valley_ratio(frame: np.ndarray, left: tuple[float, float],
                  right: tuple[float, float]) -> float:
    distance = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    samples = max(3, int(np.ceil(distance)) + 1)
    values = []
    for weight in np.linspace(0.0, 1.0, samples):
        x = int(round((1.0 - weight) * left[0] + weight * right[0]))
        y = int(round((1.0 - weight) * left[1] + weight * right[1]))
        y = int(np.clip(y, 0, frame.shape[0] - 1))
        x = int(np.clip(x, 0, frame.shape[1] - 1))
        values.append(float(frame[y, x]))
    endpoint = max(min(values[0], values[-1]), 1.0)
    interior = values[1:-1] if len(values) > 2 else values
    return float(np.clip(min(interior) / endpoint, 0.0, 2.0))


def _median_area(labels: np.ndarray, identity: int, start: int) -> float:
    areas = np.count_nonzero(labels[start:] == int(identity), axis=(1, 2))
    positive = areas[areas > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def recover_recording_start_residents(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Split complete start-boundary merged runs supported by later residence."""
    candidate = labels.copy()
    frames = len(labels)
    active = sorted(set(map(int, np.unique(labels))) - {0})
    start_limit = max(1, int(np.ceil(
        frames * float(params.get("maximum_birth_frame_fraction", 0.08)))))
    minimum_persistence = float(
        params.get("minimum_post_birth_presence_fraction", 0.5))
    minimum_separation = float(
        params.get("minimum_center_separation_radii", 0.5))
    maximum_separation = float(
        params.get("maximum_center_separation_radii", 2.5))
    maximum_valley = float(params.get("maximum_raw_valley_ratio", 0.9))
    minimum_area_ratio = float(params.get("minimum_partition_area_ratio", 0.25))
    maximum_area_ratio = float(params.get("maximum_partition_area_ratio", 2.5))
    minimum_combined = float(
        params.get("minimum_merged_to_combined_area_ratio", 0.5))
    search_scale = float(params.get("maximum_seed_search_radii", 0.75))
    sigma_scale = float(params.get("watershed_sigma_radius_fraction", 0.2))
    maximum_fragment_ratio = float(
        params.get("maximum_coreless_fragment_ratio", 0.0))
    occupied = np.zeros_like(labels, dtype=bool)
    audits: list[dict] = []

    for newcomer in active:
        presence = np.any(labels == newcomer, axis=(1, 2))
        first = int(np.flatnonzero(presence)[0])
        base = {
            "newcomer_identity": newcomer, "birth_frame": first,
            "incumbent_identity": 0, "applied": False,
            "changed_pixels": 0, "changed_frames": 0, "reason": ""}
        if first == 0 or first > start_limit:
            audits.append({**base, "reason": "outside_start_relative_birth_band"})
            continue
        post_fraction = float(np.count_nonzero(presence[first:]) /
                              max(1, frames - first))
        if post_fraction < minimum_persistence:
            audits.append({**base, "post_birth_presence_fraction": post_fraction,
                           "reason": "insufficient_later_residence"})
            continue

        components, count = ndi.label(labels[first] == newcomer, STRUCTURE)
        if count != 1:
            audits.append({**base, "post_birth_presence_fraction": post_fraction,
                           "reason": "newcomer_birth_not_single_component"})
            continue
        newcomer_mask = components == 1
        newcomer_x, newcomer_y = _centroid(newcomer_mask)
        newcomer_area = _median_area(labels, newcomer, first)
        previous_components, _ = ndi.label(labels[first - 1] > 0, STRUCTURE)
        search = max(2, int(np.ceil(search_scale * _radius(newcomer_area))))
        previous_component = _component_at(
            previous_components, newcomer_x, newcomer_y, search)
        if previous_component <= 0:
            audits.append({**base, "post_birth_presence_fraction": post_fraction,
                           "reason": "no_previous_foreground_at_newcomer"})
            continue
        prior_values = labels[first - 1][previous_components == previous_component]
        identities = np.unique(prior_values[prior_values > 0])
        if len(identities) != 1:
            audits.append({**base, "post_birth_presence_fraction": post_fraction,
                           "reason": "previous_component_not_single_owner"})
            continue
        incumbent = int(identities[0])
        detail = {**base, "incumbent_identity": incumbent,
                  "post_birth_presence_fraction": post_fraction}
        if incumbent == newcomer or not np.any(labels[first] == incumbent):
            audits.append({**detail, "reason": "incumbent_missing_after_split"})
            continue
        incumbent_components, incumbent_count = ndi.label(
            labels[first] == incumbent, STRUCTURE)
        distances = []
        for component_id in range(1, incumbent_count + 1):
            mask = incumbent_components == component_id
            x, y = _centroid(mask)
            distances.append((float(np.hypot(x - newcomer_x, y - newcomer_y)),
                              component_id, x, y))
        _, _, incumbent_x, incumbent_y = min(distances)
        incumbent_area = _median_area(labels, incumbent, first)
        scale = _radius(newcomer_area) + _radius(incumbent_area)
        normalized_separation = float(np.hypot(
            incumbent_x - newcomer_x, incumbent_y - newcomer_y) / max(scale, 1.0))
        detail["normalized_center_separation"] = normalized_separation
        if not minimum_separation <= normalized_separation <= maximum_separation:
            audits.append({**detail, "reason": "split_centers_out_of_range"})
            continue

        frame_changes: list[tuple[int, np.ndarray, int, float]] = []
        refusal = ""
        for frame in range(first):
            owner_components, _ = ndi.label(labels[frame] == incumbent, STRUCTURE)
            search = max(2, int(np.ceil(search_scale * scale)))
            at_newcomer = _component_at(
                owner_components, newcomer_x, newcomer_y, search)
            at_incumbent = _component_at(
                owner_components, incumbent_x, incumbent_y, search)
            if at_newcomer <= 0 or at_newcomer != at_incumbent:
                refusal = "later_split_markers_not_in_one_prior_owner_component"
                break
            shared = owner_components == at_newcomer
            shared_area = int(np.count_nonzero(shared))
            if shared_area < minimum_combined * (newcomer_area + incumbent_area):
                refusal = "prior_component_too_small_for_two_residents"
                break
            lost_marker = _marker(shared, newcomer_x, newcomer_y)
            host_marker = _marker(shared, incumbent_x, incumbent_y)
            if lost_marker is None or host_marker is None or lost_marker == host_marker:
                refusal = "temporal_markers_unavailable"
                break
            valley = _valley_ratio(
                raw[frame], (newcomer_x, newcomer_y),
                (incumbent_x, incumbent_y))
            if valley > maximum_valley:
                refusal = "raw_valley_not_separable"
                break
            markers = np.zeros(labels.shape[1:], np.int16)
            markers[lost_marker] = 1
            markers[host_marker] = 2
            sigma = max(0.5, sigma_scale * min(
                _radius(newcomer_area), _radius(incumbent_area)))
            elevation = -ndi.gaussian_filter(raw[frame].astype(np.float32), sigma)
            partition = watershed(
                elevation, markers=markers, mask=shared,
                connectivity=STRUCTURE)
            change = partition == 1
            remaining = partition == 2
            host_parts, host_count = ndi.label(remaining, STRUCTURE)
            host_component = int(host_parts[host_marker])
            if host_component <= 0:
                refusal = "incumbent_marker_lost"
                break
            fragments = remaining & (host_parts != host_component)
            fragment_area = int(np.count_nonzero(fragments))
            if fragment_area:
                dilated = ndi.binary_dilation(change, structure=STRUCTURE)
                touches_newcomer = all(
                    bool(np.any(dilated & (host_parts == component_id)))
                    for component_id in range(1, host_count + 1)
                    if component_id != host_component)
                fragment_ratio = float(
                    fragment_area / max(1, int(np.count_nonzero(change))))
                if (not touches_newcomer or
                        fragment_ratio > maximum_fragment_ratio):
                    refusal = "coreless_incumbent_fragments_out_of_range"
                    break
                change |= fragments
                remaining &= ~fragments
            if (ndi.label(change, STRUCTURE)[1] != 1 or
                    ndi.label(remaining, STRUCTURE)[1] != 1):
                refusal = "partition_not_two_connected_residents"
                break
            new_ratio = float(np.count_nonzero(change) / max(newcomer_area, 1.0))
            incumbent_ratio = float(
                np.count_nonzero(remaining) / max(incumbent_area, 1.0))
            if not (minimum_area_ratio <= new_ratio <= maximum_area_ratio and
                    minimum_area_ratio <= incumbent_ratio <= maximum_area_ratio):
                refusal = "partition_area_out_of_range"
                break
            if np.any(occupied[frame] & change):
                refusal = "proposal_overlaps_earlier_proposal"
                break
            frame_changes.append(
                (frame, change, int(np.count_nonzero(change)), valley))
        if refusal or len(frame_changes) != first:
            audits.append({**detail, "reason": refusal or
                           "merged_run_does_not_reach_recording_boundary"})
            continue
        for frame, change, _, _ in frame_changes:
            candidate[frame][change] = newcomer
            occupied[frame] |= change
        audits.append({
            **detail, "applied": True,
            "changed_pixels": int(sum(item[2] for item in frame_changes)),
            "changed_frames": int(len(frame_changes)),
            "maximum_raw_valley_ratio": float(max(item[3] for item in frame_changes)),
            "reason": "applied"})
    return candidate, pd.DataFrame(audits)
