"""Discover raw-supported stationary gaps and seed ownership field-wide."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from stationary_reconciliation import (
    Component, Run, build_component_runs, extract_components)


CONNECTIVITY = np.ones((3, 3), np.uint8)


@dataclass
class GapEvent:
    event_id: str
    resident: int
    initial_successor: int
    reference_run: Run
    successor: Component
    history: list[Component]
    centre_y: float
    centre_x: float
    territory_radius: int
    support_radius: int
    recall_start: int
    recall_end: int
    seed_frames: list[int]
    challengers: set[int]


@dataclass
class StationaryGapResult:
    labels: np.ndarray
    unclaimed: np.ndarray
    recall_domain: np.ndarray
    reservation_domain: np.ndarray
    event_audit: pd.DataFrame
    actions: pd.DataFrame


def _disk(shape: tuple[int, int], centre_y: float, centre_x: float,
          radius: float) -> np.ndarray:
    y, x = np.ogrid[:shape[0], :shape[1]]
    return ((y - centre_y) ** 2 + (x - centre_x) ** 2
            <= float(radius) ** 2)


def _duration(fraction: float, frames: int, minimum: int = 1) -> int:
    return max(int(minimum), int(math.ceil(float(fraction) * frames)))


def _stable_suffix(run: Run, minimum_frames: int,
                   maximum_distance_diameters: float) -> list[Component]:
    history = list(run.members[-minimum_frames:])
    while len(run.members) > len(history):
        previous = run.members[-len(history) - 1]
        centre = np.median(np.asarray(
            [part.centroid for part in history]), axis=0)
        diameter = float(np.median([part.diameter for part in history]))
        distance = float(np.linalg.norm(previous.centroid - centre)) / max(
            diameter, 1.0)
        if distance > float(maximum_distance_diameters):
            break
        history.insert(0, previous)
    return history


def _nearest_successor(run: Run, components: list[list[Component]],
                       maximum_gap: int, maximum_distance: float
                       ) -> Component | None:
    for delta in range(1, maximum_gap + 1):
        frame = run.end + delta
        if frame >= len(components):
            return None
        candidates = []
        for part in components[frame]:
            if part.identity == run.identity:
                continue
            distance = float(np.linalg.norm(part.centroid - run.centre)) / max(
                run.diameter, 1.0)
            if distance <= maximum_distance:
                candidates.append((distance, part.identity, part.index, part))
        if candidates:
            return min(candidates, key=lambda row: row[:3])[3]
    return None


def discover_gap_events(labels: np.ndarray, raw: np.ndarray, params: dict
                        ) -> tuple[list[GapEvent], pd.DataFrame]:
    """Audit all stable-run transitions containing raw-supported blank frames."""
    if labels.shape != raw.shape:
        raise ValueError("labels and aligned raw stacks must have the same shape")
    frames = len(labels)
    minimum_frames = _duration(
        params["minimum_reference_duration_fraction"], frames, 2)
    maximum_gap = _duration(
        params["maximum_successor_gap_fraction"], frames)
    prior_lookback = _duration(
        params["prior_owner_lookback_fraction"], frames)
    maximum_seed = _duration(
        params["maximum_seed_run_duration_fraction"], frames, 2)
    bookend_frames = _duration(
        params["distant_run_bookend_fraction"], frames, 2)
    components = extract_components(
        labels, int(params["minimum_component_px"]))
    runs = build_component_runs(components, params, minimum_frames)
    run_for_component = {
        part.key: run for run in runs for part in run.members}
    field_diameter = float(np.median([
        part.diameter for frame in components for part in frame]))
    territory_radius = max(1, int(round(field_diameter * float(
        params["field_median_territory_radius_diameters"]))))
    support_radius = max(1, int(round(field_diameter * float(
        params["field_median_support_radius_diameters"]))))
    maximum_distance = float(params["maximum_successor_distance_diameters"])
    rows: list[dict] = []
    accepted: list[GapEvent] = []
    for run in runs:
        if not run.stationary or run.length < minimum_frames:
            continue
        successor = _nearest_successor(
            run, components, maximum_gap, maximum_distance)
        if successor is None:
            continue
        history = _stable_suffix(
            run, minimum_frames,
            float(params["stable_suffix_extension_distance_diameters"]))
        centre = np.median(np.asarray(
            [part.centroid for part in history]), axis=0)
        support = _disk(labels.shape[1:], centre[0], centre[1],
                        support_radius)
        supported_gaps = [
            frame for frame in range(run.end + 1, successor.frame)
            if not np.any(labels[frame][support] > 0)
            and float(np.median(raw[frame][support])) > 0]
        if not supported_gaps:
            continue
        prior_other = any(
            part.identity != run.identity
            and float(np.linalg.norm(part.centroid - run.centre)) / max(
                run.diameter, 1.0) <= maximum_distance
            for frame in range(max(0, run.start - prior_lookback), run.start)
            for part in components[frame])
        resident_elsewhere = any(
            part.identity == run.identity
            and float(np.linalg.norm(part.centroid - run.centre)) / max(
                run.diameter, 1.0) > maximum_distance
            for part in components[successor.frame])
        decision = "accepted_stationary_raw_gap_event"
        if prior_other:
            decision = "refused_recent_different_local_predecessor"
        elif resident_elsewhere:
            decision = "refused_resident_active_elsewhere_at_successor"
        base = {
            "reference_identity": run.identity,
            "reference_start_frame": run.start + 1,
            "reference_end_frame": run.end + 1,
            "reference_observed_frames": run.length,
            "stable_history_start_frame": history[0].frame + 1,
            "stable_history_end_frame": history[-1].frame + 1,
            "successor_identity": successor.identity,
            "successor_frame": successor.frame + 1,
            "raw_supported_gap_frames": "|".join(
                str(frame + 1) for frame in supported_gaps),
            "centre_y": float(centre[0]), "centre_x": float(centre[1]),
            "territory_radius_px": territory_radius,
            "support_radius_px": support_radius,
            "decision": decision,
        }
        if decision != "accepted_stationary_raw_gap_event":
            rows.append(base)
            continue

        territory = _disk(labels.shape[1:], centre[0], centre[1],
                          territory_radius)
        initial = successor.identity
        alternate_seen = False
        transaction_end = frames - 1
        for frame in range(successor.frame + 1, frames):
            values = {int(value) for value in np.unique(
                labels[frame][territory]) if int(value) > 0}
            if values - {initial, run.identity}:
                alternate_seen = True
            if alternate_seen and initial in values:
                transaction_end = frame - 1
                break
        recall_start = history[0].frame
        for frame in range(history[0].frame - 1,
                           max(-1, history[0].frame - bookend_frames - 1), -1):
            if np.any(labels[frame][support] > 0):
                break
            recall_start = frame
        challengers = {
            int(value) for frame in range(successor.frame, transaction_end + 1)
            for value in np.unique(labels[frame][territory])
            if int(value) > 0 and int(value) != run.identity}
        if alternate_seen:
            seed_frames = [
                frame for frame in range(successor.frame, transaction_end + 1)
                if np.any(np.isin(labels[frame][territory], list(challengers)))]
        else:
            successor_run = run_for_component[successor.key]
            seed_frames = list(range(
                successor.frame,
                min(successor_run.end, successor.frame + maximum_seed - 1) + 1))
            cursor = seed_frames[-1] + 1
            while (cursor <= transaction_end
                   and not np.any(labels[cursor][territory] > 0)):
                cursor += 1
            if cursor <= transaction_end and cursor > seed_frames[-1] + 1:
                local = [part for part in components[cursor]
                         if part.identity == initial
                         and float(np.linalg.norm(part.centroid - centre)) / max(
                             run.diameter, 1.0) <= maximum_distance]
                if local:
                    following_run = run_for_component[local[0].key]
                    if following_run.length <= maximum_seed:
                        seed_frames.extend(range(cursor, following_run.end + 1))
        distant = [
            other for other in runs
            if other.identity == run.identity
            and other.start > successor.frame and other.end < frames - 1
            and other.stationary and other.length >= run.length
            and float(np.linalg.norm(other.centre - centre)) / max(
                run.diameter, 1.0) > maximum_distance]
        if distant:
            displaced = max(distant, key=lambda item: (item.length, -item.start))
            seed_frames.extend(range(
                max(displaced.start, displaced.end - bookend_frames + 1),
                displaced.end + 1))
        event = GapEvent(
            event_id=f"SGR-{len(accepted) + 1:04d}",
            resident=run.identity, initial_successor=initial,
            reference_run=run, successor=successor, history=history,
            centre_y=float(centre[0]), centre_x=float(centre[1]),
            territory_radius=territory_radius, support_radius=support_radius,
            recall_start=recall_start, recall_end=transaction_end,
            seed_frames=sorted(set(seed_frames)), challengers=challengers)
        accepted.append(event)
        rows.append({
            **base, "event_id": event.event_id,
            "recall_start_frame": recall_start + 1,
            "recall_end_frame": transaction_end + 1,
            "seed_frames": "|".join(str(frame + 1)
                                      for frame in event.seed_frames),
            "challenger_count": len(challengers),
        })
    return accepted, pd.DataFrame(rows)


def _raw_recall_mask(raw_frame: np.ndarray, frame_labels: np.ndarray,
                     event: GapEvent, expected_area: int,
                     minimum_area: int) -> np.ndarray | None:
    territory = _disk(raw_frame.shape, event.centre_y, event.centre_x,
                      event.territory_radius)
    available = territory & (raw_frame > 0) & (frame_labels == 0)
    numbered, count = ndi.label(available, structure=CONNECTIVITY)
    choices = []
    for value in range(1, count + 1):
        mask = numbered == value
        y, x = np.nonzero(mask)
        if len(x) < minimum_area:
            continue
        distance = float(np.hypot(float(x.mean()) - event.centre_x,
                                  float(y.mean()) - event.centre_y))
        choices.append((distance, -len(x), mask))
    if not choices:
        return None
    component = min(choices, key=lambda row: (row[0], row[1]))[2]
    y_grid, x_grid = np.indices(raw_frame.shape)
    distance = np.hypot(x_grid - event.centre_x,
                        y_grid - event.centre_y)
    positive = raw_frame[component].astype(np.float64)
    scale = max(float(np.percentile(positive, 90)), 1.0)
    score = np.zeros(raw_frame.shape, np.float64)
    score[component] = (
        np.clip(raw_frame[component] / scale, 0.0, 1.5)
        + np.exp(-0.5 * (distance[component] / 3.0) ** 2))
    seed = np.unravel_index(int(np.argmax(
        np.where(component, score, -np.inf))), raw_frame.shape)
    chosen = np.zeros(raw_frame.shape, bool)
    visited = np.zeros(raw_frame.shape, bool)
    queue = [(-float(score[seed]), int(seed[0]), int(seed[1]))]
    visited[seed] = True
    height, width = raw_frame.shape
    while queue and int(chosen.sum()) < expected_area:
        _, row, column = heapq.heappop(queue)
        if not component[row, column]:
            continue
        chosen[row, column] = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = row + dy, column + dx
                if (0 <= ny < height and 0 <= nx < width
                        and component[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    heapq.heappush(queue, (
                        -float(score[ny, nx]), int(ny), int(nx)))
    return chosen if int(chosen.sum()) >= minimum_area else None


def _component_masks(frame: np.ndarray, identity: int,
                     minimum_area: int) -> list[np.ndarray]:
    numbered, count = ndi.label(frame == identity, structure=CONNECTIVITY)
    masks = [numbered == value for value in range(1, count + 1)
             if int(np.count_nonzero(numbered == value)) >= minimum_area]
    return sorted(masks, key=lambda mask: -int(mask.sum()))


def _safe_predecessor(candidate: np.ndarray, frame: int, mask: np.ndarray,
                      forbidden: set[int]) -> int:
    if frame <= 0:
        return 0
    expanded = ndi.binary_dilation(mask, structure=CONNECTIVITY, iterations=3)
    values, counts = np.unique(candidate[frame - 1][expanded],
                               return_counts=True)
    ranked = sorted([
        (int(count), int(value)) for value, count in zip(values, counts)
        if int(value) > 0 and int(value) not in forbidden], reverse=True)
    for _, identity in ranked:
        if not np.any(candidate[frame] == identity):
            return identity
    return 0


def recover_stationary_gaps(labels: np.ndarray, unclaimed: np.ndarray,
                            raw: np.ndarray, params: dict
                            ) -> StationaryGapResult:
    """Recover every accepted raw gap and seed its generic ownership event."""
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and aligned raw stacks must match")
    events, audit = discover_gap_events(labels, raw, params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    recall_domain = np.zeros(labels.shape, np.uint16)
    reservation_domain = np.zeros(labels.shape, np.uint16)
    actions: list[dict] = []
    minimum_area = int(params["minimum_component_px"])
    for event_number, event in enumerate(events, 1):
        expected_area = max(1, int(round(float(np.median([
            part.area for part in event.history])))))
        support = _disk(labels.shape[1:], event.centre_y, event.centre_x,
                        event.support_radius)
        for frame in range(event.recall_start, event.recall_end + 1):
            if (np.any((candidate[frame] == event.resident) & support)
                    or float(np.median(raw[frame][support])) <= 0
                    or np.any(candidate[frame][support] > 0)):
                continue
            mask = _raw_recall_mask(
                raw[frame], candidate[frame], event, expected_area,
                minimum_area)
            if mask is None:
                continue
            candidate[frame][mask] = event.resident
            candidate_unclaimed[frame][mask] = 0
            recall_domain[frame][mask] = event_number
            actions.append({
                "event_id": event.event_id, "mechanism": "raw_gap_recall",
                "review_frame": frame + 1,
                "resident_identity": event.resident,
                "observed_identity": 0,
                "changed_pixels": int(mask.sum()),
                "replacement_identity": event.resident,
            })
    for event_number, event in enumerate(events, 1):
        territory = _disk(labels.shape[1:], event.centre_y, event.centre_x,
                          event.territory_radius)
        for frame in event.seed_frames:
            for challenger in sorted(event.challengers):
                for mask in _component_masks(
                        candidate[frame], challenger, minimum_area):
                    if not np.any(mask & territory):
                        continue
                    candidate[frame][mask] = event.resident
                    candidate_unclaimed[frame][mask] = 0
                    reservation_domain[frame][mask] = event_number
                    actions.append({
                        "event_id": event.event_id,
                        "mechanism": "local_successor_seed",
                        "review_frame": frame + 1,
                        "resident_identity": event.resident,
                        "observed_identity": challenger,
                        "changed_pixels": int(mask.sum()),
                        "replacement_identity": event.resident,
                    })
            if np.any((candidate[frame] == event.resident) & territory):
                for mask in _component_masks(
                        candidate[frame], event.resident, minimum_area):
                    if np.any(mask & territory):
                        continue
                    replacement = _safe_predecessor(
                        candidate, frame, mask,
                        {event.resident, *event.challengers})
                    if replacement:
                        candidate[frame][mask] = replacement
                        candidate_unclaimed[frame][mask] = 0
                        mechanism = "distant_reuse_restored"
                    else:
                        candidate[frame][mask] = 0
                        candidate_unclaimed[frame][mask] = np.where(
                            labels[frame][mask] > 0,
                            labels[frame][mask], event.resident)
                        mechanism = "distant_reuse_released"
                    reservation_domain[frame][mask] = event_number
                    actions.append({
                        "event_id": event.event_id,
                        "mechanism": mechanism,
                        "review_frame": frame + 1,
                        "resident_identity": event.resident,
                        "observed_identity": event.resident,
                        "changed_pixels": int(mask.sum()),
                        "replacement_identity": replacement,
                    })
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed foreground overlap")
    before = (labels > 0) | (unclaimed > 0)
    after = (candidate > 0) | (candidate_unclaimed > 0)
    if np.any(before & ~after):
        raise AssertionError("accepted foreground was lost")
    added = after & ~before
    if np.any(added & (raw == 0)):
        raise AssertionError("raw gap recovery added zero-signal foreground")
    return StationaryGapResult(
        labels=candidate, unclaimed=candidate_unclaimed,
        recall_domain=recall_domain,
        reservation_domain=reservation_domain,
        event_audit=audit,
        actions=pd.DataFrame(actions, columns=[
            "event_id", "mechanism", "review_frame", "resident_identity",
            "observed_identity", "changed_pixels", "replacement_identity"]))
