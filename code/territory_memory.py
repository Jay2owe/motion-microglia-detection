"""Field-wide territory tenure and prior-conditioned raw-signal recall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stationary_reconciliation import (
    Component, Run, area_similarity, build_component_runs,
    extract_components, normalized_area_correlation)


@dataclass
class TerritoryCandidate:
    event_id: str
    run: Run
    resident_identity: int
    prior_components: list[Component]
    centre_x: float
    centre_y: float
    decision: str
    accepted: bool


@dataclass
class TerritoryMemoryResult:
    labels: np.ndarray
    unclaimed: np.ndarray
    raw_additions: np.ndarray
    run_audit: pd.DataFrame
    event_audit: pd.DataFrame
    actions: pd.DataFrame


def _disk(shape: tuple[int, int], x: float, y: float,
          radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2


def _centre_surround_peak(
        image: np.ndarray, anchor_x: float, anchor_y: float,
        search_radius: int, centre_radius: int,
        surround_inner: int, surround_outer: int,
        ) -> tuple[int, int, float]:
    best = (int(round(anchor_x)), int(round(anchor_y)), float("-inf"))
    height, width = image.shape
    ax, ay = int(round(anchor_x)), int(round(anchor_y))
    for y in range(max(0, ay - search_radius),
                   min(height, ay + search_radius + 1)):
        for x in range(max(0, ax - search_radius),
                       min(width, ax + search_radius + 1)):
            if ((x - anchor_x) ** 2 + (y - anchor_y) ** 2
                    > search_radius ** 2):
                continue
            centre = _disk(image.shape, x, y, centre_radius)
            outer = _disk(image.shape, x, y, surround_outer)
            inner = _disk(image.shape, x, y, surround_inner)
            surround = outer & ~inner
            score = float(image[centre].mean() - image[surround].mean())
            if score > best[2]:
                best = (x, y, score)
    return best


def _run_match(run: Run, component: Component) -> dict[str, float]:
    distance = float(np.linalg.norm(component.centroid - run.centre))
    distance_diameters = distance / max(run.diameter, 1.0)
    similarity = float(min(component.area, run.area) /
                       max(component.area, run.area))
    correlation = normalized_area_correlation(run.template, component)
    return {
        "distance_px": distance,
        "distance_diameters": distance_diameters,
        "area_similarity": similarity,
        "normalized_area_correlation": correlation,
    }


def discover_territory_candidates(
        labels: np.ndarray, params: dict,
        ) -> tuple[list[TerritoryCandidate], pd.DataFrame, pd.DataFrame]:
    """Discover long takeovers preceded by isolated observations of an owner."""
    minimum_component = int(params["minimum_component_px"])
    components = extract_components(labels, minimum_component)
    minimum_run = max(2, int(np.ceil(
        float(params["minimum_takeover_duration_fraction"]) * len(labels))))
    prior_window = max(1, int(np.ceil(
        float(params["prior_window_fraction"]) * len(labels))))
    run_params = {
        "maximum_step_diameters": float(params["maximum_step_diameters"]),
        "minimum_area_similarity": float(params["minimum_run_area_similarity"]),
        "minimum_normalized_area_correlation": float(
            params["minimum_run_shape_correlation"]),
    }
    runs = build_component_runs(components, run_params, minimum_run)
    run_rows: list[dict] = []
    candidates: list[TerritoryCandidate] = []
    event_rows: list[dict] = []
    next_event = 1
    for run in runs:
        if run.length < minimum_run or not run.stationary:
            continue
        by_identity: dict[int, list[tuple[Component, dict[str, float]]]] = {}
        for frame in range(max(0, run.start - prior_window), run.start):
            for component in components[frame]:
                if component.identity == run.identity:
                    continue
                metrics = _run_match(run, component)
                if (metrics["distance_diameters"] <= float(
                        params["maximum_prior_distance_diameters"])
                        and metrics["area_similarity"] >= float(
                            params["minimum_prior_area_similarity"])):
                    by_identity.setdefault(component.identity, []).append(
                        (component, metrics))
        eligible: list[tuple[int, list[Component], float]] = []
        rejected_reasons: list[str] = []
        for identity, matches in sorted(by_identity.items()):
            # Keep at most one component per frame, choosing the closest.
            per_frame: dict[int, tuple[Component, dict[str, float]]] = {}
            for component, metrics in matches:
                previous = per_frame.get(component.frame)
                if previous is None or metrics["distance_diameters"] < \
                        previous[1]["distance_diameters"]:
                    per_frame[component.frame] = (component, metrics)
            ordered = [per_frame[frame] for frame in sorted(per_frame)]
            frames = [component.frame for component, _ in ordered]
            if len(frames) < int(params["minimum_isolated_prior_observations"]):
                rejected_reasons.append(
                    f"identity_{identity}_insufficient_prior_observations")
                continue
            if (params.get("require_isolated_prior_observations", True)
                    and any(second - first <= 1
                            for first, second in zip(frames, frames[1:]))):
                rejected_reasons.append(
                    f"identity_{identity}_prior_observations_not_isolated")
                continue
            if run.start - frames[-1] > int(
                    params["maximum_prior_to_takeover_gap_frames"]):
                rejected_reasons.append(
                    f"identity_{identity}_prior_observation_too_old")
                continue
            median_distance = float(np.median([
                metrics["distance_diameters"] for _, metrics in ordered]))
            eligible.append((identity,
                             [component for component, _ in ordered],
                             median_distance))

        event_id = f"TM-{next_event:04d}"
        next_event += 1
        if not eligible:
            decision = ("refused_no_isolated_prior_owner" if not rejected_reasons
                        else ";".join(rejected_reasons))
            resident = 0
            prior: list[Component] = []
            centre_x, centre_y = float(run.centre[1]), float(run.centre[0])
            accepted = False
        else:
            eligible.sort(key=lambda item: (-len(item[1]), item[2], item[0]))
            best = eligible[0]
            if len(eligible) > 1 and (len(eligible[0][1]) == len(eligible[1][1])
                    and eligible[1][2] - eligible[0][2] < float(
                        params["minimum_prior_assignment_margin"])):
                decision = "refused_ambiguous_prior_owner"
                accepted = False
            else:
                decision = "eligible_isolated_prior_owner"
                accepted = True
            resident, prior, _ = best
            centre_x = float(np.median(
                [component.centroid_x for component in prior]))
            centre_y = float(np.median(
                [component.centroid_y for component in prior]))
        candidates.append(TerritoryCandidate(
            event_id=event_id, run=run, resident_identity=int(resident),
            prior_components=prior, centre_x=centre_x, centre_y=centre_y,
            decision=decision, accepted=accepted))
        run_rows.append({
            "event_id": event_id, "run_id": run.run_id,
            "observed_identity": run.identity,
            "run_start_review_frame": run.start + 1,
            "run_end_review_frame": run.end + 1,
            "run_observed_frames": run.length,
            "median_centroid_y": float(run.centre[0]),
            "median_centroid_x": float(run.centre[1]),
            "median_area_px": run.area, "decision": decision,
        })
        event_rows.append({
            "event_id": event_id, "run_id": run.run_id,
            "resident_identity": int(resident),
            "observed_identity": run.identity,
            "prior_observed_frames": len(prior),
            "prior_first_review_frame": (prior[0].frame + 1 if prior else 0),
            "prior_last_review_frame": (prior[-1].frame + 1 if prior else 0),
            "takeover_review_frame": run.start + 1,
            "last_takeover_review_frame": run.end + 1,
            "centre_y": centre_y, "centre_x": centre_x,
            "accepted": accepted, "decision": decision,
        })

    # Multiple independent events are valid. Resolve component conflicts only.
    claimed_runs: set[tuple[int, int, int]] = set()
    for candidate, row in zip(candidates, event_rows):
        if not candidate.accepted:
            continue
        keys = {component.key for component in candidate.run.members}
        if keys & claimed_runs:
            candidate.accepted = False
            candidate.decision = "refused_competing_takeover_event"
            row["accepted"] = False
            row["decision"] = candidate.decision
            for run_row in run_rows:
                if run_row["event_id"] == candidate.event_id:
                    run_row["decision"] = candidate.decision
                    break
        else:
            candidate.decision = "accepted_isolated_prior_owner"
            row["decision"] = candidate.decision
            for run_row in run_rows:
                if run_row["event_id"] == candidate.event_id:
                    run_row["decision"] = candidate.decision
                    break
            claimed_runs |= keys
    return (candidates, pd.DataFrame(run_rows), pd.DataFrame(event_rows,
            columns=["event_id", "run_id", "resident_identity",
                     "observed_identity", "prior_observed_frames",
                     "prior_first_review_frame", "prior_last_review_frame",
                     "takeover_review_frame", "last_takeover_review_frame",
                     "centre_y", "centre_x", "accepted", "decision"]))


def _raw_supported_mask(image: np.ndarray, centre_x: float, centre_y: float,
                        radius: float, target_pixels: int,
                        available: np.ndarray) -> np.ndarray:
    eligible = (_disk(image.shape, centre_x, centre_y, radius)
                & (image > 0) & available)
    indices = np.flatnonzero(eligible.ravel())
    if not len(indices):
        return np.zeros(image.shape, dtype=bool)
    values = image.ravel()[indices]
    order = np.lexsort((indices, -values))
    selected = indices[order[:min(int(target_pixels), len(indices))]]
    mask = np.zeros(image.shape, dtype=bool)
    mask.ravel()[selected] = True
    return mask


def reconcile_territory_memory(labels: np.ndarray, unclaimed: np.ndarray,
                               raw: np.ndarray, params: dict
                               ) -> TerritoryMemoryResult:
    """Apply every unambiguous field-discovered tenure and recall event."""
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed, and aligned raw stacks must match")
    candidates, run_audit, event_audit = discover_territory_candidates(
        labels, params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    additions = np.zeros(labels.shape, np.uint8)
    actions: list[dict] = []
    territory_radius = float(params["territory_radius_px"])
    for event in candidates:
        if not event.accepted:
            continue
        territory = _disk(labels.shape[1:], event.centre_x, event.centre_y,
                          territory_radius)
        prior_frames = sorted({component.frame
                               for component in event.prior_components})
        target_pixels = max(component.area
                            for component in event.prior_components)
        for frame in range(prior_frames[0] + 1, event.run.start):
            if np.any((candidate[frame] == event.resident_identity) & territory):
                continue
            outlier = ((candidate[frame] == event.resident_identity)
                       & ~territory)
            released = int(outlier.sum())
            if released:
                candidate[frame][outlier] = 0
                candidate_unclaimed[frame][outlier] = event.resident_identity
            x, y, score = _centre_surround_peak(
                raw[frame], event.centre_x, event.centre_y,
                int(params["raw_search_radius_px"]),
                int(params["raw_centre_radius_px"]),
                int(params["raw_surround_inner_px"]),
                int(params["raw_surround_outer_px"]))
            if score < float(params["minimum_centre_surround_score"]):
                actions.append({
                    "event_id": event.event_id, "review_frame": frame + 1,
                    "mechanism": "raw_recall_refused", "resident_identity":
                    event.resident_identity, "observed_identity": 0,
                    "renamed_pixels": 0, "added_pixels": 0,
                    "released_pixels": released,
                    "centre_surround_score": score,
                    "status": "refused_insufficient_raw_contrast"})
                continue
            available = ((candidate[frame] == 0)
                         & (candidate_unclaimed[frame] == 0))
            mask = _raw_supported_mask(
                raw[frame], x, y, float(params["recall_mask_radius_px"]),
                target_pixels, available)
            added = int(mask.sum())
            if added < max(int(params["minimum_recall_pixels"]),
                           int(np.floor(target_pixels * float(
                               params["minimum_recall_area_fraction"])))):
                actions.append({
                    "event_id": event.event_id, "review_frame": frame + 1,
                    "mechanism": "raw_recall_refused", "resident_identity":
                    event.resident_identity, "observed_identity": 0,
                    "renamed_pixels": 0, "added_pixels": 0,
                    "released_pixels": released,
                    "centre_surround_score": score,
                    "status": "refused_insufficient_raw_area"})
                continue
            candidate[frame][mask] = event.resident_identity
            additions[frame][mask] = 1
            actions.append({
                "event_id": event.event_id, "review_frame": frame + 1,
                "mechanism": "raw_signal_recall",
                "resident_identity": event.resident_identity,
                "observed_identity": 0, "renamed_pixels": 0,
                "added_pixels": added, "released_pixels": released,
                "centre_surround_score": score,
                "status": "raw_supported_observation_restored"})
        for component in event.run.members:
            frame = component.frame
            claimed = ((candidate[frame] == event.run.identity) & territory)
            outlier = ((candidate[frame] == event.resident_identity)
                       & ~territory)
            renamed = int(claimed.sum())
            released = int(outlier.sum())
            if renamed:
                candidate[frame][claimed] = event.resident_identity
            if released:
                candidate[frame][outlier] = 0
                candidate_unclaimed[frame][outlier] = event.resident_identity
            if renamed or released:
                actions.append({
                    "event_id": event.event_id, "review_frame": frame + 1,
                    "mechanism": "territory_tenure_memory",
                    "resident_identity": event.resident_identity,
                    "observed_identity": event.run.identity,
                    "renamed_pixels": renamed, "added_pixels": 0,
                    "released_pixels": released,
                    "centre_surround_score": np.nan,
                    "status": "territory_reclaimed"})
    baseline_union = (labels > 0) | (unclaimed > 0)
    result_union = (candidate > 0) | (candidate_unclaimed > 0)
    if np.any(baseline_union & ~result_union):
        raise AssertionError("territory reconciliation removed foreground")
    if not np.array_equal(result_union & ~baseline_union,
                          additions.astype(bool)):
        raise AssertionError("raw-addition audit differs from foreground expansion")
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    return TerritoryMemoryResult(
        labels=candidate, unclaimed=candidate_unclaimed,
        raw_additions=additions, run_audit=run_audit,
        event_audit=event_audit,
        actions=pd.DataFrame(actions, columns=[
            "event_id", "review_frame", "mechanism", "resident_identity",
            "observed_identity", "renamed_pixels", "added_pixels",
            "released_pixels", "centre_surround_score", "status"]))
