"""Discover and repair transient local lineage ownership switches."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as connected_regions
from skimage.measure import regionprops

from tracking import motion_support


CONNECTIVITY = np.ones((3, 3), bool)


@dataclass(frozen=True)
class Component:
    number: int
    identity: int
    area: int
    centre_y: float
    centre_x: float


@dataclass
class FrameComponents:
    numbers: np.ndarray
    components: list[Component]

    def mask(self, component: Component) -> np.ndarray:
        return self.numbers == int(component.number)


@dataclass
class LocalLineageResult:
    labels: np.ndarray
    unclaimed: np.ndarray
    seed_domains: np.ndarray
    trace_domains: np.ndarray
    repair_domains: np.ndarray
    duplicate_audit: pd.DataFrame
    predecessor_audit: pd.DataFrame
    events: pd.DataFrame
    trace_frames: pd.DataFrame
    boundaries: pd.DataFrame
    actions: pd.DataFrame


def _frame_components(frame: np.ndarray) -> FrameComponents:
    numbers = connected_regions(frame, background=0, connectivity=2)
    components: list[Component] = []
    for region in regionprops(numbers):
        y, x = map(float, region.centroid)
        first_y, first_x = map(int, region.coords[0])
        identity = int(frame[first_y, first_x])
        if identity <= 0:
            continue
        components.append(Component(
            number=int(region.label), identity=identity,
            area=int(region.area), centre_y=y, centre_x=x))
    return FrameComponents(numbers=numbers, components=components)


def _centre(mask: np.ndarray) -> np.ndarray:
    y, x = ndi.center_of_mass(mask)
    return np.array([float(y), float(x)])


def _largest_core(mask: np.ndarray) -> int:
    y, x = np.nonzero(mask)
    if not len(y):
        return 0
    local = mask[y.min():y.max() + 1, x.min():x.max() + 1]
    eroded = ndi.binary_erosion(local, structure=CONNECTIVITY)
    regions, count = ndi.label(eroded, structure=CONNECTIVITY)
    if not count:
        return 0
    return int(np.max(np.bincount(regions.ravel())[1:]))


def _substantial_core_count(mask: np.ndarray, minimum_px: int) -> int:
    y, x = np.nonzero(mask)
    if not len(y):
        return 0
    local = mask[y.min():y.max() + 1, x.min():x.max() + 1]
    eroded = ndi.binary_erosion(local, structure=CONNECTIVITY)
    regions, count = ndi.label(eroded, structure=CONNECTIVITY)
    if not count:
        return 0
    return int(np.count_nonzero(
        np.bincount(regions.ravel())[1:] >= int(minimum_px)))


def _distance(first: Component, second: Component) -> float:
    return float(np.hypot(first.centre_y - second.centre_y,
                          first.centre_x - second.centre_x))


def _rank_predecessors(
        current: Component, current_frame: FrameComponents,
        previous_frame: FrameComponents, lag_frame: np.ndarray,
        maximum_distance: float, params: dict) -> list[dict]:
    current_mask = current_frame.mask(current)
    rows: list[dict] = []
    for previous in previous_frame.components:
        if previous.area < int(params["minimum_component_px"]):
            continue
        distance = _distance(current, previous)
        ratio = current.area / max(previous.area, 1)
        if (distance > maximum_distance
                or ratio < float(params["minimum_area_ratio"])
                or ratio > float(params["maximum_area_ratio"])):
            continue
        evidence = motion_support(
            previous_frame.mask(previous), current_mask, lag_frame)
        rows.append({
            "predecessor_identity": int(previous.identity),
            "predecessor_component": int(previous.number),
            "predecessor_area_px": int(previous.area),
            "distance_px": distance,
            **evidence,
        })
    return sorted(rows, key=lambda row: (
        -float(row["motion_support"]), -int(row["supportive_px"]),
        float(row["distance_px"]), int(row["predecessor_identity"]),
        int(row["predecessor_component"])))


def _winner(ranked: list[dict], params: dict) -> dict | None:
    if not ranked:
        return None
    best = ranked[0]
    second_support = (float(ranked[1]["motion_support"])
                      if len(ranked) > 1 else -np.inf)
    if (float(best["motion_support"])
            < float(params["minimum_predecessor_motion_support"])
            or int(best["supportive_px"])
            < int(params["minimum_motion_pixels"])
            or float(best["motion_support"]) - second_support
            < float(params["minimum_predecessor_margin"])):
        return None
    return best


def _owner_body(frame: np.ndarray, owner: int, reference: np.ndarray,
                maximum_distance: float, params: dict
                ) -> tuple[np.ndarray | None, dict]:
    pieces = _frame_components(frame)
    reference_area = int(reference.sum())
    reference_centre = _centre(reference)
    choices: list[tuple[float, float, Component]] = []
    for component in pieces.components:
        if (component.identity != int(owner)
                or component.area < int(params["minimum_component_px"])):
            continue
        distance = float(np.linalg.norm(
            np.array([component.centre_y, component.centre_x])
            - reference_centre))
        ratio = component.area / max(reference_area, 1)
        if (distance <= maximum_distance
                and float(params["minimum_area_ratio"]) <= ratio
                <= float(params["maximum_area_ratio"])):
            choices.append((distance, abs(float(np.log(ratio))), component))
    if not choices:
        return None, {"status": "no_owner_component"}
    _, _, base = min(choices, key=lambda row: (
        row[0], row[1], row[2].number))
    body = pieces.mask(base).copy()
    used = {int(base.number)}
    added: list[Component] = []
    minimum_core = int(params["minimum_substantial_core_px"])
    while True:
        old_cost = abs(float(np.log(body.sum() / max(reference_area, 1))))
        candidates: list[tuple[float, int, int, Component, np.ndarray]] = []
        border = ndi.binary_dilation(body, structure=CONNECTIVITY)
        added_identities = {item.identity for item in added}
        for component in pieces.components:
            if (component.number in used or component.identity == int(owner)
                    or (component.area < int(params["minimum_fragment_px"])
                        and component.identity not in added_identities)):
                continue
            fragment = pieces.mask(component)
            if not np.any(border & fragment):
                continue
            union = body | fragment
            ratio = float(union.sum() / max(reference_area, 1))
            new_cost = abs(float(np.log(ratio)))
            if (ratio < float(params["minimum_expanded_area_ratio"])
                    or ratio > float(params["maximum_expanded_area_ratio"])
                    or new_cost >= old_cost
                    - float(params["minimum_area_continuity_improvement"])
                    or _substantial_core_count(union, minimum_core) != 1
                    or min(_largest_core(body), _largest_core(fragment))
                    >= minimum_core):
                continue
            candidates.append((new_cost, component.identity,
                               component.number, component, union))
        if not candidates:
            break
        _, _, _, component, body = min(candidates, key=lambda row: row[:3])
        used.add(int(component.number))
        added.append(component)
    return body, {
        "status": "owner_body_found",
        "base_area_px": int(base.area),
        "body_area_px": int(body.sum()),
        "added_fragment_count": int(len(added)),
        "added_fragment_identities": ";".join(
            map(str, sorted({item.identity for item in added}))),
    }


def discover_duplicate_handoffs(labels: np.ndarray, lag: np.ndarray,
                                params: dict
                                ) -> tuple[list[dict], pd.DataFrame,
                                           pd.DataFrame, np.ndarray]:
    """Audit every duplicate name and return transient returning-owner seeds."""
    if labels.ndim != 3 or lag.ndim != 3:
        raise ValueError("labels and lag must be TYX stacks")
    if labels.shape[1:] != lag.shape[1:] or len(lag) < len(labels) - 1:
        raise ValueError("lag does not cover retained label transitions")
    maximum_distance = float(
        params["maximum_link_fraction_of_diagonal"]) * float(np.hypot(
            labels.shape[1], labels.shape[2]))
    duplicate_rows: list[dict] = []
    predecessor_rows: list[dict] = []
    events: list[dict] = []
    seed_domains = np.zeros(labels.shape, np.uint16)
    previous = _frame_components(labels[0])
    for frame in range(1, len(labels)):
        current = _frame_components(labels[frame])
        by_identity: dict[int, list[Component]] = {}
        for component in current.components:
            if component.area >= int(params["minimum_component_px"]):
                by_identity.setdefault(component.identity, []).append(component)
        for duplicated, components in sorted(by_identity.items()):
            if len(components) < 2:
                continue
            winners: list[tuple[Component, dict]] = []
            component_decisions: list[str] = []
            for component in sorted(components, key=lambda item: item.number):
                ranked = _rank_predecessors(
                    component, current, previous, lag[frame - 1],
                    maximum_distance, params)
                winner = _winner(ranked, params)
                if winner is not None:
                    winners.append((component, winner))
                    component_decisions.append(
                        f"{component.number}:{winner['predecessor_identity']}")
                else:
                    component_decisions.append(f"{component.number}:none")
                predecessor_rows.append({
                    "review_frame": frame + 1,
                    "duplicated_identity": duplicated,
                    "component_number": component.number,
                    "component_area_px": component.area,
                    "centre_y": component.centre_y,
                    "centre_x": component.centre_x,
                    "winner_identity": (0 if winner is None else int(
                        winner["predecessor_identity"])),
                    "winner_motion_support": (np.nan if winner is None else
                        float(winner["motion_support"])),
                    "winner_distance_px": (np.nan if winner is None else
                        float(winner["distance_px"])),
                    "ranked_predecessors": json.dumps(ranked, sort_keys=True),
                })
            incumbent = [item for item in winners
                         if int(item[1]["predecessor_identity"]) == duplicated]
            visitors = [item for item in winners
                        if int(item[1]["predecessor_identity"]) != duplicated]
            eligible_visitors = [item for item in visitors
                                 if not np.any(labels[frame] == int(
                                     item[1]["predecessor_identity"]))]
            event = None
            if not incumbent:
                decision = "refused_no_incumbent_continuation"
            elif not visitors:
                decision = "refused_no_distinct_predecessor"
            elif not eligible_visitors:
                decision = "refused_predecessor_still_present"
            elif len(eligible_visitors) != 1:
                decision = "refused_ambiguous_absent_predecessors"
            elif frame + 1 >= len(labels):
                decision = "refused_no_successor_frame"
            else:
                visitor, evidence = eligible_visitors[0]
                owner = int(evidence["predecessor_identity"])
                seed = current.mask(visitor)
                returned, return_details = _owner_body(
                    labels[frame + 1], owner, seed, maximum_distance, params)
                if returned is None:
                    decision = "refused_predecessor_did_not_return"
                else:
                    return_evidence = motion_support(
                        seed, returned, lag[frame])
                    if (float(return_evidence["motion_support"])
                            < float(params["minimum_return_motion_support"])
                            or int(return_evidence["supportive_px"])
                            < int(params["minimum_motion_pixels"])):
                        decision = "refused_return_lacks_motion_support"
                    else:
                        decision = "accepted_transient_duplicate_handoff"
                        number = len(events) + 1
                        seed_domains[frame][seed] = number
                        event = {
                            "event_number": number,
                            "event_id": f"LDH-{number:04d}",
                            "review_frame": frame + 1,
                            "source_frame": frame + 1 + int(
                                params["source_frame_offset"]),
                            "duplicated_identity": duplicated,
                            "owner": owner,
                            "seed_area_px": int(seed.sum()),
                            "predecessor_motion_support": float(
                                evidence["motion_support"]),
                            "return_motion_support": float(
                                return_evidence["motion_support"]),
                            "return_details": json.dumps(
                                return_details, sort_keys=True),
                        }
                        events.append({**event, "seed": seed})
            duplicate_rows.append({
                "review_frame": frame + 1,
                "source_frame": frame + 1 + int(
                    params["source_frame_offset"]),
                "duplicated_identity": duplicated,
                "substantial_components": len(components),
                "component_predecessor_winners": ";".join(
                    component_decisions),
                "distinct_predecessor_identities": ";".join(map(str, sorted({
                    int(item[1]["predecessor_identity"])
                    for item in winners}))),
                "accepted_event_id": "" if event is None else event["event_id"],
                "decision": decision,
            })
        previous = current
    return (events, pd.DataFrame(duplicate_rows),
            pd.DataFrame(predecessor_rows), seed_domains)


def _trace_events(labels: np.ndarray, lag: np.ndarray, events: list[dict],
                  params: dict
                  ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    domains = np.zeros(labels.shape, np.uint16)
    frame_rows: list[dict] = []
    boundary_rows: list[dict] = []
    maximum_distance = float(
        params["maximum_link_fraction_of_diagonal"]) * float(np.hypot(
            labels.shape[1], labels.shape[2]))
    for event in events:
        number = int(event["event_number"])
        owner = int(event["owner"])
        start = int(event["review_frame"]) - 1
        seed = event["seed"]
        traced: dict[int, tuple[np.ndarray, dict, dict]] = {
            start: (seed, {"status": "seed_duplicate_component"}, {
                "motion_support": np.nan, "supportive_px": 0,
                "distance_px": np.nan})}
        reference = seed
        for frame in range(start - 1, -1, -1):
            body, details = _owner_body(
                labels[frame], owner, reference, maximum_distance, params)
            if body is None:
                boundary_rows.append({
                    "event_id": event["event_id"], "direction": "backward",
                    "boundary_review_frame": frame + 1,
                    "reason": str(details["status"])})
                break
            evidence = motion_support(body, reference, lag[frame])
            if float(evidence["motion_support"]) < float(
                    params["minimum_trace_motion_support"]):
                boundary_rows.append({
                    "event_id": event["event_id"], "direction": "backward",
                    "boundary_review_frame": frame + 1,
                    "reason": "insufficient_motion_support"})
                break
            traced[frame] = (body, details, evidence)
            reference = body
        reference = seed
        for frame in range(start + 1, len(labels)):
            body, details = _owner_body(
                labels[frame], owner, reference, maximum_distance, params)
            if body is None:
                boundary_rows.append({
                    "event_id": event["event_id"], "direction": "forward",
                    "boundary_review_frame": frame + 1,
                    "reason": str(details["status"])})
                break
            evidence = motion_support(reference, body, lag[frame - 1])
            if float(evidence["motion_support"]) < float(
                    params["minimum_trace_motion_support"]):
                boundary_rows.append({
                    "event_id": event["event_id"], "direction": "forward",
                    "boundary_review_frame": frame + 1,
                    "reason": "insufficient_motion_support"})
                break
            traced[frame] = (body, details, evidence)
            reference = body
        for frame, (body, details, evidence) in sorted(traced.items()):
            occupied = domains[frame][body]
            if np.any(occupied) and not np.all(occupied == number):
                raise AssertionError("independent local-lineage events overlap")
            domains[frame][body] = number
            frame_rows.append({
                "event_id": event["event_id"], "event_number": number,
                "owner": owner, "review_frame": frame + 1,
                "source_frame": frame + 1 + int(params["source_frame_offset"]),
                "body_area_px": int(body.sum()),
                "owner_pixels": int(np.count_nonzero(
                    labels[frame][body] == owner)),
                "nonowner_pixels": int(np.count_nonzero(
                    labels[frame][body] != owner)),
                "motion_support": float(evidence["motion_support"]),
                "supportive_px": int(evidence["supportive_px"]),
                "body_details": json.dumps(details, sort_keys=True),
            })
    frame_columns = [
        "event_id", "event_number", "owner", "review_frame",
        "source_frame", "body_area_px", "owner_pixels", "nonowner_pixels",
        "motion_support", "supportive_px", "body_details"]
    return (domains, pd.DataFrame(frame_rows, columns=frame_columns),
            pd.DataFrame(boundary_rows, columns=[
                "event_id", "direction", "boundary_review_frame", "reason"]))


def reserve_local_lineages(labels: np.ndarray, unclaimed: np.ndarray,
                           lag: np.ndarray, params: dict
                           ) -> LocalLineageResult:
    """Detect transient duplicate handoffs and reserve complete local bodies."""
    if labels.shape != unclaimed.shape:
        raise ValueError("labels and unclaimed stacks must have the same shape")
    events, duplicate_audit, predecessor_audit, seeds = (
        discover_duplicate_handoffs(labels, lag, params))
    traces, trace_frames, boundaries = _trace_events(
        labels, lag, events, params)
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    repairs = np.zeros(labels.shape, np.uint16)
    action_rows: list[dict] = []
    event_table = pd.DataFrame([{key: value for key, value in event.items()
                                if key != "seed"} for event in events])
    for row in trace_frames.sort_values(
            ["event_number", "review_frame"]).itertuples(index=False):
        frame = int(row.review_frame) - 1
        mask = traces[frame] == int(row.event_number)
        owner = int(row.owner)
        if not np.any(labels[frame][mask] != owner):
            continue
        repairs[frame][mask] = int(row.event_number)
        before = candidate[frame][mask].copy()
        candidate[frame][mask] = owner
        candidate_unclaimed[frame][mask] = 0
        action_rows.append({
            "event_id": row.event_id, "review_frame": int(row.review_frame),
            "source_frame": int(row.source_frame), "owner": owner,
            "domain_area_px": int(mask.sum()),
            "changed_pixels": int(np.count_nonzero(before != owner)),
        })
    if not np.array_equal(
            (candidate > 0) | (candidate_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("local-lineage reservation changed foreground")
    if np.any((candidate > 0) & (candidate_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    return LocalLineageResult(
        labels=candidate, unclaimed=candidate_unclaimed,
        seed_domains=seeds, trace_domains=traces, repair_domains=repairs,
        duplicate_audit=duplicate_audit,
        predecessor_audit=predecessor_audit, events=event_table,
        trace_frames=trace_frames, boundaries=boundaries,
        actions=pd.DataFrame(action_rows, columns=[
            "event_id", "review_frame", "source_frame", "owner",
            "domain_area_px", "changed_pixels"]))
