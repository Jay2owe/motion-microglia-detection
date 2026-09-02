"""Field-wide stationary identity reconciliation without supplied targets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import component_accounting

CONNECTIVITY = np.ones((3, 3), np.uint8)
DISALLOWED_PRODUCER_KEYS = {
    "case_id", "problem_frame", "problem_frames", "problem_coordinate",
    "problem_coordinates", "reference_frames", "source_identities",
    "target_identity", "target_identities",
}
PIXEL_COLUMNS = [
    "review_frame", "y", "x", "resident_identity", "observed_identity",
    "event_id", "raw_intensity", "distance_diameters", "area_similarity",
    "normalized_area_correlation", "match_direction",
]

DEFAULT_FIELD_WIDE_RECONCILIATION = {
    "enabled": True,
    "minimum_component_px": 4,
    "minimum_reference_duration_fraction": 0.02,
    "maximum_step_diameters": 0.5,
    "maximum_initial_gap_fraction": 0.20,
    "maximum_distance_diameters": 0.35,
    "minimum_area_similarity": 0.50,
    "minimum_normalized_area_correlation": 0.50,
    "minimum_raw_positive_fraction": 0.10,
    "minimum_assignment_margin": 0.15,
    "continuation_maximum_distance_diameters": 0.75,
    "continuation_minimum_area_similarity": 0.35,
    "continuation_minimum_normalized_area_correlation": 0.35,
    "continuation_minimum_raw_positive_fraction": 0.10,
    "continuation_minimum_assignment_margin": 0.10,
    "enable_continuation_hysteresis": True,
    "enable_dependency_transactions": True,
    "minimum_prior_incumbent_duration_ratio": 1.0,
    "enable_backward_weak_incumbent_reassignment": True,
    "enable_transactional_resident_recovery": True,
    "recovery_maximum_distance_diameters": 1.25,
    "recovery_minimum_area_similarity": 0.10,
    "recovery_minimum_normalized_area_correlation": 0.35,
    "recovery_minimum_raw_positive_fraction": 0.10,
    "recovery_minimum_assignment_margin": 0.10,
}


def reconciliation_params(overrides: dict | None = None) -> dict:
    """Return an independent default-on parameter set with optional overrides."""
    result = dict(DEFAULT_FIELD_WIDE_RECONCILIATION)
    result.update(overrides or {})
    return result


def align_stack(array: np.ndarray, target_frames: int,
                source_frame_offset: int) -> np.ndarray:
    """Align a raw stack to an identity stack after removed source frames."""
    if len(array) == target_frames:
        return array
    start = int(source_frame_offset)
    stop = start + int(target_frames)
    if start < 0 or stop > len(array):
        raise ValueError("source frame offset cannot align raw and identity stacks")
    return array[start:stop]


@dataclass(frozen=True)
class Component:
    frame: int
    identity: int
    index: int
    y: np.ndarray
    x: np.ndarray
    area: int
    centroid_y: float
    centroid_x: float
    diameter: float
    centred_points: frozenset[tuple[int, int]]

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray([self.centroid_y, self.centroid_x], float)

    @property
    def key(self) -> tuple[int, int, int]:
        return self.frame, self.identity, self.index


@dataclass
class Run:
    run_id: int
    identity: int
    members: list[Component]
    stationary: bool = False
    median_step_diameters: float = 0.0

    @property
    def start(self) -> int:
        return self.members[0].frame

    @property
    def end(self) -> int:
        return self.members[-1].frame

    @property
    def length(self) -> int:
        return len(self.members)

    @cached_property
    def centre(self) -> np.ndarray:
        values = np.asarray([part.centroid for part in self.members])
        return np.median(values, axis=0)

    @cached_property
    def area(self) -> float:
        return float(np.median([part.area for part in self.members]))

    @cached_property
    def diameter(self) -> float:
        return float(np.median([part.diameter for part in self.members]))

    @property
    def template(self) -> Component:
        return self.members[-1]


def _centred_points(y: np.ndarray, x: np.ndarray) -> frozenset[tuple[int, int]]:
    cy, cx = float(np.mean(y)), float(np.mean(x))
    yy = np.rint(y.astype(float) - cy).astype(int)
    xx = np.rint(x.astype(float) - cx).astype(int)
    return frozenset(zip(yy.tolist(), xx.tolist()))


def _make_component(frame: int, identity: int, index: int,
                    mask: np.ndarray) -> Component:
    y, x = np.nonzero(mask)
    return _make_component_from_coordinates(frame, identity, index, y, x)


def _make_component_from_coordinates(
        frame: int, identity: int, index: int,
        y: np.ndarray, x: np.ndarray) -> Component:
    """Build a component from row-major coordinates without a full-frame mask."""
    area = int(len(y))
    return Component(
        frame=frame, identity=identity, index=index, y=y, x=x, area=area,
        centroid_y=float(np.mean(y)), centroid_x=float(np.mean(x)),
        diameter=float(np.sqrt(4.0 * area / np.pi)),
        centred_points=_centred_points(y, x),
    )


def extract_components(labels: np.ndarray,
                       minimum_component_px: int) -> list[list[Component]]:
    """Extract every sufficiently large connected labelled component."""
    result: list[list[Component]] = []
    for frame, image in enumerate(labels):
        parts: list[Component] = []
        numbered, owners, areas = \
            component_accounting.label_value_components(image)
        component_slices = ndi.find_objects(numbered)
        for identity in sorted(set(map(int, owners)) - {0}):
            value_components = np.flatnonzero(owners == identity) + 1
            component_index = 0
            for value in value_components:
                if int(areas[int(value) - 1]) < minimum_component_px:
                    continue
                component_index += 1
                component_slice = component_slices[int(value) - 1]
                if component_slice is None:
                    raise AssertionError("labelled component has no bounding box")
                local_y, local_x = np.nonzero(
                    numbered[component_slice] == int(value))
                y = local_y + int(component_slice[0].start)
                x = local_x + int(component_slice[1].start)
                parts.append(_make_component_from_coordinates(
                    frame, identity, component_index, y, x))
        result.append(parts)
    return result


def area_similarity(first: Component, second: Component) -> float:
    return float(min(first.area, second.area) / max(first.area, second.area))


def normalized_area_correlation(first: Component, second: Component) -> float:
    """Cosine-normalised binary overlap after removing centroid translation."""
    denominator = float(np.sqrt(first.area * second.area))
    return (float(len(first.centred_points & second.centred_points)) /
            denominator if denominator else 0.0)


def _match_metrics(run: Run, component: Component) -> dict[str, float]:
    distance = float(np.linalg.norm(component.centroid - run.centre))
    distance_diameters = distance / max(run.diameter, 1.0)
    similarity = float(min(component.area, run.area) /
                       max(component.area, run.area))
    correlation = normalized_area_correlation(run.template, component)
    cost = distance_diameters + (1.0 - similarity) + (1.0 - correlation)
    return {
        "distance_px": distance,
        "distance_diameters": distance_diameters,
        "area_similarity": similarity,
        "normalized_area_correlation": correlation,
        "match_cost": cost,
    }


def _gate(params: dict, name: str, prefix: str = "") -> float:
    return float(params.get(f"{prefix}{name}", params[name]))


def _eligible(metrics: dict[str, float], component: Component,
              raw: np.ndarray, run: Run, params: dict,
              gate_prefix: str = "") -> tuple[bool, int]:
    raw_positive = int(np.count_nonzero(
        raw[component.frame, component.y, component.x] > 0))
    enough_raw = raw_positive >= int(np.ceil(
        _gate(params, "minimum_raw_positive_fraction", gate_prefix)
        * run.area))
    accepted = (
        metrics["distance_diameters"] <=
        _gate(params, "maximum_distance_diameters", gate_prefix)
        and metrics["area_similarity"] >=
        _gate(params, "minimum_area_similarity", gate_prefix)
        and metrics["normalized_area_correlation"] >=
        _gate(params, "minimum_normalized_area_correlation", gate_prefix)
        and enough_raw
    )
    return bool(accepted), raw_positive


def build_component_runs(components: list[list[Component]], params: dict,
                         minimum_frames: int) -> list[Run]:
    """Link same-owner components in consecutive frames and measure motion."""
    runs: list[Run] = []
    active: dict[int, list[Run]] = {}
    next_id = 1
    max_step = float(params["maximum_step_diameters"])
    min_area = float(params["minimum_area_similarity"])
    min_shape = float(params["minimum_normalized_area_correlation"])

    for frame_parts in components:
        by_identity: dict[int, list[Component]] = {}
        for part in frame_parts:
            by_identity.setdefault(part.identity, []).append(part)
        for identity, current in by_identity.items():
            previous = [run for run in active.get(identity, [])
                        if run.end == current[0].frame - 1]
            pairs: list[tuple[float, int, int]] = []
            for old_index, run in enumerate(previous):
                old = run.members[-1]
                for new_index, part in enumerate(current):
                    distance = float(np.linalg.norm(part.centroid-old.centroid))
                    distance_diameters = distance / max(
                        np.median([old.diameter, part.diameter]), 1.0)
                    similarity = area_similarity(old, part)
                    correlation = normalized_area_correlation(old, part)
                    if (distance_diameters <= max_step
                            and similarity >= min_area
                            and correlation >= min_shape):
                        cost = (distance_diameters + 1.0 - similarity
                                + 1.0 - correlation)
                        pairs.append((cost, old_index, new_index))
            used_old: set[int] = set()
            used_new: set[int] = set()
            for _, old_index, new_index in sorted(pairs):
                if old_index in used_old or new_index in used_new:
                    continue
                previous[old_index].members.append(current[new_index])
                used_old.add(old_index)
                used_new.add(new_index)
            for new_index, part in enumerate(current):
                if new_index in used_new:
                    continue
                run = Run(next_id, identity, [part])
                next_id += 1
                runs.append(run)
                active.setdefault(identity, []).append(run)

    for run in runs:
        steps = []
        for first, second in zip(run.members, run.members[1:]):
            distance = float(np.linalg.norm(second.centroid-first.centroid))
            diameter = max(np.median([first.diameter, second.diameter]), 1.0)
            steps.append(distance / diameter)
        run.median_step_diameters = float(np.median(steps)) if steps else 0.0
        run.stationary = bool(
            run.length >= minimum_frames
            and (not steps or float(np.quantile(steps, 0.90)) <= max_step))
    return runs


def _run_table(runs: list[Run], minimum_frames: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append({
            "run_id": run.run_id, "identity": run.identity,
            "start_frame": run.start + 1, "end_frame": run.end + 1,
            "observed_frames": run.length,
            "minimum_reference_frames": minimum_frames,
            "stationary_reference": run.stationary,
            "median_step_diameters": run.median_step_diameters,
            "median_area_px": run.area,
            "equivalent_diameter_px": run.diameter,
            "median_centroid_y": float(run.centre[0]),
            "median_centroid_x": float(run.centre[1]),
        })
    return pd.DataFrame(rows)


def _has_prior_incumbent(run: Run, stationary_runs: list[Run],
                         maximum_gap: int, params: dict) -> Run | None:
    candidates = []
    minimum_ratio = float(params.get(
        "minimum_prior_incumbent_duration_ratio", 0.0))
    for other in stationary_runs:
        if other.identity == run.identity or other.end >= run.start:
            continue
        if run.start - other.end > maximum_gap:
            continue
        if other.length < run.length * minimum_ratio:
            continue
        metrics = _match_metrics(run, other.template)
        if (metrics["distance_diameters"] <=
                float(params["maximum_distance_diameters"])
                and metrics["area_similarity"] >=
                float(params["minimum_area_similarity"])
                and metrics["normalized_area_correlation"] >=
                float(params["minimum_normalized_area_correlation"])):
            candidates.append((other.end, other.length, other))
    return max(candidates, default=(0, 0, None))[2]


def _target_elsewhere_components(run: Run, matched: list[Component],
                                 components: list[list[Component]],
                                 params: dict) -> list[Component]:
    matched_frames = {part.frame for part in matched}
    elsewhere: dict[tuple[int, int, int], Component] = {}
    for frame in matched_frames:
        for part in components[frame]:
            if part.identity != run.identity:
                continue
            # Any separately labelled resident component would make relabelling
            # another component to the same identity a duplicate. A dependency
            # transaction may still proceed after another accepted event frees it.
            elsewhere[part.key] = part
    return list(elsewhere.values())


def _matching_parts(run: Run, components: list[list[Component]], raw: np.ndarray,
                    start: int, end: int, params: dict,
                    identities: str = "all",
                    gate_prefix: str = "") -> list[tuple[Component, dict, int]]:
    matches = []
    for frame in range(max(start, 0), min(end + 1, len(components))):
        for part in components[frame]:
            if identities == "same" and part.identity != run.identity:
                continue
            if identities == "different" and part.identity == run.identity:
                continue
            metrics = _match_metrics(run, part)
            accepted, raw_positive = _eligible(
                metrics, part, raw, run, params, gate_prefix)
            if accepted:
                matches.append((part, metrics, raw_positive))
    return matches


def _backward_weak_incumbent_matches(
        run: Run, stationary_runs: list[Run],
        components: list[list[Component]], raw: np.ndarray,
        maximum_gap: int, params: dict,
        ) -> list[tuple[Component, dict, int]]:
    """Recover local fragments from incumbents weaker than the current run."""
    if not params.get("enable_backward_weak_incumbent_reassignment", False):
        return []
    minimum_ratio = float(params.get(
        "minimum_prior_incumbent_duration_ratio", 0.0))
    if minimum_ratio <= 0.0:
        return []
    weak = []
    for other in stationary_runs:
        if other.identity == run.identity or other.end >= run.start:
            continue
        if run.start - other.end > maximum_gap:
            continue
        if other.length >= run.length * minimum_ratio:
            continue
        metrics = _match_metrics(run, other.template)
        if (metrics["distance_diameters"] <=
                float(params["maximum_distance_diameters"])
                and metrics["area_similarity"] >=
                float(params["minimum_area_similarity"])
                and metrics["normalized_area_correlation"] >=
                float(params["minimum_normalized_area_correlation"])):
            weak.append(other)
    if not weak:
        return []
    weak_identities = {other.identity for other in weak}
    start = min(other.start for other in weak)
    matches = _matching_parts(
        run, components, raw, start, run.start - 1, params, "different")
    by_key = {
        part.key: (part, metrics, raw_positive)
        for part, metrics, raw_positive in matches
        if part.identity in weak_identities
    }
    return [by_key[key] for key in sorted(by_key)]


def _recover_displaced_resident_alternatives(
        proposals: list[dict], claimed_by: dict[tuple[int, int, int], str],
        components: list[list[Component]], raw: np.ndarray, params: dict,
        ) -> int:
    """Recover an alternate body only after the same-labelled body is claimed."""
    if not params.get("enable_transactional_resident_recovery", False):
        return 0
    recovered = 0
    for proposal in proposals:
        if not proposal["final_decision"].startswith(
                "accepted_field_wide_takeover"):
            continue
        run = proposal["run"]
        event_id = proposal["event_id"]
        occupied_frames = {
            part.frame for part, _, _ in proposal["matched"]
        }
        end = max(occupied_frames, default=run.end)
        for frame in range(run.end + 1, end + 1):
            if frame in occupied_frames:
                continue
            same = [part for part in components[frame]
                    if part.identity == run.identity]
            if not same or not all(
                    part.key in claimed_by
                    and claimed_by[part.key] != event_id for part in same):
                continue
            alternatives = sorted([
                item for item in _matching_parts(
                    run, components, raw, frame, frame, params,
                    "different", "recovery_")
                if item[0].key not in claimed_by
            ], key=lambda item: item[1]["match_cost"])
            if not alternatives:
                continue
            if (len(alternatives) > 1
                    and alternatives[1][1]["match_cost"] -
                    alternatives[0][1]["match_cost"] <
                    _gate(params, "minimum_assignment_margin", "recovery_")):
                continue
            chosen = alternatives[0]
            proposal["matched"].append(chosen)
            proposal["recovery_keys"].add(chosen[0].key)
            proposal["recovery_dependency_event_ids"].update(
                claimed_by[part.key] for part in same)
            claimed_by[chosen[0].key] = event_id
            occupied_frames.add(frame)
            recovered += 1
    return recovered


def discover_stationary_takeovers(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame,
                   pd.DataFrame]:
    """Find and safely apply field-wide stationary-owner takeovers."""
    minimum_component_px = int(params["minimum_component_px"])
    components = extract_components(labels, minimum_component_px)
    minimum_frames = max(2, int(np.ceil(
        float(params["minimum_reference_duration_fraction"]) * len(labels))))
    maximum_gap = max(1, int(np.ceil(
        float(params["maximum_initial_gap_fraction"]) * len(labels))))
    runs = build_component_runs(components, params, minimum_frames)
    stationary_runs = [run for run in runs if run.stationary]
    run_table = _run_table(runs, minimum_frames)

    proposals: list[dict] = []
    for run in stationary_runs:
        search_end = min(len(labels) - 1, run.end + maximum_gap)
        same = _matching_parts(
            run, components, raw, run.end + 1, search_end, params, "same")
        last_local_owner = max([run.end] + [part.frame for part, _, _ in same])
        later = _matching_parts(
            run, components, raw, last_local_owner + 1, search_end,
            params, "different")
        if not later:
            continue
        first_frame = min(part.frame for part, _, _ in later)
        first_matches = sorted(
            [item for item in later if item[0].frame == first_frame],
            key=lambda item: item[1]["match_cost"])
        best_part, best_metrics, best_raw = first_matches[0]
        margin = (first_matches[1][1]["match_cost"] -
                  best_metrics["match_cost"] if len(first_matches) > 1
                  else float("inf"))

        matched: list[tuple[Component, dict, int]] = []
        last_match = first_frame
        continuation_prefix = ("continuation_" if params.get(
            "enable_continuation_hysteresis", False) else "")
        for frame in range(first_frame, len(labels)):
            if frame - last_match > maximum_gap:
                break
            frame_matches = sorted(
                _matching_parts(run, components, raw, frame, frame, params,
                                "different", continuation_prefix),
                key=lambda item: item[1]["match_cost"])
            if not frame_matches:
                continue
            if (len(frame_matches) > 1
                    and frame_matches[1][1]["match_cost"] -
                    frame_matches[0][1]["match_cost"] <
                    _gate(params, "minimum_assignment_margin",
                          continuation_prefix)):
                continue
            matched.append(frame_matches[0])
            last_match = frame

        backward = _backward_weak_incumbent_matches(
            run, stationary_runs, components, raw, maximum_gap, params)
        backward_keys = {part.key for part, _, _ in backward}
        matched = backward + matched

        prior = _has_prior_incumbent(run, stationary_runs, maximum_gap, params)
        elsewhere: list[Component] = []
        if margin < float(params["minimum_assignment_margin"]):
            decision = "rejected_ambiguous_initial_match"
        elif prior is not None:
            decision = "rejected_prior_local_incumbent"
        else:
            elsewhere = _target_elsewhere_components(
                run, [part for part, _, _ in matched], components, params)
        if (margin >= float(params["minimum_assignment_margin"])
                and prior is None and elsewhere):
            decision = "rejected_resident_active_elsewhere"
        elif margin >= float(params["minimum_assignment_margin"]) and prior is None:
            decision = "eligible_field_wide_takeover"
        proposals.append({
            "run": run, "initial": best_part, "initial_metrics": best_metrics,
            "initial_raw": best_raw, "margin": margin, "matched": matched,
            "backward_keys": backward_keys,
            "prior": prior, "elsewhere": elsewhere, "decision": decision,
        })

    candidate = labels.copy()
    renamed = np.zeros(labels.shape, bool)
    claimed_by: dict[tuple[int, int, int], str] = {}
    event_rows: list[dict] = []
    pixel_rows: list[dict] = []
    ordered = sorted(proposals, key=lambda item: (
        -item["run"].length, item["initial"].frame,
        item["initial_metrics"]["match_cost"]))
    for event_index, proposal in enumerate(ordered, start=1):
        proposal["event_id"] = f"E{event_index:04d}"
        proposal["final_decision"] = proposal["decision"]
        proposal["transaction_pass"] = None
        proposal["recovery_keys"] = set()
        proposal["recovery_dependency_event_ids"] = set()

    def accept(proposal: dict, decision: str, transaction_pass: int) -> None:
        keys = {part.key for part, _, _ in proposal["matched"]}
        proposal["final_decision"] = decision
        proposal["transaction_pass"] = transaction_pass
        for key in keys:
            claimed_by[key] = proposal["event_id"]

    for proposal in ordered:
        keys = {part.key for part, _, _ in proposal["matched"]}
        if proposal["decision"] != "eligible_field_wide_takeover":
            continue
        if keys & set(claimed_by):
            proposal["final_decision"] = "rejected_competing_stationary_owner"
        else:
            accept(proposal, "accepted_field_wide_takeover", 0)

    if params.get("enable_dependency_transactions", False):
        transaction_pass = 0
        while True:
            transaction_pass += 1
            promoted = 0
            for proposal in ordered:
                if proposal["final_decision"] != (
                        "rejected_resident_active_elsewhere"):
                    continue
                elsewhere_keys = {part.key for part in proposal["elsewhere"]}
                if not elsewhere_keys or not elsewhere_keys.issubset(claimed_by):
                    continue
                keys = {part.key for part, _, _ in proposal["matched"]}
                if keys & set(claimed_by):
                    proposal["final_decision"] = (
                        "rejected_competing_stationary_owner")
                    continue
                accept(
                    proposal,
                    "accepted_field_wide_takeover_dependency_transaction",
                    transaction_pass)
                promoted += 1
            recovered = _recover_displaced_resident_alternatives(
                ordered, claimed_by, components, raw, params)
            if promoted == 0 and recovered == 0:
                break

    for proposal in ordered:
        run = proposal["run"]
        matched = proposal["matched"]
        decision = proposal["final_decision"]
        event_id = proposal["event_id"]
        if decision.startswith("accepted_field_wide_takeover"):
            for part, metrics, _ in matched:
                observed = part.identity
                candidate[part.frame, part.y, part.x] = run.identity
                renamed[part.frame, part.y, part.x] = True
                for y, x in zip(part.y, part.x):
                    pixel_rows.append({
                        "review_frame": part.frame + 1, "y": int(y),
                        "x": int(x), "resident_identity": run.identity,
                        "observed_identity": observed, "event_id": event_id,
                        "raw_intensity": float(raw[part.frame, y, x]),
                        "distance_diameters": metrics["distance_diameters"],
                        "area_similarity": metrics["area_similarity"],
                        "normalized_area_correlation": metrics[
                            "normalized_area_correlation"],
                        "match_direction": (
                            "backward_weak_incumbent" if part.key in
                            proposal["backward_keys"] else
                            "transactional_resident_recovery" if part.key in
                            proposal["recovery_keys"] else
                            "forward_takeover_continuation"),
                    })
        initial = proposal["initial"]
        metrics = proposal["initial_metrics"]
        prior = proposal["prior"]
        event_rows.append({
            "event_id": event_id, "reference_run_id": run.run_id,
            "resident_identity": run.identity,
            "observed_identity": initial.identity,
            "reference_start_frame": run.start + 1,
            "reference_end_frame": run.end + 1,
            "reference_observed_frames": run.length,
            "takeover_frame": initial.frame + 1,
            "last_matched_frame": (max(part.frame for part, _, _ in matched) + 1
                                   if matched else initial.frame + 1),
            "matched_components": len(matched),
            "backward_matched_components": len(proposal["backward_keys"]),
            "transactional_recovery_components": len(
                proposal["recovery_keys"]),
            "matched_observed_identities": "|".join(map(str, sorted({
                part.identity for part, _, _ in matched}))),
            "centroid_y": float(run.centre[0]),
            "centroid_x": float(run.centre[1]),
            "distance_px": metrics["distance_px"],
            "distance_diameters": metrics["distance_diameters"],
            "area_similarity": metrics["area_similarity"],
            "normalized_area_correlation": metrics[
                "normalized_area_correlation"],
            "raw_positive_px": proposal["initial_raw"],
            "assignment_margin": proposal["margin"],
            "prior_incumbent_run_id": (prior.run_id if prior else None),
            "prior_incumbent_identity": (prior.identity if prior else None),
            "distant_resident_components": len(proposal["elsewhere"]),
            "distant_resident_components_freed": sum(
                part.key in claimed_by for part in proposal["elsewhere"]),
            "dependency_event_ids": "|".join(sorted({
                claimed_by[part.key] for part in proposal["elsewhere"]
                if part.key in claimed_by})),
            "recovery_dependency_event_ids": "|".join(sorted(
                proposal["recovery_dependency_event_ids"])),
            "transaction_pass": proposal["transaction_pass"],
            "decision": decision,
        })

    events = pd.DataFrame(event_rows)
    evidence = pd.DataFrame(pixel_rows, columns=PIXEL_COLUMNS)
    return candidate, renamed, run_table, events, evidence


def _assert_identity_blind_params(params: dict) -> None:
    def visit(value, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in DISALLOWED_PRODUCER_KEYS:
                    raise ValueError(
                        f"field-wide producer received forbidden key: "
                        f"{'.'.join(path + (str(key),))}")
                visit(child, path + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (str(index),))
    visit(params)


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    _assert_identity_blind_params(params)
    paths = {key: Path(value) for key, value in params.get("paths", {}).items()}
    if upstream_dir is None:
        labels_path = paths["baseline_labels"]
        unclaimed_path = paths["baseline_unclaimed"]
    else:
        labels_path = upstream_dir / "95_A3.tif"
        unclaimed_path = upstream_dir / "95_A3_unclaimed_original_ids.tif"
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw_full = tifffile.imread(paths["registered_raw"])
    raw = align_stack(raw_full, len(labels), int(params["source_frame_offset"]))
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed labels and registered raw do not align")

    candidate, renamed, runs, events, evidence = (
        discover_stationary_takeovers(labels, raw, params))
    changed = candidate != labels
    if not np.array_equal(changed, renamed):
        raise AssertionError("candidate changes differ from the audited rename mask")
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("field-wide takeover discovery changed foreground")
    if np.any(changed & (labels == candidate)):
        raise AssertionError("recorded change retained the original identity")
    if len(evidence) != int(np.count_nonzero(changed)):
        raise AssertionError("takeover pixel evidence is not one-to-one")

    output_labels = out.out / "95_A3.tif"
    output_unclaimed = out.out / "95_A3_unclaimed_original_ids.tif"
    runs_path = out.out / "stationary_component_runs.csv"
    events_path = out.out / "stationary_takeover_events.csv"
    evidence_path = out.out / "stationary_takeover_pixel_evidence.csv"
    mask_path = out.mid / "95_A3_stationary_takeover_pixels_MEASUREMENT.tif"
    metrics_path = out.out / "metrics.json"
    tifffile.imwrite(output_labels, candidate, photometric="minisblack",
                     compression="zlib")
    shutil.copyfile(unclaimed_path, output_unclaimed)
    tifffile.imwrite(mask_path, renamed.astype(np.uint8),
                    photometric="minisblack", compression="zlib")
    runs.to_csv(runs_path, index=False)
    events.to_csv(events_path, index=False)
    evidence.to_csv(evidence_path, index=False)
    decisions = (events["decision"].value_counts().to_dict()
                 if len(events) else {})
    summary = {
        "attempt_id": str(params["attempt_id"]),
        "targeting_mode": "field_wide_discovery",
        "supplied_identity_count": 0,
        "supplied_problem_frame_count": 0,
        "minimum_reference_frames": max(2, int(np.ceil(
            float(params["minimum_reference_duration_fraction"])
            * len(labels)))),
        "stationary_runs": int(runs["stationary_reference"].sum()),
        "detected_events": int(len(events)),
        "accepted_events": int(events["decision"].astype(str).str.startswith(
            "accepted_field_wide_takeover").sum()) if len(events) else 0,
        "event_decisions": {str(key): int(value)
                            for key, value in decisions.items()},
        "renamed_pixels": int(np.count_nonzero(changed)),
        "renamed_frames": int(np.count_nonzero(
            np.any(changed, axis=(1, 2)))),
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": output_labels, "unclaimed": output_unclaimed,
        "stationary_runs": runs_path, "takeover_events": events_path,
        "pixel_evidence": evidence_path, "renamed_mask": mask_path,
        "metrics": metrics_path,
    }, "summary": summary}
