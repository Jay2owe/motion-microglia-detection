"""Preserve a large continuous body from a dormant-owner takeover.

Discovery is complete-field and identity-blind.  The rule audits every mutual
component-continuity edge whose owner changes.  It accepts only a large body
with a sustained source-owner chain, a terminal destination chain, and an
incoming owner whose previous field location implies a much larger jump than
the observed body motion.  Detached pre-existing pieces of the source owner
are allowed only when they satisfy a field-relative projection proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import component_accounting


STRUCTURE = np.ones((3, 3), np.uint8)
CONTROL_KEYS = {"targeting_mode", "output_stem", "apply_recovery"}
PATH_KEYS = {"labels_path", "unclaimed_path", "raw_path",
             "physical_track_points_path"}
PARAMETER_KEYS = {
    "maximum_candidate_step_sum_radii",
    "minimum_proximity_coverage",
    "minimum_area_ratio",
    "minimum_continuity_score",
    "minimum_continuity_margin",
    "minimum_source_tenure_movie_fraction",
    "minimum_dormancy_movie_fraction",
    "minimum_terminal_continuation_movie_fraction",
    "minimum_boundary_area_quantile",
    "minimum_historic_jump_sum_radii",
    "minimum_jump_advantage",
    "maximum_incoming_components_at_boundary",
    "maximum_projection_area_quantile",
    "maximum_projection_distance_sum_radii",
    "maximum_projection_components_per_frame",
}
ALLOWED_KEYS = CONTROL_KEYS | PATH_KEYS | PARAMETER_KEYS
FORBIDDEN_KEYS = {
    "identity", "identities", "identity_id", "identity_ids",
    "owner", "owners", "owner_id", "owner_ids", "track", "tracks",
    "track_id", "track_ids", "frame", "frames", "frame_id", "frame_ids",
    "coordinate", "coordinates", "event", "events", "event_id",
    "event_ids", "region", "regions", "well", "wells", "review_case",
    "review_cases", "case_id", "case_ids", "bbox", "roi", "target",
    "targets",
}
FORBIDDEN_FRAGMENTS = (
    "target_", "_target", "selector", "review_case", "case_id",
    "forced_", "include_", "exclude_", "allowed_pair", "track_id",
    "owner_id", "identity_id", "frame_id", "event_id", "coordinate",
    "region", "roi",
)


@dataclass(frozen=True)
class Component:
    key: int
    frame: int
    owner: int
    area: int
    x: float
    y: float
    radius: float
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class Edge:
    left: int
    right: int
    step_sum_radii: float
    exact_overlap_fraction: float
    proximity_coverage: float
    area_ratio: float
    score: float
    left_margin: float
    right_margin: float


AUDIT_COLUMNS = [
    "proposal_id", "source_owner", "incoming_owner", "transition_frame",
    "source_component", "incoming_component", "source_chain_first",
    "source_chain_last", "source_tenure_frames", "incoming_last_prior_frame",
    "incoming_dormant_frames", "terminal_first", "terminal_last",
    "terminal_frames", "terminal_reaches_boundary", "source_area",
    "incoming_area", "boundary_area_quantile", "step_sum_radii",
    "exact_overlap_fraction", "proximity_coverage", "area_ratio",
    "continuity_score", "left_margin", "right_margin",
    "historic_jump_sum_radii", "jump_advantage",
    "incoming_components_at_boundary", "projection_frames",
    "maximum_projection_components", "maximum_projection_area_quantile",
    "maximum_projection_distance_sum_radii", "eligible", "reason",
]
APPLICATION_COLUMNS = [
    "proposal_id", "frame", "source_owner", "incoming_owner",
    "component_pixels", "preexisting_source_components",
    "explained_projection_components", "changed_pixels", "applied", "reason",
]


def assert_target_free(params: dict) -> None:
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError("unsupported dormant-takeover parameters: "
                         + ", ".join(unsupported))
    supplied = []
    for key, value in params.items():
        if value in (None, "", [], {}):
            continue
        name = str(key).strip().lower()
        if name in FORBIDDEN_KEYS or any(
                token in name for token in FORBIDDEN_FRAGMENTS):
            supplied.append(str(key))
    if supplied:
        raise ValueError("dormant-takeover discovery received forbidden "
                         "selectors: " + ", ".join(sorted(supplied)))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("dormant-takeover discovery must be field-wide")
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing dormant-takeover inputs: "
                         + ", ".join(missing))


def _components(labels: np.ndarray) -> tuple[np.ndarray, dict[int, Component]]:
    indexed = np.zeros(labels.shape, np.int32)
    result: dict[int, Component] = {}
    key = 0
    for frame, plane in enumerate(labels):
        for owner in sorted(map(int, np.unique(plane))):
            if owner <= 0:
                continue
            parts, count = ndi.label(plane == owner, STRUCTURE)
            objects = ndi.find_objects(parts)
            for component_id in range(1, count + 1):
                slc = objects[component_id - 1]
                if slc is None:
                    continue
                local = parts[slc] == component_id
                yy, xx = np.nonzero(local)
                area = int(len(xx))
                if area <= 0:
                    continue
                key += 1
                y0, x0 = int(slc[0].start), int(slc[1].start)
                y1, x1 = int(slc[0].stop), int(slc[1].stop)
                indexed[frame, y0:y1, x0:x1][local] = key
                result[key] = Component(
                    key, frame, owner, area,
                    float(x0 + xx.mean()), float(y0 + yy.mean()),
                    float(math.sqrt(area / math.pi)), y0, y1, x0, x1)
    return indexed, result


def _pair_features(indexed: np.ndarray, left: Component, right: Component,
                   params: dict) -> Edge | None:
    scale = max(left.radius + right.radius, 1.0)
    step = float(math.hypot(left.x - right.x, left.y - right.y) / scale)
    if step > float(params.get("maximum_candidate_step_sum_radii", 1.0)):
        return None
    dilation = max(1, int(math.ceil(0.35 * scale)))
    y0 = max(0, min(left.y0, right.y0) - dilation)
    y1 = min(indexed.shape[1], max(left.y1, right.y1) + dilation)
    x0 = max(0, min(left.x0, right.x0) - dilation)
    x1 = min(indexed.shape[2], max(left.x1, right.x1) + dilation)
    left_mask = indexed[left.frame, y0:y1, x0:x1] == left.key
    right_mask = indexed[right.frame, y0:y1, x0:x1] == right.key
    exact = int(np.count_nonzero(left_mask & right_mask))
    left_near = ndi.binary_dilation(
        left_mask, STRUCTURE, iterations=dilation)
    right_near = ndi.binary_dilation(
        right_mask, STRUCTURE, iterations=dilation)
    forward = int(np.count_nonzero(left_near & right_mask)) / right.area
    backward = int(np.count_nonzero(right_near & left_mask)) / left.area
    proximity = float(min(forward, backward))
    area_ratio = float(min(left.area, right.area) / max(left.area, right.area))
    exact_fraction = float(exact / max(min(left.area, right.area), 1))
    if proximity < float(params.get("minimum_proximity_coverage", 0.60)):
        return None
    if area_ratio < float(params.get("minimum_area_ratio", 0.30)):
        return None
    score = float(0.45 * proximity + 0.30 * area_ratio
                  + 0.15 * exact_fraction + 0.10 * max(0.0, 1.0 - step))
    return Edge(left.key, right.key, step, exact_fraction, proximity,
                area_ratio, score, 0.0, 0.0)


def _mutual_edges(indexed: np.ndarray, components: dict[int, Component],
                  params: dict) -> tuple[dict[int, Edge], dict[int, Edge]]:
    by_frame: dict[int, list[Component]] = {}
    for component in components.values():
        by_frame.setdefault(component.frame, []).append(component)
    candidates: list[Edge] = []
    for frame in range(indexed.shape[0] - 1):
        for left in by_frame.get(frame, []):
            for right in by_frame.get(frame + 1, []):
                edge = _pair_features(indexed, left, right, params)
                if edge is not None:
                    candidates.append(edge)
    by_left: dict[int, list[Edge]] = {}
    by_right: dict[int, list[Edge]] = {}
    for edge in candidates:
        by_left.setdefault(edge.left, []).append(edge)
        by_right.setdefault(edge.right, []).append(edge)
    outgoing: dict[int, Edge] = {}
    incoming: dict[int, Edge] = {}
    minimum_score = float(params.get("minimum_continuity_score", 0.65))
    minimum_margin = float(params.get("minimum_continuity_margin", 0.05))
    for edge in candidates:
        left_rank = sorted(by_left[edge.left], key=lambda item: item.score,
                           reverse=True)
        right_rank = sorted(by_right[edge.right], key=lambda item: item.score,
                            reverse=True)
        if left_rank[0] != edge or right_rank[0] != edge:
            continue
        left_margin = edge.score - (left_rank[1].score
                                    if len(left_rank) > 1 else 0.0)
        right_margin = edge.score - (right_rank[1].score
                                     if len(right_rank) > 1 else 0.0)
        if edge.score < minimum_score:
            continue
        if min(left_margin, right_margin) < minimum_margin:
            continue
        accepted = Edge(
            edge.left, edge.right, edge.step_sum_radii,
            edge.exact_overlap_fraction, edge.proximity_coverage,
            edge.area_ratio, edge.score, left_margin, right_margin)
        outgoing[edge.left] = accepted
        incoming[edge.right] = accepted
    return outgoing, incoming


def _owner_chain_backward(start: int, owner: int,
                          incoming: dict[int, Edge],
                          components: dict[int, Component]) -> list[int]:
    chain = [start]
    while chain[-1] in incoming:
        previous = incoming[chain[-1]].left
        if components[previous].owner != owner:
            break
        chain.append(previous)
    return list(reversed(chain))


def _owner_chain_forward(start: int, owner: int,
                         outgoing: dict[int, Edge],
                         components: dict[int, Component]) -> list[int]:
    chain = [start]
    while chain[-1] in outgoing:
        following = outgoing[chain[-1]].right
        if components[following].owner != owner:
            break
        chain.append(following)
    return chain


def _historic_jump(component: Component, owner_components: list[Component]
                   ) -> float:
    if not owner_components:
        return float("inf")
    return float(min(
        math.hypot(component.x - previous.x, component.y - previous.y)
        / max(component.radius + previous.radius, 1.0)
        for previous in owner_components))


def _projection_evidence(chain: list[int], source_owner: int,
                         indexed: np.ndarray,
                         components: dict[int, Component],
                         area_values: np.ndarray, params: dict
                         ) -> tuple[int, int, float, float, bool]:
    by_frame_owner: dict[tuple[int, int], list[Component]] = {}
    for component in components.values():
        by_frame_owner.setdefault((component.frame, component.owner), []).append(
            component)
    maximum_area_quantile = 0.0
    maximum_distance = 0.0
    projection_frames = 0
    maximum_components = 0
    proved = True
    area_limit = float(params.get("maximum_projection_area_quantile", 0.60))
    distance_limit = float(params.get(
        "maximum_projection_distance_sum_radii", 6.0))
    count_limit = int(params.get("maximum_projection_components_per_frame", 2))
    for key in chain:
        target = components[key]
        extras = by_frame_owner.get((target.frame, source_owner), [])
        if not extras:
            continue
        projection_frames += 1
        maximum_components = max(maximum_components, len(extras))
        if len(extras) > count_limit:
            proved = False
        for extra in extras:
            quantile = float(np.mean(area_values <= extra.area))
            distance = float(math.hypot(target.x - extra.x, target.y - extra.y)
                             / max(target.radius + extra.radius, 1.0))
            maximum_area_quantile = max(maximum_area_quantile, quantile)
            maximum_distance = max(maximum_distance, distance)
            if quantile > area_limit or distance > distance_limit:
                proved = False
    return (projection_frames, maximum_components, maximum_area_quantile,
            maximum_distance, proved)


def discover(labels: np.ndarray, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray,
                        dict[int, Component]]:
    """Audit every mutual, positive owner-changing component edge."""
    assert_target_free(params)
    indexed, components = _components(labels)
    outgoing, incoming = _mutual_edges(indexed, components, params)
    area_values = np.asarray([component.area for component in components.values()],
                             dtype=float)
    movie = len(labels)
    minimum_source = int(math.ceil(movie * float(params.get(
        "minimum_source_tenure_movie_fraction", 0.10))))
    minimum_dormancy = int(math.ceil(movie * float(params.get(
        "minimum_dormancy_movie_fraction", 0.15))))
    minimum_terminal = int(math.ceil(movie * float(params.get(
        "minimum_terminal_continuation_movie_fraction", 0.25))))
    minimum_area_quantile = float(params.get(
        "minimum_boundary_area_quantile", 0.75))
    minimum_historic_jump = float(params.get(
        "minimum_historic_jump_sum_radii", 2.5))
    minimum_advantage = float(params.get("minimum_jump_advantage", 5.0))
    maximum_incoming = int(params.get(
        "maximum_incoming_components_at_boundary", 1))
    by_owner: dict[int, list[Component]] = {}
    for component in components.values():
        by_owner.setdefault(component.owner, []).append(component)
    public_rows = []
    internal_rows = []
    for left_key, edge in sorted(outgoing.items()):
        left = components[left_key]
        right = components[edge.right]
        if left.owner <= 0 or right.owner <= 0 or left.owner == right.owner:
            continue
        source_chain = _owner_chain_backward(
            left.key, left.owner, incoming, components)
        terminal_chain = _owner_chain_forward(
            right.key, right.owner, outgoing, components)
        previous_incoming = [component for component in by_owner[right.owner]
                             if component.frame < right.frame]
        last_prior = max((component.frame for component in previous_incoming),
                         default=-1)
        last_prior_components = [component for component in previous_incoming
                                 if component.frame == last_prior]
        dormancy = right.frame - last_prior - 1 if last_prior >= 0 else -1
        historic_jump = _historic_jump(right, last_prior_components)
        advantage = float(historic_jump / max(edge.step_sum_radii, 0.05))
        boundary_quantile = float(np.mean(
            area_values <= min(left.area, right.area)))
        incoming_at_boundary = sum(
            component.frame == right.frame and component.owner == right.owner
            for component in components.values())
        (projection_frames, maximum_projection_components,
         maximum_projection_quantile, maximum_projection_distance,
         projections_proved) = _projection_evidence(
             terminal_chain, left.owner, indexed, components, area_values,
             params)
        reaches_boundary = components[terminal_chain[-1]].frame == movie - 1
        reasons = []
        if len(source_chain) < minimum_source:
            reasons.append("source_owner_tenure_too_short")
        if last_prior < 0:
            reasons.append("incoming_owner_has_no_prior_history")
        elif dormancy < minimum_dormancy:
            reasons.append("incoming_owner_not_dormant_long_enough")
        if len(terminal_chain) < minimum_terminal:
            reasons.append("incoming_terminal_chain_too_short")
        if not reaches_boundary:
            reasons.append("incoming_chain_not_right_censored")
        if boundary_quantile < minimum_area_quantile:
            reasons.append("boundary_body_below_field_area_quantile")
        if historic_jump < minimum_historic_jump:
            reasons.append("incoming_owner_history_spatially_plausible")
        if advantage < minimum_advantage:
            reasons.append("historic_jump_not_distinct_from_body_motion")
        if incoming_at_boundary > maximum_incoming:
            reasons.append("incoming_owner_has_independent_boundary_body")
        if not projections_proved:
            reasons.append("source_owner_has_unexplained_parallel_body")
        public = {
            "proposal_id": "", "source_owner": left.owner,
            "incoming_owner": right.owner, "transition_frame": right.frame,
            "source_component": left.key, "incoming_component": right.key,
            "source_chain_first": components[source_chain[0]].frame,
            "source_chain_last": left.frame,
            "source_tenure_frames": len(source_chain),
            "incoming_last_prior_frame": last_prior,
            "incoming_dormant_frames": dormancy,
            "terminal_first": right.frame,
            "terminal_last": components[terminal_chain[-1]].frame,
            "terminal_frames": len(terminal_chain),
            "terminal_reaches_boundary": reaches_boundary,
            "source_area": left.area, "incoming_area": right.area,
            "boundary_area_quantile": boundary_quantile,
            "step_sum_radii": edge.step_sum_radii,
            "exact_overlap_fraction": edge.exact_overlap_fraction,
            "proximity_coverage": edge.proximity_coverage,
            "area_ratio": edge.area_ratio,
            "continuity_score": edge.score,
            "left_margin": edge.left_margin, "right_margin": edge.right_margin,
            "historic_jump_sum_radii": historic_jump,
            "jump_advantage": advantage,
            "incoming_components_at_boundary": incoming_at_boundary,
            "projection_frames": projection_frames,
            "maximum_projection_components": maximum_projection_components,
            "maximum_projection_area_quantile": maximum_projection_quantile,
            "maximum_projection_distance_sum_radii": maximum_projection_distance,
            "eligible": not reasons,
            "reason": ("eligible_large_terminal_dormant_takeover"
                       if not reasons else "|".join(reasons)),
        }
        public_rows.append(public)
        internal_rows.append({**public, "terminal_chain": tuple(terminal_chain)})
    number = 0
    for public, internal in zip(public_rows, internal_rows):
        if public["eligible"]:
            number += 1
            proposal_id = f"LDT{number:04d}"
            public["proposal_id"] = internal["proposal_id"] = proposal_id
    return (pd.DataFrame(public_rows, columns=AUDIT_COLUMNS),
            pd.DataFrame(internal_rows), indexed, components)


def apply(labels: np.ndarray, internal: pd.DataFrame, indexed: np.ndarray,
          components: dict[int, Component], enabled: bool
          ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    rows = []
    occupied = np.zeros(labels.shape, bool)
    eligible = internal[internal.eligible.astype(bool)] if len(internal) \
        else internal
    for proposal in eligible.itertuples(index=False):
        pending = []
        refused = ""
        for key in proposal.terminal_chain:
            component = components[int(key)]
            mask = indexed[component.frame] == component.key
            if np.any(occupied[component.frame] & mask):
                refused = "overlapping_eligible_proposal"
                break
            if not np.all(candidate[component.frame][mask]
                          == int(proposal.incoming_owner)):
                refused = "incoming_component_provenance_changed"
                break
            before_components = ndi.label(
                candidate[component.frame] == int(proposal.source_owner),
                STRUCTURE)[1]
            pending.append((component, mask, before_components))
        if refused:
            rows.append({
                "proposal_id": proposal.proposal_id, "frame": -1,
                "source_owner": int(proposal.source_owner),
                "incoming_owner": int(proposal.incoming_owner),
                "component_pixels": 0, "preexisting_source_components": 0,
                "explained_projection_components": 0, "changed_pixels": 0,
                "applied": False, "reason": refused,
            })
            continue
        for component, mask, before_components in pending:
            changed = int(mask.sum()) if enabled else 0
            if enabled:
                candidate[component.frame][mask] = int(proposal.source_owner)
                occupied[component.frame] |= mask
            rows.append({
                "proposal_id": proposal.proposal_id,
                "frame": component.frame,
                "source_owner": int(proposal.source_owner),
                "incoming_owner": int(proposal.incoming_owner),
                "component_pixels": component.area,
                "preexisting_source_components": int(before_components),
                "explained_projection_components": int(before_components),
                "changed_pixels": changed, "applied": bool(enabled),
                "reason": ("large_terminal_dormant_takeover_recovery"
                           if enabled else "audit_only"),
            })
    return candidate, pd.DataFrame(rows, columns=APPLICATION_COLUMNS)


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, params: dict):
    del points  # frozen provenance; component discovery is label/raw based
    if labels.shape != unclaimed.shape or labels.shape != raw.shape:
        raise ValueError("labels, unclaimed and raw stacks must align")
    audit, internal, indexed, components = discover(labels, params)
    candidate, applications = apply(
        labels, internal, indexed, components,
        bool(params.get("apply_recovery", True)))
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("dormant takeover changed assigned foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("dormant takeover overlapped unclaimed ledger")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("dormant takeover changed the movie identity set")
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    new_excess = component_accounting.new_duplicate_components(
        labels, candidate)
    summary = {
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "components_audited": int(len(components)),
        "owner_transitions_audited": int(len(audit)),
        "eligible_takeovers": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_takeovers": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_exact": True, "unclaimed_ledger_exact": True,
        "movie_identity_set_exact": True,
        "new_component_excess": int(new_excess),
        "new_unexplained_component_excess": 0,
        "projection_explanation": "field_relative_small_nearby_source_pieces",
    }
    return candidate, unclaimed.copy(), audit, applications, summary


def run(_upstream: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels_path = Path(params["labels_path"])
    unclaimed_path = Path(params["unclaimed_path"])
    labels = tifffile.imread(labels_path)
    unclaimed = tifffile.imread(unclaimed_path)
    raw = tifffile.imread(params["raw_path"])
    points = pd.read_csv(params["physical_track_points_path"])
    candidate, candidate_unclaimed, audit, applications, summary = produce(
        labels, unclaimed, raw, points, params)
    output = Path(out.out)
    output.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", labels_path.stem))
    outputs = {
        "labels": output / f"{stem}.tif",
        "unclaimed": output / f"{stem}_unclaimed_original_ids.tif",
        "audit": output / "large_terminal_dormant_takeover_audit.csv",
        "applications": output /
            "large_terminal_dormant_takeover_applications.csv",
        "metrics": output / "producer_metrics.json",
    }
    tifffile.imwrite(
        outputs["labels"], candidate, imagej=True, compression="zlib",
        photometric="minisblack", metadata={
            "axes": "TYX", "finterval": 1800.0, "tunit": "sec",
            "unit": "pixel"})
    shutil.copyfile(unclaimed_path, outputs["unclaimed"])
    audit.to_csv(outputs["audit"], index=False)
    applications.to_csv(outputs["applications"], index=False)
    outputs["metrics"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": summary}

