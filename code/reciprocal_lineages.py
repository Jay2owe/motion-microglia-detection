"""Discover and solve recurrent reciprocal two-body lineage swaps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from local_lineage_reservations import discover_duplicate_handoffs


CONNECTIVITY = np.ones((3, 3), np.uint8)


@dataclass
class ReciprocalLineageResult:
    labels: np.ndarray
    unclaimed: np.ndarray
    repair_domains: np.ndarray
    duplicate_audit: pd.DataFrame
    predecessor_audit: pd.DataFrame
    pair_audit: pd.DataFrame
    components: pd.DataFrame
    assignments: pd.DataFrame
    mismatches: pd.DataFrame
    actions: pd.DataFrame


def frame_components(frame: np.ndarray, identities: tuple[int, int],
                     source_frame: int, minimum_area: int) -> list[dict]:
    """Return deterministic substantial components of one pair's joint mask."""
    connected, count = ndi.label(np.isin(frame, identities),
                                 structure=CONNECTIVITY)
    rows: list[dict] = []
    for component in range(1, count + 1):
        mask = connected == component
        y, x = np.nonzero(mask)
        area = int(len(x))
        if area < int(minimum_area):
            continue
        values, counts = np.unique(frame[mask], return_counts=True)
        rows.append({
            "source_frame": int(source_frame),
            "x": float(x.mean()), "y": float(y.mean()), "area": area,
            "_raw_component_number": int(component),
            "base_majority_identity": int(values[int(np.argmax(counts))]),
            "identity_a_pixels": int(np.count_nonzero(
                frame[mask] == identities[0])),
            "identity_b_pixels": int(np.count_nonzero(
                frame[mask] == identities[1])),
        })
    rows.sort(key=lambda row: (row["x"], row["y"], -row["area"]))
    for number, row in enumerate(rows, 1):
        row["component_id"] = f"S{source_frame:03d}-C{number:02d}"
    return rows


def solve_two_lanes(components: list[dict], identities: tuple[int, int],
                    fixed_assignment_frames: set[int], area_weight: float
                    ) -> list[dict]:
    """Assign every observed component to two histories by global path cost."""
    by_frame: dict[int, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for row in components:
        by_frame.setdefault(int(row["source_frame"]), []).append(row)
        by_id[str(row["component_id"])] = row
    frames = list(range(min(by_frame), max(by_frame) + 1))
    states: dict[tuple[str | None, str | None],
                 tuple[float, list[tuple[str | None, str | None]]]] = {
                     (None, None): (0.0, [])}
    for source_frame in frames:
        observed = by_frame.get(source_frame, [])
        if len(observed) > 2:
            raise AssertionError("pair has more than two substantial bodies")
        if len(observed) == 2:
            first, second = map(lambda row: str(row["component_id"]), observed)
            options = ((first, second), (second, first))
        elif len(observed) == 1:
            first = str(observed[0]["component_id"])
            options = ((first, None), (None, first))
        else:
            options = ((None, None),)
        next_states: dict[tuple[str | None, str | None],
                          tuple[float, list[tuple[str | None, str | None]]]] = {}
        for (last_a, last_b), (cost, path) in states.items():
            for current_a, current_b in options:
                if source_frame in fixed_assignment_frames:
                    if current_a is None or current_b is None:
                        continue
                    if (int(by_id[current_a]["base_majority_identity"])
                            != identities[0]
                            or int(by_id[current_b]["base_majority_identity"])
                            != identities[1]):
                        continue
                added = 0.0
                for previous, current in ((last_a, current_a),
                                          (last_b, current_b)):
                    if previous is None or current is None:
                        continue
                    before, after = by_id[previous], by_id[current]
                    elapsed = max(1, int(after["source_frame"])
                                  - int(before["source_frame"]))
                    distance_sq = ((float(after["x"]) - float(before["x"])) ** 2
                                   + (float(after["y"])
                                      - float(before["y"])) ** 2)
                    area_cost = float(area_weight) * abs(math.log(
                        float(after["area"]) / float(before["area"])))
                    added += distance_sq / elapsed + area_cost
                key = (current_a if current_a is not None else last_a,
                       current_b if current_b is not None else last_b)
                proposed = (cost + added, path + [(current_a, current_b)])
                if key not in next_states or proposed[0] < next_states[key][0]:
                    next_states[key] = proposed
        if not next_states:
            raise AssertionError("no two-lane solution survives the seed bookends")
        states = next_states
    total_cost, best_path = min(states.values(), key=lambda item: item[0])
    assignments: list[dict] = []
    for source_frame, pair in zip(frames, best_path):
        for identity, component_id in zip(identities, pair):
            component = by_id.get(component_id) if component_id else None
            assignments.append({
                "source_frame": source_frame, "identity": identity,
                "component_id": component_id or "",
                "observed": component is not None,
                "x": "" if component is None else float(component["x"]),
                "y": "" if component is None else float(component["y"]),
                "area": "" if component is None else int(component["area"]),
                "base_majority_identity": ("" if component is None else int(
                    component["base_majority_identity"])),
                "path_total_cost": float(total_cost),
            })
    table = pd.DataFrame(assignments)
    for identity in identities:
        selected = table[table.identity.astype(int) == identity]
        observed = selected.observed.astype(bool).to_numpy()
        indices = np.arange(len(selected), dtype=float)
        for axis in ("x", "y"):
            values = pd.to_numeric(
                selected[axis], errors="coerce").to_numpy(float)
            table.loc[selected.index, f"predicted_{axis}"] = np.interp(
                indices, indices[observed], values[observed])
    return table.to_dict("records")


def _joint_history(labels: np.ndarray, pair: tuple[int, int],
                   source_frame_offset: int, minimum_area: int
                   ) -> tuple[list[list[dict]], list[dict]]:
    by_review: list[list[dict]] = []
    all_rows: list[dict] = []
    for review_index, frame in enumerate(labels):
        rows = frame_components(
            frame, pair, review_index + 1 + int(source_frame_offset),
            minimum_area)
        by_review.append(rows)
        all_rows.extend(rows)
    return by_review, all_rows


def _clean_pair_frame(rows: list[dict], pair: tuple[int, int]) -> bool:
    return bool(len(rows) == 2 and {
        int(row["base_majority_identity"]) for row in rows} == set(pair))


def _bookends_before_contact(by_review: list[list[dict]], seed_index: int,
                             pair: tuple[int, int], minimum_frames: int
                             ) -> list[int]:
    index = seed_index - 1
    while index >= 0:
        if len(by_review[index]) != 1:
            index -= 1
            continue
        contact_start = index
        while contact_start > 0 and len(by_review[contact_start - 1]) == 1:
            contact_start -= 1
        start = contact_start - int(minimum_frames)
        if start >= 0:
            bookends = list(range(start, contact_start))
            if all(_clean_pair_frame(by_review[frame], pair)
                   for frame in bookends):
                return bookends
        index = contact_start - 1
    return []


def discover_candidate_pairs(labels: np.ndarray, lag: np.ndarray, params: dict
                             ) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Derive unresolved identity pairs from field-wide duplicate handoffs."""
    _, duplicate_audit, predecessor_audit, _ = discover_duplicate_handoffs(
        labels, lag, params["handoff_rule"])
    candidates: dict[tuple[int, int], dict] = {}
    for duplicate in duplicate_audit.itertuples(index=False):
        if str(duplicate.decision) != "refused_predecessor_did_not_return":
            continue
        frame_rows = predecessor_audit[
            (predecessor_audit.review_frame.astype(int)
             == int(duplicate.review_frame))
            & (predecessor_audit.duplicated_identity.astype(int)
               == int(duplicate.duplicated_identity))]
        winners = set(frame_rows.winner_identity.astype(int)) - {
            0, int(duplicate.duplicated_identity)}
        if len(winners) != 1:
            continue
        other = int(next(iter(winners)))
        pair = tuple(sorted((int(duplicate.duplicated_identity), other)))
        row = {"pair": pair, "seed_review_frame": int(duplicate.review_frame),
               "seed_duplicated_identity": int(duplicate.duplicated_identity),
               "seed_predecessor_identity": other}
        if pair not in candidates or row["seed_review_frame"] < candidates[pair][
                "seed_review_frame"]:
            candidates[pair] = row
    return (list(candidates.values()), duplicate_audit, predecessor_audit)


def reconcile_reciprocal_lineages(labels: np.ndarray, unclaimed: np.ndarray,
                                  lag: np.ndarray, params: dict
                                  ) -> ReciprocalLineageResult:
    """Discover recurrent two-body swaps and solve each complete pair history."""
    if labels.shape != unclaimed.shape:
        raise ValueError("labels and unclaimed stacks must have the same shape")
    seeds, duplicate_audit, predecessor_audit = discover_candidate_pairs(
        labels, lag, params)
    offset = int(params["source_frame_offset"])
    minimum_area = int(params["minimum_component_area"])
    pair_rows: list[dict] = []
    all_components: list[dict] = []
    all_assignments: list[dict] = []
    all_mismatches: list[dict] = []
    accepted: list[dict] = []
    for number, seed in enumerate(sorted(
            seeds, key=lambda row: (row["seed_review_frame"], row["pair"])), 1):
        pair = tuple(map(int, seed["pair"]))
        by_review, component_rows = _joint_history(
            labels, pair, offset, minimum_area)
        seed_index = int(seed["seed_review_frame"]) - 1
        bookends = _bookends_before_contact(
            by_review, seed_index, pair, int(params["minimum_seed_frames"]))
        base = {
            "candidate_id": f"RLP-{number:04d}",
            "identity_a": pair[0], "identity_b": pair[1],
            "seed_review_frame": seed["seed_review_frame"],
            "bookend_review_frames": ";".join(map(
                str, [frame + 1 for frame in bookends])),
            "start_review_frame": 0 if not bookends else bookends[0] + 1,
            "end_review_frame": len(labels),
            "maximum_substantial_components": 0,
            "lane_mismatch_components": 0,
            "lane_mismatch_frames": 0,
            "lane_mismatch_pixels": 0,
            "accepted": False,
            "decision": "refused_no_precontact_clean_bookends",
        }
        if not bookends:
            pair_rows.append(base)
            continue
        start = bookends[0]
        maximum_components = max(len(rows) for rows in by_review[start:])
        base["maximum_substantial_components"] = maximum_components
        if maximum_components > 2:
            base["decision"] = "refused_more_than_two_later_bodies"
            pair_rows.append(base)
            continue
        selected_components = [row for row in component_rows
                               if int(row["source_frame"])
                               >= start + 1 + offset]
        fixed_assignment_frames = {frame + 1 + offset for frame in bookends}
        assignments = solve_two_lanes(
            selected_components, pair, fixed_assignment_frames,
            float(params["area_weight"]))
        component_by_id = {str(row["component_id"]): row
                           for row in selected_components}
        mismatches: list[dict] = []
        for row in assignments:
            component_id = str(row["component_id"])
            if not component_id:
                continue
            component = component_by_id[component_id]
            if int(component["base_majority_identity"]) != int(row["identity"]):
                mismatches.append({
                    "candidate_id": base["candidate_id"],
                    "source_frame": int(row["source_frame"]),
                    "component_id": component_id,
                    "base_identity": int(component["base_majority_identity"]),
                    "lane_identity": int(row["identity"]),
                    "area": int(component["area"]),
                    "x": float(component["x"]), "y": float(component["y"]),
                })
        mismatch_frames = len({row["source_frame"] for row in mismatches})
        span = len(labels) - start
        base.update({
            "lane_mismatch_components": len(mismatches),
            "lane_mismatch_frames": mismatch_frames,
            "lane_mismatch_pixels": int(sum(row["area"] for row in mismatches)),
        })
        if mismatch_frames < int(math.ceil(
                float(params["minimum_mismatch_frames_fraction"]) * span)):
            base["decision"] = "refused_nonrecurrent_lane_mismatches"
            pair_rows.append(base)
            continue
        base["accepted"] = True
        base["decision"] = "accepted_recurrent_reciprocal_lineages"
        pair_rows.append(base)
        accepted.append({**base, "pair": pair, "start": start,
                         "assignments": assignments,
                         "components": selected_components,
                         "mismatches": mismatches})

    candidate = labels.copy()
    domains = np.zeros(labels.shape, np.uint16)
    action_rows: list[dict] = []
    for event_number, event in enumerate(accepted, 1):
        pair = event["pair"]
        expected = {
            (int(row["source_frame"]), str(row["component_id"])): int(
                row["identity"])
            for row in event["assignments"] if bool(row["observed"])}
        for source_frame in sorted({int(row["source_frame"])
                                    for row in event["assignments"]}):
            review_index = source_frame - offset - 1
            for component in frame_components(
                    labels[review_index], pair, source_frame, minimum_area):
                target = expected[(source_frame, str(component["component_id"]))]
                connected, _ = ndi.label(np.isin(labels[review_index], pair),
                                         structure=CONNECTIVITY)
                mask = connected == int(component["_raw_component_number"])
                change_mask = mask & (candidate[review_index] != target)
                changed = int(np.count_nonzero(
                    change_mask))
                if not changed:
                    continue
                if np.any(domains[review_index][mask]):
                    raise AssertionError("reciprocal pair repair domains overlap")
                before_identities = sorted(map(
                    int, np.unique(candidate[review_index][change_mask])))
                before_values = (before_identities[0]
                                 if len(before_identities) == 1 else
                                 "|".join(map(str, before_identities)))
                candidate[review_index][mask] = target
                domains[review_index][change_mask] = event_number
                action_rows.append({
                    "mechanism": "L_two_lineage_lane_graph",
                    "source_frame": source_frame,
                    "component_id": component["component_id"],
                    "from_identity": before_values,
                    "to_identity": target, "changed_pixels": changed,
                    "evidence": "minimum whole-track displacement and area cost",
                })
        for row in event["components"]:
            all_components.append({"candidate_id": event["candidate_id"], **row})
        for row in event["assignments"]:
            all_assignments.append({"candidate_id": event["candidate_id"], **row})
        all_mismatches.extend(event["mismatches"])
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("reciprocal lineage repair changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    return ReciprocalLineageResult(
        labels=candidate, unclaimed=unclaimed.copy(), repair_domains=domains,
        duplicate_audit=duplicate_audit,
        predecessor_audit=predecessor_audit,
        pair_audit=pd.DataFrame(pair_rows),
        components=pd.DataFrame(all_components),
        assignments=pd.DataFrame(all_assignments),
        mismatches=pd.DataFrame(all_mismatches),
        actions=pd.DataFrame(action_rows, columns=[
            "mechanism", "source_frame", "component_id", "from_identity",
            "to_identity", "changed_pixels", "evidence"]))
