"""Recover a proved multi-owner seat cycle without biological targets.

The rule looks for three or more independently established physical seats whose
owners fail within the same short interval: owners move onto one another's
seats and at least one established seat becomes ownerless while remaining
visible.  Once that closed evidence set is proved, recurrent flashes between
the same owners on a stable physical seat are corrected as part of the same
episode.  Short companion references inherit a seat only when they shared its
accepted component immediately before the exchange.

Discovery is field-wide.  Identity, track, frame, coordinate, event, region,
well and review selectors are rejected.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile

import component_accounting
from ownerless_body_recovery import evidence_image
from ownerless_cohort_latent_gap_completion import _nearest_raw_core


STRUCTURE = np.ones((3, 3), np.uint8)
CONTROL_KEYS = {"mode", "targeting_mode", "output_stem"}
PATH_KEYS = {
    "labels_path", "unclaimed_path", "raw_path",
    "physical_track_points_path", "thresholds_path",
}
PARAMETER_KEYS = {
    "minimum_anchor_run_movie_fraction",
    "minimum_anchor_run_frames",
    "minimum_anchor_owner_purity",
    "minimum_changed_post_frames",
    "maximum_joint_onset_movie_fraction",
    "minimum_joint_owners",
    "minimum_joint_foreign_edges",
    "minimum_companion_support_frames",
    "maximum_companion_gap_frames",
    "maximum_component_point_distance_radii",
    "minimum_raw_core_area_radius_ratio",
    "maximum_raw_core_area_radius_ratio",
    "maximum_core_offset_radii",
    "core_window_radius_px",
    "core_peak_fraction",
    "weak_threshold_fraction",
}
ALLOWED_KEYS = CONTROL_KEYS | PATH_KEYS | PARAMETER_KEYS
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "anchor_owner", "anchor_first_frame",
    "anchor_last_frame", "anchor_run_frames", "change_first_frame",
    "track_last_frame", "foreign_owners", "foreign_frames",
    "ownerless_frames", "joint_episode", "propagated_pair",
    "eligible", "reason",
]
SEAT_COLUMNS = [
    "proposal_id", "track_id", "anchor_track", "anchor_owner", "role",
    "first_correction_frame", "last_correction_frame", "support_frames",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "track_id", "anchor_owner", "observed_owner",
    "operation", "changed_pixels", "raw_core_area_radius_ratio",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("established-seat cycle recovery must be field-wide")
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported parameters (selectors are forbidden): "
            + ", ".join(unsupported))
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing input paths: " + ", ".join(missing))


def _visible(group: pd.DataFrame) -> pd.DataFrame:
    if "physically_visible" in group:
        mask = group.physically_visible.astype(bool)
    else:
        mask = group.state.astype(str).isin(["observed", "latent_visible"])
    return group[mask].sort_values("frame").copy()


def _owner_runs(group: pd.DataFrame) -> list[pd.DataFrame]:
    group = _visible(group)
    if group.empty:
        return []
    owner = group.candidate_owner.astype(int)
    split = (group.frame.astype(int).diff().fillna(1).ne(1)
             | owner.ne(owner.shift()).fillna(True)).cumsum()
    return [run.copy() for _, run in group.groupby(split, sort=False)]


def _component_near(frame: np.ndarray, owner: int, point,
                    maximum_distance_radii: float) -> np.ndarray | None:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    if not count:
        return None
    radius = max(float(point.radius_px), 1.0)
    choices = []
    for index in range(1, int(count) + 1):
        yy, xx = np.nonzero(components == index)
        if not len(xx):
            continue
        distance = float(np.sqrt(np.min(
            (yy - float(point.y)) ** 2 + (xx - float(point.x)) ** 2)))
        if distance <= maximum_distance_radii * radius:
            choices.append((distance, -len(xx), index))
    if not choices:
        return None
    return components == min(choices)[2]


def _anchor_candidates(points: pd.DataFrame, frame_count: int, params: dict
                       ) -> list[dict]:
    minimum_run = max(
        int(params.get("minimum_anchor_run_frames", 3)),
        int(math.ceil(frame_count * float(params.get(
            "minimum_anchor_run_movie_fraction", 0.08)))))
    minimum_purity = float(params.get("minimum_anchor_owner_purity", 0.90))
    minimum_changed = int(params.get("minimum_changed_post_frames", 1))
    candidates = []
    for track_id, original in points.groupby("track_id", sort=True):
        group = _visible(original)
        if group.empty:
            continue
        runs = _owner_runs(group)
        qualified = []
        for run in runs:
            owner = int(run.candidate_owner.iloc[0])
            if owner <= 0 or len(run) < minimum_run:
                continue
            purity = float(run.candidate_owner_purity.astype(float).median())
            if purity < minimum_purity:
                continue
            after = group[group.frame.astype(int) > int(run.frame.max())]
            changed = after[after.candidate_owner.astype(int).ne(owner)]
            if len(changed) < minimum_changed:
                continue
            qualified.append((int(run.frame.max()), run, after, changed))
        if not qualified:
            continue
        _, run, after, changed = max(qualified, key=lambda item: item[0])
        owner = int(run.candidate_owner.iloc[0])
        foreign = changed[
            changed.candidate_owner.astype(int).gt(0)
            & changed.candidate_owner.astype(int).ne(owner)]
        ownerless = changed[changed.candidate_owner.astype(int).eq(0)]
        candidates.append({
            "track_id": int(track_id), "anchor_owner": owner,
            "anchor_first_frame": int(run.frame.min()),
            "anchor_last_frame": int(run.frame.max()),
            "anchor_run_frames": int(len(run)),
            "change_first_frame": int(changed.frame.min()),
            "track_last_frame": int(group.frame.max()),
            "foreign_owners_set": set(map(
                int, foreign.candidate_owner.astype(int).unique())),
            "foreign_frames": int(len(foreign)),
            "ownerless_frames": int(len(ownerless)),
            "group": group, "anchor_run": run, "after": after,
        })
    return candidates


def _joint_episodes(candidates: list[dict], frame_count: int,
                    params: dict) -> list[dict]:
    maximum_window = max(1, int(math.ceil(frame_count * float(params.get(
        "maximum_joint_onset_movie_fraction", 0.05)))))
    minimum_owners = int(params.get("minimum_joint_owners", 3))
    minimum_edges = int(params.get("minimum_joint_foreign_edges", 2))
    ordered = sorted(candidates, key=lambda value: value["change_first_frame"])
    proved: dict[frozenset[int], dict] = {}
    for start in range(len(ordered)):
        first = ordered[start]["change_first_frame"]
        cluster = [value for value in ordered[start:]
                   if value["change_first_frame"] - first <= maximum_window]
        owners = {int(value["anchor_owner"]) for value in cluster}
        edges = {(int(value["anchor_owner"]), foreign)
                 for value in cluster
                 if value["foreign_owners_set"] <= owners
                 for foreign in value["foreign_owners_set"]
                 if foreign != value["anchor_owner"]}
        neighbours = {owner: set() for owner in owners}
        for left, right in edges:
            neighbours[left].add(right)
            neighbours[right].add(left)
        remaining = set(owners)
        while remaining:
            root = min(remaining)
            component = {root}
            frontier = [root]
            while frontier:
                node = frontier.pop()
                for neighbour in neighbours[node] - component:
                    component.add(neighbour)
                    frontier.append(neighbour)
            remaining -= component
            component_edges = {edge for edge in edges
                               if edge[0] in component and edge[1] in component}
            has_ownerless = any(
                int(value["anchor_owner"]) in component
                and value["ownerless_frames"] > 0 for value in cluster)
            if (len(component) >= minimum_owners
                    and len(component_edges) >= minimum_edges
                    and has_ownerless):
                key = frozenset(component)
                episode = proved.setdefault(
                    key, {"owners": set(component), "first_frame": first,
                          "last_joint_frame": first + maximum_window})
                episode["first_frame"] = min(episode["first_frame"], first)
                episode["last_joint_frame"] = max(
                    episode["last_joint_frame"], first + maximum_window)
    return sorted(proved.values(), key=lambda value: (
        value["first_frame"], sorted(value["owners"])))


def _simple_directed_dropout_chains(
        candidates: list[dict], episodes: list[dict]) -> list[dict]:
    """Retain episodes that prove one owner-to-owner chain ending in dropout.

    Undirected connectivity alone also admits two biologically different
    patterns: several established owners converging on one name, or one name
    branching onto several established owners.  Neither proves a displaced
    seat cycle.  A recoverable episode must instead be a simple directed path
    through every established owner, with the terminal seat remaining visible
    after its owner disappears.
    """
    proved: list[dict] = []
    for episode in episodes:
        owners = set(map(int, episode["owners"]))
        initial = [
            value for value in candidates
            if int(value["anchor_owner"]) in owners
            and int(episode["first_frame"])
            <= int(value["change_first_frame"])
            <= int(episode["last_joint_frame"])
        ]
        represented = {int(value["anchor_owner"]) for value in initial}
        edges = {
            (int(value["anchor_owner"]), int(foreign))
            for value in initial
            for foreign in value["foreign_owners_set"]
            if int(foreign) in owners
            and int(foreign) != int(value["anchor_owner"])
        }
        incoming = {owner: 0 for owner in owners}
        outgoing = {owner: 0 for owner in owners}
        for left, right in edges:
            outgoing[left] += 1
            incoming[right] += 1
        sources = [owner for owner in owners
                   if incoming[owner] == 0 and outgoing[owner] == 1]
        sinks = [owner for owner in owners
                 if incoming[owner] == 1 and outgoing[owner] == 0]
        middle = [owner for owner in owners
                  if incoming[owner] == 1 and outgoing[owner] == 1]
        sink_has_dropout = bool(sinks) and any(
            int(value["anchor_owner"]) == sinks[0]
            and int(value["ownerless_frames"]) > 0
            for value in initial)
        if (represented == owners
                and len(edges) == len(owners) - 1
                and len(sources) == 1
                and len(sinks) == 1
                and len(middle) == len(owners) - 2
                and sink_has_dropout):
            proved.append(episode)
    return proved


def discover(labels: np.ndarray, points: pd.DataFrame, params: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Find closed established-seat cycles and their stable continuations."""
    candidates = _anchor_candidates(points, len(labels), params)
    episodes = _joint_episodes(candidates, len(labels), params)
    episodes = _simple_directed_dropout_chains(candidates, episodes)
    rows = []
    selected = []
    number = 0
    for value in candidates:
        owner = int(value["anchor_owner"])
        foreign = set(value["foreign_owners_set"])
        matching = [episode for episode in episodes
                    if owner in episode["owners"]
                    and (not foreign or foreign <= episode["owners"])]
        joint = any(
            int(value["change_first_frame"]) <= int(episode["last_joint_frame"])
            for episode in matching)
        propagated = bool(foreign) and bool(matching)
        eligible = joint or propagated
        reasons = []
        if not episodes:
            reasons.append("no_closed_multi_owner_dropout_episode")
        elif not any(owner in episode["owners"] for episode in episodes):
            reasons.append("anchor_owner_outside_proved_episode")
        elif not matching:
            reasons.append("foreign_owner_outside_proved_episode")
        elif not eligible:
            reasons.append("no_joint_or_recurrent_pair_evidence")
        if eligible:
            number += 1
            proposal_id = f"SC{number:04d}"
            correction_first = min(
                (int(episode["first_frame"]) for episode in matching
                 if int(value["change_first_frame"])
                 <= int(episode["last_joint_frame"])),
                default=int(value["change_first_frame"]))
            selected.append({
                **value, "proposal_id": proposal_id,
                "correction_first_frame": correction_first,
            })
        else:
            proposal_id = ""
        rows.append({
            "proposal_id": proposal_id,
            "track_id": int(value["track_id"]),
            "anchor_owner": owner,
            "anchor_first_frame": int(value["anchor_first_frame"]),
            "anchor_last_frame": int(value["anchor_last_frame"]),
            "anchor_run_frames": int(value["anchor_run_frames"]),
            "change_first_frame": int(value["change_first_frame"]),
            "track_last_frame": int(value["track_last_frame"]),
            "foreign_owners": ";".join(map(str, sorted(foreign))),
            "foreign_frames": int(value["foreign_frames"]),
            "ownerless_frames": int(value["ownerless_frames"]),
            "joint_episode": joint, "propagated_pair": propagated,
            "eligible": eligible,
            "reason": ("eligible_established_seat_cycle" if eligible
                       else "|".join(reasons)),
        })

    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    seats = _seat_members(labels, points, selected, params)
    return audit, seats, selected


def _seat_members(labels: np.ndarray, points: pd.DataFrame,
                  selected: list[dict], params: dict) -> pd.DataFrame:
    """Attach short references that shared an anchor's pre-exchange body."""
    minimum_support = int(params.get("minimum_companion_support_frames", 2))
    maximum_gap = int(params.get("maximum_companion_gap_frames", 1))
    maximum_distance = float(params.get(
        "maximum_component_point_distance_radii", 1.0))
    groups = {int(track): _visible(group) for track, group in
              points.groupby("track_id", sort=True)}
    rows = []
    seen: set[tuple[str, int]] = set()
    selected_tracks = {int(value["track_id"]) for value in selected}
    for anchor in selected:
        proposal_id = str(anchor["proposal_id"])
        anchor_track = int(anchor["track_id"])
        owner = int(anchor["anchor_owner"])
        first = int(anchor["correction_first_frame"])
        group = groups[anchor_track]
        rows.append({
            "proposal_id": proposal_id, "track_id": anchor_track,
            "anchor_track": anchor_track, "anchor_owner": owner,
            "role": "anchor", "first_correction_frame": first,
            "last_correction_frame": int(group.frame.max()),
            "support_frames": int(anchor["anchor_run_frames"]),
        })
        seen.add((proposal_id, anchor_track))
        anchor_run = anchor["anchor_run"]
        anchor_frames = set(map(int, anchor_run.frame))
        for track_id, companion in groups.items():
            if track_id == anchor_track or (proposal_id, track_id) in seen:
                continue
            if track_id in selected_tracks:
                continue
            support = companion[
                companion.frame.astype(int).isin(anchor_frames)
                & companion.candidate_owner.astype(int).eq(owner)]
            if len(support) < minimum_support:
                continue
            shared = 0
            for point in support.itertuples(index=False):
                anchor_point = anchor_run[
                    anchor_run.frame.astype(int).eq(int(point.frame))]
                if len(anchor_point) != 1:
                    continue
                anchor_point = anchor_point.iloc[0]
                left = _component_near(
                    labels[int(point.frame)], owner, point, maximum_distance)
                right = _component_near(
                    labels[int(point.frame)], owner, anchor_point,
                    maximum_distance)
                if left is not None and right is not None and np.array_equal(left, right):
                    shared += 1
            if shared < minimum_support:
                continue
            changed = companion[
                companion.frame.astype(int).gt(int(support.frame.max()))
                & companion.candidate_owner.astype(int).ne(owner)]
            if changed.empty:
                continue
            if int(changed.frame.min()) - int(support.frame.max()) > maximum_gap + 1:
                continue
            rows.append({
                "proposal_id": proposal_id, "track_id": int(track_id),
                "anchor_track": anchor_track, "anchor_owner": owner,
                "role": "shared_component_companion",
                "first_correction_frame": int(changed.frame.min()),
                "last_correction_frame": int(companion.frame.max()),
                "support_frames": shared,
            })
            seen.add((proposal_id, int(track_id)))
    return pd.DataFrame(rows, columns=SEAT_COLUMNS)


def _points_by_frame(points: pd.DataFrame, seats: pd.DataFrame) -> dict[int, list[dict]]:
    groups = {int(track): _visible(group).set_index("frame", drop=False)
              for track, group in points.groupby("track_id", sort=True)}
    result: dict[int, list[dict]] = {}
    for seat in seats.itertuples(index=False):
        group = groups[int(seat.track_id)]
        frames = group.index[
            (group.index.astype(int) >= int(seat.first_correction_frame))
            & (group.index.astype(int) <= int(seat.last_correction_frame))]
        for frame in map(int, frames):
            point = group.loc[frame]
            if isinstance(point, pd.DataFrame):
                continue
            result.setdefault(frame, []).append({
                "proposal_id": str(seat.proposal_id),
                "track_id": int(seat.track_id),
                "anchor_track": int(seat.anchor_track),
                "anchor_owner": int(seat.anchor_owner),
                "role": str(seat.role), "point": point,
            })
    return result


def _partition(mask: np.ndarray, members: list[dict]) -> dict[int, np.ndarray]:
    yy, xx = np.nonzero(mask)
    result = {int(member["anchor_owner"]): np.zeros_like(mask)
              for member in members}
    if not len(xx):
        return result
    costs = []
    for member in members:
        point = member["point"]
        radius = max(float(point.radius_px), 1.0)
        costs.append(((yy - float(point.y)) ** 2
                      + (xx - float(point.x)) ** 2) / radius ** 2)
    chosen = np.argmin(np.asarray(costs), axis=0)
    for index, member in enumerate(members):
        owner = int(member["anchor_owner"])
        take = chosen == index
        result[owner][yy[take], xx[take]] = True
    return result


def apply(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
          points: pd.DataFrame, thresholds: pd.DataFrame,
          seats: pd.DataFrame, params: dict
          ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    candidate = labels.copy()
    threshold_by_frame = thresholds.set_index("frame")
    maximum_distance = float(params.get(
        "maximum_component_point_distance_radii", 1.0))
    minimum_raw_ratio = float(params.get(
        "minimum_raw_core_area_radius_ratio", 0.20))
    maximum_raw_ratio = float(params.get(
        "maximum_raw_core_area_radius_ratio", 2.0))
    by_frame = _points_by_frame(points, seats)
    frame_rows = []
    for frame, members in sorted(by_frame.items()):
        parent = labels[frame]
        trial = candidate[frame].copy()
        component_groups: dict[bytes, dict] = {}
        ownerless = []
        for member in members:
            point = member["point"]
            observed = int(point.candidate_owner)
            desired = int(member["anchor_owner"])
            if observed <= 0:
                if member["role"] == "anchor":
                    ownerless.append(member)
                continue
            mask = _component_near(parent, observed, point, maximum_distance)
            if mask is None:
                continue
            key = np.packbits(mask, axis=None).tobytes()
            entry = component_groups.setdefault(
                key, {"mask": mask, "observed": observed, "members": []})
            entry["members"].append(member)

        for entry in component_groups.values():
            mask = entry["mask"]
            relevant = []
            by_desired: dict[int, dict] = {}
            for member in entry["members"]:
                by_desired.setdefault(int(member["anchor_owner"]), member)
            relevant.extend(by_desired.values())
            if len(relevant) == 1:
                trial[mask] = int(relevant[0]["anchor_owner"])
                operation = "restore_component_seat"
            else:
                trial[mask] = 0
                for owner, portion in _partition(mask, relevant).items():
                    trial[portion] = int(owner)
                operation = "partition_merged_component_by_seats"
            changed = int(np.count_nonzero(trial != candidate[frame]))
            for member in relevant:
                frame_rows.append({
                    "proposal_id": member["proposal_id"], "frame": frame,
                    "track_id": member["track_id"],
                    "anchor_owner": member["anchor_owner"],
                    "observed_owner": int(entry["observed"]),
                    "operation": operation, "changed_pixels": changed,
                    "raw_core_area_radius_ratio": float("nan"),
                })

        # A duplicate physical reference can describe the same raw body.  Add
        # at most one non-overlapping core per established owner and frame.
        for member in sorted(ownerless, key=lambda item: (
                -float(item["point"].radius_px), item["track_id"])):
            point = member["point"]
            desired = int(member["anchor_owner"])
            evidence = evidence_image(raw[frame], params)
            found = _nearest_raw_core(
                raw[frame], evidence, point,
                float(threshold_by_frame.loc[frame, "weak_threshold"]), params)
            if found is None:
                continue
            core, region, details = found
            ratio = float(details["core_area_radius_ratio"])
            if not minimum_raw_ratio <= ratio <= maximum_raw_ratio:
                continue
            occupied = ((trial[region] > 0) | (unclaimed[frame][region] > 0))
            if np.any(core & occupied):
                continue
            before = trial.copy()
            trial[region][core] = desired
            frame_rows.append({
                "proposal_id": member["proposal_id"], "frame": frame,
                "track_id": member["track_id"], "anchor_owner": desired,
                "observed_owner": 0, "operation": "recover_ownerless_raw_core",
                "changed_pixels": int(np.count_nonzero(trial != before)),
                "raw_core_area_radius_ratio": ratio,
            })
        candidate[frame] = trial

    changed = candidate != labels
    additions = changed & (labels == 0) & (candidate > 0)
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    details = {
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_added_pixels": int(additions.sum()),
        "foreground_removed_pixels": int(np.count_nonzero(
            changed & (labels > 0) & (candidate == 0))),
        "unclaimed_overlap_pixels": int(np.count_nonzero(
            (candidate > 0) & (unclaimed > 0))),
        "zero_signal_additions": int(np.count_nonzero(additions & (raw == 0))),
        "new_identity_count": int(len(after_ids - before_ids)),
        "removed_identity_count": int(len(before_ids - after_ids)),
        "new_duplicate_components": int(
            component_accounting.new_duplicate_components(labels, candidate)),
        "component_excess_before": int(
            component_accounting.total_component_excess(labels)),
        "component_excess_after": int(
            component_accounting.total_component_excess(candidate)),
        "duplicate_owner_frames_before": int(
            component_accounting.duplicate_owner_frames(labels)),
        "duplicate_owner_frames_after": int(
            component_accounting.duplicate_owner_frames(candidate)),
    }
    return candidate, pd.DataFrame(frame_rows, columns=FRAME_COLUMNS), details


def produce(labels: np.ndarray, unclaimed: np.ndarray, raw: np.ndarray,
            points: pd.DataFrame, thresholds: pd.DataFrame, params: dict):
    assert_target_free(params)
    audit, seats, _ = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
        details = {key: 0 for key in (
            "changed_pixels", "changed_frames", "foreground_added_pixels",
            "foreground_removed_pixels", "unclaimed_overlap_pixels",
            "zero_signal_additions", "new_identity_count",
            "removed_identity_count", "new_duplicate_components",
            "component_excess_before", "component_excess_after",
            "duplicate_owner_frames_before", "duplicate_owner_frames_after")}
    else:
        candidate, frames, details = apply(
            labels, unclaimed, raw, points, thresholds, seats, params)
    if details["foreground_removed_pixels"]:
        raise AssertionError("seat-cycle recovery removed foreground")
    if details["unclaimed_overlap_pixels"] or details["zero_signal_additions"]:
        raise AssertionError("seat-cycle recovery violated source ledgers")
    if details["new_identity_count"] or details["removed_identity_count"]:
        raise AssertionError("seat-cycle recovery changed the movie identity set")
    if details["component_excess_after"] > details["component_excess_before"]:
        raise AssertionError("seat-cycle recovery increased component excess")
    if (details["duplicate_owner_frames_after"]
            > details["duplicate_owner_frames_before"]):
        raise AssertionError("seat-cycle recovery increased duplicate frames")
    applied = bool(details["changed_pixels"])
    audit = audit.copy()
    if len(audit):
        audit["applied"] = audit.eligible.astype(bool) & applied
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "seat_members": int(len(seats)), "applied": applied,
        **details,
    }
    return candidate, unclaimed.copy(), audit, seats, frames, metrics


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])[:len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    thresholds = pd.read_csv(params["thresholds_path"])
    candidate, candidate_unclaimed, audit, seats, frames, metrics = produce(
        labels, unclaimed, raw, points, thresholds, params)
    output_dir = Path(out.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = output_dir / f"{stem}.tif"
    unclaimed_path = output_dir / f"{stem}_unclaimed_original_ids.tif"
    if np.array_equal(candidate, labels):
        shutil.copy2(params["labels_path"], labels_path)
    else:
        tifffile.imwrite(
            labels_path, candidate, imagej=True, compression="zlib",
            metadata={"axes": "TYX", "finterval": 1800.0,
                      "tunit": "sec", "unit": "pixel"})
    shutil.copy2(params["unclaimed_path"], unclaimed_path)
    outputs = {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": output_dir / "established_seat_cycle_audit.csv",
        "seats": output_dir / "established_seat_cycle_members.csv",
        "frames": output_dir / "established_seat_cycle_frames.csv",
        "metrics": output_dir / "metrics.json",
    }
    audit.to_csv(outputs["audit"], index=False)
    seats.to_csv(outputs["seats"], index=False)
    frames.to_csv(outputs["frames"], index=False)
    outputs["metrics"].write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {"outputs": outputs, "summary": metrics}
