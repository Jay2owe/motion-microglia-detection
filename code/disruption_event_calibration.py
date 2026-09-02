"""Target-free calibration helpers for disruptive identity-event catalogues."""
from __future__ import annotations

from itertools import combinations
import re

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


_EIGHT_CONNECTED = np.ones((3, 3), np.uint8)

_SINGLE_CORE_FORBIDDEN_TARGETS = (
    "identity_ids", "track_ids", "frame_ids", "coordinates",
    "event_ids", "failure_targets",
)


def _owners(value: object) -> set[int]:
    if pd.isna(value):
        return set()
    return {int(item) for item in re.findall(r"\d+", str(value))}


def _tracks(rows: pd.DataFrame) -> set[int]:
    refs = "|".join(rows.get("source_ref", pd.Series(dtype=str)).astype(str))
    return {int(item) for item in re.findall(r"T(\d+)", refs)}


def _owner_run_length(points: pd.DataFrame, track: int, owner: int,
                      first_frame: int) -> int:
    group = points[(points.track_id.astype(int) == int(track)) &
                   (points.frame.astype(int) >= int(first_frame))].sort_values("frame")
    count = 0
    expected = int(first_frame)
    for row in group.itertuples(index=False):
        if int(row.frame) != expected or int(row.candidate_owner) != int(owner):
            break
        count += 1
        expected += 1
    return count


def _unique_continuous_successor(points: pd.DataFrame, predecessor: pd.Series,
                                 owner: int, first_frame: int,
                                 source_tracks: set[int],
                                 maximum_frame_gap: int,
                                 maximum_distance_sum_radii: float,
                                 minimum_successor_owner_frames: int,
                                 minimum_owner_purity: float) -> dict | None:
    candidates: list[dict] = []
    predecessor_frame = int(predecessor.frame)
    predecessor_radius = float(predecessor.radius_px)
    for gap in range(1, maximum_frame_gap + 2):
        frame = predecessor_frame + gap
        rows = points[(points.frame.astype(int) == frame) &
                      (points.candidate_owner.astype(int) == int(owner))]
        for row in rows.itertuples(index=False):
            track = int(row.track_id)
            if track in source_tracks:
                continue
            track_rows = points[points.track_id.astype(int) == track]
            if int(track_rows.frame.min()) != frame:
                continue
            purity = float(getattr(row, "candidate_owner_purity", 1.0))
            if purity < minimum_owner_purity:
                continue
            support = _owner_run_length(points, track, owner, frame)
            if support < minimum_successor_owner_frames:
                continue
            distance = float(np.hypot(float(row.x) - float(predecessor.x),
                                      float(row.y) - float(predecessor.y)))
            radius_sum = predecessor_radius + float(row.radius_px)
            if distance > maximum_distance_sum_radii * radius_sum:
                continue
            candidates.append({
                "owner": int(owner),
                "predecessor_track": int(predecessor.track_id),
                "successor_track": track,
                "predecessor_frame": predecessor_frame,
                "successor_frame": frame,
                "gap_frames": gap - 1,
                "distance_px": distance,
                "radius_sum_px": radius_sum,
                "distance_sum_radii": distance / max(radius_sum, 1e-9),
                "successor_owner_frames": support,
                "successor_owner_purity": purity,
            })
    if len(candidates) != 1:
        return None
    return candidates[0]


def conserved_handoff_merge_losses(
        events: pd.DataFrame, members: pd.DataFrame, points: pd.DataFrame,
        maximum_frame_gap: int = 1,
        maximum_distance_sum_radii: float = 1.0,
        minimum_successor_owner_frames: int = 2,
        minimum_owner_purity: float = 0.8,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove pure merge-loss events whose supposedly lost owner continues.

    Suppression is conservative: every source marker in the collapsed event must be
    a merge/split loss, every lost owner must have exactly one new raw-track successor,
    and that successor must start immediately, remain owner-pure, and lie within a
    data-relative sum-of-radii distance. Explicit tracks, identities, frames, or
    coordinates are never accepted as parameters.
    """
    if events.empty or members.empty:
        return events.copy(), pd.DataFrame()
    filtered = events.copy()
    audit_rows: list[dict] = []
    suppress_ids: set[str] = set()
    for event_id, event_members in members.groupby("event_id"):
        source_rows = event_members.drop_duplicates("source_id")
        if source_rows.empty or not source_rows.source_family.eq(
                "merge_split_identity_loss").all():
            continue
        event_matches: list[dict] = []
        eligible = True
        for source_id, rows in event_members.groupby("source_id"):
            source = rows.iloc[0]
            lost = _owners(source.source_accepted_before) - _owners(
                source.source_accepted_after)
            source_tracks = _tracks(rows)
            if not lost or not source_tracks:
                eligible = False
                break
            last_frame = int(source.source_last_frame)
            for owner in sorted(lost):
                predecessors = points[
                    (points.track_id.astype(int).isin(source_tracks)) &
                    (points.frame.astype(int) == last_frame) &
                    (points.candidate_owner.astype(int) == owner)]
                if len(predecessors) != 1:
                    eligible = False
                    break
                match = _unique_continuous_successor(
                    points, predecessors.iloc[0], owner, last_frame + 1,
                    source_tracks, maximum_frame_gap,
                    maximum_distance_sum_radii,
                    minimum_successor_owner_frames, minimum_owner_purity)
                if match is None:
                    eligible = False
                    break
                match.update({"event_id": str(event_id),
                              "source_id": int(source_id)})
                event_matches.append(match)
            if not eligible:
                break
        if eligible and event_matches:
            suppress_ids.add(str(event_id))
            audit_rows.extend(event_matches)
    if suppress_ids:
        filtered = filtered[~filtered.event_id.astype(str).isin(suppress_ids)].copy()
        filtered = filtered.sort_values(
            ["disruption_score", "impact", "first_frame"],
            ascending=[False, False, True]).reset_index(drop=True)
        filtered["impact_rank"] = np.arange(1, len(filtered) + 1)
    return filtered, pd.DataFrame(audit_rows)


def _point(points: pd.DataFrame, track: int, frame: int,
           owner: int) -> pd.Series | None:
    rows = points[
        (points.track_id.astype(int) == int(track))
        & (points.frame.astype(int) == int(frame))
        & (points.candidate_owner.astype(int) == int(owner))]
    return None if len(rows) != 1 else rows.iloc[0]


def _scaled_distance(left: pd.Series, right: pd.Series) -> float:
    distance = float(np.hypot(float(left.x) - float(right.x),
                              float(left.y) - float(right.y)))
    radius_sum = float(left.radius_px) + float(right.radius_px)
    return distance / max(radius_sum, 1e-9)


def reciprocal_owner_handoff_events(
        events: pd.DataFrame, members: pd.DataFrame, points: pd.DataFrame,
        maximum_exchange_distance_sum_radii: float = 1.5,
        maximum_successor_frame_gap: int = 1,
        maximum_successor_distance_sum_radii: float = 2.0,
        minimum_successor_owner_frames: int = 2,
        minimum_owner_purity: float = 0.8,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove mixed alerts caused by a reciprocal two-owner reference handoff.

    A qualifying event contains one apparent takeover and one merge-loss source.
    Across a single frame boundary, the takeover track and exactly one merge track
    exchange their two owners while all other merge tracks retain their owners.
    Both exchanges must be spatially continuous and owner-pure. The durable owner
    must continue on the takeover track, and the transient owner must have exactly
    one nearby continuing raw-track successor. No named identities, tracks, frames,
    coordinates, events, or review cases are accepted as parameters.
    """
    if events.empty or members.empty:
        return events.copy(), pd.DataFrame()
    suppress_ids: set[str] = set()
    audit_rows: list[dict] = []
    for event_id, event_members in members.groupby("event_id"):
        sources = event_members.drop_duplicates("source_id")
        takeover = sources[
            sources.source_family.eq("identity_swap_or_takeover")]
        merge = sources[
            sources.source_family.eq("merge_split_identity_loss")]
        if len(takeover) != 1 or len(merge) != 1 or len(sources) != 2:
            continue
        takeover_source = takeover.iloc[0]
        merge_source = merge.iloc[0]
        takeover_tracks = _tracks(event_members[
            event_members.source_id == takeover_source.source_id])
        merge_tracks = _tracks(event_members[
            event_members.source_id == merge_source.source_id])
        old_owners = _owners(takeover_source.source_accepted_before)
        new_owners = _owners(takeover_source.source_accepted_after)
        lost_owners = (_owners(merge_source.source_accepted_before)
                       - _owners(merge_source.source_accepted_after))
        if (len(takeover_tracks) != 1 or len(old_owners) != 1
                or len(new_owners) != 1 or len(lost_owners) != 1):
            continue
        old_owner = next(iter(old_owners))
        new_owner = next(iter(new_owners))
        if old_owner == new_owner or lost_owners != {new_owner}:
            continue
        first = int(takeover_source.source_first_frame)
        last = int(takeover_source.source_last_frame)
        if last != first + 1:
            continue
        takeover_track = next(iter(takeover_tracks))
        takeover_before = _point(points, takeover_track, first, old_owner)
        takeover_after = _point(points, takeover_track, last, new_owner)
        if takeover_before is None or takeover_after is None:
            continue
        partner_options: list[tuple[int, pd.Series, pd.Series]] = []
        for track in sorted(merge_tracks - {takeover_track}):
            partner_before = _point(points, track, first, new_owner)
            partner_after = _point(points, track, last, old_owner)
            if partner_before is not None and partner_after is not None:
                partner_options.append((track, partner_before, partner_after))
        if len(partner_options) != 1:
            continue
        partner_track, partner_before, partner_after = partner_options[0]
        purity_rows = (takeover_before, takeover_after,
                       partner_before, partner_after)
        if any(float(getattr(row, "candidate_owner_purity", 1.0))
               < minimum_owner_purity for row in purity_rows):
            continue
        new_owner_distance = _scaled_distance(
            partner_before, takeover_after)
        old_owner_distance = _scaled_distance(
            takeover_before, partner_after)
        if (new_owner_distance > maximum_exchange_distance_sum_radii
                or old_owner_distance > maximum_exchange_distance_sum_radii):
            continue
        if _owner_run_length(
                points, takeover_track, new_owner, last) \
                < minimum_successor_owner_frames:
            continue
        old_successor = _unique_continuous_successor(
            points, partner_after, old_owner, last + 1,
            {takeover_track, partner_track}, maximum_successor_frame_gap,
            maximum_successor_distance_sum_radii,
            minimum_successor_owner_frames, minimum_owner_purity)
        if old_successor is None:
            continue
        stable_other_tracks = True
        stable_owners: list[int] = []
        for track in sorted(merge_tracks - {partner_track}):
            before = points[
                (points.track_id.astype(int) == track)
                & (points.frame.astype(int) == first)]
            after = points[
                (points.track_id.astype(int) == track)
                & (points.frame.astype(int) == last)]
            if len(before) != 1 or len(after) != 1:
                stable_other_tracks = False
                break
            before_owner = int(before.iloc[0].candidate_owner)
            after_owner = int(after.iloc[0].candidate_owner)
            if before_owner <= 0 or before_owner != after_owner:
                stable_other_tracks = False
                break
            stable_owners.append(before_owner)
        if not stable_other_tracks:
            continue
        suppress_ids.add(str(event_id))
        audit_rows.append({
            "event_id": str(event_id),
            "takeover_track": takeover_track,
            "partner_track": partner_track,
            "old_owner": old_owner,
            "new_owner": new_owner,
            "first_frame": first,
            "last_frame": last,
            "old_owner_successor_track": int(
                old_successor["successor_track"]),
            "old_owner_successor_frames": int(
                old_successor["successor_owner_frames"]),
            "old_owner_exchange_distance_sum_radii": old_owner_distance,
            "new_owner_exchange_distance_sum_radii": new_owner_distance,
            "stable_other_tracks": len(stable_owners),
        })
    filtered = events.copy()
    if suppress_ids:
        filtered = filtered[
            ~filtered.event_id.astype(str).isin(suppress_ids)].copy()
        filtered = filtered.sort_values(
            ["disruption_score", "impact", "first_frame"],
            ascending=[False, False, True]).reset_index(drop=True)
        filtered["impact_rank"] = np.arange(1, len(filtered) + 1)
    return filtered, pd.DataFrame(audit_rows)


def _owner_runs(points: pd.DataFrame, track: int, owner: int) -> list[dict]:
    rows = points[
        (points.track_id.astype(int) == int(track))
        & (points.candidate_owner.astype(int) == int(owner))].sort_values("frame")
    runs: list[dict] = []
    for row in rows.itertuples(index=False):
        frame = int(row.frame)
        if runs and frame == runs[-1]["last"] + 1:
            runs[-1]["last"] = frame
            runs[-1]["rows"].append(row)
        else:
            runs.append({"first": frame, "last": frame, "rows": [row]})
    for run in runs:
        run["frames"] = len(run["rows"])
    return runs


def _overlap_event_run(points: pd.DataFrame, track: int, owner: int,
                       first_frame: int, last_frame: int) -> dict | None:
    matches = [
        run for run in _owner_runs(points, track, owner)
        if run["first"] <= last_frame and run["last"] >= first_frame
    ]
    return matches[0] if len(matches) == 1 else None


def overlapping_same_owner_handoffs(
        events: pd.DataFrame, members: pd.DataFrame, points: pd.DataFrame,
        labels: np.ndarray, maximum_overlap_frames: int = 2,
        minimum_predecessor_owner_frames: int = 5,
        minimum_successor_owner_frames: int = 5,
        maximum_distance_sum_radii: float = 1.0,
        minimum_owner_purity: float = 0.8,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove false merge alerts from overlapping same-owner references.

    A qualifying source has exactly two raw tracks and one unchanged owner. Their
    long owner runs must form a directed predecessor/successor chain whose complete
    overlap is short. At every overlap frame, both owner-pure points must be within
    one data-relative sum-of-radii distance and inside the same connected owner
    component, with no third claimant in that component. Explicit identities,
    tracks, frames, coordinates, events, or review cases are never parameters.
    """
    if events.empty or members.empty:
        return events.copy(), pd.DataFrame()
    suppress_ids: set[str] = set()
    audit_rows: list[dict] = []
    for event_id, event_members in members.groupby("event_id"):
        sources = event_members.drop_duplicates("source_id")
        if (len(sources) != 1
                or not sources.source_family.eq("separable_cells_merged").all()):
            continue
        source = sources.iloc[0]
        before = _owners(source.source_accepted_before)
        after = _owners(source.source_accepted_after)
        tracks = sorted(_tracks(event_members))
        if before != after or len(before) != 1 or len(tracks) != 2:
            continue
        owner = next(iter(before))
        first_frame = int(source.source_first_frame)
        last_frame = int(source.source_last_frame)
        left = _overlap_event_run(
            points, tracks[0], owner, first_frame, last_frame)
        right = _overlap_event_run(
            points, tracks[1], owner, first_frame, last_frame)
        if left is None or right is None:
            continue
        oriented = None
        for predecessor_track, predecessor, successor_track, successor in (
                (tracks[0], left, tracks[1], right),
                (tracks[1], right, tracks[0], left)):
            if (predecessor["first"] < successor["first"]
                    and predecessor["last"] < successor["last"]):
                oriented = (predecessor_track, predecessor,
                            successor_track, successor)
                break
        if oriented is None:
            continue
        predecessor_track, predecessor, successor_track, successor = oriented
        overlap_first = max(predecessor["first"], successor["first"])
        overlap_last = min(predecessor["last"], successor["last"])
        overlap_frames = overlap_last - overlap_first + 1
        if (overlap_frames < 1 or overlap_frames > maximum_overlap_frames
                or overlap_first != first_frame or overlap_last != last_frame
                or predecessor["frames"] < minimum_predecessor_owner_frames
                or successor["frames"] < minimum_successor_owner_frames):
            continue
        maximum_scaled_distance = 0.0
        valid = True
        for frame in range(overlap_first, overlap_last + 1):
            predecessor_point = _point(
                points, predecessor_track, frame, owner)
            successor_point = _point(points, successor_track, frame, owner)
            if predecessor_point is None or successor_point is None:
                valid = False
                break
            if any(float(getattr(point, "candidate_owner_purity", 1.0))
                   < minimum_owner_purity
                   for point in (predecessor_point, successor_point)):
                valid = False
                break
            scaled_distance = _scaled_distance(
                predecessor_point, successor_point)
            maximum_scaled_distance = max(
                maximum_scaled_distance, scaled_distance)
            if scaled_distance > maximum_distance_sum_radii:
                valid = False
                break
            components, _ = ndi.label(
                labels[frame] == owner, _EIGHT_CONNECTED)
            component_ids: list[int] = []
            for point in (predecessor_point, successor_point):
                y = int(np.clip(round(float(point.y)), 0, labels.shape[1] - 1))
                x = int(np.clip(round(float(point.x)), 0, labels.shape[2] - 1))
                component_ids.append(int(components[y, x]))
            if component_ids[0] <= 0 or component_ids[0] != component_ids[1]:
                valid = False
                break
            other_claimants = points[
                (points.frame.astype(int) == frame)
                & (points.candidate_owner.astype(int) == owner)
                & (~points.track_id.astype(int).isin(tracks))]
            for point in other_claimants.itertuples(index=False):
                y = int(np.clip(round(float(point.y)), 0, labels.shape[1] - 1))
                x = int(np.clip(round(float(point.x)), 0, labels.shape[2] - 1))
                if int(components[y, x]) == component_ids[0]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        suppress_ids.add(str(event_id))
        audit_rows.append({
            "event_id": str(event_id),
            "owner": owner,
            "predecessor_track": predecessor_track,
            "successor_track": successor_track,
            "overlap_first_frame": overlap_first,
            "overlap_last_frame": overlap_last,
            "overlap_frames": overlap_frames,
            "predecessor_owner_frames": predecessor["frames"],
            "successor_owner_frames": successor["frames"],
            "maximum_distance_sum_radii": maximum_scaled_distance,
        })
    filtered = events.copy()
    if suppress_ids:
        filtered = filtered[
            ~filtered.event_id.astype(str).isin(suppress_ids)].copy()
        filtered = filtered.sort_values(
            ["disruption_score", "impact", "first_frame"],
            ascending=[False, False, True]).reset_index(drop=True)
        filtered["impact_rank"] = np.arange(1, len(filtered) + 1)
    return filtered, pd.DataFrame(audit_rows)


def resolved_post_split_forks(
        events: pd.DataFrame, members: pd.DataFrame, points: pd.DataFrame,
        labels: np.ndarray, minimum_branch_owner_frames: int = 5,
        minimum_donor_owner_frames: int = 8,
        minimum_coexistence_frames: int = 5,
        maximum_split_distance_sum_radii: float = 1.5,
        minimum_owner_purity: float = 0.8,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove takeover alerts that are fully resolved physical forks.

    A qualifying event is one owner transition on one continuous raw track. The new
    owner must be born exactly on the transition frame and remain durable on that
    branch. Exactly one nearby raw companion must also be born on that frame, retain
    the old owner durably, and coexist with the new-owner branch. This describes a
    resolved two-body split rather than an identity takeover. Explicit identities,
    tracks, frames, coordinates, events, or review cases are never parameters.
    """
    if events.empty or members.empty:
        return events.copy(), pd.DataFrame()
    suppress_ids: set[str] = set()
    audit_rows: list[dict] = []
    active_labels = set(map(int, np.unique(labels))) - {0}
    first_claim = {
        owner: int(np.flatnonzero(np.any(
            labels == owner, axis=(1, 2)))[0]) for owner in active_labels}
    for event_id, event_members in members.groupby("event_id"):
        sources = event_members.drop_duplicates("source_id")
        if (len(sources) != 1
                or not sources.source_family.eq(
                    "identity_swap_or_takeover").all()):
            continue
        source = sources.iloc[0]
        tracks = _tracks(event_members)
        before = _owners(source.source_accepted_before)
        after = _owners(source.source_accepted_after)
        first = int(source.source_first_frame)
        last = int(source.source_last_frame)
        if (len(tracks) != 1 or len(before) != 1 or len(after) != 1
                or before == after or last != first + 1):
            continue
        branch_track = next(iter(tracks))
        old_owner = next(iter(before))
        new_owner = next(iter(after))
        if first_claim.get(new_owner) != last:
            continue
        branch_before = _point(points, branch_track, first, old_owner)
        branch_after = _point(points, branch_track, last, new_owner)
        if branch_before is None or branch_after is None:
            continue
        if any(float(getattr(point, "candidate_owner_purity", 1.0))
               < minimum_owner_purity
               for point in (branch_before, branch_after)):
            continue
        branch_support = _owner_run_length(
            points, branch_track, new_owner, last)
        if branch_support < minimum_branch_owner_frames:
            continue
        donor_options: list[dict] = []
        frame_rows = points[
            (points.frame.astype(int) == last)
            & (points.candidate_owner.astype(int) == old_owner)]
        for row in frame_rows.itertuples(index=False):
            donor_track = int(row.track_id)
            if donor_track == branch_track:
                continue
            donor_rows = points[points.track_id.astype(int) == donor_track]
            if int(donor_rows.frame.min()) != last:
                continue
            purity = float(getattr(row, "candidate_owner_purity", 1.0))
            if purity < minimum_owner_purity:
                continue
            donor_support = _owner_run_length(
                points, donor_track, old_owner, last)
            coexistence = min(branch_support, donor_support)
            if (donor_support < minimum_donor_owner_frames
                    or coexistence < minimum_coexistence_frames):
                continue
            distance = _scaled_distance(branch_after, pd.Series(row._asdict()))
            if distance > maximum_split_distance_sum_radii:
                continue
            continuous = True
            for frame in range(last, last + coexistence):
                if (_point(points, branch_track, frame, new_owner) is None
                        or _point(points, donor_track, frame, old_owner) is None):
                    continuous = False
                    break
            if not continuous:
                continue
            donor_options.append({
                "donor_track": donor_track,
                "donor_owner_frames": donor_support,
                "coexistence_frames": coexistence,
                "split_distance_sum_radii": distance,
                "donor_owner_purity": purity,
            })
        if len(donor_options) != 1:
            continue
        donor = donor_options[0]
        suppress_ids.add(str(event_id))
        audit_rows.append({
            "event_id": str(event_id),
            "branch_track": branch_track,
            "donor_track": int(donor["donor_track"]),
            "old_owner": old_owner,
            "new_owner": new_owner,
            "transition_first_frame": first,
            "split_frame": last,
            "branch_owner_frames": branch_support,
            **donor,
        })
    filtered = events.copy()
    if suppress_ids:
        filtered = filtered[
            ~filtered.event_id.astype(str).isin(suppress_ids)].copy()
        filtered = filtered.sort_values(
            ["disruption_score", "impact", "first_frame"],
            ascending=[False, False, True]).reset_index(drop=True)
        filtered["impact_rank"] = np.arange(1, len(filtered) + 1)
    return filtered, pd.DataFrame(audit_rows)


def _nearby_identity_component(
        labels: np.ndarray, frame: int, owner: int, x: float, y: float,
        maximum_distance_sum_radii: float,
        ) -> dict | None:
    """Return one uniquely nearest owner component in data-relative units."""
    components, count = ndi.label(
        labels[int(frame)] == int(owner), _EIGHT_CONNECTED)
    if not count:
        return None
    candidates: list[dict] = []
    for component in range(1, count + 1):
        mask = components == component
        area = int(np.count_nonzero(mask))
        if not area:
            continue
        centre_y, centre_x = ndi.center_of_mass(mask)
        radius = float(np.sqrt(area / np.pi))
        distance = float(np.hypot(float(centre_x) - float(x),
                                  float(centre_y) - float(y)))
        scaled = distance / max(radius, 1.0)
        if scaled <= maximum_distance_sum_radii:
            candidates.append({
                "component": component,
                "area_px": area,
                "x": float(centre_x),
                "y": float(centre_y),
                "radius_px": radius,
                "distance_px": distance,
                "distance_component_radii": scaled,
            })
    candidates.sort(key=lambda row: (
        row["distance_component_radii"], row["distance_px"],
        -row["area_px"], row["component"]))
    if not candidates:
        return None
    if (len(candidates) > 1
            and abs(candidates[0]["distance_component_radii"]
                    - candidates[1]["distance_component_radii"]) < 0.1):
        return None
    return candidates[0]


def _component_step(left: dict, right: dict) -> float:
    distance = float(np.hypot(
        float(left["x"]) - float(right["x"]),
        float(left["y"]) - float(right["y"])))
    return distance / max(
        float(left["radius_px"]) + float(right["radius_px"]), 1.0)


def _identity_gap_continuity(
        labels: np.ndarray, owner: int, event_frame: int, x: float, y: float,
        maximum_missing_frames: int,
        maximum_distance_per_frame_sum_radii: float,
        maximum_component_distance_radii: float,
        ) -> dict | None:
    """Find one spatially continuous owner path bracketing an alert frame."""
    present = np.flatnonzero(np.any(labels == int(owner), axis=(1, 2)))
    before = present[present <= int(event_frame)]
    after = present[present >= int(event_frame) + 1]
    if not len(before) or not len(after):
        return None
    left_frame = int(before[-1])
    right_frame = int(after[0])
    missing = right_frame - left_frame - 1
    if missing > int(maximum_missing_frames):
        return None
    left = _nearby_identity_component(
        labels, left_frame, owner, x, y,
        maximum_component_distance_radii)
    right = _nearby_identity_component(
        labels, right_frame, owner, x, y,
        maximum_component_distance_radii)
    if left is None or right is None:
        return None
    step = _component_step(left, right)
    per_frame = step / max(right_frame - left_frame, 1)
    if per_frame > maximum_distance_per_frame_sum_radii:
        return None
    return {
        "owner": int(owner),
        "left_frame": left_frame,
        "right_frame": right_frame,
        "missing_frames": missing,
        "step_sum_radii": step,
        "step_per_frame_sum_radii": per_frame,
    }


def conserved_reference_fork_events(
        events: pd.DataFrame, members: pd.DataFrame, points: pd.DataFrame,
        labels: np.ndarray,
        minimum_new_owner_run_frames: int = 3,
        minimum_old_owner_companion_frames: int = 3,
        maximum_transition_component_step_sum_radii: float = 2.0,
        maximum_component_distance_radii: float = 6.0,
        maximum_lost_owner_missing_frames: int = 5,
        maximum_lost_owner_step_per_frame_sum_radii: float = 0.75,
        minimum_owner_purity: float = 0.7,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Suppress mixed alerts caused by a raw reference forking between cells.

    A qualifying event contains exactly one apparent takeover and one later merge
    loss.  At the takeover boundary both accepted owners already have distinct,
    spatially continuous mask components: the raw reference follows the new owner
    while the old owner continues on one unique companion reference.  Every owner
    reported lost by the later merge marker must also have a unique, spatially
    continuous accepted-mask path, allowing only a short evidence gap.  Thus the
    accepted identities are conserved and only the detector reference graph forks.

    The function accepts no identity, track, frame, coordinate, event, or review
    targets; all candidates are discovered from the complete event catalogue.
    """
    audit_columns = [
        "event_id", "status", "takeover_track", "old_owner", "new_owner",
        "transition_first_frame", "transition_last_frame", "merge_frame",
        "old_owner_companion_track", "old_owner_companion_frames",
        "old_owner_component_step_sum_radii",
        "new_owner_component_step_sum_radii", "lost_owners",
        "maximum_lost_owner_missing_frames",
        "maximum_lost_owner_step_per_frame_sum_radii",
    ]
    if events.empty or members.empty:
        return events.copy(), pd.DataFrame(columns=audit_columns)
    suppress_ids: set[str] = set()
    audits: list[dict] = []
    for event_id, event_members in members.groupby("event_id", sort=True):
        sources = event_members.drop_duplicates("source_id")
        takeover = sources[
            sources.source_family.eq("identity_swap_or_takeover")]
        merge = sources[
            sources.source_family.eq("merge_split_identity_loss")]
        if len(takeover) != 1 or len(merge) != 1 or len(sources) != 2:
            continue
        takeover_source = takeover.iloc[0]
        merge_source = merge.iloc[0]
        takeover_tracks = _tracks(event_members[
            event_members.source_id == takeover_source.source_id])
        old_owners = _owners(takeover_source.source_accepted_before)
        new_owners = _owners(takeover_source.source_accepted_after)
        if (len(takeover_tracks) != 1 or len(old_owners) != 1
                or len(new_owners) != 1 or old_owners == new_owners):
            continue
        first = int(takeover_source.source_first_frame)
        last = int(takeover_source.source_last_frame)
        merge_frame = int(merge_source.source_last_frame)
        if last != first + 1 or merge_frame < last:
            continue
        track = next(iter(takeover_tracks))
        old_owner = next(iter(old_owners))
        new_owner = next(iter(new_owners))
        left_point = _point(points, track, first, old_owner)
        right_point = _point(points, track, last, new_owner)
        if left_point is None or right_point is None:
            continue
        if any(float(getattr(point, "candidate_owner_purity", 1.0))
               < minimum_owner_purity for point in (left_point, right_point)):
            continue
        if _owner_run_length(points, track, new_owner, last) \
                < minimum_new_owner_run_frames:
            continue

        old_left = _nearby_identity_component(
            labels, first, old_owner, float(left_point.x), float(left_point.y),
            maximum_component_distance_radii)
        old_right = _nearby_identity_component(
            labels, last, old_owner, float(right_point.x), float(right_point.y),
            maximum_component_distance_radii)
        new_left = _nearby_identity_component(
            labels, first, new_owner, float(left_point.x), float(left_point.y),
            maximum_component_distance_radii)
        new_right = _nearby_identity_component(
            labels, last, new_owner, float(right_point.x), float(right_point.y),
            maximum_component_distance_radii)
        if any(item is None for item in (
                old_left, old_right, new_left, new_right)):
            continue
        old_step = _component_step(old_left, old_right)
        new_step = _component_step(new_left, new_right)
        if (old_step > maximum_transition_component_step_sum_radii
                or new_step > maximum_transition_component_step_sum_radii):
            continue

        companion_options: list[dict] = []
        for row in points[
                (points.frame.astype(int) == last)
                & (points.candidate_owner.astype(int) == old_owner)
                & (points.track_id.astype(int) != track)].itertuples(index=False):
            if float(getattr(row, "candidate_owner_purity", 1.0)) \
                    < minimum_owner_purity:
                continue
            support = _owner_run_length(
                points, int(row.track_id), old_owner, last)
            if support < minimum_old_owner_companion_frames:
                continue
            component = _nearby_identity_component(
                labels, last, old_owner, float(row.x), float(row.y),
                maximum_component_distance_radii)
            if component is None or int(component["component"]) != int(
                    old_right["component"]):
                continue
            companion_options.append({
                "track": int(row.track_id), "frames": support,
                "purity": float(getattr(
                    row, "candidate_owner_purity", 1.0)),
            })
        if len(companion_options) != 1:
            continue

        merge_before = _owners(merge_source.source_accepted_before)
        merge_after = _owners(merge_source.source_accepted_after)
        lost = merge_before - merge_after
        if old_owner not in lost or new_owner not in merge_after or not lost:
            continue
        continuity: list[dict] = []
        valid = True
        for owner in sorted(lost):
            result = _identity_gap_continuity(
                labels, owner, merge_frame,
                float(merge_source.source_x), float(merge_source.source_y),
                maximum_lost_owner_missing_frames,
                maximum_lost_owner_step_per_frame_sum_radii,
                maximum_component_distance_radii)
            if result is None:
                valid = False
                break
            continuity.append(result)
        if not valid:
            continue
        suppress_ids.add(str(event_id))
        audits.append({
            "event_id": str(event_id),
            "status": "suppressed",
            "takeover_track": track,
            "old_owner": old_owner,
            "new_owner": new_owner,
            "transition_first_frame": first,
            "transition_last_frame": last,
            "merge_frame": merge_frame,
            "old_owner_companion_track": companion_options[0]["track"],
            "old_owner_companion_frames": companion_options[0]["frames"],
            "old_owner_component_step_sum_radii": old_step,
            "new_owner_component_step_sum_radii": new_step,
            "lost_owners": "|".join(map(str, sorted(lost))),
            "maximum_lost_owner_missing_frames": max(
                row["missing_frames"] for row in continuity),
            "maximum_lost_owner_step_per_frame_sum_radii": max(
                row["step_per_frame_sum_radii"] for row in continuity),
        })
    filtered = events.copy()
    if suppress_ids:
        filtered = filtered[
            ~filtered.event_id.astype(str).isin(suppress_ids)].copy()
        filtered = filtered.sort_values(
            ["disruption_score", "impact", "first_frame"],
            ascending=[False, False, True]).reset_index(drop=True)
        filtered["impact_rank"] = np.arange(1, len(filtered) + 1)
    return filtered, pd.DataFrame(audits, columns=audit_columns)


def assert_single_core_reference_overlap_target_free(params: dict) -> None:
    """Reject any attempt to turn the field-wide calibration into a target list."""
    supplied = [
        name for name in _SINGLE_CORE_FORBIDDEN_TARGETS if params.get(name)
    ]
    if supplied:
        raise ValueError(
            "single-core calibration received forbidden targets: "
            + ", ".join(supplied)
        )


def _single_core_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _single_core_sample_pair(
        image: np.ndarray, left: object, right: object, sigma: float) -> dict:
    """Measure whether two references sample one raw core or two raw cores."""
    smooth = ndi.gaussian_filter(image.astype(np.float32), sigma)
    distance = float(np.hypot(
        float(left.x) - float(right.x), float(left.y) - float(right.y)))
    samples = max(5, 2 * int(np.ceil(distance)) + 1)
    xs = np.linspace(float(left.x), float(right.x), samples)
    ys = np.linspace(float(left.y), float(right.y), samples)
    values = ndi.map_coordinates(smooth, [ys, xs], order=1, mode="nearest")
    endpoint_low = float(min(values[0], values[-1]))
    endpoint_high = float(max(values[0], values[-1]))
    interior = values[1:-1] if len(values) > 2 else values
    scale = max(float(left.radius_px) + float(right.radius_px), 1.0)
    return {
        "separation_sum_radii": distance / scale,
        "valley_ratio": float(np.min(interior) / max(endpoint_low, 1.0)),
        "endpoint_balance": endpoint_low / max(endpoint_high, 1.0),
        "left_endpoint": float(values[0]),
        "right_endpoint": float(values[-1]),
    }


def _single_core_track_ids(member_rows: pd.DataFrame) -> list[int]:
    values = "|".join(member_rows.source_ref.astype(str))
    return sorted({int(value) for value in re.findall(r"T(\d+)", values)})


def single_core_reference_overlap_events(
        events: pd.DataFrame, members: pd.DataFrame, points: pd.DataFrame,
        raw: np.ndarray, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Suppress alerts whose same-owner references sample one raw-signal core.

    The rule audits every eligible separable-merge event in the supplied field.
    It never accepts identities, tracks, frames, coordinates, event IDs, or
    review failures as configuration.
    """
    assert_single_core_reference_overlap_target_free(params)
    sigma = float(params.get("single_core_raw_sigma_px", 1.0))
    maximum_two_core_valley = float(
        params.get("single_core_maximum_two_core_valley_ratio", 0.80))
    minimum_two_core_balance = float(
        params.get("single_core_minimum_two_core_endpoint_balance", 0.25))
    minimum_two_core_separation = float(
        params.get(
            "single_core_minimum_two_core_separation_sum_radii", 1.0))
    minimum_two_core_frames = int(
        params.get("single_core_minimum_two_core_frames", 2))
    minimum_shared_core_valley = float(
        params.get("single_core_minimum_shared_core_valley_ratio", 0.90))
    maximum_shared_core_separation = float(
        params.get(
            "single_core_maximum_shared_core_separation_sum_radii", 1.5))
    minimum_shared_core_frames = int(
        params.get("single_core_minimum_shared_core_frames", 2))
    maximum_handoff_balance = float(
        params.get("single_core_maximum_handoff_endpoint_balance", 0.10))
    minimum_successor_tail = int(
        params.get("single_core_minimum_successor_tail_frames", 5))
    maximum_successor_strong_gap = int(
        params.get("single_core_maximum_successor_strong_gap_frames", 2))

    indexed = points.set_index(["track_id", "frame"], drop=False)
    rows: list[dict] = []
    suppressed: set[str] = set()
    candidates = events[events.family.eq("separable_cells_merged")]
    for event in candidates.itertuples(index=False):
        event_id = str(event.event_id)
        event_members = members[members.event_id.astype(str).eq(event_id)]
        source_families = set(event_members.source_family.astype(str))
        tracks = _single_core_track_ids(event_members)
        reason = "mixed_constituent_families"
        pair_records: list[dict] = []
        if source_families == {"separable_cells_merged"} and len(tracks) >= 2:
            for frame in range(int(event.first_frame), int(event.last_frame) + 1):
                local = []
                for track in tracks:
                    key = (track, frame)
                    if key in indexed.index:
                        row = indexed.loc[key]
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                        if _single_core_bool(row.physically_visible):
                            local.append(row)
                for left, right in combinations(local, 2):
                    values = _single_core_sample_pair(
                        raw[frame], left, right, sigma)
                    pair_records.append({
                        "frame": frame,
                        "left_track": int(left.track_id),
                        "right_track": int(right.track_id),
                        "both_strong": (
                            _single_core_bool(left.strong)
                            and _single_core_bool(right.strong)),
                        **values,
                    })
            reason = "insufficient_positive_single_core_evidence"

        two_core_frames = {
            record["frame"] for record in pair_records
            if record["both_strong"]
            and record["separation_sum_radii"] >= minimum_two_core_separation
            and record["endpoint_balance"] >= minimum_two_core_balance
            and record["valley_ratio"] <= maximum_two_core_valley
        }
        shared_core_frames = {
            record["frame"] for record in pair_records
            if record["separation_sum_radii"] <= maximum_shared_core_separation
            and record["endpoint_balance"] >= minimum_two_core_balance
            and record["valley_ratio"] >= minimum_shared_core_valley
        }
        handoff = False
        if (source_families == {"separable_cells_merged"}
                and len(tracks) == 2 and pair_records):
            groups = {
                track: points[points.track_id.eq(track)].sort_values("frame")
                for track in tracks
            }
            outgoing = min(
                tracks, key=lambda track: int(groups[track].frame.max()))
            successor = max(
                tracks, key=lambda track: int(groups[track].frame.max()))
            outgoing_last = int(groups[outgoing].frame.max())
            successor_last = int(groups[successor].frame.max())
            after = groups[successor][
                groups[successor].frame.astype(int) > outgoing_last]
            strong_after = after[after.strong.map(_single_core_bool)]
            first_strong_gap = (
                int(strong_after.frame.min()) - outgoing_last
                if len(strong_after) else 10**6)
            event_records = [
                record for record in pair_records
                if int(event.first_frame) <= record["frame"]
                <= int(event.last_frame)
            ]
            handoff = bool(
                outgoing != successor
                and outgoing_last <= int(event.last_frame)
                and successor_last - outgoing_last >= minimum_successor_tail
                and first_strong_gap <= maximum_successor_strong_gap
                and len(event_records) >= 2
                and all(
                    record["endpoint_balance"] <= maximum_handoff_balance
                    for record in event_records)
                and all(
                    record["separation_sum_radii"]
                    <= maximum_shared_core_separation
                    for record in event_records))

        eligible = bool(
            source_families == {"separable_cells_merged"}
            and len(two_core_frames) < minimum_two_core_frames
            and (len(shared_core_frames) >= minimum_shared_core_frames
                 or handoff))
        if eligible:
            suppressed.add(event_id)
            reason = (
                "supported_single_core_overlap"
                if len(shared_core_frames) >= minimum_shared_core_frames
                else "low_support_reference_handoff")
        rows.append({
            "event_id": event_id,
            "physical_tracks": "|".join(map(str, tracks)),
            "source_families": "|".join(sorted(source_families)),
            "pair_observations": len(pair_records),
            "two_core_evidence_frames": len(two_core_frames),
            "shared_core_evidence_frames": len(shared_core_frames),
            "low_support_handoff": handoff,
            "status": "suppressed" if eligible else "retained",
            "reason": reason,
        })
    return (
        events[~events.event_id.astype(str).isin(suppressed)].copy(),
        pd.DataFrame(rows),
    )


_ALTERNATING_REFERENCE_FORBIDDEN_TARGETS = (
    "identity_id", "track_id", "frame_id", "event_id", "coordinate",
    "review_case", "case_id", "target_identity", "target_track",
    "target_frame", "target_event", "forced_identity", "forced_interval",
)


def assert_alternating_reference_single_body_target_free(params: dict) -> None:
    """Reject dataset-local selectors in the field-wide calibration."""
    supplied = sorted(
        key for key, value in params.items()
        if value not in (None, "", [], {})
        and any(token in key.lower()
                for token in _ALTERNATING_REFERENCE_FORBIDDEN_TARGETS)
    )
    if supplied:
        raise ValueError(
            "alternating-reference calibration received forbidden targets: "
            + ", ".join(supplied))
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError(
            "alternating-reference calibration must be field-wide")


def _alternating_reference_track_ids(value: object) -> tuple[int, ...]:
    return tuple(map(int, re.findall(r"\d+", str(value))))


def _alternating_reference_features(
        left: pd.DataFrame, right: pd.DataFrame,
        frame_count: int) -> dict:
    left = left.sort_values("frame")
    right = right.sort_values("frame")
    if left.empty or right.empty:
        return {
            "left_first_frame": -1, "right_first_frame": -1,
            "left_last_frame": -1, "right_last_frame": -1,
            "same_birth_frame": False, "overlap_frames": 0,
            "both_visible_frames": 0, "left_only_visible_frames": 0,
            "right_only_visible_frames": 0,
            "both_visible_fraction": 0.0, "positive_owner_count": 0,
            "median_pair_distance_sum_radii": np.inf,
            "interior_lifetime": False,
        }
    common = left.merge(right, on="frame", suffixes=("_left", "_right"))
    if common.empty:
        median_distance = np.inf
        both_visible = left_only = right_only = 0
        owners: set[int] = set()
    else:
        distances = np.hypot(
            common.x_left.to_numpy(float) - common.x_right.to_numpy(float),
            common.y_left.to_numpy(float) - common.y_right.to_numpy(float))
        scales = np.maximum(
            common.radius_px_left.to_numpy(float)
            + common.radius_px_right.to_numpy(float), 1.0)
        median_distance = float(np.median(distances / scales))
        left_visible = common.physically_visible_left.astype(bool)
        right_visible = common.physically_visible_right.astype(bool)
        both_visible = int(np.count_nonzero(left_visible & right_visible))
        left_only = int(np.count_nonzero(left_visible & ~right_visible))
        right_only = int(np.count_nonzero(~left_visible & right_visible))
        owners = set(map(int, pd.concat([
            common.dominant_candidate_owner_left,
            common.dominant_candidate_owner_right,
        ]).to_numpy(int))) - {0}
    first_left, first_right = int(left.frame.min()), int(right.frame.min())
    last_left, last_right = int(left.frame.max()), int(right.frame.max())
    return {
        "left_first_frame": first_left,
        "right_first_frame": first_right,
        "left_last_frame": last_left,
        "right_last_frame": last_right,
        "same_birth_frame": bool(first_left == first_right),
        "overlap_frames": int(len(common)),
        "both_visible_frames": both_visible,
        "left_only_visible_frames": left_only,
        "right_only_visible_frames": right_only,
        "both_visible_fraction": float(both_visible / max(len(common), 1)),
        "positive_owner_count": int(len(owners)),
        "median_pair_distance_sum_radii": median_distance,
        "interior_lifetime": bool(
            min(first_left, first_right) > 0
            and max(last_left, last_right) < frame_count - 1),
    }


def alternating_reference_single_body_events(
        events: pd.DataFrame, points: pd.DataFrame,
        single_core_audit: pd.DataFrame, frame_count: int,
        params: dict, enabled: bool = True,
        ) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Suppress false merge alerts caused by two references to one body.

    Selection uses complete reference histories and the raw-core audit over the
    supplied field. Event, track, identity, frame, and coordinate values are
    outputs only and are never accepted as selectors.
    """
    assert_alternating_reference_single_body_target_free(params)
    maximum_pair_observations = int(params.get(
        "alternating_reference_maximum_pair_observations", 1))
    maximum_two_core = int(params.get(
        "alternating_reference_maximum_two_core_frames", 0))
    minimum_overlap = int(params.get(
        "alternating_reference_minimum_overlap_frames", 4))
    minimum_both_visible = int(params.get(
        "alternating_reference_minimum_both_visible_frames", 2))
    maximum_both_fraction = float(params.get(
        "alternating_reference_maximum_both_visible_fraction", 0.5))
    minimum_exclusive = int(params.get(
        "alternating_reference_minimum_exclusive_frames_per_branch", 1))
    maximum_distance = float(params.get(
        "alternating_reference_maximum_distance_sum_radii", 2.0))

    rows: list[dict] = []
    suppressed: set[str] = set()
    candidates = single_core_audit[
        single_core_audit.status.astype(str).eq("retained")
        & single_core_audit.source_families.astype(str).str.contains(
            "separable_cells_merged", regex=False)]
    for row in candidates.itertuples(index=False):
        tracks = _alternating_reference_track_ids(row.physical_tracks)
        reasons: list[str] = []
        if len(tracks) != 2:
            reasons.append("not_exactly_two_physical_references")
            features = _alternating_reference_features(
                points.iloc[0:0], points.iloc[0:0], frame_count)
        else:
            features = _alternating_reference_features(
                points[points.track_id.astype(int).eq(tracks[0])],
                points[points.track_id.astype(int).eq(tracks[1])],
                frame_count)
        if int(row.pair_observations) > maximum_pair_observations:
            reasons.append("event_has_repeated_pair_observations")
        if int(row.two_core_evidence_frames) > maximum_two_core:
            reasons.append("positive_two_core_raw_evidence")
        if not features["same_birth_frame"]:
            reasons.append("references_do_not_share_birth_frame")
        if features["overlap_frames"] < minimum_overlap:
            reasons.append("reference_overlap_too_short")
        if features["positive_owner_count"] != 1:
            reasons.append("references_do_not_share_one_owner")
        if features["both_visible_frames"] < minimum_both_visible:
            reasons.append("insufficient_initial_joint_visibility")
        if features["both_visible_fraction"] > maximum_both_fraction:
            reasons.append("references_remain_jointly_visible")
        if features["left_only_visible_frames"] < minimum_exclusive:
            reasons.append("left_branch_never_exclusively_visible")
        if features["right_only_visible_frames"] < minimum_exclusive:
            reasons.append("right_branch_never_exclusively_visible")
        if features["median_pair_distance_sum_radii"] > maximum_distance:
            reasons.append("reference_branches_too_distant")
        if not features["interior_lifetime"]:
            reasons.append("reference_lifetime_is_boundary_censored")
        eligible = not reasons
        status = "suppressed" if eligible and enabled else (
            "eligible_filter_disabled" if eligible else "retained")
        if status == "suppressed":
            suppressed.add(str(row.event_id))
        rows.append({
            "event_id": str(row.event_id),
            "physical_tracks": str(row.physical_tracks),
            "pair_observations": int(row.pair_observations),
            "two_core_evidence_frames": int(row.two_core_evidence_frames),
            **features,
            "status": status,
            "reason": ("same_birth_alternating_single_body"
                       if eligible else "|".join(reasons)),
        })
    filtered = events[
        ~events.event_id.astype(str).isin(suppressed)].copy()
    return filtered, pd.DataFrame(rows), suppressed
