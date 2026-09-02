from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from tracking import motion_support


CONNECTIVITY = np.ones((3, 3), bool)


@dataclass(frozen=True)
class OneCoreInteraction:
    event_id: str
    domain_number: int
    review_frame: int
    source_frame: int
    incoming_local_id: int
    outgoing_local_id: int
    incoming_track_ids: str
    outgoing_track_ids: str
    incoming_identity: int
    outgoing_identity: int
    incoming_area_px: int
    outgoing_area_px: int
    incoming_core_px: int
    outgoing_core_px: int
    union_area_px: int
    union_core_px: int
    centre_y: float
    centre_x: float
    union: np.ndarray


@dataclass
class EpisodeTrace:
    domains: np.ndarray
    frames: pd.DataFrame
    boundaries: pd.DataFrame
    seeds: pd.DataFrame


@dataclass
class PhysicalLineageResult:
    labels: np.ndarray
    unclaimed: np.ndarray
    seed_domains: np.ndarray
    interactions: pd.DataFrame
    owner_proposals: pd.DataFrame
    episode_domains: np.ndarray
    episode_frames: pd.DataFrame
    episode_boundaries: pd.DataFrame
    episode_seeds: pd.DataFrame
    episode_actions: pd.DataFrame
    alias_audit: pd.DataFrame
    alias_actions: pd.DataFrame


def largest_core(mask: np.ndarray) -> tuple[int, np.ndarray | None]:
    eroded = ndi.binary_erosion(mask, structure=CONNECTIVITY)
    components, count = ndi.label(eroded, structure=CONNECTIVITY)
    if not count:
        return 0, None
    sizes = np.bincount(components.ravel())[1:]
    number = int(np.argmax(sizes)) + 1
    return int(sizes[number - 1]), components == number


def substantial_core_count(mask: np.ndarray, minimum_px: int) -> int:
    eroded = ndi.binary_erosion(mask, structure=CONNECTIVITY)
    components, count = ndi.label(eroded, structure=CONNECTIVITY)
    if not count:
        return 0
    sizes = np.bincount(components.ravel())[1:]
    return int(np.count_nonzero(sizes >= int(minimum_px)))


def _dominant_identity(frame: np.ndarray, mask: np.ndarray) -> int:
    values = frame[mask]
    identities, counts = np.unique(values[values > 0], return_counts=True)
    if not len(identities):
        return 0
    return int(identities[int(np.argmax(counts))])


def _parse_track_ids(value: object) -> set[int]:
    text = str(value)
    if text in ("", "nan", "None"):
        return set()
    return {int(item) for item in text.split(";") if item}


def _candidate_side(track_ids: set[int], incoming_ids: set[int],
                    outgoing_ids: set[int]) -> str:
    if track_ids and track_ids <= incoming_ids and not track_ids & outgoing_ids:
        return "incoming"
    if track_ids and track_ids <= outgoing_ids and not track_ids & incoming_ids:
        return "outgoing"
    return "mixed"


def _nearest_component(mask: np.ndarray, reference: np.ndarray
                       ) -> tuple[np.ndarray | None, float, int]:
    components, count = ndi.label(mask, structure=CONNECTIVITY)
    choices: list[tuple[float, np.ndarray, int]] = []
    for number in range(1, count + 1):
        component = components == number
        core_px, core = largest_core(component)
        centre_mask = core if core is not None else component
        centre = np.column_stack(np.nonzero(centre_mask)).mean(axis=0)
        choices.append((float(np.linalg.norm(centre - reference)),
                        component, core_px))
    if not choices:
        return None, float("inf"), 0
    distance, component, core_px = min(choices, key=lambda row: row[0])
    return component, distance, core_px


def _interaction_row(event: OneCoreInteraction) -> dict:
    return {key: value for key, value in vars(event).items() if key != "union"}


def discover_one_core_interactions(
        observations: np.ndarray, observation_table: pd.DataFrame,
        neutral_tracks: np.ndarray, labels: np.ndarray,
        source_frame_offset: int, minimum_union_core_px: int,
        ) -> tuple[list[OneCoreInteraction], np.ndarray, pd.DataFrame]:
    """Discover every adjacent incoming/outgoing fragment forming one soma.

    The detector receives no identities, positions, frame ranges, or review cases.
    ``source_frame_offset`` only aligns the source movie with a trimmed label movie.
    """
    if observations.ndim != 3 or neutral_tracks.ndim != 3 or labels.ndim != 3:
        raise ValueError("observations, neutral tracks, and labels must be TYX stacks")
    if observations.shape[1:] != labels.shape[1:] \
            or neutral_tracks.shape[1:] != labels.shape[1:]:
        raise ValueError("one-core interaction inputs differ in field shape")
    required = {"t", "source_kind", "two_sided_green", "green_track_ids",
                "local_id"}
    missing = required - set(observation_table.columns)
    if missing:
        raise ValueError(f"observation table missing columns: {sorted(missing)}")

    domains = np.zeros(labels.shape, np.uint16)
    detected: list[OneCoreInteraction] = []
    offset = int(source_frame_offset)
    first_source_index = max(1, offset)
    last_source_index = min(len(observations) - 1, len(neutral_tracks) - 1,
                            len(labels) + offset - 1)
    for source_index in range(first_source_index, last_source_index + 1):
        review_index = source_index - offset
        if not 0 <= review_index < len(labels):
            continue
        source_frame = source_index + 1
        review_frame = review_index + 1
        incoming_ids = set(map(int, np.unique(
            neutral_tracks[source_index - 1]))) - {0}
        outgoing_ids = set(map(int, np.unique(
            neutral_tracks[source_index]))) - {0}
        frame_rows = observation_table[
            (observation_table.t.astype(int) == source_index)
            & (observation_table.source_kind.astype(str) == "green")
            & (~observation_table.two_sided_green.astype(bool))]
        candidates: list[tuple[object, np.ndarray, str, set[int], int]] = []
        for row in frame_rows.sort_values("local_id").itertuples():
            tracks = _parse_track_ids(row.green_track_ids)
            side = _candidate_side(tracks, incoming_ids, outgoing_ids)
            mask = observations[source_index] == int(row.local_id)
            core_px, _ = largest_core(mask)
            candidates.append((row, mask, side, tracks, core_px))

        for first_index, first in enumerate(candidates):
            row_a, mask_a, side_a, tracks_a, core_a = first
            for second in candidates[first_index + 1:]:
                row_b, mask_b, side_b, tracks_b, core_b = second
                if {side_a, side_b} != {"incoming", "outgoing"}:
                    continue
                union = mask_a | mask_b
                if ndi.label(union, structure=CONNECTIVITY)[1] != 1:
                    continue
                union_core_px, _ = largest_core(union)
                if (union_core_px < int(minimum_union_core_px)
                        or min(core_a, core_b) >= int(minimum_union_core_px)):
                    continue
                identity_a = _dominant_identity(labels[review_index], mask_a)
                identity_b = _dominant_identity(labels[review_index], mask_b)
                if side_a == "incoming":
                    incoming = (row_a, mask_a, tracks_a, identity_a, core_a)
                    outgoing = (row_b, mask_b, tracks_b, identity_b, core_b)
                else:
                    incoming = (row_b, mask_b, tracks_b, identity_b, core_b)
                    outgoing = (row_a, mask_a, tracks_a, identity_a, core_a)
                yy, xx = np.nonzero(union)
                number = len(detected) + 1
                if number > np.iinfo(domains.dtype).max:
                    raise ValueError("too many one-core interactions for uint16 audit")
                event = OneCoreInteraction(
                    event_id=f"OCI-{review_frame:03d}-{number:04d}",
                    domain_number=number,
                    review_frame=review_frame,
                    source_frame=source_frame,
                    incoming_local_id=int(incoming[0].local_id),
                    outgoing_local_id=int(outgoing[0].local_id),
                    incoming_track_ids=";".join(map(str, sorted(incoming[2]))),
                    outgoing_track_ids=";".join(map(str, sorted(outgoing[2]))),
                    incoming_identity=int(incoming[3]),
                    outgoing_identity=int(outgoing[3]),
                    incoming_area_px=int(incoming[1].sum()),
                    outgoing_area_px=int(outgoing[1].sum()),
                    incoming_core_px=int(incoming[4]),
                    outgoing_core_px=int(outgoing[4]),
                    union_area_px=int(union.sum()),
                    union_core_px=int(union_core_px),
                    centre_y=float(yy.mean()), centre_x=float(xx.mean()),
                    union=union)
                domains[review_index][union] = number
                detected.append(event)

    columns = [key for key in OneCoreInteraction.__dataclass_fields__
               if key != "union"]
    table = pd.DataFrame([_interaction_row(event) for event in detected],
                         columns=columns)
    return detected, domains, table


def _predecessor_evidence(event: OneCoreInteraction, identity: int,
                          labels: np.ndarray, lag: np.ndarray,
                          params: dict) -> dict:
    result = {
        "identity": int(identity), "predecessor_area_px": 0,
        "predecessor_core_px": 0, "predecessor_distance_px": np.inf,
        "motion_support": 0.0, "source_loss_coverage": 0.0,
        "destination_gain_coverage": 0.0, "supported": False,
    }
    if identity <= 0 or event.review_frame <= 1:
        return result
    current_index = event.review_frame - 1
    yy, xx = np.nonzero(event.union)
    reference = np.array([yy.mean(), xx.mean()], float)
    component, distance, core_px = _nearest_component(
        labels[current_index - 1] == int(identity), reference)
    if component is None:
        return result
    support = motion_support(component, event.union, lag[current_index - 1])
    supported = bool(
        core_px >= int(params["minimum_substantial_core_px"])
        and distance <= float(params["maximum_predecessor_distance_px"])
        and support["motion_support"] >= float(params["minimum_motion_support"])
        and support["source_loss_coverage"]
            >= float(params["minimum_source_loss_coverage"])
        and support["destination_gain_coverage"]
            >= float(params["minimum_destination_gain_coverage"]))
    result.update({
        "predecessor_area_px": int(component.sum()),
        "predecessor_core_px": int(core_px),
        "predecessor_distance_px": float(distance),
        "motion_support": float(support["motion_support"]),
        "source_loss_coverage": float(support["source_loss_coverage"]),
        "destination_gain_coverage": float(
            support["destination_gain_coverage"]),
        "supported": supported,
    })
    return result


def propose_interaction_owners(events: list[OneCoreInteraction],
                               labels: np.ndarray, lag: np.ndarray,
                               params: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for event in events:
        identities = sorted({event.incoming_identity,
                             event.outgoing_identity} - {0})
        evidence = [_predecessor_evidence(event, identity, labels, lag, params)
                    for identity in identities]
        supported = [row for row in evidence if row["supported"]]
        if event.incoming_identity == 0 or event.outgoing_identity == 0:
            status, owner = "refused_unclaimed_pair", 0
        elif event.incoming_identity == event.outgoing_identity:
            status, owner = ("unchanged_same_identity",
                             event.incoming_identity)
        elif len(supported) == 1:
            status, owner = ("proposed_unique_predecessor_core",
                             int(supported[0]["identity"]))
        elif not supported:
            status, owner = "refused_no_supported_predecessor_core", 0
        else:
            status, owner = "refused_multiple_supported_predecessor_cores", 0
        rows.append({
            "event_id": event.event_id,
            "domain_number": event.domain_number,
            "review_frame": event.review_frame,
            "source_frame": event.source_frame,
            "incoming_identity": event.incoming_identity,
            "outgoing_identity": event.outgoing_identity,
            "proposed_owner": owner,
            "status": status,
            "union_area_px": event.union_area_px,
            "union_core_px": event.union_core_px,
            "centre_y": event.centre_y,
            "centre_x": event.centre_x,
            "evidence": json.dumps(evidence, sort_keys=True),
        })
    return pd.DataFrame(rows, columns=[
        "event_id", "domain_number", "review_frame", "source_frame",
        "incoming_identity", "outgoing_identity", "proposed_owner", "status",
        "union_area_px", "union_core_px", "centre_y", "centre_x", "evidence"])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = ndi.label(mask, structure=CONNECTIVITY)
    return [labels == number for number in range(1, count + 1)]


def _centre(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    return float(yy.mean()), float(xx.mean())


def _component_containing(frame_foreground: np.ndarray,
                          seed: np.ndarray) -> np.ndarray:
    choices = [(int(np.count_nonzero(mask & seed)), mask)
               for mask in _components(frame_foreground) if np.any(mask & seed)]
    if not choices:
        raise AssertionError("seed domain does not intersect foreground")
    return max(choices, key=lambda item: item[0])[1]


def _successor(prior: np.ndarray, current_foreground: np.ndarray,
               lag_frame: np.ndarray, params: dict
               ) -> tuple[np.ndarray | None, dict, str]:
    py, px = _centre(prior)
    eligible = []
    for mask in _components(current_foreground):
        y, x = _centre(mask)
        distance = float(np.hypot(y - py, x - px))
        if distance > float(params["maximum_successor_distance_px"]):
            continue
        support = motion_support(prior, mask, lag_frame)
        row = {
            "motion_support": float(support["motion_support"]),
            "source_loss_coverage": float(support["source_loss_coverage"]),
            "destination_gain_coverage": float(
                support["destination_gain_coverage"]),
            "distance_px": distance,
        }
        if row["motion_support"] >= float(
                params["minimum_successor_motion_support"]):
            eligible.append((row["motion_support"], -distance, mask, row))
    eligible.sort(key=lambda item: (-item[0], -item[1]))
    if not eligible:
        return None, {}, "no_supported_successor"
    if len(eligible) > 1:
        return None, {}, "multiple_supported_successors"
    return eligible[0][2], eligible[0][3], "continued"


def trace_connected_episodes(labels: np.ndarray, unclaimed: np.ndarray,
                             lag: np.ndarray, seed_domains: np.ndarray,
                             proposals: pd.DataFrame,
                             params: dict) -> EpisodeTrace:
    foreground = (labels > 0) | (unclaimed > 0)
    domains = np.zeros(labels.shape, np.uint16)
    frame_rows: list[dict] = []
    boundary_rows: list[dict] = []
    seed_rows: list[dict] = []
    episode_seeds: dict[int, list[str]] = {}
    next_episode = 1
    selected = proposals[
        proposals.status == "proposed_unique_predecessor_core"
    ].sort_values(["review_frame", "event_id"])
    for proposal in selected.itertuples():
        start = int(proposal.review_frame) - 1
        owner = int(proposal.proposed_owner)
        seed = seed_domains[start] == int(proposal.domain_number)
        existing = sorted(set(map(int, np.unique(domains[start][seed]))) - {0})
        if existing:
            if len(existing) != 1:
                raise AssertionError("seed overlaps multiple traced episodes")
            number = existing[0]
            episode_seeds[number].append(str(proposal.event_id))
            seed_rows.append({
                "event_id": proposal.event_id, "episode_number": number,
                "status": "deduplicated_into_existing_episode"})
            continue
        number = next_episode
        next_episode += 1
        episode_seeds[number] = [str(proposal.event_id)]
        seed_rows.append({
            "event_id": proposal.event_id, "episode_number": number,
            "status": "started_episode"})
        current = _component_containing(foreground[start], seed)
        link = {"motion_support": np.nan, "source_loss_coverage": np.nan,
                "destination_gain_coverage": np.nan, "distance_px": np.nan}
        for frame in range(start, len(labels)):
            if frame > start:
                current, link, link_status = _successor(
                    prior, foreground[frame], lag[frame - 1], params)
                if current is None:
                    boundary_rows.append({
                        "episode_number": number,
                        "boundary_review_frame": frame + 1,
                        "reason": link_status})
                    break
            owner_elsewhere = bool(np.any((labels[frame] == owner) & ~current))
            cores = substantial_core_count(
                current, int(params["minimum_substantial_core_px"]))
            if frame > start and owner_elsewhere:
                boundary_rows.append({
                    "episode_number": number,
                    "boundary_review_frame": frame + 1,
                    "reason": "owner_reappeared_separately"})
                break
            if cores != int(params["maximum_substantial_cores"]):
                boundary_rows.append({
                    "episode_number": number,
                    "boundary_review_frame": frame + 1,
                    "reason": "multiple_or_missing_substantial_cores"})
                break
            if frame > start and bool(np.all(labels[frame][current] == owner)):
                boundary_rows.append({
                    "episode_number": number,
                    "boundary_review_frame": frame + 1,
                    "reason": "baseline_full_owner_restored"})
                break
            domains[frame][current] = number
            y, x = _centre(current)
            accepted_owner = int(np.count_nonzero(labels[frame][current] == owner))
            frame_rows.append({
                "episode_number": number, "owner": owner,
                "review_frame": frame + 1,
                "source_frame": frame + 1 + int(params["source_frame_offset"]),
                "area_px": int(current.sum()),
                "accepted_owner_px": accepted_owner,
                "changed_pixels": int(current.sum()) - accepted_owner,
                "substantial_cores": cores, "centre_y": y, "centre_x": x,
                **link})
            prior = current
            if frame == len(labels) - 1:
                boundary_rows.append({
                    "episode_number": number,
                    "boundary_review_frame": frame + 1,
                    "reason": "end_of_movie_after_included_frame"})
    frames = pd.DataFrame(frame_rows)
    boundaries = pd.DataFrame(boundary_rows, columns=[
        "episode_number", "boundary_review_frame", "reason"])
    seeds = pd.DataFrame(seed_rows, columns=[
        "event_id", "episode_number", "status"])
    if len(frames):
        frames["seed_event_ids"] = frames.episode_number.map(
            lambda number: ";".join(episode_seeds[int(number)]))
    return EpisodeTrace(domains=domains, frames=frames,
                        boundaries=boundaries, seeds=seeds)


def apply_episode_domains(labels: np.ndarray, unclaimed: np.ndarray,
                          domains: np.ndarray, frames: pd.DataFrame,
                          source_frame_offset: int,
                          ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    candidate_unclaimed = unclaimed.copy()
    actions = []
    for row in frames.itertuples(index=False):
        frame = int(row.review_frame) - 1
        mask = domains[frame] == int(row.episode_number)
        before = candidate[frame][mask].copy()
        owner = int(row.owner)
        candidate[frame][mask] = owner
        candidate_unclaimed[frame][mask] = 0
        actions.append({
            "episode_number": int(row.episode_number),
            "seed_event_ids": row.seed_event_ids,
            "review_frame": int(row.review_frame),
            "source_frame": int(row.review_frame) + int(source_frame_offset),
            "owner": owner, "domain_area_px": int(mask.sum()),
            "changed_pixels": int(np.count_nonzero(before != owner))})
    return candidate, candidate_unclaimed, pd.DataFrame(actions, columns=[
        "episode_number", "seed_event_ids", "review_frame", "source_frame",
        "owner", "domain_area_px", "changed_pixels"])


def discover_first_appearance_aliases(
        original_labels: np.ndarray, episode_labels: np.ndarray,
        proposals: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Turn only evidence-backed first appearances into full-history aliases."""
    minimum_core = int(params["minimum_substantial_core_px"])
    minimum_prior = int(params["minimum_owner_prior_frames"])
    presence = {
        identity: np.any(original_labels == identity, axis=(1, 2))
        for identity in set(map(int, np.unique(original_labels))) - {0}}
    rows: list[dict] = []
    for proposal in proposals.sort_values(["review_frame", "event_id"]).itertuples():
        base = {
            "event_id": str(proposal.event_id),
            "review_frame": int(proposal.review_frame),
            "source_frame": int(proposal.source_frame),
            "alias_identity": 0, "canonical_identity": 0,
            "alias_first_review_frame": 0,
            "canonical_prior_frames": 0,
            "co_present_frames": 0,
            "maximum_collapsed_substantial_cores": 0,
            "eligible": False, "accepted": False,
            "decision": "refused_not_unique_predecessor_event",
        }
        if proposal.status != "proposed_unique_predecessor_core":
            rows.append(base)
            continue
        identities = {int(proposal.incoming_identity),
                      int(proposal.outgoing_identity)} - {0}
        owner = int(proposal.proposed_owner)
        nonowners = identities - {owner}
        if len(identities) != 2 or len(nonowners) != 1 or owner <= 0:
            base["decision"] = "refused_not_a_distinct_two_identity_pair"
            rows.append(base)
            continue
        alias = int(next(iter(nonowners)))
        alias_presence = presence.get(alias, np.zeros(len(original_labels), bool))
        owner_presence = presence.get(owner, np.zeros(len(original_labels), bool))
        first = int(np.flatnonzero(alias_presence)[0] + 1)
        prior = int(np.count_nonzero(
            owner_presence[:int(proposal.review_frame) - 1]))
        # A full-history rename can only create a duplicate soma name while the
        # two identities coexist. Frames containing just one side are safe even
        # when a single branched cell has more than one eroded intensity lobe.
        union_cores = [substantial_core_count(
            (frame == alias) | (frame == owner), minimum_core)
            for frame, alias_here, owner_here in zip(
                episode_labels, alias_presence, owner_presence)
            if alias_here and owner_here]
        co_present = int(np.count_nonzero(alias_presence & owner_presence))
        maximum_cores = max(union_cores, default=0)
        base.update({
            "alias_identity": alias, "canonical_identity": owner,
            "alias_first_review_frame": first,
            "canonical_prior_frames": prior,
            "co_present_frames": co_present,
            "maximum_collapsed_substantial_cores": maximum_cores,
        })
        if first != int(proposal.review_frame):
            base["decision"] = "refused_alias_existed_before_interaction"
        elif prior < minimum_prior:
            base["decision"] = "refused_canonical_identity_lacks_prior_tenure"
        elif maximum_cores > int(params["maximum_collapsed_substantial_cores"]):
            base["decision"] = "refused_collapse_would_join_multiple_somas"
        else:
            base["eligible"] = True
            base["decision"] = "eligible_first_appearance_alias"
        rows.append(base)

    # Resolve the alias graph without assuming there is exactly one edge.
    eligible = [row for row in rows if row["eligible"]]
    by_source: dict[int, list[dict]] = {}
    for row in eligible:
        by_source.setdefault(int(row["alias_identity"]), []).append(row)
    provisional: dict[int, dict] = {}
    for source, source_rows in by_source.items():
        targets = {int(row["canonical_identity"]) for row in source_rows}
        if len(targets) != 1:
            for row in source_rows:
                row["decision"] = "refused_conflicting_canonical_identities"
            continue
        winner = min(source_rows, key=lambda row: (
            int(row["review_frame"]), str(row["event_id"])))
        provisional[source] = winner
        for row in source_rows:
            if row is not winner:
                row["decision"] = "corroborating_duplicate_alias_evidence"

    edges = {source: int(row["canonical_identity"])
             for source, row in provisional.items()}
    cycle_sources: set[int] = set()
    for start in edges:
        path: list[int] = []
        current = start
        while current in edges and current not in path:
            path.append(current)
            current = edges[current]
        if current in path:
            cycle_sources.update(path[path.index(current):])
    for source, row in provisional.items():
        if source in cycle_sources:
            row["decision"] = "refused_alias_cycle"
        else:
            row["accepted"] = True
            row["decision"] = "accepted_first_appearance_alias"
    return pd.DataFrame(rows, columns=[
        "event_id", "review_frame", "source_frame", "alias_identity",
        "canonical_identity", "alias_first_review_frame",
        "canonical_prior_frames", "co_present_frames",
        "maximum_collapsed_substantial_cores", "eligible", "accepted",
        "decision"])


def apply_lineage_aliases(labels: np.ndarray, alias_audit: pd.DataFrame
                          ) -> tuple[np.ndarray, pd.DataFrame]:
    candidate = labels.copy()
    accepted = alias_audit[alias_audit.accepted.astype(bool)].sort_values(
        ["review_frame", "event_id"])
    edges = {int(row.alias_identity): int(row.canonical_identity)
             for row in accepted.itertuples()}
    actions: list[dict] = []
    for row in accepted.itertuples():
        source = int(row.alias_identity)
        target = int(row.canonical_identity)
        seen = {source}
        while target in edges:
            if target in seen:
                raise AssertionError("accepted lineage aliases contain a cycle")
            seen.add(target)
            target = edges[target]
        for frame in range(len(candidate)):
            mask = candidate[frame] == source
            changed = int(mask.sum())
            if not changed:
                continue
            candidate[frame][mask] = target
            actions.append({
                "event_id": row.event_id, "review_frame": frame + 1,
                "alias_identity": source, "canonical_identity": target,
                "renamed_pixels": changed})
    return candidate, pd.DataFrame(actions, columns=[
        "event_id", "review_frame", "alias_identity", "canonical_identity",
        "renamed_pixels"])


def reconcile_physical_lineages(
        labels: np.ndarray, unclaimed: np.ndarray, observations: np.ndarray,
        observation_table: pd.DataFrame, neutral_tracks: np.ndarray,
        lag: np.ndarray, params: dict) -> PhysicalLineageResult:
    """Discover local one-core repairs and safe whole-lineage aliases field-wide."""
    if labels.shape != unclaimed.shape:
        raise ValueError("labels and unclaimed stacks must have the same shape")
    if lag.shape[1:] != labels.shape[1:] or len(lag) < len(labels) - 1:
        raise ValueError("lag stack does not cover the retained label transitions")
    offset = int(params["source_frame_offset"])
    owner = params["owner_rule"]
    events, seed_domains, interaction_table = discover_one_core_interactions(
        observations, observation_table, neutral_tracks, labels, offset,
        int(owner["minimum_union_core_px"]))
    proposals = propose_interaction_owners(events, labels, lag, owner)
    episode_params = dict(params["episode_rule"])
    episode_params["source_frame_offset"] = offset
    trace = trace_connected_episodes(
        labels, unclaimed, lag, seed_domains, proposals, episode_params)
    episode_labels, episode_unclaimed, episode_actions = apply_episode_domains(
        labels, unclaimed, trace.domains, trace.frames, offset)
    alias_audit = discover_first_appearance_aliases(
        labels, episode_labels, proposals, params["alias_rule"])
    candidate, alias_actions = apply_lineage_aliases(
        episode_labels, alias_audit)
    if not np.array_equal(
            (candidate > 0) | (episode_unclaimed > 0),
            (labels > 0) | (unclaimed > 0)):
        raise AssertionError("physical-lineage reconciliation changed foreground")
    if np.any((candidate > 0) & (episode_unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    return PhysicalLineageResult(
        labels=candidate, unclaimed=episode_unclaimed,
        seed_domains=seed_domains, interactions=interaction_table,
        owner_proposals=proposals, episode_domains=trace.domains,
        episode_frames=trace.frames, episode_boundaries=trace.boundaries,
        episode_seeds=trace.seeds, episode_actions=episode_actions,
        alias_audit=alias_audit, alias_actions=alias_actions)
