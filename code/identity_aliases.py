from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from persistent_merges import _identity_shape, _partition_score, _split_interval
from tracking import frame_observations, motion_support


def identity_segments(labels: np.ndarray) -> list[dict]:
    frames: dict[int, list[int]] = defaultdict(list)
    for t, frame in enumerate(labels):
        for identity in set(map(int, np.unique(frame))) - {0}:
            frames[identity].append(t)
    rows: list[dict] = []
    for identity, observed in frames.items():
        start = previous = observed[0]
        for t in observed[1:] + [observed[-1] + 2]:
            if t != previous + 1:
                rows.append({"identity": identity, "start_t": start,
                             "end_t": previous, "frames": previous - start + 1})
                start = t
            previous = t
    return rows


def _mask_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_pixels = np.column_stack(np.nonzero(first))
    second_pixels = np.column_stack(np.nonzero(second))
    if not len(first_pixels) or not len(second_pixels):
        return np.inf
    small, large = ((first_pixels, second_pixels)
                    if len(first_pixels) <= len(second_pixels)
                    else (second_pixels, first_pixels))
    return float(cKDTree(large).query(small, k=1)[0].min())


def _disappearance_hosts(cache: list[dict[int, dict]], identity: int,
                         end_t: int, params: dict) -> list[dict]:
    if end_t + 1 >= len(cache) or identity not in cache[end_t] \
            or identity in cache[end_t + 1]:
        return []
    previous, current = cache[end_t], cache[end_t + 1]
    old = previous[identity]
    rows: list[dict] = []
    for host in sorted(set(previous) & set(current) - {identity}):
        centroid_distance = float(np.linalg.norm(
            old["position"] - current[host]["position"]))
        if centroid_distance > float(params["centroid_prefilter_px"]):
            continue
        expanded = ndi.binary_dilation(current[host]["mask"], iterations=2)
        absorbed = float(np.mean(expanded[old["mask"]]))
        distance = _mask_distance(old["mask"], current[host]["mask"])
        growth = current[host]["area"] - previous[host]["area"]
        growth_fraction = growth / max(old["area"], 1)
        if (distance <= float(params["merge_reach_px"])
                and absorbed >= float(params["minimum_absorbed_fraction"])
                and float(params["minimum_host_growth_fraction"])
                <= growth_fraction <= float(params["maximum_host_growth_fraction"])):
            rows.append({
                "host_identity": int(host), "host_distance_px": distance,
                "absorbed_fraction": absorbed,
                "host_growth_fraction": growth_fraction,
            })
    return rows


def _appearance_hosts(cache: list[dict[int, dict]], identity: int,
                      start_t: int, params: dict) -> list[dict]:
    if start_t <= 0 or identity not in cache[start_t] \
            or identity in cache[start_t - 1]:
        return []
    previous, current = cache[start_t - 1], cache[start_t]
    new = current[identity]
    rows: list[dict] = []
    for host in sorted(set(previous) & set(current) - {identity}):
        centroid_distance = float(np.linalg.norm(
            new["position"] - current[host]["position"]))
        if centroid_distance > float(params["centroid_prefilter_px"]):
            continue
        distance = _mask_distance(new["mask"], current[host]["mask"])
        combined_ratio = ((new["area"] + current[host]["area"])
                          / max(previous[host]["area"], 1))
        released_fraction = ((previous[host]["area"] - current[host]["area"])
                             / max(new["area"], 1))
        if (distance <= float(params["split_reach_px"])
                and float(params["minimum_combined_area_ratio"])
                <= combined_ratio <= float(params["maximum_combined_area_ratio"])
                and float(params["minimum_released_fraction"])
                <= released_fraction <= float(params["maximum_released_fraction"])):
            rows.append({
                "host_identity": int(host), "split_distance_px": distance,
                "combined_area_ratio": float(combined_ratio),
                "released_fraction": float(released_fraction),
            })
    return rows


def alias_candidates(labels: np.ndarray, raw: np.ndarray, params: dict
                     ) -> pd.DataFrame:
    cache = [frame_observations(labels[t], raw[t]) for t in range(len(labels))]
    segments = identity_segments(labels)
    total_identity_frames: dict[int, int] = defaultdict(int)
    for segment in segments:
        total_identity_frames[int(segment["identity"])] += int(segment["frames"])
    ending: list[dict] = []
    starting: list[dict] = []
    for segment in segments:
        identity = int(segment["identity"])
        if total_identity_frames[identity] >= int(params["minimum_target_segment_frames"]):
            for host in _disappearance_hosts(
                    cache, identity, int(segment["end_t"]), params):
                ending.append({**segment, **host})
        for host in _appearance_hosts(
                cache, identity, int(segment["start_t"]), params):
            starting.append({**segment, **host})

    rows: list[dict] = []
    ending_by_host: dict[int, list[dict]] = defaultdict(list)
    starting_by_host: dict[int, list[dict]] = defaultdict(list)
    for row in ending:
        ending_by_host[int(row["host_identity"])].append(row)
    for row in starting:
        starting_by_host[int(row["host_identity"])].append(row)
    allowed_pairs = {tuple(map(int, pair)) for pair in params.get("allowed_pairs", [])}
    for host in sorted(set(ending_by_host) & set(starting_by_host)):
        for old in ending_by_host[host]:
            if allowed_pairs and (int(old["identity"]), int(host)) not in allowed_pairs:
                continue
            for new in starting_by_host[host]:
                if old["identity"] == new["identity"]:
                    continue
                gap = int(new["start_t"] - old["end_t"])
                if gap <= 1 or gap > int(params["maximum_alias_gap_frames"]):
                    continue
                if any(host not in cache[t]
                       for t in range(int(old["end_t"]), int(new["start_t"]) + 1)):
                    continue
                old_observation = cache[int(old["end_t"])][int(old["identity"])]
                new_observation = cache[int(new["start_t"])][int(new["identity"])]
                distance = float(np.linalg.norm(
                    old_observation["position"] - new_observation["position"]))
                maximum_distance = float(params["position_scale_px"]) * np.sqrt(gap)
                area_ratio = new_observation["area"] / max(old_observation["area"], 1)
                area_change = abs(float(np.log2(max(area_ratio, 1e-12))))
                mean_change = abs(float(np.log2(
                    (new_observation["mean"] + 1.0) / (old_observation["mean"] + 1.0))))
                if distance > maximum_distance \
                        or area_change > float(params["maximum_area_log2_change"]) \
                        or mean_change > float(params["maximum_mean_log2_change"]):
                    continue
                delayed = np.log2((raw[int(new["start_t"])].astype(np.float32) + 1.0)
                                  / (raw[int(old["end_t"])].astype(np.float32) + 1.0))
                support = motion_support(
                    old_observation["mask"], new_observation["mask"], delayed)
                cost = (
                    distance / max(maximum_distance, 1e-6)
                    + float(params["area_weight"]) * area_change
                    + float(params["mean_weight"]) * mean_change
                    - float(params["motion_weight"]) * support["motion_support"]
                    + float(params["gap_weight"]) * gap
                )
                rows.append({
                    "target_identity": int(old["identity"]),
                    "source_identity": int(new["identity"]),
                    "host_identity": host,
                    "target_end_t": int(old["end_t"]),
                    "target_end_imagej_frame": int(old["end_t"] + 1),
                    "source_start_t": int(new["start_t"]),
                    "source_start_imagej_frame": int(new["start_t"] + 1),
                    "source_end_t": int(new["end_t"]),
                    "source_end_imagej_frame": int(new["end_t"] + 1),
                    "gap_steps": gap, "distance_px": distance,
                    "maximum_distance_px": maximum_distance,
                    "area_ratio": float(area_ratio),
                    "area_log2_change": area_change,
                    "mean_log2_change": mean_change,
                    "delayed_motion_support": float(support["motion_support"]),
                    "cost": float(cost),
                    "disappearance_absorbed_fraction": old["absorbed_fraction"],
                    "appearance_combined_area_ratio": new["combined_area_ratio"],
                })
    return pd.DataFrame(rows)


def consolidate_host_conditioned_aliases(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    fixed = labels.copy()
    # Optional candidate guard. Once pixels have been renamed, do not let a later
    # iteration reinterpret the same physical segment in the reverse direction. The
    # default remains off so accepted production behaviour is unchanged until this
    # general rule has passed review.
    lock_renamed_domains = bool(params.get("lock_renamed_alias_domains", False))
    renamed_domain = np.zeros(labels.shape, bool)
    decisions: list[dict] = []
    all_candidates: list[pd.DataFrame] = []
    for iteration in range(1, int(params["maximum_aliases"]) + 1):
        candidates = alias_candidates(fixed, raw, params)
        if candidates.empty:
            break
        candidates = candidates.copy()
        candidates["iteration"] = iteration
        all_candidates.append(candidates)
        eligible: list[dict] = []
        for (_, _), group in candidates.groupby(
                ["source_identity", "source_start_t"]):
            # Several old segments can belong to the same established identity. They
            # are alternative endpoints, not competing biological identities.
            ordered = (group.sort_values(["cost", "target_identity"])
                       .drop_duplicates("target_identity", keep="first"))
            best = ordered.iloc[0]
            second_cost = float(ordered.iloc[1].cost) if len(ordered) > 1 else np.inf
            margin = second_cost - float(best.cost)
            if float(best.cost) <= float(params["maximum_alias_cost"]) \
                    and margin >= float(params["minimum_alias_margin"]):
                row = best.to_dict()
                row["second_best_margin"] = float(margin)
                eligible.append(row)
        if not eligible:
            break
        # A low-cost row can still be an identity swap if the proposed old and new
        # names are both visible in the source interval.  Reject that row and keep
        # considering other lost-cell claims; one conflict must not stop the entire
        # field-wide consolidation pass.
        safe = [row for row in eligible if not any(
            (np.any(fixed[t] == int(row["target_identity"]))
             and np.any(fixed[t] == int(row["source_identity"])))
            or (lock_renamed_domains and np.any(
                renamed_domain[t] & (fixed[t] == int(row["source_identity"]))))
            for t in range(int(row["source_start_t"]),
                           int(row["source_end_t"]) + 1))]
        if not safe:
            break
        best = min(safe, key=lambda row: (row["cost"], row["source_start_t"],
                                          row["source_identity"]))
        target = int(best["target_identity"])
        source = int(best["source_identity"])
        start_t, end_t = int(best["source_start_t"]), int(best["source_end_t"])
        renamed = 0
        for t in range(start_t, end_t + 1):
            mask = fixed[t] == source
            renamed += int(mask.sum())
            if lock_renamed_domains:
                renamed_domain[t] |= mask
            fixed[t][mask] = target
        decisions.append({
            "alias_id": f"AL{len(decisions) + 1:04d}",
            "iteration": iteration, **best,
            "renamed_pixels": renamed,
        })
    candidate_table = (pd.concat(all_candidates, ignore_index=True)
                       if all_candidates else pd.DataFrame())
    return fixed, pd.DataFrame(decisions), candidate_table


def _eligible_alias_rows(candidates: pd.DataFrame, params: dict) -> list[dict]:
    """Choose one unambiguous predecessor for each newly named segment."""
    eligible: list[dict] = []
    if candidates.empty:
        return eligible
    for (_, _, _), group in candidates.groupby(
            ["source_identity", "source_start_t", "source_end_t"]):
        ordered = (group.sort_values(["cost", "target_identity", "target_end_t"])
                   .drop_duplicates(["target_identity", "target_end_t"], keep="first"))
        best = ordered.iloc[0]
        second_cost = float(ordered.iloc[1].cost) if len(ordered) > 1 else np.inf
        margin = second_cost - float(best.cost)
        if (float(best.cost) <= float(params["maximum_alias_cost"])
                and margin >= float(params["minimum_alias_margin"])):
            row = best.to_dict()
            row["second_best_margin"] = float(margin)
            eligible.append(row)
    return eligible


def _recomputed_link(row: dict, labels: np.ndarray, raw: np.ndarray,
                     target: int, target_end_t: int, params: dict
                     ) -> dict | None:
    """Re-score a link from the latest accepted segment of an alias chain."""
    source = int(row["source_identity"])
    source_start_t = int(row["source_start_t"])
    host = int(row["host_identity"])
    gap = source_start_t - target_end_t
    if gap <= 0 or target not in np.unique(labels[target_end_t]) \
            or source not in np.unique(labels[source_start_t]):
        return None
    if any(host not in np.unique(labels[t])
           for t in range(target_end_t, source_start_t + 1)):
        return None

    observations = {
        t: frame_observations(labels[t], raw[t])
        for t in {target_end_t, min(target_end_t + 1, len(labels) - 1),
                  max(source_start_t - 1, 0), source_start_t}
    }
    old = observations[target_end_t].get(target)
    new = observations[source_start_t].get(source)
    if old is None or new is None:
        return None
    # A name already attached to another visible cell at the old segment's endpoint
    # signals an identity swap, not a new alias. Renaming it here would collapse two
    # established lineages; swaps need a separate permutation resolver.
    if source != target and source in observations[target_end_t]:
        return None

    # The latest segment must disappear into, and the new segment must emerge from,
    # the same continuing host. This is the chain-of-custody evidence that replaces
    # a cell-specific allow-list.
    after_end = observations[target_end_t + 1]
    before_start = observations[source_start_t - 1]
    at_start = observations[source_start_t]
    if (target in after_end or source in before_start
            or host not in observations[target_end_t] or host not in after_end
            or host not in before_start or host not in at_start):
        return None
    expanded = ndi.binary_dilation(after_end[host]["mask"], iterations=2)
    absorbed = float(np.mean(expanded[old["mask"]]))
    merge_distance = _mask_distance(old["mask"], after_end[host]["mask"])
    growth = after_end[host]["area"] - observations[target_end_t][host]["area"]
    growth_fraction = growth / max(old["area"], 1)
    split_distance = _mask_distance(new["mask"], at_start[host]["mask"])
    combined_ratio = ((new["area"] + at_start[host]["area"])
                      / max(before_start[host]["area"], 1))
    released_fraction = ((before_start[host]["area"] - at_start[host]["area"])
                         / max(new["area"], 1))
    if not (
            merge_distance <= float(params["merge_reach_px"])
            and absorbed >= float(params["minimum_absorbed_fraction"])
            and float(params["minimum_host_growth_fraction"])
            <= growth_fraction <= float(params["maximum_host_growth_fraction"])
            and split_distance <= float(params["split_reach_px"])
            and float(params["minimum_combined_area_ratio"])
            <= combined_ratio <= float(params["maximum_combined_area_ratio"])
            and float(params["minimum_released_fraction"])
            <= released_fraction <= float(params["maximum_released_fraction"])):
        return None

    distance = float(np.linalg.norm(old["position"] - new["position"]))
    maximum_distance = float(params["position_scale_px"]) * np.sqrt(gap)
    area_ratio = new["area"] / max(old["area"], 1)
    area_change = abs(float(np.log2(max(area_ratio, 1e-12))))
    mean_change = abs(float(np.log2(
        (new["mean"] + 1.0) / (old["mean"] + 1.0))))
    if (distance > maximum_distance
            or area_change > float(params["maximum_area_log2_change"])
            or mean_change > float(params["maximum_mean_log2_change"])):
        return None
    delayed = np.log2((raw[source_start_t].astype(np.float32) + 1.0)
                      / (raw[target_end_t].astype(np.float32) + 1.0))
    support = motion_support(old["mask"], new["mask"], delayed)
    cost = (
        distance / max(maximum_distance, 1e-6)
        + float(params["area_weight"]) * area_change
        + float(params["mean_weight"]) * mean_change
        - float(params["motion_weight"]) * support["motion_support"]
        + float(params["gap_weight"]) * gap
    )
    if cost > float(params["maximum_alias_cost"]):
        return None
    return {
        **row,
        "target_identity": int(target),
        "target_end_t": int(target_end_t),
        "target_end_imagej_frame": int(target_end_t + 1),
        "gap_steps": int(gap),
        "distance_px": distance,
        "maximum_distance_px": maximum_distance,
        "area_ratio": float(area_ratio),
        "area_log2_change": area_change,
        "mean_log2_change": mean_change,
        "delayed_motion_support": float(support["motion_support"]),
        "disappearance_absorbed_fraction": absorbed,
        "appearance_combined_area_ratio": float(combined_ratio),
        "cost": float(cost),
    }


def consolidate_recurrent_host_aliases(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Consolidate every repeated same-host alias chain found in the movie.

    A biological cell may be called 156, then 121, then 69 after repeatedly merging
    into cell 46. The old pair-specific resolver could repair this only if 156 and 46
    were named in configuration. This resolver discovers all such chains, accepts a
    chain only after the same host supplies at least two transitions, and refuses any
    rename that would make two simultaneously visible cells share an identity.
    """
    unrestricted = dict(params)
    unrestricted["allowed_pairs"] = []
    candidates = alias_candidates(labels, raw, unrestricted)
    eligible = _eligible_alias_rows(candidates, unrestricted)
    if not eligible:
        return labels.copy(), pd.DataFrame(), candidates, pd.DataFrame()

    by_host: dict[int, list[dict]] = defaultdict(list)
    for row in eligible:
        by_host[int(row["host_identity"])].append(row)

    proposed_groups: list[tuple[int, int, list[dict]]] = []
    audit: list[dict] = []
    minimum_links = int(unrestricted.get("minimum_recurrent_links", 2))
    for host, rows in sorted(by_host.items()):
        scratch = labels.copy()
        roots: dict[tuple[int, int], int] = {}
        latest_end: dict[int, int] = {}
        host_decisions: list[dict] = []
        for row in sorted(rows, key=lambda item: (
                int(item["source_start_t"]), float(item["cost"]),
                int(item["source_identity"]))):
            original_target = int(row["target_identity"])
            original_end = int(row["target_end_t"])
            target = roots.get((original_target, original_end), original_target)
            target_end = max(original_end, latest_end.get(target, original_end))
            source = int(row["source_identity"])
            start_t, end_t = int(row["source_start_t"]), int(row["source_end_t"])
            if target_end >= start_t:
                audit.append({**row, "host_identity": host,
                              "decision": "rejected_overlapping_chain"})
                continue
            if source != target and any(np.any(scratch[t] == target)
                                        for t in range(start_t, end_t + 1)):
                audit.append({**row, "host_identity": host,
                              "decision": "rejected_target_already_visible"})
                continue
            rescored = _recomputed_link(
                row, scratch, raw, target, target_end, unrestricted)
            if rescored is None:
                audit.append({**row, "host_identity": host,
                              "decision": "rejected_latest_link_evidence"})
                continue
            renamed = 0
            if source != target:
                for t in range(start_t, end_t + 1):
                    mask = scratch[t] == source
                    renamed += int(mask.sum())
                    scratch[t][mask] = target
            roots[(source, end_t)] = target
            latest_end[target] = end_t
            decision = {
                **rescored,
                "original_target_identity": original_target,
                "original_target_end_t": original_end,
                "original_target_end_imagej_frame": original_end + 1,
                "renamed_pixels": renamed,
                "decision": "proposed",
            }
            host_decisions.append(decision)

        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in host_decisions:
            grouped[int(row["target_identity"])].append(row)
        for target, group in grouped.items():
            if len(group) >= minimum_links:
                proposed_groups.append((host, target, group))
            else:
                for row in group:
                    audit.append({**row, "decision": "rejected_single_transition"})

    # A host that is itself being renamed cannot provide a stable chain of custody.
    # Reject the conflicting group rather than trying to infer two lineages at once.
    proposed_sources = {
        int(row["source_identity"])
        for _, _, group in proposed_groups for row in group
    }
    fixed = labels.copy()
    accepted: list[dict] = []
    for host, target, group in sorted(
            proposed_groups,
            key=lambda item: (-len(item[2]),
                              float(np.mean([row["cost"] for row in item[2]])),
                              item[0], item[1])):
        if host in proposed_sources:
            audit.extend({**row, "decision": "rejected_host_is_alias"}
                         for row in group)
            continue
        trial = fixed.copy()
        group_ok = True
        for row in sorted(group, key=lambda item: int(item["source_start_t"])):
            source = int(row["source_identity"])
            start_t, end_t = int(row["source_start_t"]), int(row["source_end_t"])
            if source != target and any(np.any(trial[t] == target)
                                        for t in range(start_t, end_t + 1)):
                group_ok = False
                break
            if source != target:
                for t in range(start_t, end_t + 1):
                    trial[t][trial[t] == source] = target
        if not group_ok:
            audit.extend({**row, "decision": "rejected_cross_group_conflict"}
                         for row in group)
            continue
        fixed = trial
        for row in sorted(group, key=lambda item: int(item["source_start_t"])):
            accepted.append({
                "alias_id": f"GA{len(accepted) + 1:04d}",
                **row,
                "decision": "accepted_recurrent_chain",
            })

    audit.extend(accepted)
    return fixed, pd.DataFrame(accepted), candidates, pd.DataFrame(audit)


def fill_alias_bridge_gaps(labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
                           aliases: pd.DataFrame, params: dict
                           ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    rows: list[dict] = []
    if aliases.empty:
        return fixed, pd.DataFrame(), inferred
    for alias in aliases.sort_values("source_start_t").itertuples():
        start_t = int(alias.target_end_t) + 1
        future_t = int(alias.source_start_t)
        target, host = int(alias.target_identity), int(alias.host_identity)
        if future_t <= start_t or any(np.any(fixed[t] == target)
                                      for t in range(start_t, future_t)):
            continue
        partition = _split_interval(
            fixed, raw, lag, start_t, future_t, target, host, params)
        if partition is None:
            continue
        fixed, frame_rows = partition
        for row in frame_rows:
            inferred[int(row["t"])][
                (fixed[int(row["t"])] == target)
                | (fixed[int(row["t"])] == host)] = True
        rows.append({
            "alias_id": alias.alias_id,
            "target_identity": target, "host_identity": host,
            "start_t": start_t, "start_imagej_frame": start_t + 1,
            "end_t": future_t - 1, "end_imagej_frame": future_t,
            "bookend_t": future_t, "bookend_imagej_frame": future_t + 1,
            "partitioned_frames": future_t - start_t,
            "mean_target_motion_support": float(np.mean(
                [row["a_motion_support"] for row in frame_rows])),
            "mean_host_motion_support": float(np.mean(
                [row["b_motion_support"] for row in frame_rows])),
            "mean_body_separation_support": float(np.mean(
                [row["body_separation_support"] for row in frame_rows])),
        })
    return fixed, pd.DataFrame(rows), inferred


def carry_recurrent_alias_pairs_to_end(
        labels: np.ndarray, raw: np.ndarray, lag: np.ndarray,
        aliases: pd.DataFrame, bridge_events: pd.DataFrame, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    fixed = labels.copy()
    inferred = np.zeros(labels.shape, bool)
    rows: list[dict] = []
    if aliases.empty or bridge_events.empty:
        return fixed, pd.DataFrame(), inferred
    pair_counts = bridge_events.groupby(
        ["target_identity", "host_identity"]).size()
    for (target, host), count in pair_counts.items():
        if int(count) < int(params["minimum_confirmed_cycles"]):
            continue
        target, host = int(target), int(host)
        start_t = int(bridge_events[
            (bridge_events.target_identity == target)
            & (bridge_events.host_identity == host)].bookend_t.max())
        observations = frame_observations(fixed[start_t], raw[start_t])
        if target not in observations or host not in observations:
            continue
        state_target, state_host = observations[target], observations[host]
        velocity_target = np.zeros(2, float)
        velocity_host = np.zeros(2, float)
        for t in range(start_t + 1, len(fixed)):
            current = frame_observations(fixed[t], raw[t])
            if target in current and host in current:
                velocity_target = (0.65 * (current[target]["position"]
                                           - state_target["position"])
                                   + 0.35 * velocity_target)
                velocity_host = (0.65 * (current[host]["position"]
                                         - state_host["position"])
                                 + 0.35 * velocity_host)
                state_target, state_host = current[target], current[host]
                continue
            present = [identity for identity in (target, host) if identity in current]
            if len(present) != 1:
                break
            remaining = present[0]
            merged_host = current[remaining]["mask"]
            predicted_target = state_target["position"] + velocity_target
            predicted_host = state_host["position"] + velocity_host
            from persistent_merges import _candidate_seeds
            seeds = _candidate_seeds(
                merged_host, raw[t], predicted_target, predicted_host, params)
            candidates: list[dict] = []
            for seed_target in seeds:
                for seed_host in seeds:
                    if seed_target == seed_host:
                        continue
                    scored = _partition_score(
                        merged_host, raw[t], state_target, state_host,
                        predicted_target, predicted_host,
                        state_target["area"], state_host["area"],
                        _identity_shape(state_target["mask"]),
                        _identity_shape(state_host["mask"]), lag[t - 1],
                        seed_target, seed_host, params)
                    if scored is not None:
                        candidates.append(scored)
            if not candidates:
                break
            best = min(candidates, key=lambda row: row["cost"])
            fixed[t][merged_host] = 0
            fixed[t][best["part_a"]] = target
            fixed[t][best["part_b"]] = host
            if not np.array_equal(fixed[t] > 0, labels[t] > 0):
                raise AssertionError("open merge carry changed foreground support")
            inferred[t][merged_host] = True
            current = frame_observations(fixed[t], raw[t])
            velocity_target = (0.65 * (current[target]["position"]
                                       - state_target["position"])
                               + 0.35 * velocity_target)
            velocity_host = (0.65 * (current[host]["position"]
                                     - state_host["position"])
                             + 0.35 * velocity_host)
            state_target, state_host = current[target], current[host]
            rows.append({
                "target_identity": target, "host_identity": host,
                "t": t, "imagej_frame": t + 1,
                "source_identity": remaining,
                "target_area_px": best["area_a_px"],
                "host_area_px": best["area_b_px"],
                "target_motion_support": best["a_motion_support"],
                "host_motion_support": best["b_motion_support"],
                "body_peak_target": best["body_peak_a"],
                "body_peak_host": best["body_peak_b"],
                "body_waist_intensity": best["body_waist_intensity"],
                "body_separation_support": best["body_separation_support"],
                "cost": best["cost"],
            })
    return fixed, pd.DataFrame(rows), inferred
