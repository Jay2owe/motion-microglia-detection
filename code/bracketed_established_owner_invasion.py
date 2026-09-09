"""Repair short A-B-A excursions when B already occupies a stable nearby seat.

Discovery is field-wide. Identities, tracks, frames, coordinates, events, and
review cases cannot be supplied as producer targets. The rule requires two
independent mathematical proofs: the invaded physical track returns to its
bookending owner A, and invading owner B remains established on another
physical track before, during, and after the excursion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tifffile

import bounded_owner_excursion as bounded
import owner_consensus
from resident_takeover import attach_owners


FORBIDDEN_PARAMETER_PARTS = (
    "identity_target", "owner_target", "track_target", "frame_target",
    "coordinate", "event_target", "region", "review_case", "case_id",
    "allowed_pair", "forced_interval", "failure_target",
)
AUDIT_COLUMNS = [
    "proposal_id", "track_id", "owner_a", "owner_b", "middle_start",
    "middle_end", "middle_frames", "pre_frames", "post_frames",
    "bookend_frames", "visible_episode_frames", "strong_episode_fraction",
    "maximum_episode_step_sum_radii", "owner_a_present_during_middle",
    "donor_track", "donor_owner_support", "donor_owner_purity",
    "donor_strong_fraction", "donor_prior_support", "donor_post_support",
    "minimum_donor_separation_sum_radii", "maximum_donor_separation_sum_radii",
    "eligible", "reason",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("bracketed established-owner invasion must be field-wide")
    supplied = sorted(
        str(key) for key, value in params.items()
        if value not in (None, "", [], {})
        and any(part in str(key).lower()
                for part in FORBIDDEN_PARAMETER_PARTS))
    if supplied:
        raise ValueError(
            "bracketed established-owner invasion received forbidden targets: "
            + ", ".join(supplied))


def _point(group: pd.DataFrame, frame: int):
    rows = group[group.frame.astype(int) == int(frame)]
    return None if len(rows) != 1 else rows.iloc[0]


def _donor_catalog(groups: dict[int, pd.DataFrame], params: dict,
                   frame_count: int) -> dict[int, list[dict]]:
    """Index globally stable donor tracks once rather than per excursion."""
    minimum_support = int(math.ceil(
        float(params.get("minimum_donor_owner_support_movie_fraction", 0.25))
        * frame_count))
    minimum_purity = float(params.get("minimum_donor_owner_purity", 0.80))
    minimum_strong = float(params.get("minimum_donor_strong_fraction", 0.80))
    catalog: dict[int, list[dict]] = {}
    for track, group in groups.items():
        visible = group[group.physically_visible.astype(bool)]
        if visible.empty:
            continue
        counts = visible.candidate_owner.astype(int).value_counts()
        counts = counts[counts.index.astype(int) > 0]
        if counts.empty:
            continue
        owner = int(counts.index[0])
        support = int(counts.iloc[0])
        purity = float(support / len(visible))
        strong_fraction = float(visible.strong.astype(bool).mean())
        if (support < minimum_support or purity < minimum_purity
                or strong_fraction < minimum_strong):
            continue
        catalog.setdefault(owner, []).append({
            "track": int(track), "group": group, "visible": visible,
            "owned": visible[visible.candidate_owner.astype(int) == owner],
            "support": support, "purity": purity,
            "strong_fraction": strong_fraction,
        })
    return catalog


def _established_donor(
        groups: dict[int, pd.DataFrame], donor_catalog: dict[int, list[dict]],
        resident_track: int, owner_b: int, middle_frames: list[int],
        params: dict, frame_count: int
        ) -> tuple[dict | None, str]:
    minimum_prior = int(math.ceil(
        float(params.get("minimum_donor_prior_movie_fraction", 0.03))
        * frame_count))
    minimum_post = int(math.ceil(
        float(params.get("minimum_donor_post_movie_fraction", 0.03))
        * frame_count))
    minimum_separation = float(params.get(
        "minimum_donor_separation_sum_radii", 1.20))
    maximum_separation = float(params.get(
        "maximum_donor_separation_sum_radii", 4.00))
    resident = groups[int(resident_track)]
    first, last = min(middle_frames), max(middle_frames)
    options: list[dict] = []
    for detail in donor_catalog.get(int(owner_b), []):
        track = int(detail["track"])
        if track == int(resident_track):
            continue
        group = detail["group"]
        visible = detail["visible"]
        owned = detail["owned"]
        support = int(detail["support"])
        purity = float(detail["purity"])
        strong_fraction = float(detail["strong_fraction"])
        interval = visible[visible.frame.astype(int).isin(middle_frames)]
        if (len(interval) != len(middle_frames)
                or not interval.candidate_owner.astype(int).eq(owner_b).all()):
            continue
        prior = int(((owned.frame.astype(int) < first)).sum())
        post = int(((owned.frame.astype(int) > last)).sum())
        if prior < minimum_prior or post < minimum_post:
            continue
        separations: list[float] = []
        valid = True
        for frame in middle_frames:
            target = _point(resident, frame)
            donor = _point(group, frame)
            if target is None or donor is None:
                valid = False
                break
            separation = bounded._scaled_step(target, donor)
            if not minimum_separation <= separation <= maximum_separation:
                valid = False
                break
            separations.append(float(separation))
        if valid:
            options.append({
                "track": int(track), "support": support, "purity": purity,
                "strong_fraction": strong_fraction, "prior": prior,
                "post": post, "minimum_separation": min(separations),
                "maximum_separation": max(separations),
            })
    if not options:
        return None, "established_donor_missing"
    options.sort(key=lambda row: (-row["support"], -row["purity"], row["track"]))
    if len(options) > 1:
        margin = int(params.get("minimum_donor_support_margin_frames", 3))
        if options[0]["support"] - options[1]["support"] < margin:
            return None, "established_donor_ambiguous"
    return options[0], "eligible"


def discover(labels: np.ndarray, points: pd.DataFrame,
             params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every A-B-A physical-track run triple in the complete field."""
    assert_target_free(params)
    attached = attach_owners(points, labels).sort_values(
        ["track_id", "frame"]).reset_index(drop=True)
    groups = {int(track): group.sort_values("frame")
              for track, group in attached.groupby("track_id", sort=True)}
    frame_count = int(len(labels))
    donor_catalog = _donor_catalog(groups, params, frame_count)
    maximum_middle = int(math.ceil(
        float(params.get("maximum_middle_movie_fraction", 0.08))
        * frame_count))
    minimum_bookend = int(math.ceil(
        float(params.get("minimum_total_bookend_movie_fraction", 0.11))
        * frame_count))
    minimum_long_flank = int(math.ceil(
        float(params.get("minimum_long_flank_movie_fraction", 0.05))
        * frame_count))
    minimum_strong = float(params.get("minimum_episode_strong_fraction", 0.50))
    maximum_step = float(params.get("maximum_episode_step_sum_radii", 1.50))
    rows: list[dict] = []
    internals: list[dict] = []
    for track, group in groups.items():
        runs = bounded._positive_runs(group)
        for position in range(1, len(runs) - 1):
            pre, middle, post = runs[position - 1:position + 2]
            if pre["owner"] != post["owner"] or pre["owner"] == middle["owner"]:
                continue
            owner_a, owner_b = int(pre["owner"]), int(middle["owner"])
            middle_frames = list(map(int, middle["frames"]))
            episode_frames = list(range(int(pre["start"]), int(post["end"]) + 1))
            episode = group[group.frame.astype(int).isin(episode_frames)]
            visible_episode = episode[episode.physically_visible.astype(bool)]
            strong_fraction = (float(visible_episode.strong.astype(bool).mean())
                               if len(visible_episode) else 0.0)
            points_in_order = list(visible_episode.sort_values("frame")
                                   .itertuples(index=False))
            maximum_seen_step = max(
                (bounded._scaled_step(left, right)
                 for left, right in zip(points_in_order[:-1], points_in_order[1:])),
                default=float("inf"))
            owner_a_present = int(sum(
                bool(np.any(labels[frame] == owner_a))
                for frame in middle_frames))
            reasons: list[str] = []
            if len(middle_frames) > maximum_middle:
                reasons.append("middle_too_long")
            bookends = int(len(pre["frames"]) + len(post["frames"]))
            if bookends < minimum_bookend:
                reasons.append("insufficient_total_bookend_support")
            if max(len(pre["frames"]), len(post["frames"])) < minimum_long_flank:
                reasons.append("insufficient_long_flank_support")
            if (len(episode) != len(episode_frames)
                    or len(visible_episode) != len(episode_frames)):
                reasons.append("episode_not_physically_continuous")
            if strong_fraction < minimum_strong:
                reasons.append("episode_too_dim")
            if maximum_seen_step > maximum_step:
                reasons.append("episode_physical_discontinuity")
            if owner_a_present:
                reasons.append("bookending_owner_present_elsewhere")
            donor, donor_reason = _established_donor(
                groups, donor_catalog, int(track), owner_b, middle_frames,
                params, frame_count)
            if donor is None:
                reasons.append(donor_reason)
                donor = {
                    "track": 0, "support": 0, "purity": 0.0,
                    "strong_fraction": 0.0, "prior": 0, "post": 0,
                    "minimum_separation": float("nan"),
                    "maximum_separation": float("nan"),
                }
            public = {
                "proposal_id": "", "track_id": int(track),
                "owner_a": owner_a, "owner_b": owner_b,
                "middle_start": int(middle["start"]),
                "middle_end": int(middle["end"]),
                "middle_frames": len(middle_frames),
                "pre_frames": len(pre["frames"]),
                "post_frames": len(post["frames"]),
                "bookend_frames": bookends,
                "visible_episode_frames": int(len(visible_episode)),
                "strong_episode_fraction": strong_fraction,
                "maximum_episode_step_sum_radii": maximum_seen_step,
                "owner_a_present_during_middle": owner_a_present,
                "donor_track": int(donor["track"]),
                "donor_owner_support": int(donor["support"]),
                "donor_owner_purity": float(donor["purity"]),
                "donor_strong_fraction": float(donor["strong_fraction"]),
                "donor_prior_support": int(donor["prior"]),
                "donor_post_support": int(donor["post"]),
                "minimum_donor_separation_sum_radii": float(
                    donor["minimum_separation"]),
                "maximum_donor_separation_sum_radii": float(
                    donor["maximum_separation"]),
                "eligible": not reasons,
                "reason": ("eligible_bracketed_established_owner_invasion"
                           if not reasons else "|".join(reasons)),
            }
            rows.append(public)
            internals.append({**public, "companion_track": int(donor["track"]),
                              "frame_values": middle_frames})
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            public["proposal_id"] = internal["proposal_id"] = f"BEI{number:04d}"
    return (pd.DataFrame(rows, columns=AUDIT_COLUMNS), pd.DataFrame(internals))


def _copy_sidecars(upstream_dir: Path | None, params: dict, out) -> dict:
    outputs = {}
    empty_headers = {
        "application_audit.csv": (
            "proposal_id,physical_track,assigned_identity,outcome,reason,"
            "changed_pixels,changed_frames\n"),
        "conflicted_lineage_audit.csv": (
            "proposal_id,physical_track,assigned_identity,outcome,reason,"
            "changed_pixels,changed_frames\n"),
    }
    for name in ("application_audit.csv", "conflicted_lineage_audit.csv"):
        source = upstream_dir / name if upstream_dir is not None else None
        configured = params.get(name.removesuffix(".csv") + "_path")
        if source is None or not source.is_file():
            source = Path(configured) if configured else None
        target = out.out / name
        if source is not None and source.is_file():
            shutil.copyfile(source, target)
        else:
            target.write_text(empty_headers[name], encoding="utf-8")
        outputs[name.removesuffix(".csv")] = target
    return outputs


def _save_labels(source: Path, target: Path, labels: np.ndarray,
                 candidate: np.ndarray) -> None:
    """Preserve the exact parent TIFF whenever the rule is a no-op."""
    if np.array_equal(candidate, labels):
        shutil.copyfile(source, target)
        return
    tifffile.imwrite(
        target, candidate, imagej=True, compression="zlib",
        metadata={"axes": "TYX", "finterval": 1800.0,
                  "tunit": "sec", "unit": "pixel"})


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    assert_target_free(params)
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    raw = tifffile.imread(params["raw_path"])
    if len(raw) != len(labels):
        offset = int(params.get("raw_frame_offset", 2))
        raw = raw[offset:offset + len(labels)]
    points = pd.read_csv(params["physical_track_points_path"])
    audit, internal = discover(labels, points, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        applications = pd.DataFrame(columns=bounded.APPLICATION_COLUMNS)
    else:
        candidate, applications = bounded.apply(
            labels, raw, points, internal, params)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("bracketed invasion changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed outputs overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("bracketed invasion changed identity set")
    duplicates = owner_consensus.count_new_duplicate_components(labels, candidate)
    if duplicates:
        raise AssertionError("bracketed invasion created duplicate components")
    output_stem = str(params.get("output_stem", Path(params["labels_path"]).stem))
    labels_path = out.out / f"{output_stem}.tif"
    unclaimed_path = out.out / f"{output_stem}_unclaimed_original_ids.tif"
    _save_labels(Path(params["labels_path"]), labels_path, labels, candidate)
    shutil.copyfile(params["unclaimed_path"], unclaimed_path)
    audit_path = out.out / "bracketed_established_owner_invasion_audit.csv"
    application_path = out.out / "bracketed_established_owner_invasion_applications.csv"
    audit.to_csv(audit_path, index=False)
    applications.to_csv(application_path, index=False)
    changed = candidate != labels
    applied = applications[applications.applied.astype(bool)] \
        if len(applications) else applications
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "tracks", "frames", "coordinates", "events",
            "regions", "review_cases")},
        "run_triples_audited": int(len(audit)),
        "eligible_invasions": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_invasions": int(applied.proposal_id.nunique())
            if len(applied) else 0,
        "applied_frames": int(applied.frame.astype(int).nunique())
            if len(applied) else 0,
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": before_ids == after_ids,
        "new_duplicate_components": int(duplicates),
    }
    metrics_path = out.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs = {"labels": labels_path, "unclaimed": unclaimed_path,
               "audit": audit_path, "applications": application_path,
               "metrics": metrics_path}
    outputs.update(_copy_sidecars(upstream_dir, params, out))
    return {"outputs": outputs, "summary": metrics}
