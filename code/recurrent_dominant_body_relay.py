"""Preserve an established dominant body through recurrent owner relays.

Discovery is complete-field and target-free.  A candidate begins with a strict
reciprocal area inversion after a long resident history.  The physical body is
then followed by dimensionless area conservation and centroid motion.  A repair
is permitted only when the same trajectory is repeatedly carried by multiple
foreign owners; this distinguishes a multi-owner relay from ordinary growth or
a single terminal exchange.

Identity, frame, coordinate, event, region, well and review selectors are not
accepted.  Reviewed names and intervals belong only in regression metadata.
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


STRUCTURE = np.ones((3, 3), np.uint8)
CONTROL_KEYS = {"mode", "targeting_mode", "output_stem"}
PATH_KEYS = {"labels_path", "unclaimed_path"}
PARAMETER_KEYS = {
    "minimum_prior_support_movie_fraction",
    "minimum_entry_body_ratio",
    "minimum_entry_area_cost_margin",
    "maximum_entry_cross_area_cost",
    "maximum_entry_separation_sum_radii",
    "minimum_trajectory_area_ratio",
    "maximum_trajectory_area_ratio",
    "maximum_trajectory_step_sum_radii",
    "trajectory_motion_weight",
    "trajectory_area_alpha",
    "maximum_missing_trajectory_frames",
    "minimum_foreign_occupancy_movie_fraction",
    "minimum_foreign_owner_runs",
    "minimum_distinct_foreign_owners",
    "minimum_trajectory_coverage_fraction",
    "maximum_prefix_area_interquartile_ratio",
}
ALLOWED_KEYS = CONTROL_KEYS | PATH_KEYS | PARAMETER_KEYS
AUDIT_COLUMNS = [
    "proposal_id", "resident_owner", "first_frame", "last_frame",
    "prior_support", "prior_area_q1", "prior_area_median",
    "prior_area_q3", "prefix_area_interquartile_ratio",
    "entry_claimant_owner", "entry_prior_resident_area",
    "entry_prior_claimant_area", "entry_current_resident_area",
    "entry_current_claimant_area", "entry_prior_body_ratio",
    "entry_current_body_ratio", "entry_same_area_cost",
    "entry_cross_area_cost", "entry_area_cost_margin",
    "entry_resident_to_claimant_separation_sum_radii",
    "trajectory_frames", "trajectory_coverage_fraction",
    "foreign_occupancy_frames", "foreign_occupancy_fraction",
    "foreign_owner_runs", "distinct_foreign_owners",
    "changed_pixels", "changed_frames", "eligible", "applied", "reason",
]
FRAME_COLUMNS = [
    "proposal_id", "frame", "resident_owner", "observed_owner",
    "operation", "observed_area", "expected_area", "area_ratio",
    "step_sum_radii", "trajectory_cost", "changed_pixels",
]


def assert_target_free(params: dict) -> None:
    if params.get("targeting_mode", "field_wide_discovery") != \
            "field_wide_discovery":
        raise ValueError("dominant-body relay must use field-wide discovery")
    unsupported = sorted(str(key) for key in params if key not in ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported parameters (selectors are forbidden): "
            + ", ".join(unsupported))
    missing = sorted(key for key in PATH_KEYS if not params.get(key))
    if missing:
        raise ValueError("missing input paths: " + ", ".join(missing))


def _fraction(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return value


def _components(frame: np.ndarray, owner: int) -> list[np.ndarray]:
    components, count = ndi.label(frame == int(owner), STRUCTURE)
    values = [components == index for index in range(1, count + 1)]
    return sorted(values, key=lambda mask: int(mask.sum()), reverse=True)


def _body(frame: np.ndarray, owner: int) -> dict | None:
    values = _components(frame, owner)
    if not values:
        return None
    mask = values[0]
    area = int(mask.sum())
    return {
        "owner": int(owner), "mask": mask, "area": area,
        "centroid": np.mean(np.argwhere(mask), axis=0),
        "radius": math.sqrt(area / math.pi),
        "secondary_area": sum(int(value.sum()) for value in values[1:]),
        "components": len(values),
    }


def _bodies(labels: np.ndarray) -> tuple[list[dict[int, dict]], list[int]]:
    owners = sorted(set(map(int, np.unique(labels))) - {0})
    frames: list[dict[int, dict]] = []
    for frame in labels:
        rows: dict[int, dict] = {}
        for owner in sorted(set(map(int, np.unique(frame))) - {0}):
            value = _body(frame, owner)
            if value is not None:
                rows[owner] = value
        frames.append(rows)
    return frames, owners


def _distance(left: dict, right: dict) -> float:
    return float(np.linalg.norm(left["centroid"] - right["centroid"])) / max(
        float(left["radius"] + right["radius"]), 1.0)


def _area_cost(left: float, right: float) -> float:
    return abs(math.log(max(float(left), 1.0) / max(float(right), 1.0)))


def _runs(values: list[int]) -> int:
    return sum(index == 0 or value != values[index - 1]
               for index, value in enumerate(values))


def _strict_entries(frame_bodies: list[dict[int, dict]], owners: list[int],
                    params: dict) -> list[dict]:
    frame_count = len(frame_bodies)
    minimum_prior = max(3, math.ceil(frame_count * _fraction(
        params, "minimum_prior_support_movie_fraction", 0.25)))
    minimum_ratio = float(params.get("minimum_entry_body_ratio", 2.0))
    minimum_margin = float(params.get(
        "minimum_entry_area_cost_margin", 0.75))
    maximum_cross = float(params.get("maximum_entry_cross_area_cost", 1.25))
    maximum_separation = float(params.get(
        "maximum_entry_separation_sum_radii", 3.0))
    entries: list[dict] = []
    for frame in range(1, frame_count):
        prior, current = frame_bodies[frame - 1], frame_bodies[frame]
        for resident in owners:
            if resident not in prior or resident not in current:
                continue
            prior_support = sum(resident in frame_bodies[index]
                                for index in range(frame))
            if prior_support < minimum_prior:
                continue
            for claimant in sorted(set(prior) & set(current) - {resident}):
                pa = float(prior[resident]["area"])
                pb = float(prior[claimant]["area"])
                ca = float(current[resident]["area"])
                cb = float(current[claimant]["area"])
                if pa <= pb or ca >= cb:
                    continue
                prior_ratio = pa / max(pb, 1.0)
                current_ratio = cb / max(ca, 1.0)
                same_cost = _area_cost(pa, ca) + _area_cost(pb, cb)
                cross_cost = _area_cost(pa, cb) + _area_cost(pb, ca)
                margin = same_cost - cross_cost
                separation = _distance(prior[resident], current[claimant])
                if (min(prior_ratio, current_ratio) < minimum_ratio
                        or margin < minimum_margin
                        or cross_cost > maximum_cross
                        or separation > maximum_separation):
                    continue
                entries.append({
                    "resident": resident, "claimant": claimant,
                    "frame": frame, "prior_support": prior_support,
                    "pa": pa, "pb": pb, "ca": ca, "cb": cb,
                    "prior_ratio": prior_ratio,
                    "current_ratio": current_ratio,
                    "same_cost": same_cost, "cross_cost": cross_cost,
                    "margin": margin, "separation": separation,
                })
    return entries


def _follow(frame_bodies: list[dict[int, dict]], entry: dict,
            params: dict) -> list[dict]:
    first = int(entry["frame"])
    previous = frame_bodies[first - 1][int(entry["resident"])]
    expected_area = float(previous["area"])
    minimum_ratio = float(params.get("minimum_trajectory_area_ratio", 0.45))
    maximum_ratio = float(params.get("maximum_trajectory_area_ratio", 2.20))
    maximum_step = float(params.get(
        "maximum_trajectory_step_sum_radii", 3.0))
    motion_weight = float(params.get("trajectory_motion_weight", 0.5))
    alpha = float(params.get("trajectory_area_alpha", 0.25))
    maximum_missing = int(params.get("maximum_missing_trajectory_frames", 1))
    missing = 0
    rows: list[dict] = []
    for frame in range(first, len(frame_bodies)):
        choices: list[tuple[float, int, dict, float, float]] = []
        for owner, body in frame_bodies[frame].items():
            ratio = float(body["area"] / max(expected_area, 1.0))
            if ratio < minimum_ratio or ratio > maximum_ratio:
                continue
            step = _distance(previous, body)
            if step > maximum_step:
                continue
            cost = _area_cost(expected_area, body["area"]) + motion_weight * step
            choices.append((cost, int(owner), body, ratio, step))
        if not choices:
            missing += 1
            if missing > maximum_missing:
                break
            continue
        missing = 0
        cost, owner, body, ratio, step = min(
            choices, key=lambda item: (item[0], item[1]))
        rows.append({
            "frame": frame, "observed_owner": owner,
            "observed_area": int(body["area"]),
            "expected_area": expected_area, "area_ratio": ratio,
            "step_sum_radii": step, "trajectory_cost": cost,
        })
        expected_area = ((1.0 - alpha) * expected_area
                         + alpha * float(body["area"]))
        previous = body
    return rows


def discover(labels: np.ndarray, params: dict
             ) -> tuple[pd.DataFrame, list[dict]]:
    assert_target_free(params)
    frame_bodies, owners = _bodies(labels)
    entries = _strict_entries(frame_bodies, owners, params)
    frame_count = len(labels)
    minimum_foreign = max(2, math.ceil(frame_count * _fraction(
        params, "minimum_foreign_occupancy_movie_fraction", 0.15)))
    minimum_runs = int(params.get("minimum_foreign_owner_runs", 3))
    minimum_distinct = int(params.get("minimum_distinct_foreign_owners", 2))
    minimum_coverage = float(params.get(
        "minimum_trajectory_coverage_fraction", 0.80))
    maximum_prefix_iqr_ratio = float(params.get(
        "maximum_prefix_area_interquartile_ratio", 2.0))
    rows: list[dict] = []
    internals: list[dict] = []
    seen: set[tuple[int, int]] = set()
    claimed_residents: set[int] = set()
    for entry in entries:
        key = (int(entry["resident"]), int(entry["frame"]))
        if key in seen:
            continue
        seen.add(key)
        trajectory = _follow(frame_bodies, entry, params)
        resident = int(entry["resident"])
        foreign = [int(row["observed_owner"]) for row in trajectory
                   if int(row["observed_owner"]) != resident]
        foreign_runs = _runs(foreign) if foreign else 0
        distinct = len(set(foreign))
        span = (int(trajectory[-1]["frame"]) - int(entry["frame"]) + 1
                if trajectory else 0)
        coverage = len(trajectory) / max(span, 1)
        prior_areas = [float(frame_bodies[frame][resident]["area"])
                       for frame in range(int(entry["frame"]))
                       if resident in frame_bodies[frame]]
        q1, median, q3 = (np.percentile(prior_areas, [25, 50, 75])
                          if prior_areas else (0.0, 0.0, 0.0))
        iqr_ratio = float(q3 / max(q1, 1.0))
        reasons: list[str] = []
        if len(foreign) < minimum_foreign:
            reasons.append("insufficient_foreign_occupancy")
        if foreign_runs < minimum_runs:
            reasons.append("insufficient_foreign_owner_runs")
        if distinct < minimum_distinct:
            reasons.append("single_owner_exchange_not_multi_owner_relay")
        if coverage < minimum_coverage:
            reasons.append("trajectory_coverage_too_low")
        if iqr_ratio > maximum_prefix_iqr_ratio:
            reasons.append("resident_prefix_area_not_stable")
        if not reasons and resident in claimed_residents:
            reasons.append("later_entry_inside_existing_resident_trajectory")
        if not reasons:
            claimed_residents.add(resident)
        public = {
            "proposal_id": "", "resident_owner": resident,
            "first_frame": int(entry["frame"]),
            "last_frame": int(trajectory[-1]["frame"])
                if trajectory else int(entry["frame"]),
            "prior_support": int(entry["prior_support"]),
            "prior_area_q1": float(q1),
            "prior_area_median": float(median),
            "prior_area_q3": float(q3),
            "prefix_area_interquartile_ratio": iqr_ratio,
            "entry_claimant_owner": int(entry["claimant"]),
            "entry_prior_resident_area": entry["pa"],
            "entry_prior_claimant_area": entry["pb"],
            "entry_current_resident_area": entry["ca"],
            "entry_current_claimant_area": entry["cb"],
            "entry_prior_body_ratio": entry["prior_ratio"],
            "entry_current_body_ratio": entry["current_ratio"],
            "entry_same_area_cost": entry["same_cost"],
            "entry_cross_area_cost": entry["cross_cost"],
            "entry_area_cost_margin": entry["margin"],
            "entry_resident_to_claimant_separation_sum_radii":
                entry["separation"],
            "trajectory_frames": len(trajectory),
            "trajectory_coverage_fraction": coverage,
            "foreign_occupancy_frames": len(foreign),
            "foreign_occupancy_fraction": len(foreign) / max(len(trajectory), 1),
            "foreign_owner_runs": foreign_runs,
            "distinct_foreign_owners": distinct,
            "changed_pixels": 0, "changed_frames": 0,
            "eligible": not reasons, "applied": False,
            "reason": ("eligible_recurrent_multi_owner_body_relay"
                       if not reasons else "|".join(reasons)),
        }
        rows.append(public)
        internals.append({**public, "trajectory": trajectory})
    number = 0
    for public, internal in zip(rows, internals):
        if public["eligible"]:
            number += 1
            proposal_id = f"MR{number:04d}"
            public["proposal_id"] = internal["proposal_id"] = proposal_id
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS), internals


def apply(labels: np.ndarray, audit: pd.DataFrame, internals: list[dict]
          ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    candidate = labels.copy()
    frame_rows: list[dict] = []
    occupied = np.zeros(len(labels), bool)
    for proposal in sorted(
            (value for value in internals if bool(value["eligible"])),
            key=lambda value: (-int(value["foreign_occupancy_frames"]),
                               int(value["first_frame"]))):
        resident = int(proposal["resident_owner"])
        trial = candidate.copy()
        pending: list[dict] = []
        conflict = False
        for row in proposal["trajectory"]:
            frame = int(row["frame"])
            observed = int(row["observed_owner"])
            if observed == resident:
                continue
            if occupied[frame]:
                conflict = True
                break
            resident_mask = trial[frame] == resident
            observed_mask = trial[frame] == observed
            if not np.any(observed_mask):
                conflict = True
                break
            if np.any(resident_mask):
                trial[frame][resident_mask] = observed
                operation = "reciprocal_frame_exchange"
            else:
                operation = "restore_absent_resident"
            trial[frame][observed_mask] = resident
            changed = int(np.count_nonzero(trial[frame] != candidate[frame]))
            pending.append({
                "proposal_id": proposal["proposal_id"], "frame": frame,
                "resident_owner": resident, "observed_owner": observed,
                "operation": operation,
                "observed_area": int(row["observed_area"]),
                "expected_area": float(row["expected_area"]),
                "area_ratio": float(row["area_ratio"]),
                "step_sum_radii": float(row["step_sum_radii"]),
                "trajectory_cost": float(row["trajectory_cost"]),
                "changed_pixels": changed,
            })
        index = audit.index[
            audit.resident_owner.astype(int).eq(resident)
            & audit.first_frame.astype(int).eq(int(proposal["first_frame"]))]
        if conflict or not pending:
            if len(index):
                audit.loc[index[0], "eligible"] = False
                audit.loc[index[0], "reason"] = (
                    "overlapping_proposal" if conflict else "already_consistent")
            continue
        candidate = trial
        for row in pending:
            occupied[int(row["frame"])] = True
        frame_rows.extend(pending)
        if len(index):
            audit.loc[index[0], "applied"] = True
            audit.loc[index[0], "reason"] = "applied_recurrent_multi_owner_body_relay"
            audit.loc[index[0], "changed_pixels"] = sum(
                int(row["changed_pixels"]) for row in pending)
            audit.loc[index[0], "changed_frames"] = len(pending)
    return candidate, audit, pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)


def produce(labels: np.ndarray, unclaimed: np.ndarray, params: dict):
    audit, internals = discover(labels, params)
    if params.get("mode", "candidate") == "baseline":
        candidate = labels.copy()
        frames = pd.DataFrame(columns=FRAME_COLUMNS)
    else:
        candidate, audit, frames = apply(labels, audit, internals)
    if not np.array_equal(candidate > 0, labels > 0):
        raise AssertionError("dominant-body relay changed foreground")
    if np.any((candidate > 0) & (unclaimed > 0)):
        raise AssertionError("assigned and unclaimed ledgers overlap")
    before_ids = set(map(int, np.unique(labels))) - {0}
    after_ids = set(map(int, np.unique(candidate))) - {0}
    if before_ids != after_ids:
        raise AssertionError("dominant-body relay changed movie identity set")
    changed = candidate != labels
    applied = audit[audit.applied.astype(bool)] if len(audit) else audit
    metrics = {
        "mode": params.get("mode", "candidate"),
        "targeting_mode": "field_wide_discovery",
        "target_counts": {key: 0 for key in (
            "identities", "owners", "tracks", "frames", "coordinates",
            "events", "regions", "wells", "review_cases")},
        "proposals_audited": int(len(audit)),
        "eligible_proposals": int(audit.eligible.astype(bool).sum())
            if len(audit) else 0,
        "applied_proposals": int(len(applied)),
        "changed_pixels": int(changed.sum()),
        "changed_frames": int(np.count_nonzero(
            changed.reshape(len(changed), -1).any(axis=1))),
        "foreground_ledger_exact": True,
        "preexisting_unclaimed_exact": True,
        "identity_set_exact": True,
        "new_identity_count": 0,
        "removed_identity_count": 0,
        "frame_object_bijection_preserved": True,
    }
    return candidate, unclaimed.copy(), audit, frames, metrics


def run(upstream_dir: Path | None, params: dict, out) -> dict:
    del upstream_dir
    labels = tifffile.imread(params["labels_path"])
    unclaimed = tifffile.imread(params["unclaimed_path"])
    candidate, candidate_unclaimed, audit, frames, metrics = produce(
        labels, unclaimed, params)
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
    audit_path = output_dir / "recurrent_dominant_body_relay_audit.csv"
    frames_path = output_dir / "recurrent_dominant_body_relay_frames.csv"
    metrics_path = output_dir / "metrics.json"
    audit.to_csv(audit_path, index=False)
    frames.to_csv(frames_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")
    return {"outputs": {
        "labels": labels_path, "unclaimed": unclaimed_path,
        "audit": audit_path, "frames": frames_path,
        "metrics": metrics_path,
    }, "summary": metrics}


