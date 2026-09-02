from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


CONNECTIVITY = np.ones((3, 3), np.uint8)


@dataclass(frozen=True)
class TwoCoreParams:
    minimum_object_px: int = 40
    erosion_iterations: int = 1
    minimum_core_px: int = 12
    minimum_core_fraction: float = 0.06
    minimum_core_separation_px: float = 7.0
    minimum_core_intensity_ratio: float = 0.35
    raw_sigma_px: float = 1.3


@dataclass(frozen=True)
class TransferParams:
    minimum_prior_area_px: int = 30
    maximum_area_retention: float = 0.65
    minimum_direct_transfer_fraction: float = 0.25
    minimum_host_growth_px: int = 20


def identity_components(frame: np.ndarray, identity: int
                        ) -> list[tuple[int, np.ndarray]]:
    target = frame == identity
    yy, xx = np.nonzero(target)
    if not len(yy):
        return []
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    components, count = ndi.label(
        target[y0:y1, x0:x1], structure=CONNECTIVITY)
    result: list[tuple[int, np.ndarray]] = []
    for number in range(1, count + 1):
        component = np.zeros(frame.shape, bool)
        component[y0:y1, x0:x1] = components == number
        result.append((number, component))
    return result


def eroded_cores(mask: np.ndarray, iterations: int = 1
                 ) -> list[tuple[int, np.ndarray]]:
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return []
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    eroded = ndi.binary_erosion(
        mask[y0:y1, x0:x1], structure=CONNECTIVITY,
        iterations=iterations)
    components, count = ndi.label(eroded, structure=CONNECTIVITY)
    cores: list[tuple[int, np.ndarray]] = []
    for number in range(1, count + 1):
        local = components == number
        core = np.zeros(mask.shape, bool)
        core[y0:y1, x0:x1] = local
        cores.append((int(np.count_nonzero(local)), core))
    return sorted(cores, key=lambda row: row[0], reverse=True)


def audit_two_core_objects(
        labels: np.ndarray, raw: np.ndarray,
        params: TwoCoreParams = TwoCoreParams()) -> pd.DataFrame:
    """Find one labelled object containing two substantial soma-like cores.

    Eroding one pixel removes a thin bridge. A suspicious object is one that then
    separates into two substantial, similarly bright cores. This is independent of
    the identity count, so a parameter sweep cannot look better merely by naming two
    neighbouring somas as one cell.
    """
    if labels.shape != raw.shape:
        raise ValueError("labels and raw must have the same T,Y,X shape")
    rows: list[dict] = []
    for t in range(len(labels)):
        smooth = ndi.gaussian_filter(
            raw[t].astype(np.float32), sigma=params.raw_sigma_px)
        for identity_value in np.unique(labels[t]):
            identity = int(identity_value)
            if identity <= 0:
                continue
            for component, mask in identity_components(labels[t], identity):
                area = int(mask.sum())
                if area < params.minimum_object_px:
                    continue
                cores = eroded_cores(mask, params.erosion_iterations)
                if len(cores) < 2:
                    continue
                first_area, first = cores[0]
                second_area, second = cores[1]
                minimum_core = max(
                    params.minimum_core_px,
                    int(np.ceil(params.minimum_core_fraction * area)))
                if second_area < minimum_core:
                    continue
                first_position = np.mean(
                    np.column_stack(np.nonzero(first)), axis=0)
                second_position = np.mean(
                    np.column_stack(np.nonzero(second)), axis=0)
                separation = float(np.linalg.norm(
                    first_position - second_position))
                if separation < params.minimum_core_separation_px:
                    continue
                first_intensity = float(np.percentile(smooth[first], 95))
                second_intensity = float(np.percentile(smooth[second], 95))
                intensity_ratio = min(first_intensity, second_intensity) / max(
                    first_intensity, second_intensity, 1.0)
                if intensity_ratio < params.minimum_core_intensity_ratio:
                    continue
                position = np.mean(np.column_stack(np.nonzero(mask)), axis=0)
                core_balance = second_area / max(first_area, 1)
                core_fraction = (first_area + second_area) / max(area, 1)
                score = (
                    core_balance * intensity_ratio
                    * min(separation / 12.0, 1.0)
                    * min(core_fraction / 0.30, 1.0)
                )
                rows.append({
                    "object_id": f"F{t + 1:03d}-I{identity:03d}-C{component:02d}",
                    "t": t, "imagej_frame": t + 1,
                    "identity": identity, "component": component,
                    "area_px": area,
                    "first_core_px": first_area,
                    "second_core_px": second_area,
                    "core_balance": float(core_balance),
                    "core_fraction": float(core_fraction),
                    "core_separation_px": separation,
                    "first_core_intensity": first_intensity,
                    "second_core_intensity": second_intensity,
                    "core_intensity_ratio": float(intensity_ratio),
                    "centre_y": float(position[0]),
                    "centre_x": float(position[1]),
                    "first_core_y": float(first_position[0]),
                    "first_core_x": float(first_position[1]),
                    "second_core_y": float(second_position[0]),
                    "second_core_x": float(second_position[1]),
                    "two_core_score": float(score),
                })
    columns = [
        "object_id", "t", "imagej_frame", "identity", "component",
        "area_px", "first_core_px", "second_core_px", "core_balance",
        "core_fraction", "core_separation_px", "first_core_intensity",
        "second_core_intensity", "core_intensity_ratio", "centre_y",
        "centre_x", "first_core_y", "first_core_x", "second_core_y",
        "second_core_x", "two_core_score",
    ]
    return pd.DataFrame(rows, columns=columns)


def audit_identity_body_transfers(
        labels: np.ndarray,
        params: TransferParams = TransferParams()) -> pd.DataFrame:
    """Find identities whose body is reassigned before their name disappears.

    The key failure is a donor that loses most of its area while a neighbouring host
    gains those exact pixels. The donor may still own a small process, so auditing only
    completely vanished names misses the event.
    """
    rows: list[dict] = []
    for t in range(1, len(labels)):
        previous, current = labels[t - 1], labels[t]
        previous_values, previous_counts = np.unique(
            previous[previous > 0], return_counts=True)
        current_values, current_counts = np.unique(
            current[current > 0], return_counts=True)
        previous_areas = dict(zip(
            map(int, previous_values), map(int, previous_counts)))
        current_areas = dict(zip(
            map(int, current_values), map(int, current_counts)))
        for donor, prior_area in previous_areas.items():
            if prior_area < params.minimum_prior_area_px:
                continue
            current_area = current_areas.get(donor, 0)
            area_retention = current_area / prior_area
            if area_retention > params.maximum_area_retention:
                continue
            values, counts = np.unique(
                current[previous == donor], return_counts=True)
            choices = [(int(count), int(value))
                       for value, count in zip(values, counts)
                       if int(value) > 0 and int(value) != donor]
            if not choices:
                continue
            transferred_px, host = max(choices)
            transfer_fraction = transferred_px / prior_area
            host_growth = (current_areas.get(host, 0)
                           - previous_areas.get(host, 0))
            if (transfer_fraction < params.minimum_direct_transfer_fraction
                    or host_growth < params.minimum_host_growth_px):
                continue
            transfer_mask = (previous == donor) & (current == host)
            position = np.mean(
                np.column_stack(np.nonzero(transfer_mask)), axis=0)
            collapse_fraction = 1.0 - area_retention
            score = (
                transfer_fraction * collapse_fraction
                * min(host_growth / max(prior_area, 1), 1.0)
            )
            rows.append({
                "event_id": f"BT{len(rows) + 1:04d}",
                "t": t, "imagej_frame": t + 1,
                "donor_identity": donor, "host_identity": host,
                "donor_prior_area_px": prior_area,
                "donor_current_area_px": current_area,
                "donor_still_present": bool(current_area > 0),
                "donor_area_retention": float(area_retention),
                "donor_collapse_fraction": float(collapse_fraction),
                "direct_transfer_px": transferred_px,
                "direct_transfer_fraction": float(transfer_fraction),
                "host_prior_area_px": previous_areas.get(host, 0),
                "host_current_area_px": current_areas.get(host, 0),
                "host_growth_px": int(host_growth),
                "transfer_y": float(position[0]),
                "transfer_x": float(position[1]),
                "body_transfer_score": float(score),
            })
    columns = [
        "event_id", "t", "imagej_frame", "donor_identity",
        "host_identity", "donor_prior_area_px", "donor_current_area_px",
        "donor_still_present", "donor_area_retention",
        "donor_collapse_fraction", "direct_transfer_px",
        "direct_transfer_fraction", "host_prior_area_px",
        "host_current_area_px", "host_growth_px", "transfer_y",
        "transfer_x", "body_transfer_score",
    ]
    return pd.DataFrame(rows, columns=columns)


def link_transfers_to_two_core_hosts(
        transfers: pd.DataFrame, objects: pd.DataFrame,
        maximum_centre_distance_px: float = 48.0) -> pd.DataFrame:
    linked = transfers.copy()
    linked["host_has_two_cores"] = False
    linked["host_two_core_object_id"] = ""
    linked["host_two_core_score"] = 0.0
    for index, event in linked.iterrows():
        candidates = objects[
            (objects.imagej_frame == int(event.imagej_frame))
            & (objects.identity == int(event.host_identity))].copy()
        if candidates.empty:
            continue
        candidates["distance"] = np.hypot(
            candidates.centre_y - float(event.transfer_y),
            candidates.centre_x - float(event.transfer_x))
        best = candidates.sort_values(
            ["distance", "two_core_score"], ascending=[True, False]).iloc[0]
        if float(best.distance) > maximum_centre_distance_px:
            continue
        linked.at[index, "host_has_two_cores"] = True
        linked.at[index, "host_two_core_object_id"] = str(best.object_id)
        linked.at[index, "host_two_core_score"] = float(best.two_core_score)
    return linked


def group_two_core_runs(
        objects: pd.DataFrame, maximum_gap_frames: int = 2,
        maximum_shift_px_per_frame: float = 18.0) -> pd.DataFrame:
    """Collapse repeated frames of one suspicious object into review cases."""
    runs: list[dict] = []
    members: list[list[dict]] = []
    ordered = objects.sort_values(
        ["imagej_frame", "two_core_score"], ascending=[True, False])
    for row in ordered.to_dict("records"):
        choices: list[tuple[float, int]] = []
        for index, run in enumerate(runs):
            gap = int(row["imagej_frame"] - run["last_imagej_frame"])
            if (row["identity"] != run["identity"]
                    or gap < 1 or gap > maximum_gap_frames):
                continue
            distance = float(np.hypot(
                row["centre_y"] - run["last_centre_y"],
                row["centre_x"] - run["last_centre_x"]))
            if distance <= maximum_shift_px_per_frame * gap:
                choices.append((distance, index))
        if choices:
            _, selected = min(choices)
            run = runs[selected]
            run["last_imagej_frame"] = int(row["imagej_frame"])
            run["last_centre_y"] = float(row["centre_y"])
            run["last_centre_x"] = float(row["centre_x"])
            members[selected].append(row)
        else:
            runs.append({
                "identity": int(row["identity"]),
                "first_imagej_frame": int(row["imagej_frame"]),
                "last_imagej_frame": int(row["imagej_frame"]),
                "last_centre_y": float(row["centre_y"]),
                "last_centre_x": float(row["centre_x"]),
            })
            members.append([row])
    summaries: list[dict] = []
    for number, (run, rows) in enumerate(zip(runs, members), start=1):
        representative = max(rows, key=lambda row: row["two_core_score"])
        summaries.append({
            "run_id": f"TC{number:04d}",
            "identity": int(run["identity"]),
            "first_imagej_frame": int(run["first_imagej_frame"]),
            "last_imagej_frame": int(run["last_imagej_frame"]),
            "flagged_frames": len(rows),
            "representative_imagej_frame": int(
                representative["imagej_frame"]),
            "representative_object_id": str(representative["object_id"]),
            "centre_y": float(representative["centre_y"]),
            "centre_x": float(representative["centre_x"]),
            "maximum_two_core_score": float(
                representative["two_core_score"]),
            "maximum_core_separation_px": float(max(
                row["core_separation_px"] for row in rows)),
            "minimum_core_intensity_ratio": float(min(
                row["core_intensity_ratio"] for row in rows)),
        })
    columns = [
        "run_id", "identity", "first_imagej_frame", "last_imagej_frame",
        "flagged_frames", "representative_imagej_frame",
        "representative_object_id", "centre_y", "centre_x",
        "maximum_two_core_score", "maximum_core_separation_px",
        "minimum_core_intensity_ratio",
    ]
    return pd.DataFrame(summaries, columns=columns)


def score_review_gold_standard(labels: np.ndarray,
                               gold_standard: pd.DataFrame) -> pd.DataFrame:
    """Score whether the two reviewed raw cores receive the required identities."""
    rows: list[dict] = []
    for case in gold_standard.itertuples():
        t = int(case.representative_imagej_frame) - 1
        first = int(labels[t, int(round(case.first_core_y)),
                           int(round(case.first_core_x))])
        second = int(labels[t, int(round(case.second_core_y)),
                            int(round(case.second_core_x))])
        both_detected = first > 0 and second > 0
        separated = both_detected and first != second
        same_identity = both_detected and first == second
        expected = str(case.truth)
        passed = (separated if expected == "undersegmentation"
                  else same_identity if expected == "single_cell"
                  else False)
        rows.append({
            "review_rank": int(case.review_rank),
            "run_id": str(case.run_id),
            "truth": expected,
            "first_core_identity": first,
            "second_core_identity": second,
            "both_cores_detected": bool(both_detected),
            "cores_separated": bool(separated),
            "cores_same_identity": bool(same_identity),
            "gate_passed": bool(passed),
        })
    return pd.DataFrame(rows)
