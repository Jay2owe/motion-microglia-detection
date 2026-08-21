from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def _core_area(mask: np.ndarray) -> int:
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), bool))
    components, count = ndi.label(eroded, structure=np.ones((3, 3), bool))
    if not count:
        return 0
    sizes = np.bincount(components.ravel())[1:]
    return int(sizes.max()) if len(sizes) else 0


def _observations(frame: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    object_slices = ndi.find_objects(frame)
    for identity in sorted(set(map(int, np.unique(frame))) - {0}):
        slices = object_slices[identity - 1]
        if slices is None:
            continue
        y_slice, x_slice = slices
        local_mask = frame[y_slice, x_slice] == identity
        yy, xx = np.nonzero(local_mask)
        bounds = (int(y_slice.start), int(y_slice.stop),
                  int(x_slice.start), int(x_slice.stop))
        rows.append({
            "identity": identity,
            "local_mask": local_mask,
            "area": int(local_mask.sum()),
            "core_area": _core_area(local_mask),
            "position": np.array([
                yy.mean() + y_slice.start, xx.mean() + x_slice.start], float),
            "bounds": bounds,
        })
    return rows


def _intersection(first: dict, second: dict) -> int:
    a, b = first["bounds"], second["bounds"]
    y0, y1 = max(a[0], b[0]), min(a[1], b[1])
    x0, x1 = max(a[2], b[2]), min(a[3], b[3])
    if y0 >= y1 or x0 >= x1:
        return 0
    first_local = first["local_mask"][
        y0 - a[0]:y1 - a[0], x0 - a[2]:x1 - a[2]]
    second_local = second["local_mask"][
        y0 - b[0]:y1 - b[0], x0 - b[2]:x1 - b[2]]
    return int(np.count_nonzero(
        first_local & second_local))


def _paint(frame: np.ndarray, observation: dict, identity: int) -> None:
    y0, y1, x0, x1 = observation["bounds"]
    view = frame[y0:y1, x0:x1]
    view[observation["local_mask"]] = identity


def _overlap_tables(previous: list[dict], current: list[dict]) -> dict[str, np.ndarray]:
    shape = (len(previous), len(current))
    intersection = np.zeros(shape, np.int64)
    iou = np.zeros(shape, float)
    source = np.zeros(shape, float)
    destination = np.zeros(shape, float)
    for row, old in enumerate(previous):
        for col, new in enumerate(current):
            pixels = _intersection(old, new)
            if not pixels:
                continue
            union = old["area"] + new["area"] - pixels
            intersection[row, col] = pixels
            iou[row, col] = pixels / max(union, 1)
            source[row, col] = pixels / max(old["area"], 1)
            destination[row, col] = pixels / max(new["area"], 1)
    return {"intersection": intersection, "iou": iou,
            "source": source, "destination": destination}


def _strong_mutual_links(previous: list[dict], current: list[dict],
                         params: dict) -> list[dict]:
    if not previous or not current:
        return []
    table = _overlap_tables(previous, current)
    intersection, iou = table["intersection"], table["iou"]
    source, destination = table["source"], table["destination"]
    rows: list[dict] = []
    for row, old in enumerate(previous):
        if not np.any(intersection[row]):
            continue
        col = int(np.argmax(iou[row]))
        if int(np.argmax(iou[:, col])) != row:
            continue
        row_other = np.delete(iou[row], col)
        col_other = np.delete(iou[:, col], row)
        second = max(float(row_other.max()) if len(row_other) else 0.0,
                     float(col_other.max()) if len(col_other) else 0.0)
        margin = float(iou[row, col]) - second
        source_other = np.delete(source[row], col)
        destination_other = np.delete(destination[:, col], row)
        competitor = max(
            float(source_other.max()) if len(source_other) else 0.0,
            float(destination_other.max()) if len(destination_other) else 0.0)
        ratio = current[col]["area"] / max(previous[row]["area"], 1)
        allowed = bool(
            intersection[row, col] >= int(params.get(
                "minimum_intersection_px", 12))
            and iou[row, col] >= float(params.get(
                "minimum_intersection_over_union", 0.50))
            and source[row, col] >= float(params.get(
                "minimum_bidirectional_coverage", 0.70))
            and destination[row, col] >= float(params.get(
                "minimum_bidirectional_coverage", 0.70))
            and float(params.get("minimum_area_ratio", 0.50))
            <= ratio <= float(params.get("maximum_area_ratio", 2.00))
            and margin >= float(params.get("minimum_iou_margin", 0.20))
            and competitor < float(params.get(
                "maximum_competitor_coverage", 0.20)))
        rows.append({
            "previous_index": row, "current_index": col,
            "previous_identity": int(old["identity"]),
            "current_identity": int(current[col]["identity"]),
            "intersection_px": int(intersection[row, col]),
            "intersection_over_union": float(iou[row, col]),
            "source_coverage": float(source[row, col]),
            "destination_coverage": float(destination[row, col]),
            "area_ratio": float(ratio),
            "iou_margin": margin,
            "maximum_competitor_coverage": competitor,
            "previous_core_area_px": int(old["core_area"]),
            "current_core_area_px": int(current[col]["core_area"]),
            "centroid_displacement_px": float(np.linalg.norm(
                old["position"] - current[col]["position"])),
            "allowed": allowed,
        })
    return rows


def occupancy_replacement_audit(labels: np.ndarray, params: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for t in range(1, len(labels)):
        previous, current = _observations(labels[t - 1]), _observations(labels[t])
        for row in _strong_mutual_links(previous, current, params):
            rows.append({
                "from_t": t - 1, "to_t": t,
                "from_imagej_frame": t, "to_imagej_frame": t + 1,
                "identity_changed": bool(
                    row["previous_identity"] != row["current_identity"]),
                **row,
            })
    return pd.DataFrame(rows)


def _propagate_frame(previous_frame: np.ndarray, current_input: np.ndarray,
                     params: dict, attach_coreless: bool,
                     direction: str, from_t: int, to_t: int,
                     ) -> tuple[np.ndarray, list[dict]]:
    previous, current = _observations(previous_frame), _observations(current_input)
    links = [row for row in _strong_mutual_links(previous, current, params)
             if row["allowed"]]
    result = np.zeros_like(current_input)
    assigned_current: set[int] = set()
    used_identities: set[int] = set()
    decisions: list[dict] = []
    substantial: list[tuple[int, np.ndarray]] = []
    for row in sorted(links, key=lambda value: (
            -value["intersection_over_union"], value["previous_identity"])):
        col = int(row["current_index"])
        identity = int(row["previous_identity"])
        _paint(result, current[col], identity)
        assigned_current.add(col)
        used_identities.add(identity)
        if current[col]["core_area"] >= int(params.get(
                "minimum_substantial_core_px", 12)):
            substantial.append((identity, current[col]["position"]))
        if identity != int(current[col]["identity"]):
            decisions.append({
                "method": "same_territory_inheritance",
                "direction": direction, "from_t": from_t, "to_t": to_t,
                "from_imagej_frame": from_t + 1,
                "to_imagej_frame": to_t + 1,
                "input_identity": int(current[col]["identity"]),
                "output_identity": identity,
                **{key: value for key, value in row.items()
                   if key not in {"previous_index", "current_index"}},
            })

    unclaimed_input_names = [
        identity for identity in sorted(set(map(int, np.unique(current_input))) - {0})
        if identity not in used_identities]
    next_identity = max(
        int(previous_frame.max()), int(current_input.max()), 0) + 1
    for col, observation in sorted(enumerate(current),
                                   key=lambda pair: -pair[1]["area"]):
        if col in assigned_current:
            continue
        preferred = int(observation["identity"])
        assigned = preferred if preferred not in used_identities else 0
        method = "unmatched_input_identity"
        if (not assigned and attach_coreless
                and observation["core_area"] < int(params.get(
                    "minimum_substantial_core_px", 12)) and substantial):
            host = min(substantial, key=lambda item: float(np.linalg.norm(
                observation["position"] - item[1])))
            distance = float(np.linalg.norm(observation["position"] - host[1]))
            if distance <= float(params.get(
                    "maximum_coreless_host_distance_px", 24.0)):
                assigned = int(host[0])
                method = "coreless_fragment_attached_to_host"
        if not assigned:
            available = [value for value in unclaimed_input_names
                         if value not in used_identities]
            if available:
                assigned = int(available[0])
                method = "displacement_cycle_spare_identity"
            else:
                while next_identity in used_identities:
                    next_identity += 1
                assigned = next_identity
                next_identity += 1
                method = "unresolved_new_identity"
        _paint(result, observation, assigned)
        # A coreless host attachment may share its host's name by design.
        if method != "coreless_fragment_attached_to_host":
            used_identities.add(assigned)
        if observation["core_area"] >= int(params.get(
                "minimum_substantial_core_px", 12)):
            substantial.append((assigned, observation["position"]))
        if assigned != preferred or method == "coreless_fragment_attached_to_host":
            decisions.append({
                "method": method, "direction": direction,
                "from_t": from_t, "to_t": to_t,
                "from_imagej_frame": from_t + 1,
                "to_imagej_frame": to_t + 1,
                "input_identity": preferred, "output_identity": assigned,
                "area_px": int(observation["area"]),
                "core_area_px": int(observation["core_area"]),
            })
    if not np.array_equal(result > 0, current_input > 0):
        raise AssertionError("territory propagation changed foreground support")
    return result, decisions


def reconcile_same_territory_replacements(
        labels: np.ndarray, anchor_t: int, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Carry identities over unambiguous adjacent occupancy in both directions."""
    result = np.zeros_like(labels)
    result[anchor_t] = labels[anchor_t]
    decisions: list[dict] = []
    for t in range(anchor_t + 1, len(labels)):
        result[t], rows = _propagate_frame(
            result[t - 1], labels[t], params, False, "forward", t - 1, t)
        decisions.extend(rows)
    for t in range(anchor_t - 1, -1, -1):
        result[t], rows = _propagate_frame(
            result[t + 1], labels[t], params, False, "backward", t + 1, t)
        decisions.extend(rows)
    audit = occupancy_replacement_audit(labels, params)
    if not np.array_equal(result > 0, labels > 0):
        raise AssertionError("same-territory reunion changed foreground support")
    return result, pd.DataFrame(decisions), audit


def optimise_territory_ownership_graph(
        labels: np.ndarray, anchor_t: int, params: dict,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Traverse the whole-movie occupancy graph and keep coreless fragments as territory.

    Substantial somas inherit identity through mutual overlap.  If that reservation
    displaces a coreless fragment, the fragment attaches to the nearest substantial
    host while the absent identity remains latent for later reappearance.
    """
    result = np.zeros_like(labels)
    result[anchor_t] = labels[anchor_t]
    decisions: list[dict] = []
    for t in range(anchor_t + 1, len(labels)):
        result[t], rows = _propagate_frame(
            result[t - 1], labels[t], params, True, "forward", t - 1, t)
        decisions.extend(rows)
    for t in range(anchor_t - 1, -1, -1):
        result[t], rows = _propagate_frame(
            result[t + 1], labels[t], params, True, "backward", t + 1, t)
        decisions.extend(rows)
    audit = occupancy_replacement_audit(labels, params)
    if not np.array_equal(result > 0, labels > 0):
        raise AssertionError("territory ownership graph changed foreground support")
    return result, pd.DataFrame(decisions), audit
