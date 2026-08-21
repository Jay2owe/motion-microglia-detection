from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def identity_confidence_table(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> pd.DataFrame:
    """Classify identities before tracking so small/dim objects cannot lead it."""
    if labels.shape != raw.shape:
        raise ValueError("labels and raw have incompatible shapes")
    ring_width = int(params.get("confidence_background_ring_px", 3))
    rows: list[dict] = []
    for t, frame in enumerate(labels):
        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            mask = frame == identity
            ring = ndi.binary_dilation(mask, iterations=ring_width) & ~mask
            foreground_free = ring & (frame == 0)
            background = raw[t][foreground_free]
            background_median = (float(np.median(background))
                                 if background.size else 0.0)
            mean = float(raw[t][mask].mean())
            rows.append({
                "identity": identity,
                "t": t,
                "imagej_frame": t + 1,
                "area_px": int(mask.sum()),
                "mean_intensity": mean,
                "local_background_median": background_median,
                "local_contrast_ratio": ((mean + 1.0)
                                         / (background_median + 1.0)),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "identity", "observed_frames", "first_imagej_frame",
            "last_imagej_frame", "median_area_px", "median_mean_intensity",
            "median_local_contrast_ratio", "confidence_layer", "confident"])
    frame_table = pd.DataFrame(rows)
    summary = (frame_table.groupby("identity", as_index=False)
               .agg(observed_frames=("t", "size"),
                    first_imagej_frame=("imagej_frame", "min"),
                    last_imagej_frame=("imagej_frame", "max"),
                    median_area_px=("area_px", "median"),
                    median_mean_intensity=("mean_intensity", "median"),
                    median_local_contrast_ratio=("local_contrast_ratio", "median")))
    summary["confident"] = (
        (summary.observed_frames >= int(params.get(
            "minimum_confident_observations", 8)))
        & (summary.median_area_px >= float(params.get(
            "minimum_confident_area_px", 80.0)))
        & (summary.median_local_contrast_ratio >= float(params.get(
            "minimum_confident_local_contrast", 1.05))))
    summary["confidence_layer"] = np.where(
        summary.confident, "layer_1_established", "layer_2_small_or_dim")
    return summary


def multilayer_identity_confidence_table(
        labels: np.ndarray, raw: np.ndarray, params: dict,
        ) -> pd.DataFrame:
    """Split the established group by motion and nearby smaller neighbours.

    Motion uses the 75th percentile of adjacent-frame centroid steps so an
    occasional outline wobble does not turn a static soma into a mobile one.
    Neighbour distance is the gap between equivalent-area circles, not raw
    centroid distance, so a small cell next to a large soma is treated as near.
    """
    summary = identity_confidence_table(labels, raw, {
        "minimum_confident_observations": int(params.get(
            "minimum_confident_observations", 8)),
        "minimum_confident_area_px": float(params.get(
            "minimum_large_area_px", 80.0)),
        "minimum_confident_local_contrast": float(params.get(
            "minimum_local_contrast", 1.05)),
        "confidence_background_ring_px": int(params.get(
            "confidence_background_ring_px", 3)),
    })
    if summary.empty:
        for column in ("median_step_px", "p75_step_px",
                       "near_smaller_frame_fraction"):
            summary[column] = pd.Series(dtype=float)
        return summary

    confident_ids = set(summary.loc[
        summary.confident, "identity"].astype(int))
    frame_objects: dict[int, list[dict]] = {}
    histories: dict[int, list[tuple[int, np.ndarray]]] = {}
    for t, frame in enumerate(labels):
        objects: list[dict] = []
        for identity in sorted(set(map(int, np.unique(frame))) - {0}):
            coordinates = np.column_stack(np.nonzero(frame == identity))
            area = int(len(coordinates))
            position = coordinates.mean(axis=0)
            objects.append({
                "identity": identity,
                "position": position,
                "radius": float(np.sqrt(area / np.pi)),
            })
            histories.setdefault(identity, []).append((t, position))
        frame_objects[t] = objects

    step_metrics: dict[int, tuple[float, float]] = {}
    for identity, history in histories.items():
        steps = [
            float(np.linalg.norm(later_position - earlier_position))
            for (earlier_t, earlier_position), (later_t, later_position)
            in zip(history, history[1:]) if later_t == earlier_t + 1]
        step_metrics[identity] = (
            float(np.median(steps)) if steps else np.nan,
            float(np.percentile(steps, 75)) if steps else np.nan,
        )

    smaller_ids = (set(map(int, summary.identity)) - confident_ids)
    near_distance = float(params.get("near_smaller_edge_gap_px", 8.0))
    nearby: dict[int, list[bool]] = {identity: [] for identity in confident_ids}
    for objects in frame_objects.values():
        smaller = [row for row in objects if row["identity"] in smaller_ids]
        for row in objects:
            identity = int(row["identity"])
            if identity not in confident_ids:
                continue
            gaps = [max(
                float(np.linalg.norm(row["position"] - other["position"]))
                - float(row["radius"]) - float(other["radius"]), 0.0)
                for other in smaller]
            nearby[identity].append(bool(gaps and min(gaps) <= near_distance))

    summary["median_step_px"] = summary.identity.map(
        lambda value: step_metrics.get(int(value), (np.nan, np.nan))[0])
    summary["p75_step_px"] = summary.identity.map(
        lambda value: step_metrics.get(int(value), (np.nan, np.nan))[1])
    summary["near_smaller_frame_fraction"] = summary.identity.map(
        lambda value: float(np.mean(nearby.get(int(value), [False]))))

    maximum_static_step = float(params.get(
        "static_maximum_p75_step_px", 2.0))
    minimum_near_fraction = float(params.get(
        "minimum_near_smaller_frame_fraction", 0.10))
    layers: list[str] = []
    for row in summary.itertuples():
        if not bool(row.confident):
            layer = "small_or_dim"
        elif (not np.isfinite(float(row.p75_step_px))
              or float(row.p75_step_px) > maximum_static_step):
            layer = "large_mobile"
        elif float(row.near_smaller_frame_fraction) >= minimum_near_fraction:
            layer = "large_static_near_smaller"
        else:
            layer = "large_static_isolated"
        layers.append(layer)
    summary["confidence_layer"] = layers
    return summary


def high_confidence_labels(labels: np.ndarray, confidence: pd.DataFrame,
                           ) -> np.ndarray:
    """Return only first-layer identities without renumbering them."""
    high_ids = set(confidence.loc[confidence.confident, "identity"].astype(int))
    return np.where(np.isin(labels, list(high_ids)), labels, 0).astype(labels.dtype)


def restore_deferred_identity_layer(
        tracked: np.ndarray, physical_objects: np.ndarray,
        reference_labels: np.ndarray, high_identity_ids: set[int],
        ) -> tuple[np.ndarray, pd.DataFrame]:
    """Add small/dim identities after established tracking without stealing it."""
    if tracked.shape != physical_objects.shape or tracked.shape != reference_labels.shape:
        raise ValueError("tracked, physical and reference labels are incompatible")
    result = tracked.copy()
    rows: list[dict] = []
    high_ids = {int(value) for value in high_identity_ids}
    # First return any unclaimed established name to its own reference soma. A
    # genuinely redirected established lineage is already present elsewhere and
    # therefore remains untouched for the red-led evidence to resolve.
    for t, frame in enumerate(physical_objects):
        records: list[tuple[int, np.ndarray, int, int]] = []
        used_high = set(map(int, np.unique(result[t]))) & high_ids
        for local in sorted(set(map(int, np.unique(frame))) - {0}):
            mask = frame == local
            tracked_values, tracked_counts = np.unique(
                result[t][mask], return_counts=True)
            tracked_choices = [(int(count), int(value)) for value, count
                               in zip(tracked_values, tracked_counts)
                               if int(value) > 0]
            tracked_identity = max(tracked_choices)[1] if tracked_choices else 0
            old_values, old_counts = np.unique(
                reference_labels[t][mask], return_counts=True)
            old_choices = [(int(count), int(value)) for value, count
                           in zip(old_values, old_counts)
                           if int(value) > 0]
            reference_identity = max(old_choices)[1] if old_choices else 0
            records.append((local, mask, tracked_identity, reference_identity))
        for local, mask, tracked_identity, reference_identity in records:
            if (reference_identity not in high_ids
                    or tracked_identity in high_ids
                    or reference_identity in used_high):
                continue
            result[t][mask] = reference_identity
            used_high.add(reference_identity)
            rows.append({
                "t": t, "imagej_frame": t + 1, "local_object": local,
                "tracked_identity": tracked_identity,
                "restored_identity": reference_identity,
                "pixels": int(mask.sum()),
                "restoration_layer": "unclaimed_established_name",
            })

    for t, frame in enumerate(physical_objects):
        for local in sorted(set(map(int, np.unique(frame))) - {0}):
            mask = frame == local
            tracked_values, tracked_counts = np.unique(
                result[t][mask], return_counts=True)
            tracked_choices = [(int(count), int(value)) for value, count
                               in zip(tracked_values, tracked_counts)
                               if int(value) > 0]
            tracked_identity = max(tracked_choices)[1] if tracked_choices else 0
            if tracked_identity in high_ids:
                continue
            old_values, old_counts = np.unique(
                reference_labels[t][mask], return_counts=True)
            old_choices = [(int(count), int(value)) for value, count
                           in zip(old_values, old_counts)
                           if int(value) > 0 and int(value) not in high_ids]
            if not old_choices:
                continue
            restored = max(old_choices)[1]
            result[t][mask] = restored
            rows.append({
                "t": t, "imagej_frame": t + 1, "local_object": local,
                "tracked_identity": tracked_identity,
                "restored_identity": restored,
                "pixels": int(mask.sum()),
                "restoration_layer": "deferred_small_or_dim_name",
            })
    reference_ids = set(map(int, np.unique(reference_labels))) - {0}
    novel_ids = (set(map(int, np.unique(result))) - {0}) - reference_ids
    if novel_ids:
        for t in range(len(result)):
            present = (set(map(int, np.unique(result[t]))) - {0}) & novel_ids
            if not present:
                continue
            changed = int(np.count_nonzero(result[t] != reference_labels[t]))
            result[t] = reference_labels[t]
            rows.append({
                "t": t, "imagej_frame": t + 1, "local_object": 0,
                "tracked_identity": ";".join(map(str, sorted(present))),
                "restored_identity": 0,
                "pixels": changed,
                "restoration_layer": "ambiguous_frame_reverted_no_new_name",
            })
    return result, pd.DataFrame(rows)
