from __future__ import annotations

import numpy as np
import pandas as pd

from common import validate_labels
from tracking import track_from_anchor


def _parse_green_tracks(value: object) -> set[int]:
    text = str(value)
    if text in ("", "nan"):
        return set()
    return {int(item) for item in text.split(";") if item}


def green_evidence_for_fixed_objects(
        fixed_objects: np.ndarray, observation_labels: np.ndarray,
        observation_table: pd.DataFrame) -> pd.DataFrame:
    """Transfer green motion-track evidence onto the fixed final objects by overlap."""
    lookup = {
        (int(row.t), int(row.local_id)): _parse_green_tracks(row.green_track_ids)
        for row in observation_table.itertuples()
    }
    rows: list[dict] = []
    for t in range(len(fixed_objects)):
        for local_id in np.unique(fixed_objects[t]):
            local_id = int(local_id)
            if local_id <= 0:
                continue
            overlaps = set(map(int, np.unique(
                observation_labels[t][fixed_objects[t] == local_id]))) - {0}
            tracks: set[int] = set()
            for observation_id in overlaps:
                tracks |= lookup.get((t, observation_id), set())
            rows.append({
                "t": t,
                "local_id": local_id,
                "green_track_ids": ";".join(map(str, sorted(tracks))),
            })
    return pd.DataFrame(rows)


def _frame_bijections(source: np.ndarray, candidate: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for t in range(len(source)):
        source_ids = sorted(set(map(int, np.unique(source[t]))) - {0})
        candidate_ids = sorted(set(map(int, np.unique(candidate[t]))) - {0})
        if len(source_ids) != len(candidate_ids):
            raise AssertionError(f"frame {t}: fixed-object count changed")
        mapped_targets: list[int] = []
        for source_id in source_ids:
            values = set(map(int, np.unique(candidate[t][source[t] == source_id]))) - {0}
            if len(values) != 1:
                raise AssertionError(
                    f"frame {t}: source object {source_id} did not receive one identity")
            target_id = values.pop()
            mapped_targets.append(target_id)
            rows.append({
                "t": t, "imagej_frame": t + 1,
                "source_object": source_id, "persistent_identity": target_id,
            })
        if len(set(mapped_targets)) != len(mapped_targets):
            raise AssertionError(f"frame {t}: two simultaneous objects share one identity")
    return pd.DataFrame(rows)


def retrack_fixed_objects(
        fixed_objects: np.ndarray, raw: np.ndarray, lag: np.ndarray,
        anchor_t: int, params: dict,
        green_table: pd.DataFrame | None = None,
        ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Reassign identity names while preserving every supplied frame object exactly."""
    if fixed_objects.shape != raw.shape:
        raise ValueError("fixed objects and raw movie differ in shape")
    validate_labels(fixed_objects)
    candidate, links = track_from_anchor(
        fixed_objects, raw, lag, anchor_t, params, green_table)
    validate_labels(candidate)
    if not np.array_equal(candidate > 0, fixed_objects > 0):
        raise AssertionError("persistence changed foreground support")
    mappings = _frame_bijections(fixed_objects, candidate)
    return candidate, links, mappings
