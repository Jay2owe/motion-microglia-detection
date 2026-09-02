"""Exact connected-component accounting shared by accepted-history rules."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as label_values


STRUCTURE = np.ones((3, 3), np.uint8)


def component_excess(frame: np.ndarray, identity: int) -> int:
    """Return components beyond the first for one identity in one frame."""
    count = int(ndi.label(frame == int(identity), STRUCTURE)[1])
    return max(0, count - 1)


def new_duplicate_components(baseline: np.ndarray,
                             candidate: np.ndarray) -> int:
    """Count new same-identity components using only changed owner-frames.

    An unchanged identity mask has exactly the same component count. Restricting
    the calculation to identities whose pixels changed is therefore
    mathematically identical to scanning every identity in every frame.
    """
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate label stacks must align")
    added = 0
    changed_frames = np.flatnonzero(np.any(
        baseline != candidate, axis=tuple(range(1, baseline.ndim))))
    for frame in changed_frames:
        before = baseline[int(frame)]
        after = candidate[int(frame)]
        changed = before != after
        owners = np.unique(np.concatenate((before[changed], after[changed])))
        for identity in owners:
            if int(identity) <= 0:
                continue
            added += max(
                0,
                component_excess(after, int(identity))
                - component_excess(before, int(identity)),
            )
    return int(added)


def label_value_components(
        frame: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label equal-valued 8-connected regions in one image pass."""
    components = label_values(frame, background=0, connectivity=2)
    count = int(components.max())
    if count == 0:
        return (components, np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64))
    indices = np.arange(1, count + 1)
    areas = np.bincount(components.ravel(), minlength=count + 1)[1:]
    owners = ndi.maximum(frame, labels=components, index=indices).astype(int)
    return components, owners, areas.astype(np.int64, copy=False)


def component_areas_by_identity(frame: np.ndarray) -> dict[int, np.ndarray]:
    """Return all 8-connected component areas after one value-aware pass."""
    _, owners, areas = label_value_components(frame)
    result: dict[int, list[int]] = {}
    for owner, area in zip(owners, areas):
        result.setdefault(int(owner), []).append(int(area))
    return {owner: np.asarray(values, dtype=np.int64)
            for owner, values in result.items() if owner > 0}


def total_component_excess(stack: np.ndarray,
                           minimum_area: int = 1) -> int:
    """Count components beyond the first for every owner and frame."""
    total = 0
    for frame in stack:
        for areas in component_areas_by_identity(frame).values():
            substantial = int(np.count_nonzero(areas >= int(minimum_area)))
            total += max(0, substantial - 1)
    return int(total)


def duplicate_owner_frames(stack: np.ndarray,
                           minimum_area: int = 1) -> int:
    """Count owner-frames containing more than one substantial component."""
    rows = 0
    for frame in stack:
        for areas in component_areas_by_identity(frame).values():
            rows += int(np.count_nonzero(areas >= int(minimum_area)) > 1)
    return int(rows)
