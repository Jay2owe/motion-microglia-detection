"""Geometric contact between identity pairs, frame by frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from analysis.fields import longest_run as _longest_run
from analysis.registry import Column, MeasurementContext, Output, register


DEFAULTS = {
    "dilation_px": 2,
    "min_shared_boundary_px": 1,
    "include_unclaimed": False,
}

# ``_longest_run`` moved to ``analysis.fields`` when ``object_geometry`` needed
# the same count of consecutive observed frames. Same arithmetic, one copy.


#: A row here is a *pair*, not a cell, which is why both identities are named
#: rather than one being the index. ``dilation_px`` is on every row because
#: contact is a threshold and not an observation: two cells that touch at three
#: pixels of dilation and not at one are a different claim, and a table that
#: dropped the setting would read as though there were only one.
PRODUCES = (
    Column("identity_a", "First cell of the pair", "", "reference"),
    Column("identity_b", "Second cell of the pair", "", "reference"),
    Column("dilation_px", "Dilation that counted as touching", "px", "reference"),
    # contacts_frame - one row per pair per frame
    Column("shared_boundary_px", "Shared boundary", "px", "surveillance"),
    Column("centroid_distance", "Distance between centroids", "px", "motility"),
    Column("overlap_px", "Pixels claimed by both", "px", "unclaimed"),
    Column("boundary_share_a", "Share of the first cell's boundary in contact", "fraction", "surveillance"),
    Column("boundary_share_b", "Share of the second cell's boundary in contact", "fraction", "surveillance"),
    Column("handoff_coincident", "A tracker handoff happened here", "0 or 1", "inferred"),
    # contacts - one row per pair over the recording
    Column("frames_in_contact", "Frames in contact", "frames", "surveillance"),
    Column("hours_in_contact", "Time in contact", "h", "surveillance"),
    Column("first_hour", "First contact", "h", "surveillance"),
    Column("last_hour", "Last contact", "h", "surveillance"),
    Column("longest_bout_frames", "Longest unbroken contact", "frames", "surveillance"),
    Column("mean_shared_boundary_px", "Shared boundary, mean", "px", "surveillance"),
)


WRITES = (
    Output("contacts",       grain=("identity_a", "identity_b", "dilation_px")),
    Output("contacts_frame", grain=("identity_a", "identity_b", "frame_index", "dilation_px")),
)


@register(
    name="contacts",
    description="Identity pairs sharing a boundary in each frame and over the recording",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("contacts")}
    configured = params["dilation_px"]
    dilations = [int(value) for value in configured] if isinstance(configured, (list, tuple)) else [int(configured)]
    if any(value < 1 for value in dilations):
        raise ValueError("contacts dilation_px must be one or more positive integers")
    minimum = int(params["min_shared_boundary_px"])
    rows: list[dict] = []

    for frame_index in range(context.n_frames):
        frame = context.labels[frame_index]
        masks: dict[int, np.ndarray] = {
            int(identity): frame == identity for identity in np.unique(frame) if identity
        }
        if bool(params["include_unclaimed"]) and context.unclaimed is not None:
            masks[0] = context.unclaimed[frame_index] > 0
        identities = sorted(masks)
        boundaries = {
            identity: mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
            for identity, mask in masks.items()
        }
        centres = {identity: np.asarray(ndi.center_of_mass(mask), dtype=float) for identity, mask in masks.items()}
        for dilation in dilations:
            structure = ndi.iterate_structure(ndi.generate_binary_structure(2, 2), dilation)
            expanded = {
                identity: ndi.binary_dilation(mask, structure=structure)
                for identity, mask in masks.items()
            }
            # Ask the dilated masks which labels they can actually reach before
            # computing shared boundaries.  A field with 80 cells has 3,160
            # possible pairs but only tens of neighbours; testing every full-
            # frame boolean pair made runtime grow with empty space.
            candidates: set[tuple[int, int]] = set()
            for identity, reached in expanded.items():
                for neighbour in (int(value) for value in np.unique(frame[reached]) if value):
                    if neighbour != identity and neighbour in masks:
                        candidates.add(tuple(sorted((identity, neighbour))))
                if 0 in masks and identity != 0 and np.any(reached & masks[0]):
                    candidates.add((0, identity))

            for identity_a, identity_b in sorted(candidates):
                contact_a = boundaries[identity_a] & expanded[identity_b]
                contact_b = boundaries[identity_b] & expanded[identity_a]
                contact_a_px = int(contact_a.sum())
                contact_b_px = int(contact_b.sum())
                shared = max(contact_a_px, contact_b_px)
                if shared < minimum:
                    continue
                boundary_a = int(boundaries[identity_a].sum())
                boundary_b = int(boundaries[identity_b].sum())
                distance_px = float(np.linalg.norm(centres[identity_a] - centres[identity_b]))
                rows.append(
                    {
                        "identity_a": identity_a,
                        "identity_b": identity_b,
                        "frame_index": frame_index,
                        "hours": context.scale.hours(frame_index + context.source_frame_offset),
                        "dilation_px": dilation,
                        "shared_boundary_px": shared,
                        "centroid_distance": context.scale.length(distance_px),
                        "overlap_px": int(np.count_nonzero(masks[identity_a] & masks[identity_b])),
                        "boundary_share_a": float(contact_a_px / boundary_a) if boundary_a else np.nan,
                        "boundary_share_b": float(contact_b_px / boundary_b) if boundary_b else np.nan,
                        # Accepted handoff events are not present in the
                        # image-only context. The explicit false column is
                        # retained so a later join can flag, never drop, them.
                        "handoff_coincident": False,
                    }
                )

    columns = [
        "identity_a", "identity_b", "frame_index", "hours", "dilation_px",
        "shared_boundary_px", "centroid_distance", "overlap_px",
        "boundary_share_a", "boundary_share_b", "handoff_coincident",
    ]
    contacts_frame = pd.DataFrame(rows, columns=columns)
    summary_rows: list[dict] = []
    if not contacts_frame.empty:
        for keys, group in contacts_frame.groupby(["identity_a", "identity_b", "dilation_px"], sort=True):
            frames = group["frame_index"].to_numpy(int)
            summary_rows.append(
                {
                    "identity_a": int(keys[0]),
                    "identity_b": int(keys[1]),
                    "dilation_px": int(keys[2]),
                    "frames_in_contact": int(group["frame_index"].nunique()),
                    "hours_in_contact": context.scale.hours(group["frame_index"].nunique()),
                    "first_hour": float(group["hours"].min()),
                    "last_hour": float(group["hours"].max()),
                    "longest_bout_frames": _longest_run(frames),
                    "mean_shared_boundary_px": float(group["shared_boundary_px"].mean()),
                    "overlap_px": int(group["overlap_px"].sum()),
                    "handoff_coincident": bool(group["handoff_coincident"].any()),
                }
            )
    contacts = pd.DataFrame(summary_rows, columns=[
        "identity_a", "identity_b", "dilation_px", "frames_in_contact",
        "hours_in_contact", "first_hour", "last_hour", "longest_bout_frames",
        "mean_shared_boundary_px", "overlap_px", "handoff_coincident",
    ])
    return {"contacts": contacts, "contacts_frame": contacts_frame}
