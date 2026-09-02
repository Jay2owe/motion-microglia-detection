"""How much of each outline's identity the tracker reconstructed.

Every other module measures what is inside an outline. This one measures how the
outline came to have the name it has: of the pixels this cell owns in this
frame, for how many did the microscope show this cell there, and for how many
did the tracker decide the owner by reconstruction?

A weather map is the everyday version. Some of it is stations reporting and some
is interpolation between stations. Both are drawn the same way and both are
usually right, but a reader who cannot tell them apart cannot tell which parts
of the map to lean on.

Read the flag carefully, because it is easy to overstate. It is about the
*name*, not the picture, and the sidecar separates the two rather than asking a
reader to take that on trust. On the accepted `95_A3`, of 404,337 flagged
pixels inside an outline, 403,765 were shown by the microscope with only the
owner supplied by inference (``renamed_px``) and 572 are outline the tracker
supplied where the segmentation saw nothing (``added_px``). So a high fraction
does not mean the cell was drawn out of nothing; it means that which cell those
pixels belong to rests on the tracker's reasoning rather than on seeing that
cell in that frame. Every per-cell number depends on exactly that, which is why
it is worth counting.

Nor is it a defect being reported. Reconstruction is what lets a cell keep its
name through a frame where it dimmed below threshold or sat inside a merged
object, and without it the tables would be full of spurious deaths and births.
But a turnover figure computed over an attribution the tracker reasoned its way
to is partly measuring the reasoning, and that is only checkable if the share is
written down.

Columns per cell per frame:

``inferred_px``
    pixels of this cell's footprint whose owner was reconstructed.
``observed_px``
    the rest - ``area_px`` minus ``inferred_px``, by construction.
``renamed_px``
    the part of ``inferred_px`` the microscope did show: something was
    there, and the tracker decided whose it was.
``added_px``
    the part of ``inferred_px`` the microscope did not show: no detection
    here at all, so this pixel of the outline is the tracker's. Expect a
    very small number, and read a large one as a warning about the frame.

    On the pinned movie it is not spread thinly: the 572 supplied pixels are
    17 whole cell-frames of 6,073, each with ``observed_px`` and
    ``renamed_px`` at zero, where a held position was carried through a frame
    the detection did not support. A row with ``added_px == area`` is that
    case, and is the one worth looking at.

Plus ``unresolved_px``: foreground overlapping this cell that the tracker
refused to attribute to anyone. The per-frame table carries the same counts for
the whole field, including the unresolved foreground that sits outside every
outline and so belongs to no cell at all. Forcing it onto the nearest identity
would invent exactly the attribution the tracker declined to make.

Nothing here changes a measured value. It says what the measurements rest on;
what to do about that is a scientific decision, not a module's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register


#: The per-cell counts and their whole-field counterparts. The ``inferred``
#: role is deliberately not the ``named`` one: these labels have to keep saying
#: "reconstructed" wherever they land, because the whole point of the module is
#: that a reader can tell the two apart.
PRODUCES = (
    # provenance - one row per cell per frame
    Column("inferred_px", "Pixels attributed by reconstruction", "px", "inferred"),
    Column("observed_px", "Pixels attributed by observation", "px", "named"),
    Column("inferred_fraction", "Reconstructed share of the outline", "fraction", "inferred"),
    Column("renamed_px", "Pixels seen, owner reconstructed", "px", "inferred"),
    Column("added_px", "Outline pixels the detection never saw", "px", "inferred"),
    Column("unresolved_px", "Unresolved pixels overlapping the cell", "px", "unclaimed"),
    # provenance_frame - one row per frame, whole field
    Column("inferred_px_field", "Reconstructed pixels in the field", "px", "inferred"),
    Column("inferred_px_named", "Reconstructed pixels inside an outline", "px", "inferred"),
    Column("inferred_px_unowned", "Reconstructed pixels outside every outline", "px", "inferred"),
    Column("inferred_fraction_field", "Reconstructed share of named foreground", "fraction", "inferred"),
    Column("added_px_field", "Outline pixels the detection never saw, whole field", "px", "inferred"),
    Column("unresolved_px_field", "Unresolved pixels in the field", "px", "unclaimed"),
    Column("unresolved_px_unowned", "Unresolved pixels outside every outline", "px", "unclaimed"),
)


WRITES = (
    Output("provenance",       grain=("identity", "frame_index"), fold=True),
    Output("provenance_frame", grain=("frame_index",)),
)


@register(
    name="provenance",
    description="How much of each cell's footprint was attributed by reconstruction rather than by observation",
    requires=("labels", "inferred"),
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    labels = context.labels
    inferred = context.inferred
    unresolved = context.unresolved
    added = context.added
    n_frames = context.n_frames

    rows: list[dict] = []
    frame_inferred = np.zeros(n_frames, dtype=np.int64)
    frame_inferred_unowned = np.zeros(n_frames, dtype=np.int64)
    frame_unresolved = np.zeros(n_frames, dtype=np.int64)
    frame_unresolved_unowned = np.zeros(n_frames, dtype=np.int64)
    frame_added = np.zeros(n_frames, dtype=np.int64)

    for frame_index in range(n_frames):
        frame = labels[frame_index]
        reconstructed = inferred[frame_index]
        undecided = None if unresolved is None else unresolved[frame_index]
        supplied = None if added is None else added[frame_index]
        named = frame > 0

        frame_inferred[frame_index] = int(np.count_nonzero(reconstructed))
        frame_inferred_unowned[frame_index] = int(
            np.count_nonzero(reconstructed & ~named))
        if undecided is not None:
            frame_unresolved[frame_index] = int(np.count_nonzero(undecided))
            frame_unresolved_unowned[frame_index] = int(
                np.count_nonzero(undecided & ~named))
        if supplied is not None:
            frame_added[frame_index] = int(np.count_nonzero(supplied))

        for identity in (int(value) for value in np.unique(frame) if value):
            mask = frame == identity
            area = int(np.count_nonzero(mask))
            marked = int(np.count_nonzero(mask & reconstructed))
            rows.append(
                {
                    "identity": identity,
                    "frame_index": frame_index,
                    "inferred_px": marked,
                    "observed_px": area - marked,
                    # np.nan, not 0: a cell with no pixels was not attributed by
                    # observation, it was not measured at all.
                    "inferred_fraction": marked / area if area else np.nan,
                    # Counted against the flag rather than subtracted from
                    # it, so a sidecar where the two ever stop nesting says
                    # so here instead of quietly reporting a negative.
                    "renamed_px": marked if supplied is None else int(
                        np.count_nonzero(mask & reconstructed & ~supplied)),
                    "added_px": 0 if supplied is None else int(
                        np.count_nonzero(mask & supplied)),
                    "unresolved_px": 0 if undecided is None else int(
                        np.count_nonzero(mask & undecided)),
                }
            )

    per_cell_frame = pd.DataFrame(rows, columns=[
        "identity", "frame_index", "inferred_px", "observed_px",
        "inferred_fraction", "renamed_px", "added_px", "unresolved_px",
    ])

    per_frame = context.frame_table()
    per_frame["inferred_px_field"] = frame_inferred
    per_frame["inferred_px_unowned"] = frame_inferred_unowned
    per_frame["unresolved_px_field"] = frame_unresolved
    per_frame["unresolved_px_unowned"] = frame_unresolved_unowned
    # Bit 2 is only ever set inside an outline, so it has no unowned
    # counterpart to report.
    per_frame["added_px_field"] = frame_added
    named_px = np.array(
        [int(np.count_nonzero(labels[i])) for i in range(n_frames)], dtype=np.int64)
    per_frame["inferred_px_named"] = frame_inferred - frame_inferred_unowned
    per_frame["inferred_fraction_field"] = np.where(
        named_px > 0,
        (frame_inferred - frame_inferred_unowned) / np.where(named_px > 0, named_px, 1),
        np.nan,
    )

    return {"provenance": per_cell_frame, "provenance_frame": per_frame}
