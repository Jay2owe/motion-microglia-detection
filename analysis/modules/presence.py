"""Whether each cell's name is on the picture, frame by frame.

Every other module measures a cell in the frames where it appears. This one
measures the frames where it does not. A microscope roll-call: the register is
read out in every frame, and what is recorded is who answered, who was expected
and did not, and who had not yet arrived.

The distinction matters because a missing name is not a missing cell. A cell
that spends six frames inside a merged object is still there and still lit; the
outline it is inside simply carries the other cell's number. Silence in the
tables therefore has two very different causes, and nothing downstream can tell
them apart unless the absence itself is written down.

Three states per cell per frame:

``named``
    the identity carries a mask in this frame.
``unclaimed``
    the frame lies inside the cell's own lifespan, first appearance to last,
    but the name is not on screen. This is the gap the tracker is judged on.
``outside_lifespan``
    before the cell first appeared or after it was last seen. Not a fault, and
    counted separately so it can never be added to the gaps by accident.

The per-frame table adds the other half of the same question - foreground
pixels the pipeline declines to attribute to *any* identity. Keeping them
visible rather than forcing them onto the nearest name is a deliberate choice
of the tracking half, and it is only checkable if it is counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register

#: The three states, in the order a figure should stack or legend them.
STATES = ("named", "unclaimed", "outside_lifespan")


#: Two tables: the roll-call per cell per frame, and the field totals per frame.
#: ``state`` is the only categorical column the package writes, and its three
#: values each have their own colour role - see ``STATES`` above - so the column
#: itself takes ``reference`` rather than pretending to be one of them.
PRODUCES = (
    # presence - one row per cell per frame
    Column("state", "Whether the name is on screen", "named, unclaimed or outside lifespan", "reference"),
    Column("first_frame_index", "First frame held", "frame", "reference"),
    Column("last_frame_index", "Last frame held", "frame", "reference"),
    # presence_frame - one row per frame, whole field
    Column("identities_named", "Named cells", "count", "named"),
    Column("identities_unclaimed", "Expected cells without an identity in this frame", "count", "unclaimed"),
    Column("identities_expected", "Expected cells within lifespan", "count", "reference"),
    Column("identities_total", "Cell identities in the recording", "count", "reference"),
    Column("assigned_px", "Foreground pixels assigned to a cell", "px", "named"),
    Column("unclaimed_px", "Unclaimed foreground pixels", "px", "unclaimed"),
    Column("foreground_px", "All detected foreground pixels", "px", "reference"),
    Column("unclaimed_fraction", "Unclaimed foreground share", "fraction", "unclaimed"),
)


#: ``presence`` is at cell-frame grain and is still not folded into
#: ``cell_frame``. It carries a row for every cell in every frame *including*
#: the frames where the cell is absent, which is the whole point of the module,
#: so it has more rows than ``cell_frame`` has. Folding it would either lose the
#: absences or fill cell_frame with mostly blank rows.
WRITES = (
    Output("presence",       grain=("identity", "frame_index")),
    Output("presence_frame", grain=("frame_index",)),
)


@register(
    name="presence",
    description="Whether each identity's name is on screen in each frame, and how much foreground carries no name",
    requires=("labels",),
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    labels = context.labels
    n_frames = context.n_frames

    # One pass over the stack: who is on screen in each frame, and how many
    # pixels carry a name at all.
    on_screen: list[set[int]] = []
    assigned_px = np.zeros(n_frames, dtype=np.int64)
    for frame_index in range(n_frames):
        frame = labels[frame_index]
        on_screen.append({int(value) for value in np.unique(frame) if value})
        assigned_px[frame_index] = int(np.count_nonzero(frame))

    identities = sorted({identity for frame in on_screen for identity in frame})
    frames = context.frame_table()
    hours = frames["hours"].to_numpy(float)

    rows: list[dict] = []
    for identity in identities:
        seen = [index for index in range(n_frames) if identity in on_screen[index]]
        first, last = seen[0], seen[-1]
        for frame_index in range(n_frames):
            if identity in on_screen[frame_index]:
                state = "named"
            elif first <= frame_index <= last:
                state = "unclaimed"
            else:
                state = "outside_lifespan"
            rows.append(
                {
                    "identity": identity,
                    "frame_index": frame_index,
                    "hours": hours[frame_index],
                    "state": state,
                    "first_frame_index": first,
                    "last_frame_index": last,
                }
            )
    presence = pd.DataFrame(rows, columns=[
        "identity", "frame_index", "hours", "state", "first_frame_index", "last_frame_index",
    ])

    # Per frame: how many names answered, how many were expected, and how much
    # of the foreground nobody claimed.
    per_frame = frames.copy()
    if presence.empty:
        per_frame["identities_named"] = 0
        per_frame["identities_expected"] = 0
        per_frame["identities_unclaimed"] = 0
    else:
        counts = (
            presence.pivot_table(
                index="frame_index", columns="state", values="identity", aggfunc="count"
            )
            .reindex(columns=list(STATES), fill_value=0)
            .fillna(0)
            .astype(int)
        )
        per_frame = per_frame.merge(
            counts.rename(columns={
                "named": "identities_named",
                "unclaimed": "identities_unclaimed",
            })[["identities_named", "identities_unclaimed"]].reset_index(),
            on="frame_index",
            how="left",
        )
        per_frame["identities_expected"] = (
            per_frame["identities_named"] + per_frame["identities_unclaimed"]
        )

    per_frame["identities_total"] = len(identities)
    per_frame["assigned_px"] = assigned_px
    if context.unclaimed is not None:
        unclaimed_px = np.array(
            [int(np.count_nonzero(context.unclaimed[i])) for i in range(n_frames)], dtype=np.int64
        )
        per_frame["unclaimed_px"] = unclaimed_px
        foreground = unclaimed_px + assigned_px
        per_frame["foreground_px"] = foreground
        per_frame["unclaimed_fraction"] = np.where(
            foreground > 0, unclaimed_px / np.where(foreground > 0, foreground, 1), np.nan
        )

    return {"presence": presence, "presence_frame": per_frame}
