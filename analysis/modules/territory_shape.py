"""Territory against the cell that made it.

``territory`` says how big a cell's eventual footprint is and what shape it is.
This says how that footprint compares with the cell itself: a patch worth ten
cell-areas is a cell that ranged; a patch worth two is a cell that sat.

Three divisions, and every one of them is between numbers two other modules
already measured. That is why this is derived rather than part of ``territory``:
the instantaneous area is ``morphology``'s, and recomputing it inside
``territory`` would be a second pass over the pixels to avoid a join the roll-up
does for nothing.

``coverage_share`` used to live inside figure 26's builder, where it was a
number only that one page could see. It is a column now.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register_derived


DEFAULTS: dict = {}


#: ``coverage_share`` keeps the wording it had while it was a figure's private
#: arithmetic, so no axis that already draws it changes.
PRODUCES = (
    # territory_shape_frame - one row per cell per frame
    Column("coverage_share", "Eventual territory reached", "fraction", "surveillance"),
    Column("area_over_territory", "Cell area against its eventual territory",
           "fraction", "surveillance"),
    # territory_shape - one row per cell
    Column("territory_over_area", "Eventual territory in cell-areas", "ratio", "surveillance"),
)


WRITES = (
    Output("territory_shape",       grain=("identity",),               fold=True),
    Output("territory_shape_frame", grain=("identity", "frame_index"), fold=True),
)


@register_derived(
    name="territory_shape",
    description="Eventual territory against the cell that made it, per frame and per cell",
    needs_columns=("identity", "frame_index", "cumulative_unique_px"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def derive(cell_frame: pd.DataFrame, context: MeasurementContext) -> dict[str, pd.DataFrame]:
    keys = ["identity", "frame_index"]
    working = cell_frame[
        [column for column in (*keys, "cumulative_unique_px", "area_px")
         if column in cell_frame.columns]
    ].copy()

    # The eventual territory is the last cumulative count, which is its
    # largest. Taking it from this table rather than from ``territory``'s
    # per-cell row keeps the module reading one table, and the two are the same
    # number by construction.
    eventual = working.groupby("identity")["cumulative_unique_px"].transform("max")

    frame_table = working[keys].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        frame_table["coverage_share"] = (
            working["cumulative_unique_px"] / eventual.replace(0, np.nan)
        )
        if "area_px" in working.columns:
            frame_table["area_over_territory"] = (
                working["area_px"] / eventual.replace(0, np.nan)
            )

    per_cell = pd.DataFrame(
        {"identity": sorted(working["identity"].unique())}
    ).set_index("identity")
    per_cell["union_px_seen"] = working.groupby("identity")["cumulative_unique_px"].max()
    if "area_px" in working.columns:
        median_area = working.groupby("identity")["area_px"].median()
        per_cell["territory_over_area"] = (
            per_cell["union_px_seen"] / median_area.replace(0, np.nan)
        )
    per_cell = per_cell.drop(columns=["union_px_seen"]).reset_index()

    return {
        "territory_shape": per_cell,
        "territory_shape_frame": frame_table.sort_values(keys).reset_index(drop=True),
    }
