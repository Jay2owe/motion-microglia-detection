"""Reference shapes in the image, described as shapes.

Everything else in the package relates a cell to another cell, or to nothing.
Blood vessels, plaques, a wound edge, a scaffold, a second cell type - a
microglia's whole environment - had nowhere to be. This module is the first
half of putting it there: **get the shapes in, and say what they are.**

The everyday version is a map with two layers. One layer is where the people
are; the other is where the roads are. Drawing the roads is one job; asking who
lives near a road is a different one. This module draws the roads.
``object_geometry`` asks the other question, and it is deliberately not here:
the two need different things, and a module that measures shapes without ever
looking at a cell is a module whose numbers cannot be wrong about a cell.

Nothing here knows what an object is. There is no "vessels are tubes, so
measure tortuosity". The measurements are the same generic ones ``morphology``
takes of a cell, because what these shapes are is the user's to write in the
configuration's ``description``, and it travels into the manifest untouched.

**One row per object per frame per set.** Long rather than wide, for the same
reason the channel tables are: which set a row is about is a value in the
``object_set`` column, so declaring a second set adds rows and never adds
columns.

Two things a reader should know before using ``object``:

* **The label is whatever the input file says it is.** This package does not
  track objects across frames and does not repair a mask that renumbers itself
  every frame. ``object_frames_present`` is the column that makes that visible:
  if every object shows 1, the numbering is not stable and nothing that depends
  on an object keeping its number means anything.
* **A pixel the alignment could not reach reads as empty ground.** A label
  image has no spare value for "could not look", so the count lives in
  ``manifest.json`` under ``objects:<name>`` as ``unreachable_px`` instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from analysis.registry import (Column, MeasurementContext, ObjectStack, Output,
                               register)


#: All one colour family, like ``channel``: how many object sets there are and
#: what they are called is configuration, so a per-set role could not be
#: declared here. A figure drawing two sets separates them by series.
PRODUCES = (
    Column("object_set", "Object set", "", "object"),
    Column("object", "Object", "identifier", "object"),
    Column("object_area_px", "Object area", "px", "object"),
    Column("object_perimeter_px", "Object perimeter", "px", "object"),
    Column("object_centroid_y", "Object centre, row", "px", "object"),
    Column("object_centroid_x", "Object centre, column", "px", "object"),
    Column("object_solidity", "Object solidity", "0-1", "object"),
    Column("object_equivalent_diameter_px", "Object equivalent diameter", "px", "object"),
    Column("object_touches_border", "Object touches the field edge", "0 or 1", "object"),
    Column("object_count", "Objects in this set this frame", "count", "object"),
    Column("object_set_px", "Field this set covers", "px", "object"),
    Column("object_field_share", "Measurable field this set covers", "fraction", "object"),
    Column("object_frames_present", "Frames this object number appears in", "frames", "object"),
)


#: Does not fold. A roll-up is keyed on a cell, a frame or both, and this is
#: keyed on an object - which is not a cell and has no roll-up of its own.
WRITES = (
    Output("objects", grain=("object_set", "object", "frame_index")),
)


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    """Whether a region reaches the edge of the field it was measured in.

    An object at the edge continues outside the image, so its area and its
    shape are of the part that was photographed rather than of the object. The
    same test ``morphology`` applies to a cell, applied to a shape.
    """
    min_row, min_col, max_row, max_col = bbox
    return bool(min_row == 0 or min_col == 0
                or max_row == shape[0] or max_col == shape[1])


def _set_rows(context: MeasurementContext, stack: ObjectStack) -> list[dict]:
    """Every object in every frame of one set."""
    field = context.labels.shape[1:]
    rows: list[dict] = []

    for frame_index in range(context.n_frames):
        frame = stack.frame(frame_index)
        present = int(np.count_nonzero(frame))
        # The measurable field, which is the whole frame unless the movie
        # declares a mask - see ``MeasurementContext.valid_px``.
        field_px = float(context.valid_px(frame_index))
        regions = regionprops(frame)
        for region in regions:
            rows.append({
                "object_set": stack.name,
                "object": int(region.label),
                "frame_index": frame_index,
                "object_area_px": float(region.area),
                "object_perimeter_px": float(region.perimeter),
                "object_centroid_y": float(region.centroid[0]),
                "object_centroid_x": float(region.centroid[1]),
                "object_solidity": float(region.solidity),
                "object_equivalent_diameter_px": float(region.equivalent_diameter_area),
                "object_touches_border": _touches_border(region.bbox, field),
                # Both about the frame rather than about this object, and on
                # every row on purpose: a distance to "the nearest of four
                # objects" and to "the nearest of forty" are different claims,
                # and a reader should not have to go and count.
                "object_count": len(regions),
                "object_set_px": present,
                "object_field_share": present / field_px if field_px else np.nan,
            })
    return rows


@register(
    name="objects",
    description="Reference shapes - vessels, plaques, a wound edge - described as shapes",
    requires=("labels", "objects"),
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    rows: list[dict] = []
    # Configuration order, not alphabetical: the first set a user lists is the
    # one they care about most, and it should be the first series drawn.
    for name in context.object_set_names:
        rows.extend(_set_rows(context, context.objects[name]))

    table = pd.DataFrame(rows)
    if table.empty:
        return {"objects": pd.DataFrame(
            columns=["object_set", "object", "frame_index",
                     *[column.name for column in PRODUCES
                       if column.name not in ("object_set", "object")]])}

    # The honesty column. An input that renumbers its objects every frame
    # gives every object a 1 here, and nothing downstream that assumes an
    # object keeps its number can be believed.
    table["object_frames_present"] = table.groupby(
        ["object_set", "object"])["frame_index"].transform("count")
    return {"objects": table.sort_values(
        ["object_set", "object", "frame_index"]).reset_index(drop=True)}
