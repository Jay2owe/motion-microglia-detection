"""How each cell sits against the reference shapes.

``objects`` draws the roads. This asks who lives near one: for every cell, in
every frame, against every declared set - how far away the nearest shape is,
how much of the cell is on top of one, whether it is touching, and for how long
it stays there.

This is the largest gap in the package closing. ``neighbours`` and ``contacts``
measure a cell against another cell and that was the whole of what a cell could
be related to; "how far is this microglia from the nearest vessel" and "how
many hours did it spend on a plaque" had no column to be written in.

The geometry is not new. One exact Euclidean distance transform per frame per
set answers, for every pixel of the field at once, both how far the nearest
object is and which object it is - the same technique ``neighbours`` uses to
build ``domain_px``, pointed at a second set of shapes. Nothing here loops over
objects per cell.

Two things to read together rather than apart:

* **Distance 0 and overlap are not the same claim.** A cell brushing an object
  and a cell half buried in one both read 0 outline-to-outline.
  ``object_overlap_share`` is what separates them, and any figure or sentence
  drawn from one of these columns needs the other beside it.
* **Contact duration assumes the object keeps its number.** ``nearest_object``
  is the label out of the input file. If that file renumbers its objects every
  frame, this column names a different object each frame and
  ``object_longest_contact_frames`` is meaningless while looking perfectly
  reasonable. ``object_frames_present`` in ``objects.csv`` is the check: if
  every object shows 1, the numbering is not stable.

``dilation_px`` is on every row for the reason ``contacts`` puts it on every
row of its own table: touching is a stated tolerance, not an observation, and
two tables built at different tolerances must not be mistakable for one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from analysis.fields import ground_within, longest_run
from analysis.registry import (Column, MeasurementContext, ObjectStack, Output,
                               register)

DEFAULTS = {
    # The same name and the same default ``contacts`` uses for cell-to-cell
    # adjacency. Whether a shape deserves the same tolerance as a cell is a
    # judgement rather than a constant - on the pinned movie the median cell is
    # about 11 px across, so every pixel of dilation is a tenth of a cell - and
    # it is a setting so that the judgement is visible and changeable.
    "dilation_px": 2,
    # The neighbourhood radius, resolved the way ``neighbours`` resolves its
    # own: a multiple of the cells' own reach, because a pixel constant tuned
    # at one magnification is wrong at the next. ``radius_px`` overrides it.
    "radius_reach_multiple": 4.0,
    "radius_px": None,
}


#: Everything is prefixed ``object_`` or ``nearest_object`` to stay clear of the
#: cell-to-cell columns ``neighbours`` owns. ``nearest_edge_px`` already means
#: *the nearest other cell*; reusing it and meaning something else here would
#: make the axis label depend on which module sorted first.
PRODUCES = (
    # cell_objects - one row per cell per frame per object set
    Column("nearest_object", "The nearest object, outline to outline", "identifier", "object"),
    Column("nearest_object_edge_px", "Distance to the nearest object, outline to outline",
           "px", "object"),
    Column("nearest_object_centre", "The nearest object, centre to centre", "identifier", "object"),
    Column("nearest_object_centre_px", "Distance to the nearest object, centre to centre",
           "px", "object"),
    Column("object_overlap_px", "Cell pixels inside an object", "px", "object"),
    Column("object_overlap_share", "Cell inside an object", "fraction", "object"),
    Column("object_touching", "Cell touches an object", "0 or 1", "object"),
    Column("objects_within_radius", "Objects within the neighbourhood radius", "count", "object"),
    Column("object_cover_within_radius", "Neighbourhood covered by this set", "fraction", "object"),
    Column("object_radius_px", "Neighbourhood radius", "px", "object"),
    # Declared identically to the way the modules that own them declare them, so
    # a reader of this table does not have to open another to know what
    # tolerance it was built at or how many shapes the nearest one was nearest
    # of. ``declared_columns`` allows two producers that agree and refuses two
    # that do not, which is exactly the guarantee wanted here.
    Column("dilation_px", "Dilation that counted as touching", "px", "reference"),
    Column("object_count", "Objects in this set this frame", "count", "object"),
    # cell_object_tracks - one row per cell per object set
    Column("object_frames", "Frames measured against this set", "frames", "object"),
    Column("object_contact_frames", "Frames touching an object", "frames", "object"),
    Column("object_contact_share", "Observed frames touching an object", "fraction", "object"),
    Column("object_contact_hours", "Time touching an object", "h", "object"),
    Column("object_longest_contact_frames", "Longest unbroken contact", "frames", "object"),
    Column("object_distance_median_px", "Distance to the nearest object, median", "px", "object"),
    Column("object_distance_min_px", "Closest approach to an object", "px", "object"),
    Column("object_overlap_share_median", "Cell inside an object, median", "fraction", "object"),
)


#: Neither folds. Both carry ``object_set`` on top of a roll-up's grain, so
#: folding one in would need a column per set and the schema would depend on
#: the configuration - the same reason the channel tables are files of their own.
WRITES = (
    Output("cell_objects",       grain=("identity", "frame_index", "object_set")),
    Output("cell_object_tracks", grain=("identity", "object_set")),
)


def _nearest_map(objects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel: how far the nearest object is, and which one it is.

    One exact Euclidean distance transform for the whole frame. ``edt`` of the
    empty ground measures each empty pixel's distance to the nearest object
    pixel and returns 0 inside an object, which is correct and is why distance
    and overlap have to be read together. The indices it returns point at the
    nearest object pixel, and that pixel carries its own label.
    """
    distance, indices = ndi.distance_transform_edt(objects == 0, return_indices=True)
    return distance, objects[indices[0], indices[1]]


def _set_rows(context: MeasurementContext, stack: ObjectStack,
              ) -> tuple[list[dict], list[float], list[np.ndarray]]:
    """Everything except the radius-dependent columns, in one pass.

    The radius is a multiple of the cells' own reach, which is not known until
    the cells have been measured, so those two columns wait for a second pass
    over numbers rather than over pixels. The reach is collected here for the
    same reason ``neighbours`` collects it in its own first pass.
    """
    rows: list[dict] = []
    reaches: list[float] = []
    centres: list[np.ndarray] = []

    for frame_index in range(context.n_frames):
        labels = context.labels[frame_index]
        objects = stack.frame(frame_index)
        present = [int(value) for value in np.unique(objects) if value]
        if present:
            distance, owner = _nearest_map(objects)
            object_centroids = np.array(
                ndi.center_of_mass(objects > 0, objects, present), dtype=float)
        else:
            distance = owner = object_centroids = None

        for zero_based, window in enumerate(ndi.find_objects(labels)):
            if window is None:
                continue
            identity = zero_based + 1
            local = labels[window] == identity
            ys, xs = np.nonzero(local)
            if ys.size == 0:
                continue
            top, left = window[0].start, window[1].start
            centre = np.array([ys.mean() + top, xs.mean() + left])
            radii = np.hypot(ys + top - centre[0], xs + left - centre[1])
            reaches.append(float(np.percentile(radii, 95)))
            centres.append(centre)

            row = {
                "identity": identity,
                "frame_index": frame_index,
                "object_set": stack.name,
                "object_count": len(present),
            }
            if present:
                gaps = distance[window][local]
                closest = int(np.argmin(gaps))
                overlap = int(np.count_nonzero(objects[window][local]))
                to_centres = np.linalg.norm(object_centroids - centre, axis=1)
                nearest_centre = int(np.argmin(to_centres))
                row.update({
                    "nearest_object": int(owner[window][local][closest]),
                    "nearest_object_edge_px": float(gaps[closest]),
                    "nearest_object_centre": present[nearest_centre],
                    "nearest_object_centre_px": float(to_centres[nearest_centre]),
                    "object_overlap_px": overlap,
                    "object_overlap_share": float(overlap / ys.size),
                })
            else:
                # No shape in this frame is not the same as a shape infinitely
                # far away, and it is certainly not a distance of zero. Every
                # geometric column is blank and ``object_count`` says why.
                row.update({
                    "nearest_object": np.nan,
                    "nearest_object_edge_px": np.nan,
                    "nearest_object_centre": np.nan,
                    "nearest_object_centre_px": np.nan,
                    "object_overlap_px": 0,
                    "object_overlap_share": 0.0,
                })
            rows.append(row)
    return rows, reaches, centres


def _add_radius_columns(rows: list[dict], centres: list[np.ndarray],
                        context: MeasurementContext, stacks: dict[str, ObjectStack],
                        radius: float) -> None:
    """The two columns that need a neighbourhood, filled in place.

    The denominator is how much of the *measurable* field lies within the
    radius, so a cell near the edge is not credited with neighbourhood the
    microscope never photographed. With no mask declared that is the whole
    field and one convolution answers it for the movie; a per-frame mask is the
    only reason to do it once per frame, and it is done only then. The
    numerator is one more per frame per set, or one in total for a static set.
    """
    field = context.labels.shape[1:]
    per_frame_mask = context.valid is not None and context.valid.ndim == 3
    field_within: dict[int, np.ndarray] = {}
    if not per_frame_mask:
        field_within[0] = ground_within(
            context.valid_frame(0).astype(float), radius)
    covered: dict[tuple[str, int], np.ndarray] = {}
    centroids: dict[tuple[str, int], np.ndarray] = {}

    for row, centre in zip(rows, centres):
        name = row["object_set"]
        stack = stacks[name]
        # A static set has one map, so its neighbourhood cover is computed once
        # and looked up under frame 0 for every frame.
        key = (name, 0 if stack.static else int(row["frame_index"]))
        if key not in covered:
            frame = stack.frame(int(row["frame_index"]))
            covered[key] = ground_within((frame > 0).astype(float), radius)
            present = [int(value) for value in np.unique(frame) if value]
            centroids[key] = (
                np.array(ndi.center_of_mass(frame > 0, frame, present), dtype=float)
                if present else np.empty((0, 2))
            )
        y = int(np.clip(round(float(centre[0])), 0, field[0] - 1))
        x = int(np.clip(round(float(centre[1])), 0, field[1] - 1))
        measurable = 0 if not per_frame_mask else int(row["frame_index"])
        if measurable not in field_within:
            field_within[measurable] = ground_within(
                context.valid_frame(measurable).astype(float), radius)
        denominator = float(field_within[measurable][y, x])
        row["object_radius_px"] = radius
        row["object_cover_within_radius"] = (
            float(covered[key][y, x] / denominator) if denominator > 0 else np.nan
        )
        here = centroids[key]
        row["objects_within_radius"] = (
            int(np.count_nonzero(np.linalg.norm(here - centre, axis=1) <= radius))
            if here.size else 0
        )


def _track_rows(cells: pd.DataFrame, context: MeasurementContext) -> list[dict]:
    """One row per cell per object set: how close, how often, for how long."""
    rows: list[dict] = []
    for (name, identity), group in cells.groupby(["object_set", "identity"], sort=True):
        touching = group.loc[group["object_touching"] == True]      # noqa: E712
        frames = touching["frame_index"].to_numpy(dtype=int)
        observed = int(len(group))
        distances = group["nearest_object_edge_px"].dropna()
        shares = group["object_overlap_share"].dropna()
        rows.append({
            "identity": int(identity),
            "object_set": name,
            "object_frames": observed,
            "object_contact_frames": int(len(frames)),
            "object_contact_share": float(len(frames) / observed) if observed else np.nan,
            # The interval is a setting. A hard-coded half hour would label a
            # fifteen-minute recording as twice its length.
            "object_contact_hours": context.scale.hours(len(frames)),
            # Consecutive *observed* frames: a frame the cell was not seen in
            # breaks the run rather than bridging it, because nothing was
            # observed to be touching during a frame nobody looked at.
            "object_longest_contact_frames": longest_run(frames),
            "object_distance_median_px": (
                float(distances.median()) if not distances.empty else np.nan),
            "object_distance_min_px": (
                float(distances.min()) if not distances.empty else np.nan),
            "object_overlap_share_median": (
                float(shares.median()) if not shares.empty else np.nan),
        })
    return rows


@register(
    name="object_geometry",
    description="How far each cell is from the reference shapes, how much it overlaps, for how long",
    requires=("labels", "objects"),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("object_geometry")}
    dilation = float(params["dilation_px"])

    rows: list[dict] = []
    reaches: list[float] = []
    centres: list[np.ndarray] = []
    # Configuration order, not alphabetical: the first set a user lists is the
    # one they care about most, and it should be the first series drawn.
    for name in context.object_set_names:
        set_rows, set_reaches, set_centres = _set_rows(context, context.objects[name])
        rows.extend(set_rows)
        reaches.extend(set_reaches)
        centres.extend(set_centres)

    empty_cells = ["identity", "frame_index", "object_set", *[
        column.name for column in PRODUCES]]
    if not rows:
        return {
            "cell_objects": pd.DataFrame(columns=empty_cells),
            "cell_object_tracks": pd.DataFrame(columns=[
                "identity", "object_set", "object_frames", "object_contact_frames",
                "object_contact_share", "object_contact_hours",
                "object_longest_contact_frames", "object_distance_median_px",
                "object_distance_min_px", "object_overlap_share_median"]),
        }

    reach = float(np.nanmedian(reaches)) if reaches else np.nan
    radius = (float(params["radius_px"]) if params["radius_px"] is not None
              else float(params["radius_reach_multiple"]) * reach)
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    _add_radius_columns(rows, centres, context, context.objects, radius)

    cells = pd.DataFrame(rows)
    cells["dilation_px"] = dilation
    cells["object_touching"] = cells["nearest_object_edge_px"] <= dilation
    cells = cells.sort_values(
        ["object_set", "identity", "frame_index"]).reset_index(drop=True)
    tracks = pd.DataFrame(_track_rows(cells, context))
    if not tracks.empty:
        tracks["dilation_px"] = dilation
    return {"cell_objects": cells, "cell_object_tracks": tracks}
