"""How far away the other cells are.

Every other module in the package treats a cell as if it were alone. ``contacts``
is the one exception and it only sees pairs whose outlines already touch, so
nothing could say whether a cell sits in a crowd or in open ground. Microglia in
tissue hold non-overlapping domains and survey them, and that tiling is one of
the defining properties of the population.

Two distances, because they are not substitutes. Centre to centre is how far
apart the cells *are*; outline to outline is how far apart their *edges* are,
which for a cell that reaches with processes is the distance that decides
whether they can meet. On the pinned movie the two correlate at r = 0.91 and
still differ threefold in value - 21.6 px against 7.2 px at the median - and the
outline distance is the smaller one in every single cell-frame.

Three things are worth knowing before reading a number out of here:

* **A neighbour that vanished is not a neighbour that moved.** The tracker's
  gaps change which cells are on screen frame to frame, so a distance can jump
  because somebody disappeared. ``neighbour_cells_present`` rides along on every
  row so that is visible rather than inferred.
* **``domain_px`` sums to the whole field.** Every pixel is closest to
  somebody, so a frame with fewer visible cells hands everyone a bigger domain.
  ``domain_occupancy`` - the cell's own area over its domain - is the version
  that survives that, and is the one to prefer.
* **Density needs an area, and the field is an arbitrary crop.** The default
  divides by the ground the cells ever occupy rather than by the whole imaged
  field; on the pinned movie those differ by 4.4x. ``local_density_area_px``
  carries the denominator actually used on every row, so the number can be
  converted rather than guessed at.
* **``nearest_neighbour_index`` does not use that ground, and must not.** Its
  null model is "the same cells placed at random in the study area", so the
  study area has to be defined without reference to where the cells went. The
  ever-occupied mask is the cells' own footprints, so using it would be
  circular - and not harmlessly: on the pinned movie it moves the index from
  0.91 to 1.90, which is the difference between reporting random placement and
  reporting a regular tiling. The index divides by the *measurable* field,
  always, whatever ``density_ground`` says. That is the movie's ``valid_mask``
  where one is declared and the whole frame where it is not, and it is the
  right third option precisely because the instrument defines it rather than
  the cells do.

A cell at the field edge has neighbours outside the image that cannot be seen,
so its distances are biased long. ``morphology.touches_border`` already flags
that per cell-frame and this module deliberately does not repeat it or drop the
row: a NaN would delete a stranger's edge cells silently. On the pinned movie no
cell-frame touches the border at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from analysis.fields import ground_within
from analysis.registry import Column, MeasurementContext, Output, register


DEFAULTS = {
    # The neighbourhood radius is a multiple of the cells' own reach rather
    # than a pixel constant, because a constant tuned on one magnification is
    # wrong for the next dataset. ``radius_px`` overrides it outright.
    "radius_reach_multiple": 4.0,
    "radius_px": None,
    # How many nearest cells the mean distance averages over.
    "neighbour_k": 6,
    # "ever_occupied" is every pixel any cell ever held; "field" is the whole
    # imaged frame, including ground no cell ever visits.
    "density_ground": "ever_occupied",
}


# ``ground_within`` used to live here. It moved to ``analysis.fields`` when
# ``object_geometry`` needed the same neighbourhood: two copies of one
# convolution would be two modules quietly disagreeing about how big a
# neighbourhood is, and the arithmetic is unchanged by the move.


#: ``nearest_neighbour_index`` is the population readout: below 1 the cells are
#: clustered, near 1 they are placed at random, above 1 they are regularly
#: tiled. It is the only column here that is about the field rather than about
#: a cell, which is why it is at frame grain.
PRODUCES = (
    # neighbours - one row per cell per frame
    Column("nearest_neighbour_px", "Distance to the nearest cell, centre to centre",
           "px", "reference"),
    Column("nearest_neighbour_identity", "The nearest cell, centre to centre",
           "identifier", "reference"),
    Column("nearest_edge_px", "Distance to the nearest cell, outline to outline",
           "px", "reference"),
    Column("nearest_edge_identity", "The nearest cell, outline to outline",
           "identifier", "reference"),
    Column("neighbours_within_radius", "Other cells within the neighbourhood radius",
           "count", "reference"),
    Column("neighbour_mean_distance_px", "Distance to the nearest few cells, mean",
           "px", "reference"),
    Column("domain_px", "Ground closer to this cell than to any other", "px", "surveillance"),
    Column("domain_occupancy", "Domain the cell itself fills", "fraction", "surveillance"),
    Column("local_density", "Other cells in the neighbourhood, per ground",
           "cells per 1000 px²", "reference"),
    Column("local_density_area_px", "Neighbourhood ground the density divides by",
           "px", "reference"),
    Column("neighbour_cells_present", "Cells on screen this frame", "count", "reference"),
    # neighbour_frame - one row per frame
    Column("nearest_neighbour_index", "Spacing against random placement", "ratio", "reference"),
    Column("nearest_neighbour_mean_px", "Distance to the nearest cell, mean over cells",
           "px", "reference"),
    Column("nearest_neighbour_expected_px", "Distance to the nearest cell if placed at random",
           "px", "reference"),
    Column("neighbour_cells_measured", "Cells the index was computed from", "count", "reference"),
    Column("neighbour_radius_px", "Neighbourhood radius", "px", "reference"),
    Column("neighbour_ground_px", "Measurable field the spacing index divides by",
           "px", "reference"),
)


WRITES = (
    Output("neighbours",      grain=("identity", "frame_index"), fold=True),
    Output("neighbour_frame", grain=("frame_index",),            fold=True),
)


@register(
    name="neighbours",
    description="Nearest-neighbour distance, local density, Voronoi-style domains and field tiling",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("neighbours")}
    labels = context.labels
    field = labels.shape[1:]

    ground_choice = str(params["density_ground"])
    if ground_choice == "ever_occupied":
        ground = np.any(labels > 0, axis=0)
    elif ground_choice == "field":
        ground = np.ones(field, dtype=bool)
    else:
        raise ValueError(
            f"neighbours density_ground={ground_choice!r}; expected "
            "'ever_occupied' or 'field'"
        )
    # The study area for the spacing index, kept separate from the density
    # ground on purpose - see the third bullet at the top of this file. It is
    # per frame because a mask may be, and it is exactly the old constant when
    # no mask is declared: ``valid_px`` computes the whole field as a product
    # of side lengths rather than by summing an array of True, so an undeclared
    # mask cannot move a number in its last decimal place.
    valid_px = [context.valid_px(index) for index in range(context.n_frames)]

    # The density ground is the occupiable ground that can also be measured.
    # A cell near a blank margin should not be credited with neighbourhood the
    # microscope did not deliver, and intersecting here rather than at the read
    # site means the convolution below already accounts for it.
    if context.valid is not None:
        if context.valid.ndim == 3:
            ground_by_frame = [ground & context.valid[index]
                               for index in range(context.n_frames)]
        else:
            ground = ground & context.valid
            ground_by_frame = None
    else:
        ground_by_frame = None

    # Everything except the radius-dependent columns, in one pass. The radius
    # is a multiple of the cells' own reach, which is not known until the cells
    # have been measured, so those columns wait for a second, cheap pass over
    # numbers rather than pixels.
    rows: list[dict] = []
    reaches: list[float] = []
    distances_to_others: list[np.ndarray] = []
    centroid_positions: list[np.ndarray] = []

    for frame_index in range(context.n_frames):
        frame = labels[frame_index]
        present = [int(value) for value in np.unique(frame) if value]
        if not present:
            continue

        pixels = {identity: np.argwhere(frame == identity) for identity in present}
        centroids = {identity: points.mean(axis=0) for identity, points in pixels.items()}
        trees = {identity: cKDTree(points) for identity, points in pixels.items()}

        # Which cell each pixel of the field is closest to. One distance
        # transform per frame answers it for every pixel at once: the indices
        # it returns point at the nearest foreground pixel, and that pixel
        # carries its owner's label.
        _, indices = ndi.distance_transform_edt(frame == 0, return_indices=True)
        owner = frame[indices[0], indices[1]]
        domain_counts = {
            identity: int(count) for identity, count in
            zip(*np.unique(owner[owner > 0], return_counts=True))
        }

        order = {identity: position for position, identity in enumerate(present)}
        coordinates = np.array([centroids[identity] for identity in present])
        separation = np.linalg.norm(
            coordinates[:, None, :] - coordinates[None, :, :], axis=2
        )
        np.fill_diagonal(separation, np.inf)

        edge = np.full((len(present), len(present)), np.inf)
        for position, identity in enumerate(present):
            for other_position in range(position + 1, len(present)):
                other = present[other_position]
                gap = float(trees[other].query(pixels[identity], k=1)[0].min())
                edge[position, other_position] = gap
                edge[other_position, position] = gap

        for identity in present:
            position = order[identity]
            own = pixels[identity]
            area = float(len(own))
            radii = np.linalg.norm(own - centroids[identity], axis=1)
            reaches.append(float(np.percentile(radii, 95)) if radii.size else np.nan)

            centre_gaps = separation[position]
            edge_gaps = edge[position]
            finite = np.isfinite(centre_gaps)
            domain = domain_counts.get(identity, 0)
            row = {
                "identity": identity,
                "frame_index": frame_index,
                "neighbour_cells_present": len(present),
                "domain_px": domain,
                "domain_occupancy": area / domain if domain else np.nan,
            }
            if finite.any():
                nearest = int(np.argmin(centre_gaps))
                nearest_edge = int(np.argmin(edge_gaps))
                row["nearest_neighbour_px"] = float(centre_gaps[nearest])
                row["nearest_neighbour_identity"] = present[nearest]
                row["nearest_edge_px"] = float(edge_gaps[nearest_edge])
                row["nearest_edge_identity"] = present[nearest_edge]
            else:
                row["nearest_neighbour_px"] = np.nan
                row["nearest_neighbour_identity"] = np.nan
                row["nearest_edge_px"] = np.nan
                row["nearest_edge_identity"] = np.nan
            rows.append(row)
            distances_to_others.append(np.sort(centre_gaps[finite]))
            centroid_positions.append(centroids[identity])

    if not rows:
        empty_cells = ["identity", "frame_index", *[
            column.name for column in PRODUCES
            if column.name not in {"nearest_neighbour_index", "nearest_neighbour_mean_px",
                                   "nearest_neighbour_expected_px", "neighbour_cells_measured",
                                   "neighbour_radius_px", "neighbour_ground_px"}]]
        return {
            "neighbours": pd.DataFrame(columns=empty_cells),
            "neighbour_frame": pd.DataFrame(columns=[
                "frame_index", "nearest_neighbour_index", "nearest_neighbour_mean_px",
                "nearest_neighbour_expected_px", "neighbour_cells_measured",
                "neighbour_radius_px", "neighbour_ground_px"]),
        }

    reach = float(np.nanmedian(reaches)) if reaches else np.nan
    radius = (
        float(params["radius_px"]) if params["radius_px"] is not None
        else float(params["radius_reach_multiple"]) * reach
    )
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    # One convolution for the whole movie in the ordinary case, because the
    # ground does not change with time. A per-frame mask is the only reason to
    # do it ninety-nine times, and it is done only then.
    within_by_frame = (
        [ground_within(frame_ground, radius) for frame_ground in ground_by_frame]
        if ground_by_frame is not None else None
    )
    within = ground_within(ground, radius) if within_by_frame is None else None
    k = int(params["neighbour_k"])

    for row, sorted_gaps in zip(rows, distances_to_others):
        row["neighbours_within_radius"] = int(np.count_nonzero(sorted_gaps <= radius))
        row["neighbour_mean_distance_px"] = (
            float(np.mean(sorted_gaps[:k])) if sorted_gaps.size else np.nan
        )
    neighbours = pd.DataFrame(rows)

    # The neighbourhood ground is read at the cell's own position, so a cell
    # sitting at the edge of the occupied ground is divided by less of it than
    # one in the middle - which is the correction, not a defect.
    positions = np.rint(np.array(centroid_positions)).astype(int)
    rows_y = np.clip(positions[:, 0], 0, field[0] - 1)
    rows_x = np.clip(positions[:, 1], 0, field[1] - 1)
    if within_by_frame is None:
        ground_area = within[rows_y, rows_x]
    else:
        ground_area = np.array([
            within_by_frame[int(row["frame_index"])][y, x]
            for row, y, x in zip(rows, rows_y, rows_x)
        ])
    neighbours["local_density_area_px"] = ground_area
    with np.errstate(invalid="ignore", divide="ignore"):
        neighbours["local_density"] = 1000.0 * neighbours["neighbours_within_radius"] / np.where(
            ground_area > 0, ground_area, np.nan)

    per_frame = []
    for frame_index, group in neighbours.groupby("frame_index", sort=True):
        measured = group["nearest_neighbour_px"].dropna()
        cells = int(len(measured))
        observed = float(measured.mean()) if cells else np.nan
        # A frame with nothing measurable gives NaN rather than an infinity:
        # a blank plots as absent, an infinity plots.
        field_px = valid_px[int(frame_index)]
        expected = (
            0.5 / np.sqrt(cells / field_px) if cells and field_px else np.nan
        )
        per_frame.append(
            {
                "frame_index": int(frame_index),
                "nearest_neighbour_mean_px": observed,
                "nearest_neighbour_expected_px": expected,
                "nearest_neighbour_index": (
                    observed / expected if cells and np.isfinite(expected) and expected
                    else np.nan
                ),
                "neighbour_cells_measured": cells,
                "neighbour_radius_px": radius,
                "neighbour_ground_px": field_px,
            }
        )

    return {
        "neighbours": neighbours.sort_values(["identity", "frame_index"]).reset_index(drop=True),
        "neighbour_frame": pd.DataFrame(per_frame),
    }
