"""Long-form radial occupancy and classical Sholl intersections.

The profile is built twice, under two ring widths, because the two answer
different questions and neither answers the other's:

``cell``    the rings divide the cell's **own** 95% reach, so every cell gets
            ``n_rings`` rings that all land inside it. Ring 3 is "half way out"
            for every cell, whatever its size, which is what makes shapes
            comparable between a small cell and a large one. Ring 3 is *not* the
            same distance for two cells, so this scaling cannot answer "how far
            out, in pixels".
``global``  one width for the whole movie, so ring 3 is the same distance for
            every cell and profiles can be pooled on an absolute axis. A small
            cell then occupies only the first ring or two, and its outer rings
            are empty rather than missing.

Both are written to ``sholl.csv``, distinguished by the ``scaling`` column, and
every summary column appears twice with a ``_scale_cell`` / ``_scale_global``
suffix. The alternative - picking one - was tried and is what produced the
defect below.

**Why the global width is a percentile and not the maximum.** It used to be the
largest reach anywhere in the movie divided by ``n_rings``. On the reference
dataset one two-piece object - a tracking artefact whose halves sat 120 px apart
- pushed that maximum to 124 px and so set a 20.7 px ring against a typical cell
reach of 8.4 px mean, 7.5 px median. 96% of cell-frames then occupied exactly
one ring and 58.5% had no skeleton crossing at any radius: a profile with no
shape, from which no summary can be computed. ``global_reach_percentile``
defaults to 99 so that one outlier cannot set the scale for everyone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from analysis.registry import Column, MeasurementContext, Output, register


DEFAULTS = {
    "n_rings": 6,
    "ring_width": None,
    "global_reach_percentile": 99.0,
    "centre": "soma",
    "exclude_other_identities": True,
}

#: The two ring widths every cell-frame is measured under, and the suffix each
#: one's summary columns carry. Written out as data rather than branched on in
#: three places, so adding a third scaling stays a one-line change.
SCALINGS = ("cell", "global")
SUFFIX = {"cell": "_scale_cell", "global": "_scale_global"}


def _centre(mask: np.ndarray, raw: np.ndarray | None, mode: str) -> tuple[float, float]:
    centroid = ndi.center_of_mass(mask)
    if mode in {"centroid", "centre", "center"} or raw is None:
        return float(centroid[0]), float(centroid[1])
    weights = np.asarray(raw, dtype=float).copy()
    values = weights[mask]
    if not values.size:
        return float(centroid[0]), float(centroid[1])
    weights = np.where(mask, np.maximum(weights - float(np.nanmin(values)), 0.0), 0.0)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        return float(centroid[0]), float(centroid[1])
    centre = ndi.center_of_mass(weights)
    return float(centre[0]), float(centre[1])


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value at which half the weight lies to either side.

    Reported beside the weighted mean because occupancy against radius is
    right-skewed: a few pixels far out drag a mean that no ring actually sits at.
    """
    if not values.size or float(weights.sum()) <= 0:
        return float("nan")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    crossing = np.searchsorted(np.cumsum(weights), 0.5 * weights.sum())
    return float(values[min(int(crossing), values.size - 1)])


def _summarise_profile(rings: list[dict], width_px: float) -> dict[str, float]:
    """Reduce one cell-frame's ring rows to the classical Sholl readouts.

    ``rings`` is in ring order and is always ``n_rings`` long: an unoccupied ring
    is a zero, not a missing row. Every integral below therefore runs over the
    whole profile as defined, not over the part the cell happens to fill, and the
    two are different numbers whenever a cell is smaller than the profile.
    """
    intersections = np.array([row["intersections"] for row in rings], dtype=float)
    occupied = np.array([row["occupied_px"] for row in rings], dtype=float)
    annulus = np.array([row["annulus_px"] for row in rings], dtype=float)
    occupancy = np.divide(occupied, annulus, out=np.full_like(occupied, np.nan),
                          where=annulus > 0)
    mid = (np.arange(len(rings)) + 0.5) * width_px
    outer = (np.arange(len(rings)) + 1.0) * width_px

    peak = float(intersections.max()) if intersections.size else np.nan
    at_peak = int(np.argmax(intersections)) if intersections.size else 0
    inner_crossings = float(intersections[0]) if intersections.size else np.nan
    filled = np.nonzero(occupied > 0)[0]

    # The decay slope, log(crossings / ring area) against radius. Fitted only
    # where there is something to fit; ``points`` is reported so a reader can see
    # a two-point "regression" for what it is.
    usable = (intersections > 0) & (annulus > 0)
    slope = r_squared = np.nan
    points = int(usable.sum())
    if points >= 2:
        y = np.log10(intersections[usable] / annulus[usable])
        x = mid[usable]
        slope = float(np.polyfit(x, y, 1)[0])
        predicted = np.polyval(np.polyfit(x, y, 1), x)
        total = float(((y - y.mean()) ** 2).sum())
        r_squared = float(1.0 - ((y - predicted) ** 2).sum() / total) if total > 0 else np.nan

    inner_occupancy = float(occupancy[0]) if occupancy.size else np.nan
    outer_occupancy = float(occupancy[filled[-1]]) if filled.size else np.nan
    return {
        "max_intersections": peak,
        "critical_radius": float(mid[at_peak]) if intersections.size else np.nan,
        "auc": float(intersections.sum() * width_px),
        "ramification_index": (float(peak / inner_crossings)
                               if inner_crossings and np.isfinite(inner_crossings)
                               else np.nan),
        "regression_coefficient": slope,
        "regression_r2": r_squared,
        "regression_points": float(points),
        "enclosing_radius": float(outer[filled[-1]]) if filled.size else np.nan,
        "mean_radius": (float((mid * occupied).sum() / occupied.sum())
                        if occupied.sum() > 0 else np.nan),
        "median_radius": _weighted_median(mid, occupied),
        "occupancy_inner": inner_occupancy,
        "occupancy_outer": outer_occupancy,
        "occupancy_ratio": (float(inner_occupancy / outer_occupancy)
                            if outer_occupancy else np.nan),
        "rings_occupied": float((occupied > 0).sum()),
    }


#: Every summary the profile is reduced to, as label and unit. Written once and
#: expanded per scaling below, so the two suffixed copies cannot drift apart.
_SUMMARIES = (
    ("max_intersections", "Peak radial crossings", "count"),
    ("critical_radius", "Radius of peak crossings", "px"),
    ("auc", "Area under the crossing profile", "count.px"),
    ("ramification_index", "Peak crossings / innermost-ring crossings", "ratio"),
    ("regression_coefficient", "Sholl decay slope, log(crossings/ring area) per unit radius", "1/px"),
    ("regression_r2", "Fit quality of the decay slope", "fraction"),
    ("regression_points", "Rings that entered the decay fit", "count"),
    ("enclosing_radius", "Outer radius of the outermost occupied ring", "px"),
    ("mean_radius", "Occupancy-weighted mean radius", "px"),
    ("median_radius", "Occupancy-weighted median radius", "px"),
    ("occupancy_inner", "Occupancy of the innermost ring", "fraction"),
    ("occupancy_outer", "Occupancy of the outermost occupied ring", "fraction"),
    ("occupancy_ratio", "Innermost / outermost ring occupancy", "ratio"),
    ("rings_occupied", "Rings holding any pixel of this cell", "count"),
)

#: What the suffix means, spelled out in every label rather than left to a
#: reader to infer from the column name. Half of these columns are the same
#: quantity measured on a different x-axis; a label that does not say which is
#: how somebody averages the two.
_SCALE_NOTE = {
    "cell": "rings dividing the cell's own reach",
    "global": "rings of one fixed width across the movie",
}


#: Long form: one row per cell per frame per scaling per ring, so ``scaling``,
#: ``ring`` and its two radii are part of the measurement rather than an index.
#: Everything takes the ``morphology`` role because a Sholl profile is a shape
#: description measured a different way, and splitting it off into its own
#: colour would make one cell's shape two colours on the same page.
PRODUCES = (
    # sholl - one row per cell per frame per scaling per ring
    Column("scaling", "Ring-width scaling", "", "morphology"),
    Column("ring", "Radial ring", "count", "morphology"),
    Column("radius_inner", "Ring inner radius", "px", "morphology"),
    Column("radius_outer", "Ring outer radius", "px", "morphology"),
    Column("ring_width", "Width of one ring", "px", "morphology"),
    Column("annulus_px", "Pixels in the ring", "px", "morphology"),
    Column("occupied_px", "Occupied pixels in the ring", "px", "morphology"),
    Column("occupancy", "Fraction of each radial ring occupied", "fraction", "morphology"),
    Column("intersections", "Boundary crossings per radial ring", "count", "morphology"),
    # sholl_reach - one row per cell per frame; these three do not depend on the
    # rings at all, being computed from raw pixel distances, so they carry no
    # scaling suffix and are unchanged by anything above.
    Column("reach_p95", "Radius containing 95% of occupied pixels", "px", "morphology"),
    Column("centre_y", "Ring centre, y", "px", "morphology"),
    Column("centre_x", "Ring centre, x", "px", "morphology"),
    Column("ring_width_scale_cell", "Ring width, cell-scaled", "px", "morphology"),
    Column("ring_width_scale_global", "Ring width, movie-scaled", "px", "morphology"),
) + tuple(
    Column(f"sholl_{name}{SUFFIX[scaling]}",
           f"{label} ({_SCALE_NOTE[scaling]})", unit, "morphology")
    for scaling in SCALINGS
    for name, label, unit in _SUMMARIES
)


WRITES = (
    Output("sholl",       grain=("identity", "frame_index", "scaling", "ring")),
    Output("sholl_reach", grain=("identity", "frame_index"), fold=True),
)


@register(
    name="sholl",
    description="Annular occupancy and skeleton crossings around each cell soma, "
                "at two ring widths",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("sholl")}
    n_rings = int(params["n_rings"])
    if n_rings < 1:
        raise ValueError("sholl n_rings must be at least 1")

    entries: list[tuple[int, int, np.ndarray, tuple[float, float], np.ndarray]] = []
    reaches: list[float] = []
    for frame_index in range(context.n_frames):
        frame = context.labels[frame_index]
        raw = context.raw[frame_index] if context.raw is not None else None
        for identity in (int(value) for value in np.unique(frame) if value):
            mask = frame == identity
            centre = _centre(mask, raw, str(params["centre"]))
            yy, xx = np.nonzero(mask)
            distances = np.hypot(yy - centre[0], xx - centre[1])
            if distances.size:
                reaches.append(float(np.percentile(distances, 95)))
            entries.append((identity, frame_index, mask, centre, distances))

    configured_width = params.get("ring_width")
    if configured_width not in (None, "", 0, 0.0):
        # Configured spatial values are in the run's length unit. Convert back
        # only through Scale, never through a bare conversion in a figure. A
        # configured width sets the global scaling; the cell scaling is by
        # definition the cell's own reach and has nothing to configure.
        global_width = (
            float(configured_width) / context.scale.microns_per_pixel
            if context.scale.calibrated else float(configured_width)
        )
    else:
        percentile = float(params["global_reach_percentile"])
        reference = float(np.percentile(reaches, percentile)) if reaches else 0.0
        global_width = reference / n_rings if reference > 0 else 1.0
    global_width = max(float(global_width), 1e-6)

    yy_grid, xx_grid = np.indices(context.labels.shape[1:])
    rows: list[dict] = []
    reach_rows: list[dict] = []
    for identity, frame_index, mask, centre, cell_distances in entries:
        # Everything below the ring loop is computed once and reused by both
        # scalings: on the reference dataset this setup is 23% of the module's
        # time and the ring loop the other 77%, so a second scaling costs 77%
        # more rather than twice as much.
        distance = np.hypot(yy_grid - centre[0], xx_grid - centre[1])
        skeleton = skeletonize(mask)
        field_available = np.ones(mask.shape, dtype=bool)
        if bool(params["exclude_other_identities"]):
            frame = context.labels[frame_index]
            field_available = (frame == 0) | (frame == identity)
        hours = context.scale.hours(frame_index + context.source_frame_offset)

        reach_px = float(np.percentile(cell_distances, 95)) if cell_distances.size else np.nan
        cell_width = (max(reach_px / n_rings, 1e-6)
                      if np.isfinite(reach_px) and reach_px > 0 else global_width)
        widths = {"cell": cell_width, "global": global_width}

        summary: dict[str, float] = {}
        for scaling in SCALINGS:
            width_px = widths[scaling]
            profile: list[dict] = []
            for ring in range(n_rings):
                inner_px = ring * width_px
                outer_px = (ring + 1) * width_px
                annulus = (distance >= inner_px) & (distance < outer_px) & field_available
                occupied = annulus & mask
                mid = (inner_px + outer_px) / 2.0
                shell = skeleton & (np.abs(distance - mid) <= 0.75)
                intersections = int(ndi.label(shell, structure=np.ones((3, 3), dtype=int))[1])
                annulus_px = int(annulus.sum())
                profile.append(
                    {
                        "identity": identity,
                        "frame_index": frame_index,
                        "hours": hours,
                        "scaling": scaling,
                        "ring": ring,
                        "radius_inner": context.scale.length(inner_px),
                        "radius_outer": context.scale.length(outer_px),
                        "ring_width": context.scale.length(width_px),
                        "annulus_px": annulus_px,
                        "occupied_px": int(occupied.sum()),
                        "occupancy": float(occupied.sum() / annulus_px) if annulus_px else np.nan,
                        "intersections": intersections,
                    }
                )
            rows.extend(profile)
            reduced = _summarise_profile(profile, width_px)
            for name, _, unit in _SUMMARIES:
                value = reduced[name]
                # Radii are summarised in pixels and converted once, here, so a
                # calibrated run reports them in the same unit as every other
                # length rather than in whatever the ring arithmetic used.
                if unit == "px" and np.isfinite(value):
                    value = context.scale.length(value)
                summary[f"sholl_{name}{SUFFIX[scaling]}"] = value

        reach_rows.append(
            {
                "identity": identity,
                "frame_index": frame_index,
                "hours": hours,
                "reach_p95": context.scale.length(reach_px) if np.isfinite(reach_px) else np.nan,
                "centre_y": centre[0],
                "centre_x": centre[1],
                "ring_width_scale_cell": context.scale.length(cell_width),
                "ring_width_scale_global": context.scale.length(global_width),
                **summary,
            }
        )

    return {"sholl": pd.DataFrame(rows), "sholl_reach": pd.DataFrame(reach_rows)}
