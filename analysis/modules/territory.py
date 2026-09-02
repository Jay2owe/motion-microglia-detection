"""Per-pixel occupancy history across an entire recording.

Two questions, kept apart. How *much* ground a cell eventually held, split by
how often it held it - the core, fringe and transient bands - and what that
ground *looks like*: a tight disc around the body, or a sprawl along one axis
with the cell sitting at one end of it. The second set is ordinary
``regionprops`` on the union mask, the same call ``morphology`` makes on the
per-frame outline, so ``territory_circularity`` and ``circularity`` can be read
side by side and a cell whose patch is rounder than its own outline is one that
swept in every direction.

``count_unclaimed_as_visited`` applies to the field-wide ``revisit_count``
stack only. The per-cell union is built from ``labels == identity``, so a
foreground pixel that carries no name is ground the field was visited on and
nobody's territory. The two genuinely mean different things and are left
disagreeing on purpose; on the pinned movie 880 cell-frames (10.7%) are
``unclaimed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

from analysis.registry import Column, MeasurementContext, Output, register


DEFAULTS = {
    "core_fraction": 0.8,
    "transient_fraction": 0.2,
    "count_unclaimed_as_visited": True,
    # Below this a union is a handful of pixels whose discrete perimeter is not
    # an estimate of anything, so the ratios built on it are dropped and the
    # sizes kept. The same reasoning, and the same default, as
    # ``morphology.min_area_px_for_shape_ratios``.
    "min_union_px_for_shape": 8,
    # No outline can be rounder than a circle. Above this the discrete
    # perimeter estimator has failed, which it does on a territory small enough
    # to be a few pixels square. Same cap, same reasoning, as
    # ``morphology.max_plausible_circularity``.
    "max_plausible_circularity": 1.15,
}


def _union_shape(union: np.ndarray, max_circularity: float) -> dict:
    """Form of the eventual footprint, measured the way an outline is.

    ``regionprops`` on a single-label mask, so disconnected pieces are one
    region: a cell that was pinched into two blobs still has one territory.
    ``territory_radius_p95_px`` is the 95th percentile of the distance from the
    patch's own centre to its pixels - how far the footprint reaches, without
    one stray pixel setting the number.
    """
    coordinates = np.argwhere(union)
    if coordinates.size == 0:
        return {}
    centre = coordinates.mean(axis=0)
    radii = np.hypot(coordinates[:, 0] - centre[0], coordinates[:, 1] - centre[1])
    region = regionprops(union.astype(np.uint8))[0]
    perimeter = float(region.perimeter)
    area = float(region.area)
    minor = float(region.axis_minor_length)
    circularity = 4.0 * np.pi * area / perimeter ** 2 if perimeter > 0 else np.nan
    return {
        "union_perimeter_px": perimeter,
        "territory_circularity": (
            circularity if np.isfinite(circularity) and circularity <= max_circularity
            else np.nan
        ),
        "territory_solidity": float(region.solidity),
        "territory_major_axis_px": float(region.axis_major_length),
        "territory_minor_axis_px": minor,
        "territory_aspect_ratio": (
            float(region.axis_major_length) / minor if minor > 0 else np.nan
        ),
        "territory_orientation_rad": float(region.orientation),
        "territory_centroid_y": float(centre[0]),
        "territory_centroid_x": float(centre[1]),
        "territory_radius_p95_px": float(np.percentile(radii, 95)),
        "territory_components": int(ndi.label(union, structure=np.ones((3, 3)))[1]),
    }


#: Territory splits into three bands by how often a pixel was occupied - core,
#: fringe, transient - and the roles follow that reading rather than the module
#: name: the core is the ``stable`` colour because it is the part of the cell
#: that stays, and the fringe the ``motility`` one because it is the part that
#: moves. The per-pixel stacks this module also returns are images, not tables,
#: and carry no columns to declare.
PRODUCES = (
    # territory - one row per cell
    Column("observed_frames", "Frames the cell was seen in", "frames", "reference"),
    Column("union_px", "Eventual territory", "px", "surveillance"),
    Column("core_px", "Frequently occupied territory", "px", "stable"),
    Column("fringe_px", "Intermittently occupied territory", "px", "motility"),
    Column("transient_px", "Briefly occupied territory", "px", "surveillance"),
    Column("core_share", "Frequently occupied share of eventual territory", "fraction", "stable"),
    Column("half_coverage_hours", "Time to reach half of eventual territory", "h", "surveillance"),
    Column("saturation_hours", "Time to reach near-complete territory", "h", "surveillance"),
    # territory, the shape of that ground rather than the amount of it.
    # `core_share` and `territory_solidity` are the pair most likely to be
    # taken for each other: the first is how *often* the pixels were occupied,
    # the second how ragged the outline around them is. A cell can hold a
    # ragged patch faithfully, or sweep a smooth one once.
    Column("union_perimeter_px", "Outline of the eventual territory", "px", "surveillance"),
    Column("territory_circularity", "Roundness of the eventual territory", "0-1", "surveillance"),
    Column("territory_solidity", "Territory filled of the hull around it", "0-1", "surveillance"),
    Column("territory_major_axis_px", "Territory major axis", "px", "surveillance"),
    Column("territory_minor_axis_px", "Territory minor axis", "px", "surveillance"),
    Column("territory_aspect_ratio", "Territory elongation", "ratio", "surveillance"),
    Column("territory_orientation_rad", "Territory orientation", "rad", "surveillance"),
    Column("territory_centroid_y", "Territory centre, y", "px", "surveillance"),
    Column("territory_centroid_x", "Territory centre, x", "px", "surveillance"),
    Column("territory_radius_p95_px", "Territory reach from its own centre", "px", "surveillance"),
    Column("territory_components", "Disconnected pieces of the territory", "count", "surveillance"),
    Column("territory_centroid_offset_px",
           "Cell's usual position from the centre of its territory", "px", "surveillance"),
    Column("territory_overlap_px", "Territory shared with any other cell", "px", "surveillance"),
    Column("territory_overlap_share", "Shared share of eventual territory", "fraction", "surveillance"),
    Column("territory_sharing_cells", "Cells sharing any ground", "count", "surveillance"),
    # territory_frame - one row per cell per frame
    Column("cumulative_unique_px", "Territory reached so far", "px", "surveillance"),
    Column("new_px", "Territory reached for the first time", "px", "surveillance"),
    Column("revisit_fraction", "Occupied pixels visited before", "fraction", "surveillance"),
    # territory_overlap - one row per pair of cells that share any ground. The
    # pair keys are spelled the way `contacts` spells them, because a reader
    # joining the two tables should not have to translate between them.
    Column("identity_a", "First cell of the pair", "", "reference"),
    Column("identity_b", "Second cell of the pair", "", "reference"),
    Column("shared_territory_px", "Ground both cells held at some point", "px", "surveillance"),
    Column("shared_territory_share_a",
           "Shared share of the first cell's territory", "fraction", "surveillance"),
    Column("shared_territory_share_b",
           "Shared share of the second cell's territory", "fraction", "surveillance"),
)


#: ``territory_overlap`` is a file rather than a fold because it is keyed on a
#: pair and no roll-up is. It is sparse on purpose: only pairs that share at
#: least one pixel get a row, which on the pinned movie is 177 of the 3,403
#: possible pairs. A dense table would be 95% zeros, and the zeros are the
#: uninteresting half - microglia hold non-overlapping domains, so the question
#: is which cells break that, not which cells keep it.
WRITES = (
    Output("territory",         grain=("identity",),               fold=True),
    Output("territory_frame",   grain=("identity", "frame_index"), fold=True),
    Output("territory_overlap", grain=("identity_a", "identity_b")),
)


@register(
    name="territory",
    description="Per-pixel revisit, first ownership and per-cell cumulative coverage",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame | np.ndarray]:
    params = {**DEFAULTS, **context.module_params("territory")}
    core_fraction = float(params["core_fraction"])
    transient_fraction = float(params["transient_fraction"])
    if not 0 <= transient_fraction <= core_fraction <= 1:
        raise ValueError("territory fractions must satisfy 0 <= transient <= core <= 1")

    labels = context.labels
    shape = labels.shape[1:]
    any_foreground = labels > 0
    if bool(params["count_unclaimed_as_visited"]) and context.unclaimed is not None:
        any_foreground = any_foreground | (context.unclaimed > 0)
    revisit_count = any_foreground.sum(axis=0).astype(np.uint32)
    never = (~any_foreground.any(axis=0)).astype(np.uint8)

    first_owner = np.zeros(shape, dtype=np.uint32)
    owners: dict[int, np.ndarray] = {
        identity: np.zeros(shape, dtype=bool) for identity in context.identities
    }
    for frame_index in range(context.n_frames):
        frame = labels[frame_index]
        unclaimed = (first_owner == 0) & (frame > 0)
        first_owner[unclaimed] = frame[unclaimed].astype(np.uint32)
        for identity in (int(value) for value in np.unique(frame) if value):
            owners.setdefault(identity, np.zeros(shape, dtype=bool))
            owners[identity] |= frame == identity
    owner_count = np.zeros(shape, dtype=np.uint16)
    for occupied in owners.values():
        owner_count += occupied.astype(np.uint16)

    territory_rows: list[dict] = []
    frame_rows: list[dict] = []
    unions: dict[int, np.ndarray] = {}
    for identity in sorted(owners):
        occupied_frames = [
            frame_index for frame_index in range(context.n_frames)
            if np.any(labels[frame_index] == identity)
        ]
        if not occupied_frames:
            continue
        counts = np.sum(labels == identity, axis=0)
        observed = len(occupied_frames)
        union = counts > 0
        core = counts >= max(1, int(np.ceil(core_fraction * observed)))
        transient = union & (counts < max(1, transient_fraction * observed))
        fringe = union & ~core & ~transient

        cumulative = np.zeros(shape, dtype=bool)
        cumulative_counts: list[int] = []
        centres: list[tuple[float, float]] = []
        for frame_index in occupied_frames:
            current = labels[frame_index] == identity
            before = cumulative.copy()
            cumulative |= current
            cumulative_count = int(cumulative.sum())
            cumulative_counts.append(cumulative_count)
            # Where the cell was, frame by frame, kept only to answer whether
            # it sits in the middle of its patch or at one edge. The median of
            # these is deliberately not ``motility``'s soma centre: this module
            # reads pixels and nothing else, and a plain centroid is the centre
            # of the same outline the territory is built from.
            centre = ndi.center_of_mass(current)
            centres.append((float(centre[0]), float(centre[1])))
            frame_rows.append(
                {
                    "identity": identity,
                    "frame_index": frame_index,
                    "hours": context.scale.hours(frame_index + context.source_frame_offset),
                    "cumulative_unique_px": cumulative_count,
                    "new_px": int(np.count_nonzero(current & ~before)),
                    "revisit_fraction": (
                        float(np.count_nonzero(current & before) / current.sum())
                        if current.any() else np.nan
                    ),
                }
            )

        union_px = int(union.sum())
        half_target = 0.5 * union_px
        saturation_target = 0.95 * union_px

        def first_hour(target: float) -> float:
            for frame_index, count in zip(occupied_frames, cumulative_counts):
                if count >= target:
                    return context.scale.hours(frame_index + context.source_frame_offset)
            return float("nan")

        row = {
            "identity": identity,
            "observed_frames": observed,
            "union_px": union_px,
            "core_px": int(core.sum()),
            "fringe_px": int(fringe.sum()),
            "transient_px": int(transient.sum()),
            "core_share": float(core.sum() / union_px) if union_px else np.nan,
            "half_coverage_hours": first_hour(half_target),
            "saturation_hours": first_hour(saturation_target),
        }
        if union_px >= int(params["min_union_px_for_shape"]):
            shape_columns = _union_shape(
                union, float(params["max_plausible_circularity"]))
            row.update(shape_columns)
            typical = np.median(np.asarray(centres, dtype=float), axis=0)
            row["territory_centroid_offset_px"] = float(
                np.hypot(shape_columns["territory_centroid_y"] - typical[0],
                         shape_columns["territory_centroid_x"] - typical[1])
            )
        territory_rows.append(row)
        unions[identity] = union

    # Which cells share ground with which. Every pair is tested and only the
    # ones that touch are written, so the file says who overlaps rather than
    # asking a reader to filter 3,403 rows down to the 177 that do.
    overlap_rows: list[dict] = []
    shared_px: dict[int, int] = {}
    sharing_cells: dict[int, int] = {}
    ordered = sorted(unions)
    for position, identity in enumerate(ordered):
        for other in ordered[position + 1:]:
            both = int(np.count_nonzero(unions[identity] & unions[other]))
            if not both:
                continue
            overlap_rows.append(
                {
                    "identity_a": identity,
                    "identity_b": other,
                    "shared_territory_px": both,
                    "shared_territory_share_a": both / float(unions[identity].sum()),
                    "shared_territory_share_b": both / float(unions[other].sum()),
                }
            )
            sharing_cells[identity] = sharing_cells.get(identity, 0) + 1
            sharing_cells[other] = sharing_cells.get(other, 0) + 1

    # The per-cell total is the ground shared with *anybody*, not the sum of
    # the pairwise overlaps: a pixel three cells all held would otherwise be
    # counted twice and could push the share above 1. ``owner_count`` above is
    # already how many cells ever held each pixel, so more than one owner is
    # the whole test.
    contested = owner_count > 1
    for identity in ordered:
        shared_px[identity] = int(np.count_nonzero(unions[identity] & contested))

    for row in territory_rows:
        identity = row["identity"]
        if identity not in unions:
            continue
        row["territory_overlap_px"] = shared_px.get(identity, 0)
        row["territory_overlap_share"] = (
            shared_px.get(identity, 0) / row["union_px"] if row["union_px"] else np.nan
        )
        row["territory_sharing_cells"] = sharing_cells.get(identity, 0)

    return {
        "territory": pd.DataFrame(territory_rows),
        "territory_frame": pd.DataFrame(frame_rows),
        "territory_overlap": pd.DataFrame(
            overlap_rows,
            columns=["identity_a", "identity_b", "shared_territory_px",
                     "shared_territory_share_a", "shared_territory_share_b"],
        ),
        "revisit_count": revisit_count,
        "first_owner": first_owner,
        "owner_count": owner_count,
        "never_visited": never,
    }

