"""Shape of each cell in each frame.

Microglial activation state is read off shape: a resting cell is a small soma
with long thin branches, so it has a large perimeter for its area, low
solidity and many skeleton endpoints. An activated cell rounds up - perimeter
falls, solidity rises, branches disappear. Every column here is a different way
of asking that same question, and they are kept side by side because different
papers use different ones.

One everyday comparison: solidity is how much of a rubber band stretched around
the cell is actually filled by the cell. A starfish has low solidity; a pebble
has high solidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from analysis.registry import Column, MeasurementContext, Output, register

DEFAULTS = {
    "compute_skeleton": True,
    "min_area_px_for_skeleton": 12,
    # Below this area a region is a label fragment, not a cell: its perimeter is
    # a handful of pixels, so shape ratios built on perimeter blow up. The row is
    # kept and flagged rather than dropped, so nothing disappears silently.
    "min_area_px_for_shape_ratios": 8,
    # No real outline can be rounder than a circle. Anything above this is the
    # discrete perimeter estimator failing, which it does when an identity is
    # scattered into disconnected specks in one frame. Those rows keep their
    # area and their component count and lose only the ratios built on
    # perimeter, so the failure is visible instead of being plotted.
    "max_plausible_circularity": 1.15,
}

_NEIGHBOURS = np.ones((3, 3), dtype=np.uint8)


def skeleton_stats(mask: np.ndarray, skeleton: np.ndarray | None = None) -> dict:
    """Length and branching of the cell's medial skeleton.

    ``skeleton_px`` is a process-length proxy: the number of pixels left when
    the cell is thinned to a one-pixel-wide line drawing. ``endpoints`` are
    process tips, ``junctions`` are branch points.

    ``skeleton_branches`` counts the pieces left when every junction *pixel* is
    removed, which under-counts wherever a junction is a cluster rather than a
    point - on the pinned movie, 88.8% of them. It is kept exactly as it was
    because ``regimes`` clusters on it and changing it would move every regime
    in every run. ``branch_count``, beside it, is the cluster-contracted count
    from ``branch_geometry``, and the two are not the same number.
    """
    if skeleton is None:
        skeleton = skeletonize(mask)
    total = int(skeleton.sum())
    if total == 0:
        return {"skeleton_px": 0, "skeleton_endpoints": 0, "skeleton_junctions": 0,
                "skeleton_branches": 0}

    degree = ndi.convolve(skeleton.astype(np.uint8), _NEIGHBOURS, mode="constant") - skeleton
    endpoints = int(((degree == 1) & skeleton).sum())
    junctions = int(((degree >= 3) & skeleton).sum())

    segments = skeleton & ~((degree >= 3) & skeleton)
    branches = int(ndi.label(segments, structure=np.ones((3, 3)))[1])

    return {
        "skeleton_px": total,
        "skeleton_endpoints": endpoints,
        "skeleton_junctions": junctions,
        "skeleton_branches": branches,
    }


def _skeleton_graph(skeleton: np.ndarray) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Every skeleton pixel, who it touches, and how many that is."""
    coordinates = np.argwhere(skeleton)
    lookup = {(int(y), int(x)): i for i, (y, x) in enumerate(coordinates)}
    neighbours: list[list[int]] = [[] for _ in coordinates]
    for index, (y, x) in enumerate(coordinates):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                other = lookup.get((int(y) + dy, int(x) + dx))
                if other is not None:
                    neighbours[index].append(other)
    degree = np.array([len(entry) for entry in neighbours], dtype=int)
    return coordinates, neighbours, degree


def branch_geometry(skeleton: np.ndarray, anchor: tuple[float, float]) -> list[dict]:
    """One row per branch of the cell's stick figure, with no length filter.

    **Why the junctions are contracted first.** Where two processes meet, the
    thinned skeleton almost never leaves a single pixel: on the pinned movie
    88.8% of branch points are clusters of two or more. Cutting the skeleton at
    every junction pixel - which is what a naive split does, and what
    ``skeleton_stats.skeleton_branches`` still does - therefore deletes the
    shortest branches outright, and the branch count comes to depend on how
    thick the join happened to be. Each cluster is collapsed to one node here,
    so a two-pixel branch between two fat junctions survives to be counted.

    **No minimum length, on purpose.** A median cell in this dataset is 11 px
    across and its branches are around 2 px long, so any floor that excludes
    thinning artefacts also excludes most real branches - at 3 px the average
    cell has 1.6 left. Rather than choose that floor here, every branch is
    written out with its length and its position, and whoever reads the table
    filters it themselves: by length, or by keeping only the larger cells. The
    per-cell-frame summaries beside it are computed over *all* branches, so
    ``mean_branch_px`` is an average over a handful of mostly-tiny segments and
    should be read with ``branch_count`` next to it.

    ``branch_px`` is the distance walked along the branch (diagonal steps count
    as sqrt 2); ``branch_pixels`` is how many pixels that path passes through.
    They differ by about one at this scale, which is why both are here.
    """
    coordinates, neighbours, degree = _skeleton_graph(skeleton)
    if len(coordinates) == 0:
        return []

    # Junction clusters become one node each; every tip becomes a node of its own.
    is_junction = degree >= 3
    cluster = np.full(len(coordinates), -1, dtype=int)
    clusters = 0
    for index in range(len(coordinates)):
        if not is_junction[index] or cluster[index] >= 0:
            continue
        stack = [index]
        cluster[index] = clusters
        while stack:
            current = stack.pop()
            for other in neighbours[current]:
                if is_junction[other] and cluster[other] < 0:
                    cluster[other] = clusters
                    stack.append(other)
        clusters += 1

    node_of: dict[int, int] = {i: int(cluster[i]) for i in range(len(coordinates))
                               if cluster[i] >= 0}
    tips: set[int] = set()
    for index in range(len(coordinates)):
        if degree[index] <= 1:
            node_of[index] = clusters + len(tips)
            tips.add(node_of[index])

    def step(a: int, b: int) -> float:
        return float(np.hypot(*(coordinates[a] - coordinates[b])))

    paths: list[list[int]] = []
    walked: set[tuple[int, int]] = set()
    for start in sorted(node_of):
        for first in neighbours[start]:
            if node_of.get(first) == node_of[start] or (start, first) in walked:
                continue
            path = [start, first]
            previous, current = start, first
            while current not in node_of:
                onward = [j for j in neighbours[current] if j != previous]
                if not onward:
                    break
                previous, current = current, onward[0]
                path.append(current)
            walked.add((start, first))
            if current in node_of and len(path) > 1:
                walked.add((current, path[-2]))
            paths.append(path)

    # A closed loop has no tips and no junctions, so nothing above seeds a walk.
    if not paths and len(coordinates) > 1:
        paths = [list(range(len(coordinates)))]

    # How many junctions from the root each node is. The root is the node
    # nearest the cell's own centre, so order 0 is a branch leaving the body.
    ends = {}
    for number, path in enumerate(paths):
        first, last = node_of.get(path[0]), node_of.get(path[-1])
        ends[number] = (first, last)
    adjacency: dict[int, set[int]] = {}
    for first, last in ends.values():
        for a, b in ((first, last), (last, first)):
            if a is not None:
                adjacency.setdefault(a, set())
                if b is not None:
                    adjacency[a].add(b)
    root = None
    if node_of:
        nearest = min(node_of, key=lambda i: float(
            np.hypot(coordinates[i][0] - anchor[0], coordinates[i][1] - anchor[1])))
        root = node_of[nearest]
    order: dict[int, int] = {}
    if root is not None:
        order[root] = 0
        frontier = [root]
        while frontier:
            nxt = []
            for node in frontier:
                for other in adjacency.get(node, ()):
                    if other not in order:
                        order[other] = order[node] + 1
                        nxt.append(other)
            frontier = nxt

    rows: list[dict] = []
    for number, path in enumerate(paths):
        length = sum(step(path[i], path[i + 1]) for i in range(len(path) - 1))
        first, last = ends[number]
        depths = [order[node] for node in (first, last) if node in order]
        far = max(
            float(np.hypot(coordinates[end][0] - anchor[0], coordinates[end][1] - anchor[1]))
            for end in (path[0], path[-1])
        )
        rows.append(
            {
                "branch": number,
                "branch_px": length,
                "branch_pixels": len(path),
                "branch_order": min(depths) if depths else np.nan,
                "branch_is_tip": bool({first, last} & tips),
                "branch_end_distance_px": far,
            }
        )
    return rows


def _anchor(mask: np.ndarray, signal: np.ndarray | None) -> tuple[tuple[float, float], bool]:
    """Where the cell body is, and whether that point is inside the outline.

    The brightness-weighted centre when there is a raw stack - for these
    reporters the signal piles up in the soma - and the plain centroid when
    there is not, so a movie without ``raw`` still gets branch orders. The flag
    matters because a C-shaped cell can put its own centre outside itself, which
    on the pinned movie happens in about 3% of cell-frames; the branch orders
    are then measured from a point the cell does not occupy.
    """
    if signal is not None and np.any(signal > 0):
        centre = ndi.center_of_mass(np.where(mask, signal, 0.0))
    else:
        centre = ndi.center_of_mass(mask)
    y, x = float(centre[0]), float(centre[1])
    row, column = int(round(y)), int(round(x))
    inside = (
        0 <= row < mask.shape[0] and 0 <= column < mask.shape[1] and bool(mask[row, column])
    )
    return (y, x), inside


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    min_row, min_col, max_row, max_col = bbox
    return min_row == 0 or min_col == 0 or max_row == shape[0] or max_col == shape[1]


#: What this module writes, and how the rest of the package should say it. The
#: centroid is declared identically by ``motility``, which writes it too; the
#: join keeps one copy and both have to mean the same thing by it.
PRODUCES = (
    Column("centroid_y", "Centroid, y", "px", "morphology"),
    Column("centroid_x", "Centroid, x", "px", "morphology"),
    Column("area_px", "Area", "px", "morphology"),
    Column("perimeter_px", "Perimeter", "px", "morphology"),
    Column("convex_area_px", "Convex area", "px", "morphology"),
    Column("equivalent_diameter_px", "Equivalent diameter", "px", "morphology"),
    Column("major_axis_px", "Major axis", "px", "morphology"),
    Column("minor_axis_px", "Minor axis", "px", "morphology"),
    Column("orientation_rad", "Orientation", "rad", "morphology"),
    Column("eccentricity", "Eccentricity", "0-1", "morphology"),
    Column("solidity", "Solidity", "0-1", "morphology"),
    Column("extent", "Extent", "0-1", "morphology"),
    Column("euler_number", "Euler number", "count", "morphology"),
    Column("circularity", "Circularity", "0-1", "morphology"),
    Column("ramification_index", "Ramification", "perimeter for area", "morphology"),
    Column("aspect_ratio", "Aspect ratio", "ratio", "morphology"),
    Column("is_fragment", "Fragment of a cell", "0 or 1", "morphology"),
    Column("n_components", "Disconnected pieces", "count", "morphology"),
    Column("shape_ratios_valid", "Shape ratios usable", "0 or 1", "morphology"),
    Column("touches_border", "Touches the field edge", "0 or 1", "morphology"),
    Column("skeleton_px", "Skeleton length", "px", "morphology"),
    Column("skeleton_endpoints", "Process tips", "count", "morphology"),
    Column("skeleton_junctions", "Branch points", "count", "morphology"),
    Column("skeleton_branches", "Process branches", "count", "morphology"),
    Column("soma_centre_inside", "Cell body centre falls inside the outline", "0 or 1", "morphology"),
    # Branch geometry, summarised per cell-frame. Every one of these is over
    # *all* branches with no minimum length, so read them with `branch_count`
    # beside them - see `branch_geometry` for why there is no floor.
    Column("branch_count", "Process branches, junction clusters contracted", "count", "morphology"),
    Column("longest_branch_px", "Longest process branch", "px", "morphology"),
    Column("mean_branch_px", "Process branch length, mean", "px", "morphology"),
    Column("median_branch_px", "Process branch length, median", "px", "morphology"),
    Column("total_branch_px", "Process branch length, total", "px", "morphology"),
    Column("tip_branch_count", "Branches ending in a free tip", "count", "morphology"),
    Column("max_branch_order", "Junctions from the body to the furthest branch", "count", "morphology"),
    # branches - one row per branch per cell per frame
    Column("branch", "Which branch of this cell-frame", "identifier", "reference"),
    Column("branch_px", "Length walked along the branch", "px", "morphology"),
    Column("branch_pixels", "Pixels the branch passes through", "count", "morphology"),
    Column("branch_order", "Junctions between this branch and the body", "count", "morphology"),
    Column("branch_is_tip", "Branch ends in a free tip", "0 or 1", "morphology"),
    Column("branch_end_distance_px", "Far end of the branch from the body", "px", "morphology"),
)


#: ``branches`` is a file of its own, not a fold: one row per branch is a finer
#: grain than any roll-up has. It is written unfiltered so that choosing a
#: minimum branch length stays the reader's decision rather than this module's.
WRITES = (
    Output("morphology", grain=("identity", "frame_index"), fold=True),
    Output("branches",   grain=("identity", "frame_index", "branch")),
)


@register(
    name="morphology",
    description="Per-cell shape: area, perimeter, circularity, solidity and skeleton branching",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("morphology")}
    scale = context.scale
    field_shape = context.labels.shape[1:]
    rows: list[dict] = []
    branch_rows: list[dict] = []

    for frame_index in range(context.n_frames):
        frame = context.labels[frame_index]
        signal = context.raw[frame_index] if context.raw is not None else None
        for region in regionprops(frame):
            area = float(region.area)
            perimeter = float(region.perimeter)
            convex_area = float(region.area_convex)
            minor = float(region.axis_minor_length)
            is_fragment = area < params["min_area_px_for_shape_ratios"]
            n_components = int(ndi.label(region.image, structure=np.ones((3, 3)))[1])
            circularity = 4.0 * np.pi * area / perimeter ** 2 if perimeter > 0 else np.nan
            shape_ratios_valid = bool(
                (not is_fragment)
                and perimeter > 0
                and np.isfinite(circularity)
                and circularity <= params["max_plausible_circularity"]
            )
            if not shape_ratios_valid:
                circularity = np.nan

            row = {
                "identity": int(region.label),
                "frame_index": frame_index,
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "area_px": area,
                "perimeter_px": perimeter,
                "convex_area_px": convex_area,
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "major_axis_px": float(region.axis_major_length),
                "minor_axis_px": minor,
                "orientation_rad": float(region.orientation),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "extent": float(region.extent),
                "euler_number": int(region.euler_number),
                "circularity": circularity,
                # Perimeter for a given area, normalised so a disc is 1.
                "ramification_index": (
                    perimeter / (2.0 * np.sqrt(np.pi * area)) if shape_ratios_valid else np.nan
                ),
                "aspect_ratio": (
                    float(region.axis_major_length) / minor if minor > 0 else np.nan
                ),
                "is_fragment": bool(is_fragment),
                "n_components": n_components,
                "shape_ratios_valid": shape_ratios_valid,
                "touches_border": _touches_border(region.bbox, field_shape),
            }

            if params["compute_skeleton"] and area >= params["min_area_px_for_skeleton"]:
                skeleton = skeletonize(region.image)
                row.update(skeleton_stats(region.image, skeleton))

                # The anchor the branch orders are measured from: the cell body,
                # taken from the raw signal where there is one. `region.image`
                # is cropped to the bounding box, so the raw has to be cropped
                # the same way before the two can be weighed against each other.
                top, left, bottom, right = region.bbox
                window = signal[top:bottom, left:right].astype(np.float64) if signal is not None else None
                anchor, inside = _anchor(region.image, window)
                row["soma_centre_inside"] = inside

                branches = branch_geometry(skeleton, anchor)
                lengths = np.array([b["branch_px"] for b in branches], dtype=float)
                orders = np.array([b["branch_order"] for b in branches], dtype=float)
                row.update({
                    "branch_count": len(branches),
                    "longest_branch_px": float(lengths.max()) if lengths.size else np.nan,
                    "mean_branch_px": float(lengths.mean()) if lengths.size else np.nan,
                    "median_branch_px": float(np.median(lengths)) if lengths.size else np.nan,
                    "total_branch_px": float(lengths.sum()) if lengths.size else 0.0,
                    "tip_branch_count": int(sum(b["branch_is_tip"] for b in branches)),
                    "max_branch_order": (
                        float(np.nanmax(orders)) if orders.size and np.any(np.isfinite(orders))
                        else np.nan
                    ),
                })
                for branch in branches:
                    branch_rows.append({
                        "identity": int(region.label),
                        "frame_index": frame_index,
                        **branch,
                    })
            else:
                row.update({"skeleton_px": np.nan, "skeleton_endpoints": np.nan,
                            "skeleton_junctions": np.nan, "skeleton_branches": np.nan,
                            "soma_centre_inside": np.nan, "branch_count": np.nan,
                            "longest_branch_px": np.nan, "mean_branch_px": np.nan,
                            "median_branch_px": np.nan, "total_branch_px": np.nan,
                            "tip_branch_count": np.nan, "max_branch_order": np.nan})

            if scale.calibrated:
                row["area_um2"] = scale.area(area)
                row["perimeter_um"] = scale.length(perimeter)
                row["equivalent_diameter_um"] = scale.length(row["equivalent_diameter_px"])
                row["skeleton_length_um"] = scale.length(row["skeleton_px"])

            rows.append(row)

    cell_frame = pd.DataFrame(rows).sort_values(["identity", "frame_index"]).reset_index(drop=True)
    branches = pd.DataFrame(
        branch_rows,
        columns=["identity", "frame_index", "branch", "branch_px", "branch_pixels",
                 "branch_order", "branch_is_tip", "branch_end_distance_px"],
    ).sort_values(["identity", "frame_index", "branch"]).reset_index(drop=True)
    return {"morphology": cell_frame, "branches": branches}
