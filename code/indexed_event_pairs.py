"""Exact spatial indexing for disruption-event collapse pairs."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import math

import numpy as np


def _all_pairs(events: list[dict], maximum_frame_gap: int,
               maximum_distance_px: float) -> Iterator[tuple[int, int]]:
    """Preserve the accepted search for unusual, non-indexable inputs."""
    for left_index, left in enumerate(events):
        for right_index in range(left_index + 1, len(events)):
            right = events[right_index]
            frame_gap = max(
                0,
                max(left["first_frame"], right["first_frame"])
                - min(left["last_frame"], right["last_frame"]) - 1,
            )
            distance = float(np.hypot(
                left["x"] - right["x"], left["y"] - right["y"]))
            if (frame_gap <= maximum_frame_gap
                    and distance <= maximum_distance_px):
                yield left_index, right_index


def candidate_pairs(events: list[dict], maximum_frame_gap: int = 3,
                    maximum_distance_px: float = 18.0,
                    *, diagnostics: dict | None = None
                    ) -> Iterator[tuple[int, int]]:
    """Yield the exact accepted pairs after indexed, vectorised screening.

    The square width equals the accepted distance threshold. A qualifying
    earlier point can therefore occur only in the current square or one of its
    eight neighbours. The accepted frame-gap and Euclidean-distance equations
    still make every decision.
    """
    count = len(events)
    possible_pairs = count * (count - 1) // 2
    checked_pairs = 0

    if count < 2:
        if diagnostics is not None:
            diagnostics.update({
                "possible_pairs": possible_pairs,
                "spatial_candidate_pairs": 0,
                "qualifying_pairs": 0,
                "fallback_all_pairs": False,
            })
        return

    starts = np.fromiter(
        (event["first_frame"] for event in events), dtype=np.int64, count=count)
    ends = np.fromiter(
        (event["last_frame"] for event in events), dtype=np.int64, count=count)
    xs = np.fromiter((event["x"] for event in events), dtype=np.float64,
                     count=count)
    ys = np.fromiter((event["y"] for event in events), dtype=np.float64,
                     count=count)

    indexable = (
        maximum_frame_gap >= 0
        and math.isfinite(maximum_distance_px)
        and maximum_distance_px > 0
        and bool(np.isfinite(xs).all())
        and bool(np.isfinite(ys).all())
    )
    if not indexable:
        qualifying = 0
        for pair in _all_pairs(
                events, maximum_frame_gap, maximum_distance_px):
            qualifying += 1
            yield pair
        if diagnostics is not None:
            diagnostics.update({
                "possible_pairs": possible_pairs,
                "spatial_candidate_pairs": possible_pairs,
                "qualifying_pairs": qualifying,
                "fallback_all_pairs": True,
            })
        return

    width = float(maximum_distance_px)
    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    qualifying = 0
    for right_index in range(count):
        cell_x = math.floor(xs[right_index] / width)
        cell_y = math.floor(ys[right_index] / width)
        blocks = [
            cells[(cell_x + offset_x, cell_y + offset_y)]
            for offset_x in (-1, 0, 1)
            for offset_y in (-1, 0, 1)
            if cells.get((cell_x + offset_x, cell_y + offset_y))
        ]
        if blocks:
            candidates = np.fromiter(
                (index for block in blocks for index in block),
                dtype=np.int64,
                count=sum(map(len, blocks)),
            )
            checked_pairs += int(candidates.size)
            frame_gaps = np.maximum(
                0,
                np.maximum(starts[candidates], starts[right_index])
                - np.minimum(ends[candidates], ends[right_index]) - 1,
            )
            temporally_close = frame_gaps <= maximum_frame_gap
            nearby = candidates[temporally_close]
            if nearby.size:
                distances = np.hypot(
                    xs[nearby] - xs[right_index],
                    ys[nearby] - ys[right_index],
                )
                for left_index in nearby[distances <= maximum_distance_px]:
                    qualifying += 1
                    yield int(left_index), right_index
        cells[(cell_x, cell_y)].append(right_index)

    if diagnostics is not None:
        diagnostics.update({
            "possible_pairs": possible_pairs,
            "spatial_candidate_pairs": checked_pairs,
            "qualifying_pairs": qualifying,
            "fallback_all_pairs": False,
        })
