"""General acceptance gate for newly allocated identity fragments.

The gate compares a frozen accepted label movie with a candidate.  It receives
no cell, frame, event, or coordinate targets.  A new identity is considered an
unanchored fragment when all of its candidate pixels were ownerless in the
accepted labels, its complete lifetime is temporally and spatially interior,
and its span is shorter than a movie-relative minimum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MINIMUM_UNANCHORED_LIFESPAN_FRACTION = 0.15
DEFAULT_SPATIAL_MARGIN_FRACTION = 0.01


def score_new_identity_fragments(
        baseline: np.ndarray,
        candidate: np.ndarray,
        *,
        minimum_unanchored_lifespan_fraction: float =
        DEFAULT_MINIMUM_UNANCHORED_LIFESPAN_FRACTION,
        spatial_margin_fraction: float = DEFAULT_SPATIAL_MARGIN_FRACTION,
) -> tuple[pd.DataFrame, dict]:
    """Score every identity that exists only in ``candidate``.

    The confidence is one for identities linked to pre-existing named pixels,
    censored by a movie/image boundary, or long enough to establish an
    independent lineage.  Otherwise it is the observed span divided by the
    movie-relative minimum span.  Promotion requires a minimum confidence of
    one, so short ownerless interior allocations cannot improve persistence by
    defining their own conveniently short lifespan.
    """
    baseline = np.asarray(baseline)
    candidate = np.asarray(candidate)
    if baseline.shape != candidate.shape or baseline.ndim != 3:
        raise ValueError("baseline and candidate must be matching T-Y-X arrays")
    if not 0 < minimum_unanchored_lifespan_fraction <= 1:
        raise ValueError("minimum lifespan fraction must be in (0, 1]")
    if not 0 <= spatial_margin_fraction < 0.5:
        raise ValueError("spatial margin fraction must be in [0, 0.5)")

    frame_count, height, width = candidate.shape
    baseline_ids = set(np.unique(baseline).astype(int)) - {0}
    candidate_ids = set(np.unique(candidate).astype(int)) - {0}
    new_ids = sorted(candidate_ids - baseline_ids)
    minimum_span_frames = max(
        1, int(np.ceil(minimum_unanchored_lifespan_fraction * frame_count)))
    margin = max(1, int(np.ceil(
        spatial_margin_fraction * min(height, width))))

    rows: list[dict] = []
    for identity in new_ids:
        mask = candidate == identity
        frame_presence = mask.reshape(frame_count, -1).any(axis=1)
        frames = np.flatnonzero(frame_presence)
        first_frame = int(frames[0])
        last_frame = int(frames[-1])
        span_frames = last_frame - first_frame + 1
        named_frames = int(len(frames))
        named_pixels = int(mask.sum())
        inherited_named_pixels = int(np.count_nonzero(mask & (baseline > 0)))
        temporally_interior = first_frame > 0 and last_frame < frame_count - 1
        spatial_boundary_contact = bool(
            mask[:, :margin].any() or mask[:, -margin:].any()
            or mask[:, :, :margin].any() or mask[:, :, -margin:].any())
        unanchored = inherited_named_pixels == 0
        exposed = (
            unanchored
            and temporally_interior
            and not spatial_boundary_contact
        )
        confidence = (
            min(1.0, span_frames / minimum_span_frames)
            if exposed else 1.0
        )
        rejected = exposed and span_frames < minimum_span_frames
        rows.append({
            "identity": identity,
            "first_frame_index": first_frame,
            "last_frame_index": last_frame,
            "span_frames": span_frames,
            "named_frames": named_frames,
            "named_pixels": named_pixels,
            "inherited_named_pixels": inherited_named_pixels,
            "temporally_interior": temporally_interior,
            "spatial_boundary_contact": spatial_boundary_contact,
            "unanchored": unanchored,
            "minimum_span_frames": minimum_span_frames,
            "fragmentation_confidence": confidence,
            "status": "rejected_short_unanchored_fragment" if rejected else "pass",
        })

    columns = [
        "identity", "first_frame_index", "last_frame_index", "span_frames",
        "named_frames", "named_pixels", "inherited_named_pixels",
        "temporally_interior", "spatial_boundary_contact", "unanchored",
        "minimum_span_frames", "fragmentation_confidence", "status",
    ]
    table = pd.DataFrame(rows, columns=columns)
    rejected_count = int((table.status != "pass").sum()) if len(table) else 0
    minimum_confidence = (
        float(table.fragmentation_confidence.min()) if len(table) else 1.0)
    summary = {
        "formula": (
            "min(1, new_identity_span / "
            "ceil(minimum_fraction * movie_frames)); exempt when linked to "
            "accepted foreground or censored by a temporal/spatial boundary"
        ),
        "movie_frames": frame_count,
        "minimum_unanchored_lifespan_fraction":
            minimum_unanchored_lifespan_fraction,
        "minimum_span_frames": minimum_span_frames,
        "spatial_margin_fraction": spatial_margin_fraction,
        "spatial_margin_pixels": margin,
        "new_identities": len(table),
        "rejected_fragments": rejected_count,
        "minimum_fragmentation_confidence": minimum_confidence,
        "promotion_gate_passed": rejected_count == 0,
        "identity_target_count": 0,
        "frame_target_count": 0,
        "coordinate_target_count": 0,
        "event_target_count": 0,
    }
    return table, summary
