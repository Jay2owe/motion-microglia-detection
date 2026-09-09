"""How much of each cell is rebuilt from one frame to the next.

Microglia mostly stay put and instead sweep the tissue by extending and
retracting processes. A cell can therefore have almost zero centroid movement
and still be highly active, which motility alone would score as inert.

The measurement is a direct comparison of the cell's own pixels between
consecutive frames: pixels it kept, pixels it grew into, pixels it gave up. The
turnover index is the fraction of the cell that is not the same pixels it was
30 minutes ago - think of it as how much of a coastline the tide redraws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import Column, MeasurementContext, Output, register

DEFAULTS = {
    "max_gap_frames_for_turnover": 1,
}

DEFAULT_EVIDENCE_CHANNELS = ["base", "lost", "held", "gained", "trail"]

DEFAULTS_EVIDENCE = {
    "channel_names": DEFAULT_EVIDENCE_CHANNELS,
    # Which channel, if any, encodes how long ago a pixel was last disturbed.
    # Named rather than assumed to be the fifth, because a tracker that writes
    # four channels or writes them in another order is not a broken tracker.
    "trail_channel": "trail",
    # How many steps that channel's ramp has. ``None`` infers it from the
    # stack, which works whenever the ramp is evenly spaced and short; set it
    # to a number when a tracker writes something the inference declines.
    "trail_levels": None,
}


def _channel_scale(channel: np.ndarray) -> float:
    """The value in this channel that means "as much as this channel goes".

    Needed because the evidence stack is graded, not binary, and a total in raw
    units is unreadable without knowing what full scale is. An integer channel
    is scaled by what its dtype holds - which is what an encoder writing a ramp
    into a 16-bit image is working to - and a float channel by what it actually
    reaches, because there is nothing else to go on.
    """
    if channel.dtype == np.bool_:
        return 1.0
    if np.issubdtype(channel.dtype, np.integer):
        return float(np.iinfo(channel.dtype).max)
    finite = channel[np.isfinite(channel)]
    largest = float(finite.max()) if finite.size else 0.0
    return largest if largest > 0 else 1.0


def _trail_levels(channel: np.ndarray, scale: float, limit: int = 64) -> int | None:
    """How many steps the trail ramp has, read off the stack, or ``None``.

    A trail channel encodes recency as brightness: a pixel disturbed in this
    transition is written at full scale and the value steps down once per
    transition until it runs out. The number of steps is the tracker's, not
    this package's, so it is measured rather than assumed - the distinct values
    are counted, the step between them is taken as full scale divided by that
    count, and every value has to sit on that grid or the answer is ``None``.

    ``None`` is the honest outcome for a continuous channel or one with more
    steps than a ramp plausibly has, and it leaves the age columns blank rather
    than filling them with a number derived from a guess.
    """
    values = np.unique(channel)
    values = values[values > 0].astype(np.float64)
    if values.size == 0 or values.size > limit or scale <= 0:
        return None
    step = float(values.min())
    if values.size > 1:
        step = min(step, float(np.diff(values).min()))
    if step <= 0:
        return None
    levels = int(round(scale / step))
    if not 1 <= levels <= limit:
        return None
    # Every value must sit on the grid this many steps implies. A tenth of a
    # step is the tolerance, which is wide enough for a ramp rounded into
    # integers and far too narrow for a channel that is not a ramp at all.
    on_grid = values / scale * levels
    if np.abs(on_grid - np.rint(on_grid)).max() > 0.1:
        return None
    return levels


def _trail_age(values: np.ndarray, scale: float, levels: int) -> np.ndarray:
    """Transitions since each trailed pixel was last disturbed.

    Brightest is freshest, so the ramp is read upside down: full scale is one
    transition ago, one step above zero is ``levels`` transitions ago. Pixels
    with no trail at all are dropped rather than counted as infinitely old,
    because "never disturbed within memory" and "disturbed a long time ago" are
    different statements and only the second is a measurement.
    """
    trailed = values[values > 0]
    if trailed.size == 0:
        return np.empty(0, dtype=np.float64)
    step = np.rint(trailed.astype(np.float64) / scale * levels)
    return (levels + 1) - step


#: Every number here describes a transition between two frames rather than a
#: state, which is why ``turnover_index`` is explicitly a fraction per
#: ``{interval}`` - a 15 min recording must not be labelled 30 min.
PRODUCES = (
    Column("from_frame_index", "Frame the change started from", "frame", "reference"),
    Column("gap_frames", "Frames missing", "frames", "reference"),
    Column("turnover_index", "Footprint turnover fraction",
           "fraction per {interval}", "surveillance"),
    Column("jaccard", "Footprint overlap with previous frame", "0-1", "surveillance"),
    Column("held_px", "Pixels held", "px", "surveillance"),
    Column("gained_px", "Pixels gained", "px", "surveillance"),
    Column("lost_px", "Pixels lost", "px", "surveillance"),
    Column("area_change_px", "Area change", "px", "surveillance"),
    Column("extension_bias", "Extension bias", "-1 to 1", "surveillance"),
    Column("balanced_turnover_fraction", "Balanced footprint turnover", "fraction",
           "surveillance"),
)


WRITES = (
    Output("surveillance", grain=("identity", "frame_index"), fold=True),
)


@register(
    name="surveillance",
    description="Per-cell pixel turnover between frames: processes extending and retracting",
    requires=("labels",),
    defaults=DEFAULTS,
    produces=PRODUCES,
    writes=WRITES,
)
def measure(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    params = {**DEFAULTS, **context.module_params("surveillance")}
    max_gap = int(params["max_gap_frames_for_turnover"])
    rows: list[dict] = []

    previous: dict[int, tuple[int, np.ndarray]] = {}
    for frame_index in range(context.n_frames):
        labels = context.labels[frame_index]
        present = [int(v) for v in np.unique(labels) if v]
        current: dict[int, np.ndarray] = {}
        for identity in present:
            mask = labels == identity
            current[identity] = mask
            if identity not in previous:
                continue
            last_frame, last_mask = previous[identity]
            gap = frame_index - last_frame
            if gap > max_gap:
                continue
            held = int(np.count_nonzero(mask & last_mask))
            gained = int(np.count_nonzero(mask & ~last_mask))
            lost = int(np.count_nonzero(last_mask & ~mask))
            union = held + gained + lost
            total = int(mask.sum()) + int(last_mask.sum())
            rows.append(
                {
                    "identity": identity,
                    "frame_index": frame_index,
                    "from_frame_index": last_frame,
                    "gap_frames": gap,
                    "held_px": held,
                    "gained_px": gained,
                    "lost_px": lost,
                    "turnover_index": (gained + lost) / total if total else np.nan,
                    "jaccard": held / union if union else np.nan,
                    "area_change_px": int(mask.sum()) - int(last_mask.sum()),
                    "extension_bias": (
                        (gained - lost) / (gained + lost) if (gained + lost) else np.nan
                    ),
                    "balanced_turnover_fraction": (
                        1 - abs(gained - lost) / (gained + lost)
                        if (gained + lost) else np.nan
                    ),
                }
            )
        previous = {identity: (frame_index, mask) for identity, mask in current.items()} | {
            identity: value for identity, value in previous.items() if identity not in current
        }

    return {"surveillance": pd.DataFrame(rows)}


#: Two tables: one row per cell per frame, and one row per frame for the whole
#: field. The ``frame_`` and ``claimed_`` prefixes are the same four counts over
#: two different populations - everything the evidence stack marked, and the
#: part of it that fell inside somebody's outline - so the labels say which.
PRODUCES_EVIDENCE = (
    # motion_evidence - one row per cell per frame
    Column("footprint_px", "Footprint", "px", "evidence"),
    Column("evidence_base_px", "Evidence base", "px", "evidence"),
    Column("evidence_lost_px", "Evidence lost", "px", "evidence_lost"),
    Column("evidence_held_px", "Evidence held", "px", "evidence_held"),
    Column("evidence_gained_px", "Evidence gained", "px", "evidence_gained"),
    Column("evidence_trail_px", "Evidence trail", "px", "evidence"),
    Column("evidence_motion_fraction", "Own pixels moving", "fraction", "evidence"),
    # The same five channels totalled rather than counted. A count answers "how
    # many pixels changed"; a total answers "how much signal changed", and the
    # two come apart exactly where it matters - a bright process sweeping across
    # a few pixels against a dim smear over many. Everything is a share of full
    # scale so the numbers are readable without knowing the encoder's dtype.
    Column("evidence_base_sum", "Evidence base, total", "share of full scale", "evidence"),
    Column("evidence_lost_sum", "Evidence lost, total", "share of full scale", "evidence_lost"),
    Column("evidence_held_sum", "Evidence held, total", "share of full scale", "evidence_held"),
    Column("evidence_gained_sum", "Evidence gained, total", "share of full scale", "evidence_gained"),
    Column("evidence_trail_sum", "Evidence trail, total", "share of full scale", "evidence"),
    # How long ago the ground under this cell was last disturbed. Not derivable
    # from the counts above: the trail channel encodes recency in its
    # brightness, and counting its pixels throws exactly that away.
    Column("trail_age_frames_mean", "Ground last disturbed", "frames", "surveillance"),
    Column("trail_age_frames_min", "Freshest ground under the cell", "frames", "surveillance"),
    Column("trail_age_hours_mean", "Ground last disturbed", "h", "surveillance"),
    # motion_evidence_frame - one row per frame, whole field
    Column("from_frame_index", "Frame the change started from", "frame", "reference"),
    Column("to_frame_index", "Frame the change ended at", "frame", "reference"),
    Column("frame_base_px", "Evidence base, whole field", "px", "evidence"),
    Column("frame_lost_px", "Evidence lost, whole field", "px", "evidence_lost"),
    Column("frame_held_px", "Evidence held, whole field", "px", "evidence_held"),
    Column("frame_gained_px", "Evidence gained, whole field", "px", "evidence_gained"),
    Column("frame_trail_px", "Evidence trail, whole field", "px", "evidence"),
    Column("frame_gained_share", "New motion-evidence share", "fraction", "evidence_gained"),
    Column("claimed_base_px", "Evidence base inside an outline", "px", "evidence"),
    Column("claimed_lost_px", "Evidence lost inside an outline", "px", "evidence_lost"),
    Column("claimed_held_px", "Evidence held inside an outline", "px", "evidence_held"),
    Column("claimed_gained_px", "Evidence gained inside an outline", "px", "evidence_gained"),
    Column("claimed_trail_px", "Evidence trail inside an outline", "px", "evidence"),
    Column("frame_base_sum", "Evidence base, whole field total", "share of full scale", "evidence"),
    Column("frame_lost_sum", "Evidence lost, whole field total", "share of full scale", "evidence_lost"),
    Column("frame_held_sum", "Evidence held, whole field total", "share of full scale", "evidence_held"),
    Column("frame_gained_sum", "Evidence gained, whole field total", "share of full scale", "evidence_gained"),
    Column("frame_trail_sum", "Evidence trail, whole field total", "share of full scale", "evidence"),
    Column("claimed_base_sum", "Evidence base inside an outline, total", "share of full scale", "evidence"),
    Column("claimed_lost_sum", "Evidence lost inside an outline, total", "share of full scale", "evidence_lost"),
    Column("claimed_held_sum", "Evidence held inside an outline, total", "share of full scale", "evidence_held"),
    Column("claimed_gained_sum", "Evidence gained inside an outline, total", "share of full scale", "evidence_gained"),
    Column("claimed_trail_sum", "Evidence trail inside an outline, total", "share of full scale", "evidence"),
    # The field's own memory, which is what says whether ground the cells are
    # not on now was swept an hour ago or two days ago.
    Column("frame_trail_age_mean", "Field last disturbed", "frames", "surveillance"),
    Column("frame_trail_age_median", "Field last disturbed, median", "frames", "surveillance"),
    Column("trail_levels", "Steps in the trail ramp", "steps", "reference"),
)


#: ``motion_evidence_frame`` is one row per *gap between* frames, keyed on
#: ``from_frame_index``, so it has one row fewer than there are frames. It
#: reads like frame grain and is not, which is why it is not folded into
#: ``frame_summary``: the last frame would come out blank.
WRITES_EVIDENCE = (
    Output("motion_evidence",       grain=("identity", "frame_index"), fold=True),
    Output("motion_evidence_frame", grain=("from_frame_index",)),
)


@register(
    name="motion_evidence",
    description="Motion-evidence pixels attributed to each cell, from the tracker's own evidence stack",
    requires=("labels", "evidence"),
    defaults=DEFAULTS_EVIDENCE,
    produces=PRODUCES_EVIDENCE,
    writes=WRITES_EVIDENCE,
)
def measure_evidence(context: MeasurementContext) -> dict[str, pd.DataFrame]:
    """Two tables: evidence attributed to each cell, and the whole-frame total.

    The per-cell table answers "how much of this cell moved". The frame table
    answers a prior question the per-cell one cannot: how much evidence there
    was at all. They differ by the evidence that fell on no named cell, which is
    exactly the part a tracking failure hides in - so the two are kept apart
    rather than one being derived from the other.
    """
    params = {**DEFAULTS_EVIDENCE, **context.module_params("motion_evidence")}
    names = list(params["channel_names"])
    evidence = context.evidence
    n_channels = evidence.shape[1]
    if len(names) < n_channels:
        names = names + [f"channel_{i}" for i in range(len(names), n_channels)]

    # Full scale per channel, once for the movie rather than once per cell.
    scales = [_channel_scale(evidence[:, channel]) for channel in range(n_channels)]

    # Which channel carries recency, and how many steps its ramp has. Both may
    # come out absent, and then the age columns stay blank for the whole run
    # rather than being filled from an assumption about somebody else's encoder.
    trail_name = params.get("trail_channel")
    trail_channel = names.index(trail_name) if trail_name in names[:n_channels] else None
    levels = params.get("trail_levels")
    if trail_channel is not None:
        levels = (
            int(levels) if levels
            else _trail_levels(evidence[:, trail_channel], scales[trail_channel])
        )
    else:
        levels = None

    rows: list[dict] = []
    # The last frame has no outgoing transition inside the label range.
    for frame_index in range(context.n_frames - 1):
        labels = context.labels[frame_index]
        next_labels = context.labels[frame_index + 1]
        present = [int(v) for v in np.unique(labels) if v]
        for identity in present:
            footprint = (labels == identity) | (next_labels == identity)
            if not footprint.any():
                continue
            row = {
                "identity": identity,
                "frame_index": frame_index,
                "footprint_px": int(footprint.sum()),
            }
            for channel in range(n_channels):
                inside = evidence[frame_index, channel][footprint]
                row[f"evidence_{names[channel]}_px"] = int(np.count_nonzero(inside))
                # float64 accumulator, said out loud. Whether a uint16 sum
                # widens far enough is a numpy-version and platform question,
                # and a field-wide total of a 16-bit channel is comfortably
                # past what 32 bits hold.
                row[f"evidence_{names[channel]}_sum"] = float(
                    inside.sum(dtype=np.float64) / scales[channel]
                )
            if levels:
                ages = _trail_age(
                    evidence[frame_index, trail_channel][footprint],
                    scales[trail_channel], levels,
                )
                if ages.size:
                    row["trail_age_frames_mean"] = float(ages.mean())
                    row["trail_age_frames_min"] = float(ages.min())
                    row["trail_age_hours_mean"] = context.scale.hours(float(ages.mean()))
            rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty and "evidence_gained_px" in table and "evidence_lost_px" in table:
        moved = table["evidence_gained_px"] + table["evidence_lost_px"]
        table["evidence_motion_fraction"] = moved / table["footprint_px"]

    # The same channels over the whole field, one row per transition. The last
    # frame has no outgoing transition, so it carries no row rather than a zero.
    frames = context.frame_table().iloc[: context.n_frames - 1].copy()
    frames = frames.rename(columns={"frame_index": "from_frame_index"})
    frames.insert(1, "to_frame_index", frames["from_frame_index"] + 1)
    for channel in range(n_channels):
        frames[f"frame_{names[channel]}_px"] = [
            int(np.count_nonzero(evidence[index, channel]))
            for index in range(context.n_frames - 1)
        ]
        frames[f"frame_{names[channel]}_sum"] = [
            float(evidence[index, channel].sum(dtype=np.float64) / scales[channel])
            for index in range(context.n_frames - 1)
        ]
    if levels:
        # The field's memory, over every pixel that carries a trail rather than
        # only the ones a cell is standing on now. `trail_levels` rides along as
        # a column because it decides what the two ages mean and it was read off
        # the stack, so a reader who wants to check the arithmetic can.
        field_ages = [
            _trail_age(evidence[index, trail_channel], scales[trail_channel], levels)
            for index in range(context.n_frames - 1)
        ]
        frames["frame_trail_age_mean"] = [
            float(a.mean()) if a.size else np.nan for a in field_ages
        ]
        frames["frame_trail_age_median"] = [
            float(np.median(a)) if a.size else np.nan for a in field_ages
        ]
        frames["trail_levels"] = int(levels)
    if "frame_gained_px" in frames and "frame_lost_px" in frames:
        # Signal that leaves one place arrives in another, so these two should
        # track each other. Their ratio is the check; a frame where one runs
        # away from the other is a frame where evidence is being invented.
        total = frames["frame_gained_px"] + frames["frame_lost_px"]
        frames["frame_gained_share"] = frames["frame_gained_px"] / total.replace(0, np.nan)
    if not table.empty:
        claimed = table.groupby("frame_index")[
            [column for column in table.columns if column.startswith("evidence_")
             and column.endswith(("_px", "_sum"))]
        ].sum()
        claimed.columns = [column.replace("evidence_", "claimed_") for column in claimed.columns]
        frames = frames.merge(
            claimed.reset_index().rename(columns={"frame_index": "from_frame_index"}),
            on="from_frame_index", how="left",
        )

    return {"motion_evidence": table, "motion_evidence_frame": frames}
