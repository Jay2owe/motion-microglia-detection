"""Roll-ups.

Three levels, and every figure in the package is drawn from one of them:

* ``cell_frame`` - one row per cell per frame, the raw material;
* ``cell_summary`` - one row per cell, for comparing cells;
* ``frame_summary`` - one row per frame, for the population over time.

Nothing here re-measures anything. If a number is wrong the fault is in a
measurement module, not in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.registry import MeasurementContext, Output, declared_tables

#: The three tables this file builds, declared the same way a module declares
#: its own. Nothing else in the package knows what one row of ``cell_summary``
#: is, and the writer needs to: a module's table folds into whichever roll-up
#: shares its grain, so the roll-ups have to say what their grain is.
#:
#: They are declared here rather than in ``registry`` because this is the file
#: that builds them, which is the same rule every module follows.
ROLLUPS: tuple[Output, ...] = (
    Output("cell_frame", grain=("identity", "frame_index")),
    Output("cell_summary", grain=("identity",)),
    Output("frame_summary", grain=("frame_index",)),
)

# Columns worth a per-cell median and spread. Anything absent is skipped, so a
# run with modules switched off still produces a valid summary.
SUMMARY_METRICS = [
    "area_px",
    "perimeter_px",
    "circularity",
    "solidity",
    "ramification_index",
    "aspect_ratio",
    "skeleton_px",
    "skeleton_endpoints",
    "skeleton_branches",
    "signal_mean",
    "corrected_mean",
    "integrated_density",
    "corrected_integrated_density",
    "signal_to_background",
    "punctate_fraction",
    "punctateness",
    "signal_cv",
    "dff",
    "background_median",
    "turnover_index",
    "jaccard",
    "extension_bias",
    "step_px_gapless",
    "soma_step_px_gapless",
    "evidence_motion_fraction",
    "inferred_fraction",
    # Four of the 28 Sholl summaries, chosen by what they add rather than by how
    # stable they look. The stable-looking ones are cell size again: per cell,
    # `sholl_auc` correlates with `area_px` at r = 0.97 and is 99% predicted by
    # the four shape columns already on this list, so it would add a column and
    # no information. These four are the exceptions.
    "sholl_regression_coefficient_scale_global",  # the standard decay slope, absolute axis
    "sholl_occupancy_outer_scale_cell",           # periphery fill; |r| with area 0.05
    "sholl_occupancy_ratio_scale_cell",           # centre against edge; |r| with area 0.03
    "sholl_max_intersections_scale_global",       # the column readers expect; 92% predictable
    # Neighbour geometry. Both distances, because they are not substitutes:
    # centre-to-centre and outline-to-outline correlate at r = 0.91 and still
    # differ threefold. `local_density` and `domain_occupancy` are the two that
    # say whether a cell is crowded, and neither is derivable from the others.
    "nearest_neighbour_px",
    "nearest_edge_px",
    "local_density",
    "domain_occupancy",
    # Branch geometry. `longest_branch_px` is the one branch statistic that
    # survives this resolution - a maximum needs one real branch to mean
    # something, where a mean over 8.8 mostly-tiny segments does not - and
    # `branch_count` is here because the others cannot be read without it.
    "longest_branch_px",
    "branch_count",
    # How long ago the ground each cell is standing on was last disturbed,
    # read off the trail channel's brightness. Not a restatement of any
    # turnover column: those say how much this cell changed, this says how
    # recently anything happened where it is - including before it got there.
    "trail_age_frames_mean",
]


def side_column_names(context: MeasurementContext) -> set[str]:
    """Every column the user's own spreadsheets contribute, under their prefixes.

    The one place that answers "did this column come from a measurement or from
    somebody's notes?". Asked by the run, which shows a derived module only the
    side columns the configuration named for it, and by ``coupling``, which
    refuses a pair naming a side column it was not given rather than skipping it
    the way it skips a measured column a switched-off module never wrote.

    The keys are excluded because they are not the user's numbers: the loader
    renamed the user's key column to the key it joins on, so ``identity`` and
    ``frame_index`` here are this package's own.
    """
    names: set[str] = set()
    for table in context.side.values():
        names.update(c for c in table.columns if c not in ("identity", "frame_index"))
    return names


def join_side(table: pd.DataFrame, context: MeasurementContext, key: str) -> pd.DataFrame:
    """Attach every side table keyed on ``key`` to ``table``.

    A left join, always. A frame the user's log does not mention, or a cell
    their spreadsheet has no row for, keeps its row and gets a blank - the
    alternative is a measured cell disappearing from the output because
    somebody's notes were incomplete.

    Which key a side table carries is read off the table itself rather than
    from a setting carried alongside it: the loader renamed the user's key
    column to the key it joins on, so a table holding ``identity`` is a
    per-cell table and there is nothing else to know.
    """
    for name in sorted(context.side):
        side = context.side[name]
        if side.empty or key not in side.columns:
            continue
        extra = [column for column in side.columns
                 if column != key and column not in table.columns]
        if not extra:
            continue
        table = table.merge(side[[key, *extra]], on=key, how="left")
    return table


def _spread(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float(np.percentile(valid, 75) - np.percentile(valid, 25))


#: Every metric is summarised four ways, not one. Which statistic is used is a
#: judgement, and a table that offers only the median makes that judgement for
#: the reader invisibly. It is not a cosmetic difference: across the 50 metrics
#: on this list, mean and median rank the cells differently in 48, and in 11 the
#: two disagree by more than a quarter of the spread between cells.
#: ``extension_bias`` is the sharp case at a rank agreement of 0.66 - which cell
#: extends most has a different answer under each, and only one was available.
#:
#: Median pairs with the inter-quartile range and mean with the standard
#: deviation, so each central value travels with the spread that belongs to it.
_STATISTICS = {
    "median": lambda grouped, metric: grouped[metric].median(),
    "iqr": lambda grouped, metric: grouped[metric].apply(_spread),
    "mean": lambda grouped, metric: grouped[metric].mean(),
    "sd": lambda grouped, metric: grouped[metric].std(),
}


def build_cell_summary(
    cell_frame: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    context: MeasurementContext,
) -> pd.DataFrame:
    metrics = [column for column in SUMMARY_METRICS if column in cell_frame.columns]
    grouped = cell_frame.groupby("identity", sort=True)

    summary = pd.DataFrame({"identity": sorted(cell_frame["identity"].unique())}).set_index("identity")
    summary["observed_frames"] = grouped["frame_index"].count()
    summary["first_frame_index"] = grouped["frame_index"].min()
    summary["last_frame_index"] = grouped["frame_index"].max()
    summary["span_frames"] = summary["last_frame_index"] - summary["first_frame_index"] + 1
    summary["gap_frames"] = summary["span_frames"] - summary["observed_frames"]
    summary["coverage"] = summary["observed_frames"] / context.n_frames
    summary["first_hour"] = context.scale.hours(summary["first_frame_index"] + context.source_frame_offset)
    summary["last_hour"] = context.scale.hours(summary["last_frame_index"] + context.source_frame_offset)

    if "touches_border" in cell_frame.columns:
        summary["border_frames"] = grouped["touches_border"].sum().astype(int)
        summary["ever_touches_border"] = summary["border_frames"] > 0

    for metric in metrics:
        for statistic, reduce in _STATISTICS.items():
            summary[f"{metric}_{statistic}"] = reduce(grouped, metric)

    summary = summary.reset_index()

    # Any module that declared a per-cell table as a fold has its columns
    # merged in here rather than written to a file of its own. This used to be
    # one hard-coded merge for ``motility_tracks``; the rule is now the same one
    # the cell-frame join uses, so a new per-cell measurement lands in
    # ``cell_summary`` without this file being edited.
    #
    # A column the summary already has is dropped rather than merged: a
    # median computed here and the same name arriving from a module would
    # otherwise become ``_x`` and ``_y``, and every figure would have to guess.
    for name, output in sorted(declared_tables().items()):
        if not output.fold or output.grain != ("identity",):
            continue
        folded = tables.get(name)
        if folded is None or folded.empty or "identity" not in folded.columns:
            continue
        columns = [c for c in folded.columns
                   if c == "identity" or c not in summary.columns]
        summary = summary.merge(folded[columns], on="identity", how="left")

    # Whatever the user knows about each cell that the pictures do not say.
    return join_side(summary, context, "identity")


def build_frame_summary(
    cell_frame: pd.DataFrame,
    context: MeasurementContext,
    tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    frames = context.frame_table()
    labels = context.labels

    assigned = np.array([int((labels[i] > 0).sum()) for i in range(context.n_frames)])
    # How much of each frame could be measured at all. Present and honest even
    # when no mask is declared - the whole field and 1.0 - rather than absent,
    # so a reader never has to work out whether the denominator was masked.
    field_px = int(labels.shape[1] * labels.shape[2])
    frames["valid_px"] = [context.valid_px(i) for i in range(context.n_frames)]
    frames["valid_share"] = frames["valid_px"] / float(field_px)
    frames["identities_present"] = [
        int(len([v for v in np.unique(labels[i]) if v])) for i in range(context.n_frames)
    ]
    frames["assigned_px"] = assigned
    if context.unclaimed is not None:
        frames["unclaimed_px"] = [
            int((context.unclaimed[i] > 0).sum()) for i in range(context.n_frames)
        ]
        frames["unclaimed_fraction"] = frames["unclaimed_px"] / (
            frames["unclaimed_px"] + frames["assigned_px"]
        ).replace(0, np.nan)

    metrics = [column for column in SUMMARY_METRICS if column in cell_frame.columns]
    if metrics:
        # The same reasoning as ``_STATISTICS``: the population value for a
        # frame is a choice of statistic. This table never carried a spread, so
        # it gains the mean beside the median and nothing else; adding a spread
        # here is a separate decision.
        grouped = cell_frame.groupby("frame_index")[metrics]
        aggregate = pd.concat(
            [grouped.median().add_suffix("_median"),
             grouped.mean().add_suffix("_mean")], axis=1)
        frames = frames.merge(aggregate.reset_index(), on="frame_index", how="left")

    # A module that measured something about the whole frame rather than about
    # a cell folds its columns in here, by the same rule ``build_cell_summary``
    # applies at one row per cell. Without this a table could declare
    # ``fold=True`` at frame grain, pass every declaration test, and then
    # quietly never be written anywhere.
    for name, output in sorted(declared_tables().items()):
        if not output.fold or output.grain != ("frame_index",):
            continue
        folded = (tables or {}).get(name)
        if folded is None or folded.empty or "frame_index" not in folded.columns:
            continue
        columns = [c for c in folded.columns
                   if c == "frame_index" or c not in frames.columns]
        frames = frames.merge(folded[columns], on="frame_index", how="left")

    # Whatever the user knows about each timepoint that the pictures do not say.
    return join_side(frames, context, "frame_index")


def build_movie_summary(
    cell_frame: pd.DataFrame,
    cell_summary: pd.DataFrame,
    frame_summary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    context: MeasurementContext,
) -> dict:
    scale = context.scale
    summary: dict = {
        "stem": context.stem,
        "frames": context.n_frames,
        "hours_covered": round(scale.hours(context.n_frames - 1), 3),
        "minutes_per_frame": scale.minutes_per_frame,
        "identities": len(context.identities),
        "cell_frame_rows": int(len(cell_frame)),
        "scale": scale.describe(),
        "coverage": {
            "median_identities_per_frame": float(frame_summary["identities_present"].median()),
            "cells_present_every_frame": int(
                (cell_summary["observed_frames"] == context.n_frames).sum()
            ),
            "total_gap_frames": int(cell_summary["gap_frames"].sum()),
        },
    }

    if "area_px_median" in cell_summary:
        summary["morphology"] = {
            "median_cell_area_px": float(cell_summary["area_px_median"].median()),
            "median_solidity": float(cell_summary["solidity_median"].median())
            if "solidity_median" in cell_summary
            else None,
            "median_skeleton_px": float(cell_summary["skeleton_px_median"].median())
            if "skeleton_px_median" in cell_summary
            else None,
        }

    if "corrected_mean_median" in cell_summary:
        zero_background_fraction = (
            float(cell_frame["background_is_zero"].mean())
            if "background_is_zero" in cell_frame
            else None
        )
        background_is_zero = (
            zero_background_fraction is not None and zero_background_fraction > 0.9
        )
        summary["intensity"] = {
            "median_corrected_mean": float(cell_summary["corrected_mean_median"].median()),
            "input_already_background_subtracted": background_is_zero,
            "zero_background_fraction": zero_background_fraction,
            "median_punctateness_p90_over_median": float(
                cell_summary["punctateness_median"].median()
            )
            if "punctateness_median" in cell_summary
            else None,
            "median_signal_to_background": None
            if background_is_zero
            else float(cell_summary["signal_to_background_median"].median()),
        }

    if "turnover_index" in cell_frame:
        summary["surveillance"] = {
            "median_turnover_index": float(cell_frame["turnover_index"].median()),
            "median_jaccard": float(cell_frame["jaccard"].median())
            if "jaccard" in cell_frame
            else None,
        }

    provenance = tables.get("provenance")
    if provenance is not None and not provenance.empty:
        inferred_px = int(provenance["inferred_px"].sum())
        footprint = inferred_px + int(provenance["observed_px"].sum())
        per_frame = tables.get("provenance_frame")
        summary["provenance"] = {
            "labelled_px": footprint,
            "inferred_px": inferred_px,
            "inferred_fraction_of_footprint": (
                inferred_px / footprint if footprint else None
            ),
            # The split that stops "reconstructed" being read as
            # "invented": almost all of it is a name the tracker
            # worked out over a pixel the microscope did show.
            "renamed_px": int(provenance["renamed_px"].sum())
            if "renamed_px" in provenance
            else None,
            "added_px": int(provenance["added_px"].sum())
            if "added_px" in provenance
            else None,
            "added_fraction_of_footprint": (
                int(provenance["added_px"].sum()) / footprint
                if footprint and "added_px" in provenance
                else None
            ),
            "cell_frames": int(len(provenance)),
            "cell_frames_touched": int((provenance["inferred_px"] > 0).sum()),
            "cell_frames_with_added_px": int((provenance["added_px"] > 0).sum())
            if "added_px" in provenance
            else None,
            # A supplied pixel is not usually a rim on a real outline. On the
            # pinned movie all 572 of them are whole cell-frames whose entire
            # footprint the detection missed, which is a far more legible fact
            # than the pixel count and a much easier one to check.
            "cell_frames_entirely_supplied": int(
                ((provenance["added_px"] > 0)
                 & (provenance["observed_px"] == 0)
                 & (provenance["renamed_px"] == 0)).sum()
            )
            if "added_px" in provenance
            else None,
            "identities": int(provenance["identity"].nunique()),
            "identities_touched": int(
                provenance.loc[provenance["inferred_px"] > 0, "identity"].nunique()
            ),
            "unresolved_px_outside_any_cell": (
                int(per_frame["unresolved_px_unowned"].sum())
                if per_frame is not None and "unresolved_px_unowned" in per_frame
                else None
            ),
        }

    tracks = tables.get("motility_tracks")
    if tracks is not None and not tracks.empty:
        summary["motility"] = {
            "tracks": int(len(tracks)),
            "median_step_px": float(tracks["median_step_px"].median()),
            "median_straightness": float(tracks["straightness"].median()),
            "median_msd_alpha": float(tracks["msd_alpha"].median()),
            "median_speed_" + scale.speed_unit: float(tracks["mean_speed"].median()),
        }

    rhythms = tables.get("rhythms")
    if rhythms is not None and not rhythms.empty:
        per_metric = {}
        for metric, group in rhythms.groupby("metric"):
            per_metric[metric] = {
                "cells_tested": int(len(group)),
                "primary_rhythm_test": str(group["primary_rhythm_test"].iloc[0]),
                "period_estimation_method": str(
                    group.get("period_estimation_method", group["primary_rhythm_test"]).iloc[0]
                ),
                "rhythmic_by_primary_test": int(
                    group["rhythmic"].fillna(False).astype(bool).sum()),
                "rhythmic_by_cosinor": (
                    int(group["rhythmic_cosinor"].sum())
                    if group["rhythmic_cosinor"].notna().any() else None
                ),
                "rhythmic_by_both_tests": (
                    int(group["rhythmic_both"].sum())
                    if group["rhythmic_both"].notna().any() else None
                ),
                "median_best_period_hours": float(group["best_period_hours"].median())
                if "best_period_hours" in group
                else None,
                "median_free_period_hours": float(group["best_period_hours"].median())
                if "best_period_hours" in group else None,
                "median_peak_hour": float(
                    (group["best_phase_hours"] if "best_phase_hours" in group
                     else group["cosinor_peak_hour"]).median()
                ),
            }
        population = tables.get("rhythms_population")
        if population is not None and not population.empty:
            for _, row in population[population["population"] == "all_tested"].iterrows():
                if row["metric"] in per_metric:
                    per_metric[row["metric"]].update(
                        {
                            "phase_vector_length": float(row["vector_length"]),
                            "phase_rayleigh_p": float(row["rayleigh_p_value"]),
                            "population_mean_peak_hour": float(row["mean_peak_hour"]),
                        }
                    )

        null = tables.get("rhythms_null")
        if null is not None and not null.empty:
            for _, row in null.iterrows():
                if row["metric"] in per_metric:
                    per_metric[row["metric"]].update(
                        {
                            "null_false_positive_rate_primary": float(
                                row["false_positive_rate_primary"]),
                            "excess_over_null_primary": float(
                                row["excess_over_null_primary"]),
                            "null_false_positive_rate_both": float(row["false_positive_rate_both"]),
                            "excess_over_null": float(row["excess_over_null_both"]),
                        }
                    )
        summary["rhythms"] = {
            "cycles_covered": float(rhythms["cycles_covered"].max()),
            "period_underdetermined": bool(rhythms["period_underdetermined"].all()),
            "null_model": str(null["null_model"].iloc[0]) if null is not None and not null.empty else None,
            "per_metric": per_metric,
        }

    return summary
